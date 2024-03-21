"""AffineRANSACRegressor.py

Modified version of sklearn RANSAC heavily optimized for runtime and only
  supporting regressions utilized by msemalign.

Copyright (C) 2018-2023 Max Planck Institute for Neurobiology of Behavior

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

# This was mostly ripped off of sklearn, but with the intent of trying to optimize for runtime.
# It is frequently used in emalign for outlier detection, and the sklearn version became prohibitively slow.
# Modifying this was much easier than completely rewriting RANSAC from scratch, or trying to modify
#   one of many many (semi-complete or problematic) versions available on github.

import numpy as np
import warnings
import traceback
import time

# import scipy.linalg as lin

# removed all dependencies on sklearn.
# from ..base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone
# from ..base import MultiOutputMixin
# from ..utils import check_random_state, check_consistent_length
# from ..utils.random import sample_without_replacement
# from ..utils.validation import check_is_fitted, _check_sample_weight
# from ..utils.validation import _deprecate_positional_args
# from ._base import LinearRegression
# from ..utils.validation import has_fit_parameter
# from ..exceptions import ConvergenceWarning

import multiprocessing as mp
import queue

from .procrustes import RigidRegression, RigidRegression_types

# for now avoid complete dependency on the msem package
try:
    from .zimages import zimages
    queue_timeout = zimages.queue_timeout
except:
    queue_timeout = 180

_EPSILON = np.spacing(1)


# <<< methods for repeating ransac in parallel workers

def ransac_job(ind, inds, ransac, Xpts, ypts, result_queue, verbose):
    if verbose: print('\tworker%d started' % (ind,))

    coef = None; mask = None; fail_cnt = 0
    for i in range(inds.size):
        try:
            # NOTE: dangerous try-except-pass, do not put anything else inside this try.
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                ransac.fit(Xpts, ypts)
                ccoef = ransac.estimator_.coef_.copy(); cmask = ransac.inlier_mask_.copy()
                fail_cnt = 0
                # xxx - should we still take these fits if it hits max_trials without early stopping?
                #if ransac.n_trials_ < ransac.max_trials:
                #    fail_cnt = 0
                #else:
                #    fail_cnt += 1
        except:
            print('ransac fit failed:')
            print(traceback.format_exc())
            ccoef = None; cmask = None; fail_cnt += 1
        if mask is None or (cmask is not None and cmask.sum() > mask.sum()):
            coef = ccoef; mask = cmask
            if verbose and mask is not None: print(mask.sum())

        result = {'ind':inds[i], 'coef':coef, 'mask':mask, 'fail_cnt':fail_cnt, 'iworker':ind}
        result_queue.put(result)
    #for i in range(inds.size):

    if verbose: print('\tworker%d completed' % (ind, ))
#def ransac_job

# found repeating ransac a few times better than simply increasing ransac max_trials.
# this is a worker parallized version that runs repeats in parallel.
def _ransac_repeat(Xpts, ypts, ransac, ransac_repeats, verbose=False, nworkers=1):
    # run the ransac repeats in parallel
    workers = [None]*nworkers
    result_queue = mp.Queue(ransac_repeats)
    inds = np.arange(ransac_repeats)
    inds = np.array_split(inds, nworkers)

    for i in range(nworkers):
        workers[i] = mp.Process(target=ransac_job, daemon=True,
                args=(i, inds[i], ransac, Xpts, ypts, result_queue, False))
        workers[i].start()
    # NOTE: only call join after queue is emptied
    # https://stackoverflow.com/questions/45948463/python-multiprocessing-join-deadlock-depends-on-worker-function

    # for multiple processes save memory by only allocating vertices for this process.
    nprint = 0
    dt = time.time()
    coef = None; mask = None #; fail_cnt = 0
    worker_cnts = np.zeros((nworkers,), dtype=np.int64)
    dead_workers = np.zeros((nworkers,), dtype=bool)
    #for i in range(ransac_repeats):
    i = 0
    while i < ransac_repeats:
        if verbose and i>0 and nprint > 0 and i%nprint==0:
            print('{} through q in {:.2f} s, worker_cnts:'.format(nprint,time.time()-dt,)); dt = time.time()
            print(worker_cnts)

        try:
            res = result_queue.get(block=True, timeout=queue_timeout)
        except queue.Empty:
            for x in range(nworkers):
                if not workers[x].is_alive() and worker_cnts[x] != inds[x].size:
                    if dead_workers[x]:
                        print('worker {} is dead and worker cnt is {} / {}'.format(x,worker_cnts[x],inds[x].size))
                        assert(False) # a worker exitted with an error or was killed without finishing
                    else:
                        # to make sure this is not a race condition, try the queue again before error exit
                        dead_workers[x] = 1
            continue

        if mask is None or (res['mask'] is not None and res['mask'].sum() > mask.sum()):
            coef = res['coef']; mask = res['mask'] #; fail_cnt = res['fail_cnt']

        worker_cnts[res['iworker']] += 1
        i += 1
    assert(result_queue.empty())
    [x.join() for x in workers]
    [x.close() for x in workers]

    return coef, mask
#def _ransac_repeat

# methods for repeating ransac in parallel workers >>>


def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.
    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.
    n_samples : int
        Total number of samples in the data.
    min_samples : int
        Minimum number of samples chosen randomly from original data.
    probability : float
        Probability (confidence) that one outlier-free sample is generated.
    Returns
    -------
    trials : int
        Number of trials.
    """
    inlier_ratio = n_inliers / float(n_samples)
    nom = max(_EPSILON, 1 - probability)
    denom = max(_EPSILON, 1 - inlier_ratio ** min_samples)
    if nom == 1:
        return 0
    if denom == 1:
        return float('inf')
    return abs(float(np.ceil(np.log(nom) / np.log(denom))))


# https://stats.stackexchange.com/questions/296005/the-expected-number-of-unique-elements-drawn-with-replacement
# https://math.stackexchange.com/questions/1467153/
#   finding-variance-of-number-of-distinct-values-chosen-from-a-set-with-replacement

#def local_sample_without_replacement(N, size, n_collision_stds=6):
#    # N = n*m             # total number of matrix elements
#    # k = int(N*density)  # number of desired nonzero matrix elements
#    k = np.prod(size)
#
#    # When sampling (1..N) with replacement p times we get
#    # on average k unique values,
#    #     k = N*(1 - ((N - 1)/N)**p)
#    # or inversely
#    p = int(np.log(1 - k / N) / np.log(1 - 1 / N))
#    n_collisions = p - k
#    # the variance of the number of collisions can be estimated with possion distribution.
#    #std_collisions = int(np.sqrt(n_collisions))
#    std_collisions = max([1, int(np.sqrt(n_collisions))])
#    n_samples = k + n_collisions + n_collision_stds*std_collisions
#    inds = np.random.randint(0, N, size=n_samples)
#    _, iinds = np.unique(inds, return_index=True)
#    inds = inds[np.sort(iinds)]
#    assert( k <= inds.size ) # increase n_collision_stds
#
#    return inds[:k].reshape(size)


def r2_score(y_true, y_pred, *, sample_weight=None,
             multioutput="uniform_average"):
    """R^2 (coefficient of determination) regression score function.
    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a R^2 score of 0.0.
    Read more in the :ref:`User Guide <r2_score>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average', 'variance_weighted'}, \
            array-like of shape (n_outputs,) or None, default='uniform_average'
        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.
        Default is "uniform_average".
        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.
        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.
        .. versionchanged:: 0.19
            Default value of multioutput is 'uniform_average'.
    Returns
    -------
    z : float or ndarray of floats
        The R^2 score or ndarray of scores if 'multioutput' is
        'raw_values'.
    Notes
    -----
    This is not a symmetric function.
    Unlike most other scores, R^2 score may be negative (it need not actually
    be the square of a quantity R).
    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.
    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    Examples
    --------
    >>> from sklearn.metrics import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)
    0.948...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> r2_score(y_true, y_pred,
    ...          multioutput='variance_weighted')
    0.938...
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1, 2, 3]
    >>> r2_score(y_true, y_pred)
    1.0
    >>> y_true = [1, 2, 3]
    >>> y_pred = [2, 2, 2]
    >>> r2_score(y_true, y_pred)
    0.0
    >>> y_true = [1, 2, 3]
    >>> y_pred = [3, 2, 1]
    >>> r2_score(y_true, y_pred)
    -3.0
    """
    # y_type, y_true, y_pred, multioutput = _check_reg_targets(
    #     y_true, y_pred, multioutput)
    # check_consistent_length(y_true, y_pred, sample_weight)

    # if _num_samples(y_pred) < 2:
    #     msg = "R^2 score is not well-defined with less than two samples."
    #     warnings.warn(msg, UndefinedMetricWarning)
    #     return float('nan')
    if y_pred.shape[0] < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg)
        return float('nan')

    if sample_weight is not None:
        #sample_weight = column_or_1d(sample_weight)
        # weight = sample_weight[:, np.newaxis]
        weight = sample_weight.reshape(-1)[:, np.newaxis]
    else:
        weight = 1.

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,
                                                      dtype=np.float64)
    denominator = (weight * (y_true - np.average(
        y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0,
                                                          dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            # return scores individually
            return output_scores
        elif multioutput == 'uniform_average':
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)


class AffineRANSACRegressor():
    def __init__(self, min_samples=None,
                 residual_threshold=None,
                 max_trials=100,
                 max_skips=np.inf,
                 stop_score=np.inf,
                 stop_n_inliers=np.inf,
                 stop_probability=0.99,
                 rigid_type=RigidRegression_types.affine,
                 loss='absolute_loss',):

        # self.base_estimator = base_estimator
        # instantiate a placeholder object for the estimator.
        # this version always uses a LinearRegression (least squares) estimator.
        # self.estimator_ = type('BaseEstimator', (object,), {})()
        self.estimator_ = RigidRegression(rigid_type=rigid_type)
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        # self.is_data_valid = is_data_valid
        # self.is_model_valid = is_model_valid
        self.max_trials = int(max_trials)
        self.max_skips = max_skips
        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        # self.random_state = random_state
        self.loss = loss

    def fit(self, X, y, use_score=False):
        """Fit estimator using RANSAC algorithm.
        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_features]
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        Raises
        ------
        ValueError
            If no valid consensus set could be found. This occurs if
            `is_data_valid` and `is_model_valid` return False for all
            `max_trials` randomly chosen sub-samples.
        """
        # # Need to validate separately here.
        # # We can't pass multi_ouput=True because that would allow y to be csr.
        # check_X_params = dict(accept_sparse='csr')
        # check_y_params = dict(ensure_2d=False)
        # X, y = self._validate_data(X, y, validate_separately=(check_X_params,
        #                                                       check_y_params))
        # check_consistent_length(X, y)

        # if self.base_estimator is not None:
        #     base_estimator = clone(self.base_estimator)
        # else:
        #     base_estimator = LinearRegression()

        if self.min_samples is None:
            # assume linear model by default
            min_samples = X.shape[1] + 1
        elif 0 < self.min_samples < 1:
            min_samples = np.ceil(self.min_samples * X.shape[0])
        elif self.min_samples >= 1:
            if self.min_samples % 1 != 0:
                raise ValueError("Absolute number of samples must be an "
                                 "integer value.")
            min_samples = self.min_samples
        else:
            raise ValueError("Value for `min_samples` must be scalar and "
                             "positive.")
        if min_samples > X.shape[0]:
            raise ValueError("`min_samples` may not be larger than number "
                             "of samples: n_samples = %d." % (X.shape[0]))

        if self.stop_probability < 0 or self.stop_probability > 1:
            raise ValueError("`stop_probability` must be in range [0, 1].")

        if self.residual_threshold is None:
            # MAD (median absolute deviation)
            residual_threshold = np.median(np.abs(y - np.median(y)))
        else:
            residual_threshold = self.residual_threshold

        if self.loss == "absolute_loss":
            if y.ndim == 1:
                loss_function = lambda y_true, y_pred: np.abs(y_true - y_pred)
            else:
                loss_function = lambda \
                    y_true, y_pred: np.sum(np.abs(y_true - y_pred), axis=1)

        elif self.loss == "squared_loss":
            if y.ndim == 1:
                loss_function = lambda y_true, y_pred: (y_true - y_pred) ** 2
            else:
                loss_function = lambda \
                    y_true, y_pred: np.sum((y_true - y_pred) ** 2, axis=1)

        elif callable(self.loss):
            loss_function = self.loss

        else:
            raise ValueError(
                "loss should be 'absolute_loss', 'squared_loss' or a callable."
                "Got %s. " % self.loss)


        # random_state = check_random_state(self.random_state)

        # try:  # Not all estimator accept a random_state
        #     base_estimator.set_params(random_state=random_state)
        # except ValueError:
        #     pass
        #
        # estimator_fit_has_sample_weight = has_fit_parameter(base_estimator,
        #                                                     "sample_weight")
        # estimator_name = type(base_estimator).__name__
        # if (sample_weight is not None and not
        #         estimator_fit_has_sample_weight):
        #     raise ValueError("%s does not support sample_weight. Samples"
        #                      " weights are only used for the calibration"
        #                      " itself." % estimator_name)
        # if sample_weight is not None:
        #     sample_weight = _check_sample_weight(sample_weight, X)

        n_inliers_best = 1
        score_best = -np.inf
        inlier_mask_best = None
        X_inlier_best = None
        y_inlier_best = None
        #inlier_best_idxs_subset = None
        self.n_skips_no_inliers_ = 0
        self.n_skips_invalid_data_ = 0
        self.n_skips_invalid_model_ = 0

        # number of data samples
        n_samples = X.shape[0]
        sample_idxs = np.arange(n_samples)

        self.n_trials_ = 0
        max_trials = self.max_trials
        niters_per_rand = 1000
        rand_rng = np.random.default_rng()
        # https://stackoverflow.com/questions/47675003/how-to-create-2d-array-with-numpy-random-choice-for-every-rows
        # all_subset_idxs = np.random.default_rng().random((max_trials, n_samples)).argpartition(min_samples,
        #         axis=1)[:,:min_samples]
        while self.n_trials_ < max_trials:
            # to save on memory, compromise between rand generation every iteration, and one massive one.
            rand_iter = self.n_trials_ % niters_per_rand
            if rand_iter == 0:
                all_subset_idxs = rand_rng.random((niters_per_rand, n_samples)).\
                        argpartition(min_samples, axis=1)[:,:min_samples]

            self.n_trials_ += 1

            if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ + \
                    self.n_skips_invalid_model_) > self.max_skips:
                break

            # choose random sample set
            # subset_idxs = sample_without_replacement(n_samples, min_samples,
            #                                          random_state=random_state)
            #subset_idxs = local_sample_without_replacement(n_samples, min_samples)
            #subset_idxs = np.random.choice(n_samples, size=min_samples, replace=False)
            #subset_idxs = all_subset_idxs[self.n_trials_-1,:] # fastest option
            subset_idxs = all_subset_idxs[rand_iter,:] # fastest option, but tradeoff for memory usage
            X_subset = X[subset_idxs]
            y_subset = y[subset_idxs]

            # # check if random sample set is valid
            # if (self.is_data_valid is not None
            #         and not self.is_data_valid(X_subset, y_subset)):
            #     self.n_skips_invalid_data_ += 1
            #     continue

            # # fit model for current random sample set
            # if sample_weight is None:
            #     base_estimator.fit(X_subset, y_subset)
            # else:
            #     base_estimator.fit(X_subset, y_subset,
            #                        sample_weight=sample_weight[subset_idxs])
            # OLD:
            # if self.rigid:
            #     self.estimator_.fit(X_subset, y_subset, fast=True)
            # else:
            #     self.estimator_.coef_ = lin.lstsq(X_subset, y_subset, cond=None, check_finite=False,
            #             lapack_driver=self._lapack_driver)[0].T
            self.estimator_.fit(X_subset, y_subset, fast=True)

            # # check if estimated model is valid
            # if (self.is_model_valid is not None and not
            #         self.is_model_valid(base_estimator, X_subset, y_subset)):
            #     self.n_skips_invalid_model_ += 1
            #     continue
            # check if estimated model is valid
            if not self.estimator_.is_fitted_:
                self.n_skips_invalid_model_ += 1
                continue

            # residuals of all data for current random sample model
            # y_pred = base_estimator.predict(X)
            # OLD:
            # if self.rigid:
            #     y_pred = self.estimator_.predict(X, fast=True)
            # else:
            #     y_pred = np.dot(X, self.estimator_.coef_.T)
            y_pred = self.estimator_.predict(X, fast=True)
            residuals_subset = loss_function(y, y_pred)

            # classify data into inliers and outliers
            inlier_mask_subset = residuals_subset < residual_threshold
            n_inliers_subset = np.sum(inlier_mask_subset)

            # less inliers -> skip current random sample
            if n_inliers_subset < n_inliers_best:
                self.n_skips_no_inliers_ += 1
                continue

            # extract inlier data set
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            X_inlier_subset = X[inlier_idxs_subset]
            y_inlier_subset = y[inlier_idxs_subset]

            # score of inlier data set
            if use_score:
                # score_subset = base_estimator.score(X_inlier_subset,
                #                                     y_inlier_subset)
                score_subset = self.score(X_inlier_subset,
                                          y_inlier_subset)
            else:
                score_subset = -np.abs(residuals_subset).sum()

            # same number of inliers but worse score -> skip current random
            # sample
            if (n_inliers_subset == n_inliers_best and score_subset < score_best):
                continue

            # save current random sample as best sample
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            X_inlier_best = X_inlier_subset
            y_inlier_best = y_inlier_subset
            #inlier_best_idxs_subset = inlier_idxs_subset

            max_trials = min(
                max_trials,
                _dynamic_max_trials(n_inliers_best, n_samples,
                                    min_samples, self.stop_probability))

            # break if sufficient number of inliers or score is reached
            if n_inliers_best >= self.stop_n_inliers or \
                            score_best >= self.stop_score:
                break
        #while self.n_trials_ < max_trials:

        # if none of the iterations met the required criteria
        if inlier_mask_best is None:
            # if ((self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
            #         self.n_skips_invalid_model_) > self.max_skips):
            if self.n_skips_no_inliers_ > self.max_skips:
                raise ValueError(
                    "RANSAC skipped more iterations than `max_skips` without"
                    " finding a valid consensus set. Iterations were skipped"
                    " because each randomly chosen sub-sample failed the"
                    " passing criteria. See estimator attributes for"
                    " diagnostics (n_skips*).")
            else:
                raise ValueError(
                    "RANSAC could not find a valid consensus set. All"
                    " `max_trials` iterations were skipped because each"
                    " randomly chosen sub-sample failed the passing criteria."
                    " See estimator attributes for diagnostics (n_skips*).")
        else:
            # if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
            #         self.n_skips_invalid_model_) > self.max_skips:
            if self.n_skips_no_inliers_ > self.max_skips:
                warnings.warn("RANSAC found a valid consensus set but exited"
                              " early due to skipping more iterations than"
                              " `max_skips`. See estimator attributes for"
                              " diagnostics (n_skips*).",)
                              #ConvergenceWarning)

        # # estimate final model using all inliers
        # if sample_weight is None:
        #     base_estimator.fit(X_inlier_best, y_inlier_best)
        # else:
        #     base_estimator.fit(
        #         X_inlier_best,
        #         y_inlier_best,
        #         sample_weight=sample_weight[inlier_best_idxs_subset])
        # OLD:
        # if self.rigid:
        #     self.estimator_.fit(X_inlier_best, y_inlier_best, fast=True)
        # else:
        #     self.estimator_.coef_ = lin.lstsq(X_inlier_best, y_inlier_best, cond=None, check_finite=False,
        #             lapack_driver=self._lapack_driver)[0].T
        self.estimator_.fit(X_inlier_best, y_inlier_best, fast=True)

        if not self.estimator_.is_fitted_:
            raise ValueError("RANSAC Best inlier mask could not be refit.")

        # self.estimator_ = base_estimator
        self.inlier_mask_ = inlier_mask_best
        return self

    def predict(self, X):
        """Predict using the estimated model.
        This is a wrapper for `estimator_.predict(X)`.
        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
        Returns
        -------
        y : array, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        # check_is_fitted(self)

        # return self.estimator_.predict(X)
        # OLD:
        # if self.rigid:
        #     y_pred = self.estimator_.predict(X, fast=True)
        # else:
        #     y_pred = np.dot(X, self.estimator_.coef_.T)
        return self.estimator_.predict(X, fast=True)

    def score(self, X, y, sample_weight=None):
        """Returns the score of the prediction.
        This is a wrapper for `estimator_.score(X, y)`.
        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data.
        y : array, shape = [n_samples] or [n_samples, n_targets]
            Target values.
        Returns
        -------
        z : float
            Score of the prediction.
        """
        # check_is_fitted(self)

        # return self.estimator_.score(X, y)

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)


    # def _more_tags(self):
    #     return {
    #         '_xfail_checks': {
    #             'check_sample_weights_invariance':
    #             'zero sample_weight is not equivalent to removing samples',
    #         }
    #     }
