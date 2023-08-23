"""procrustes.py

Implements full affine and optimally constrained affine fitting with the
  same interface as required by scikit-learn.

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

import numpy as np
import scipy.linalg as lin
from enum import IntEnum

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
#from sklearn.utils.validation import check_array, check_is_fitted


class RigidRegression_types(IntEnum):
    # for ease of use with AffineRANSACRegressor decided to implement the full affine in this class also
    affine = 0
    rigid = 1; scale = 2; nonuniform_scale = 3


class RigidRegression(BaseEstimator, RegressorMixin):
    def __init__(self, demo_param='demo_param', rigid_type=RigidRegression_types.rigid):
        self.demo_param = demo_param
        self._rigid_type = rigid_type
        if self._rigid_type == RigidRegression_types.affine:
            self._xform = RigidRegression.affine_transform
        elif self._rigid_type == RigidRegression_types.rigid:
            self._xform = RigidRegression.rigid_transform
        elif self._rigid_type == RigidRegression_types.scale:
            self._xform = RigidRegression.rigid_scale_transform
        elif self._rigid_type == RigidRegression_types.nonuniform_scale:
            self._xform = RigidRegression.rigid_nonuniform_scale_transform
        else:
            assert(False) # bad rigid_type specified

    # NOTE: X and y are simply the points, i.e., the transformation includes a translation even though
    #   X is not augmented to contain a constant column of ones (as opposed to scipy linear regression).
    def fit(self, X, y, fast=False):
        if not fast:
            X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True, multi_output=True)

        self.coef_, self.translation_, a, self.scale_ = self._xform(X,y)

        if self.coef_ is not None:
            # computed angle is only valid for 2d rotations
            if X.shape[1] == 2: self.angle_ = a
            self.is_fitted_ = True
        else:
            self.is_fitted_ = False

        # `fit` should always return `self`
        return self

    def predict(self, X, fast=False):
        if not fast:
            X = check_array(X, accept_sparse=True)
            check_is_fitted(self, 'is_fitted_')

        if self._rigid_type == RigidRegression_types.affine:
            return np.dot(X, self.coef_.T)
        else:
            return np.dot(X, self.coef_[:-1,:-1].T) + self.translation_

    # Input: A are source points, B are dest points, both npts x ndims
    #   i.e., calculate rotation and translation for A to match B
    # NOTE: this method must contain the column of ones in the source in order to fit the translation.
    # xxx - make this consistent between the methods
    @staticmethod
    def affine_transform(A, B):
        # xxx - expose?
        #_lapack_driver = 'gelsd' # default for scipy lstsq
        _lapack_driver = 'gelsy' # much faster for this application

        Ra = lin.lstsq(A, B, cond=None, check_finite=False, lapack_driver=_lapack_driver)[0].T
        # xxx - could use the method from plot_aggregation.py to decompose the affine matrix
        return Ra, None, None, None

    # Regression for the orthonormal prucrustes problem, i.e., rigid body.
    # Also known as the Kabsch–Umeyama algorithm.
    # Only rotation and translation (and optionally scale) are fit.
    # modified from http://nghiaho.com/uploads/code/rigid_transform_3D.py_
    # See also:
    #   https://en.wikipedia.org/wiki/Kabsch_algorithm
    # Input: A are source points, B are dest points, both npts x ndims
    #   i.e., calculate rotation and translation for A to match B
    # returns R = ndims+1 x ndims+1 augmented rotation matrix
    #         t = ndims translation vector
    #         a = rotation angle in radians, NOTE: only valid for 2D rotation
    #         s = uniform scale, if scale==True, else 1.
    @staticmethod
    def rigid_transform(A, B, scale=False):
        npts, ndims = A.shape
        assert( npts == B.shape[0] and ndims == B.shape[1] ) # must be same number and dimension of points
        cA = A.mean(axis=0, dtype=np.double); cB = B.mean(axis=0, dtype=np.double)
        AA = A - cA; BB = B - cB
        U, s, Vt = lin.svd(np.dot(AA.T,BB),overwrite_a=True,full_matrices=False); V = Vt.T

        R = np.dot(V,U.T)
        if lin.det(R) < 0:
            # this prevents fitting reflection which is not a rotation
            V[:,-1] = -V[:,-1]
            s[-1] = -s[-1]
            R = np.dot(V,U.T)
        if scale:
            # this is the Umeyama addition to the Kabsch algorithm; the paper is math dense...
            # see also:
            #   https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
            #   SO - 13432805/finding-translation-and-scale-on-two-sets-of-points-to-get-least-square-error-in
            s = s.sum() / npts / np.var(AA) / ndims
        else:
            s = 1.
        #https://math.stackexchange.com/questions/301319/derive-a-rotation-from-a-2d-rotation-matrix
        a = np.arctan2(R[1,0], R[0,0])

        # scale the rotation matrix, then fit the translation
        R *= s; t = -np.dot(R, cA.T) + cB.T

        # return the augmented affine (rotation and translation) matrix
        Ra = np.zeros((ndims+1,ndims+1), dtype=np.double)
        Ra[:ndims,:ndims] = R; Ra[:ndims,ndims] = t; Ra[ndims,ndims] = 1

        return Ra, t, a, np.array([s])

    @staticmethod
    def rigid_scale_transform(A, B):
        return RigidRegression.rigid_transform(A, B, scale=True)

    # SO - 3955634/how-to-find-a-transformation-non-uniform-scaling-and-similarity-that-maps-one
    # see also derive_nonuniform_rigid_fit.py, essentially outlines solving the minimization problem.
    # Input: A are source points, B are dest points, both npts x ndims
    #   i.e., calculate rotation and translation for A to match B
    # NOTE: only works for 2d points.
    # returns R = ndims+1 x ndims+1 augmented rotation + nonuniform scale affine matrix
    #         t = ndims translation vector
    #         a = rotation angle in radians
    #         s = [sx, sy] solved scale components
    @staticmethod
    def rigid_nonuniform_scale_transform(A, B, tol=None):
        npts, ndims = A.shape
        assert( npts == B.shape[0] and ndims == B.shape[1] ) # must be same number and dimension of points
        assert( ndims == 2 ) # only found solution for 2D
        cA = A.mean(axis=0, dtype=np.double); cB = B.mean(axis=0, dtype=np.double)
        AA = A - cA; BB = B - cB
        if tol is None: tol = np.finfo(np.double).eps

        _B = -2*np.dot(AA.T,BB)
        _C = np.dot(AA.T,AA)
        # the formulas for the solution come from derive_nonuniform_rigid_fit.py
        b11,b12,b21,b22 = _B.flat[:]
        c11,c12,c21,c22 = _C.flat[:]
        denom = (b11*b12*c22 - b21*b22*c11)
        if (np.abs(denom) < tol) or (np.abs(2*c11) < tol) or (np.abs(2*c22) < tol):
            return None, None, None, None
        #k = (-b11**2*c22 + b12**2*c22 + b21**2*c11 - b22**2*c11)/(b11*b12*c22 - b21*b22*c11)
        k = (-b11**2*c22 + b12**2*c22 + b21**2*c11 - b22**2*c11) / denom
        kk = 2*k/np.sqrt(k**2 + 4)
        r1 = np.array([-np.sqrt(-kk + 2)/2, np.sqrt(-kk + 2)/2, -np.sqrt( kk + 2)/2, np.sqrt( kk + 2)/2])
        r2 = np.array([ np.sqrt( kk + 2)/2, np.sqrt( kk + 2)/2,  np.sqrt(-kk + 2)/2, np.sqrt(-kk + 2)/2])
        sx = -(b11*r1 + b12*r2)/(2*c11)
        sy =  (b21*r2 - b22*r1)/(2*c22)

        # flip the signs on R/S if both sx and sy are negative.
        sel = np.logical_and(sx < 0, sy < 0)
        r1 = [-x if y else x for x,y in zip(r1,sel)]
        r2 = [-x if y else x for x,y in zip(r2,sel)]
        sx[sel] = -sx[sel]; sy[sel] = -sy[sel]
        # after flipping the signs, reject any solutions where sx and sy are not both positive.
        sel = np.logical_and(sx >= 0, sy >= 0)
        r1 = [x for x,y in zip(r1,sel) if y]
        r2 = [x for x,y in zip(r2,sel) if y]
        sx = sx[sel]; sy = sy[sel]

        # used 2D specific solved r1,r2,sx,sy to create affine matrix.
        R = [np.array([[c, -s], [s, c]]) for c,s in zip(r1,r2)]
        S = [np.array([[x, 0], [0, y]]) for x,y in zip(sx,sy)]
        RS = [np.dot(x,y) for x,y in zip(R,S)]

        # solve for remaining translation after xform matrix applied.
        t = [-np.dot(x, cA.T) + cB.T for x in RS]

        #https://math.stackexchange.com/questions/301319/derive-a-rotation-from-a-2d-rotation-matrix
        a = np.arctan2(r2, r1)

        nsolns = len(R)
        if nsolns > 0:
            # of the possible realisitc solutions, take the one that gives the best fit variance account for.
            var = np.var(BB)
            vaf = np.zeros(nsolns)
            for i in range(nsolns):
                xpts = np.dot(A, RS[i].T) + t[i]
                mse = ((xpts - B)**2).mean()
                vaf[i] = 1 - mse/var
            mi = np.argmax(vaf)
            RS = RS[mi]; t = t[mi]; a = a[mi]
            s = np.array([sx[mi], sy[mi]])

            # return the augmented affine (rotation and translation) matrix
            Ra = np.zeros((ndims+1,ndims+1), dtype=np.double)
            Ra[:ndims,:ndims] = RS; Ra[:ndims,ndims] = t; Ra[ndims,ndims] = 1
        else:
            # have to deal with the possiblity that all solutions are not realistic.
            # this could be the possibility when they all involve reflections for example, which we do not want.
            Ra = t = a = s = None

        return Ra, t, a, s
    #def rigid_nonuniform_scale_transform(A, B):
