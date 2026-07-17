
# generate a pipeline file for creating multiple sequences based on iterations,
#   when multiple solutions to TSP exist.
# different solvestrs represent different preprocessing of the percent matches matrix.
# based on the same loop structure in workflow.bash order-iter

import argparse


## argparse

parser = argparse.ArgumentParser(description='run_wafer_aggregator.py')
parser.add_argument('--cluster', nargs=1, type=str, default=['soma'],
                    choices=['soma', 'axon'], help='which cluster to generate for')
args = parser.parse_args()
args = vars(args)

## params that are set by command line arguments

# which cluster to generate for
cluster = args['cluster'][0]


## fixed params

iter_strs = {
'soma':
"""
solve___{}   {:<12} run_wafer_solver-solve-{:<7}           CPU/GPU/CPU-72c  8/8/8     0/0/0  1/1/1  0  0
affines_{}   solve___{}   run_wafer_solver-affines-{:<7}         GPU              16        4      0      1  0
maffines{}   affines_{}   run_wafer_solver-merge_affines-{:<7}   CPU/GPU/CPU-72c  8/8/8     0/0/0  0/0/0  0  0
writeseq{}   maffines{}   run_wafer_aggregator-status-{:<7}      CPU/GPU/CPU-72c  8/8/8     0/0/0  0/0/0  0  0
""",

'axon':
"""
solve___{}   {:<12} run_wafer_solver-solve-{:<7}           p.axon/p.gpu   2/2     0/0   1/1   0    0
affines_{}   solve___{}   run_wafer_solver-affines-{:<7}         p.gpu          6       1     0     0    0
maffines{}   affines_{}   run_wafer_solver-merge_affines-{:<7}   p.axon/p.gpu   2/2     0/0   0/0   0    0
writeseq{}   maffines{}   run_wafer_aggregator-status-{:<7}      p.axon/p.gpu   2/2     0/0   0/0   0    0
"""
}

# normal order solving runs
# could modify depending on the difficulty of the order solving
solvestrs=["all", "top6", "top12", "top24", "norm", "normm"]
niters=5

## order solving sensitivity test
#solvestrs=["zero", "one", "two", "five", "ten", "fifteen", "twenty", "thirty", "fourty", "fifty"]
#niters=1


cnt=1
for i in range(len(solvestrs)):
    for j in range(niters):
        args = [cnt, cnt, solvestrs[i] + '-{}'.format(j+1)]*4
        args[1] = ',' if i==0 and j==0 else 'writeseq{}'.format(cnt-1)
        print(iter_strs[cluster].format(*args))
        cnt += 1
