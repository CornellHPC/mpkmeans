This contains instructions for agents to set up an evluation workflow for the mixed precision k-means algorithm implemented in this codebase

The workflow should involve three scripts:
(1) Dataset prep -- download all needed datasets and prepare them with any necessary preprocessing into a specified directory
(2) Jobscript generation -- generate slurm jobscripts for running experiments - there should be fine grained scripts i.e. one per config and a global run_all.sh which will run all of them
(3) Plotting -- process the results of the experiments into plots

It's fine to have more scripts in this workflow, but the user should only need to run these three. 

Binary:

use the build/mpkmeans_bench binary 
run with and without fp32 accum
maxiters=400 and --convergence flag always


Datasets:

synthetically generated with default -b
synthetically generated with -b 5
synthetically generated with -b 2
synthetically generated with -b 0

synthetic datasets should do n=200k and n=1M with sweeps of d and k from 32 to 1024 at powers of 2

libSVM datasets
- MNIST8m
- covtype.binary
- Wiki10-31K
- news20
- real-sim
- epsilon

vector indexing datasets -- described in (https://github.com/cwida/SuperKMeans/blob/main/BENCHMARKING.md)
- openAI
- arxiv
- wiki

All should be run with and without z-score normalization


