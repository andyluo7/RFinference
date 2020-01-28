import os
import sys
import time

num_trees = [10**2, 10**3, 10**4, 10**5]
num_samples = [10**0, 10**1, 10**2, 10**3, 10**4, 10**5]
modes = [0] #0 for Classifier, 1 for Regressor


start_time = time.clock()

for numSamples in num_samples:
    for nTrees in num_trees:
        for mode in modes:
            execString = "python3 run_Xl_benchmark_single.py "+ sys.argv[1] + " " + sys.argv[2] + " " + str(nTrees) + " " + str(mode) + " " + str(numSamples)
            os.system(execString)
