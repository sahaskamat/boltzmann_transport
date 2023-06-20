import multiprocessing
from joblib import delayed, Parallel
import numpy as np

#find number of cpus
try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2   # arbitrary default

if cpus >60: cpus =60 #joblib breaks if you use too many CPUS (>61)

def makelist_parallel(sigmaarea_function,inputslist):
    """
    Inputs:
    function (lambda function that returns the conductivity tensor and area sigma,area after taking in one input)
    inputslist (list of inputs over which function is calculated)

    Makes list of conductivity tensors by repeatedly applying function to inputslist parallelly

    Outputs: sigmalist,rholist,arealist
    """
    #execute sigma_function paralelly over thetalist to obtain sigmas and areas
    sigmaarealist = Parallel(n_jobs=int(cpus))(delayed(sigmaarea_function)(inputinstance) for inputinstance in inputslist)

    sigmalist = [sigmaarealist[i][0] for i in range(len(sigmaarealist))]
    arealist = [sigmaarealist[i][1] for i in range(len(sigmaarealist))]

    rholist = [] #invert sigmas to find rhos
    for sigma in sigmalist:
        rholist.append(np.linalg.inv(sigma))

    return sigmalist,rholist,arealist