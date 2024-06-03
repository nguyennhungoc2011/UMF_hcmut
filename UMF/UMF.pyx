########################################################
# UMF.pyx
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Updated: 2024/5/30 by Ngoc Nhu Nguyen
########################################################

import time
import numpy as np
cimport numpy as np # import C-API
from libcpp cimport bool


#########################################################
# Make declarations on functions from cpp file
#
cdef extern from "c_UMF.h":
    void UMF(double *removedData, double *historyData, int numUser, int numService, int dim, 
    double lmda, int maxIter, double convergeThreshold, double eta, 
    double beta, double alpha, bool debugMode, double *Udata, double *Sdata, double *p,
    double *q, double *predData)
#########################################################


#########################################################
# Function to perform the prediction algorithm
# Wrap up the C++ implementation
#
def predict(removedMatrix, historyMatrix, U, S, p, q, para):  
    cdef int numService = removedMatrix.shape[1] 
    cdef int numUser = removedMatrix.shape[0] 
    cdef int dim = para['dimension']
    cdef double lmda = para['lambda']
    cdef int maxIter = para['maxIter']
    cdef double convergeThreshold = para['convergeThreshold']
    cdef double eta = para['eta']
    cdef double beta = para['beta']
    cdef double alpha = para['alpha']
    cdef bool debugMode = para['debugMode']
    cdef np.ndarray[double, ndim=2, mode='c'] predMatrix = \
        np.zeros((numUser, numService), dtype=np.float64)

    # wrap up c_UMF.cpp
    UMF(
        <double *> (<np.ndarray[double, ndim=2, mode='c']> removedMatrix).data,
        <double *> (<np.ndarray[double, ndim=2, mode='c']> historyMatrix).data,
        numUser,
        numService,
        dim,
        lmda,
        maxIter,
        convergeThreshold,
        eta,
        beta,
        alpha,
        debugMode,
        <double *> (<np.ndarray[double, ndim=2, mode='c']> U).data,
        <double *> (<np.ndarray[double, ndim=2, mode='c']> S).data,
        <double *> (<np.ndarray[double, ndim=1, mode='c']> p).data,
        <double *> (<np.ndarray[double, ndim=1, mode='c']> q).data,
        <double *> predMatrix.data
        )

    return predMatrix
#########################################################




