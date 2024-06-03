########################################################
# evaluator.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Updated: 2024/5/30 by Ngoc Nhu Nguyen
########################################################

import numpy as np 
import time
from utils import logger
import evallib
import UMF
from scipy import stats, special
import multiprocessing
import random

#======================================================#
# Function to plot distribution
#======================================================#
def plotdistribution(datalist, para):
    dataType = 'Response_Time'
    color = 'black'

    if para['dataType'] is 'tp':
      dataType = 'Throughput'
      color = 'red'
    attribute = datalist[3]

    plt.figure(figsize=(10, 6))
    plt.hist(attribute, bins=50, density=True, alpha=0.6, color=color)
    plt.title('Distribution of '+ dataType)
    plt.xlabel(dataType)
    plt.ylabel('Probability Density')
    plt.grid(True)

    # Save the figure
    plt.savefig('wsdream_'+ dataType +'_distribution.png')

#======================================================#
# Function to evalute the approach at all settings
#======================================================#
def execute(tensor, para):
    # loop over each density and each round
    if para['parallelMode']: # run on multiple processes
        pool = multiprocessing.Pool()
        for den in para['density']: 
            for roundId in xrange(para['rounds']):
                pool.apply_async(executeOneSetting, (tensor, den, roundId, para))
        pool.close()
        pool.join()
    else: # run on single processes
        for den in para['density']:
            for roundId in xrange(para['rounds']):
                executeOneSetting(tensor, den, roundId, para)
    # summarize the dumped results
    print('# of Timeslices: '+ str(tensor.shape[2]))
    evallib.summarizeResult(para, tensor.shape[2])


#======================================================#
# Function to run the prediction approach at one setting
#======================================================#
def executeOneSetting(tensor, density, roundId, para):
    logger.info('density=%.2f, %2d-round starts.'%(density, roundId + 1))
    (numUser, numService, numTime) = tensor.shape
    dim = para['dimension']

    # initialization
    U = np.random.rand(numUser, dim)
    S = np.random.rand(numService, dim)
    p = np.zeros(numUser)
    q = np.zeros(numService)
    historyData = np.zeros((numUser,numService))
    # run for each time slice
    for sliceId in xrange(0, numTime):
        # boxcox data transformation
        matrix = tensor[:, :, sliceId]
        dataVector = matrix[:]
        (transfVector, alpha) = stats.boxcox(dataVector[dataVector > 0])
        maxV = np.max(transfVector)
        minV = np.min(transfVector)
        transfMatrix = matrix.copy()
        transfMatrix[transfMatrix != -1] = stats.boxcox(transfMatrix[transfMatrix != -1], alpha)
        transfMatrix[transfMatrix != -1] = (transfMatrix[transfMatrix != -1] - minV) / (maxV - minV)

        # remove data entries to generate trainMatrix and testMatrix  
        seedID = roundId + sliceId * 100
        (trainMatrix, testMatrix) = evallib.removeEntries(matrix, density, seedID)
        trainMatrix = np.where(trainMatrix > 0, transfMatrix, 0)
        (testVecX, testVecY) = np.where(testMatrix)     
        testVec = matrix[testVecX, testVecY]

        # invocation to the prediction function
        startTime = time.clock() 
        predictedMatrix = UMF.predict(trainMatrix, historyData, U, S, p, q, para)
        historyData = trainMatrix.copy()     
        runningTime = float(time.clock() - startTime)

        # evaluate the estimation error  
        predVec = predictedMatrix[testVecX, testVecY]
        predVec = (maxV - minV) * predVec + minV
        predVec = evallib.argBoxcox(predVec, alpha)
        evalResult = evallib.errMetric(testVec, predVec, para['metrics'])
        result = (evalResult, runningTime)

        # dump the result at each density
        outFile = '%s%s_%s_result_%02d_%.2f_round%02d.tmp'%(para['outPath'], 
            para['dataName'], para['dataType'], sliceId + 1, density, roundId + 1)
        evallib.dumpresult(outFile, result)
        logger.info('sliceId=%02d done.'%(sliceId + 1))
    #plotdistribution(transfDataList, para)
    logger.info('density=%.2f, %2d-round done.'%(density, roundId + 1))
    logger.info('----------------------------------------------')








