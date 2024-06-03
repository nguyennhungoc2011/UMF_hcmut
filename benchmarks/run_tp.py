########################################################
# run_rt.py 
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Updated: 2024/5/30 by Ngoc Nhu Nguyen
########################################################

import numpy as np
import os, sys, time
from commons.utils import logger
from commons import utils
from commons import evaluator
from commons import dataloader
 

# parameter config area
para = {'dataPath': '../data/',
        'dataName': 'dataset2',
        'dataType': 'tp', # set the dataType as 'rt' or 'tp'
        'outPath': 'result/',
        'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NPRE'], # delete where appropriate      
        'density': np.arange(0.05, 0.31, 0.05), # matrix density
        'rounds': 10, # how many runs are performed at each matrix density
        'dimension': 10, # dimenisionality of the latent factors
        'eta': 0.8, # learning rate
        'lambda': 0.0002, # regularization parameter
        'maxIter': 300, # the max iterations
        'convergeThreshold': 1e-2, # stopping criteria for convergence
        'beta': 0.3, # the controlling weight of exponential moving average
        'alpha': 0.6, # the controlling weight of exponential moving average
        'saveTimeInfo': True, # whether to keep track of the running time
        'saveLog': True, # whether to save log into file
        'debugMode': False, # whether to record the debug info
        'parallelMode': True # whether to leverage multiprocessing for speedup
        }


startTime = time.time() # start timing
utils.setConfig(para) # set configuration
logger.info('==============================================')
logger.info('UMF: Updated Adaptive Matrix Factorization [TPDS]')

# load the dataset
dataTensor = dataloader.load(para)

# evaluate QoS prediction algorithm
evaluator.execute(dataTensor, para)

logger.info('All done. Elaspsed time: ' + utils.formatElapsedTime(time.time() - startTime)) # end timing
logger.info('==============================================')
