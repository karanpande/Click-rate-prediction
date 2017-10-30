# Click Trough Rate (CTR)
# Language: Python
# Dataset: Criteo public dataset
# Data information:
# •	Label - Target variable that indicates if an ad was clicked (1) or not (0).
# •	I1-I13 - A total of 13 columns of integer features (mostly count features).
# •	C1-C26 - A total of 26 columns of categorical features. The values of these #features have been hashed onto 32 bits for anonymization purposes. 
# The semantic of the features is undisclosed.
# When a value is missing, the field is empty.
# Dataset: Dataset consist of two files 
# 1: Train.csv
# 2: Test.csv

# Code

# Importing the dataset

import os.path
baseDir = os.path.join('data')
inputPath = os.path.join('cs190', 'dac_sample.txt')
fileName = os.path.join(baseDir, inputPath)
 
if os.path.isfile(fileName):
    rawData = (sc
               .textFile(fileName, 2)
               .map(lambda x: x.replace('\t', ',')))  # work with either ',' or '\t' separated data
    print rawData.take(1)

import glob
from io import BytesIO
import os.path
import tarfile
import urllib
import urlparse
 

url = 'http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz'
 
url = url.strip()
 
if 'rawData' in locals():
    print 'rawData already loaded.  Nothing to do.'
elif not url.endswith('dac_sample.tar.gz'):
    print 'Check your download url.  Are you downloading the Sample dataset?'
else:
    try:
        tmp = BytesIO()
        urlHandle = urllib.urlopen(url)
        tmp.write(urlHandle.read())
        tmp.seek(0)
        tarFile = tarfile.open(fileobj=tmp)
 
        dacSample = tarFile.extractfile('dac_sample.txt')
        dacSample = [unicode(x.replace('\n', '').replace('\t', ',')) for x in dacSample]
        rawData  = (sc
                    .parallelize(dacSample, 1)  # Create an RDD
                    .zipWithIndex()  # Enumerate lines
                    .map(lambda (v, i): (i, v))  # Use line index as key
                    .partitionBy(2, lambda i: not (i < 50026))  # Match sc.textFile partitioning
                    .map(lambda (i, v): v))  # Remove index
        print 'rawData loaded from url'
        print rawData.take(1)
    except IOError:
        print 'Unable to unpack: {0}'.format(url)


# Loading and splitting the dataset into training,validation and testing (ratio .8:.1:.1)


weights = [.8, .1, .1]
seed = 42
# Use randomSplit with weights and seed
rawTrainData, rawValidationData, rawTestData = rawData.randomSplit(weights,seed)
# Cache the data
rawTrainData.cache()
rawValidationData.cache()
rawTestData.cache()
 
nTrain = rawTrainData.count()
nVal = rawValidationData.count()
nTest = rawTestData.count()
print nTrain, nVal, nTest, nTrain + nVal + nTest
print rawData.take(1)




# HASHING FEATURE
from collections import defaultdict
import hashlib
 
def hashFunction(numBuckets, rawFeats, printMapping=False):
        mapping = {}
    for ind, category in rawFeats:
        featureString = category + str(ind)
        mapping[featureString] = int(int(hashlib.md5(featureString).hexdigest(), 16) % numBuckets)
    if(printMapping): print mapping
    sparseFeatures = defaultdict(float)
    for bucket in mapping.values():
        sparseFeatures[bucket] += 1.0
    return dict(sparseFeatures)



# Creating hashed features


def parseHashPoint(point, numBuckets):
    
    fields = point.split(',')
    label = fields[0]
    features = parsePoint(point)
    return LabeledPoint(label, SparseVector(numBuckets, hashFunction(numBuckets, features)))
 
numBucketsCTR = 2 ** 15
hashTrainData = rawTrainData.map(lambda point: parseHashPoint(point, numBucketsCTR))
hashTrainData.cache()
hashValidationData = rawValidationData.map(lambda point: parseHashPoint(point, numBucketsCTR))
hashValidationData.cache()
hashTestData = rawTestData.map(lambda point: parseHashPoint(point, numBucketsCTR))
hashTestData.cache()
 
print hashTrainData.take(1)


# Logistic model with hashed features

numIters = 500
regType = 'l2'
includeIntercept = True

bestModel = None
bestLogLoss = 10
stepSizes = [1, 10]
regParams = [1e-6, 1e-3]
for stepSize in stepSizes:
    for regParam in regParams:
        model = (LogisticRegressionWithSGD
                 .train(hashTrainData, numIters, stepSize, regParam=regParam, regType=regType,
                        intercept=includeIntercept))
        logLossVa = evaluateResults(model, hashValidationData)
        print ('\tstepSize = {0:.1f}, regParam = {1:.0e}: logloss = {2:.3f}'
               .format(stepSize, regParam, logLossVa))
        if (logLossVa < bestLogLoss):
            bestModel = model
            bestLogLoss = logLossVa
 
print ('Hashed Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossValBase, bestLogLoss))



# LOGLOSS Test on the Test dataset


logLossTest = evaluateResults(bestModel, hashTestData)
 
# Log loss for the baseline model
logLossTestBaseline = hashTestData.map(lambda lp: computeLogLoss(classOneFracTrain, lp.label)).mean()
 
print ('Hashed Features Test Log Loss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossTestBaseline, logLossTest))


###########################################

# Result
# Hashed Features Test
# Log Loss: Baseline = 0.537 
# LogReg = 0.457
############################################
