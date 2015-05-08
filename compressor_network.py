# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:09:30 2015

@author: jan
"""

import theano
import cPickle
import numpy
import paths
from blocks.bricks import Rectifier, Softmax, MLP
from blocks.initialization import IsotropicGaussian, Constant
from blocks.filter import VariableFilter
import blocks.graph
from sklearn import preprocessing

#%%
trainIds = numpy.loadtxt(paths.pathToFBANKTrain,dtype='str_',usecols=(0,))

f = file(paths.pathToSaveFBANKTrain,'rb')
fbank_data = cPickle.load(f)
f.close()
f = file(paths.pathToSentenceGroupIds,'rb')
sentenceGroupIds = cPickle.load(f)
f.close()
f = file(paths.pathToSave39Labels,'rb')
fbank_labels = cPickle.load(f)
f.close()

scaler = preprocessing.StandardScaler().fit(fbank_data)

fbank_std = scaler.transform(fbank_data)

speakerIds = set(sentenceId.split('_')[0] for sentenceId in trainIds)

speakerIdsDict = dict(zip(speakerIds,range(len(speakerIds))))

speakerIdsInt = numpy.asarray([speakerIdsDict[sentenceId.split('_')[0]] for sentenceId in trainIds])

#%%

if 'tX' in globals():
    tX = None
if 'vX' in globals():
    vX = None

rng = numpy.random.RandomState(123)
randomOrderSentIds = rng.permutation(list(set(sentenceGroupIds)))

validationSetSize = 0
validationSet = list()
i = 0
while validationSetSize/float(fbank_data.shape[0]) < 0.2:
    validationSet.append(randomOrderSentIds[i])
    validationSetSize = numpy.sum(numpy.in1d(sentenceGroupIds,(validationSet)))
    i+=1

validationSelection = numpy.in1d(sentenceGroupIds,(validationSet))
trainingSelection = validationSelection==0

print '#Training: {}, #Validation: {}'.format(numpy.sum(trainingSelection==1),numpy.sum(validationSelection==1))

tX = theano.shared(numpy.asarray(fbank_std[trainingSelection,:],dtype=theano.config.floatX),borrow=True)
vX = theano.shared(numpy.asarray(fbank_std[validationSelection,:],dtype=theano.config.floatX),borrow=True)
vYb = theano.tensor.cast(theano.shared(numpy.asarray(fbank_labels[validationSelection],dtype=theano.config.floatX),borrow=True),'int32')
tYb = theano.tensor.cast(theano.shared(numpy.asarray(fbank_labels[trainingSelection],dtype=theano.config.floatX),borrow=True),'int32')

#%%
x0 = theano.tensor.matrix('x0')
x1 = theano.tensor.matrix('x1')
x2 = theano.tensor.matrix('x2')
x3 = theano.tensor.matrix('x3')
y_b = theano.tensor.ivector('y_b')
idx = theano.tensor.lscalar('idx')
shuffIdx = theano.tensor.imatrix('shuffIdx')

batch_size = 128
NBatches = int(numpy.floor(
	tX.get_value(borrow=True).shape[0]/batch_size))
validationBatchSize = 2048;
NTestBatches = int(numpy.floor(
	vX.get_value(borrow=True).shape[0]/validationBatchSize))
 
#%%

def momentum_sgd(cost, params, momentum, lr_var):
    assert momentum < 1 and momentum >= 0
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - lr_var*param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*theano.tensor.grad(cost, param)))
    return updates
    
def setupNN(NNParam):
    NNWidth = NNParam['NNWidth']
    WeightStdDev = NNParam['WeightStdDev']
    L2Weight = NNParam['L2Weight']
    DropOutProb = NNParam['DropOutProb']
    InitialLearningRate = NNParam['InitialLearningRate']
    x = theano.tensor.concatenate([x0, x1, x2, x3], axis=1)
    mlp = MLP(activations=[Rectifier(), Rectifier(), Rectifier(), Rectifier(), Rectifier()], dims=[69*4, NNWidth, NNWidth, NNWidth, NNWidth, 100],
           weights_init=IsotropicGaussian(WeightStdDev),
           biases_init=Constant(0))

    x_forward = mlp.apply(x)
    mlp_sm = MLP(activations=[None], dims=[100, 39],
           weights_init=IsotropicGaussian(WeightStdDev),
           biases_init=Constant(0))
    y_hat_b = Softmax().apply(mlp_sm.apply(x_forward))
    mlp.initialize()
    mlp_sm.initialize()
    cg = blocks.graph.ComputationGraph(y_hat_b)
    parameters = list()
    for p in cg.parameters:
        parameters.append(p)
    weights = VariableFilter(roles=[blocks.roles.WEIGHT])(cg.variables)
    cg_dropout = blocks.graph.apply_dropout(cg,[weights[3]] , DropOutProb)
    y_hat_b_do = cg_dropout.outputs[0]
    pred_b = theano.tensor.argmax(cg.outputs[0],axis=1)
    err_b = theano.tensor.mean(theano.tensor.eq(pred_b,y_b))
    cW = 0
    for W in weights:
        cW += (W**2).sum()
    cost = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(y_hat_b_do, y_b))  + cW*L2Weight


    Learning_Rate_Decay = numpy.float32(0.98)
    learning_rate_theano = theano.shared(numpy.float32(InitialLearningRate), name='learning_rate')

    learning_rate_update = theano.function(inputs=[],outputs=learning_rate_theano,updates=[(learning_rate_theano,learning_rate_theano*Learning_Rate_Decay)])
    update_proc = momentum_sgd(cost,parameters,0.8, learning_rate_theano)

    #train
    training_proc = theano.function(
        	inputs=[shuffIdx], outputs=cost, updates=update_proc,
        	givens={x0:tX[theano.tensor.flatten(shuffIdx[:,0])],
                x1:tX[theano.tensor.flatten(shuffIdx[:,1])],
                x2:tX[theano.tensor.flatten(shuffIdx[:,2])],
                x3:tX[theano.tensor.flatten(shuffIdx[:,3])],
                y_b:tYb[theano.tensor.flatten(shuffIdx[:,1])]}) 
    #test
    test_on_testing_proc = theano.function(
        	inputs=[shuffIdx], outputs=[err_b], 
        	givens={x0:vX[shuffIdx[:,0]],x1:vX[shuffIdx[:,1]],x2:vX[shuffIdx[:,2]],x3:vX[shuffIdx[:,3]],y_b:vYb[shuffIdx[:,1]]}) 
       
    test_on_training_proc = theano.function(
        	inputs=[shuffIdx], outputs=[err_b], 
        	givens={x0:tX[shuffIdx[:,0]],x1:tX[shuffIdx[:,1]],x2:tX[shuffIdx[:,2]],x3:tX[shuffIdx[:,3]],y_b:tYb[shuffIdx[:,1]]}) 

    forward_proc = theano.function(inputs=[x0,x1,x2,x3],outputs=[x_forward])
    return (learning_rate_update, training_proc, test_on_testing_proc,test_on_training_proc,forward_proc)

def runNN(NNProcs, NBatches, NEpochs):
    recorded_data = numpy.zeros(shape=(NEpochs,4))
    learning_rate_update = NNProcs[0]
    training_proc = NNProcs[1]
    test_on_testing_proc = NNProcs[2]
    test_on_training_proc = NNProcs[3]
    NTrainingPoints = numpy.sum(trainingSelection==1)
    NValidationPoints = numpy.sum(validationSelection==1)
    epochPermMult   = numpy.zeros(shape=(NTrainingPoints,4)).astype('int32')
    trainingBatchIds = numpy.arange(NTrainingPoints)
    trainingBatchMult = numpy.zeros(shape=(NTrainingPoints,4)).astype('int32')
    trainingBatchMult[:,1] = trainingBatchIds-1
    trainingBatchMult[:,0] = trainingBatchIds-2    
    trainingBatchMult[:,2] = trainingBatchIds  
    trainingBatchMult[:,3] = trainingBatchIds+1
    trainingBatchMult[trainingBatchMult[:,1]<0,1] = 0
    trainingBatchMult[trainingBatchMult[:,0]<0,0] = 0
    trainingBatchMult[trainingBatchMult[:,3]>=NTrainingPoints,3] = NTrainingPoints-1
    
    validationBatchIds = numpy.arange(NValidationPoints).astype('int32')
    validationBatchMult = numpy.zeros(shape=(NValidationPoints,4)).astype('int32')
    validationBatchMult[:,1] = validationBatchIds-1
    validationBatchMult[:,0] = validationBatchIds-2
    validationBatchMult[:,2] = validationBatchIds
    validationBatchMult[:,3] = validationBatchIds+1
    validationBatchMult[validationBatchMult[:,0]<0,0] = 0
    validationBatchMult[validationBatchMult[:,1]<0,1] = 0
    validationBatchMult[validationBatchMult[:,3]>=NValidationPoints,3] = NValidationPoints-1
    NValidationBatches = int(numpy.floor(NValidationPoints/validationBatchSize))
    NTrainBatches = int(numpy.floor(NTrainingPoints/validationBatchSize))
    for epoch in xrange(NEpochs):
        epochPerm = rng.permutation(NTrainingPoints).astype('int32')
        epochPermMult[:,1] = epochPerm-1
        epochPermMult[:,0] = epochPerm-2
        epochPermMult[:,2] = epochPerm
        epochPermMult[:,3] = epochPerm+1
        epochPermMult[epochPermMult[:,3]>=NTrainingPoints,3] = NTrainingPoints-1
        epochPermMult[epochPermMult[:,0]<=0,0] = 0
        epochPermMult[epochPermMult[:,1]<=0,1] = 0
        avg_cost = 0
        for minibatch_i in xrange(NBatches):
            avg_cost += training_proc(epochPermMult[minibatch_i*batch_size:(minibatch_i+1)*batch_size,:])  
    
        current_accuracy = numpy.mean([test_on_testing_proc(validationBatchMult[m*validationBatchSize:(m+1)*validationBatchSize,:]) for m in xrange(NValidationBatches)])
        trainingDataAcc = numpy.mean([test_on_training_proc(trainingBatchMult[m*validationBatchSize:(m+1)*validationBatchSize,:]) for m in xrange(NTrainBatches)])
        current_learning_rate = learning_rate_update()
        print 'Epoch %i, Vali Acc: %0.4f, Train Acc: %0.4f, Cost: %0.3f, LR: %0.3f' % (epoch,current_accuracy,numpy.mean(trainingDataAcc),avg_cost/NBatches,current_learning_rate)
        
        #print 'Accuracy on training data: %f' % numpy.mean(trainingDataAcc)    
        recorded_data[epoch,:] = (avg_cost/NBatches, current_learning_rate, trainingDataAcc, current_accuracy)
        
    return recorded_data

#%%
import time
start_time = time.clock()

NEpochs = 30

ILR = ( 0.1, )
NNW = (250,)
L2W = (0.00015,)
DOP = (0.1,)
WSD = (0.05,)

Grid = {'NNWidth':NNW,'WeightStdDev':WSD,'L2Weight':L2W,'DropOutProb':DOP,'InitialLearningRate':ILR}

currentIndex = dict()
maxIndex = dict()
keyList = list()

for k in Grid.iterkeys():
    currentIndex[k] = 0
    maxIndex[k] = len(Grid[k])-1
    keyList.append(k)
resultList = list()
stop=False   
while not stop:
    NNParam = dict()
    for k in Grid.iterkeys():
        NNParam[k] = Grid[k][currentIndex[k]]
    print 'Running NN on {}'.format(NNParam)
    NNProcedures = setupNN(NNParam)
    recData = runNN(NNProcedures,NBatches, NEpochs)
    print 'Maximum accuracy on test: {}'.format(numpy.max(recData[:,3]))
    resultList.append((NNParam,recData))
    currentIndex[keyList[0]]+=1
    for i in range(len(keyList)):
        key=keyList[i]
        if currentIndex[key] > maxIndex[key]:
            currentIndex[key] = 0
            if i == len(keyList)-1:
                stop = True
                break
            else:
                currentIndex[keyList[i+1]]+=1
            
stop_time = time.clock()

 
#%%

def getParallelBatchIds(N):
    batchIds = numpy.arange(N).astype('int32')
    batchMult = numpy.zeros(shape=(N,4)).astype('int32')
    batchMult[:,1] = batchIds-1
    batchMult[:,0] = batchIds-2
    batchMult[:,2] = batchIds
    batchMult[:,3] = batchIds+1
    batchMult[batchMult[:,0]<0,0] = 0
    batchMult[batchMult[:,1]<0,1] = 0
    batchMult[batchMult[:,3]>=N,3] = N-1
    return batchMult
 
forward_proc = NNProcedures[4]
trainingData = numpy.zeros(shape=(sum(trainingSelection==True),100))
validationData = numpy.zeros(shape=(sum(validationSelection==True),100))
trainingIndex = 0
validationIndex = 0
trainingIndices = numpy.ravel(numpy.asarray(numpy.where(trainingSelection)))
fw_batch_size = 1
trainBatches = getParallelBatchIds(trainingIndices.shape[0])
for i in range(int(numpy.ceil(trainingIndices.shape[0]/fw_batch_size))):
    start = i*fw_batch_size
    stop = min(((i+1)*fw_batch_size,trainingIndices.shape[0]))
    trainingData[start:stop,:] = forward_proc(numpy.asarray(fbank_std[trainingIndices[trainBatches[start:stop,0]],:],dtype=theano.config.floatX),
                       numpy.asarray(fbank_std[trainingIndices[trainBatches[start:stop,1]],:],dtype=theano.config.floatX),
                        numpy.asarray(fbank_std[trainingIndices[trainBatches[start:stop,2]],:],dtype=theano.config.floatX),
                        numpy.asarray(fbank_std[trainingIndices[trainBatches[start:stop,3]],:],dtype=theano.config.floatX))
validationIndices = numpy.ravel(numpy.asarray(numpy.where(validationSelection)))
validationBatches = getParallelBatchIds(validationIndices.shape[0])
for i in range(int(numpy.ceil(validationIndices.shape[0]/fw_batch_size))):
    start = i*fw_batch_size
    stop = min(((i+1)*fw_batch_size,validationIndices.shape[0]))
    validationData[start:stop,:] = forward_proc(numpy.asarray(fbank_std[validationIndices[validationBatches[start:stop,0]],:],dtype=theano.config.floatX),
                       numpy.asarray(fbank_std[validationIndices[validationBatches[start:stop,1]],:],dtype=theano.config.floatX),
                        numpy.asarray(fbank_std[validationIndices[validationBatches[start:stop,2]],:],dtype=theano.config.floatX),
                        numpy.asarray(fbank_std[validationIndices[validationBatches[start:stop,3]],:],dtype=theano.config.floatX))


#for i in range(trainingSelection.shape[0]):
#    res = forward_proc(numpy.asarray(fbank_std[(i, 0),:],dtype=theano.config.floatX))
#    if trainingSelection[i]:
#        trainingData[trainingIndex,:] = res[0][0]
#        trainingIndex+=1
#    else:
#        validationData[validationIndex,:] = res[0][0]
#        validationIndex+=1
    
trainingSentences = trainIds[trainingSelection]
validationSentences = trainIds[validationSelection]

validationLabels = fbank_labels[validationSelection]
trainingLabels = fbank_labels[trainingSelection]
#%%
import coding
training_packed = coding.pack_sentences(trainingData,trainingLabels,trainingSentences)
validation_packed = coding.pack_sentences(validationData,validationLabels,validationSentences)

#%%
coding.print_svm_data(training_packed,'../data/fbank_nn_svm_training.save')
coding.print_svm_data(validation_packed,'../data/fbank_nn_svm_validation.save')

coding.print_validation_codes(validation_packed,'../data/fbank_nn_svm_validation.code')

#%%

testSentences = numpy.loadtxt(paths.pathToFBANKTest,dtype='str_',usecols=(0,))

f = file(paths.pathToSaveFBANKTest,'rb')
fbank_test = cPickle.load(f)
f.close()

fbank_test_std = scaler.transform(fbank_test)

testingData = numpy.zeros(shape=(fbank_test_std.shape[0],100))
print 'Forwarding test data'
fw_batch_size = 1
testBatches = getParallelBatchIds(fbank_test_std.shape[0])
for i in range(int(numpy.ceil(fbank_test_std.shape[0]/fw_batch_size))):
    start = i*fw_batch_size
    stop = min(((i+1)*fw_batch_size,fbank_test_std.shape[0]))
    testingData[start:stop,:] = forward_proc(numpy.asarray(fbank_test_std[testBatches[start:stop,0],:],dtype=theano.config.floatX),
                       numpy.asarray(fbank_test_std[testBatches[start:stop,1],:],dtype=theano.config.floatX),
                        numpy.asarray(fbank_test_std[testBatches[start:stop,2],:],dtype=theano.config.floatX),
                        numpy.asarray(fbank_test_std[testBatches[start:stop,3],:],dtype=theano.config.floatX))

testLabels = numpy.zeros(shape=(fbank_test_std.shape[0],))
test_packed = coding.pack_sentences(testingData,testLabels,testSentences)
coding.print_svm_data(test_packed,'../data/fbank_nn_svm_test.save')
coding.print_sentence_ids(test_packed,paths.pathToFBANKSVMTestingSents)

#%%

phonemeStreakLengths = dict([(i, list()) for i in range(39)])

for xs, ys, sent in validation_packed:
    currPhon = ys[0]
    currLen = 1
    for i in range(1,len(ys)):
        if ys[i] != currPhon:
            phonemeStreakLengths[currPhon].append(currLen)
            currLen = 1
            currPhon = ys[i]
        else:
            currLen+=1
    phonemeStreakLengths[currPhon].append(currLen)
    
#%%
    
for k,v in phonemeStreakLengths.iteritems():
    arr = numpy.array(v)
    print 'Phoneme {}: [{},{}] ~={}'.format(k,numpy.min(arr),numpy.max(arr),numpy.mean(arr))
    
#%% 

tM = numpy.zeros(shape=(39,39))
for xs, ys, sent in training_packed:
    lastPhon = ys[0]
    for i in range(1,len(ys)):
        tM[ys[i-1],ys[i]]+=1
for xs, ys, sent in validation_packed:
    lastPhon = ys[0]
    for i in range(1,len(ys)):
        tM[ys[i-1],ys[i]]+=1
for i in range(39):
    string = ''
    for j in range(39):
        string += '{0:.3f} '.format(tM[i,j]/numpy.sum(tM[i,:])*100)
    print string
        
