# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:06:02 2015

@author: Jason
"""

import paths
import numpy
import cPickle
import coding
#%% Read Data

trainIds = numpy.loadtxt(paths.pathToFBANKTrain,dtype='str_',usecols=(0,))
f = file(paths.pathToSaveFBANKTrain,'rb')
fbank_feat = cPickle.load(f)
f.close()

testIds = numpy.loadtxt(paths.pathToFBANKTest,dtype='str_',usecols=(0,))
f = file(paths.pathToSaveFBANKTest,'rb')
fbank_test_feat = cPickle.load(f)
f.close()

f = file(paths.pathToSave48Labels,'rb')
fbank_labels = cPickle.load(f)
f.close()


#%% Extract Sentence Ids
    

data_points = coding.pack_sentences(fbank_feat,fbank_labels, trainIds)
test_points = coding.pack_sentences(fbank_test_feat, numpy.zeros(shape=(fbank_test_feat.shape[0],)), testIds)

#%% Make a split

rng = numpy.random.RandomState(1234)
selection = rng.uniform(size=(len(data_points),)) < 0.8
training_index = numpy.arange(len(data_points))[selection]
validation_index = numpy.arange(len(data_points))[selection==False]

training_data = list()
for k in training_index:
    training_data.append(data_points[k])

validation_data = list()
for k in validation_index:
    validation_data.append(data_points[k])


#%% 
           

f = file(paths.pathToFBANKSVMTraining,'wb')

sentenceId = -1
for xs, ys, sent in training_data:
    sentenceId += 1
    symbolId = -1
    for k in range(len(ys)):
        symbolId+=1
        y = ys[k]
        x = xs[k*69:(k+1)*69]
        f.write('%i %i %i %i %i ' % (int(len(training_data)), sentenceId, int(len(ys)), symbolId, y) + ' '.join([str(x_) for x_ in x]) + '\n' )
        

f.close()

f = file(paths.pathToFBANKSVMValidate,'wb')
f_sent = file(paths.pathToFBANKSVMValidateSents,'wb')

sentenceId = -1
for xs, ys, sent in validation_data:
    sentenceId += 1
    symbolId = -1
    for k in range(len(ys)):
        symbolId+=1
        y = ys[k]
        x = xs[k*69:(k+1)*69]
        f.write('%i %i %i %i %i ' % (int(len(validation_data)), sentenceId, int(len(ys)), symbolId, y) + ' '.join([str(x_) for x_ in x]) + '\n' )
    f_sent.write(sent+'\n')

f_sent.close()
f.close()

f = file(paths.pathToFBANKSVMTesting,'wb')
f_sent = file(paths.pathToFBANKSVMTestingSents,'wb')

sentenceId = -1
for xs, ys, sent in test_points:
    sentenceId += 1
    symbolId = -1
    for k in range(len(ys)):
        symbolId+=1
        y = ys[k]
        x = xs[k*69:(k+1)*69]
        f.write('%i %i %i %i %i ' % (int(len(test_points)), sentenceId, int(len(ys)), symbolId, y) + ' '.join([str(x_) for x_ in x]) + '\n' )
    f_sent.write(sent+'\n')

f_sent.close()
f.close()


#%%
  
f = file(paths.pathToFBANKSVMValidateCodes,'wb')
for xs, ys, sent in validation_data:
    f.write(coding.computeResponse(ys)+'\n')
f.close()

