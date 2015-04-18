# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:06:02 2015

@author: Jason
"""

import paths
import numpy
import cPickle
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

phonemes = numpy.loadtxt(paths.pathToChrMap,dtype='str_',usecols=(0,))
phonId = numpy.loadtxt(paths.pathToChrMap,dtype='int',usecols=(1,))

#%% Extract Sentence Ids


ph48_39 = numpy.loadtxt(paths.pathToMapPhones,dtype='str_',delimiter='\t')
phi_48 = dict(zip(numpy.arange(0,48),ph48_39[:,0]))
phonemes2id = dict(zip(phonemes,phonId))


def pack_sentences(features, labels, ids):
    data_points = list()
    
    currentX = list()
    currentY = list()
    currentSentenceId = ''
    vectorIndex = -1
    for feature, label, frameId in zip(features, labels, ids):
        frameData = frameId.split('_')
        sentenceId = frameData[0]+'_'+frameData[1]    
        if not sentenceId == currentSentenceId:
            if len(currentX) > 0:
                data_points.append((numpy.asarray(currentX),numpy.asarray(currentY), currentSentenceId))
                currentX = list()
                currentY = list()
            vectorIndex += 1
        currentSentenceId =  sentenceId
        phoneme = phi_48[label]
        phonemeNo = phonemes2id[phoneme]
        currentY.append(phonemeNo)
        for index in range(69):
            currentX.append(feature[index])
    if len(currentX) > 0:
        data_points.append((numpy.asarray(currentX),numpy.asarray(currentY), currentSentenceId))
    return data_points
    

data_points = pack_sentences(fbank_feat,fbank_labels, trainIds)
test_points = pack_sentences(fbank_test_feat, numpy.zeros(shape=(fbank_test_feat.shape[0],)), testIds)

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

phonPhones = numpy.loadtxt(paths.pathToChrMap,dtype='str_',usecols=(0,))
phonId = numpy.loadtxt(paths.pathToChrMap,dtype='int',usecols=(1,))
phonLetters = numpy.loadtxt(paths.pathToChrMap,dtype='str_',usecols=(2,))
ph48_39 = numpy.loadtxt(paths.pathToMapPhones,dtype='str_',delimiter='\t')
phonemeId2ph48 = dict(zip(phonId,phonPhones))
ph482ph39 = dict(zip(ph48_39[:,0],ph48_39[:,1]))
phones2Letter = dict(zip(phonPhones,phonLetters))

def computeResponse(ys):
    letters = (phones2Letter[ph482ph39[phonemeId2ph48[y]]] for y in ys)
    letters_out = list()
    lastLetter = ''
    for l in letters:
        if l == lastLetter:
            continue
        letters_out.append(l)
        lastLetter = l
    code = ''.join(letters_out).replace('K',' ').strip().replace(' ','K')
    return code
    
f = file(paths.pathToFBANKSVMValidateCodes,'wb')
for xs, ys, sent in validation_data:
    f.write(computeResponse(ys)+'\n')
f.close()

