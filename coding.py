# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:47:12 2015

@author: jan
"""

import numpy
import paths

#paths.pathToChrMap has been replaced here
phonemes = numpy.loadtxt('48_idx_chr.map_b',dtype='str_',usecols=(0,))
phonId = numpy.loadtxt('48_idx_chr.map_b',dtype='int',usecols=(1,))
letters = numpy.loadtxt('48_idx_chr.map_b',dtype='str_',usecols=(2,))

#paths.pathToMapPhones has been replaced here
ph48_39 = numpy.loadtxt('48_39.map',dtype='str_',delimiter='\t')
ph48_39_dict = dict(ph48_39)
ph39_48 = dict(zip(ph48_39[:,1],ph48_39[:,0]))
phi_48 = dict(zip(numpy.arange(0,48),ph48_39[:,0]))
ph39_i = dict(zip(set(ph48_39[:,1]),numpy.arange(0,39)))
phi_39 = dict(zip(numpy.arange(0,39),set(ph48_39[:,1])))
phonemes2Letters = dict(zip(phonemes,letters))
id2letter = dict(zip(phonId,letters))

i39_L39 = dict()
for k,v in phi_39.items():
    i39_L39[k] = phonemes2Letters[v]
    
i48_i39 = dict()
for k,v in phi_48.items():
    i48_i39[k] = ph39_i[ph48_39_dict[v]]

i39_i48 = dict()
for k,v in i48_i39.iteritems():
    i39_i48[v] = k

def computeResponse(ys):
    letters = (i39_L39[y] for y in ys)
    letters_out = list()
    lastLetter = ''
    for l in letters:
        if l == lastLetter:
            continue
        letters_out.append(l)
        lastLetter = l
    code = ''.join(letters_out).replace('L',' ').strip().replace(' ','L')
    return code
    
def pack_sentences(features, labels, ids):
    data_points = list()
    useMapping = True
    if numpy.max(labels) == 38:
        useMapping = False
    currentX = list()
    currentY = list()
    currentSentenceId = ''
    vectorIndex = -1
    maxLength = 0;
    for feature, label, frameId in zip(features, labels, ids):
        frameData = frameId.split('_')
        sentenceId = frameData[0]+'_'+frameData[1]    
        if not sentenceId == currentSentenceId:
            if len(currentX) > 0:
                data_points.append((numpy.asarray(currentX),numpy.asarray(currentY), currentSentenceId))
                if len(currentY) > maxLength:
                    maxLength = len(currentY)
                currentX = list()
                currentY = list()
            vectorIndex += 1
        currentSentenceId =  sentenceId
        if useMapping:
            currentY.append(i48_i39[label])
        else:
            currentY.append(label)    
        for index in range(features.shape[1]):
            currentX.append(feature[index])
    if len(currentX) > 0:
        data_points.append((numpy.asarray(currentX),numpy.asarray(currentY), currentSentenceId))
    #print 'Longest sentence has {0} observations'.format(maxLength)
    return data_points

def print_svm_data(data, outputFile):
    f = file(outputFile,'wb')
    sentenceId = -1
    for xs, ys, sent in data:
        sentenceId += 1
        symbolId = -1
        N = int(len(xs)/len(ys))
        for k in range(len(ys)):
            symbolId+=1
            y = ys[k]
            x = xs[k*N:(k+1)*N]
            f.write('%i %i %i %i %i ' % (int(len(data)), sentenceId, int(len(ys)), symbolId, y) + ' '.join([str(x_) for x_ in x]) + '\n' )
    f.close()

def print_sentence_ids(data, outputFile):
    with file(outputFile,'wb') as outFile:
        for xs, ys, sent in data:
            outFile.write(sent+'\n')

def print_validation_codes(data,outputFile):
    f = file(outputFile,'wb')
    for xs, ys, sent in data:
        f.write(computeResponse(ys)+'\n')
    f.close()

def print_svm_hmm_data(data, outputFile):
    f = file(outputFile,'wb')
    sentenceId = -1
    for xs,ys, sent in data:
        sentenceId +=1
        N = int(len(xs)/len(ys))
        for k in range(len(ys)):
            y = ys[k]
            x = xs[k*N:(k+1)*N]
            fS=' '.join(map(':'.join,zip(map(str,range(1,N+1)),map(str,x))))
            f.write('%d qid:%d ' % (y+1,sentenceId) + fS + '\n')
    f.close()