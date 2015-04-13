# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:32:59 2015

@author: Jason
"""

import paths
import numpy
import cPickle
import csv
#%% Read Data

ids = numpy.loadtxt(paths.pathToFBANKTrain,dtype='str_',usecols=(0,))
f = file(paths.pathToSaveFBANKTrain,'rb')
fbank_feat = cPickle.load(f)
f.close()

f = file(paths.pathToSave48Labels,'rb')
fbank_labels = cPickle.load(f)
f.close()

phonemes = numpy.loadtxt(paths.pathToChrMap,dtype='str_',usecols=(0,))
phonId = numpy.loadtxt(paths.pathToChrMap,dtype='int',usecols=(1,))

#%% Extract Utterance

utt_id = 'faem0_si1392'

ph48_39 = numpy.loadtxt(paths.pathToMapPhones,dtype='str_',delimiter='\t')
ph48_39_dict = dict(ph48_39)
phi_48 = dict(zip(numpy.arange(0,48),ph48_39[:,0]))
phonemes2id = dict(zip(phonemes,phonId))

feature_vector = numpy.zeros(shape=(69*48+48*48,))

processed = False
previousNo = -1
for feature, label, frameId in zip(fbank_feat,fbank_labels, ids):
    if not frameId.startswith(utt_id):
        if processed:
            break
        continue
    processed = True
    phoneme = phi_48[label]
    phonemeNo = phonemes2id[phoneme]
    offset = phonemeNo * 69
    for index in range(69):
        feature_vector[offset+index] += feature[index]
    base = 69*48
    if previousNo >= 0:
        offset = previousNo * 48
        feature_vector[offset + base + phonemeNo] += 1
    previousNo = phonemeNo
    
with open('hw2a_prediction.csv','wb') as csvfile:
    csvw = csv.writer(csvfile,delimiter=',')
    csvw.writerow(['id','feature'])
    for row in range(feature_vector.shape[0]):
        csvw.writerow([('%s_%i') % (utt_id,row),feature_vector[row]])
