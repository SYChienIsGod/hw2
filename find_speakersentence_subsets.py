# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:05:06 2015

@author: jan
"""

import paths
import numpy
import networkx as nx
import cPickle

trainIds = numpy.loadtxt(paths.pathToFBANKTrain,dtype='str_',usecols=(0,))

#%%
speakerSentences = [(sentence.split('_')[0],sentence.split('_')[1]) for sentence in trainIds]

G = nx.Graph()

for speaker, sentence in speakerSentences:
    G.add_edge(speaker,sentence)
    
subsets = list(nx.connected_components(G))

#%%

speakerSets = list()

for subset in subsets:
    speakerSet = list()
    for elem in subset:
        if elem[0] in ('f','m'):
            speakerSet.append(elem)
    speakerSets.append(speakerSet)
    
#%%

sentenceGroupId = list()
sentenceGroupInst = [0] * len(speakerSets)
for speaker, sentence in speakerSentences:
    for i in range(len(speakerSets)):
        if speaker in speakerSets[i]:
            sentenceGroupId.append(i)
            sentenceGroupInst[i] += 1
            
#%% Save the information
            
f = file(paths.pathToSentenceGroupIds,'wb')
cPickle.dump(sentenceGroupId,f)
f.close()