# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:47:12 2015

@author: jan
"""

import numpy
import paths

phonemes = numpy.loadtxt(paths.pathToChrMap,dtype='str_',usecols=(0,))
phonId = numpy.loadtxt(paths.pathToChrMap,dtype='int',usecols=(1,))
letters = numpy.loadtxt(paths.pathToChrMap,dtype='str_',usecols=(2,))


ph48_39 = numpy.loadtxt(paths.pathToMapPhones,dtype='str_',delimiter='\t')
ph48_39_dict = dict(ph48_39)
ph39_48 = dict(zip(ph48_39[:,1],ph48_39[:,0]))
phi_48 = dict(zip(numpy.arange(0,48),ph48_39[:,0]))
ph39_i = dict(zip(set(ph48_39[:,1]),numpy.arange(0,39)))
phi_39 = dict(zip(numpy.arange(0,39),set(ph48_39[:,1])))
phonemes2Letters = dict(zip(phonemes,letters))

i39_L39 = dict()
for k,v in phi_39.items():
    i39_L39[k] = phonemes2Letters[v]
    
i48_i39 = dict()
for k,v in phi_48.items():
    i48_i39[k] = ph39_i[ph48_39_dict[v]]

def computeResponse(ys):
    letters = (i39_L39[y] for y in ys)
    letters_out = list()
    lastLetter = ''
    for l in letters:
        if l == lastLetter:
            continue
        letters_out.append(l)
        lastLetter = l
    code = ''.join(letters_out).replace('K',' ').strip().replace(' ','K')
    return code