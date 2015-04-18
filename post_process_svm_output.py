# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:02:34 2015

@author: Jason
"""

import paths
import numpy
import Levenshtein

validation = False # Set this to false for testing -> generates the submission output
pathToSVMValidationOutput = '../fbank_svm_validate.output'
pathToSVMTestOutput = '../fbank_svm_test.output'
pathToSubmission = '../hw2_submission.csv'

if validation:
    pathToSVMOut = pathToSVMValidationOutput
else:
    pathToSVMOut = pathToSVMTestOutput

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

codes = list()

for line in file(pathToSVMOut,'rb'):
    ys = (int(s) for s in line.strip().split(' '))
    codes.append(computeResponse(ys))
    
if validation:
    refCodes = numpy.loadtxt(paths.pathToFBANKSVMValidateCodes,dtype='str')
    for refC,recC in zip(refCodes,codes):
        print refC + ' ' + recC
    lens = numpy.asarray([len(refC) for refC in refCodes])
    editDist = numpy.asarray([Levenshtein.distance(refC,recC) for refC,recC in zip(refCodes,codes)])
    accs = (lens-editDist)/lens
    print 'Average accuracy is %f' % numpy.mean(accs)
else:
    sentenceIds = numpy.loadtxt(paths.pathToFBANKSVMTestingSents,dtype='str')
    f = open(pathToSubmission,'wb')
    f.write('id,phone_sequence\n')
    for sent,code in zip(sentenceIds,codes):
        f.write('%s,%s\n' % (sent,code))
    f.close()


    
    


    