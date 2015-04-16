# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:02:34 2015

@author: Jason
"""

import paths
import numpy
import Levenshtein

validation = True # Set this to false for testing -> generates the submission output
pathToSVMValidationOutput = '../svm_struct/fbank_svm_validate.out'
pathToSVMTestOutput = '../svm_struct/fbank_svm_test.out'
pathToSubmission = '../hw2_submission.csv'

if validation:
    pathToSVMOut = pathToSVMValidationOutput
else:
    pathToSVMOut = pathToSVMTestOutput

phonId = numpy.loadtxt(paths.pathToChrMap,dtype='int',usecols=(1,))
phonLetters = numpy.loadtxt(paths.pathToChrMap,dtype='str_',usecols=(2,))
phonemeId2Letter = dict(zip(phonId,phonLetters))

def computeResponse(ys):
    letters = (phonemeId2Letter[y] for y in ys)
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
    ys = (int(s) for s in line.split(' '))
    codes.append(computeResponse(ys))
    
if validation:
    refCodes = numpy.loadtxt(paths.pathToFBANKSVMValidateCodes,dtype='str')
    lens = numpy.asarray(len(refC) for refC in refCodes)
    editDist = numpy.asarray(Levenshtein.distance(refC,recC) for refC,recC in zip(refCodes,codes))
    accs = (lens-editDist)/lens
    print 'Average accuracy is %f' % numpy.mean(accs)
else:
    sentenceIds = numpy.loadtxt(paths.pathToFBANKSVMTestingSents,dtype='str')
    f = open(pathToSubmission,'wb')
    f.write('id,phone_sequence\n')
    for sent,code in zip(sentenceIds,codes):
        f.write('%s,%s\n' % (sent,code))
    f.close()


    
    


    