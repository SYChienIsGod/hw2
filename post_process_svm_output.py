# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:02:34 2015

@author: Jason
"""

import paths
import numpy
import Levenshtein
import coding

validation = True # Set this to false for testing -> generates the submission output


if validation:
    pathToSVMOut = paths.pathToSVMValidationOutput
else:
    pathToSVMOut = paths.pathToSVMTestOutput

codes = list()

for line in file(pathToSVMOut,'rb'):
    ys = (int(s) for s in line.strip().split(' '))
    codes.append(coding.computeResponse(ys))
    
if validation:
    refCodes = numpy.loadtxt(paths.pathToFBANKSVMValidateCodes,dtype='str')
    for refC,recC in zip(refCodes,codes):
        print refC + ' ' + recC
    lens = numpy.asarray([len(refC) for refC in refCodes])
    editDist = numpy.asarray([Levenshtein.distance(refC,recC) for refC,recC in zip(refCodes,codes)])
    accs = (lens-editDist)/lens
    print 'Average edit distance is %f' % numpy.mean(editDist)
    print 'Average accuracy is %f' % numpy.mean(accs)
else:
    sentenceIds = numpy.loadtxt(paths.pathToFBANKSVMTestingSents,dtype='str')
    f = open(paths.pathToSubmission,'wb')
    f.write('id,phone_sequence\n')
    for sent,code in zip(sentenceIds,codes):
        f.write('%s,%s\n' % (sent,code))
    f.close()


    
    


    