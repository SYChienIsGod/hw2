# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:02:34 2015

@author: Jason
"""

import paths
import numpy
import Levenshtein
import coding

validation = False # Set this to false for testing -> generates the submission output


svmOutput = '../data/fbank_nn_svm_validation1000.output'
validationCodes = paths.pathToFBANKSVMValidateCodes
svmTestOutput = 'fbank_nn_svm_test1000.output' #This has been changed to work with make run

if validation:
    pathToSVMOut = svmOutput #paths.pathToSVMValidationOutput
else:
    pathToSVMOut = svmTestOutput

codes = list()


def correctY(ys):
    for i in range(1,len(ys)-1):
        if ys[i-1] == ys[i+1] and ys[i] != ys[i-1]:
            ys[i] = ys[i-1]
    for i in range(1,len(ys)-1):
        if ys[i-1] != ys[i] and ys[i+1] != ys[i]:
            ys[i] = ys[i-1]
    return ys
ysList=list()
for line in file(pathToSVMOut,'rb'):
    ys = correctY([int(s) for s in line.strip().split(' ')])
    ysList.append(correctY(ys))
    codes.append(coding.computeResponse(ys))
    
if validation:
    refCodes = numpy.loadtxt(validationCodes,dtype='str')
    for refC,recC in zip(refCodes,codes):
        print refC + ' ' + recC
    lens = numpy.asarray([len(refC) for refC in refCodes])
    editDist = numpy.asarray([Levenshtein.distance(refC,recC) for refC,recC in zip(refCodes,codes)])
    accs = (lens-editDist)/lens
    print 'Average edit distance is %f' % numpy.mean(editDist)
    print 'Average accuracy is %f' % numpy.mean(accs)
else:
    # paths.pathToFBANKSVMTestingSents
    sentenceIds = numpy.loadtxt('fbank_svm_testing.sents',dtype='str') #This has been changed to work with make run
    # paths.pathToSubmission
    f = open('output.kaggle','wb') 
    f.write('id,phone_sequence\n')
    for sent,code in zip(sentenceIds,codes):
        f.write('%s,%s\n' % (sent,code))
    f.close()

#%%
#phonemeStreakLengths = dict([(i, list()) for i in range(39)])
#
#for ys in ysList:
#    currPhon = ys[0]
#    currLen = 1
#    for i in range(1,len(ys)):
#        if ys[i] != currPhon:
#            phonemeStreakLengths[currPhon].append(currLen)
#            currLen = 1
#            currPhon = ys[i]
#        else:
#            currLen+=1
#    phonemeStreakLengths[currPhon].append(currLen)
#    
#%%
    
#for k,v in phonemeStreakLengths.iteritems():
#    arr = numpy.array(v)
#    print 'Phoneme {}: [{},{}] ~={}'.format(k,numpy.min(arr),numpy.max(arr),numpy.mean(arr))
    
#%%

#tM = numpy.zeros(shape=(39,39))
#for ys in ysList:
#    lastPhon = ys[0]
#    for i in range(1,len(ys)):
#        tM[ys[i-1],ys[i]]+=1
#for i in range(39):
#    string = ''
#    for j in range(39):
#        string += '{0:.3f} '.format(tM[i,j]/numpy.sum(tM[i,:])*100)
#    print string