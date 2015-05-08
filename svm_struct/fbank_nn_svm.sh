#!/bin/bash -vx
./svm_empty_learn -c $1 ../../data/fbank_nn_svm_training.save fbank_nn_svm_c$1.model
./svm_empty_classify ../../data/fbank_nn_svm_validation.save fbank_nn_svm_c$1.model ../../data/fbank_nn_svm_validation$1.output
./svm_empty_classify ../../data/fbank_nn_svm_test.save fbank_nn_svm_c$1.model ../../data/fbank_nn_svm_test$1.output
