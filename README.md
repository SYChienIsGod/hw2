# Homework 2 Part A (finished) 
Use hw2a_single_feature_vector_generation.py

# Homework 2 Part B

## Building
Replace the three files "Makefile", "svm_struct_api.c" and "svm_struct_api_types.h" with the ones from the svm_struct folder in the repository. Invoke make to build the software.

### Prerequisites

The python library python-Levenshtein is needed to compute the edit distance, available on PyPi (i.e. pip).

## Running
Invoke "svm_feature_vectors.py" to generate feature vectors for training, validation and testing set along with a list of sentence ids for validation and testing and the true code for the validation set (will be needed later on).

Start learning (takes a long time...)
`svm_empty_learn.exe -c 1 pathToTrainingData pathToModel`

Test (goes quickly)
`svm_empty_classify.exe pathToTestData pathToModel pathToOutput`

Use post_process_svm_output.py to either compute the average edit distance or write a submission (can be modified via a variable)

## ToDo

- Check the Viterbi Decoder for Bugs


