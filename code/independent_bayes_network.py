import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
train_path = sys.argv[1]
test_path = sys.argv[2]
validation_dataset = sys.argv[3]

"""
train_dataset = np.loadtxt(r"D:\Bayesian\complete_datasets\msnbc.ts.data",delimiter=',',dtype='int')
test_dataset = np.loadtxt(r"D:\Bayesian\complete_datasets\msnbc.test.data", delimiter=',',dtype='int')
validation_dataset =  np.loadtxt(r"D:\Bayesian\complete_datasets\msnbc.valid.data", delimiter=',',dtype='int')
"""

train_dataset = np.loadtxt(train_path,delimiter=',',dtype='int')
test_dataset = np.loadtxt(test_path, delimiter=',',dtype='int')
validation_dataset =  np.loadtxt(validation_dataset,delimiter=',',dtype='int')

def generate_probability_matrix(train_dataset):
    vocab_size = np.float(len(train_dataset)+2)
    probability_matrix = np.zeros((2, train_dataset.shape[1]))
    probability_of_ones = ((np.sum(train_dataset,axis=0)+1)/vocab_size)
    for i in range(len(probability_of_ones)):
        probability_matrix[1, i] = probability_of_ones[i]
        probability_matrix[0, i] = 1 - probability_of_ones[i]
    return probability_matrix

def predictions(test_dataset,matrix):
    overall_likelyhood = 0.0
    for row in range(test_dataset.shape[0]):
        probability_current = 0.0
        for col in range(test_dataset.shape[1]):
            probability_current = probability_current + np.log2(matrix[test_dataset[row, col],col])
        overall_likelyhood = overall_likelyhood + probability_current
    return overall_likelyhood,overall_likelyhood/len(test_dataset)

matrix = generate_probability_matrix(train_dataset)
likelyhood,avg_likelyhood = predictions(test_dataset,matrix)
#print(likelyhood)
print(avg_likelyhood)