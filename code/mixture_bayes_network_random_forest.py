
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from programs.tree_bayesian_network_chow_liu import generate_mutual_info_matrix,generate_parents_depth_first_search,find_max_spanning_tree,get_related_vertices,generate_probability_matrix
import random



train_path = sys.argv[1]
test_path = sys.argv[2]
validation_dataset = sys.argv[3]
K = int(sys.argv[4])

train_dataset = np.loadtxt(train_path,delimiter=',',dtype='int')
test_dataset = np.loadtxt(test_path, delimiter=',',dtype='int')
validation_dataset =  np.loadtxt(validation_dataset,delimiter=',',dtype='int')
"""
train_dataset = np.loadtxt(r"D:\Bayesian\complete_datasets\nltcs.ts.data",delimiter=',',dtype='int')
test_dataset = np.loadtxt(r"D:\Bayesian\complete_datasets\nltcs.test.data", delimiter=',',dtype='int')
validation_dataset = np.loadtxt(r"D:\Bayesian\complete_datasets\nltcs.valid.data", delimiter=',',dtype='int')
"""

number_of_features = train_dataset.shape[1]
def bootstraping_generate_k_samples(K):
    data_points_in_each_bag = int(0.632*len(train_dataset))
    training_data_k_bags = {}
    for k in range(K):
        random_samples = random.sample(range(len(train_dataset)),data_points_in_each_bag)
        training_data_k_bags[k] = train_dataset[random_samples,:]
    return training_data_k_bags

def prediction(test_data,prior_matrix, parents,complete_probability_matrix, predictions,cluster_probs):
    for i in range(len(test_data)):
        prob_test_example = 0
        for j in range(number_of_features):
            t = test_data[i,j]
            k = parents[0,j]
            if k==-1:
                prob_test_example = prob_test_example + np.log2(prior_matrix[t,j])
            else:
                try:
                    joint = complete_probability_matrix[j][k][test_data[i,j]][test_data[i,k]]
                except:
                    try:
                        joint = complete_probability_matrix[k][j][test_data[i,k]][test_data[i,j]]
                    except:
                        print('Issue with count strorage or accessing countMatrix')
                prob_test_example = prob_test_example + np.log2((joint/float(prior_matrix[test_data[i,k],k])))
        predictions[i,0] = predictions[i,0]+(cluster_probs*prob_test_example)
    return predictions

def set_r_hyperparmeter(mutual_info_matrix,r):
    count,visited = 0,set()
    while(count<r):
        node1,node2 = random.randint(0,number_of_features-1), random.randint(0,number_of_features-1)
        while (((node1, node2) in visited) or ((node2, node1) in visited) or (node1 == node2)):
            node1, node2 = random.randint(0,number_of_features-1),random.randint(0,number_of_features-1)
        visited.add((node1, node2))
        mutual_info_matrix[node1][node2]=0
        count=count+1
    return mutual_info_matrix


list_of_r = [0,2,5]
execution_count=10
current_iteration_trace = 0
likelihood_trace=np.zeros((1,execution_count))
while(current_iteration_trace<execution_count):
    bootstraped_data_samples = bootstraping_generate_k_samples(K)
    predictions = np.zeros((len(test_dataset), 1))
    cluster_probs = np.ones((1, K)) / np.sum(np.ones((1, K)))
    print("RunNumber-->", current_iteration_trace + 1)
    for k in range(K):
        print( "Tree number-->", k)
        prior_matrix = generate_probability_matrix(bootstraped_data_samples[k])
        mutual_info_matrix, complete_probability_matrix = generate_mutual_info_matrix(train_dataset, prior_matrix)
        mutual_info_matrix=set_r_hyperparmeter(mutual_info_matrix,list_of_r[1])
        max_spanning_tree = find_max_spanning_tree(mutual_info_matrix)
        adjacency_list = get_related_vertices(max_spanning_tree)
        parents = generate_parents_depth_first_search(adjacency_list, random.randint(0, len(adjacency_list) - 1))
        prediction(test_dataset, prior_matrix, parents, complete_probability_matrix,predictions,cluster_probs[0,k])
    avgProb = np.mean(predictions)
    print("predictions mean=", avgProb)
    likelihood_trace[0, current_iteration_trace] = avgProb
    current_iteration_trace = current_iteration_trace + 1
    print("")
print("M=", np.mean(likelihood_trace))
print("SD=", np.std(likelihood_trace))