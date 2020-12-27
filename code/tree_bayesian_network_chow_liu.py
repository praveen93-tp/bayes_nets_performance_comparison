import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scipy.sparse.csgraph import minimum_spanning_tree
import random

train_path = sys.argv[1]
test_path = sys.argv[2]
validation_dataset = sys.argv[3]

train_dataset = np.loadtxt(train_path,delimiter=',',dtype='int')
test_dataset = np.loadtxt(test_path, delimiter=',',dtype='int')
validation_dataset =  np.loadtxt(validation_dataset,delimiter=',',dtype='int')

"""
train_dataset = np.loadtxt(r"D:\Bayesian\complete_datasets\kdd.ts.data",delimiter=',',dtype='int')
test_dataset = np.loadtxt(r"D:\Bayesian\complete_datasets\kdd.test.data", delimiter=',',dtype='int')
validation_dataset =  np.loadtxt(r"D:\Bayesian\complete_datasets\kdd.valid.data", delimiter=',',dtype='int')
"""
def generate_probability_matrix(train_dataset):
    vocab_size = np.float(len(train_dataset)+2)
    probability_matrix = np.zeros((2,train_dataset.shape[1]))
    probability_of_ones = ((np.sum(train_dataset,axis=0)+1)/vocab_size)
    for i in range(len(probability_of_ones)):
        probability_matrix[1, i] = probability_of_ones[i]
        probability_matrix[0, i] = 1 - probability_of_ones[i]
    return probability_matrix

def calcualte_pairwise_probability_matrix(train_dataset,column1,column2):
     pairwise_matrix = np.zeros((2,2))
     vocab_size = np.float(len(train_dataset)+4)
     for row in range(len(train_dataset)):
       pairwise_matrix[train_dataset[row,column1],train_dataset[row,column2]] = pairwise_matrix[train_dataset[row,column1],train_dataset[row,column2]]+1
     pairwise_matrix = ((pairwise_matrix+1)/(vocab_size))
     return pairwise_matrix

def generate_mutual_info_matrix(train_dataset,probability_matrix):
  mutual_info_matrix = np.zeros((train_dataset.shape[1],train_dataset.shape[1]))
  complete_probability_matrix = {}
  for i in range(train_dataset.shape[1]):
      complete_probability_matrix[i] = {}
      for j in range(i+1,train_dataset.shape[1]):
          pairwise_matrix = calcualte_pairwise_probability_matrix(train_dataset,i,j)
          mutual_info_matrix[i][j] = pairwise_matrix[0][0]*np.log2(pairwise_matrix[0][0]/(probability_matrix[0,i]*probability_matrix[0,j]))+\
                                     pairwise_matrix[0][1]*np.log2(pairwise_matrix[0][1]/(probability_matrix[0,i]*probability_matrix[1,j]))+\
                                     pairwise_matrix[1][0]*np.log2(pairwise_matrix[1][0]/(probability_matrix[1,i]*probability_matrix[0,j]))+\
                                     pairwise_matrix[1][1]*np.log2(pairwise_matrix[1][1]/(probability_matrix[1,i]*probability_matrix[1,j]))
          complete_probability_matrix[i][j] = [[pairwise_matrix[0][0],pairwise_matrix[0][1]],[pairwise_matrix[1][0],pairwise_matrix[1][1]]]
  return mutual_info_matrix,complete_probability_matrix

def find_max_spanning_tree(mutual_info_matrix):
    max_tree = minimum_spanning_tree((mutual_info_matrix*(-1)))
    max_tree = max_tree.toarray()
    return max_tree

def get_related_vertices(max_spanning_tree):
    related_vertices_set = {}
    for i in range(len(max_spanning_tree)):
        related_vertices_set[i] = set()
    for i in range(len(max_spanning_tree)):
        current_vertices = max_spanning_tree[i].nonzero()[0]
        if(len(current_vertices)>0):
            for j in range(len(current_vertices)):
                related_vertices_set[i].add(current_vertices[j])
                related_vertices_set[current_vertices[j]].add(i)
    return related_vertices_set

def generate_parents_depth_first_search(adjacancy_list,start_vertex):
    path = np.zeros((1,len(adjacancy_list)), dtype=int)
    visited, stack, parent = set(), [start_vertex], []
    while stack:
        current_vertex = stack.pop()
        if current_vertex not in visited:
            visited.add(current_vertex)
            if not parent:
                path[0, current_vertex] = -1
            else:
                parent_check = parent.pop()
                path[0, current_vertex] = parent_check
                if ((adjacancy_list[parent_check].issubset(visited)) == False):
                    parent.append(parent_check)
            if ((adjacancy_list[current_vertex].issubset(visited)) == False):
                parent.append(current_vertex)
            stack.extend(adjacancy_list[current_vertex] - visited)
    return path

def prediction(test_dataset,prior_matrix,parent,complete_probability_matrix):
    avg_probs = 0.0
    for i in range(test_dataset.shape[0]):
        prob_current_test_ex = 0
        for j in range(test_dataset.shape[1]):
            current_parent = parent[0,j]
            if(current_parent== -1):
                prob_current_test_ex = prob_current_test_ex + np.log2(prior_matrix[test_dataset[i,j],j])
            else:
                try:
                 joint_prob = complete_probability_matrix[j][current_parent][test_dataset[i,j]][test_dataset[i,current_parent]]
                except:
                    try:
                        joint_prob = complete_probability_matrix[current_parent][j][test_dataset[i,current_parent]][test_dataset[i,j]]
                    except:
                        sys.exit(0)
                prob_current_test_ex = prob_current_test_ex + np.log2(joint_prob/prior_matrix[test_dataset[i,current_parent],parent[0,j]])
        avg_probs = avg_probs + prob_current_test_ex
    return avg_probs/len(test_dataset)

def main():
    prior_matrix = generate_probability_matrix(train_dataset)
    mutual_info_matrix,complete_probability_matrix = generate_mutual_info_matrix(train_dataset,prior_matrix)
    max_spanning_tree = find_max_spanning_tree(mutual_info_matrix)
    adjacency_list = get_related_vertices(max_spanning_tree)
    parents = generate_parents_depth_first_search(adjacency_list, random.randint(0,len(adjacency_list)-1))
    avg_pred = prediction(test_dataset,prior_matrix,parents,complete_probability_matrix)
    print(avg_pred)

if __name__=="__main__":
    main()