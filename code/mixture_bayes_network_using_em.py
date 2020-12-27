import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from programs.tree_bayesian_network_chow_liu import generate_parents_depth_first_search,find_max_spanning_tree,get_related_vertices
import random


def generate_random_graph(number_of_nodes):
    adj_matrix = np.zeros((number_of_nodes,number_of_nodes))
    choosen_vertex,visited,number_of_edges = set(),set(),0
    while((number_of_edges<number_of_nodes-1) and (visited!=set(range(number_of_nodes)))):
        node1,node2 = random.randint(0,number_of_nodes-1),random.randint(0,number_of_nodes-1)
        while(((node1, node2) in choosen_vertex)or((node2,node1)in choosen_vertex) or (node1==node2)):
            node1, node2 = random.randint(0, number_of_nodes - 1), random.randint(0, number_of_nodes - 1)
        choosen_vertex.add((node1,node2))
        adj_matrix[node1][node2] = 1
        number_of_edges=number_of_edges+1
        visited.add(node1)
        visited.add(node2)
    return adj_matrix

def k_graph_initialization(K):
    K_parents = {}
    for i in range(K):
        graph = generate_random_graph(number_of_features)
        maxTree = find_max_spanning_tree(graph)
        adj_list = get_related_vertices(maxTree)
        parents = generate_parents_depth_first_search(adj_list, random.randint(0, len(adj_list) - 1))
        K_parents[i] = parents
    return K_parents

def k_joint_prob_initialization(K,K_parents):
    joint_prob_init = {}
    for i in range(K):
        joint_prob_init[i] = {}
        for j in range(number_of_features):
            joint_prob_init[i][j] = {}
            current_parent = K_parents[i][0,j]
            if current_parent==-1:
                joint_prob_init[i][j][current_parent] = (random.random())
            else:
                joint_prob_init[i][j][current_parent] = (random.random(),random.random())
    return joint_prob_init

def calculate_mutual_information(prob_table_computed_data,clusterNumber):
    joint_prob = {}
    prior_prob = {}
    mutual_info_matrix = np.zeros((number_of_features,number_of_features))
    vocab = round(4)
    for i in range(len(train_dataset)):
        vocab += prob_table_computed_data[i,clusterNumber]
    for j in range(number_of_features):
        count_ones=2
        for i in range(len(train_dataset)):
            if train_dataset[i,j]==1:
                count_ones+=prob_table_computed_data[i,clusterNumber]
        prior_prob[j]=count_ones/vocab
    for j in range(number_of_features):
        joint_prob[j]={}
        for k in range(j+1,number_of_features):
            count_temp = np.ones((2,2))
            for i in range(len(train_dataset)):
                count_temp[train_dataset[i,j],train_dataset[i,k]] += prob_table_computed_data[i,clusterNumber]
            count_temp = count_temp/vocab
            mutual_info_matrix[j][k] = count_temp[0,0]*np.log2((count_temp[0,0]/((1-prior_prob[j])*(1-prior_prob[k]))))+\
                                       count_temp[0,1]*np.log2(count_temp[0,1]/((1-prior_prob[j])*(prior_prob[k])))+\
                                       count_temp[1,0]*np.log2((count_temp[1,0]/((prior_prob[j])*(1-prior_prob[k]))))+\
                                       count_temp[1,1]*np.log2(count_temp[1,1]/((prior_prob[j])*(prior_prob[k])))
            joint_prob[j][k]=[[count_temp[0,0],count_temp[0,1]],[count_temp[1,0],count_temp[1,1]]]
    return mutual_info_matrix, joint_prob, prior_prob

def findProb(joint_prob,j,j_val,k,k_val):
    prob = 0
    try:
        prob = joint_prob[j][k][j_val][k_val]
    except:
        try:
            prob = joint_prob[k][j][k_val][j_val]
        except:
            print('issue accessing joint probability')
    return prob

def update_joint_probabilities(joint_prob_updated,k_parents_updated,k, joint_prob, prior_prob):
    joint_prob_updated[k]={}
    for j in range(number_of_features):
        joint_prob_updated[k][j]={}
        parent = k_parents_updated[k][0,j]
        if parent==-1:
            joint_prob_updated[k][j][parent]=prior_prob[j]
        else:
            a= findProb(joint_prob,j,1,parent,1)/prior_prob[parent]
            b = findProb(joint_prob,j,1,parent,0)/(1-prior_prob[parent])
            joint_prob_updated[k][j][parent]=(a,b)
    return joint_prob_updated

def check_possible_combinations_train(prob_table_training_data,joint_prob_init,i,j,k,current_parent):
    if (train_dataset[i,j]==1 and train_dataset[i,current_parent]==1):
        prob_table_training_data[i, k]=prob_table_training_data[i,k]*(joint_prob_init[k][j][current_parent][0])
    elif (train_dataset[i,j]==0 and train_dataset[i,current_parent]==1):
        prob_table_training_data[i, k]=prob_table_training_data[i,k]*(1-joint_prob_init[k][j][current_parent][0])
    elif (train_dataset[i,j]==1 and train_dataset[i,current_parent]==0):
        prob_table_training_data[i,k]=prob_table_training_data[i,k]*(joint_prob_init[k][j][current_parent][1])
    else:
        prob_table_training_data[i,k]=prob_table_training_data[i,k]*(1-joint_prob_init[k][j][current_parent][1])
    return prob_table_training_data

def generate_data_from_param(K,joint_prob_init,cluster_prob_init,k_parents):
    prob_table_training_data = np.ones((len(train_dataset),K))
    for i in range(len(train_dataset)):
        for k in range(K):
            prob_table_training_data[i,k]=cluster_prob_init[0,k]
            for j in range(number_of_features):
                parent = k_parents[k][0,j]
                if parent == -1:
                    if train_dataset[i,j]==1:
                        prob_table_training_data[i,k]=prob_table_training_data[i,k]*(joint_prob_init[k][j][parent])
                    else:
                        prob_table_training_data[i,k]=prob_table_training_data[i,k]*(1-joint_prob_init[k][j][parent])
                else:
                    prob_table_training_data = check_possible_combinations_train(prob_table_training_data,joint_prob_init,i,j,k,parent)
        prob_table_training_data = prob_table_training_data / prob_table_training_data.sum(axis=1, keepdims=True)
    return prob_table_training_data

def generate_parm_from_data(prob_table_computed_data,K):
    cluster_prob_updated = np.zeros((1,K))
    for k in range(K):
        cluster_prob_updated[0,k]=0
        for i in range(len(train_dataset)):
            cluster_prob_updated[0,k]+=prob_table_computed_data[i,k]
        cluster_prob_updated[0,k]= cluster_prob_updated[0,k]/float(len(train_dataset))
    joint_prob_updated= {}
    k_parents_updated={}
    for k in range(K):
         mutual_info_matrix, joint_prob, prior_prob = calculate_mutual_information(prob_table_computed_data,k)
         max_spanning_tree= find_max_spanning_tree(mutual_info_matrix)
         adj_list = get_related_vertices(max_spanning_tree)
         parents = generate_parents_depth_first_search(adj_list,random.randint(0,len(adj_list)-1))
         k_parents_updated[k]=parents
         joint_prob_updated = update_joint_probabilities(joint_prob_updated,k_parents_updated,k, joint_prob, prior_prob)
    return cluster_prob_updated, k_parents_updated, joint_prob_updated


def check_possible_combinations_test(i,j,k,parent,joint_prob_updated,prob_test_curr_k):
    if (test_dataset[i, j] == 1 and test_dataset[i, parent] == 1):
        prob_test_curr_k = prob_test_curr_k * (joint_prob_updated[k][j][parent][0])
    elif (test_dataset[i, j] == 0 and test_dataset[i, parent] == 1):
        prob_test_curr_k = prob_test_curr_k * (1 - joint_prob_updated[k][j][parent][0])
    elif (test_dataset[i, j] == 1 and test_dataset[i, parent] == 0):
        prob_test_curr_k = prob_test_curr_k * (joint_prob_updated[k][j][parent][1])
    else:
        prob_test_curr_k = prob_test_curr_k* (1-joint_prob_updated[k][j][parent][1])
    return prob_test_curr_k

def prediction(test_dataset,cluster_prob_updated, k_parents_updated, joint_prob_updated):
    avg_prob = 0
    for i in range(len(test_dataset)):
        prob_current_test_ex = 0
        for k in range(K):
            prob_test_curr_k = 1
            for j in range(number_of_features):
                parent = k_parents_updated[k][0,j]
                if parent == -1:
                    if test_dataset[i,j]==1:
                        prob_test_curr_k=prob_test_curr_k*(joint_prob_updated[k][j][parent])
                    else:
                        prob_test_curr_k=prob_test_curr_k*(1-joint_prob_updated[k][j][parent])
                else:
                    prob_test_curr_k = check_possible_combinations_test(i,j,k,parent,joint_prob_updated,prob_test_curr_k)
            prob_current_test_ex += cluster_prob_updated[0,k]*prob_test_curr_k
        avg_prob+= np.log2(prob_current_test_ex)
    avg_prob= avg_prob/float(len(test_dataset))
    print("Log Likelihood=",avg_prob)
    print("")
    return avg_prob

"""
train_dataset = np.loadtxt(r"D:\Bayesian\complete_datasets\nltcs.ts.data",delimiter=',',dtype='int')
test_dataset = np.loadtxt(r"D:\Bayesian\complete_datasets\nltcs.test.data", delimiter=',',dtype='int')
validation_dataset = np.loadtxt(r"D:\Bayesian\complete_datasets\nltcs.valid.data", delimiter=',',dtype='int')
K=5
"""

train_path = sys.argv[1]
test_path = sys.argv[2]
validation_dataset = sys.argv[3]
K = int(sys.argv[4])

train_dataset = np.loadtxt(train_path,delimiter=',',dtype='int')
test_dataset = np.loadtxt(test_path, delimiter=',',dtype='int')
validation_dataset =  np.loadtxt(validation_dataset,delimiter=',',dtype='int')


number_of_features = train_dataset.shape[1]

maximum_iterations=100
maximum_runs = 10
current_run_trace = 0
Likelihood=np.zeros((1,maximum_runs))
while(current_run_trace<maximum_runs):
    k_parents = k_graph_initialization(K)
    cluster_prob_initial = np.ones((1, K)) / np.sum(np.ones((1, K)))
    joint_prob = k_joint_prob_initialization(K, k_parents)
    current_iteration,flag=1,1
    print("Run Number-->", current_run_trace + 1)
    while(current_iteration<=maximum_iterations and flag==1):
        print("IterationNumber-->",current_iteration)
        flag=0
        prob_table_training_data = generate_data_from_param(K,joint_prob,cluster_prob_initial,k_parents)
        cluster_prob_updated, k_parents, joint_prob = generate_parm_from_data(prob_table_training_data,K)
        if (current_iteration!=0):
            for k in range(0,K):
                if(abs(cluster_prob_updated[0,k]-cluster_prob_initial[0,k])>0.001):
                    flag = 1
            cluster_prob_initial = cluster_prob_updated
        else:
            flag = 1
            cluster_prob_initial = cluster_prob_updated
        current_iteration+=1
    print("Converged at IterationNumber=",current_iteration-1)
    Likelihood[0,current_run_trace]= prediction(validation_dataset,cluster_prob_updated, k_parents, joint_prob)
    current_run_trace= current_run_trace+1
print("")
print("Mean=",np.mean(Likelihood))
print("Standard Dev=",np.std(Likelihood))