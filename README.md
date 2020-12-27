# bayes_nets_performance_comparison


Steps to execute the programs

move into the directory /code inside the Bayesian Folder in the project
to execute independent_bayesian_network.py and tree_bayesian_network_chow_liu.py 
    use the following format python <program_name> <train_file_path> <test_file_path> <validation_file_path>
	eg: python independent_bayesian_network.py  "D:\Bayesian\complete_datasets\nltcs.ts.data" "D:\Bayesian\complete_datasets\nltcs.test.data" "D:\Bayesian\complete_datasets\nltcs.valid.data"   

to execute the other two files:
    use the following format python <program_name> <train_file_path> <test_file_path> <validation_file_path> K-value
	  eg: python mixture_bnetwork_random_forest.py "D:\Bayesian\complete_datasets\nltcs.ts.data" "D:\Bayesian\complete_datasets\nltcs.test.data" "D:\Bayesian\complete_datasets\nltcs.valid.data" 5
