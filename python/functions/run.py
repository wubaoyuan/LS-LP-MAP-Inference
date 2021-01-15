from compute_log_potential import compute_log_potential
from ADMM_algorithm import ADMM_algorithm
from read_uai import read_uai
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import argparse

  
def Run_ADMM(args):
# %% Set parameters and load data information 
    # Set parameters   
    initial_rho = args.rho_initial
    learning_fact = args.learning_fact
    rho_upper = args.rho_upper_bound
    max_iters = args.max_iter
    u_factor_solution = args.u_factor_solution
    dataset_name = args.dataset_name
    file_index = args.file_index
    
    # Run function read_uai to get data from uai file   
    # Get data of variable, factor and extra variable nodes
    [struct_node_factor_edge, file_name] = read_uai(file_index, dataset_name)
    node_structure = struct_node_factor_edge['node_structure']
    factor_structure = struct_node_factor_edge['factor_structure']
    edge_structure = struct_node_factor_edge['edge_structure']
    
    # Number of variable nodes
    numNodes = node_structure['numNodes']
    # State of each variable node
    nodeStates = node_structure['numStates']
    # Number of factor nodes
    numFactors = factor_structure['numFactors']
    # State of each factor node
    factorStates = factor_structure['numStates']
    
    
# %% Random initialization of variable nodes, factor nodes
    # Use list to store state of each variable node
    u_node_0_cell = list(np.zeros(numNodes))
    #initialize state of each node
    node_state_0 = np.zeros(numNodes)
    for n_idx in range(numNodes):
        u_node_0_cell[n_idx] = np.zeros(nodeStates[n_idx])
        # Generate a random state of each node
        node_state_0[n_idx] = random.randint(1, nodeStates[n_idx])
        # Assign the random state to node, resulting in an one-hot vector
        u_node_0_cell[n_idx][int(node_state_0[n_idx])-1] = 1
    u_node_0_cell = np.array(u_node_0_cell)
    
    # Generate initial states of each factor node
    u_factor_0_cell = list(np.zeros(numFactors))
    for f_idx in range(numFactors):
        # Get states, number of variable nodes connected to this factor node
        node_list_in_this_factor = factor_structure['nodesList'][f_idx]-1
        node_state_in_this_factor = node_state_0[node_list_in_this_factor]
        num_node_in_this_factor = node_list_in_this_factor.size
        
    # Determine the index of the factor configuration in all possible configurations
        node_numStates_in_this_factor = nodeStates[node_list_in_this_factor]
        cum = 0
        for n_idx in range(num_node_in_this_factor-1):
            # Get current states of variable nodes
            cur_state = node_state_in_this_factor[n_idx]
            cum = cum + (cur_state-1)*np.prod(node_numStates_in_this_factor[n_idx+1:])
            
        cum = cum + node_state_in_this_factor[-1]
        # Using above results, assign each factor node a initial state
        u_factor_0_cell[f_idx] = np.zeros(int(factorStates[f_idx]))
        u_factor_0_cell[f_idx][int(cum)-1] = 1
    u_factor_0_cell = np.array(u_factor_0_cell)


    # initialize a parameter dictionary   
    ADMM_params = {'opt': 1,'is_verbose': True,'stop_threshold': 1e-6, 
                     'std_threshold': 1e-6,'gamma_val': 1,'gamma_factor': 1, 
                     'max_iters': max_iters,'initial_rho': initial_rho, 
                     'learning_fact': 1 + learning_fact/100,'rho_upper': rho_upper, 
                     'history_size': 10,'u_node_0': [], 'u_factor_0': [], 
                     'rel_tol': 1e-6,'imsize': [],'save_dir': [], 'projection_lp': 2, 
                     'y_node_update_maxIter': 100, 'u_factor_solution': u_factor_solution}
    # Assign initial states of variable and factor nodes
    ADMM_params['u_node_0'] = u_node_0_cell
    ADMM_params['u_factor_0'] = u_factor_0_cell
        
    
    # Compute the initial objective value
    logPotential_0 = compute_log_potential(u_node_0_cell, u_factor_0_cell, node_structure, factor_structure)
    ADMM_params['logPotential_0'] = logPotential_0
    
# %% Execute ADMM algorithm to get results
    # Compute final result using ADMM
    ADMM_result = ADMM_algorithm(node_structure, factor_structure, edge_structure, ADMM_params)

    u_node_sol_ADMM = ADMM_result['u_node_sol_ADMM']
    v_factor_sol_ADMM = ADMM_result['v_factor_sol_ADMM']
    u_factor_sol_ADMM = ADMM_result['u_factor_sol_ADMM']
    obj_list_ADMM = ADMM_result['obj_list_ADMM']
    constraint_violation_lp_ADMM = ADMM_result['constraint_violation_lp_ADMM']
    constraint_violation_consis_ADMM = ADMM_result['constraint_violation_consis_ADMM']
    time_elapsed_ADMM = ADMM_result['time_elapsed_ADMM']
    
    # Compute objective value based on final results
    logPotential_ADMM = compute_log_potential(u_node_sol_ADMM, u_factor_sol_ADMM, node_structure, factor_structure)
    
    # Plot the figure of how objective value changes during iteration
    plt.figure()
    plt.plot(np.arange(len(obj_list_ADMM))+1,obj_list_ADMM)
    plt.xlabel('iterations')
    plt.ylabel('objective value')
    plt.show()
    
    # construct a dictionary to store results
    result_struct_ADMM = {'logPotCurve': obj_list_ADMM,
                          'logPotFinal': logPotential_ADMM,
                          'lpbox_violation': constraint_violation_lp_ADMM,
                          'local_consistency_violation': constraint_violation_consis_ADMM,
                          'nodeLabel': u_node_sol_ADMM,
                          'factorLabel': u_factor_sol_ADMM,
                          'time_elapsed': time_elapsed_ADMM,
                          'ADMM_params': ADMM_params
                          }
    
# %% Save the result dictionary in a npy file
    file_name = file_name[:-4]
    os.chdir('../../python')
    # Create a file named Results if not exist
    if not os.path.exists('results'):
        os.makedirs('results')
    os.chdir('results')
    # Create a file name specific dataset_name if not exist
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)    
    # Save the result in the npy file
    np.save(dataset_name + '\\result_' + file_name + '.npy', result_struct_ADMM)
    return 0
    
# %% Main function
def main():
    parser = argparse.ArgumentParser()
    # rho_initial available choices: [5e-2, 1e-1, 1e0, 5e0, 1e1, 1e2, 1e3, 1e4]
    parser.add_argument('--rho_initial', type = float, default = 5e-2)
    # learning_fact available choices: [1.01, 1.03, 1.05, 1.1, 1.2]
    parser.add_argument('--learning_fact', type = float, default = 1.01)
    # rho_upper_bound available choices: [1e6,1e8]
    parser.add_argument('--rho_upper_bound', type = int, default = 1e6)
    # max_iter available choices: [500, 1000]
    parser.add_argument('--max_iter', type = int, default = 500)
    # u_factor_solution available choices: ['linear-proximal', 'exact-qpc']
    parser.add_argument('--u_factor_solution', type = str, default = 'linear-proximal')
    # dataset available choices: ['Grids', 'inpainting4', 'inpainting8', 'scene', 'Segmentation']
    parser.add_argument('--dataset_name', type = str, default = 'Grids')
    # file_index available choices are decided by specific dataset
    parser.add_argument('--file_index', type = int, default = 0)
    args = parser.parse_args()

    Run_ADMM(args)    
    
    return 0

    
# %% Execute the main function  
if __name__=='__main__':
    main()
    
    