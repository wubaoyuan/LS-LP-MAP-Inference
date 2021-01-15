import numpy as np
from compute_std_obj import compute_std_obj
import time
from project_shifted_Lp_ball import project_shifted_Lp_ball
from compute_log_potential import compute_log_potential
from project_simplex import project_simplex
from cvxopt import solvers, matrix
import cvxopt

def ADMM_algorithm(node_structure, factor_structure, edge_structure, all_params):

# %% Get parameters and state of nodes
    #------------------------ parameters ----------------------
    stop_threshold = all_params['stop_threshold']
    std_threshold = all_params['std_threshold']
    rho_upper = all_params['rho_upper']
    max_iters = all_params['max_iters']
    initial_rho = all_params['initial_rho']
    gamma_val = all_params['gamma_val']
    learning_fact = all_params['learning_fact']
    history_size = all_params['history_size']
    
    #---------------------- node parameters ----------------------
    numNodes = node_structure['numNodes']
    nodeStates = node_structure['numStates']
    numFactors = factor_structure['numFactors']
    factorStates = factor_structure['numStates']
    
    numEdges = edge_structure['numEdges']
    factorIndexAll = edge_structure['factorNodePair_List'][0]
    nodeIndexAll = edge_structure['factorNodePair_List'][1]
    
    thetaNodeCell = node_structure['nodeScore']
    thetaFactorCell = factor_structure['factorScore']

# %% Get information of initialization based on dictionary 'all_params'    
    # initialize information of variable, factor and extra variable nodes
    u_node_sol = all_params['u_node_0']
    u_factor_sol = all_params['u_factor_0'] 
    v_factor_sol = np.array([0 for i in range(u_factor_sol.size)]).reshape(u_factor_sol.shape)

    # initialize lambda of factor nodes based on factor structure in a list
    lambda_factor = [0 for i in range(numFactors)]
    for i in range(numFactors):
        lambda_factor[i] = np.zeros(int(factorStates[i]))
    lambda_factor = np.array(lambda_factor)
    # initialize lambda of edges in a list
    lambda_edge = [0 for i in range(numEdges)]
    # Get node states connected to each edge
    nodeState_of_each_edge = nodeStates[nodeIndexAll-1] 

    for e_idx in range(numEdges):
        lambda_edge[e_idx] = np.zeros(nodeState_of_each_edge[e_idx])
        
    # violation of equivalence constrains
    constraint_violation_lp = []
    # violation of local consistency constraint
    constraint_violation_local_consistency = []
    
    # initial value of rho_i and rho_alpha
    rho_factor = 3*initial_rho
    rho_edge = initial_rho
    
    # compute standard deviation of objective value if length of obj_list >= history size
    obj_list = []
    std_obj = compute_std_obj(obj_list, history_size)
    
    # Get index of each node where state equals 1
    prev_idx_vec = np.where(u_node_sol == 1)[1]
    succ_change = 1
    best_bin_obj = 0
    
    # ADMM algorithm
    time_elapsed = 0
    constraint_stopping = True
    iter = 1
    
# %% ADMM algorithm
    while constraint_stopping & (std_obj >= std_threshold) & (iter <= max_iters):
        
        print('-------------- start of [iter %d] ---------------\n'%(iter))
        
        
# %% Part 1: update extra variable nodes
        print(' + update v_factor ... ')
        # record time
        time_start = time.time()
        
        v_factor_sol_transform = u_factor_sol + lambda_factor/rho_factor
        v_factor_sol_vec = [v_factor_sol_transform[i][j] for i in range(v_factor_sol_transform.shape[0]) 
                                for j in range(v_factor_sol_transform[i].size)]
        v_factor_sol_vec = np.array(v_factor_sol_vec)
        # Make a projection onto sphere to get the updated results
        v_factor_sol_vec_proj = project_shifted_Lp_ball(v_factor_sol_vec, 
                    0.5*np.ones(v_factor_sol_vec.size), all_params['projection_lp'])
        
        # Assign the updated projection value to each extra variable node 
        cum = 0
        for f_idx in range(numFactors):
            num_state_this_factor = factorStates[f_idx]
            cum_new = cum + num_state_this_factor
            # return num_state_this_factor
            v_factor_sol[f_idx] = v_factor_sol_vec_proj[cum:cum_new]
            
        time_end = time.time()
        t1 = time_end - time_start
        # Show the execution time
        print(' finished in [%3.3f] seconds \n'%(t1))
        time_elapsed =  time_elapsed + t1
    
# %% Part 2:  update factor nodes
        print(' + update u_factor ... ')
        time_start = time.time()
        # This is a convex problem, so we use active-set algorithm in quadratic programming solver
        for f_idx in range(numFactors):
            edgeIndexList = np.nonzero(factorIndexAll == (f_idx+1))[0]
            nodeIndexList = nodeIndexAll[edgeIndexList]

            # Find initial A matrix and b vector
            A_matrix = rho_factor * np.eye(factorStates[f_idx])
            b_vec = thetaFactorCell[f_idx].T + rho_factor * v_factor_sol[f_idx] - lambda_factor[f_idx]

            # update A matrix and b vector in each edge
            for e_idx in range(len(edgeIndexList)):
                edge = edgeIndexList[e_idx]
                node = nodeIndexList[e_idx]
                M_matrix = edge_structure['M'][edge]
                A_matrix = A_matrix + rho_edge * M_matrix.T * M_matrix 
                b_vec = b_vec + M_matrix.T * (rho_edge * u_node_sol[node-1] + lambda_edge[edge])
            
            # Solve the constrained QP with QPC
            if all_params['u_factor_solution'] == 'exact-qpc':
                fac_state = len(b_vec)
                Aeq = matrix(np.ones(fac_state)).T
                beq = matrix(np.ones(1))
                lb = np.zeros(fac_state)
                ub = np.ones(fac_state)
                bound_vec = matrix(np.concatenate((ub, lb)))
                bound_matrix = matrix(np.concatenate((np.eye(fac_state), -np.eye(fac_state)), 0))
                b_vec = matrix(b_vec)
                cvxopt.solvers.options['show_progress'] = False
                sol = solvers.qp(matrix(A_matrix), -matrix(b_vec), 
                                                 bound_matrix, bound_vec, Aeq, beq) 
                u_factor_sol[f_idx] = np.array(sol['x']).reshape(fac_state,)
                
            # Linear-proximal approximation
            elif all_params['u_factor_solution'] == 'linear-proximal':
                eta = 1.01 * (np.linalg.norm(A_matrix))**2
                temp_vec = u_factor_sol[f_idx] + (b_vec - np.dot(A_matrix, u_factor_sol[f_idx])) / eta
                u_factor_sol[f_idx] = project_simplex(np.array(temp_vec)[0])
        
        time_end = time.time()
        t2 = time_end - time_start
        print(' finished in [%3.3f] seconds \n'%(t2))
        time_elapsed =  time_elapsed + t2
                
        
# %% Part 3: update variable nodes
        print(' + update u_node ... ')
        time_start = time.time()
        
        u_node_sol = [0 for i in range(numNodes)]
        for node in range(numNodes):
            # nodeScore in node_structure
            numerator = thetaNodeCell[node]
            # Find index of connected edges
            edgeIndexList = np.nonzero(nodeIndexAll == (node+1))[0]
            # Find number of connected edges
            numEdgesConnected = edgeIndexList.size
            # Find index of connected factor nodes
            factorIndexList = node_structure['factorNeighborCell'][node]
            
            # update numerator based on edges, factor nodes and M related to this variable node
            for f_idx in range(factorIndexList.size):
                edge = edgeIndexList[f_idx]
                fac = factorIndexList[f_idx]
                M_matrix = edge_structure['M'][edge]
                numerator = numerator + (rho_edge * M_matrix * u_factor_sol[fac-1] - lambda_edge[edge])
                
            # Update denominator
            denominator = numEdgesConnected * rho_edge
            # Update miu of this variable node
            # Consider the case when connected edge is none
            if denominator == 0:
                u_node_i = numerator / sum(numerator)
            else:
                u_node_i = numerator / denominator           
            u_node_sol[node] = u_node_i
        u_node_sol = np.array(u_node_sol)
        
        # Output running time of updating variable nodes
        time_end = time.time()
        t3 = time_end - time_start
        print(' finished in [%3.3f] seconds \n'%(t3))
        time_elapsed = time_elapsed + t3
        
# %% Part 4: update dual variables
        
        print('+ update dual variables ... ')
        time_start = time.time()
        
        # Update lambda_i
        if iter < all_params['y_node_update_maxIter']:
            lambda_factor = lambda_factor + gamma_val * rho_factor * (u_factor_sol - v_factor_sol)
           
        # Update lambda_i_alpha
        for e_idx in range(numEdges):
            node = nodeIndexAll[e_idx]
            fac = factorIndexAll[e_idx]
            M_matrix = edge_structure['M'][e_idx]
            lambda_edge[e_idx] = lambda_edge[e_idx] + gamma_val * rho_edge * (u_node_sol[node-1] - M_matrix * u_factor_sol[fac-1])
                            
        time_end = time.time()
        t4 = time_end - time_start
        print(' finished in [%3.3f] seconds \n'%(t4))
        time_elapsed = time_elapsed + t4
        
# %% Evaluate constraint violation and objective values
        # lp-box consistency, compute violation of equivalence constraint
        temp_violation_lp = np.sqrt(np.sum(np.linalg.norm((u_factor_sol - v_factor_sol)**2)))
        # Judge whether the algorithm should stop
        constraint_stopping = temp_violation_lp >= stop_threshold
        # Store all violations of equivalence constraint in each iteration
        constraint_violation_lp = [constraint_violation_lp, temp_violation_lp]
        
        # Local consistency, compute violation of local consistency constraint
        constraint_violation_local_consistency_vec = [0 for i in range(numEdges)]
        for e_idx in range(numEdges):
            fac = factorIndexAll[e_idx]
            node = nodeIndexAll[e_idx]
            M_matrix = edge_structure['M'][e_idx]
            constraint_violation_local_consistency_vec[e_idx] = (np.linalg.norm(u_node_sol[node-1] - M_matrix * u_factor_sol[fac-1]))**2
        
        # Store all violations of local consistency in each iteration
        constraint_violation_local_consistency = [constraint_violation_local_consistency, np.sqrt(np.sum(constraint_violation_local_consistency_vec))]
        obj_list.append(compute_log_potential(u_node_sol, u_factor_sol, node_structure, factor_structure)) 
        std_obj = compute_std_obj(obj_list, history_size)
        
# %% Find and show overall results
        cur_idx_vec = np.array([np.nonzero(u_node_sol[i] == np.max(u_node_sol[i]))[0][0] for i in range(len(u_node_sol))])
        if iter > 10:
            succ_change = sum(cur_idx_vec != prev_idx_vec)/len(cur_idx_vec)
            
        prev_idx_vec = cur_idx_vec
        cur_obj = obj_list[-1]
        cur_bin_obj = 0
        
        if best_bin_obj >= cur_bin_obj:
            best_bin_obj = cur_bin_obj
            
        print('------ end of [iter %d]: std = [%3.2f]; obj = [%5.2f]; , rho = [%3.3f], time = [%3.3f]; sc = [%3.3f]; lp_vio=[%3.3f]; consistency_vio=[%4.3f] -------\n'%(iter, np.log10(std_obj), cur_obj, rho_factor, time_elapsed, succ_change, constraint_violation_lp[-1], constraint_violation_local_consistency[-1]))
        print('\n')
        
# %% Update other variables
        
        if iter % 3 == 0:
            rho_factor = min(rho_upper, learning_fact * rho_factor)
            rho_edge = min(rho_upper, learning_fact * rho_edge)
            gamma_val = max(gamma_val * all_params['gamma_factor'], 1)
            
        iter = iter + 1

# %% Return
    result_dict = {'u_node_sol_ADMM': u_node_sol,
                   'v_factor_sol_ADMM': v_factor_sol,
                   'u_factor_sol_ADMM': u_factor_sol,
                   'obj_list_ADMM': obj_list,
                   'constraint_violation_lp_ADMM': constraint_violation_lp,
                   'constraint_violation_consis_ADMM': constraint_violation_local_consistency,
                   'time_elapsed_ADMM': time_elapsed}
    
    return result_dict
    
        
        
        

    
    

    
    
    
    
    
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    