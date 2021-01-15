import numpy as np
import os
import scipy.sparse as sp
import glob


def read_uai(file_index, dataset_name):
    path = os.path.join('..', '..', 'dataset', dataset_name)
    os.chdir(path)
    files = glob.glob('*.uai')
    filename = files[file_index]
    f = open(filename)
    lines = f.readlines()
    data = []
    for line in lines:
        data.append(line[:-1])

    # Derive number of variable and factor nodes, states of variable nodes 
    numNodes = int(data[1])
    # Get node states
    nodeStates = np.array(data[2].split()).astype(int)
    # Number of variable and factor nodes in total
    numFactorAll = int(data[3])
    
    
# %% Obtain the node and factor index 
    # Find the node index and index of nodes that a factor node connects
    node_or_factor_index = {'node': [], 'factor': []}
    # Record two indexes of nodes that a factor node connects
    nodeIndexCell_of_factor = []
    # Number of factor nodes
    numFactors = 0
    
    for f in range(numFactorAll):
        if len(data[4].split('\t')) == 1:
            aa = data[4 + f].split()
        else:
            aa = data[4 + f].split('\t')
        if aa[0] == '1':
            node_idx = int(aa[1]) + 1
            node_or_factor_index['node'].append(node_idx)
        
        else:
            # Return a vector with node index
            node_list = np.array(aa[1:]).astype(int) + 1
            # Record the indexes of nodes that this factor connects
            node_or_factor_index['factor'].append(node_list)
            
            nodeIndexCell_of_factor.append(node_list)
            
            numFactors = numFactors + 1
            
# %% Obtain the node and factor scores
    # Store the scores of each node (length is equal to number of states)
    unaryScoresCell = [0] * numNodes
    # Store the scores of state of each factor
    # Length = # of connected nodes * # of node states
    factorScoresCell = [0] * numFactors
    factorStates = np.zeros(numFactors)
    fac_idx = 0
    eps = 2.2204e-16
    for f in range(numFactorAll):
        # The value in data_cell is potential value, 
        # We need to save logPot into factors
        score = np.array(data[4 + numFactorAll + f * 3 + 2].split()).astype(float)
        aa = np.log(score + 1e-20 * eps)
        if len(aa) != 0:
            if f < numNodes:
                # return [node_or_factor_index['node'],numFactorAll,numNodes]
                node_idx = node_or_factor_index['node'][f]
                unaryScoresCell[node_idx - 1] = aa
            else:
                factorScoresCell[fac_idx] = aa
                factorStates[fac_idx] = len(aa)
                fac_idx = fac_idx + 1
    
    # If there is no unary score for some nodes, set the scores (logPot) as 0
    for n_idx in range(numNodes):
        if len(unaryScoresCell[n_idx]) == 0:
            unaryScoresCell[n_idx] = np.zeros(nodeStates[n_idx])
          
    
# %% node_structure, including the number of nodes, the unary score of different states of each node
    node_structure = {'numNodes': numNodes, 'numStates': nodeStates}
    node_structure['nodeScore'] = np.array(unaryScoresCell)
    
# %% factor_structure, the factor size, including nodes, and scores of different configurations of each factor
    factor_structure = {'numFactors': numFactors, 'nodesList': nodeIndexCell_of_factor}
    factor_structure['numStates'] = factorStates.astype(int)
    factor_structure['factorScore'] = np.array(factorScoresCell)
    
# %% edge_structure, the edge connecting factors and nodes
    edge_structure = {'numEdges': [], 'factorNodePair_List': [np.array([]), np.array([])]}
    temp_num = 0
    for f_idx in range(factor_structure['numFactors']):
        # Index of variable nodes connected to this factor node
        nodeList_in_factor = factor_structure['nodesList'][f_idx]
        # Number of variable nodes connected to this factor node
        numNodes_in_factor = len(nodeList_in_factor)
        
        # Index of factor node
        edge_structure['factorNodePair_List'][0] = np.concatenate((edge_structure['factorNodePair_List'][0], 
                                                    (f_idx + 1) * np.ones(numNodes_in_factor))).astype(int)
        # Index of nodes connected to this factor node
        edge_structure['factorNodePair_List'][1] = np.concatenate((edge_structure['factorNodePair_List'][1], 
                                                    factor_structure['nodesList'][f_idx])).astype(int)
        temp_num = temp_num + numNodes_in_factor
        
    edge_structure['numEdges'] = len(edge_structure['factorNodePair_List'][0])
    edge_structure['M'] = [0] * edge_structure['numEdges']
    # Define a new field recording information of connected factor nodes 
    node_structure['factorNeighborCell'] = [0] * node_structure['numNodes']
    for node_idx in range(node_structure['numNodes']):
        edge_idx = edge_structure['factorNodePair_List'][1] == (node_idx + 1)
        # The indexes of factors nodes connected to this variable node
        node_structure['factorNeighborCell'][node_idx] = edge_structure['factorNodePair_List'][0][edge_idx == 1].astype(int)
    
# %% define the M matrix for each edge, i.e., the consistency matrix
    for node_idx in range(numNodes):
        node_state_i = nodeStates[node_idx]
        # Identity matrix with dimension equal to number of node states
        basic_mat = np.eye(node_state_i)
        # Find index of connected edges
        edgeIndexList = np.where(edge_structure['factorNodePair_List'][1] == (node_idx + 1))[0]
        # Index of connected factor nodes
        factorIndexList = node_structure['factorNeighborCell'][node_idx]
        
        for f_idx in range(len(factorIndexList)):
            edge = edgeIndexList[f_idx]
            fac = factorIndexList[f_idx]
            
            # Index of connected nodes
            node_vec_of_this_fac = factor_structure['nodesList'][fac - 1]
            # Number of states of connected nodes
            node_state_vec_of_this_fac = nodeStates[node_vec_of_this_fac - 1]
            # Find the order of this node in connected nodes of connected factor node
            node_order = np.where(node_vec_of_this_fac == (node_idx + 1))[0] + 1
            
            if node_order == 1:
                # If this node is the first node in this factor
                M_matrix = np.kron(basic_mat, np.ones(np.prod(node_state_vec_of_this_fac[1:])))
            elif node_order == len(node_state_vec_of_this_fac):
                # If this node is the last node in this factor
                M_matrix = np.kron(np.ones(np.prod(node_state_vec_of_this_fac[:-1])), basic_mat)
            else:
                # If this node is a middle node in this factor
                before_states = np.prod(node_state_vec_of_this_fac[:node_order - 2])
                after_states = np.prod(node_state_vec_of_this_fac[node_order:])
                M_temp = np.kron(basic_mat, np.ones(after_states))
                M_matrix = np.kron(np.ones(before_states), M_temp)
        
            edge_structure['M'][edge] = sp.csr_matrix(M_matrix)
    
# %% Combine information of node, factor and edge and save them        
    struct_node_factor_edge = {'node_structure': node_structure, 
                               'factor_structure': factor_structure,
                               'edge_structure': edge_structure}
    
    return [struct_node_factor_edge, filename]


            
        
            


    



    
    
    
    
    
