
function run(initial_rho, learning_fact, rho_upper, max_iters, u_factor_solution, dataset_name, file_index)


%% Set parameters and load data information
struct_node_factor_edge = read_uai(dataset_name, file_index);
chdir(fullfile('..', '..', 'matlab', 'functions'));
% load information(node, factor and edge) of specific structure
% corresponding to data_index
node_structure = struct_node_factor_edge.node_structure;
factor_structure = struct_node_factor_edge.factor_structure;
edge_structure = struct_node_factor_edge.edge_structure;
clear struct_node_factor_edge

% get information of nodes
% number of variable nodes in the graph, i.e. |V|
numNodes = node_structure.numNodes;
% number of states of each variable node. nodeStates is a list
% state is an one-hot numerical array, e.g [0,0,1,0]
nodeStates = node_structure.numStates;

% get information of factors
numFactors = factor_structure.numFactors;
% states of factor nodes in the graph. Each factor node has multiple states
factorStates = factor_structure.numStates;


%% Random initialization of variable nodes, factor nodes
% generate initial states of variable nodes
% use cell array to store states of each variable node
u_node_0_cell = cell(1, numNodes);
% initialize the state of each node
node_state_0 = zeros(1, numNodes); 
for n_idx = 1:numNodes
    u_node_0_cell{n_idx} = zeros(nodeStates(n_idx), 1); 
    % generate a random state for each node
    node_state_0(n_idx) = randi([1, nodeStates(n_idx)], 1, 1);  
    % assign the random state to node, resulting in an one-hot vector
    u_node_0_cell{n_idx}(node_state_0(n_idx)) = 1; 
end

% generate initial states of factor nodes
% use cell array to store states of each factor node
u_factor_0_cell = cell(1, numFactors); 
for f_idx = 1:numFactors
    % get states,number of variable nodes connected to this factor node
    % nodesList is a field in factor_structure containing connected nodes
    node_list_in_this_factor = factor_structure.nodesList{f_idx};
    node_state_in_this_factor = node_state_0(node_list_in_this_factor);
    num_node_in_this_factor = numel(node_list_in_this_factor);
    
    % determine the index of the factor configuration in all possible configurations
    node_numStates_in_this_factor = nodeStates(node_list_in_this_factor);
    cum = 0; 
    for n_idx = 1: num_node_in_this_factor-1
        % get current states of variable nodes
        cur_state = node_state_in_this_factor(n_idx); 
        cum = cum + (cur_state-1) * prod( node_numStates_in_this_factor(n_idx+1:end) ); 
    end
    cum = cum + node_state_in_this_factor(end);
    % using above result, assign each factor node a initial state
    u_factor_0_cell{f_idx} = zeros(factorStates(f_idx), 1);
    u_factor_0_cell{f_idx}(cum) = 1;
    
end
% initialize a parameter dictionary
ADMM_params = struct('opt',1,'is_verbose',true,'stop_threshold',1e-6, ...
                     'std_threshold',1e-6,'gamma_val',1,'gamma_factor',1, ...
                     'max_iters', max_iters,'initial_rho', initial_rho, ...
                     'learning_fact', 1 + learning_fact/100,  'rho_upper', rho_upper, ...
                     'history_size',10,'u_node_0',[], 'u_factor_0', [], ...
                     'rel_tol',1e-6,'imsize',[],'save_dir',[], 'projection_lp', 2, ...
                     'y_node_update_maxIter', 100, 'u_factor_solution', u_factor_solution);
                 
% assign initial states of variable and factor nodes 
ADMM_params.u_node_0 = u_node_0_cell; 
ADMM_params.u_factor_0 = u_factor_0_cell; 

% compute the initial objective value
logPotential_0 = compute_log_potential(u_node_0_cell, u_factor_0_cell, node_structure, factor_structure);
ADMM_params.logPotential_0 = logPotential_0; 


%% Execute ADMM algorithm to get results
% compute final result using ADMM algorithm
[best_sol_ADMM, u_node_sol_ADMM, v_factor_sol_ADMM, u_factor_sol_ADMM, ...
 obj_list_ADMM, constraint_violation_lp_ADMM, ...
 constraint_violation_consis_ADMM, time_elapsed_ADMM] = ...
 ADMM_algorithm(node_structure, ...
 factor_structure,edge_structure, ADMM_params);

% compute continuous and discrete objective values based on final results
[logPotential_ADMM, logPotential_ADMM_dis] = ...
    compute_log_potential(u_node_sol_ADMM, u_factor_sol_ADMM, ...
    node_structure, factor_structure);

% plot the figure of how objective value changes during iteration
figure; plot(1:numel(obj_list_ADMM), obj_list_ADMM)

% construct a dictionary to store results
result_struct_ADMM = struct('logPotCurve', obj_list_ADMM, ...
							'logPotFinal', logPotential_ADMM, ...
							'logPotFinal_dis', logPotential_ADMM_dis, ...
                            'lpbox_violation', constraint_violation_lp_ADMM, ...
							'local_consistency_violation', constraint_violation_consis_ADMM, ...
							'nodeLabel', {u_node_sol_ADMM}, ...
							'factorLabel', {u_factor_sol_ADMM}, ...
                            'time_elapsed', time_elapsed_ADMM, ...
							'ADMM_params', {ADMM_params});

                        
%% Store the results
% build a file to store running results if not exist
chdir('..')
if ~exist('results')
    mkdir('results')
end
chdir('results')
if ~exist(dataset_name)
    mkdir(dataset_name)
end
chdir('../functions')
% build full file name
save_path = fullfile('..', 'results', dataset_name);
% create a name of the file that stores results
save_name = strcat('Result_ADMM_', '_type_', u_factor_solution, '_', generate_clock_str(), '.mat');
% save the results in the specified path with name above
chdir(save_path)
save(save_name, 'result_struct_ADMM');


end 
