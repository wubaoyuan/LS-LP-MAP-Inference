function [best_sol,u_node_sol, v_factor_sol, u_factor_sol, obj_list, ...
          constraint_violation_lp, constraint_violation_local_consistency, time_elapsed] ...
               = ADMM_algorithm(node_structure, ...
                 factor_structure,edge_structure, all_params)
       
             
%% Get parameters and state of nodes 

%------------------------ parameters ----------------------
opt = all_params.opt;
is_verbose = all_params.is_verbose;
stop_threshold = all_params.stop_threshold;
std_threshold = all_params.std_threshold;
rho_upper = all_params.rho_upper;
max_iters = all_params.max_iters;
initial_rho = all_params.initial_rho;
gamma_val = all_params.gamma_val;
learning_fact = all_params.learning_fact; 
% how far back in time to see whether the cost has stabilized
history_size = all_params.history_size; 

%---------------------- node parameters ----------------------
numNodes = node_structure.numNodes; 
nodeStates = node_structure.numStates; 

numFactors = factor_structure.numFactors; 
factorStates = factor_structure.numStates; 

numEdges = edge_structure.numEdges; 
% get index of factor and variable nodes, each edge connect a pair of nodes
factorIndexAll  = edge_structure.factorNodePair_List(1,:); 
nodeIndexAll  = edge_structure.factorNodePair_List(2,:); 

% 1 x numNodes cell, each cell is  numNodeStates(i) x 1 vector
thetaNodeCell  = node_structure.nodeScore;    
% 1 x numFactors cell, each cell is  numFactorStates(i) x 1 vector
thetaFactorCell = factor_structure.factorScore; 

%% Get information of initialization based on input variable 'all_params'

% initialize information of variable, factor and extra variable nodes
u_node_sol  = all_params.u_node_0; % 1 x numNodes cell 
u_factor_sol = all_params.u_factor_0; % 1 x numFactors cell 
v_factor_sol = u_factor_sol; 

% initialize lambda of factor nodes based on factor stucture in a cell array
lambda_factor = cellfun(@(x) x .* 0, u_factor_sol, 'UniformOutput',false);
% initialize lambda of edges in a cell array
lambda_edge = cell(1, numEdges); 
% node_idx = edge_structure.factorNodePair_List(2, e_idx);
nodeState_of_each_edge = nodeStates( edge_structure.factorNodePair_List(2, :) ); 
% set each edge a zero vector based on state number of nodes
for e_idx = 1:numEdges
      lambda_edge{e_idx} = zeros(nodeState_of_each_edge(e_idx), 1);  % nodeState x 1 matrix 
end

% violation of equivalence constrains
constraint_violation_lp = []; 
% violation of local consistency constraint
constraint_violation_local_consistency = []; 

% initial value of rho_i and rho_alpha
rho_factor = 3.*initial_rho; 
rho_edge = 1*initial_rho;

% compute standard deviation of objective value if length of obj_list >= history size
obj_list = [];
[std_obj] = compute_std_obj(obj_list,history_size);

[~, prev_idx] = cellfun(@max, u_node_sol, 'UniformOutput',false);
succ_change = 1;
best_sol = [];
best_bin_obj = 0;
prev_idx_vec = cell2mat( prev_idx ); 


%% ADMM algorithm
% update variables in the order: extra variable nodes, factor nodes, variable nodes, dual variables

time_elapsed = 0;
constraint_stopping = true;
iter = 1; 

% set the conditions that will stop the algorithm
while constraint_stopping && std_obj>=std_threshold && iter<=max_iters

    fprintf('------------- start of [iter #%d]: ------------------\n',iter); 
    
%% Part 1: update extra variable nodes
    
    fprintf('+ update v_factor ... '); 
    % time recording
    tic; 
    
    % the first step to update extra variable nodes
    % v_factor_sol = u_factor_sol + lambda_factor ./ rho_factor
    v_factor_sol_transform = cellfun(@(x, y) (x + y./ rho_factor)', u_factor_sol, lambda_factor, 'UniformOutput',false);    
    v_factor_sol_vec = [v_factor_sol_transform{:}]';
    clear v_factor_sol_transform
    
    % make a projection onto sphere to get the updated results
    v_factor_sol_vec_proj = project_shifted_Lp_ball( v_factor_sol_vec, 0.5*ones(numel(v_factor_sol_vec) ,1), all_params.projection_lp); 
    
    % assign the updated projection value to each extra variable node 
    cum = 0; 
    for f_idx = 1:numFactors
        num_state_this_factor = factorStates(f_idx);
        cum_new = cum + num_state_this_factor;
        v_factor_sol{f_idx} = v_factor_sol_vec_proj(cum+1: cum_new);
        cum = cum_new; 
    end
    
    t = toc;  
    % show the execution time
    fprintf(' finished in [%3.3f] seconds \n',t);
    time_elapsed = time_elapsed+t;
    
%% Part 2:  update factor nodes
    fprintf('+ update u_factor ... ');
    tic;  

    % This is a convex problem, so we use active-set algorithm in quadratic
    % programming solver to handle it.
    options = optimoptions(@quadprog,'Display','off','Algorithm', 'interior-point-convex');
    for f_idx = 1:numFactors
        edgeIndexList = find(factorIndexAll == f_idx);
        nodeIndexList = nodeIndexAll(edgeIndexList);
        
        % find the initial A matrix and b vector
        A_matrix = rho_factor .* eye(factorStates(f_idx));
        b_vec = thetaFactorCell{f_idx}+ rho_factor .* v_factor_sol{f_idx} - lambda_factor{f_idx};

        % update A matrix and b vector in each edge
        for e_idx = 1:numel(edgeIndexList)
            edge = edgeIndexList(e_idx);
            node = nodeIndexList(e_idx);
            M_matrix = edge_structure.M{edge};
            A_matrix = A_matrix + rho_edge .* M_matrix' * M_matrix;
            b_vec = b_vec +  M_matrix' * ( rho_edge .* u_node_sol{node} + lambda_edge{edge} ); 
        end
        
      
       % update miu of factor nodes through the specified algorithm --- u_factor_solution in params           
       % solution 1: solve the constrained QP with QPC 
       if strcmp(all_params.u_factor_solution, 'exact-qpc')  
           fac_state = length(b_vec);
           Aeq = ones(1, fac_state); beq = 1; 
           lb = zeros(fac_state, 1);  ub = ones(fac_state, 1);
           u_factor_sol{f_idx} = quadprog(A_matrix, -b_vec, ...
                  [], [], Aeq, beq, lb, ub, [], options);

       % solution 2: linear-proximal approximation, using simplex projection 
       elseif strcmp(all_params.u_factor_solution, 'linear-proximal')         
          eta = 1.01 * norm(A_matrix, 'fro')^2;
          temp_vec = u_factor_sol{f_idx} + (b_vec - A_matrix * u_factor_sol{f_idx}) ./ eta;
          %temp_vec = b_vec./eta + (eye(size(A_matrix)) - A_matrix./eta) * u_factor_sol{f_idx};
          u_factor_sol{f_idx} = project_simplex(temp_vec')'; 
       end
    end
 
    t = toc;   
    fprintf(' finished in [%3.3f] seconds \n',t);
    
%% Part 3: update variable nodes

    fprintf('+ update u_node ... '); 
    tic; 

    for node = 1:numNodes
        % nodeScore in node_structure
        numerator = thetaNodeCell{node};
        % find index of connected edges
        edgeIndexList = find( nodeIndexAll == node);
        % find number of connected edges
        numEdgesConnected = length(edgeIndexList);
        % find index of connected factor nodes
        factorIndexList = node_structure.factorNeighborCell{node};
        % update numerator(b) based on edges, factor nodes and M related to
        % this variable node
        for f_idx = 1:numel(factorIndexList)
            edge = edgeIndexList(f_idx);
            fac = factorIndexList(f_idx);
            M_matrix = edge_structure.M{edge};
            numerator = numerator + rho_edge .* M_matrix * u_factor_sol{fac} - lambda_edge{edge};
        end
        
        % update the denominator(a)
        denominator = numEdgesConnected * rho_edge;
        % update miu of this variable node
        % consider the case when connected edge is none
        if denominator==0 
            u_node_i = numerator ./ sum(numerator);
        else
            u_node_i = numerator ./denominator;
        end
        u_node_sol{node} = u_node_i;
    end
    % output the running time of updating variable node
    t = toc;    
    fprintf('finished in [%3.3f] seconds \n',t); 
    time_elapsed = time_elapsed+t;

%% Part 4: update dual variables

    fprintf('+ update dual variables ... '); 
    tic; 
    
    % update lambda_i
    if iter < all_params.y_node_update_maxIter          
        lambda_factor = cellfun(@(x, y, z) x + gamma_val * rho_factor .* (y - z), ...
              lambda_factor, u_factor_sol, v_factor_sol, 'UniformOutput',false);
    end 

    for e_idx = 1:numEdges
         node = nodeIndexAll(e_idx);  
         fac = factorIndexAll(e_idx); 
         M_matrix = edge_structure.M{e_idx};
         lambda_edge{e_idx} = lambda_edge{e_idx} +  gamma_val * rho_edge .* ( u_node_sol{node} -  M_matrix * u_factor_sol{fac} ); 
    end
    
    t = toc;  
    fprintf(' finished in [%3.3f] seconds \n',t); 
    time_elapsed = time_elapsed+t;

%% Evaluate constraint violation and objective values
    
    % lp-box consistency,compute violation of equivalence constraint 
    temp_violation_lp = sqrt( sum( cell2mat( cellfun( @(x, y) norm(x-y, 'fro'), u_factor_sol, v_factor_sol,  'UniformOutput',false) ).^2 ) );
    % judge whether the algorithm should stop
    constraint_stopping = temp_violation_lp>=stop_threshold;
    % store all violations of equivalence constraint in each iteration
    constraint_violation_lp = [constraint_violation_lp; temp_violation_lp];
    clear temp_violation_lp
    
    % local consistency, compute violation of local consistency constraint
    constraint_violation_local_consistency_vec = zeros(numEdges, 1);
    for e_idx = 1:numEdges
        fac = edge_structure.factorNodePair_List(1, e_idx); 
        node = edge_structure.factorNodePair_List(2, e_idx); 
        M_matrix = edge_structure.M{e_idx}; 
        constraint_violation_local_consistency_vec(e_idx) = ...
            norm(u_node_sol{node} - M_matrix * u_factor_sol{fac}, 'fro');
    end
    
    % store all violations of local consistency in each iteration
    constraint_violation_local_consistency = [constraint_violation_local_consistency; ...
        sqrt( sum(constraint_violation_local_consistency_vec.^2) )];
    clear constraint_violation_local_consistency_vec
    
    % compute objective value and its standard deviation
    obj_list = [obj_list; compute_log_potential(u_node_sol, u_factor_sol, node_structure, factor_structure)];
    std_obj = compute_std_obj(obj_list,history_size);

%% Find and show overall results
    % try a different stopping criterion
    %[~, cur_idx]  = cellfun(@max, u_node_sol, 'UniformOutput',false);
    cur_idx = cellfun(@max, u_node_sol, 'UniformOutput',false);
    cur_idx_vec = cell2mat( cur_idx );
    if iter > 10
        succ_change = sum(cur_idx_vec~=prev_idx_vec)/numel(cur_idx_vec); 
    end
    prev_idx = cur_idx;
    prev_idx_vec = cell2mat(prev_idx);
    cur_obj = obj_list(end);
    
    % cur_sol = cellfun(@(x) ( decimalToBinaryVector(x, 2, 'LSBFirst') )', cur_idx, 'UniformOutput',false);
    % cur_bin_obj = compute_cost(cur_sol, u_factor_sol, node_structure, factor_structure);
    cur_sol = [];
    cur_bin_obj = 0;
    
    % maintain best binary solution so far; in case the cost function
    % oscillates
    if best_bin_obj >= cur_bin_obj
        best_bin_obj = cur_bin_obj;
        best_sol = u_node_sol;
    end
    
    % print the overall results
    fprintf('------ end of [iter #%d]: std = [%3.2f]; obj = [%5.2f]; , rho = [%3.3f], time = [%3.3f]; sc = [%3.3f]; lp_vio=[%3.3f]; consistency_vio=[%4.3f] -------\n',... 
                  iter,log10(std_obj), cur_obj, rho_factor, time_elapsed, succ_change, constraint_violation_lp(end), constraint_violation_local_consistency(end)); 
    fprintf('\n');
    
%% Update other variables    

    % update rho in each three iterations
    if mod(iter,3)==0
        rho_factor = min(rho_upper, learning_fact*rho_factor);
        rho_edge = min(rho_upper, learning_fact*rho_edge);
        gamma_val = max(gamma_val*all_params.gamma_factor,1);
    end  
    % increment counter
    iter = iter + 1;
    
end
% The end of ADMM algorithm

return;
% The end of whole function
