%--------------------------------------------------------------------------
function struct_node_factor_edge = read_uai(dataset_name, data_index)
%-------------------------------------------------------------------------- 
% change the working directory to the file of specific dataset
chdir(fullfile('..', '..', 'dataset', dataset_name));
% Get name of all uai files
fileNameList = dir( '*.uai' );
data_name = fileNameList(data_index).name;
   
fid = fopen([data_name]);

i = 1;
% test for end of the file
while ~feof(fid)  % if it is not the end row
     % read lines in the uai file
     data_cell{i}=fgets(fid); 
     i=i+1; 
end 
fclose(fid);  clear i
data_cell = data_cell';

% derive number of variable and factor nodes, states of variable nodes 
numNodes = str2num( data_cell{2} );
nodeStates = str2num( data_cell{3} );
% number of variable and factor nodes in total
% 1160
numFactorsAll = str2num( data_cell{4} ); 

%% obtain the node and factor index 
% find the node index and index of nodes that a factor node connects
node_or_factor_index = cell(2, numFactorsAll); 
% record two indexes of nodes that a factor node connects
nodeIndexCell_of_factor = []; 
% number of factor nodes
numFactors = 0; 
for f = 1:numFactorsAll
    aa = str2num( data_cell{4+f} );
    if aa(1) == 1
        node_idx = aa(2) + 1; 
        % unaryScoresCell{node} = aa(2);
        node_or_factor_index{1, f} = 'node'; 
        % record the index of nodes
        node_or_factor_index{2, f} = node_idx; 
    elseif aa(1) > 1
        % return a column vector with node index
        node_list = aa(2:end)' + 1; 
        node_or_factor_index{1, f} = 'factor'; 
        % record the indexes of nodes that this factor connects
        node_or_factor_index{2, f} = node_list; 
        
        nodeIndexCell_of_factor{numFactors + 1} = node_list; 
        
        numFactors = numFactors + 1; 
    end
end


%% obtain the node and factor scores
% store the scores of each node (length is equal to number of states)
unaryScoresCell = cell(1, numNodes);
% store the scores of state of each factor
% length = # of connected nodes * # of node states
factorScoresCell = cell(1, numFactors); 
factorStates = zeros(1, numFactors);
fac_idx = 0; 
for f = 1:numFactorsAll
    % the value in data_cell is potential value, 
    % but we need to save logPot into factors
    %%%%%%%%
    aa = log(str2num( data_cell{5 + numFactorsAll + (f-1)*3+2} ) + 1e-20*eps);  % be careful here, (f-1)*3+2
    if ~isempty(aa)
        if strcmp(node_or_factor_index{1, f}, 'node')
            node_idx = node_or_factor_index{2, f}; 
            unaryScoresCell{node_idx} = aa'; 
            
        elseif strcmp(node_or_factor_index{1, f}, 'factor') == 1
            factorScoresCell{fac_idx+1} = aa'; 
            factorStates(fac_idx+1) = numel(aa);
            fac_idx = fac_idx + 1;
        end
    end
end

% if there is no unary score for some nodes, set the scores (logPot) as 0
for n_idx = 1:numNodes
    if isempty(unaryScoresCell{n_idx})
        unaryScoresCell{n_idx} = zeros(nodeStates(n_idx), 1); 
    end
end


%% node_structure, including the number of nodes, the unary score of different states of each node
node_structure = struct('numNodes', numNodes, 'numStates', nodeStates); % , 'nodeOrder', nodeIndex'
node_structure.nodeScore = unaryScoresCell;

%% factor_structure, the factor size, including nodes, and scores of different configurations of each factor
%factor_structure = struct('numFactors', numFactors, 'nodesListCell', nodeIndexCell_of_factor, 'numStates', factorStates, 'factorScore', factorScoresCell); 

factor_structure = struct('numFactors', numFactors); 
factor_structure.nodesList = nodeIndexCell_of_factor;
factor_structure.numStates = factorStates;
factor_structure.factorScore = factorScoresCell;


%% edge_structure, the edge connecting factors and nodes
edge_structure = struct('numEdges', [], 'factorNodePair_List', []);
temp_num = 0; 
for f_idx = 1:factor_structure.numFactors
    % index of variable nodes connected to this factor node
    nodeList_in_factor =  factor_structure.nodesList{f_idx};
    % number of variable nodes connected to this factor node 
    numNodes_in_factor = numel(nodeList_in_factor);
    
    % the 1st row is the index of factor node
    edge_structure.factorNodePair_List(1, temp_num + 1 : temp_num+numNodes_in_factor) = ...
        f_idx .* ones(1, numNodes_in_factor);    
    % the 2nd row is the index of nodes connected to this factor node
    edge_structure.factorNodePair_List(2, temp_num + 1 : temp_num+numNodes_in_factor) = ...
        factor_structure.nodesList{f_idx}';      
    
   temp_num = temp_num + numNodes_in_factor; 
end
clear temp_num
%%%% 得看一下factorNodePair_List
edge_structure.numEdges = size(edge_structure.factorNodePair_List, 2); 

% define a new field recording information of connected factor nodes 
node_structure.factorNeighborCell = cell(1, node_structure.numNodes);
for node_idx = 1:node_structure.numNodes
    edge_idx = edge_structure.factorNodePair_List(2, :) == node_idx; 
    % the indexes of factors nodes connected to this variable node 
    node_structure.factorNeighborCell{node_idx} = ...
        edge_structure.factorNodePair_List(1, edge_idx); 
end

%% define the M matrix for each edge, i.e., the consistency matrix
for node_idx = 1:numNodes
    
    % node_idx = nodeIndex(node_idx);

    node_state_i = nodeStates(node_idx); 
    % identity matrix with dimension equal to number of node states
    basic_mat = eye(node_state_i);
    
    % find index of connected edges
    edgeIndexList = find( edge_structure.factorNodePair_List(2,:) == node_idx); 
    % number of connected edges
    numEdgesConnected = length(edgeIndexList);
    % index of connected factor nodes
    factorIndexList = node_structure.factorNeighborCell{node_idx}; 
    for f_idx = 1:numel(factorIndexList)
        edge = edgeIndexList(f_idx); 
        fac = factorIndexList(f_idx);
        % factor_state_i = factorStates(f_idx); 
        
        % index of connected nodes
        node_vec_of_this_fac = factor_structure.nodesList{fac};
        % number of states of connected nodes
        node_state_vec_of_this_fac = nodeStates(node_vec_of_this_fac);
        % find the order of this node in connected nodes of connected
        % factor node 
        node_order = find( node_vec_of_this_fac == node_idx ); 

        if node_order == 1                                                
            % if this node is the first node in this factor
            M_matrix = kron( basic_mat, ones(1, prod(node_state_vec_of_this_fac(2:end))) ); 
        elseif node_order == numel(node_state_vec_of_this_fac)   
            % if this node is the last node in this factor
            M_matrix = kron( ones(1, prod(node_state_vec_of_this_fac(1:end-1))), basic_mat ); 
        else
            % if this node is a middle node in this factor
            before_states = prod(node_state_vec_of_this_fac(1:node_order-1)); 
            after_states = prod(node_state_vec_of_this_fac(1+node_order : end)); 
            M_temp = kron( basic_mat, ones(1, after_states) ); 
            M_matrix = kron( ones(1, before_states), M_temp );
        end

        edge_structure.M{edge} = sparse(M_matrix); 
    end
end

%% combine node, factor and edge structure
% construct a struct to store information of nodes, factors and edges 
struct_node_factor_edge = struct('node_structure', node_structure, ...
                                 'factor_structure', factor_structure, ...
                                 'edge_structure', edge_structure);

