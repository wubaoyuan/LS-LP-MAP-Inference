%% log_potential function, larger is better 
function [obj, obj_dis] = compute_log_potential(u_node_score, u_factor_score, node_structure, factor_structure)
% obj = - sum_i sum_{x_i} ( theta_i' * u_i ) - sum_{\alpha} sum_{x_\alphai} theta_\alpha' * u_alpha
% theta_i = - node_structure.nodeScore;  theta_\alpha = - factor_structure.factorScore; 

% obj = sum(sum(u_node_sol .* node_structure.nodeScore)) + sum(sum(u_factor_sol .* factor_structure.factorScore)); 

obj_cell_node = cellfun(@(x,y) x.*y,  u_node_score, node_structure.nodeScore, 'UniformOutput',false);
obj_node = sum( cellfun(@sum, obj_cell_node,'UniformOutput', true) );
clear obj_cell_node

obj_cell_factor = cellfun(@(x,y) x.*y,  u_factor_score, factor_structure.factorScore, 'UniformOutput',false);
obj_factor = sum( cellfun(@sum, obj_cell_factor,'UniformOutput', true) );
clear obj_cell_factor

obj = obj_node + obj_factor;

if nargout == 2
    %% compute the discrete objective value
    u_node_score_dis = cell_continuous_to_discrete(u_node_score);
    u_factor_score_dis = cell_continuous_to_discrete(u_factor_score);

    obj_cell_node_dis = cellfun(@(x,y) x.*y,  u_node_score_dis, node_structure.nodeScore, 'UniformOutput',false);
    obj_node_dis = sum( cellfun(@sum, obj_cell_node_dis,'UniformOutput', true) );
    clear obj_cell_node_dis

    obj_cell_factor_dis = cellfun(@(x,y) x.*y,  u_factor_score_dis, factor_structure.factorScore, 'UniformOutput',false);
    obj_factor_dis = sum( cellfun(@sum, obj_cell_factor_dis,'UniformOutput', true) );
    clear obj_cell_factor

    obj_dis = obj_node_dis + obj_factor_dis;
end

end

function discrete_cell = cell_continuous_to_discrete(continuous_cell)

numCell = length(continuous_cell); 
discrete_cell = cell(1, numCell); 
for c = 1:numCell
    vec = continuous_cell{c}; 
    bb = zeros(length(vec), 1); 
    [~, loc_max] = max(vec);
    bb(loc_max) = 1;
    discrete_cell{c} = bb;
end

end