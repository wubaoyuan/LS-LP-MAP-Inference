%% computes the std of the objective history

function [std_obj] = compute_std_obj(obj_list,history_size)
std_obj = Inf;
if numel(obj_list)>=history_size
    std_obj = std(obj_list(end-history_size+1:end));
    
    % normalize 
    std_obj = std_obj/abs(obj_list(end));
end
return;