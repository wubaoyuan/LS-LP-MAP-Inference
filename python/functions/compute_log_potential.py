# Log_potential function
def compute_log_potential(u_node_score, u_factor_score, node_structure, factor_structure):
    
    nodeScore = node_structure['nodeScore'] 
    obj_node = sum(sum(u_node_score * nodeScore))

    factorScore = factor_structure['factorScore']
    result = [u_factor_score[i] * factorScore[i].reshape(1,factorScore[i].size) for i in range(u_factor_score.shape[0])]
    result1 = [sum(sum(result[i])) for i in range(len(result))]
    obj_factor = sum(result1)
    
    obj_total = obj_factor + obj_node
           
    return obj_total