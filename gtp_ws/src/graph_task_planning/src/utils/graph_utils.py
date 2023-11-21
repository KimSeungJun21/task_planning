import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, unbatch_edge_index

def find_edge_index(ei, src, dest):
    ei_target = torch.tensor([src,dest])
    for idx in range(ei.size(-1)):
        # if torch.equal(ei[:,idx], ei_target):
        if ei[0,idx]==src and ei[1,idx]==dest:
            # print(ei[:,idx])
            # print(src, dest)
            return idx
    return None

def to_fully_connected_graph(x, ei, ea):
    num_node = len(x)
    ea_dim = ea.size(-1)
    ei_list = []
    ea_list = []
    for src in range(num_node):
        for dest in range(num_node):
            ei_list.append(torch.tensor([[src],[dest]]))
            ei_idx = find_edge_index(ei, src, dest)
            ea_list.append(ea[ei_idx,:] if ei_idx is not None else torch.zeros(ea_dim))
    fcg_ei = torch.cat(ei_list, dim=-1)
    fcg_ea = torch.stack(ea_list, dim=0)
    return x, fcg_ei, fcg_ea

def to_dense_graph(ei, ea):
    # num_node = len(x)
    # ea_dim = ea.size(-1)
    dense_mask = []
    for idx in range(ea.size(0)):
        # dense_mask.append(True if torch.is_nonzero(ea[idx,:]) else False)##############수정필요
        dense_mask.append(True if torch.any(ea[idx,:]) else False)
    # if batch == None:
    #     return x, ei[:,dense_mask], ea[dense_mask,:]
    # else:
    #     return x, ei[:,dense_mask], ea[dense_mask,:], batch[dense_mask]
    return ei[:,dense_mask], ea[dense_mask,:]

def get_edge_index_from_edge_attr(ei_csv, ea_csv):
    # get node_name_to_id dictionary
    ei_index = ei_csv.index.to_list()
    node_name_to_id = {v:k for k, v in enumerate(ei_index)}

    ea = torch.Tensor(ea_csv.values) # dataframe to tensor
    ea = ea.to(dtype = torch.float32)

    # write edge_index from index(ID) of edge_attr dataframe
    ei_list = []
    for ei in ea_csv.index.to_list():
        [src, dest] = ei[2:-2].split('\', \'')
        #ei_list.append(torch.tensor([[int(src)], [int(dest)]]))
        ei_list.append(torch.Tensor([[int(node_name_to_id[src])],[int(node_name_to_id[dest])]]))

    ei = torch.cat(ei_list, dim=1)
    return ei, ea

def concat_current_goal_graphs(state_ei, state_ea, goal_ei, goal_ea):
    edge_attr_dim = state_ea.size(-1)
    #match state and goal edge_attr and then concat
    ei_set = set()
    for i_s in range(state_ei.size(dim=-1)):
        ei_set.add(tuple(state_ei[:,i_s].tolist()))
    for i_g in range(goal_ei.size(dim=-1)):
        ei_set.add(tuple(goal_ei[:,i_g].tolist()))
    cat_ei_list = list(map(lambda x:torch.Tensor(x).unsqueeze(-1), ei_set))
    cat_ei = torch.cat(cat_ei_list, dim=1)
    cat_ei = cat_ei.type(torch.long)

    cat_ei_len = cat_ei.size(dim=-1)

    cat_ea = torch.zeros((cat_ei_len, 2*edge_attr_dim))

    for i in range(cat_ei_len):
        for i_s in range(state_ei.size(dim=-1)):
            if torch.equal(cat_ei[:,i],state_ei[:, i_s]):
                cat_ea[i, :edge_attr_dim] = state_ea[i_s,:]
        for i_g in range(goal_ei.size(dim=-1)):
            if torch.equal(cat_ei[:,i],goal_ei[:, i_g]):
                cat_ea[i, edge_attr_dim:] = goal_ea[i_g,:]
    return cat_ei, cat_ea

def concat_fcgs(cur_x, cur_ea, goal_x, goal_ea):
    cur_pos = cur_x[:, -6:]
    goal_pos = goal_x[:, -6:]
    diff_pos = goal_pos - cur_pos
    cat_x = torch.cat([cur_x[:,:-6], diff_pos], dim=-1)
    cat_ea = torch.cat([cur_ea, goal_ea], dim=-1)
    return cat_x, cat_ea

def remove_const_edges(ei, ea):
    #input: concatenated graph(current-goal)
    change_mask = []
    edge_dim = int(ea.size(1)/2)
    for idx in range(ea.size(0)):
        before = ea[idx,:edge_dim]
        after = ea[idx,edge_dim:]
        # print(before)
        # print(after)
        if torch.equal(before,after):
            change_mask.append(False)
        else:
            change_mask.append(True)
            # print(idx)
            # print(ei[:,idx])
            # input()
    return ei[:,change_mask], ea[change_mask,:]

def extract_key_nodes(ei):
    key_node_set = set()
    ei_elements = ei.tolist()[0]
    # print(ei_elements)
    for key_node in ei_elements:
        key_node_set.add(key_node)
    #add robot_hand
    key_node_set.add(0)
    return list(key_node_set)

def concat_edges(state_ei, state_ea, goal_ei, goal_ea):
    #match state and goal edge_attr and then concat
    edge_attr_dim = state_ea.size(-1)
    ei_set = set()
    for i_s in range(state_ei.size(dim=-1)):
        ei_set.add(tuple(state_ei[:,i_s].tolist()))
    for i_g in range(goal_ei.size(dim=-1)):
        ei_set.add(tuple(goal_ei[:,i_g].tolist()))
    cat_ei_list = list(map(lambda x:torch.Tensor(x).unsqueeze(-1), ei_set))
    cat_ei = torch.cat(cat_ei_list, dim=1)
    cat_ei = cat_ei.type(torch.long)
    cat_ei_len = cat_ei.size(dim=-1)
    cat_ea = torch.zeros((cat_ei_len, 2*edge_attr_dim))
    for i in range(cat_ei_len):
        for i_s in range(state_ei.size(dim=-1)):
            if torch.equal(cat_ei[:,i],state_ei[:, i_s]):
                cat_ea[i, :edge_attr_dim] = state_ea[i_s,:]
        for i_g in range(goal_ei.size(dim=-1)):
            if torch.equal(cat_ei[:,i],goal_ei[:, i_g]):
                cat_ea[i, edge_attr_dim:] = goal_ea[i_g,:]
    return cat_ei, cat_ea

def batch_split(batched_graph):
    split_list = []
    x = batched_graph['x']
    ei = batched_graph['edge_index']
    ea = batched_graph['edge_attr']
    batch = batched_graph['batch']

    x_unbatch, x_mask = to_dense_batch(x, batch)
    ei_unbatch = unbatch_edge_index(ei, batch)
    ea_unbatch = []
    start_idx = 0
    for ei_split in ei_unbatch:
        num_edge = ei_split.size(-1)
        end_idx = start_idx+num_edge
        ea_unbatch.append(ea[start_idx:end_idx,:])
        start_idx = end_idx
    
    split_list = [Data(x=x_unbatch[idx,:],
                       edge_index=ei_unbatch[idx],
                       edge_attr=ea_unbatch[idx],
                       batch=torch.zeros_like(batch[batch==idx])) for idx in range(len(ea_unbatch))]

    return split_list

def key_edge_compare(cur_ei, cur_ea, goal_ei, goal_ea):
    cur_edges = []
    goal_edges = []
    key_edges = [] # goal에는 있는데 current에는 없는 edge의 list
    for cur_idx in range(cur_ei.size(-1)):
        ei = cur_ei[:,cur_idx]
        ea = cur_ea[cur_idx,:]
        cur_edges.append(GraphEdge(ei,ea))
    for goal_idx in range(goal_ei.size(-1)):
        ei = goal_ei[:,goal_idx]
        ea = goal_ea[goal_idx,:]
        key_candidate= GraphEdge(ei,ea)
        goal_edges.append(key_candidate)
        is_key = True
        # print('key_edge:\n',key_candidate)
        for comp in cur_edges:
            if edge_equality_checking(key_candidate, comp) is True:
                is_key = False
                break
        if is_key:
            # print('key detected')
            key_edges.append(key_candidate)
            # input()

    return len(key_edges)

class GraphEdge():
    def __init__(self, ei, ea):
        [self.src, self.dest] = ei.tolist()
        self.ea = ea
    def __str__(self):
        return f"GraphEdge\n  src={self.src}\n  dest={self.dest}\n  ea={self.ea.tolist()}"
    
def edge_equality_checking(edge1, edge2):
    if edge1.src==edge2.src and edge1.dest==edge2.dest:
        if torch.equal(edge1.ea, edge2.ea):
            return True
    return False
# def graph_shuffle(ei, shuffle_dict):
#     # edge index value replace
#     def replace_func(x):
#         return shuffle_dict.get(x, x)
    
#     shuffled_ei = ei.clone().apply_(replace_func)

#     # shuffled_ei = ei[:,shuffled_edge_idx]
#     # shuffled_ea = ea[shuffled_edge_idx]

#     return shuffled_ei, shuffled_obj_list