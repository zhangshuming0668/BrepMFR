# -*- coding: utf-8 -*-
import torch
import dgl
import sys
sys.path.append('..')
from models.modules.utils.macro import *

def pad_mask_unsqueeze(x, padlen):  #x[num_nodes]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_ones([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_1d_unsqueeze(x, padlen):  #x[num_nodes]
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_face_unsqueeze(x, padlen):  #x[num_nodes]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen):  # x[num_nodes, num_nodes]
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_d2_pos_unsqueeze(x, padlen): # x[num_nodes, num_nodes, 32]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, 64], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_ang_pos_unsqueeze(x, padlen): # x[num_nodes, num_nodes, 32]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, 64], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)
     
def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):  #x[num_nodes, num_nodes, max_dist]
    xlen1, xlen2, xlen3 = x.size() 
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = -1 * x.new_ones([padlen1, padlen2, padlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    return x.unsqueeze(0)


def collator(items, multi_hop_max_dist, spatial_pos_max):  #items({PYGGraph_1, PYGGraph_1_mian}, {PYGGraph_2, PYGGraph_2_mian}, ..., PYGGraph_batchsize)
    items = [
        (
            item.graph,
            item.node_data,  #node_data[num_nodes, U_grid, V_grid, pnt_feature]
            item.face_area,
            item.face_type,
            item.face_loop,
            item.face_adj,
            item.edge_data,  #edge_data[num_edges, U_grid, pnt_feature]
            item.edge_type,
            item.edge_len,
            item.edge_ang,
            item.edge_conv,
            item.node_degree,
            item.attn_bias,
            item.spatial_pos,
            item.d2_distance,
            item.angle_distance,
            item.edge_path[:, :, :multi_hop_max_dist],  #[num_nodes, num_nodes, max_dist]
            item.label_feature,  #[num_nodes]
            item.data_id
        )
        for item in items
    ]

    (
        graphs,
        node_datas,
        face_areas,
        face_types,
        face_loops,
        face_adjs,
        edge_datas,
        edge_types,
        edge_lens,
        edge_angs,
        edge_convs,
        node_degrees,
        attn_biases,
        spatial_poses,
        d2_distances,
        angle_distances,
        edge_paths,
        label_features,
        data_ids
    ) = zip(*items)  #解压缩

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
        
    max_node_num = max(i.size(0) for i in node_datas)  #计算这批数据中图节点的最大数量 
    max_edge_num = max(i.size(0) for i in edge_datas)
    max_dist = max(i.size(-1) for i in edge_paths)  #计算节点间的最大距离 针对某些图的max_dist都小于multi_hop_max_dist的情况
    max_dist = max(max_dist, multi_hop_max_dist)

    #对数据进行打包并返回, 将各数据调整到同一长度，以max_node_num为准  
    #图长度掩码
    padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in node_datas]
    padding_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in padding_mask_list])
    
    #边长度掩码
    edge_padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in edge_datas]
    edge_padding_mask = torch.cat([pad_mask_unsqueeze(i, max_edge_num) for i in edge_padding_mask_list])
    
    #节点特征
    node_data = torch.cat([i for i in node_datas])  #node_datas(batch_size, [num_nodes, U_grid, V_grid, pnt_feature])

    # face_area = torch.cat([pad_face_unsqueeze(i, max_node_num) for i in face_areas])
    face_area = torch.cat([i for i in face_areas])

    # face_type = torch.cat([pad_face_unsqueeze(i, max_node_num) for i in face_types])
    face_type = torch.cat([i for i in face_types])

    face_loop = torch.cat([i for i in face_loops])

    face_adj = torch.cat([i for i in face_adjs])
    
    #边特征
    edge_data = torch.cat([i for i in edge_datas])  #edge_datas(batch_size, [num_edges, U_grid, pnt_feature])元组
    edge_type = torch.cat([i for i in edge_types])
    edge_len = torch.cat([i for i in edge_lens])
    edge_ang = torch.cat([i for i in edge_angs])
    edge_conv = torch.cat([i for i in edge_convs])
    
    #边编码输入
    edge_path = torch.cat(     #edges_paths(batch_size, [num_nodes, num_nodes, max_dist])
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_paths]
    ).long()
    
    #注意力矩阵
    attn_bias = torch.cat(      #attn_bias(batch_size, [num_nodes+1, num_nodes+1]) 多了一个全图的虚拟节点
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
   
    #空间编码
    spatial_pos = torch.cat(   #spatial_pos(batch_size, [num_nodes, num_nodes])
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )    
    d2_distance = torch.cat(
        [pad_d2_pos_unsqueeze(i, max_node_num) for i in d2_distances]
    )
    angle_distance = torch.cat(
        [pad_ang_pos_unsqueeze(i, max_node_num) for i in angle_distances]
    )
    
    #中心性编码
    # in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]) #in_degree(batch_size, [num_nodes])
    in_degree = torch.cat([i for i in node_degrees])

    batched_graph = dgl.batch([i for i in graphs])

    # face feature type
    batched_label_feature = torch.cat([i for i in label_features])

    data_ids = torch.tensor([i for i in data_ids])

    batch_data = dict(
        padding_mask = padding_mask,       #[batch_size, max_node_num]
        edge_padding_mask = edge_padding_mask,  #[batch_size, max_edge_num]
        graph=batched_graph,

        node_data = node_data,             #[total_node_num, U_grid, V_grid, pnt_feature]
        face_area = face_area,             #[total_node_num]
        face_type = face_type,             #[total_node_num]
        face_loop = face_loop,
        face_adj = face_adj,

        edge_data = edge_data,             #[total_edge_num, U_grid, pnt_feature]
        edge_type = edge_type,             #[total_edge_num]
        edge_len = edge_len,
        edge_ang = edge_ang,
        edge_conv = edge_conv,

        in_degree = in_degree,             #[batch_size, max_node_num]
        out_degree = in_degree,            #[batch_size, max_node_num] #无向图
        attn_bias = attn_bias,             #[batch_size, max_node_num+1, max_node_num+1]
        spatial_pos = spatial_pos,         #[batch_size, max_node_num, max_node_num]
        d2_distance = d2_distance,         #[batch_size, max_node_num, max_node_num, 64]
        angle_distance = angle_distance,   #[batch_size, max_node_num, max_node_num, 64]
        edge_path = edge_path,             #[batch_size, max_node_num, max_node_num, max_dist] 空位用-1填充

        label_feature = batched_label_feature, #[total_node_num]
        id = data_ids
    )
    return batch_data


def collator_st(items, multi_hop_max_dist, spatial_pos_max):
    # source_CAD-----------------------------------------------------------------
    items_source = [
        (
            item["source_data"].graph,
            item["source_data"].node_data,  # node_data[num_nodes, U_grid, V_grid, pnt_feature]
            item["source_data"].face_area,
            item["source_data"].face_type,
            item["source_data"].face_loop,
            item["source_data"].face_adj,
            item["source_data"].edge_data,  # edge_data[num_edges, U_grid, pnt_feature]
            item["source_data"].edge_type,
            item["source_data"].edge_len,
            item["source_data"].edge_ang,
            item["source_data"].edge_conv,
            item["source_data"].in_degree,
            item["source_data"].attn_bias,
            item["source_data"].spatial_pos,
            item["source_data"].d2_distance,
            item["source_data"].angle_distance,
            item["source_data"].edge_path[:, :, :multi_hop_max_dist],  # [num_nodes, num_nodes, max_dist]
            item["source_data"].label_feature,
            item["source_data"].data_id
        )
        for item in items
    ]
    # target_CAD------------------------------------------------------------------
    items_target = [
        (
            item["target_data"].graph,
            item["target_data"].node_data,  # node_data[num_nodes, U_grid, V_grid, pnt_feature]
            item["target_data"].face_area,
            item["target_data"].face_type,
            item["target_data"].face_loop,
            item["target_data"].face_adj,
            item["target_data"].edge_data,  # edge_data[num_edges, U_grid, pnt_feature]
            item["target_data"].edge_type,
            item["target_data"].edge_len,
            item["target_data"].edge_ang,
            item["target_data"].edge_conv,
            item["target_data"].in_degree,
            item["target_data"].attn_bias,
            item["target_data"].spatial_pos,
            item["target_data"].d2_distance,
            item["target_data"].angle_distance,
            item["target_data"].edge_path[:, :, :multi_hop_max_dist],  # [num_nodes, num_nodes, max_dist]
            item["target_data"].label_feature,
            item["target_data"].data_id
        )
        for item in items
    ]
    items = items_source + items_target

    (
        graphs,
        node_datas,
        face_areas,
        face_types,
        face_loops,
        face_adjs,
        edge_datas,
        edge_types,
        edge_lens,
        edge_angs,
        edge_convs,
        in_degrees,
        attn_biases,
        spatial_poses,
        d2_distancees,
        angle_distancees,
        edge_paths,
        label_features,
        data_ids
    ) = zip(*items)  # 解压缩

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")

    batched_graph = dgl.batch([i for i in graphs])

    max_node_num = max(i.size(0) for i in node_datas)  # 计算这批数据中图节点的最大数量
    max_edge_num = max(i.size(0) for i in edge_datas)
    max_dist = max(i.size(-1) for i in edge_paths)  # 计算节点间的最大距离 针对某些图的max_dist都小于multi_hop_max_dist的情况
    max_dist = max(max_dist, multi_hop_max_dist)

    # 对数据进行打包并返回, 将各数据调整到同一长度，以max_node_num为准
    # 图长度掩码
    padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in node_datas]
    padding_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in padding_mask_list])

    # 边长度掩码
    edge_padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in edge_datas]
    edge_padding_mask = torch.cat([pad_mask_unsqueeze(i, max_edge_num) for i in edge_padding_mask_list])

    # 节点特征
    node_data = torch.cat([i for i in node_datas])  # node_datas(batch_size, [num_nodes, U_grid, V_grid, pnt_feature])

    # face_area = torch.cat([pad_face_unsqueeze(i, max_node_num) for i in face_areas])
    face_area = torch.cat([i for i in face_areas])

    # face_type = torch.cat([pad_face_unsqueeze(i, max_node_num) for i in face_types])
    face_type = torch.cat([i for i in face_types])

    face_loop = torch.cat([i for i in face_loops])

    face_adj = torch.cat([i for i in face_adjs])

    # 边特征
    edge_data = torch.cat([i for i in edge_datas])  # edge_datas(batch_size, [num_edges, U_grid, pnt_feature])元组
    edge_type = torch.cat([i for i in edge_types])
    edge_len = torch.cat([i for i in edge_lens])
    edge_ang = torch.cat([i for i in edge_angs])
    edge_conv = torch.cat([i for i in edge_convs])

    # 边编码输入
    edge_path = torch.cat(  # edges_paths(batch_size, [num_nodes, num_nodes, max_dist])
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_paths]
    ).long()

    # 注意力矩阵
    attn_bias = torch.cat(  # attn_bias(batch_size, [num_nodes+1, num_nodes+1]) 多了一个全图的虚拟节点
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )

    # 空间编码
    spatial_pos = torch.cat(  # spatial_pos(batch_size, [num_nodes, num_nodes])
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    d2_distance = torch.cat(
        [pad_d2_pos_unsqueeze(i, max_node_num) for i in d2_distancees]
    )
    angle_distance = torch.cat(
        [pad_ang_pos_unsqueeze(i, max_node_num) for i in angle_distancees]
    )

    # 中心性编码
    # in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]) #in_degree(batch_size, [num_nodes])
    in_degree = torch.cat([i for i in in_degrees])

    batched_label_feature = torch.cat([i for i in label_features])

    data_ids = torch.tensor([i for i in data_ids])

    batch_data = dict(
        padding_mask=padding_mask,  # [batch_size, max_node_num]
        edge_padding_mask=edge_padding_mask,  # [batch_size, max_edge_num]
        graph=batched_graph,

        node_data=node_data,  # [total_node_num, U_grid, V_grid, pnt_feature]
        face_area=face_area,  # [batch_size, max_node_num] / [total_node_num]
        face_type=face_type,  # [batch_size, max_node_num] / [total_node_num]
        face_loop=face_loop,
        face_adj=face_adj,

        edge_data=edge_data,  # [total_edge_num, U_grid, pnt_feature]
        edge_type=edge_type,
        edge_len=edge_len,
        edge_ang=edge_ang,
        edge_conv=edge_conv,

        in_degree=in_degree,  # [batch_size, max_node_num]
        out_degree=in_degree, # [batch_size, max_node_num] #无向图
        attn_bias=attn_bias,  # [batch_size, max_node_num+1, max_node_num+1]
        spatial_pos=spatial_pos,  # [batch_size, max_node_num, max_node_num]
        d2_distance=d2_distance,  # [batch_size, max_node_num, max_node_num, 64]
        angle_distance=angle_distance,  # [batch_size, max_node_num, max_node_num, 64]
        edge_path=edge_path,  # [batch_size, max_node_num, max_node_num, max_dist] 空位用-1填充

        label_feature=batched_label_feature,  # [total_node_num]
        id=data_ids
    )
    return batch_data