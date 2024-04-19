from typing import Callable, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from .multihead_attention import MultiheadAttention
from .feature_encoders import SurfaceEncoder, CurveEncoder

class GraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        pre_layernorm: bool = False,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.pre_layernorm = pre_layernorm

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=self.__class__.__name__
        )

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x, attn


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class NonLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(output_dim)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = F.relu(self.bn1(self.linear1(inp)))
        x = F.relu(self.bn2(self.linear2(x)))
        return x


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
            self, num_heads, num_degree, hidden_dim, n_layers
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        # node_feature encode
        self.surf_encoder = SurfaceEncoder(
            in_channels=7, output_dims=int(0.5*hidden_dim)
        )
        # self.face_area_encoder = nn.Linear(1, int(0.125*hidden_dim), bias=False)
        self.face_area_encoder = NonLinear(1, int(0.125*hidden_dim))
        self.face_type_encoder = nn.Embedding(6, int(0.125*hidden_dim), padding_idx=0)
        self.face_loop_encoder = nn.Embedding(256, int(0.125*hidden_dim), padding_idx=0)
        self.degree_encoder = nn.Embedding(num_degree, int(0.125*hidden_dim), padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, x, face_area, face_type, face_loop, face_degree, padding_mask):
        # x [total_node_num, U_grid, V_grid, pnt_feature]
        # padding_mask [batch_size, max_node_num] 记录每个graph的实际长度，空位记为True
        n_graph, n_node = padding_mask.size()[:2]
        node_pos = torch.where(padding_mask == False)
        x = x.permute(0, 3, 1, 2)
        x_ = self.surf_encoder(x)  # [total_nodes, n_hidden]
        face_area_ = self.face_area_encoder(face_area.unsqueeze(dim=1))  # [total_nodes, n_hidden]
        face_type_ = self.face_type_encoder(face_type)  # [total_nodes, n_hidden]
        face_loop_ = self.face_loop_encoder(face_loop)
        face_degree_ = self.degree_encoder(face_degree)
        node_feature = torch.cat((x_, face_area_, face_type_, face_loop_, face_degree_), dim=-1)

        face_feature = torch.zeros([n_graph, n_node, self.hidden_dim], device=x.device, dtype=x.dtype)
        face_feature[node_pos] = node_feature[:]  # [total_nodes, n_hidden]->[n_graph, max_node_num, n_hidden] 空节点用0.0填充

        # 增加一个全局虚拟节点 [n_graph, 1, n_hidden]
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, face_feature], dim=1)  # [n_graph, max_node_num+1, n_hidden]
        return graph_node_feature, node_feature


class _MLP(nn.Module):
    """"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        MLP with linear output
        Args:
            num_layers (int): The number of linear layers in the MLP
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden feature dimensions for all hidden layers
            output_dim (int): Output feature dimension

        Raises:
            ValueError: If the given number of layers is <1
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # TODO: this could move inside the above loop
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class _EdgeConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        out_feats,
        node_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        """
        This module implements Eq. 2 from the paper where the edge features are
        updated using the node features at the endpoints.

        Args:
            edge_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_EdgeConv, self).__init__()
        self.proj = _MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        src, dst = graph.edges()
        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst])
        agg = proj1 + proj2
        h = self.mlp((1 + self.eps) * efeat + agg)
        h = F.leaky_relu(self.batchnorm(h), inplace= True)
        return h


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """
    def __init__(
            self,
            dim_node,
            num_heads,
            num_spatial,
            num_edge_dis,
            edge_type,
            multi_hop_max_dist,
            n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        # spatial_feature encode
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        # D2 A3 encoder
        # self.d2_pos_encoder = nn.Linear(64, num_heads, bias=False)
        # self.ang_pos_encoder = nn.Linear(64, num_heads, bias=False)
        self.d2_pos_encoder = NonLinear(64, num_heads)
        self.ang_pos_encoder = NonLinear(64, num_heads)

        # edge_feature encode
        self.curv_encoder = CurveEncoder(in_channels=7, output_dims=num_heads)
        self.edge_type_encoder = nn.Embedding(6, num_heads, padding_idx=0)
        self.edge_len_encoder = NonLinear(1, num_heads)
        self.edge_ang_encoder = NonLinear(1, num_heads)
        self.edge_conv_encoder = nn.Embedding(3, num_heads, padding_idx=0)

        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
            self.node_cat = _EdgeConv(
                edge_feats=num_heads,
                out_feats=num_heads,
                node_feats=dim_node,
            )

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, attn_bias, spatial_pos, d2_distance, ang_distance, edge_data, edge_type, edge_len, edge_ang, edge_conv, edge_path, edge_padding_mask, graph, node_feat):
        n_graph, n_node = edge_path.size()[:2]

        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # [n_graph, n_head, n_node+1, n_node+1] 描述每一头注意力下各节点之间的关系矩阵

        # spatial_pos 空间编码------------------------------------------------------------------------------------------------------------
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos)  # spatial_pos_bias[n_graph, n_node, n_node, n_head]
        spatial_pos_bias = spatial_pos_bias.permute(0, 3, 1, 2)  # spatial_pos_bias[n_graph, n_head, n_node, n_node]
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here 设置全局虚拟节点到其他节点的距离
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        # spatial_pos 空间编码------------------------------------------------------------------------------------------------------------

        # 欧氏空间编码---------------------------------------------------------------------------------------------------------------------
        # 在空间编码中增加面-面之间的D2距离
        d2_distance = d2_distance.reshape(-1, 64)
        d2_pos_bias = self.d2_pos_encoder(d2_distance)  # [n_graph, n_node, n_node, 64] -> [n_graph, n_node, n_node, n_head]
        d2_pos_bias = d2_pos_bias.reshape(n_graph, n_node, n_node, self.num_heads)
        d2_pos_bias = d2_pos_bias.permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + d2_pos_bias

        # 在空间编码中增加面-面之间的角度编码
        ang_distance = ang_distance.reshape(-1, 64)
        ang_pos_bias = self.ang_pos_encoder(ang_distance)  # [n_graph, n_node, n_node, 64] -> [n_graph, n_node, n_node, n_head]
        ang_pos_bias = ang_pos_bias.reshape(n_graph, n_node, n_node, self.num_heads)
        ang_pos_bias = ang_pos_bias.permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + ang_pos_bias
        # 欧氏空间编码---------------------------------------------------------------------------------------------------------------------

        # edge_feature 边编码------------------------------------------------------------------------------------------------------------
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()  # 记录任意两节点之间的距离[batch_size, max_node_num, max_node_num] 自己到自己的距离记为1
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1  空位（可以看作是虚拟节点） 统一为1，自己到自己的距离记为1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)  # 调整后两个直接相连的节点之间距离也是1
            spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)

            # 缩减edge_input
            max_dist = self.multi_hop_max_dist
            edge_pos = torch.where(edge_padding_mask == False)  # edge_padding_mask [batch_size, max_edges_num]

            # 调整维度，进行curv_encode
            edge_data = edge_data.permute(0, 2, 1)
            edge_data_ = self.curv_encoder(edge_data)  # [total_edges, n_head]
            edge_type_ = self.edge_type_encoder(edge_type)
            edge_len_ = self.edge_len_encoder(edge_len.unsqueeze(dim=1))
            edge_ang_ = self.edge_ang_encoder(edge_ang.unsqueeze(dim=1))
            edge_conv_ = self.edge_conv_encoder(edge_conv)
            edge_feat = edge_data_ + edge_type_ + edge_len_ + edge_ang_ + edge_conv_

            # add node_feature to edge_feature
            edge_feat_ = self.node_cat(graph, node_feat, edge_feat)  # [total_edges, n_head]

            # edge_input扩充 [total_edges, n_head]->[n_graph, max_node_num, max_node_num, max_dist, n_head]
            n_edge = edge_padding_mask.size(1)
            edge_feature = torch.zeros([n_graph, (n_edge + 1), edge_feat_.size(-1)], device=edge_data.device, dtype=edge_data.dtype)
            edge_feature[edge_pos] = edge_feat_[:][:]  # edge_feature[n_graph, max_edge_num+1, n_head]

            edge_path = edge_path.reshape(n_graph, n_node * n_node * max_dist)
            dim_0 = torch.arange(n_graph, device=edge_path.device).reshape(n_graph, 1).long()
            edge_bias = edge_feature[dim_0, edge_path]
            edge_bias = edge_bias.reshape(n_graph, n_node, n_node, max_dist, self.num_heads)

            edge_bias = edge_bias.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            # permute转为[max_dist, n_graph, max_node_num, max_node_num, n_head]
            # reshape转为[max_dist, ---, n_head]

            # 乘以edge_dis_encoder系数，边特征权重按距离递减，超出max_dist后减为0
            edge_bias = torch.bmm(
                edge_bias,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            edge_bias = edge_bias.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)  # edge_input[n_graph, max_node_num, max_node_num, max_dist, n_head]
            # 各个edge上的特征求和取均值
            edge_bias = (edge_bias.sum(-2) / (spatial_pos_.float().unsqueeze(-1))) # edge_input[n_graph, max_node_num, max_node_num, n_head]
            edge_bias = edge_bias.permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_bias
        # edge_feature 边编码------------------------------------------------------------------------------------------------------------

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset
        return graph_attn_bias