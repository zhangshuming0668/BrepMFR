from typing import Optional, Tuple
import torch
import torch.nn as nn
from fairseq.modules import FairseqDropout, LayerDropModuleList, LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .layers.multihead_attention import MultiheadAttention
from .layers.brep_encoder_layer import GraphEncoderLayer, GraphNodeFeature, GraphAttnBias


def init_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class BrepEncoder(nn.Module):
    def __init__(
        self,
        num_degree: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 4,
        embedding_dim: int = 128,
        ffn_embedding_dim: int = 128,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        apply_params_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ) -> None:
        super().__init__()

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_params_init = apply_params_init
        self.traceable = traceable

        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_degree=num_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.graph_attn_bias = GraphAttnBias(
            dim_node=embedding_dim,
            num_heads=num_attention_heads,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            n_layers=num_encoder_layers,
        )

        self.embed_scale = embed_scale
        
        self.tanh = nn.Tanh()

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if pre_layernorm:
            self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_params_init:
            self.apply(init_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def build_graph_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        pre_layernorm,
    ):
        return GraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            pre_layernorm=pre_layernorm,
        )

    def forward(
        self,
        batch_data,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        padding_mask = batch_data["padding_mask"]
        n_graph, n_node = padding_mask.size()[:2]
        
        if token_embeddings is not None:
            x = token_embeddings
        else:
            x, x_0= self.graph_node_feature(batch_data["node_data"],
                                        batch_data["face_area"],
                                        batch_data["face_type"],
                                        batch_data["face_loop"],
                                        batch_data["in_degree"],
                                        batch_data["padding_mask"])  
            # x[batch_size, max_node, embedding_dim] 每个节点均嵌入到长度embedding_dim的向量

        if perturb is not None:
            x[:, 1:, :] += perturb

        attn_bias = self.graph_attn_bias(batch_data["attn_bias"],
                                         batch_data["spatial_pos"],
                                         batch_data["d2_distance"],
                                         batch_data["angle_distance"],
                                         batch_data["edge_data"],
                                         batch_data["edge_type"],
                                         batch_data["edge_len"],
                                         batch_data["edge_ang"],
                                         batch_data["edge_conv"],
                                         batch_data["edge_path"],
                                         batch_data["edge_padding_mask"],
                                         batch_data["graph"],
                                         x_0
                                         )  #attn_bias[n_graph, n_head, max_node_num, max_node_num]

        # compute padding mask. This is needed for multi-head attention
        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        # B x (T+1) 比data_x多了一个节点，这是全局的虚拟节点 padding_mask中等于True的位置标示该位置masked

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)  #类似torch.nn.LayerNorm(embedding_dim)  x [batch_size, max_node, embedding_dim]

        x = self.dropout_module(x)
        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1) # x [max_node, batch_size, embedding_dim]

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            if not last_state_only:
                inner_states.append(x)

        x = self.tanh(x)
        graph_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), graph_rep
        else:
            return inner_states, graph_rep
