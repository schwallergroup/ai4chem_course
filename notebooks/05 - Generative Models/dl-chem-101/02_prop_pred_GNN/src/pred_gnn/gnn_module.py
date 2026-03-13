""" gnn_module.py

Hold basic GNN Types:
1. GGNN
2. NNConv

These classes should accept graphs and return featurizations at each node

The calling class should be responsible for pooling however is best
"""
import numpy as np

import torch
import torch.nn as nn

import dgl.function as fn
from dgllife.model import MPNNGNN
from dgl.backend import pytorch as dgl_F


class MoleculeGNN(nn.Module):
    """MoleculeGNN Module"""

    def __init__(
            self,
            hidden_size: int,
            num_step_message_passing: int = 4,
            gnn_node_feats: int = 74,
            gnn_edge_feats: int = 4,  # 12,
            mpnn_type: str = "NNConv",
            node_feat_symbol="h",
            set_transform_layers: int = 2,
            **kwargs):
        """__init__.
        Args:
            hidden_size (int): Hidden size
            num_mol_layers (int): Number of layers to encode for the molecule
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.gnn_edge_feats = gnn_edge_feats
        self.gnn_node_feats = gnn_node_feats
        self.node_feat_symbol = node_feat_symbol

        self.mpnn_type = mpnn_type
        node_out_feats = self.hidden_size
        if self.mpnn_type == "NNConv":
            # https://github.com/awslabs/dgl-lifesci/blob/master/examples/property_prediction/csv_data_configuration/
            # Note: Edge hidden feats seems way too bloated in this model which
            # is driving params way up when node_out_feats is increased
            self.gnn = MPNNGNN(
                node_in_feats=self.gnn_node_feats,
                edge_in_feats=self.gnn_edge_feats,
                node_out_feats=self.hidden_size,
                edge_hidden_feats=self.hidden_size // 4,  #32,
                num_step_message_passing=num_step_message_passing,
            )
        elif self.mpnn_type == "GGNN":
            self.gnn = GGNN(
                hidden_size=self.hidden_size,
                edge_feats=self.gnn_edge_feats,
                node_feats=self.gnn_node_feats,
                num_step_message_passing=num_step_message_passing,
            )
        else:
            raise ValueError()

        # Keeping d_head only to 2x increase in size to avoid memory. Orig
        # transformer uses 4x
        self.set_transformer = SetTransformerEncoder(
            d_model=node_out_feats,
            n_heads=4,
            d_head=node_out_feats // 4,
            d_ff=hidden_size,
            n_layers=set_transform_layers,
        )

    def forward(self, g):
        """encode batch of molecule graph"""
        node_feats = g.ndata[self.node_feat_symbol]
        device = g.device
        if self.mpnn_type == "NNConv":
            edge_feats = g.edata.get(
                "e", torch.empty(0, self.gnn_edge_feats, device=device))
            output = self.gnn(g, node_feats, edge_feats)
        elif self.mpnn_type == "GGNN":
            # Get graph output
            output = self.gnn(g, self.node_feat_symbol, "e")
        else:
            raise NotImplementedError()

        output = self.set_transformer(g, output)
        return output


class GGNN(nn.Module):

    def __init__(self,
                 hidden_size=64,
                 edge_feats=4,
                 node_feats=74,
                 num_step_message_passing=4,
                 **kwargs):
        """GGNN.

        Define a gated graph neural network

        This is very similar to the NNConv models.

        Args:
            input_size (int): Size of edge features into the graph
            hidden_size (int): Hidden size
            edge_feats (int): Number of edge feats. Must be onehot!
            node_feats (int): Num of node feats (default 74)
            num_step_message_passing (int): Number of message passing steps
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_step_message_passing = num_step_message_passing
        self.edge_feats = edge_feats
        self.node_feats = node_feats
        self.input_project = nn.Linear(self.node_feats, self.hidden_size)

        # Define LSTM Cell
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.edge_transform_weights = torch.nn.Parameter(
            torch.randn(self.edge_feats, self.hidden_size, self.hidden_size))
        self.edge_transform_bias = torch.nn.Parameter(
            torch.randn(self.edge_feats, 1))

        ### Compute Output transformation
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )
        self.tan_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        )

    def message_pass(self, edges):
        """Pass messages along the edges"""
        src_feat = edges.src["_h"]

        # Linear layer
        messages = (torch.einsum("nio,ni->no", edges.data["w"], src_feat) +
                    edges.data["b"])
        messages = nn.functional.relu(messages)
        return {"m": messages}

    def forward(self, graph, nfeat_name="h", efeat_name="e"):
        """forward.

        Args:
            graph (dgl graph): Graph object
            nfeat_name (str): Name of node feat data
            efeat_name (str): Name of e feat

        Return:
            h: Hidden state at each node

        """
        ndata = graph.ndata[nfeat_name]
        with graph.local_scope():

            # Set initial hidden
            h_init = self.input_project(ndata)
            graph.ndata.update({"_h": h_init})

            # Make these 1 dimensional, assuming they were onehot
            efeats = graph.edata[efeat_name].argmax(1)
            for layer in range(self.num_step_message_passing):
                h_in = graph.ndata["_h"]
                edge_transforms = self.edge_transform_weights[efeats]
                edge_biases = self.edge_transform_bias[efeats]
                graph.edata.update({"w": edge_transforms, "b": edge_biases})
                graph.update_all(self.message_pass, fn.sum("m", "_h"))
                h_out = graph.ndata["_h"]
                h_out = self.gru(h_out, h_in)
                graph.ndata.update({"_h": h_out})

            # Read out at the end
            # Because we have other layers, don't get this
            node_level_output = torch.cat([h_init, h_out], -1)
            # gate_weight = self.gate_mlp(h_out)
            # output_val = gate_weight * self.tan_mlp(node_level_output)
            output_val = self.tan_mlp(node_level_output)
        return output_val


# DGL Models
# https://docs.dgl.ai/en/0.6.x/_modules/dgl/nn/pytorch/glob.html#SetTransformerDecoder


class MultiHeadAttention(nn.Module):
    r"""Multi-Head Attention block, used in Transformer, Set Transformer and so on.

    Parameters
    ----------
    d_model : int
        The feature size (input and output) in Multi-Head Attention layer.
    num_heads : int
        The number of heads.
    d_head : int
        The hidden size per head.
    d_ff : int
        The inner hidden size in the Feed-Forward Neural Network.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Notes
    -----
    This module was used in SetTransformer layer.
    """

    def __init__(self,
                 d_model,
                 num_heads,
                 d_head,
                 d_ff,
                 dropouth=0.0,
                 dropouta=0.0):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.proj_q = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_k = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_v = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_o = nn.Linear(num_heads * d_head, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropouth),
            nn.Linear(d_ff, d_model),
        )
        self.droph = nn.Dropout(dropouth)
        self.dropa = nn.Dropout(dropouta)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_inter = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def self_attention(self, x, mem, lengths_x, lengths_mem):
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x)
        max_len_mem = max(lengths_mem)
        device = x.device
        lengths_x = torch.tensor(lengths_x, dtype=torch.int64, device=device)
        lengths_mem = torch.tensor(lengths_mem,
                                   dtype=torch.int64,
                                   device=device)

        queries = self.proj_q(x).view(-1, self.num_heads, self.d_head)
        keys = self.proj_k(mem).view(-1, self.num_heads, self.d_head)
        values = self.proj_v(mem).view(-1, self.num_heads, self.d_head)

        # padding to (B, max_len_x/mem, num_heads, d_head)
        queries = dgl_F.pad_packed_tensor(queries, lengths_x, 0)
        keys = dgl_F.pad_packed_tensor(keys, lengths_mem, 0)
        values = dgl_F.pad_packed_tensor(values, lengths_mem, 0)

        # attention score with shape (B, num_heads, max_len_x, max_len_mem)
        e = torch.einsum("bxhd,byhd->bhxy", queries, keys)
        # normalize
        e = e / np.sqrt(self.d_head)

        # generate mask
        mask = _gen_mask(lengths_x, lengths_mem, max_len_x, max_len_mem)
        e = e.masked_fill(mask == 0, -float("inf"))

        # apply softmax
        alpha = torch.softmax(e, dim=-1)
        # the following line addresses the NaN issue, see
        # https://github.com/dmlc/dgl/issues/2657
        alpha = alpha.masked_fill(mask == 0, 0.0)

        # sum of value weighted by alpha
        out = torch.einsum("bhxy,byhd->bxhd", alpha, values)
        # project to output
        out = self.proj_o(out.contiguous().view(batch_size, max_len_x,
                                                self.num_heads * self.d_head))
        # pack tensor
        out = dgl_F.pack_padded_tensor(out, lengths_x)
        return out

    def forward(self, x, mem, lengths_x, lengths_mem):
        """
        Compute multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor used to compute queries.
        mem : torch.Tensor
            The memory tensor used to compute keys and values.
        lengths_x : list
            The array of node numbers, used to segment x.
        lengths_mem : list
            The array of node numbers, used to segment mem.
        """

        ### Following a _pre_ transformer

        # intra norm
        x = x + self.self_attention(self.norm_in(x), mem, lengths_x,
                                    lengths_mem)

        # inter norm
        x = x + self.ffn(self.norm_inter(x))

        ## intra norm
        # x = self.norm_in(x + out)

        ## inter norm
        # x = self.norm_inter(x + self.ffn(x))
        return x


class SetAttentionBlock(nn.Module):
    r"""SAB block introduced in Set-Transformer paper.

    Parameters
    ----------
    d_model : int
        The feature size (input and output) in Multi-Head Attention layer.
    num_heads : int
        The number of heads.
    d_head : int
        The hidden size per head.
    d_ff : int
        The inner hidden size in the Feed-Forward Neural Network.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Notes
    -----
    This module was used in SetTransformer layer.
    """

    def __init__(self,
                 d_model,
                 num_heads,
                 d_head,
                 d_ff,
                 dropouth=0.0,
                 dropouta=0.0):
        super(SetAttentionBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model,
                                      num_heads,
                                      d_head,
                                      d_ff,
                                      dropouth=dropouth,
                                      dropouta=dropouta)

    def forward(self, feat, lengths):
        """
        Compute a Set Attention Block.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature.
        lengths : list
            The array of node numbers, used to segment feat tensor.
        """
        return self.mha(feat, feat, lengths, lengths)


class SetTransformerEncoder(nn.Module):
    r"""

    Description
    -----------
    The Encoder module in `Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks <https://arxiv.org/pdf/1810.00825.pdf>`__.

    Parameters
    ----------
    d_model : int
        The hidden size of the model.
    n_heads : int
        The number of heads.
    d_head : int
        The hidden size of each head.
    d_ff : int
        The kernel size in FFN (Positionwise Feed-Forward Network) layer.
    n_layers : int
        The number of layers.
    block_type : str
        Building block type: 'sab' (Set Attention Block) or 'isab' (Induced
        Set Attention Block).
    m : int or None
        The number of induced vectors in ISAB Block. Set to None if block type
        is 'sab'.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Examples
    --------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import SetTransformerEncoder
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = torch.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = torch.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> set_trans_enc = SetTransformerEncoder(5, 4, 4, 20)  # create a settrans encoder.

    Case 1: Input a single graph

    >>> set_trans_enc(g1, g1_node_feats)
    tensor([[ 0.1262, -1.9081,  0.7287,  0.1678,  0.8854],
            [-0.0634, -1.1996,  0.6955, -0.9230,  1.4904],
            [-0.9972, -0.7924,  0.6907, -0.5221,  1.6211]],
           grad_fn=<NativeLayerNormBackward>)

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = torch.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> set_trans_enc(batch_g, batch_f)
    tensor([[ 0.1262, -1.9081,  0.7287,  0.1678,  0.8854],
            [-0.0634, -1.1996,  0.6955, -0.9230,  1.4904],
            [-0.9972, -0.7924,  0.6907, -0.5221,  1.6211],
            [-0.7973, -1.3203,  0.0634,  0.5237,  1.5306],
            [-0.4497, -1.0920,  0.8470, -0.8030,  1.4977],
            [-0.4940, -1.6045,  0.2363,  0.4885,  1.3737],
            [-0.9840, -1.0913, -0.0099,  0.4653,  1.6199]],
           grad_fn=<NativeLayerNormBackward>)

    See Also
    --------
    SetTransformerDecoder

    Notes
    -----
    SetTransformerEncoder is not a readout layer, the tensor it returned is nodewise
    representation instead out graphwise representation, and the SetTransformerDecoder
    would return a graph readout tensor.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        d_head,
        d_ff,
        n_layers=1,
        block_type="sab",
        m=None,
        dropouth=0.0,
        dropouta=0.0,
    ):
        super(SetTransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.block_type = block_type
        self.m = m
        layers = []
        if block_type == "isab" and m is None:
            raise KeyError(
                "The number of inducing points is not specified in ISAB block."
            )

        for _ in range(n_layers):
            if block_type == "sab":
                layers.append(
                    SetAttentionBlock(
                        d_model,
                        n_heads,
                        d_head,
                        d_ff,
                        dropouth=dropouth,
                        dropouta=dropouta,
                    ))
            elif block_type == "isab":
                # layers.append(
                #    InducedSetAttentionBlock(m, d_model, n_heads, d_head, d_ff,
                #                             dropouth=dropouth, dropouta=dropouta))
                raise NotImplementedError()
            else:
                raise KeyError(
                    "Unrecognized block type {}: we only support sab/isab")

        self.layers = nn.ModuleList(layers)

    def forward(self, graph, feat):
        """
        Compute the Encoder part of Set Transformer.

        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the
            number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(N, D)`.
        """
        lengths = graph.batch_num_nodes()
        for layer in self.layers:
            feat = layer(feat, lengths)
        return feat


def _gen_mask(lengths_x, lengths_y, max_len_x, max_len_y):
    """Generate binary mask array for given x and y input pairs.

    Parameters
    ----------
    lengths_x : Tensor
        The int tensor indicates the segment information of x.
    lengths_y : Tensor
        The int tensor indicates the segment information of y.
    max_len_x : int
        The maximum element in lengths_x.
    max_len_y : int
        The maximum element in lengths_y.

    Returns
    -------
    Tensor
        the mask tensor with shape (batch_size, 1, max_len_x, max_len_y)
    """
    device = lengths_x.device
    # x_mask: (batch_size, max_len_x)
    x_mask = torch.arange(max_len_x,
                          device=device).unsqueeze(0) < lengths_x.unsqueeze(1)
    # y_mask: (batch_size, max_len_y)
    y_mask = torch.arange(max_len_y,
                          device=device).unsqueeze(0) < lengths_y.unsqueeze(1)
    # mask: (batch_size, 1, max_len_x, max_len_y)
    mask = (x_mask.unsqueeze(-1) & y_mask.unsqueeze(-2)).unsqueeze(1)
    return mask


def pad_packed_tensor(input, lengths, value):
    """pad_packed_tensor"""
    old_shape = input.shape
    device = input.device
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
    else:
        lengths = lengths.to(device)
    max_len = (lengths.max()).item()

    batch_size = len(lengths)
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value)

    # Initialize a tensor with an index for every value in the array
    index = torch.ones(len(input), dtype=torch.int64, device=device)

    # Row shifts
    row_shifts = torch.cumsum(max_len - lengths, 0)

    # Calculate shifts for second row, third row... nth row (not the n+1th row)
    # Expand this out to match the shape of all entries after the first row
    row_shifts_expanded = row_shifts[:-1].repeat_interleave(lengths[1:])

    # Add this to the list of inds _after_ the first row
    cumsum_inds = torch.cumsum(index, 0) - 1
    cumsum_inds[lengths[0]:] += row_shifts_expanded
    x[cumsum_inds] = input
    return x.view(batch_size, max_len, *old_shape[1:])
