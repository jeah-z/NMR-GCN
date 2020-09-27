# -*- coding:utf-8 -*-

import dgl
import torch as th
import torch.nn as nn
from .layers import AtomEmbedding, Interaction, ShiftSoftplus, RBFLayer
from torch.nn import init


class SchNetModel(nn.Module):
    """
    SchNet Model from:
        Schütt, Kristof, et al.
        SchNet: A continuous-filter convolutional neural network
        for modeling quantum interactions. (NIPS'2017)
    """

    def __init__(self,
                 dim=64,
                 cutoff=5.0,
                 output_dim=1,
                 width=1,
                 n_conv=3,
                 norm=False,
                 atom_ref=None,
                 pre_train=None):
        """
        Args:
            dim: dimension of features
            output_dim: dimension of prediction
            cutoff: radius cutoff
            width: width in the RBF function
            n_conv: number of interaction layers
            atom_ref: used as the initial value of atom embeddings,
                      or set to None with random initialization
            norm: normalization
        """
        super().__init__()
        self.name = "SchNet"
        self._dim = dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm
        self.activation = ShiftSoftplus()

        if atom_ref is not None:
            self.e0 = AtomEmbedding(1, pre_train=atom_ref)
        if pre_train is None:
            self.embedding_layer = AtomEmbedding(dim)
        else:
            self.embedding_layer = AtomEmbedding(pre_train=pre_train)
        self.rbf_layer = RBFLayer(0, cutoff, width)
        self.conv_layers = nn.ModuleList(
            [Interaction(self.rbf_layer._fan_out, dim) for i in range(n_conv)])

        self.atom_dense_layer1 = nn.Linear(dim, 64)
        self.atom_dense_layer2 = nn.Linear(64, output_dim)
        self.dense_layer1 = nn.Linear(2, 16)
        self.dense_layer2 = nn.Linear(16, 64)
        self.dense_layer3 = nn.Linear(64, 16)
        self.dense_layer4 = nn.Linear(16, 1)
        self.dense_layer5 = nn.Linear(1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()




    def set_mean_std(self, mean, std, device="cpu"):
        self.mean_per_atom = th.tensor(mean, device=device)
        self.std_per_atom = th.tensor(std, device=device)

    def forward(self, g, mask, qm):
        """g is the DGL.graph"""

        self.embedding_layer(g)
        if self.atom_ref is not None:
            self.e0(g, "e0")
        self.rbf_layer(g)
        for idx in range(self.n_conv):
            self.conv_layers[idx](g)

        atom = self.atom_dense_layer1(g.ndata["node"])
        atom = self.activation(atom)
        res = self.atom_dense_layer2(atom)
        g.ndata["res"] = res

        if self.atom_ref is not None:
            g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]

        if self.norm:
            g.ndata["res"] = g.ndata[
                "res"] * self.std_per_atom + self.mean_per_atom
        # print('res before *mask %s'%(g.ndata["res"]))
        # print('mask =  %s'%(mask.squeeze(0)))
        # res = dgl.sum_nodes(g, "res")
        g.ndata["res"] = g.ndata["res"] * mask.squeeze(0)
        # print('res after *mask %s'%(g.ndata["res"]))
        res = dgl.sum_nodes(g, "res")
        # print('\n\nres =  %s'%(res))
        # print('qm =  %s'%(qm))
        res_qm = th.cat((res, qm), 1)
        # print('res_qm =  %s\n\n'%(res_qm))
        pred = self.relu(self.dense_layer1(res_qm))
        pred = self.relu(self.dense_layer2(pred))
        pred = self.relu(self.dense_layer3(pred))
        pred = self.relu(self.dense_layer4(pred))
        pred = self.dense_layer5(pred)

        # print('res after sum_nodes %s'%(res))
        return pred


if __name__ == "__main__":
    g = dgl.DGLGraph()
    g.add_nodes(2)
    g.add_edges([0, 0, 1, 1], [1, 0, 1, 0])
    g.edata["distance"] = th.tensor([1.0, 3.0, 2.0, 4.0]).reshape(-1, 1)
    g.ndata["node_type"] = th.LongTensor([1, 2])
    model = SchNetModel(dim=1)
    atom = model(g)
    print(atom)
