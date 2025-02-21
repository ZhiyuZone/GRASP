import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
import torch.nn as nn


def get_noise(x, sigma=0.0, k=0, p=0.707):
    n, d = x.size()

    if k == 0:
        noise = torch.normal(0.0, sigma, size=x.size(), device=x.device)

    elif k == 1.5:
        noise = torch.normal(0.0, sigma, size=x.size(), device=x.device)
        noise[torch.rand(n) < p] = torch.normal(0.0, sigma, size=(1, d), device=x.device)
        # noise[torch.rand(n) < p**0.5] = torch.normal(0.0, sigma, size=(1, d), device=x.device)

    elif k > 0:
        noise = torch.normal(0.0, sigma, size=(k, d), device=x.device)
        noise = noise[torch.randint(0, k, (n,), device=x.device)]

        # weight = 1 / torch.sqrt(torch.tensor(k, dtype=torch.float32))
        # noise = torch.normal(0.0, sigma * weight, x.size(), device=x.device)
        # n = x.size(0)
        # indices = torch.arange(n, device=x.device).view(-1, 1)
        # indices = torch.cat([indices, torch.randint(0, n, (n, k-1), device=x.device)], dim=1)
        # noise = torch.sum(noise[indices], dim=1)

    return noise


class MPNNs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3, 
                 dropout=0.5, heads=1, pre_ln=False, pre_linear=False, res=False, ln=False, bn=False, jk=False, gnn='gcn', 
                 defence_sigma=0.0, defence_k=1, defence_p=0.7, defence_limit=False):
        super(MPNNs, self).__init__()

        self.dropout = dropout
        self.pre_ln = pre_ln

        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk
        self.defence_sigma = defence_sigma
        self.defence_k = defence_k
        self.defence_p = defence_p
        self.defence_limit = defence_limit

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)
        
        if not self.pre_linear:
            if gnn=='gat':
                self.local_convs.append(GATConv(in_channels, hidden_channels, heads=heads,
                    concat=True, add_self_loops=False, bias=False))
            elif gnn=='sage':
                self.local_convs.append(SAGEConv(in_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(in_channels, hidden_channels,
                        cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(in_channels))
            local_layers = local_layers - 1
            
        for _ in range(local_layers):
            if gnn=='gat':
                self.local_convs.append(GATConv(hidden_channels, hidden_channels, heads=heads,
                    concat=True, add_self_loops=False, bias=False))
            elif gnn=='sage':
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels,
                        cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hidden_channels))

        # self.projector = torch.nn.Linear(hidden_channels, hidden_channels)
        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

        # self.projector = torch.nn.Linear(hidden_channels, out_channels)
        # self.pred_local = torch.nn.Linear(out_channels, out_channels)


    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        # self.projector.reset_parameters()
        self.pred_local.reset_parameters()


    def forward(self, x, edge_index, get_embedding=False, dense_adj=False):

        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final = 0

        if dense_adj:
            for i, local_conv in enumerate(self.local_convs):
                if self.res:
                    x = edge_index @ local_conv.lin(x) + local_conv.bias + self.lins[i](x)
                else:
                    x = edge_index @ local_conv.lin(x) + local_conv.bias
                # --------------------------------------------------
                if self.defence_sigma > 0.0:
                    if (not self.defence_limit) or (i == len(self.local_convs)-1):
                        x = x + get_noise(x, sigma=self.defence_sigma, k=self.defence_k)
                # --------------------------------------------------
                if self.ln:
                    x = self.lns[i](x)
                elif self.bn:
                    x = self.bns[i](x)
                else:
                    pass
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.jk:
                    x_final = x_final + x
                else:
                    x_final = x

        else:
            for i, local_conv in enumerate(self.local_convs):
                if self.res:
                    x = local_conv(x, edge_index) + self.lins[i](x)
                else:
                    x = local_conv(x, edge_index)
                # --------------------------------------------------
                # if self.defence_sigma > 0.0:
                #     if (not self.defence_limit) or (i == len(self.local_convs)-1):
                #         x = x + get_noise(x, sigma=self.defence_sigma, k=self.defence_k, p=self.defence_p)
                # --------------------------------------------------
                if self.ln:
                    x = self.lns[i](x)
                elif self.bn:
                    x = self.bns[i](x)
                else:
                    pass
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                # --------------------------------------------------
                if self.defence_sigma > 0.0:
                    if (not self.defence_limit) or (i == len(self.local_convs)-1):
                        x = x + get_noise(x, sigma=self.defence_sigma, k=self.defence_k, p=self.defence_p)
                # --------------------------------------------------

                if self.jk:
                    x_final = x_final + x
                else:
                    x_final = x

        # x_final = x_final + self.projector(x_final)

        x = self.pred_local(x_final)

        if get_embedding:
            return x_final, x
        else:
            return x
