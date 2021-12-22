import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, state_dim, hypernet_embed, embed_dim):
        super(Attention, self).__init__()
        self.state_dim = state_dim
        self.hypernet_embed = hypernet_embed
        self.embed_dim = embed_dim
        self.w_k = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed // 2))
        self.w_q = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed // 2))
        self.w_v = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed // 2))
        self.w_out = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed // 2))
        self.V = nn.Linear(self.state_dim, self.embed_dim)

    def forward(self, x, state):
        x = x.transpose(1, 2)
        b, t, _ = x.shape
        e = self.hypernet_embed // 2

        wk = torch.abs(self.w_k(state).view(-1, 1, e))
        wv = torch.abs(self.w_v(state).view(-1, 1, e))
        wq = torch.abs(self.w_q(state).view(-1, 1, e))

        keys = torch.bmm(x, wk)
        queries = torch.bmm(x, wq)
        values = torch.bmm(x, wv)

        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / np.sqrt(e)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).reshape(b, t, e)

        wout = torch.abs(self.w_out(state).view(-1, e, 1))
        v = self.V(state).view(-1, t, 1)
        out = F.elu(torch.bmm(out, wout) + v)
        return out.transpose(1, 2)

class myQMixer(nn.Module):
    def __init__(self, args):
        super(myQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.adain_w_1 = nn.Linear(self.state_dim, self.n_agents * self.embed_dim)
            self.adain_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.adain_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.n_agents * self.embed_dim))
            self.adain_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        self.attn = Attention(self.state_dim, self.args.hypernet_embed, self.embed_dim)
        
        self.V = nn.Linear(self.state_dim, 1)

        self.bn1 = nn.BatchNorm1d(self.n_agents)
        self.bn2 = nn.BatchNorm1d(self.embed_dim)
        
    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        y1 = torch.abs(self.adain_w_1(states))
        y2 = self.hyper_b_1(states)
        y1 = y1.view(-1, self.n_agents, self.embed_dim)
        y2 = y2.view(-1, 1, self.embed_dim)
        # Compute first layer
        # agent_qs = self.bn1(agent_qs.transpose(1, 2)).transpose(1, 2)
        hidden = F.elu(torch.bmm(agent_qs, y1) + y2)
        hidden = self.attn(hidden, states)
        # Second layer
        y3 = torch.abs(self.adain_w_final(states))
        y3 = y3.view(-1, self.embed_dim, 1)
        y4 = self.V(states).view(-1, 1, 1)
        # Compute final output
        # hidden = self.bn2(hidden.transpose(1, 2)).transpose(1, 2)
        y = torch.bmm(hidden, y3) + y4
        # y = self.w_final(hidden)
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
