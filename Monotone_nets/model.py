# Monotone Neural Network, Codes from paper 'Monotone, Bi-Lipschitz, and Polyak-{\L}ojasiewicz Networks'
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence, Callable

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


def cayley(W: torch.Tensor) -> torch.Tensor:
    cout, cin = W.shape
    if cin > cout:
        return cayley(W.T).T
    U, V = W[:cin, :], W[cin:, :]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)
    A = U - U.T + V.T @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=0)

class MonLipNet(nn.Module):
    def __init__(self,
                 features: int,
                 unit_features: Sequence[int],
                 mu: float = 0.1, #lower bound for monotonicity
                 nu: float = 10., #upper bound for monotonicity
                 act_fn: Callable = None):
        super().__init__()
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        self.mu = mu
        self.nu = nu
        self.units = unit_features
        self.act_fn = act_fn
        self.Fq = nn.Parameter(torch.empty(sum(self.units), features)).to(self.device)
        nn.init.xavier_normal_(self.Fq)
        self.fq = nn.Parameter(torch.empty((1,))).to(self.device)
        nn.init.constant_(self.fq, self.Fq.norm())
        self.by = nn.Parameter(torch.zeros(features)).to(self.device)
        Fr, fr, b = [], [], []
        if act_fn is not None:
            d = []
        nz_1 = 0
        for nz in self.units:
            R = nn.Parameter(torch.empty((nz, nz+nz_1))).to(self.device)
            nn.init.xavier_normal_(R)
            r = nn.Parameter(torch.empty((1,))).to(self.device)
            nn.init.constant_(r, R.norm())
            Fr.append(R)
            fr.append(r)
            b.append(nn.Parameter(torch.zeros(nz)))
            if act_fn is not None:
                d.append(nn.Parameter(torch.zeros(nz)).to(self.device))
            nz_1 = nz
        self.Fr = nn.ParameterList(Fr).to(self.device)
        self.fr = nn.ParameterList(fr).to(self.device)
        self.b = nn.ParameterList(b).to(self.device)
        if act_fn is not None:
            self.d = nn.ParameterList(d)
        # cached weights
        self.Q = None
        self.R = None

    def forward(self, x):
        sqrt_gam = math.sqrt(self.nu - self.mu)
        sqrt_2 = math.sqrt(2.)
        if self.training:
            self.Q, self.R = None, None
            Q = cayley(self.fq * self.Fq / self.Fq.norm()).to(self.device)
            R = [cayley(fr * Fr / Fr.norm()) for Fr, fr in zip(self.Fr, self.fr)]
        else:
            if self.Q is None:
                with torch.no_grad():
                    self.Q = cayley(self.fq * self.Fq / self.Fq.norm())
                    self.R = [cayley(fr * Fr / Fr.norm()) for Fr, fr in zip(self.Fr, self.fr)]
            Q, R = self.Q, self.R

        xh = sqrt_gam * x.to(self.device) @ Q.T
        yh = []
        hk_1 = xh[..., :0]
        idx = 0
        for k, nz in enumerate(self.units):
            xk = xh[..., idx:idx+nz]
            gh = torch.cat((xk, hk_1), dim=-1).to(self.device)
            if self.act_fn is None:
                gh = sqrt_2 * F.relu(sqrt_2 * gh @ R[k].T + self.b[k]) @ R[k]
            else:
                gh = sqrt_2 * (gh @ R[k].T) + self.b[k]
                gh = self.act_fn(gh * torch.exp(self.d[k])) * torch.exp(-self.d[k])
                gh = sqrt_2 * gh @ R[k]
            hk = gh[..., :nz] - xk
            gk = gh[..., nz:]
            yh.append(hk_1-gk)
            idx += nz
            hk_1 = hk
        yh.append(hk_1)

        yh = torch.cat(yh, dim=-1).to(self.device)
        y = 0.5 * ((self.mu + self.nu) * x.to(self.device) + sqrt_gam * yh @ Q) + self.by

        return y
    
## Monotone net using ICNN, based on https://github.com/JieFeng-cse/Online-Event-Triggered-Switching-for-Frequency-Control
# ICNN
class ReHU(nn.Module):
    """ Rectified Huber unit"""
    def __init__(self, d):
        super().__init__()
        self.a = 1/d
        self.b = -d/2

    def forward(self, x):
        return torch.max(torch.clamp(torch.sign(x)*self.a/2*x**2,min=0,max=-self.b),x+self.b)


class ICNN(nn.Module):
    def __init__(self, layer_sizes, activation=F.relu_):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0]))
                                   for l in layer_sizes[1:]])
        self.U = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1], layer_sizes[i]))
                                   for i in range(1,len(layer_sizes)-1)])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(l)) for l in layer_sizes[1:]])
        self.act = activation
        self.reset_parameters()
        self.rehu = ReHU(1.0)
        self.act2 = torch.nn.Tanh()
        self.beta = nn.Parameter(torch.tensor(5.0), requires_grad=False)
        self.act_sp = torch.nn.Softplus()

    def reset_parameters(self):
        # copying from PyTorch Linear
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=0.1**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=0.1**0.5)
        for i,b in enumerate(self.bias):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        beta = torch.abs(self.beta)
        z = F.linear(x, self.W[0], self.bias[0])
        z = self.act_sp(z*beta)/beta

        for W,b,U in zip(self.W[1:-1], self.bias[1:-1], self.U[:-1]):
            z = F.linear(x, W, b) + F.linear(z, F.softplus(U)) / U.shape[0]
            z = self.act_sp(z*beta)/beta

        V_res = F.linear(x, self.W[-1], self.bias[-1]) + F.linear(z, F.softplus(self.U[-1])) / self.U[-1].shape[0]
        return V_res * 0.1 #here 0.1 is only for scalaring purpose

class Mono_ICNN_grad(nn.Module):
    def __init__(self, obs_dim, hidden_dim,distribued=False):
        super(Mono_ICNN_grad, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")

        self.rehu = ReHU(float(7.0))
        self.icnn = ICNN([obs_dim, hidden_dim, hidden_dim, 1], activation=torch.nn.Softplus())
        self.distributed = distribued

    def forward(self, state):
        output = self.icnn(state)
        compute_batch_jacobian = torch.vmap(torch.func.jacrev(self.icnn))
        y = compute_batch_jacobian(state).squeeze()
        return y

def init_NNs(input_dim=12):
    features = input_dim
    units = [100, 100]
    mu, nu = 0.001, 10
    # FTN feed-through neural network
    FTN_net = MonLipNet(features, units, mu, nu).to(device)
    FTN_optimizer = torch.optim.Adam(FTN_net.parameters(), lr=2e-3)

    # define the monotone neural network with gradient of SCNN
    hidden_dim = 100
    cvx_net = Mono_ICNN_grad(features, hidden_dim).to(device)

    # params_to_optimize = [param for g in policy_net.g_list for param in g.parameters()]
    cvx_optimizer = torch.optim.Adam(cvx_net.parameters(), lr=1e-3)
    # Loss function
    criterion = nn.MSELoss()

    simple_NN = MLP(features, 100, features).to(device)
    NN_optimizer = torch.optim.Adam(simple_NN.parameters(), lr=1e-3)
    return FTN_net, FTN_optimizer, cvx_net, cvx_optimizer, simple_NN, NN_optimizer, criterion