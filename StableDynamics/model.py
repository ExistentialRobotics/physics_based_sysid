import torch
from torch import nn
import torch.nn.functional as F
import math

VERIFY = False
V_SCALE = 0.01

global V_WRAP, SCALE_FX
V_WRAP = False
SCALE_FX = False
SMOOTH_V = 6

class Dynamics(nn.Module):
    def __init__(self, fhat, V, alpha=0.01,no_proj=False):
        super().__init__()
        self.fhat = fhat
        self.V = V
        self.alpha = alpha
        self.no_proj = no_proj

    def forward(self, x):
        fx = self.fhat(x)
        if SCALE_FX:
            fx = fx / fx.norm(p=2, dim=1, keepdim=True).clamp(min=1.0)

        Vx = self.V(x)
        gV = torch.autograd.grad([a for a in Vx], [x], create_graph=True, only_inputs=True)[0]
        if self.no_proj:
          rv = fx
        else:
          rv = fx - gV * (F.relu((gV*fx).sum(dim=1) + self.alpha*Vx[:,0])/(gV**2).sum(dim=1))[:,None]

        if VERIFY:
            # Verify that rv has no positive component along gV.
            # This helps us catch:
            #   (1) numeric error in the symbolic gradient calculation, and
            #   (2) Violation of the Lyapunov function when Euler integration is used.
            verify = (gV * rv).sum(dim=1)
            num_violation = len([v for v in verify if v > 0]) # (1)
            new_V = self.V(x + V_SCALE * rv)
            if (new_V > Vx).any(): # (2)
                err = sorted([v for v in (new_V - Vx).detach().cpu().numpy().ravel() if v > 0], reverse=True)

        return rv

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

    def reset_parameters(self):
        # copying from PyTorch Linear
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=5**0.5)
        for i,b in enumerate(self.bias):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        z = F.linear(x, self.W[0], self.bias[0])
        z = self.act(z)

        for W,b,U in zip(self.W[1:-1], self.bias[1:-1], self.U[:-1]):
            z = F.linear(x, W, b) + F.linear(z, F.softplus(U)) / U.shape[0]
            z = self.act(z)

        return F.linear(x, self.W[-1], self.bias[-1]) + F.linear(z, F.softplus(self.U[-1])) / self.U[-1].shape[0]



class ReHU(nn.Module):
    """ Rectified Huber unit"""
    def __init__(self, d):
        super().__init__()
        self.a = 1/d
        self.b = -d/2

    def forward(self, x):
        return torch.max(torch.clamp(torch.sign(x)*self.a/2*x**2,min=0,max=-self.b),x+self.b)

class MakePSD(nn.Module):
    def __init__(self, f, n, eps=0.01, d=1.0):
        super().__init__()
        self.f = f
        self.zero = torch.nn.Parameter(f(torch.zeros(1,n)), requires_grad=False)
        self.eps = eps
        self.d = d
        self.rehu = ReHU(self.d)

    def forward(self, x):
        smoothed_output = self.rehu(self.f(x) - self.zero)
        quadratic_under = self.eps*(x**2).sum(1,keepdim=True)
        return smoothed_output + quadratic_under

def loss_stable(model, Ypred, Yactual, X, no_proj=False):
    # Force smoothness in V:
    # penalty for new_V being larget than old V:
    Vloss = torch.tensor(0)
    if SMOOTH_V:
        V = model.V
        # Successor to X:

        if no_proj:
          succ_X = (X + V_SCALE * Ypred)
        else:
          succ_X = (X + V_SCALE * Yactual).detach()
        if V_WRAP:
            while torch.any(succ_X < -math.pi):
                succ_X[succ_X < -math.pi] = succ_X[succ_X < -math.pi] + 2 * math.pi
            while torch.any(succ_X >= math.pi):
                succ_X[succ_X >= math.pi] = succ_X[succ_X >= math.pi] - 2 * math.pi
            succ_X.requires_grad_()

        Vloss = (V(succ_X) - V(X)).clamp(min=0).mean()

    l2loss = ((Ypred - Yactual)**2).mean() #L2 loss

    return (l2loss + SMOOTH_V * Vloss, l2loss, Vloss)