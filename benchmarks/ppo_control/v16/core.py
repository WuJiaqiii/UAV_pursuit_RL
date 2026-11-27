import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

def gae(rewards,values,next_values,terminated,truncated,gamma=.99,lam=.95):
    """Bootstrap at truncation, cut recursion at either episode boundary."""
    if not all(x.shape==rewards.shape for x in (values,next_values,terminated,truncated)): raise ValueError('shape mismatch')
    adv=torch.zeros_like(rewards); carry=torch.zeros_like(rewards[0]); term=terminated.bool(); done=term|truncated.bool()
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*next_values[t]*(~term[t])-values[t]
        carry=delta+gamma*lam*(~done[t])*carry; adv[t]=carry
    return adv,adv+values

def net(n,out):
    return nn.Sequential(nn.Linear(n,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,out))

class Policy(nn.Module):
    def __init__(self,n,action_dim=1):
        super().__init__(); self.actor=net(n,action_dim); self.critic=net(n,1); self.log_std=nn.Parameter(torch.zeros(action_dim)-.5)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.orthogonal_(m.weight,math.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01); nn.init.orthogonal_(self.critic[-1].weight,1.)
    def dist(self,x): return Normal(self.actor(x),self.log_std.clamp(-5,2).exp())
    def value(self,x): return self.critic(x).squeeze(-1)
    def logp(self,x,z):
        # Stable tanh Jacobian, including near saturation.
        return (self.dist(x).log_prob(z)-2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))).sum(-1)
    def sample(self,x):
        z=self.dist(x).sample(); return z.tanh(),z,self.logp(x,z)
    def forward(self,x): return self.actor(x).tanh()

def clipped_objective(new,old,adv,clip=.2):
    ratio=(new-old).exp(); return -torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean(),ratio

def explained_variance(y,p):
    v=y.var(unbiased=False); return float(1-(y-p).var(unbiased=False)/v) if v>1e-12 else None
