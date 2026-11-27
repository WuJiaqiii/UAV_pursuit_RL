"""Non-flight benchmarks, scalar actions in [-1,1]. No external target or vehicle."""
import numpy as np

class Bench:
    def __init__(self, name='linear', seed=0):
        if name not in ('linear','pendulum','setpoint','reference'): raise ValueError(name)
        self.name=name; self.rng=np.random.default_rng(seed); self.horizon=200 if name=='pendulum' else 100
        self.obs_dim=3 if name=='pendulum' else 4
    def ref(self):
        if self.name=='reference':
            t=self.t*.05; return np.sin(t),np.cos(t)
        return (1.,0.) if self.name=='setpoint' else (0.,0.)
    def obs(self):
        if self.name=='pendulum': return np.array([np.cos(self.x),np.sin(self.x),self.v],np.float32)
        r,rv=self.ref(); return np.array([self.x,self.v,r,rv],np.float32)
    def reset(self, seed=None):
        if seed is not None: self.rng=np.random.default_rng(seed)
        self.t=0; self.x=self.rng.uniform(-np.pi,np.pi) if self.name=='pendulum' else self.rng.uniform(-1,1); self.v=self.rng.uniform(-.5,.5); return self.obs()
    def step(self, action):
        a=float(np.clip(np.asarray(action).reshape(-1)[0],-1,1)); r,rv=self.ref()
        if self.name=='pendulum':
            theta=(self.x+np.pi)%(2*np.pi)-np.pi; error=theta**2+.1*self.v**2; effort=.001*(2*a)**2
            self.v=float(np.clip(self.v+(.05)*(15*np.sin(self.x)+6*a),-8,8)); self.x+=.05*self.v
        else:
            error=(self.x-r)**2+.1*(self.v-rv)**2; effort=.01*a*a
            self.v+=.05*(3*a-.4*self.v); self.x+=.05*self.v
        self.t+=1; return self.obs(),-float(error+effort),False,self.t>=self.horizon,{'error':float(error),'effort':float(effort),'action':a}
    def reference_action(self,obs):
        if self.name=='pendulum': raise ValueError('No analytic pendulum reference controller supplied')
        x,v,r,rv=obs; return np.array([np.clip((2*(r-x)+1.5*(rv-v))/3,-1,1)],np.float32)
