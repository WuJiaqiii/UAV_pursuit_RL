"""Independent mathematical and standard non-flight control benchmarks."""
import numpy as np
from v15.envs import Bench

REGISTRY = {
    'linear': ('L0', 'local'), 'pendulum': ('L0', 'local'),
    'setpoint': ('L0', 'local'), 'reference': ('L0', 'local'),
    'linear2': ('L1', 'local'), 'linear4': ('L1', 'local'),
    'linear8': ('L1', 'local'), 'linear16': ('L1', 'local'),
    'coupled4': ('L2', 'local'), 'coupled8': ('L2', 'local'),
    'official_pendulum': ('L3', 'Pendulum-v1'),
    'reacher': ('L3', 'Reacher-v5'), 'doublependulum': ('L3', 'InvertedDoublePendulum-v5'),
    'swimmer': ('L3', 'Swimmer-v5'), 'hopper': ('L3', 'Hopper-v5'),
}

class DependencyBlocked(RuntimeError):
    pass

class LocalScalar(Bench):
    action_dim = 1
    def close(self): pass

class MultiLinear:
    """Dimension-normalized quadratic cost, discrete semi-implicit integration."""
    def __init__(self, dim, seed=0, coupling=0.):
        self.action_dim=dim; self.obs_dim=4*dim; self.horizon=100
        self.rng=np.random.default_rng(seed); self.dt=.05
        self.dim=dim; self.coupling=float(coupling)
        # Symmetric nearest-neighbour coupling; same coordinates and costs as scalar case.
        adj=np.zeros((dim,dim))
        for i in range(dim-1): adj[i,i+1]=adj[i+1,i]=1
        self.K=self.coupling*(np.diag(adj.sum(1))-adj)
        self.B=3*np.eye(dim)+self.coupling*adj
        self.D=.4*np.eye(dim)
        self.A_d=np.block([[np.eye(dim)-self.dt**2*self.K,self.dt*(np.eye(dim)-self.dt*self.D)],[-self.dt*self.K,np.eye(dim)-self.dt*self.D]])
        self.B_d=np.vstack([self.dt**2*self.B,self.dt*self.B])
        self._gain=None
    def obs(self):
        return np.concatenate([self.x,self.v,np.zeros(2*self.dim)]).astype(np.float32)
    def reset(self, seed=None):
        if seed is not None: self.rng=np.random.default_rng(seed)
        self.t=0; self.x=self.rng.uniform(-1,1,self.dim); self.v=self.rng.uniform(-.5,.5,self.dim)
        return self.obs()
    def step(self, action):
        a=np.asarray(action,dtype=float)
        if a.shape!=(self.dim,) or not np.isfinite(a).all(): raise ValueError('invalid action shape or value')
        a=np.clip(a,-1,1); errors=self.x**2+.1*self.v**2; effort=.01*a**2
        self.v+=self.dt*(self.B@a-self.D@self.v-self.K@self.x); self.x+=self.dt*self.v; self.t+=1
        return self.obs(),-float(np.mean(errors+effort)),False,self.t>=self.horizon,{'error':float(errors.mean()),'effort':float(effort.mean()),'action':a.copy(),'error_by_dim':errors.tolist()}
    def reference_action(self, obs):
        from scipy.linalg import solve_discrete_are
        if self._gain is None:
            Q=np.diag(np.r_[np.ones(self.dim),np.full(self.dim,.1)])/self.dim
            R=.01*np.eye(self.dim)/self.dim
            P=solve_discrete_are(self.A_d,self.B_d,Q,R)
            self._gain=np.linalg.solve(R+self.B_d.T@P@self.B_d,self.B_d.T@P@self.A_d)
        return np.clip(-self._gain@np.asarray(obs)[:2*self.dim],-1,1).astype(np.float32)
    def close(self): pass

class Official:
    """Map normalized actions to the native finite Box exactly once."""
    def __init__(self, env_id, seed):
        try:
            import gymnasium as gym
            self.env=gym.make(env_id)
        except (ImportError, ModuleNotFoundError) as e:
            raise DependencyBlocked(f'{env_id}: {e}') from e
        except Exception as e:
            if type(e).__name__ in ('DependencyNotInstalled','NamespaceNotFound','NameNotFound','VersionNotFound'):
                raise DependencyBlocked(f'{env_id}: {e}') from e
            raise
        self.seed=seed; self.first=True
        space=self.env.action_space; obs=self.env.observation_space
        if len(space.shape)!=1 or len(obs.shape)!=1 or not np.isfinite(space.low).all() or not np.isfinite(space.high).all() or np.any(space.high<=space.low):
            self.env.close(); raise ValueError('expected flat finite Box action and flat observation')
        self.low=space.low; self.high=space.high; self.action_dim=space.shape[0]; self.obs_dim=obs.shape[0]
        self.horizon=self.env.spec.max_episode_steps
        if not self.horizon: self.env.close(); raise ValueError('missing official time limit')
    def reset(self, seed=None):
        if seed is None and self.first: seed=self.seed
        self.first=False; o,_=self.env.reset(seed=seed); return np.asarray(o,np.float32)
    def step(self,a):
        a=np.asarray(a,np.float32)
        if a.shape!=(self.action_dim,) or not np.isfinite(a).all(): raise ValueError('invalid action')
        a=np.clip(a,-1,1); physical=self.low+(a+1)*.5*(self.high-self.low)
        o,r,term,trunc,info=self.env.step(physical)
        return np.asarray(o,np.float32),float(r),bool(term),bool(trunc),{'action':a,'error':None,'effort':float(np.mean(a**2))}
    def close(self): self.env.close()

def make_env(name,seed=0):
    if name not in REGISTRY: raise ValueError(name)
    if name in ('linear','pendulum','setpoint','reference'): return LocalScalar(name,seed)
    if name.startswith('linear'): return MultiLinear(int(name[6:]),seed)
    if name.startswith('coupled'): return MultiLinear(int(name[7:]),seed,coupling=.6)
    return Official(REGISTRY[name][1],seed)
