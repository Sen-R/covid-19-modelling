import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import gamma
from math import ceil

class SEIR:
    state_labels = ['S', 'E', 'I', 'R']
    
    def __init__(self, a):
        self._a = a
        self.state_shape = a.shape[1:]

    @staticmethod
    def make_state(S, E=None, I=None, R=None):
        if E is None:
            E = np.zeros_like(S)
        if I is None:
            I = np.zeros_like(S)
        if R is None:
            R = np.zeros_like(S)
        a = np.stack((S, E, I, R))
        return SEIR(a)

    @staticmethod
    def from_vector(v, state_shape):
        return SEIR(v.reshape(4, *state_shape))

    def __repr__(self):
        return "SEIR(S={}, E={}, I={}, R={})".format(repr(self.S), repr(self.E),
                                                     repr(self.I), repr(self.R))
    
    @property
    def S(self):
        return self._a[0]

    @property
    def E(self):
        return self._a[1]

    @property
    def I(self):
        return self._a[2]

    @property
    def R(self):
        return self._a[3]

    @property
    def vector(self):
        return self._a.ravel()

    @property
    def is_scalar(self):
        return (np.ndim(self._a[0])==0)

    def to_dict(self):
        """
        Returns dict representation of the state
        """
        return dict(zip(self.state_labels, self._a))
    
class SEIRModel:
    params_names = ['R_0', 'T_inc', 'T_inf', 'initial_state', 'T_start']

    def __init__(self, R_0, T_inc, T_inf, initial_state, T_start = 0):
        params = locals()
        self.set_params({n:params[n] for n in self.params_names})
            
    def get_params(self):
        return {n: getattr(self, n) for n in self.params_names}
                
    def set_params(self, params_dict):
        for n, v in params_dict.items():
            if n in self.params_names:
                setattr(self, n, v)

    @property
    def T_start(self):
        return self._T_start

    @T_start.setter
    def T_start(self, value):
        self._T_start = value
            
    @property
    def R_0(self):
        return self._R_0

    @R_0.setter
    def R_0(self, value):
        self._R_0 = value if callable(value) else (lambda _: value)

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value):
        self._state_shape = value.state_shape
        self._initial_state = value

    def _ydot(self, t, y):
        y = SEIR.from_vector(y, self._state_shape)
        N = y.S + y.E + y.I + y.R
        lambda_ = np.tensordot(self.R_0(t), y.I / (N * self.T_inf),
                               len(y.state_shape))
        Sd = - lambda_ * y.S
        Ed = lambda_ * y.S - y.E / self.T_inc
        Id = y.E / self.T_inc - y.I / self.T_inf
        Rd = y.I / self.T_inf
        return SEIR.make_state(S=Sd, E=Ed, I=Id, R=Rd).vector

    def simulate(self, T_end):
        """
        Runs the simulation and returns the simulated trajectory
        """
        if T_end <= self.T_start:
            raise ValueError('T_end should be after T_start')
        T_span = ceil(T_end - self.T_start)
        T_end = T_span + self.T_start
        t_eval = np.linspace(self.T_start, T_end, T_span+1)
        ivp = solve_ivp(self._ydot, (self.T_start, T_end),
                        self.initial_state.vector, t_eval=t_eval)
        if ivp.status != 0:
            print(ivp.message)
        path = ivp.y.T.reshape((T_span + 1, 4, *self._state_shape))
        result = {'t': ivp.t}
        result.update(zip(['S', 'E', 'I', 'R'], path.swapaxes(0,1)))
        return result

    def predict(self, ts, which=None):
        """
        Runs the simulation and then provides estimates for the times
        provided. NB times are assumed to be ordered.

        By default provides predictions for all simulated variables. You
        can provide a list of variables you are interested in by setting
        the argument `which`.
        """
        if len(ts)<1:
            raise ValueError('No times provided')
        sim = self.simulate(ts[-1])
        if which is None:
            which = sim.keys()
        which = ['t'] + which
        return {var: np.interp(ts, sim['t'], sim[var]) for var in which}
        
    @staticmethod
    def calibrate_parameters(early_growth_rate,
                             mean_generation_time,
                             lat_fraction):
        """
        Calculate SEIR parameters R_0, T_inc and T_inf calibrated to
        the specified initial daily case/death growth rate, the infection
        mean generation time, and the fraction of this time that the
        infection is assumed to be latent (i.e. non-infectious, E stage).
        """
        T_inc = lat_fraction * mean_generation_time
        T_inf = mean_generation_time - T_inc
        pos_eigenvalue = np.log(1+early_growth_rate)
        R_0 = (1 + pos_eigenvalue * T_inc) * (1 + pos_eigenvalue * T_inf)
        return {'R_0':R_0, 'T_inc':T_inc, 'T_inf':T_inf}
         
    
class SEIRObsModel(SEIRModel):
    def __init__(self, R_0, T_inc, T_inf, initial_state, cdr, cfr,
                 cv_detect, T_detect, cv_recover, T_recover,
                 cv_death, T_death, T_start=0):
        self._obs_params = {}
        params = locals()
        self.set_params({n:params[n] for n in (self.params_names +
                                               self.obs_params_names)})

    obs_params_names = ['cdr', 'cfr', 'cv_detect', 'T_detect', 'cv_recover',
                        'T_recover', 'cv_death', 'T_death']

    def get_params(self):
        return {n: getattr(self, n) for n in (self.params_names +
                                              self.obs_params_names)}

    def set_params(self, params_dict):
        super().set_params(params_dict)
        if len(self._state_shape) > 0:
            raise NotImplementedError('SEIRObsModel does not yet support '
                                      'multidimensional states')
        for n, v in params_dict.items():
            if n in self.obs_params_names:
                setattr(self, n, v)
                
        n_detect, n_recover, n_death = [cv**(-2) for cv in
                                        (self.cv_detect,
                                         self.cv_recover,
                                         self.cv_death)]
        self.p_detect = self.cdr * dg_weights(n_detect,
                                              self.T_detect/n_detect,
                                              int(self.T_detect*30))
        
        self.p_recover = (1-self.cfr) * dg_weights(n_recover,
                                                   self.T_recover/n_recover,
                                                   int(self.T_recover*30))
        
        self.p_death = self.cfr * dg_weights(n_death,
                                             self.T_death/n_death,
                                             int(self.T_death*30))
        
    def simulate(self, T_end):
        sim = super().simulate(T_end)
        sim['All exposed'] = sim['E'] + sim['I'] + sim['R']
        sim['Daily exposed'] = np.diff(sim['All exposed'], axis=0, prepend=0)
        sim['Daily new cases'] = dp_convolve(sim['Daily exposed'],
                                             self.p_detect)
        sim['Daily recovered'] = dp_convolve(sim['Daily new cases'],
                                             self.p_recover)
        sim['Daily deaths'] = dp_convolve(sim['Daily new cases'],
                                          self.p_death)
        sim['All cases'] = np.cumsum(sim['Daily new cases'], axis=0)
        sim['All recovered'] = np.cumsum(sim['Daily recovered'], axis=0)
        sim['All deaths'] = np.cumsum(sim['Daily deaths'], axis=0)
        sim['Active cases'] = (sim['All cases'] -
                               sim['All recovered'] -
                               sim['All deaths'])
        return sim
        
    def score(self, cases, recovered, deaths, obs_threshold=10, weights=None):
        """
        Each of cases, recovered and deaths is assumed to be a length-2
        sequence: the first element is an array of times and the second
        element is an array of observations.
        """
        return self._score(self._series_to_tuple_if_applicable(cases),
                           self._series_to_tuple_if_applicable(recovered),
                           self._series_to_tuple_if_applicable(deaths),
                           obs_threshold, weights)

    @staticmethod
    def _series_to_tuple_if_applicable(s):
        if s is None:
            return ([], [])
        elif hasattr(s, 'values') and hasattr(s, 'index'):
            return s.index, s.values
        elif len(s)==2:
            return s
        else:
            raise ValueError('Unrecognised format')
    
    def _score(self, cases, recovered, deaths, obs_threshold=10,
               weights=None):
        if weights is None:
            weights = [1/3, 1/3, 1/3]
        t_max = max(s[0][-1] for s in (cases, recovered, deaths)
                    if len(s[0])>0)
        sim = self.simulate(t_max)
        score = 0
        w_total = 0
        for w, k, obs in zip(weights,
                             ['Daily new cases', 'Daily recovered',
                              'Daily deaths'],
                             [cases, recovered, deaths]):
            pred = np.interp(obs[0], sim['t'], sim[k])
            score += w * log_error(obs[1], pred, obs_threshold)
            w_total += w * len(obs[0])
        return score / w_total

    def fit_score(self, cases, recovered, deaths, params_to_vary=None,
                  obs_threshold=10, weights=None):
        """
        params_to_vary should be a dict with keys that are the parameters
        to vary and values that are (min, max) bounds.
        """
        cases = self._series_to_tuple_if_applicable(cases)
        recovered = self._series_to_tuple_if_applicable(recovered)
        deaths = self._series_to_tuple_if_applicable(deaths)
        
        if params_to_vary is None:
            params_to_vary = {'T_start':(-30, 30)}

        params_names = list(params_to_vary.keys())
        params_bounds = list(params_to_vary.values())
        def optimisee(params_values):
            self.set_params(dict(zip(params_names, params_values)))
            return self._score(cases, recovered, deaths, obs_threshold, weights)
        
        current_params = self.get_params()
        params_0 = [current_params[n] for n in params_names]
        opt = minimize(optimisee, params_0, bounds=params_bounds)
        if not opt.success:
            print(opt.message)
        self.set_params(dict(zip(params_names, opt.x)))
        return opt.fun

    def fit(self, cases, recovered, deaths, params_to_vary=None,
            obs_threshold=10, weights=None):
        self.fit_score(cases, recovered, deaths, params_to_vary,
                       obs_threshold, weights)
        return self

def dg_weights(n, mean, n_days):
    """ Discretise a gamma distribution over n_days timesteps
    and return the weights """
    dist = gamma(n, scale=mean)
    edges = np.linspace(-0.5, n_days-0.5, n_days+1)
    p = np.diff(dist.cdf(edges))
    p /= np.sum(p)
    return p

def dp_convolve(signal, p):
    """ Convolve kernel p over a signal forward in time, returning
    a vector of equal length to signal """
    return np.convolve(signal, p, mode='full')[:len(signal)]

def log_error(y_obs, y_pred, obs_threshold):
    if len(y_obs)==0:
        return 0.
    return np.sum(np.abs(np.log((obs_threshold + y_pred)/
                                (obs_threshold + y_obs))))
