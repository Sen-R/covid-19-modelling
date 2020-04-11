import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.stats import linregress, gamma
from collections import namedtuple

class SEIR:
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

    @property
    def state_labels(self):
        return ['S', 'E', 'I', 'R']

    def to_dict(self):
        """
        Returns dict representation of the state
        """
        return dict([*zip(['S', 'E', 'I', 'R'], self._a)])
    
class SEIRModel:
    def __init__(self,
                 initial_state,
                 R_0=None,
                 T_inc=None,
                 T_inf=None,
                 early_growth_rate=None,
                 mean_generation_time=None,
                 lat_fraction=None):
        #TODO: input validation
        std_parm = ((R_0 is not None) and (T_inc is not None) and
                    (T_inf is not None))
        alt_parm = ((early_growth_rate is not None) and
                    (mean_generation_time is not None) and
                    (lat_fraction is not None))
        if not (alt_parm or std_parm):
            raise ValueError('You have to either provide all of R_0, T_inc '
                             'and T_inf or all of early_growth_rate, '
                             'mean_generation_time and lat_frac')
        if alt_parm:
            if not initial_state.is_scalar:
                raise NotImplementedError('Currently can only parameterise a '
                                          'multi (tensor-valued) state model '
                                          'in terms of an R_0 matrix, T_inc '
                                          'and T_inf')
            params = SEIRModel.calibrate_parameters(early_growth_rate,
                                                    mean_generation_time,
                                                    lat_fraction)
            R_0, T_inc, T_inf = params['R_0'], params['T_inc'], params['T_inf']
        self.R_0 = R_0 if callable(R_0) else (lambda _: R_0)
        self.T_inc = T_inc
        self.T_inf = T_inf
        self.initial_state = initial_state
        self.reset_cache()

    @property
    def R_0(self):
        return self._R_0

    @R_0.setter
    def R_0(self, value):
        self._R_0 = value if callable(value) else (lambda _: value)
        self.reset_cache()

    @property
    def T_inc(self):
        return self._T_inc

    @T_inc.setter
    def T_inc(self, value):
        self._T_inc = value
        self.reset_cache()

    @property
    def T_inf(self):
        return self._T_inf

    @T_inf.setter
    def T_inf(self, value):
        self._T_inf = value
        self.reset_cache()

    def reset_cache(self):
        self._path = None

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value):
        self._state_shape = value.state_shape
        self._initial_state = value
        self.reset_cache()

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

    def predict(self, sim_days, T_start=0):
        """
        Run the simulation if not already cached and return result
        formatted as DataFrame.
        """
        if self._path is None or len(self._path) < (sim_days + 1):
            self.simulate(sim_days)
        path = self._path.reshape((sim_days+1, 4, *self._state_shape))
        result = {'t': self._t[:sim_days+1] - self._t[0] + T_start}
        result.update([*zip(['S', 'E', 'I', 'R'],
                            path[:sim_days+1].swapaxes(0,1))])
        return result
        
    def simulate(self, sim_days):
        """
        Runs the simulation and stores the result. Does not return
        anything.
        """
        sim_days = int(sim_days)
        t_eval = np.linspace(0, sim_days, sim_days+1)
        ivp = solve_ivp(self._ydot, (0, sim_days),
                        self.initial_state.vector, t_eval=t_eval)
        if ivp.status != 0:
            print(ivp.message)
        self._path = ivp.y.T
        self._t = ivp.t
    
    def _path_as_df(self):
        """
        Converts the raw path (stored as a multidimensional numpy array)
        into a DataFrame.
        """
        data = [pd.Series(SEIR.from_vector(s, self._state_shape).to_dict())
                for s in self._path]
        return pd.DataFrame(data, index=self._t,
                            columns=self.initial_state.state_labels)
    
class SEIRObsModel(SEIRModel):
    def __init__(self, f_cdr, f_cfr, cv_detect, T_detect, cv_resolve, T_resolve,
                 cv_death, T_death, start_date=0, **SEIR_params):
        #TODO: input validation
        n_detect, n_resolve, n_death = [cv**(-2) for cv in
                                        (cv_detect, cv_resolve, cv_death)]
        self._fullpath = None
        self.p_symptoms = f_cdr * dg_weights(n_detect,
                                              T_detect/n_detect,
                                              int(T_detect*30))
        self.p_resolve = (1-f_cfr) * dg_weights(n_resolve,
                                                T_resolve/n_resolve,
                                                int(T_resolve*30))
        self.p_death = f_cfr * dg_weights(n_death,
                                          T_death/n_death,
                                          int(T_death*30))
        self.start_date = start_date
        super().__init__(**SEIR_params)

    @property
    def start_date(self):
        return self._start_date

    @start_date.setter
    def start_date(self, start_date):
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        self._start_date = start_date
        self._offset = (pd.Timedelta(1, 'D')
                        if isinstance(self._start_date, pd.Timestamp)
                        else 1.)

    def reset_cache(self):
        self._fullpath = None
        super().reset_cache()
        
    def fit(self, cases, recovered, deaths, sim_days, obs_threshold=10,
            weights=None):
        """
        Fit day 0 of the simulation to observed cases, recovered and 
        deaths. Weights is a 3-tuple specifying the relative weights to
        place on case data, recovery data and death data respectively when
        calculating the score.
        """
        self._fit_score_inner(cases, recovered, deaths, sim_days,
                              obs_threshold, weights, fit=True)
        return self

    def score(self, cases, recovered, deaths, sim_days, obs_threshold=10,
              weights=None):
        return self._fit_score_inner(cases, recovered, deaths, sim_days,
                                     obs_threshold, weights, fit=False)

    def fit_score(self, cases, recovered, deaths, sim_days, obs_threshold=10,
                  weights=None):
        return self._fit_score_inner(cases, recovered, deaths, sim_days,
                                     obs_threshold, weights, fit=True)

    def _fit_score_inner(self, cases, recovered, deaths, sim_days,
                         obs_threshold=10, weights=None, fit=True):
        """
        Fit day 0 of the simulation to observed cases, recovered and 
        deaths. Weights is a 3-tuple specifying the relative weights to
        place on case data, recovery data and death data respectively when
        calculating the score.

        This function returns the score.
        """
        if self._fullpath is None:
            self.predict(sim_days)
            
        if weights is None:
            weights = [1/3, 1/3, 1/3]
        w_c, w_r, w_d = weights

        cases, recovered, deaths = [s.dropna()
                                     for s in (cases, recovered, deaths)]

        total_weight = (len(cases) * w_c + len(recovered) * w_r +
                        len(deaths) * w_d)

        def error_ts(days_shift, s_obs, s_pred):
            if len(s_obs)==0:
                return 0.
            obs_days = (s_obs.index - self.start_date) / self._offset
            s_pred = np.interp(obs_days + days_shift, s_pred.index,
                               s_pred.values)
            return np.sum(np.abs((np.log((obs_threshold+s_obs.values)/
                                         (obs_threshold+s_pred)))))
            
        def score(days_shift):
            result = (w_c * error_ts(days_shift, cases,
                                     self._fullpath['All cases']) +
                      w_r * error_ts(days_shift, recovered,
                                     self._fullpath['All recovered']) +
                      w_d * error_ts(days_shift, deaths,
                                     self._fullpath['All deaths']))
            return result / total_weight

        if not fit:
            return score(0)
        else:
            opt = minimize_scalar(score)
            if not opt.success:
                print(opt)
                raise RuntimeError('Optimiser failed')
            opt_shift = round(opt.x)
            self.start_date -= opt_shift * self._offset
            return opt.fun
    
    def predict(self, sim_days):
        if self._fullpath is not None and len(self._fullpath)>sim_days:
            sim =  self._fullpath.iloc[:sim_days+1]
        else:
            seir_path = super().predict(sim_days)
            sim = pd.Series(seir_path['E'] + seir_path['I'] + seir_path['R'],
                            index=seir_path['t'],
                            name='All exposed').to_frame()
            sim['Daily exposed'] = sim['All exposed'].diff()
            sim['Daily exposed'].iloc[0] = 0.
            sim['Daily cases'] = dp_convolve(sim['Daily exposed'],
                                             self.p_symptoms)
            sim['Daily recovered'] = dp_convolve(sim['Daily cases'],
                                                  self.p_resolve)
            sim['Daily deaths'] = dp_convolve(sim['Daily cases'],
                                              self.p_death)
            sim['All cases'] = sim['Daily cases'].cumsum()
            sim['All recovered'] = sim['Daily recovered'].cumsum()
            sim['All deaths'] = sim['Daily deaths'].cumsum()
            sim['Active cases'] = (sim['All cases'] -
                                   sim['All recovered'] -
                                   sim['All deaths'])
            self._fullpath = sim
        result = sim.copy()
        if isinstance(self.start_date, pd.Timestamp):
            result.index = self.start_date + pd.TimedeltaIndex(sim.index, 'D')
        else:
            result.index = self.start_date + sim.index.values
        result.index.name = 'Date'
        return result


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

def piecewise_linear_R_0_profile(dates, R_0s, baseline_simulation):
    dates = (pd.to_datetime(dates) - baseline_simulation.index[0]).days.values
    return lambda t: np.interp(t, dates, R_0s)
