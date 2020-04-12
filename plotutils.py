import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
from ipywidgets import interactive, FloatSlider, fixed, Text, BoundedIntText
from ipywidgets import Layout, Dropdown
from outbreak_modelling import *
from datautils import dt_to_number, number_to_dt

def sims_to_longform(sims):
    """
    Convert one or more simulations to long-form format for plotting
    with seaborn. The input `sims` should take the form of a dict:
    keys should be strings labelling the simulations; values should
    be the sim DataFrames themselves.
    """
    result = pd.concat({k: v.rename_axis('PROJECTION', axis=1).stack().rename('Value')
                        for k, v in sims.items()})
    result.index.rename('SIMULATION NAME', level=0, inplace=True)
    result = result.to_frame().reset_index()
    result['PROJECTION'] = result['PROJECTION'].astype('category')
    return result

def pivot_plot_data(plot_data):
    plot_data = plot_data.set_index(['Date', 'SIMULATION NAME', 'PROJECTION'])
    plot_data = plot_data.squeeze().rename(None)
    plot_data = plot_data.unstack(['SIMULATION NAME', 'PROJECTION'])
    return plot_data

def plot_simulations(sims, observations, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8))
    plt.sca(ax)
    plot_data = sims_to_longform(sims)
    plot_what = ['All cases', 'All deaths', 'Daily new cases', 'Daily deaths']
    sns.lineplot(data=plot_data,
                 x='Date', y='Value', hue='PROJECTION',
                 style='SIMULATION NAME',
                 hue_order=plot_what,
                 dashes = ['', (2, 4)])
    ax.plot([], [], ' ', label='OBSERVATIONS')
    ax.set_prop_cycle(None)
    observations = observations.set_index('Date')
    observations[plot_what].plot(ax=ax, marker='o', ls='',
                                        markersize=2)
    ax.set_yscale('log')
    ax.set_ylim(1, None)
    ax.yaxis.set_major_locator(ticker.LogLocator(10., (1.,), 15, 15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(10., range(10), 15, 15))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid(which='minor', axis='y', alpha=0.2)
    ax.legend().set_title('')
    ax.legend()
    ax.set_ylabel('')
    ax.set_xlabel('Date')
    #plt.xticks(rotation=90)
    return pivot_plot_data(plot_data)

def explore_simulation(initial_growth_rate,
		       serial_interval,
		       latent_fraction,
		       cdr, cfr,
		       T_detect, T_recover, T_death,
		       cv_detect, cv_recover, cv_death,
		       R_0_lockdown,
                       lockdown_release_date,
                       lockdown_release_timeframe_weeks,
                       sim_time_weeks,
                       weights,
                       observations,
                       day_zero):
    initial_growth_rate /= 100
    cdr /= 100
    cfr /= 100
    cv_detect /= 100
    cv_recover /= 100
    cv_death /= 100
    
    try:
        lockdown_release_date = pd.to_datetime(lockdown_release_date)
    except (TypeError, ValueError) as e:
        print('Error understanding lockdown release date:\n')
        print(e)
        return
    lockdown_release_end = (lockdown_release_date +
                            pd.to_timedelta(7*lockdown_release_timeframe_weeks,
                                            'D'))
    if lockdown_release_date < pd.to_datetime('20/04/01'):
        print('Lockdown cannot be released before April 2020')
        return

    SEIRObs_parameters = SEIRModel.calibrate_parameters(initial_growth_rate,
                                                        serial_interval,
                                                        latent_fraction)
    SEIRObs_parameters.update({'cdr': cdr, 'cfr': cfr,
                               'cv_detect': cv_detect, 'T_detect': T_detect,
                               'cv_recover': cv_recover, 'T_recover': T_recover,
                               'cv_death': cv_death, 'T_death': T_death,
                               'initial_state': SEIR.make_state(S=6.64e7, I=1),
                               'T_start':0})
    ld_model = SEIRObsModel(**SEIRObs_parameters)
    R_0_ld = pw_linear_fn(dt_to_number(pd.to_datetime(['2020/03/10',
                                                       '2020/03/26',
                                                       lockdown_release_date,
                                                       lockdown_release_end]),
                                       day_zero),
                          [ld_model.R_0(0), R_0_lockdown, R_0_lockdown,
                           ld_model.R_0(0)])
    ld_model.R_0 = R_0_ld
    ld_model.fit(observations['Daily new cases'], None,
                 observations['Daily deaths'], weights=weights)
    sim_ld = pd.DataFrame(ld_model.simulate(7*sim_time_weeks)).set_index('t')
    sim_ld.index = number_to_dt(sim_ld.index, day_zero).rename('Date')
    _, (axt, axb) = plt.subplots(2, 1, figsize=(12, 16),
                                 gridspec_kw={'height_ratios': [1, 3]})
    plt.sca(axt)
    plt.plot(sim_ld.index, R_0_ld(dt_to_number(sim_ld.index, day_zero)))
    plt.ylim(0, None)
    plt.ylabel('$R_0(t)$')
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.title('$R_0$ profile')
    plot_data = plot_simulations({'UK lockdown': sim_ld},
                                 observations,
                                 ax=axb)
    axb.set_title('Projections')
    for ax in (axt, axb):
        ax.axvspan(pd.to_datetime('2020/03/10'),
                   pd.to_datetime('2020/03/26'), color='orange', alpha=0.03)
        ax.axvspan(pd.to_datetime('2020/03/26'),
                   lockdown_release_date, color='r', alpha=0.04)
        ax.axvspan(lockdown_release_date, lockdown_release_end, color='g', alpha=0.05)
    plt.subplots_adjust(hspace=0.5)
    plot_data.to_csv('simdata/last-simulation.csv')
    return plot_data

def my_slider(value, mymin, mymax, step, description):
    return FloatSlider(value=value, min=mymin, max=mymax, step=step,
                       description=description,
                       layout=Layout(width='500px'),
                       style={'description_width': 'initial'})

def my_text_box(value, mymin, mymax, step, description):
    return BoundedIntText(value=value, min=mymin, max=mymax, step=step,
                          description=description,
                          style={'description_width': 'initial'})

def interactive_simulation(observations, day_zero):
    return interactive(explore_simulation,
                       {'manual':True},
                       initial_growth_rate = my_slider(26, 5, 50, 1, 'Initial growth rate, %'),
                       serial_interval = my_slider(6.5, 2, 10, 0.5, 'Mean serial interval, days'),
                       latent_fraction = my_slider(0.71, 0.1, 0.9, 0.1, 'Latent period fraction'),
                       cdr = my_slider(4.4, 0.1, 10, 0.1, 'Case detection rate, %'),
                       cfr = my_slider(33, 1, 100, 1, 'Case fatality rate, %'),
                       T_detect = my_slider(11, 1, 30, 1, 'Time to detection, days'),
                       T_recover = my_slider(9, 1, 30, 1, 'Time to recovery, days'),
                       T_death = my_slider(10, 1, 56, 1, 'Time to death, days'),
                       cv_detect = my_slider(33, 1, 99, 1, 'Detection time variability, %'),
                       cv_recover = my_slider(33, 1, 99, 1, 'Recovery time variability, %'),
                       cv_death = my_slider(20, 1, 99, 1, 'Death time variability, %'),
		       R_0_lockdown = my_slider(1.2, 0.1, 4, 0.1, '$R_0$ during lockdown'),
                       lockdown_release_date = Text(value='2020/06/30',
                                                    description='Lockdown release date',
                                                    style={'description_width': 'initial'}),
                       
                       lockdown_release_timeframe_weeks = my_text_box(26, 1, 9999, 1, 'Number of weeks for lockdown release'),
                       sim_time_weeks = my_text_box(52, 1, 999, 1, 'Simulation length, weeks'),
                       weights = Dropdown(options=[('Cases', [1, 0, 0]),
                                                   ('Deaths', [0, 0, 1]),
                                                   ('Cases & deaths',
                                                    [.5, 0, .5])],
                                          description='Fit to ',
                                          style={'description_width': 'initial'}),
                       observations = fixed(observations),
                       day_zero = fixed(day_zero))

