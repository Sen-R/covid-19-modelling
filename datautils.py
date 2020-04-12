import numpy as np
import pandas as pd

def load_data():
    data = {'All cases': 'jhdata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
            'All deaths': 'jhdata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
            'All recovered': 'jhdata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'}
    for k in data:
        data[k] = pd.read_csv(data[k])
    data = pd.concat(data, axis=0)
    data.index.names=['statistic', 'old_index']
    data = data.reset_index().drop('old_index', axis=1)
    return data

def single_country_data(data, country, unrestricted_dates):
    """ unrestricted_dates assumed to be a (start_date, end_date) tuple """
    ts = data[(data['Country/Region']==country) & (data['Province/State'].isnull())]
    ts = ts.set_index('statistic').drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).T
    ts.index.name='Date'
    ts.index = pd.to_datetime(ts.index)
    ts['Active cases'] = (ts['All cases'] - ts['All deaths'] -
                          ts['All recovered'])
    ts['Daily new cases'] = ts['All cases'].diff().fillna(0)
    ts['Daily deaths'] = ts['All deaths'].diff().fillna(0)
    ts['Daily recovered'] = ts['All recovered'].diff().fillna(0)    
    
    ts['phase'] = np.nan
    ts.loc[ts.index[0], 'phase'] = 'isolated'
    ts.loc[unrestricted_dates[0], 'phase'] = 'unrestricted'
    ts.loc[pd.to_datetime(unrestricted_dates[1])+pd.to_timedelta(1, 'D'), 'phase'] = 'suppressed'
    ts['phase'].fillna(method='ffill', inplace=True)
    ts['phase'] = pd.Categorical(ts['phase'])
    return ts

def load_uk_data():
    """ Load data from https://github.com/tomwhite/covid-19-uk-data """
    twd = pd.read_csv('twdata/data/covid-19-indicators-uk.csv')
    twd['Date'] = pd.to_datetime(twd['Date'])
    twd = twd.set_index(['Date', 'Country', 'Indicator']).squeeze().rename(None)
    twd = twd.unstack(['Indicator']).rename_axis(None, axis=1)
    twd.rename({'ConfirmedCases':'All cases',
                'Deaths':'All deaths',
                'Tests':'All tests'},
               axis=1, inplace=True)
    for dk, ak in zip(['Daily new cases', 'Daily deaths', 'Daily tests'],
                      ['All cases', 'All deaths', 'All tests']):
        differ = lambda s: pd.Series(np.diff(s, axis=0, prepend=0),
                                     index=s.index,
                                     name=s.name)
        twd[dk] = twd.groupby(level='Country')[ak].apply(differ)
    twd = twd.reset_index()
    return twd

def dt_to_number(dts, day_zero):
    return (dts - day_zero)/pd.Timedelta(1, 'D')

def number_to_dt(ts, day_zero):
    return day_zero + ts*pd.Timedelta(1, 'D')
