# Data vis for NYT coronavirus data
# data taken from https://github.com/nytimes/covid-19-data/

import sys
import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
import matplotlib
import matplotlib.pyplot as plt

#matplotlib.use('TkAgg')

import matplotlib.dates as mdates
from datetime import date
from datetime import datetime

doplot = True 
dolegend = True

# Today's date

today = mdates.date2num(date.today())

# Load Data from GitHub
url_states = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv' 
url_counties = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'

# Live data
url_us_live = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us.csv'
url_states_live = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-states.csv'
url_counties_live = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-counties.csv'

csv_us_live = read_csv(url_us_live, parse_dates=['date'])
csv_states_live = read_csv(url_states_live, parse_dates=['date'], index_col=['state'])
csv_counties_live = read_csv(url_counties_live, parse_dates=['date'], index_col=['county']) 

# end live data

csv_states = read_csv(url_states, parse_dates=['date'], index_col=['state']) # index data by date
csv_counties = read_csv(url_counties, parse_dates=['date'], index_col=['county']) # index data by county

last_date = csv_states['date'].iloc[-1]
last_date = last_date.to_pydatetime()
deltat = (datetime.now() - last_date).days

csv_country = csv_states.groupby(['date']).sum()
csv_country = csv_country.reset_index()

# Get all US data

states = csv_states.values # parse to np array
counties = csv_counties.values

# Inputs

# Filter by input state
my_state, my_county = sys.argv[1], sys.argv[2]

my_thresh = 0.97**(-1)
if len(sys.argv) > 3:
    my_thresh = float(sys.argv[3])**(-1)

my_state_data = csv_states.filter(like=my_state, axis=0) # filter csv to only selected county/state

my_county_data = csv_counties[csv_counties['state'] == my_state]

my_state_live = csv_states_live.filter(like=my_state, axis=0) # same for live data
my_county_live = csv_counties_live.filter(like=my_county, axis=0)

'''
print(my_state_live)
csv_states_live.at[my_state, 'cases'] -= 100
my_state_live = csv_states_live.filter(like=my_state, axis=0)
print(my_state_live)
'''

# Filter by county
my_county_data = my_county_data.filter(like=my_county, axis=0)

# Analysis
# Find Doubling Time; i.e. fit to 2^t/\tau; find \tau; probably only consider data when cases > 10
# plot best fit line of exponential growth

fit_thresh = 100 #int(sys.argv[3])

def logfit(dataframe, label, thresh=my_thresh): #(0.75)**(-1)): # fits to when population was 75% of current value
    df = dataframe[ dataframe[label] > dataframe[label].max()/thresh]
    x,y = mdates.date2num(df['date'].values), df[label].values

    if x.size == 0 or y.size == 0:
        return 0 # empty data
    else:
        z = np.polyfit(x, np.log(y), 1) # linear fit the log
        pfit = np.poly1d(z)
        xfit, yfit = mdates.num2date(x), np.exp(pfit(x))
        projection = np.exp(pfit(today))

        return xfit, yfit, np.log(2)/z[0], projection # returns doubling time tau; i.e. y = 2^(t/tau) in days

# Also compute first difference; i.e. plot new cases vs cases

def firstDiff(df, label):
    offset = np.array([0])
    data = df[label].values
    return np.append(offset, np.diff(data))

def rollingmean(arr, window=7): # implement rolling average for data series
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def linapprox(df, label, win=7):
    arr = df[label].diff().rolling(window=win).mean()
    m = arr[len(arr) - 1] # last value of rolling mean gets most recent slope
    y0 = df[label].max()
    return y0+(m*deltat)

def expapprox(n, tau):
    return n*2**(1.0/tau)

# Plotting

if doplot:
    # Linear approximation of cases today; get rolling mean of first derivative and scale to today 

    my_state_current_cases = my_state_data['cases'].max()
    my_state_current_deaths = my_state_data['deaths'].max()
    my_county_current_cases = my_county_data['cases'].max()
    my_county_current_deaths = my_county_data['deaths'].max()
    my_country_current_cases = csv_country['cases'].max()
    my_country_current_deaths = csv_country['deaths'].max()

    fig, axs = plt.subplots(2, 3, sharex='col')
    (ax5, ax1, ax2), (ax6, ax3, ax4) = axs

    fig.suptitle('COVID-19 Cases in %s (Data provided by NYT)\n\n' % my_state, fontsize=16)

    # State cases vs t
    xfit, yfit, tau1, p1 = logfit(my_state_data, 'cases')
    ax1.plot(xfit, yfit, '--', color='orange')

    xfit, yfit, tau2, p2 = logfit(my_state_data, 'deaths')
    ax1.plot(xfit, yfit, '--', color='red')

    my_state_data.plot('date', 'cases', kind='scatter', logy=True, grid=True, ax=ax1, color='orange', s=1)
    my_state_data.plot('date', 'deaths', kind='scatter', logy=True, grid=True, ax=ax1, color='red', s=1)

    my_state_live.plot('date', 'cases', kind='scatter', logy=True, grid=True, ax=ax1, color='blue', s=2)
    my_state_live.plot('date', 'deaths', kind='scatter', logy=True, grid=True, ax=ax1, color='blue', s=2)
    
    ax1.set(xlabel='Date', ylabel='Occurences', title='All Cases in %s\n(Est. Cases/Deaths today: %d/%d)' % (my_state, expapprox(my_state_current_cases, tau1), expapprox(my_state_current_deaths, tau2)))#linapprox(my_state_data, 'cases'), linapprox(my_state_data, 'deaths')))
    if dolegend:
        ax1.legend(['$t_d \\approx %.2f$ days' % tau1, '$t_d \\approx %.2f$ days' % tau2], loc='upper left')

    # County cases vs t
    xfit, yfit, tau1, p1 = logfit(my_county_data, 'cases') #logfit(False, True)
    ax2.plot(xfit, yfit, '--', color='orange')
    
    xfit, yfit, tau2, p2 = logfit(my_county_data, 'deaths')
    ax2.plot(xfit, yfit, '--', color='red')

    my_county_data.plot('date', 'cases', kind='scatter', logy=True, grid=True, ax=ax2, color='orange', sharex='col', s=1)
    my_county_data.plot('date', 'deaths', kind='scatter', logy=True, grid=True, ax=ax2, color='red', s=1)

    my_county_live.plot('date', 'cases', kind='scatter', logy=True, grid=True, ax=ax2, color='blue', sharex='col', s=2)
    my_county_live.plot('date', 'deaths', kind='scatter', logy=True, grid=True, ax=ax2, color='blue', sharex='col', s=2)


    ax2.set(xlabel='Date', ylabel='Occurences', title='All Cases in %s, %s\n(Est. Cases/Deaths today: %d/%d)' % (my_county, my_state, expapprox(my_county_current_cases, tau1), expapprox(my_county_current_deaths, tau2)))#linapprox(my_county_data, 'cases'), linapprox(my_county_data, 'deaths')))
    if dolegend:
        ax2.legend(['$t_d \\approx %.2f$ days' % tau1, '$t_d \\approx %.2f$ days' % tau2], loc='upper left')
    plt.setp(ax2.get_xticklabels(), rotation=45)


    # State diff vs cases

    ax3.plot(my_state_data['date'].values, my_state_data['cases'].diff(), '.', color='orange', markersize=2)
    ax3.plot(my_state_data['date'].values, my_state_data['deaths'].diff(), '.', color='red', markersize=2)
    ax3.plot(my_state_data['date'].values, my_state_data['cases'].diff().rolling(window=7).mean(), '-', color='orange', markersize=2)
    ax3.plot(my_state_data['date'].values, my_state_data['deaths'].diff().rolling(window=7).mean(), '-', color='red', markersize=2)
    ax3.grid(True, which='major', axis='both')

    csv_states_live.at[my_state, 'cases'] -= my_state_current_cases
    csv_states_live.at[my_state, 'deaths'] -= my_state_current_deaths
    my_state_live = csv_states_live.filter(like=my_state, axis=0)
    
    my_state_live.plot('date', 'cases', kind='scatter', logy=True, grid=True, ax=ax3, color='blue', s=2)
    my_state_live.plot('date', 'deaths', kind='scatter', logy=True, grid=True, ax=ax3, color='blue', s=2)
    
    ax3.set(xlabel='Date', ylabel='Occurences', title='New Cases in %s\n(Total Cases/Deaths: %d/%d)' % (my_state, my_state_current_cases, my_state_current_deaths), yscale='log')

    plt.setp(ax3.get_xticklabels(), rotation=45)
    #ax3.legend(['New Cases', 'New Deaths'], loc='upper left')


    # County diff vs cases
    ax4.plot(my_county_data['date'].values, my_county_data['cases'].diff(), '.', color='orange', markersize=2)
    ax4.plot(my_county_data['date'].values, my_county_data['deaths'].diff(), '.', color='red', markersize=2)
    ax4.plot(my_county_data['date'].values, my_county_data['cases'].diff().rolling(window=7).mean(), '-', color='orange', markersize=2)
    ax4.plot(my_county_data['date'].values, my_county_data['deaths'].diff().rolling(window=7).mean(), '-', color='red', markersize=2)

    csv_counties_live.at[my_county, 'cases'] -= my_county_current_cases
    csv_counties_live.at[my_county, 'deaths'] -= my_county_current_deaths
    my_county_live = csv_counties_live.filter(like=my_county, axis=0)

    my_county_live.plot('date', 'cases', kind='scatter', logy=True, grid=True, ax=ax4, color='blue', s=2)
    my_county_live.plot('date', 'deaths', kind='scatter', logy=True, grid=True, ax=ax4, color='blue', s=2)

    plt.setp(ax4.get_xticklabels(), rotation=45)
    plt.grid(True, which='major', axis='both')
    ax4.set(xlabel='Date', ylabel='Occurences', title='New Cases in %s, %s\n(Total Cases/Deaths: %d/%d)' % (my_county, my_state, my_county_current_cases, my_county_current_deaths), yscale='log')
    #ax4.legend(['New Cases', 'New Deaths'], loc='upper left')

    # Country Level Data
    xfit, yfit, tau1, p1 = logfit(csv_country, 'cases')
    ax5.plot(xfit, yfit, '--', color='orange')
    xfit, yfit, tau2, p2 = logfit(csv_country, 'deaths')
    ax5.plot(xfit, yfit, '--', color='red')
    csv_country.plot('date', 'cases', kind='scatter', logy=True, grid=True, ax=ax5, color='orange', sharex='col', s=1)
    csv_country.plot('date', 'deaths', kind='scatter', logy=True, grid=True, ax=ax5, color='red', sharex='col', s=1)

    # Country live
    csv_us_live.plot('date', 'cases', kind='scatter', logy=True, grid=True, ax=ax5, color='blue', sharex='col', s=2)
    csv_us_live.plot('date', 'deaths', kind='scatter', logy=True, grid=True, ax=ax5, color='blue', sharex='col', s=2)

    ax5.set(xlabel='Date', ylabel='Occurences', title='All Cases in US ($t_d =$ doubling time)\n(Est. Cases/Deaths today: %d/%d)' % (expapprox(my_country_current_cases, tau1), expapprox(my_country_current_deaths, tau2)))#(linapprox(csv_country, 'cases'), linapprox(csv_country, 'deaths')))
    if dolegend:
        ax5.legend(['$t_d \\approx %.2f$ days' % tau1, '$t_d \\approx %.2f$ days' % tau2, 'Cases','Deaths'], loc='upper left')

    # Country diff vs cases
    ax6.plot(csv_country['date'].values, csv_country['cases'].diff(), '.', color='orange', markersize=2)
    ax6.plot(csv_country['date'].values, csv_country['deaths'].diff(), '.', color='red', markersize=2)
    ax6.plot(csv_country['date'].values, csv_country['cases'].diff().rolling(window=7).mean(), '-', color='orange') # rolling
    ax6.plot(csv_country['date'].values, csv_country['deaths'].diff().rolling(window=7).mean(), '-', color='red')

    csv_us_live['cases'] -= my_country_current_cases
    csv_us_live['deaths'] -= my_country_current_deaths
    csv_us_live.plot('date', 'cases', kind='scatter', logy=True, grid=True, ax=ax6, color='blue', sharex='col', s=2)
    csv_us_live.plot('date', 'deaths', kind='scatter', logy=True, grid=True, ax=ax6, color='blue', sharex='col', s=2)

    ax6.grid(True, which='major', axis='both')
    plt.setp(ax6.get_xticklabels(), rotation=45)

    #ax6.set_yscale("log")
    ax6.set(xlabel='Date', ylabel='Occurence', title='New Cases in US\n(Total Cases/Deaths: %d/%d)' % (my_country_current_cases, my_country_current_deaths), yscale='log')
    if dolegend:
        ax6.legend(['New Cases', 'New Deaths','7-Day Avg. Cases', '7-Day Avg. Deaths'], loc='upper left')

    # Generate Plot
    plt.show()
