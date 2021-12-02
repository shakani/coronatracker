Country, state, and county level data visualization of COVID-19 data (cases and deaths) from New York Times (https://github.com/nytimes/covid-19-data/). 

Usage: python3 coronatracker.py [State] [County] [optional float between 0 and 1 for fitting]

Top row plots provide daily accumulated COVID-19 case number (orange) and deaths (red) in (left to right) the US, queued State, and queued County on a logarithmic scale. Solid lines represent 7-day moving averages of data. Data is fit (dashed lines) to 2^(t/t_d), where t_d is doubling time. Blue dots represent today's data.
Optional floating point argument, p, fits the last (1-p) percent of data. For example, queuing p = 0.95 will fit data between current load and when it was 5% lower. 

Bottom row similarly plots first difference of cases/deaths (i.e. new cases deaths) along with a 7-day moving average. Total cases/deaths are provided in the bottom row's plot headings. Estimated cases/deaths for today are provided in upper row's plot headings. Estimated data is determined by fitting data to 2^(t/t_d) and extrapolating to today. 

Data fitting is performed by linear least squares regression on the log of cases/deaths. 
