Country, state, and county level data visualization of COVID-19 data (cases and deaths) from New York Times (https://github.com/nytimes/covid-19-data/). 

Usage: python3 coronatracker.py [State] [County] [optional float between 0 and 1 for fitting]

Top row provides daily accumulated COVID-19 case number (orange) and deaths (red) in (left to right) the US, queued State, and queued County on a logarithmic scale. Solid lines represent 7-day moving averages of data. Data is fit to 2^(t/t_d), where t_d is doubling time. 
Optional floating point argument, p, fits the last (1-p) percent of data. For example, queuing p = 0.95 will fit data between current load and when it was 5% lower. 
