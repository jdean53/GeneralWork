'''
This script consists of completed functions I write within the Sto Cal II class
'''

'''
NOTE: walk_params and build_tree come from Homework 1 & 3
'''

'''Given an Up and Down factor (and Time), will output the binomial tree mu and sigma params'''
def walk_params(u,d,T,N):
    '''
    Given up/down factors and Time Step info, calculates the drift and vol term for a binomial recombinding tree
    ---
    Parameters:  
    u - (float) up term (NOTE: Function will add one to this, so to double a number on up input 1 (1x+1=2x))  
    d - (float) down term (NOTE: Function subtracts this from 1, so to halve a number input 0.5)  
    T - (float) Time to Maturity (in years)  
    N - (int) Time Steps  
    ---
    Returns:  
    mu - (float) Estimated drift parameter  
    simgma - (float) Estimated vol parameter
    '''
    import numpy as np
    dt = T/N
    mu = (np.log(1+u) + np.log(1-d)) / (2*dt)
    sigma = (np.log(1+u) - mu*dt) / (np.sqrt(dt))

    return mu, sigma

'''Prices a european option with a binomial self-combining tree -- CRR Model'''
def build_tree(N, T, mu, sigma, s0, k, r, option_type, nature):
    '''
    Builds a self combining stock-price tree used to price an option  
    ---  
    Parameters:  
    N - (int) Number of periods  
    T - (float) time (in years)  
    mu - (float) drift parameter (can be calculated using walk_params())  
    sigma - (float) vol parameter (can be calculated using walk_params())  
    s0 - (float) current stock price  
    k - (int) option strike price  
    r - (float) money-market interest rate  
    option_type - (str) Call or Put option (input 'c' or 'p')  
    nature - (str) European or American Option (input 'american' or 'european')  
    ---  
    Returns:  
    tree - (DataFrame) Option tree as a df
    '''

    '''Required Packages'''
    import numpy as np
    import pandas as pd

    '''Establish time delta -- each variable represents
    T - time to maturity in years
    N - number of periods (i.e. time steps)'''
    dt = T / N

    '''Establish up and down factos'''
    u = np.exp(mu*dt + sigma*np.sqrt(dt))
    d = np.exp(mu*dt - sigma*np.sqrt(dt))
    
    '''Establish risk neutral probability'''
    q = ((1+r*dt) - d) / (u - d)
    
    '''Constructs the data frame -- each column and its representation
    index - number
    layer - time step
    stock_price - stock price at that node and layer
    phi - shares in replicating protfolio
    psi - amount of funds in money market
    option value - option price at that node'''
    tree = pd.DataFrame(data=None, index=range(0,sum(range(N+2))))
    tree.index.name = 'node'
    tree['layer'] = ""
    tree['stock_price'] = 0
    tree['phi'] = 0
    tree['psi'] = 0
    
    '''inputs intial stock price'''
    tree['stock_price'].loc[0] = s0

    '''Inputs layers (time steps) and further stock prices'''
    node = 0
    for i in range(0,N+2):
        if i < (N+1):
            tree.iloc[range(node,node+i+1),0] = i #Layer
            tree.iloc[range(node,node+i+1),1] = s0 * d ** (np.arange(i,-1,-1)) * u ** (np.arange(0,i+1,1)) #Stock Price
            node += i + 1

    '''Puts intrinsic option value in at every node'''
    if option_type == 'c':
        tree['option_value'] = tree['stock_price'].apply(lambda x: x-k if x-k > 0 else 0)
    elif option_type == 'p':
        tree['option_value'] = tree['stock_price'].apply(lambda x: k-x if k-x > 0 else 0)
    
    '''resets node to N-1'''
    
    node= node - N
    
    '''Calculates option value and replicating portfolio going backwards'''
    for i in reversed(range(N)):
        price_range = (1 / (1+r*dt)) * (q * tree.iloc[range(node,node+i+1),4].to_numpy() + (1-q) * tree.iloc[range(node-1,node+i),4].to_numpy()) # price option
        if nature == 'european':
            tree.iloc[range(node-i-2,node-1),4] = price_range
        elif nature == 'american':
            tree.iloc[range(node-i-2,node-1),4] = np.maximum(price_range, tree.iloc[range(node-i-2,node-1),4].to_numpy())
        tree.iloc[range(node-i-2,node-1),2] = (tree.iloc[range(node,node+i+1),4].to_numpy() - tree.iloc[range(node-1,node+i),4].to_numpy()) / (tree.iloc[range(node,node+i+1),1].to_numpy() - tree.iloc[range(node-1,node+i),1].to_numpy()) # get delta
        tree.iloc[range(node-i-2,node-1),3] = (tree.iloc[range(node,node+i+1),4].to_numpy() - tree.iloc[range(node-i-2,node-1),2].to_numpy() * tree.iloc[range(node,node+i+1),1].to_numpy()) / (1 + r*dt) # Money Market
        node = node - i - 1

    return tree

'''
NOTE: asset_data, maturity, and rates_data comes from Homework 4
'''

def asset_data(ticker, start_date, end_date):
    '''
    Function Designed to Pull Equity Data
    ----  
    Parameters:  
    ticker - (str) Ticker of desired equity  
    start_date - (str) Date to start pulling data from  
    end_data - (str) Date to end data pull, typically today  
    ---  
    Returns:  
    asset - 
    data - 
    '''
    import datetime
    import yfinance as yf
    import pandas as pd
    if type(start_date) == datetime.datetime:
        start_date = start_date.strftime('%Y-%m-%d')
    else:
        pass
    
    if type(end_date) == datetime.datetime:
        end_date = end_date.strftime('%Y-%m-%d')
    else:
        pass

    asset = yf.Ticker(ticker)
    data = yf.download(ticker, start_date, end_date)                                # Pull Data
    
    return asset, data

def maturity(timedelta, asset):
    '''
    Function Designed to find option expiry closest to desired expiry
    ----
    Parameters:  
    timedelta - (float) time until expiry (EXPRESSED IN YEARS)  
    asset - (yfinance.ticker.Ticker) yfinance object whos options are being analyzed (use asset_data() function)  
    --- 
    Returns:  
    expiry - 
    years_to_maturity - 
    '''  
    
    import datetime

    days = timedelta*365                                                            # Time Expressed in years converted to days
    expiries = asset.options
    distance = []
    for expiry in expiries:                                                         # Calculate Distance from Desired Expiry
        distance.append(abs(datetime.datetime.strptime(expiry, '%Y-%m-%d') - 
                    (datetime.datetime.today() + 
                    datetime.timedelta(days))))

    expiry = expiries[distance.index(min(distance))]                                # Use minimum distance
    years_to_expiry = float((datetime.datetime.strptime(expiry, '%Y-%m-%d') - datetime.datetime.today()).days) / 365
    return expiry, years_to_expiry

def rates_data():
    '''
    Function designed to pull current yield on the US 10 year note  
    NOTE: Pulls the value as a number, not a percentage
    '''
    import yfinance as yf
    import datetime
    import pandas as pd
    rates = yf.download('^tnx', datetime.datetime.today())
    rf = rates['Adj Close'][0] / 100
    return rf

def estimate_params(M, data):
    '''
    Function designed to estimate Mu and Sigma parameters for use in Option Pricing
    ---
    M - Sample Size to consider  
    data - Asset DataFrame pulled from Asset_Data() function  
    '''
    import numpy as np
    import pandas as pd
    dt = 1/252                                              # 1 day timestep
    
    y = np.log(1 + data['Adj Close'].pct_change())[-M:]     # Log Returns
    
    sigma_sq = np.var(y) / dt
    sigma = np.sqrt(sigma_sq)                               # Sigma Estimator

    mu = np.mean(y) / dt + 0.5*sigma_sq                     # Mu Estimator
    
    return mu, sigma

'''
Multi-Stock Model Functions
'''

def multi_stock_data(tick_list, start_date, end_date):
    '''
    Pulls Adjusted Closing Price for Equity Ticker List  
    ---  
    Parameters:  
    tick_list - (list) List of Equities  
    start_date - (str) Date to start pulling data from  
    end_data - (str) Date to end data pull, typically today 
    ---
    Returns:  
    df - (DataFrame) DataFrame of Adjusted Closing Prices
    '''
    import yfinance as yf
    import pandas as pd

    df = pd.DataFrame()
    
    for ticker in tick_list:
        df[ticker] = yf.download(ticker, start_date, end_date)['Adj Close']
    
    return df

def estimate_params_multi_stock(tick_list, df, M):
    '''
    Estimates Drift, Vol, and Correlation parameters for a Multi Stock B Model  
    Feeds functions for parameter estimation of Multi Stock W Model
    ---  
    Parameters:  
    tick_list - (list) List of Equities  
    df - (DataFrame) Adjusted Closing Prices of Equities listed (pulled from multi_stock_data() function)  
    M - (int) Lookback period (in trading days)  
    ---
    Returns:  
    estimates - (DataFrame) Drift and Vol estimates  
    rho - (DataFrame) Correlation Matrix  
    '''
    import pandas as pd
    import numpy as np

    if M > len(df):
        M = len(df)
    else:
        pass

    y_df = np.log(df) - np.log(df.shift(1))
    estimates = pd.DataFrame(columns=tick_list,index=['drift','vol'])
    
    for ticker in tick_list:
        prices = df[ticker]
        y = y_df[ticker][-M:]
        sigma_sq = np.var(y) * 252
        sigma = np.sqrt(sigma_sq)
        mu = np.mean(y) * 252 + 0.5*sigma_sq
        estimates[ticker] = [mu, sigma]
    
    rho=y_df.corr()
    
    return estimates, rho

def estimate_w_form_params(vol, rho):
    '''
    Estimates the Sigma Matrix of the W-Form of the Multi Stock Model  
    ---
    Parameters:  
    vol - (pd.series) Volatility Vector  
    rho - (DataFrame) B-Form Correlation Matrix  
    Returns:  
    Sigma - (np.array) Sigma Matrix of W-Form
    '''
    import pandas as pd
    import numpy as np
    
    vol = vol.to_numpy()
    rho = rho.to_numpy()
    n = len(vol)
    C = np.zeros((n,n))
    
    for i in range(0,n):
        for j in range(0,n):
            C[i,j] = rho[i,j] * vol[i] * vol[j]
    
    Sigma = np.linalg.cholesky(C)
    
    return Sigma

def diversification_measure(vol, Sigma):
    '''
    Measures portfolio diversification according to estimates in B and W form of Multi Stock model  
    ---  
    Parameters:  
    vol - (pd.series) B-Form Volatilities  
    Sigma - (np.array) W-Form Volatilities  
    ---  
    Returns:  
    R - (np.array) Diversification Matrix  
    d - (float) Diversification measure
    '''
    import pandas as pd
    import numpy as np
    
    vol = vol.to_numpy()
    n = len(vol)
    R = np.zeros((n,n))
    
    for i in range(0,n):
        for j in range(0,n):
            R[i,j] = Sigma[i,j] / vol[i]
    
    d = np.linalg.det(R) ** (1 / (n-1))
    
    return R, d

'''bsm related functions'''

def price_bsm(s0, k, r, t, sigma):
    '''
    Prices a European Call Option using the Black-Scholes Formula
    ---  
    Parameters:  
    s0 - Current stock price
    k - strike price
    r - risk-free interest rate (per annum)
    t - time to maturity, in years
    sigma - asset volatitly  
    ---  
    Returns:  
    price (float) - Price of the option  
    '''
    import numpy as np
    from scipy.stats import norm
    d_plus = (np.log(s0/k) + (r + ((sigma ** 2) / 2)) * t) / (sigma * np.sqrt(t))
    d_less = (np.log(s0/k) + (r - ((sigma ** 2) / 2)) * t) / (sigma * np.sqrt(t))
    price = norm.cdf(d_plus) * s0 - norm.cdf(d_less) * k * np.exp(-r * t)

    return price

def put_call_parity(option_price, option_type, s0, k, r, t):
    '''
    Uses the price of a european option to return its opposite's price
    ---
    option_price - price of given option
    option_type - type of given option ('p' for put and 'c' for call)
    s0 - current asset price
    k - strike price
    r - risk-free interest rate (per annum)
    t - time to maturity, in years
    '''
    import numpy as np
    if option_type == 'c':
        price = - s0 + np.exp(-r * t)*k + option_price
    elif option_type == 'p':
        price = s0 - np.exp(-r * t)*k + option_price 
    
    return price

def bsm_greeks(s0, k, r, t, sigma):
    '''
    Uses the Black-Scholes formula to estimate the 5 major greeks (delta, gamma, theta, vega, rho)  
    ---  
    Parameters:  
    s0 (float) - Current stock price  
    k (float) - strike price  
    r (float) - risk-free interest rate (per annum)  
    t (float) - time to maturity, in years  
    sigma (float) - asset volatitly  
    ---  
    Returns:  
    greeks (dataframe) - calculated greeks in a dataframe indexed by name  
    '''
    import pandas as pd                                
    import numpy as np
    from scipy.stats import norm
    greeks = pd.DataFrame(index=['delta','gamma','theta','vega','rho'],columns=['greeks'])
    d_plus = (np.log(s0/k) + (r + ((sigma ** 2) / 2)) * t) / (sigma * np.sqrt(t))
    d_less = (np.log(s0/k) + (r - ((sigma ** 2) / 2)) * t) / (sigma * np.sqrt(t))
    greeks.loc['delta'] = norm.cdf(d_plus)
    greeks.loc['gamma'] = (1 / (np.sqrt(t) * sigma * s0)) * norm.pdf(d_plus)
    greeks.loc['theta'] = (((-1)*s0) / (2 * np.sqrt(t))) * norm.pdf(d_plus) - (r * k * np.exp((-r) * t) * norm.cdf(d_less))
    greeks.loc['vega'] = s0 * np.sqrt(t) * norm.pdf(d_plus)
    greeks.loc['rho'] = np.sqrt(t) * k * np.exp((-1) * r * t) * norm.cdf(d_less)
    return greeks
    

'''Multi-Stock Functions'''

def multi_stock_data(tick_list, start_date, end_date):
    '''
    Pulls Adjusted Closing Price for Equity Ticker List  
    ---  
    Parameters:  
    tick_list - (list) List of Equities  
    start_date - (str) Date to start pulling data from  
    end_data - (str) Date to end data pull, typically today 
    ---
    Returns:  
    df - (DataFrame) DataFrame of Adjusted Closing Prices
    '''
    import yfinance as yf
    import pandas as pd

    df = pd.DataFrame()
    
    for ticker in tick_list:
        df[ticker] = yf.download(ticker, start_date, end_date)['Adj Close']
    
    return df

def estimate_params_multi_stock(tick_list, df, M):
    '''
    Estimates Drift, Vol, and Correlation parameters for a Multi Stock B Model  
    Feeds functions for parameter estimation of Multi Stock W Model
    ---  
    Parameters:  
    tick_list - (list) List of Equities  
    df - (DataFrame) Adjusted Closing Prices of Equities listed (pulled from multi_stock_data() function)  
    M - (int) Lookback period (in trading days)  
    ---
    Returns:  
    estimates - (DataFrame) Drift and Vol estimates  
    rho - (DataFrame) Correlation Matrix  
    '''
    import pandas as pd
    import numpy as np

    if M > len(df):
        M = len(df)
    else:
        pass

    y_df = np.log(df) - np.log(df.shift(1))
    estimates = pd.DataFrame(columns=tick_list,index=['drift','vol'])
    
    for ticker in tick_list:
        prices = df[ticker]
        y = y_df[ticker][-M:]
        sigma_sq = np.var(y) * 252
        sigma = np.sqrt(sigma_sq)
        mu = np.mean(y) * 252 + 0.5*sigma_sq
        estimates[ticker] = [mu, sigma]
    
    rho=y_df.corr()
    
    return estimates, rho

def estimate_w_form_params(vol, rho):
    '''
    Estimates the Sigma Matrix of the W-Form of the Multi Stock Model  
    ---
    Parameters:  
    vol - (pd.series) Volatility Vector  
    rho - (DataFrame) B-Form Correlation Matrix  
    Returns:  
    Sigma - (np.array) Sigma Matrix of W-Form
    '''
    import pandas as pd
    import numpy as np
    
    vol = vol.to_numpy()
    rho = rho.to_numpy()
    n = len(vol)
    C = np.zeros((n,n))
    
    for i in range(0,n):
        for j in range(0,n):
            C[i,j] = rho[i,j] * vol[i] * vol[j]
    
    Sigma = np.linalg.cholesky(C)
    
    return Sigma

def diversification_measure(vol, Sigma):
    '''
    Measures portfolio diversification according to estimates in B and W form of Multi Stock model  
    ---  
    Parameters:  
    vol - (pd.series) B-Form Volatilities  
    Sigma - (np.array) W-Form Volatilities  
    ---  
    Returns:  
    R - (np.array) Diversification Matrix  
    d - (float) Diversification measure
    '''
    import pandas as pd
    import numpy as np
    
    vol = vol.to_numpy()
    n = len(vol)
    R = np.zeros((n,n))
    
    for i in range(0,n):
        for j in range(0,n):
            R[i,j] = Sigma[i,j] / vol[i]
    
    d = np.linalg.det(R) ** (1 / (n-1))
    
    return R, d

def trading_pairs(tick_list, legs):
    '''
    Returns all possible trading pairs for a list of equities  
    ---
    Parameters:  
    tick_list - (list) List of Equities to Pair  
    legs - (int) Number of legs for each trade  
    ---
    Returns:  
    Pairs - (list) List of trading pairs (list of tuples)
    '''
    import itertools
    
    if legs > len(tick_list):
        legs = len(tick_list)
    
    pairs = list(itertools.combinations(tick_list, r=legs))
    
    return pairs

