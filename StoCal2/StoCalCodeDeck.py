'''
This script consists of completed functions I write within the Sto Cal II class
'''

'''Given an Up and Down factor (and Time), will output the binomial tree mu and sigma params'''
def walk_params(u,d,T,N):
    '''Required Packages'''
    import numpy as np

    '''get time step'''
    dt = T/N

    '''get mean'''
    mu = (np.log(1+u) + np.log(1-d)) / (2*dt)
    
    '''get variance'''
    sigma = (np.log(1+u) - mu*dt) / (np.sqrt(dt))

    return mu, sigma

'''Prices a european option with a binomial self-combining tree -- CRR Model'''
def build_tree(N, T, mu, sigma, s0, k, r, option_type):
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
        tree.iloc[range(node-i-2,node-1),4] = (1 / (1+r*dt)) * (q * tree.iloc[range(node,node+i+1),4].to_numpy() + (1-q) * tree.iloc[range(node-1,node+i),4].to_numpy()) # price option
        tree.iloc[range(node-i-2,node-1),2] = (tree.iloc[range(node,node+i+1),4].to_numpy() - tree.iloc[range(node-1,node+i),4].to_numpy()) / (tree.iloc[range(node,node+i+1),1].to_numpy() - tree.iloc[range(node-1,node+i),1].to_numpy()) # get delta
        tree.iloc[range(node-i-2,node-1),3] = (tree.iloc[range(node,node+i+1),4].to_numpy() - tree.iloc[range(node-i-2,node-1),2].to_numpy() * tree.iloc[range(node,node+i+1),1].to_numpy()) / (1 + r*dt) # Money Market
        node = node - i - 1


    return tree