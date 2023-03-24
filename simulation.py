import numpy as np
from arch import arch_model
from arch.bootstrap import StationaryBootstrap

def generate_garch_paths(historical_data,N_paths=100,T_steps=360,block_size=9):
    """
    Generate future paths of asset returns using GJR-GARCH model and block-resampling bootstrap
    param: historical_data - np.array with historical asset returns
    param: N_paths - number of paths to generate
    param: T_steps - length of each path
    param: block_size - average length of block for resampling
    returns: np.ndarray of shape (T_steps, number of assets, N_paths)
    """
    T_hist,N_assets=historical_data.shape
    # Fit models
    fit_results = []
    for r in historical_data.T:
        model = arch_model(100*r, p=1, o=1, q=1,dist='skewt')
        fit_results.append(model.fit(disp="off"))
        
    # Generate block-bootstrap indices' sequences 
    n_iter=T_steps//T_hist + (1 if T_steps%T_hist>0 else 0)
    sampling_paths=np.zeros((n_iter*T_hist,N_paths),dtype=int)
    for k in range(n_iter):
        bs = StationaryBootstrap(block_size,np.arange(T_hist))
        sampling_paths[k*T_hist:(k+1)*T_hist,:]=np.column_stack([x[0][0] for x in bs.bootstrap(N_paths)])
        
    sim = np.zeros((n_iter*T_hist, N_assets, N_paths))
    for i in range(N_assets):
        sim[:,i,:]=0.01*np.column_stack([fit_results[i].params.mu+fit_results[i].resid[p] for p in sampling_paths]).T
    return sim[:T_steps,:,:]

def generate_norm_paths(historical_data,N_paths=100,T_steps=360):
    mu = historical_data.mean(axis=0)
    si = historical_data.std(axis=0)
    X = historical_data - mu[np.newaxis,:]
    # Divide each column of X by its standard deviation
    X = X / si[np.newaxis,:]
    # Calculate the correlation matrix
    Ro = np.dot(X.T, X) / (X.shape[0] - 1)
    L = np.linalg.cholesky(Ro)
    # Generate a random matrix of normally distributed variables with the given mean and standard deviation
    results=np.zeros((T_steps,len(mu),N_paths))
    for a in range(len(mu)):
        Z = np.random.normal(mu[a], si[a], (T_steps,N_paths))
        results[:,a,:] = Z
    for p in range(N_paths):
        results[:,:,p]=np.dot(L,results[:,:,p].T).T
    
    return results