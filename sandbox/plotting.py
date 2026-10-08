import numpy as np

def normalize_data(theta,rho):
    """Normalize the data to the value at theta=0."""
    theta = np.asarray(theta)
    rho = np.asarray(rho)
    rho_norm = rho/np.interp(0,theta,rho)
    return (theta,rho_norm)