import numpy as np

def crepe(deltak,g,strength,spread_xy): 
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    return (strength/(spread_xy**2))*np.exp(-(qx**2 + qy**2)/(spread_xy**2))

def pancake(deltak,g,strength,spread_xy,spread_z):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    return (strength/((spread_z)*(spread_xy**2)))*np.exp(-(qx**2 + qy**2)/(spread_xy**2))*np.exp(-(qz**2)/(spread_z**2))

def fourfold_crepe(deltak,g,strength,spread_xy,spread_iso_xy,spread_z):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    return (strength/(spread_xy**2))*(np.exp(-(qx**2)/(spread_xy**2)) + np.exp(-(qy**2)/(spread_xy**2)))*np.exp(-(qx**2 + qy**2)/(spread_iso_xy**2))

def fourfold_pancake(deltak,g,strength,spread_xy,spread_iso_xy,spread_z):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    return (strength/(spread_xy**2))*(np.exp(-(qx**2)/(spread_xy**2)) + np.exp(-(qy**2)/(spread_xy**2)))*np.exp(-(qx**2 + qy**2)/(spread_iso_xy**2))*np.exp(-(qz**2)/(spread_z**2))

def offset_inverse_square(deltak,g,strength,spread_xy):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    return (strength/(spread_xy**2))/(qx**2 + qy**2 + spread_xy**2)\
    
def cos2theta(deltak,g,strength,spread_xy,angular_exponent):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    theta = np.arctan2(qy,qx)
    return (strength/(spread_xy**2))*np.exp(-(qx**2 + qy**2)/(spread_xy**2))*((np.cos(2*theta)**4)**angular_exponent)

def cos2theta_kz(deltak,g,strength,spread_xy,angular_exponent,spread_z):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    theta = np.arctan2(qy,qx)
    return (strength/(spread_xy**2))*np.exp(-(qx**2 + qy**2)/(spread_xy**2))*((np.cos(2*theta)**4)**angular_exponent)*np.exp(-(qz**2)/(spread_z**2))

def pipi(deltak,g,strength,spread_xy,n):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    return (strength/spread_xy**2)*np.exp(-(((np.abs(qx)-g[0]/2)**2 + (np.abs(qy)-g[1]/2)**2)/(spread_xy**2))**n)