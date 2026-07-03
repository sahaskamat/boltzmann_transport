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

def pipidelta(deltak,g,strength,spread_xy,n):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    #secretly delta in qz, implemented through conductivity.py
    return (strength/spread_xy**2)*np.exp(-(((np.abs(qx)-g[0]/2)**2 + (np.abs(qy)-g[1]/2)**2)/(spread_xy**2))**n)

def pipizero(deltak,g,strength,spread_xy,spread_z,n):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    return (strength/((spread_xy**2)*spread_z))*np.exp(-(((np.abs(qx)-g[0]/2)**2 + (np.abs(qy)-g[1]/2)**2)/(spread_xy**2))**n)*np.exp(-(qz**2)/(spread_z**2))

def pipipi(deltak,g,strength,spread_xy,spread_z,n):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    return (strength/((spread_xy**2)*spread_z))*np.exp(-(((np.abs(qx)-g[0]/2)**2 + (np.abs(qy)-g[1]/2)**2)/(spread_xy**2))**n)*np.exp(-((np.abs(qz)-g[2]/2)**2)/(spread_z**2))

def pipizero_plus_fwd(deltak,g,strength,spread_xy,spread_z,n,strength_fwd,spread_z_fwd,spread_xy_fwd):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    pipizero =  (strength/((spread_xy**2)*spread_z))*np.exp(-(((np.abs(qx)-g[0]/2)**2 + (np.abs(qy)-g[1]/2)**2)/(spread_xy**2))**n)*np.exp(-(qz**2)/(spread_z**2))
    fwd = (strength_fwd/((spread_z_fwd)*(spread_xy_fwd**2)))*np.exp(-(qx**2 + qy**2)/(spread_xy_fwd**2))*np.exp(-(qz**2)/(spread_z_fwd**2))
    return pipizero + fwd

def piminusdeltapiminusdeltazero(deltak,g,strength,spread_xy,spread_z,delta,n):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]

    piminusdelta = (g[0]/2)*(1-delta)
    piplusdelta = (g[0]/2)*(1+delta)
    pi = (g[0]/2)

    pi_piminusdelta_gaussian = np.exp(-(((np.abs(qx)-piminusdelta)**2 + (np.abs(qy)-pi)**2)/(spread_xy**2))**n)
    pi_piplusdelta_gaussian = np.exp(-(((np.abs(qx)-piplusdelta)**2 + (np.abs(qy)-pi)**2)/(spread_xy**2))**n)
    piminusdelta_pi_gaussian = np.exp(-(((np.abs(qx)-pi)**2 + (np.abs(qy)-piminusdelta)**2)/(spread_xy**2))**n)
    piplusdelta_pi_gaussian = np.exp(-(((np.abs(qx)-pi)**2 + (np.abs(qy)-piplusdelta)**2)/(spread_xy**2))**n)

    zaxis_gaussian = np.exp(-(qz**2)/(spread_z**2))

    prefactor = (strength/((spread_xy**2)*spread_z))*0.25

    return prefactor*(pi_piminusdelta_gaussian+pi_piplusdelta_gaussian+piminusdelta_pi_gaussian+piplusdelta_pi_gaussian)*zaxis_gaussian