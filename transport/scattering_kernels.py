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

def pipizero(deltak,g,strength,spread_xy,spread_z,n):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    return (strength/((spread_xy**2)*spread_z))*np.exp(-(((np.abs(qx)-g[0]/2)**2 + (np.abs(qy)-g[1]/2)**2)/(spread_xy**2))**n)*np.exp(-(qz**2)/(spread_z**2))

def pipizero_plus_fwd(deltak,g,strength,spread_xy,spread_z,n):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]
    return (strength/((spread_xy**2)*spread_z))*np.exp(-(((np.abs(qx)-g[0]/2)**2 + (np.abs(qy)-g[1]/2)**2)/(spread_xy**2))**n)*np.exp(-(qz**2)/(spread_z**2))

def piminusdelta_plus_fwd(deltak,g,strength_pipi,spread_z_pipi,strength_fwd):
    qx,qy,qz = deltak[:,0],deltak[:,1],deltak[:,2]

    #scattering peaked at (pi-delta,pi-delta,0)
    spread_xy = 0.04*g[0]
    delta = 0.05*g[0]
    piminusdelta = g[0]*0.5 - delta
    pi = g[0]*0.5
    n = 1
    piminusdelta = (strength_pipi/((spread_xy**2)*spread_z_pipi))*(np.exp(-(((np.abs(qx)-piminusdelta)**2 + (np.abs(qy)-pi)**2)/(spread_xy**2))**n)+np.exp(-(((np.abs(qx)-pi)**2 + (np.abs(qy)-piminusdelta)**2)/(spread_xy**2))**n))*np.exp(-(qz**2)/(spread_z_pipi**2))

    #forward scattering
    spread_xy_fwd = 0.5*g[0]
    forward = (strength_fwd/(spread_xy_fwd**2))*np.exp((qx**2 + qy**2)/(spread_xy_fwd**2))

    return piminusdelta + forward