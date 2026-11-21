from transport.wrappers import create_dispersion,create_FS,create_SM,rhozz_vs_theta,load_data_feb2026
import numpy as np
import json
import os
import csv

def calculate_ADMR(B,T,T1multvalue,T11multvalue,Tzmultvalue,mumultvalue,invtau_iso,scattering_params_list,res_z,res_xy):
    """
    Calculates ADMR curves at phi=0 and phi=45 for given values of scattering parameters
    scattering_params_list should have the format 
    [str:scattering kernel name,bool:delta in k,dict:kwargs dict for each scattering kernel]
    see examples/example_ADMR_with_wrapper.py for a usecase
    """
    dispersionInstance = create_dispersion(T,T1multvalue,T11multvalue,Tzmultvalue,mumultvalue)
    FSinstance = create_FS(dispersionInstance,res_z,res_xy)
    FSinstance.calculateDoping()

    SMinstance = create_SM(dispersionInstance,FSinstance,invtau_iso,scattering_params_list=scattering_params_list)

    theta_array = np.sort(np.concatenate((np.linspace(-14,100,20),[0])))
    rho_array_0 = rhozz_vs_theta(B,phi=0,theta_array=theta_array,scatteringMatrixInstance=SMinstance)
    rho_array_45 = rhozz_vs_theta(B,phi=45,theta_array=theta_array,scatteringMatrixInstance=SMinstance)

    return theta_array,rho_array_0/10000,rho_array_45/10000

def calculate_ADMR_reduced(B,T,T1multvalue,T11multvalue,Tzmultvalue,mumultvalue,invtau_iso,scattering_params_list,res_z,res_xy):
    """
    Calculates ADMR curves at phi=0 and phi=45 for given values of scattering parameters
    scattering_params_list should have the format 
    [str:scattering kernel name,bool:delta in k,dict:kwargs dict for each scattering kernel]
    see examples/example_ADMR_with_wrapper.py for a usecase
    """
    dispersionInstance = create_dispersion(T,T1multvalue,T11multvalue,Tzmultvalue,mumultvalue)
    FSinstance = create_FS(dispersionInstance,res_z,res_xy)
    FSinstance.calculateDoping()

    SMinstance = create_SM(dispersionInstance,FSinstance,invtau_iso,scattering_params_list=scattering_params_list)

    theta_array = np.array([0,90])
    rho_array_0 = rhozz_vs_theta(B,phi=0,theta_array=theta_array,scatteringMatrixInstance=SMinstance)
    rho_array_45 = rhozz_vs_theta(B,phi=45,theta_array=theta_array,scatteringMatrixInstance=SMinstance)

    return theta_array,rho_array_0/10000,rho_array_45/10000

def save_log(log_filename,sample,field,temp,cost,B,T,T1multvalue,T11multvalue,Tzmultvalue,mumultvalue,invtau_iso,scattering_params_list,res_z,res_xy):

    file_exists = os.path.exists(log_filename)

    with open(log_filename, "a") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["cost","sample","field","temp", "B", "T", "T1multvalue", "T11multvalue","Tzmultvalue", "mumultvalue", "invtau_iso","scattering_params_list", "res_z", "res_xy"])

        writer.writerow(writer.writerow([cost,sample,field,temp, B, T, T1multvalue, T11multvalue,Tzmultvalue, mumultvalue, invtau_iso,json.dumps(scattering_params_list), res_z, res_xy]))

def load_and_interpolate(theta_interp,sample,field,phi,temp):

    theta,rho = load_data_feb2026(sample,field,phi,temp)
    rho_interp = np.interp(theta_interp,theta,rho)
    return theta_interp,rho_interp

def chi_square(arr1,arr2):
    return np.sum((arr1-arr2)**2)

def plot(theta,rho_0_calc,rho_45_calc,rho_0_data,rho_45_data):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\rho$ (mOhm cm)")
    
    ax.plot(theta,rho_0_calc,label=r"Model $\phi=0$")
    ax.plot(theta,rho_45_calc,label=r"Model $\phi=45$")
    ax.plot(theta,rho_0_data,"o",label=r"Data $\phi=0$")
    ax.plot(theta,rho_45_data,"o",label=r"Data $\phi=45$")

    plt.legend()
    plt.show()