from globaloptimization_v2 import calculate_ADMR,load_and_interpolate,chi_square,save_log

def costfunction(Tzmultvalue,invtau_iso,strength,spread_xy):

    B,T,T1multvalue,T11multvalue,Tzmultvalue,mumultvalue = 41.5,190e-3,-0.132,0.066,Tzmultvalue,0.86
    invtau_iso = invtau_iso
    scattering_params_list = [["pipidelta_exp",True,{"strength":strength,"spread_xy":spread_xy,"n":4}]]
    res_z,res_xy = 20,100

    theta_calc,rho_0_calc,rho_45_calc = calculate_ADMR(B,T,T1multvalue,T11multvalue,Tzmultvalue,mumultvalue,invtau_iso,scattering_params_list,res_z,res_xy)

    theta_data,rho_0_data = load_and_interpolate(theta_calc,sample="LSCO-2609-27E",field=41.5,phi=0,temp=20)
    theta_data,rho_45_data = load_and_interpolate(theta_calc,sample="LSCO-2609-27E",field=41.5,phi=45,temp=20)

    cost = chi_square(rho_0_data,rho_0_calc) + chi_square(rho_45_data,rho_45_calc)

    save_log(log_filename="cost_log.csv",sample="LSCO-2609-27E",field=41.5,temp=20,cost=cost,B=B,T=T,T1multvalue=T1multvalue,T11multvalue=T11multvalue,Tzmultvalue=Tzmultvalue,mumultvalue=mumultvalue,invtau_iso=invtau_iso,scattering_params_list=scattering_params_list,res_z=res_z,res_xy=res_xy)

    return cost