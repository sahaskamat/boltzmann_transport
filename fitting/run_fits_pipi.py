import os
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"

from fitting.globaloptimization import fit_data
from scipy.optimize import Bounds

bounds = Bounds([0.02,0.1,12,0.05,0.5],[0.045,30,30,0.5,8])

#fit 24 percent data
fixedparams = (190e-3,-0.132,0.066,0.81) #T,T1multvalue,T11multvalue,mumultvalue
x = (0.03048,6.8,1000000,0.143,6.3)

#x = fit_data(sample="2511A",doping="24",temp=35,theta_max=99,field=45.0,fixedparams=fixedparams,delta_in_k=True,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
#x = fit_data(sample="2511A",doping="24",temp=30,theta_max=99,field=45.0,fixedparams=fixedparams,delta_in_k=True,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
#x = fit_data(sample="2511A",doping="24",temp=25,theta_max=99,field=45.0,fixedparams=fixedparams,delta_in_k=True,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
#x = fit_data(sample="2511A",doping="24",temp=20,theta_max=60,field=45.0,fixedparams=fixedparams,delta_in_k=True,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)

#fit 22 percent data
fixedparams = (190e-3,-0.134,0.067,0.805) #T,T1multvalue,T11multvalue,mumultvalue
x = (0.03215,9,26,0.1363138263900155,6)

x = fit_data(sample="2601C",doping="22",temp=35,theta_max=99,field=45.0,fixedparams=fixedparams,delta_in_k=True,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
x = fit_data(sample="2601C",doping="22",temp=30,theta_max=99,field=45.0,fixedparams=fixedparams,delta_in_k=True,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
x = fit_data(sample="2601C",doping="22",temp=25,theta_max=70,field=45.0,fixedparams=fixedparams,delta_in_k=True,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
x = fit_data(sample="2601C",doping="22",temp=20,theta_max=45,field=45.0,fixedparams=fixedparams,delta_in_k=True,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
