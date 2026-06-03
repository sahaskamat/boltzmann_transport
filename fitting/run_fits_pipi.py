import os
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"

from fitting.globaloptimization import fit_data
from scipy.optimize import Bounds

x = (0.08909408166377256,10,21,0.1369513035574913,0.4,2.318873746507372,0.35,0.5,0.5) #initial guess to start with
bounds = Bounds([0.06,0,0,0.05,0.05,0,0,0.2,0.2],[0.13,30,1000,0.4,1,12,10,1,1])

#fit 22 percent data
#fixedparams = (190e-3,-0.134,0.067,0.805) #T,T1multvalue,T11multvalue,mumultvalue

#x = fit_data(sample="2601C",doping="22",temp=35,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False,plot=False)
#x = fit_data(sample="2601C",doping="22",temp=30,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False,plot=False)
#x = fit_data(sample="2601C",doping="22",temp=25,theta_max=70,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False,plot=False)
#x = fit_data(sample="2601C",doping="22",temp=20,theta_max=45,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False,plot=False)

#fit 24 percent data
fixedparams = (190e-3,-0.132,0.066,0.81) #T,T1multvalue,T11multvalue,mumultvalue
x = (0.08909408166377256,10,21,0.1369513035574913,0.4,2.318873746507372,0.35,0.5,0.5)

x = fit_data(sample="2511A",doping="24",temp=35,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False,plot=False)
x = fit_data(sample="2511A",doping="24",temp=30,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False,plot=False)
x = fit_data(sample="2511A",doping="24",temp=25,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False,plot=False)
x = fit_data(sample="2511A",doping="24",temp=20,theta_max=60,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False,plot=False)


