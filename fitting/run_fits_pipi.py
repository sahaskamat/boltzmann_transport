from fitting.globaloptimization import fit_data
from scipy.optimize import Bounds

#fit 24 percent data
fixedparams = (190e-3,-0.132,0.066,0.81) #T,T1multvalue,T11multvalue,mumultvalue
x = (0.07923269529579888,15.448737809202544,9.218302798224329,0.14625518207122723,3.9) #initial guess to start with
bounds = Bounds([0.06,0,0,0.05,2],[0.08,30,1000,0.4,4])

x = fit_data(sample="2511A",doping="24",temp=35,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False)
x = fit_data(sample="2511A",doping="24",temp=30,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False)
x = fit_data(sample="2511A",doping="24",temp=25,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False)
x = fit_data(sample="2511A",doping="24",temp=20,theta_max=60,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False)

#fit 22 percent data
fixedparams = (190e-3,-0.134,0.067,0.805) #T,T1multvalue,T11multvalue,mumultvalue

x = fit_data(sample="2601C",doping="22",temp=35,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False)
x = fit_data(sample="2601C",doping="22",temp=30,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False)
x = fit_data(sample="2601C",doping="22",temp=25,theta_max=70,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False)
x = fit_data(sample="2601C",doping="22",temp=20,theta_max=45,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=False)
