from fitting.globaloptimization import fit_data
from scipy.optimize import Bounds

x = (0.07,13.64440988094479,83.15247343896175,0.10296026520227448,0.1,1.5230931197615067) #initial guess to start with
bounds = Bounds([0.06,0,0,0.05,0.05,0],[0.09,30,1000,0.4,0.4,12])

#fit 22 percent data
fixedparams = (190e-3,-0.134,0.067,0.805) #T,T1multvalue,T11multvalue,mumultvalue

#x = fit_data(sample="2601C",doping="22",temp=35,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
#x = fit_data(sample="2601C",doping="22",temp=30,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
#x = fit_data(sample="2601C",doping="22",temp=25,theta_max=70,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
x = (0.0821,9.8,20.8,0.11172046324944028,0.3944838562322481,1.64)
x = fit_data(sample="2601C",doping="22",temp=20,theta_max=45,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)

#fit 24 percent data
fixedparams = (190e-3,-0.132,0.066,0.81) #T,T1multvalue,T11multvalue,mumultvalue
x = (0.0811,13.5,20,0.12,0.39,3.2)

x = fit_data(sample="2511A",doping="24",temp=35,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
x = fit_data(sample="2511A",doping="24",temp=30,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
x = fit_data(sample="2511A",doping="24",temp=25,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)
x = fit_data(sample="2511A",doping="24",temp=20,theta_max=60,field=45.0,fixedparams=fixedparams,initialguess=x,bounds=bounds,parallel_over_theta=True,plot=True)


