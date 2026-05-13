from fitting.globaloptimization import fit_data

#fit 24 percent data
fixedparams = (190e-3,-0.132,0.066,0.81) #T,T1multvalue,T11multvalue,mumultvalue
x = (0.11200399313795195,13.714769432884154,78.1863057646791,0.10248028810608001,1.4870367496605148) #initial guess to start with

#x = fit_data(sample="2511A",doping="24",temp=30,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x)
x = fit_data(sample="2511A",doping="24",temp=35,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x)
x = fit_data(sample="2511A",doping="24",temp=25,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x)
x = fit_data(sample="2511A",doping="24",temp=20,theta_max=60,field=45.0,fixedparams=fixedparams,initialguess=x)

#fit 22 percent data
fixedparams = (190e-3,-0.134,0.067,0.805) #T,T1multvalue,T11multvalue,mumultvalue

x = fit_data(sample="2601C",doping="22",temp=35,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x)
x = fit_data(sample="2601C",doping="22",temp=30,theta_max=99,field=45.0,fixedparams=fixedparams,initialguess=x)
x = fit_data(sample="2601C",doping="22",temp=25,theta_max=70,field=45.0,fixedparams=fixedparams,initialguess=x)
x = fit_data(sample="2601C",doping="22",temp=20,theta_max=45,field=45.0,fixedparams=fixedparams,initialguess=x)
