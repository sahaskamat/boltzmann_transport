from fitting.basinhopping import fit_data

#fit 22 percent data
fixedparams = (190e-3,-0.134,0.067,0.805) #T,T1multvalue,T11multvalue,mumultvalue
tz = fit_data(sample="2601C",doping="22",temp=35,theta_max=99,field=45,fixedparams=fixedparams,fixTz=False)
fit_data(sample="2601C",doping="22",temp=30,theta_max=99,field=45,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz)
fit_data(sample="2601C",doping="22",temp=25,theta_max=70,field=45,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz)
fit_data(sample="2601C",doping="22",temp=20,theta_max=45,field=45,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz)

#fit 24 percent data
fixedparams = (190e-3,-0.132,0.066,0.81) #T,T1multvalue,T11multvalue,mumultvalue
tz = fit_data(sample="2511A",doping="24",temp=35,theta_max=99,field=45,fixedparams=fixedparams,fixTz=False)
fit_data(sample="2511A",doping="24",temp=30,theta_max=99,field=45,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz)
fit_data(sample="2511A",doping="24",temp=25,theta_max=99,field=45,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz)
fit_data(sample="2511A",doping="24",temp=20,theta_max=60,field=45,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz)
