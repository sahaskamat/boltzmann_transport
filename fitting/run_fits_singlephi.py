from fitting.basinhopping_singlephi import fit_data

#fit 22 percent data
fixedparams = (190e-3,-0.134,0.067,0.805) #T,T1multvalue,T11multvalue,mumultvalue
tz0,invtau_iso0,invtau_aniso0 = fit_data(sample="2601C",doping="22",temp=35,theta_max=99,phi=30,field=41.5,fixedparams=fixedparams,fixTz=False,initial_guess=(0.0794,14.0468,189.855))

invtau_iso,invtau_aniso = fit_data(sample="2601C",doping="22",temp=30,theta_max=99,field=41.5,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz0,initial_guess=(tz0,invtau_iso0,invtau_aniso0))
invtau_iso,invtau_aniso = fit_data(sample="2601C",doping="22",temp=25,theta_max=70,field=41.5,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz0,initial_guess=(tz0,invtau_iso,invtau_aniso))
invtau_iso,invtau_aniso = fit_data(sample="2601C",doping="22",temp=20,theta_max=45,field=41.5,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz0,initial_guess=(tz0,invtau_iso,invtau_aniso))

invtau_iso,invtau_aniso = fit_data(sample="2601C",doping="22",temp=40,theta_max=99,phi=30,field=41.5,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz0,initial_guess=(tz0,invtau_iso0,invtau_aniso0))
invtau_iso,invtau_aniso = fit_data(sample="2601C",doping="22",temp=50,theta_max=99,phi=30,field=41.5,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz0,initial_guess=(tz0,invtau_iso0,invtau_aniso0))
invtau_iso,invtau_aniso = fit_data(sample="2601C",doping="22",temp=70,theta_max=99,phi=30,field=41.5,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz0,initial_guess=(tz0,invtau_iso0,invtau_aniso0))
invtau_iso,invtau_aniso = fit_data(sample="2601C",doping="22",temp=100,theta_max=99,phi=30,field=41.5,fixedparams=fixedparams,fixTz=True,tzfixedvalue=tz0,initial_guess=(tz0,invtau_iso0,invtau_aniso0))