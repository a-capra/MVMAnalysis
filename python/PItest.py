import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import pandas as pd
import json
import glob, os
import re

from checkers import read_jsonfile
from plotO2 import bisec

cinH2O2mbar=0.025

def getFilePathInfo(absolute):
  dirname = os.path.dirname(absolute)
  basename = os.path.basename(absolute)
  info = os.path.splitext(basename)
  filename = info[0]
  extend = info[1]
  return dirname, filename, extend

from math import pi, sqrt
def Venturi(dp,d1,d2,rho=1.204,side='lhs'):
  #https://en.wikipedia.org/wiki/Venturi_effect#Instrumentation_and_measurement
  # di in mm and rho in kg/m^3
  if d2 >= d1:
    print('Waring: the Venturi choke should be the second argument, I will swap them for you...')
    temp=d2
    d2=d1
    d1=temp
  r1=0.5*d1*0.001
  r2=0.5*d2*0.001 # ID in mm to radius in m
  dp*=100. # mbar to Pa
  A1=r1**2*pi
  A2=r2**2*pi
  
  if side == 'lhs':
    den=A1**2/A2**2-1.
    Q=A1*sqrt(2.0*dp/rho/den)
  elif side == 'rhs':
    den=1.-A2**2/A1**2
    Q=A2*sqrt(2.0*dp/rho/den)
  else: Q=-1.0
  return Q*60000. # m^3/s to slpm

def PressureTest(dirs):
  fin=glob.iglob(os.path.join(dirs, '*.json'))
  p1a=np.array([])
  p2a=np.array([])
  pmean_1a=np.array([])
  pmean_2a=np.array([])
  pstd_1a=np.array([])
  pstd_2a=np.array([])
  pmin_1a=np.array([])
  pmax_1a=np.array([])
  pmin_2a=np.array([])
  pmax_2a=np.array([])

  color=iter(cm.rainbow(np.linspace(0,1,15)))
  pcut=4.5 #mbar

  for i,f in enumerate(fin):
    if os.path.isfile(f):
      df=read_jsonfile(fname=f)
      size=len(df.index)

      _,fn,_=getFilePathInfo(f)
      regex=re.compile(r'\d+')
      out=regex.findall(f)
      p1=float(out[1])*cinH2O2mbar
      p2=float(out[3])*cinH2O2mbar

      ttot=df['time'].iloc[-1]
      if ttot < 5.: continue

      pmean=df['p_patient'].mean()
      pstd=df['p_patient'].std()
      pmin=df['p_patient'].min()
      pmax=df['p_patient'].max()
      
      if p2 == 0: 
        p1a=np.append(p1a,p1)
        pmean_1a=np.append(pmean_1a,pmean)
        pstd_1a=np.append(pstd_1a,pstd)
        pmin_1a=np.append(pmin_1a,pmin)
        pmax_1a=np.append(pmax_1a,pmax)
        pp=df['p_patient']*-1.0
        pref=p1
      elif p1 == 0.:
        p2a=np.append(p2a,p2)
        pmean_2a=np.append(pmean_2a,pmean)
        pstd_2a=np.append(pstd_2a,pstd)
        pmin_2a=np.append(pmin_2a,pmin)
        pmax_2a=np.append(pmax_2a,pmax)
        pp=df['p_patient']
        pref=p2
      #print(f'Reading {fn} as {i}, length: {size}, p1={p1}mbar, p2={p2}mbar, time={ttot}s, p average={pmean}mbar')
      #print('Reading {:s} as {:0d}, length: {:d}, p1={:1.1f}mbar, p2={:1.1f}mbar, time={:1.3f}s, p={:1.3f}mbar'.format(fn,i,size,p1,p2,ttot,pmean))

      if pmean > pcut:
        c=next(color)
        axs[0].plot(df['time'],pp,'-',c=c)
        if pref<0.: pref*=-1.
        axs[0].plot(df['time'],np.full(len(df['time'].index),pref),'--',c=c)


    else:
      print(f'{f} not a file')

  axs[0].set_title('Waveform Pressure > %1.1fmbar'%pcut)

  min_subtr_1a = pmean_1a - pmin_1a
  max_subtr_1a = pmax_1a - pmean_1a
  min_subtr_2a = pmean_2a - pmin_2a
  max_subtr_2a = pmax_2a - pmean_2a
  xpoints=np.linspace(0.0,p2a.max(),10)

  axs[1].set_title('Pressure Test')
  axs[1].errorbar(p1a,pmean_1a*-1.,yerr=[min_subtr_1a, max_subtr_1a],label='port -1',fmt='.')
  axs[1].errorbar(p2a,pmean_2a,yerr=[min_subtr_2a, max_subtr_2a],label='port 2',fmt='.')
  axs[1].plot(xpoints,bisec(xpoints),'--',c='Gray',label='not the fit')
  axs[1].set_xlabel('Dwyer Gauge [mbar]')
  axs[1].set_ylabel('TE 5525DSO-DB001DS pressure sensor [mbar]')
  axs[1].legend()
 
  axs[2].set_title('Difference between Sensors')
  axs[2].errorbar(p1a,-1*pmean_1a-p1a,yerr=[min_subtr_1a, max_subtr_1a],label='port -1',fmt='.')
  axs[2].errorbar(p2a,pmean_2a-p2a,yerr=[min_subtr_2a, max_subtr_2a],label='port 2',fmt='.')
  axs[2].plot(xpoints,np.zeros(xpoints.size),'--',c='Gray',label='not the fit')
  axs[2].set_xlabel('Dwyer Gauge [mbar]')
  axs[2].set_ylabel('TE sensor - Dwyer Gauge [mbar]')
  axs[2].legend()



def pol2(x,a,b,c):
  return a*x**2+b*x+c

def pol2x(x,a,b):
  return a*x**2+b*x

def pol4(x,a,b,c,d,e):
  return a*x**4+b*x**3+c*x**2+d*x+e

def polonehalf(x,a,b,c):
  return a*np.sqrt(x+b)

def pol1(x,m,q):
  return m*x+q

def FlowTest(dirs):
  # https://www.mathesongas.com/pdfs/flowchart/604%20(E700)/AIR%20604%20(E700)%20SS%200%20PSIG.pdf
  ro,cal=np.loadtxt(dirs+'AIR 604 (E700) SS 0 PSIG.txt',unpack=True) # slpm
  mnro,mxro=ro.min(),ro.max()
  flowCal = interpolate.interp1d(ro, cal)

  ta=np.array([])

  fin=glob.iglob(os.path.join(dirs, '*.json'))
  fa=np.array([])
  faerr=np.array([])
  #volNM=np.array([])

  # MVM readout
  pmean_a=np.array([])
  pstd_a=np.array([])

  flowavg_a=np.array([])
  flowerr_a=np.array([])
  volavg_a=np.array([])
  volstd_a=np.array([])


  # http://treymed.com/respmech/ezflow/adultpos.pdf
  pez,fez=np.loadtxt(dirs+'EZ-flow_calib.dat',unpack=True)
  pdrop=1.8*0.981#cmH2O->mbar
  ezcal,_=curve_fit(pol4,pez,fez)
  #ezcal,_=curve_fit(pol2,pez,fez)

 
  plt.subplot(221)
  plt.scatter(pez,fez,c='r',marker='o',label='EZ calib')
  pxxx=np.linspace(0.,1.,100)
  plt.plot(pxxx, pol4(pxxx, *ezcal), 'k--', 
            label='fit: a=%5.3f, b=%5.3f, c=%5.3f,\nd=%5.3f, e=%5.3f' % tuple(ezcal))
  #plt.plot(pez, pol2(pxxx, *ezcal), 'k--', 
  #          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(ezcal))

  idx=np.array([])
  for i,f in enumerate(fin):
    if os.path.isfile(f):
      df=read_jsonfile(fname=f)
      size=len(df.index)
      idx=np.append(idx,i)

      _,fn,_=getFilePathInfo(f)
      regex=re.compile(r'\d+')
      out=regex.findall(f)
      flow_nocal=float(out[0])

      ttot=df['time'].iloc[-1]
      if ttot < 5.: continue

      try:
        flow=flowCal(float(out[0]))
      except ValueError:
        print(f'Reading {fn} as {i}, length: {size}, uncalibrated flow={flow_nocal}slpm, time={ttot}s')
        continue

      pmean=df['p_patient'].mean()
      pstd=df['p_patient'].std()
      
      try:
        #ferr=(flowCal(float(out[0])+0.5)-flowCal(float(out[0])-0.5))/sqrt(12.0)
        ferr=(flowCal(float(out[0])+1.)-flowCal(float(out[0])-1.))/sqrt(12.0)
      except:
        ferr=(flowCal(float(out[0])+1.)-flowCal(float(out[0])))/sqrt(12.0)

      
      flowavg=df['flux_inhale'].mean()
      flowerr=df['flux_inhale'].std()
      #flowavg=df['f_total'].mean()
      #flowerr=df['f_total'].std()
      #flowavg=df['f_vent_raw'].mean()
      #flowerr=df['f_vent_raw'].std()
      volavg=df['v_total'].mean()
      volstd=df['v_total'].std()

      fa=np.append(fa,flow) # filename calibrated
      faerr=np.append(faerr,ferr)

      ta=np.append(ta,ttot)
      pmean_a=np.append(pmean_a,pmean)
      pstd_a=np.append(pstd_a,pstd)

      flowavg_a=np.append(flowavg_a,flowavg)
      flowerr_a=np.append(flowerr_a,flowerr)

      volavg_a=np.append(volavg_a,volavg)
      volstd_a=np.append(volstd_a,volstd)

      print('{:d}) p={:1.3f}mbar q={:1.1f}slpm err={:1.5}'.format(i,pmean,flow,ferr))

    else:
      print(f'{f} not a file')

  ##### "Calibration" #####

  pshift_val=pmean_a.min()
  print('pressure offset: {:1.3f}mbar'.format(pshift_val))
  porig=pmean_a
  pshift=pmean_a-pshift_val
  #print(porig,'\n',pshift)
  #popt,_=curve_fit(pol2,pshift,fa,sigma=faerr,absolute_sigma=True)
  popt,_=curve_fit(pol4,pshift,fa,sigma=faerr,absolute_sigma=True)
  #popt,_=curve_fit(polonehalf,pshift,fa,sigma=faerr,absolute_sigma=True)
  
  plt.title('Flow Test')
  plt.errorbar(porig,fa,xerr=pstd_a,yerr=faerr,label='Matheson calib',fmt='ok')
  plt.errorbar(pshift,fa,xerr=pstd_a,yerr=faerr,label='Matheson calib -{:1.3f}mbar'.format(pshift_val),fmt='.b')
  plt.xlabel('TE 5525DSO-DB001DS pressure sensor [mbar]')
  plt.ylabel('Flow [slpm]')
  #plt.plot(pxxx, pol2(pxxx, *popt), 'k-', 
  #          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
  plt.plot(pxxx, pol4(pxxx, *popt), 'k-', 
            label='fit: a=%5.3f, b=%5.3f, c=%5.3f,\nd=%5.3f, e=%5.3f' % tuple(popt))
  #plt.plot(pxxx, polonehalf(pxxx, *popt), 'k-', 
  #          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
  
  #left,_=plt.xlim()
  plt.xlim(-0.01,pshift.max()*1.2)
  #bottom,_=plt.ylim()
  plt.ylim(-1.,fa.max()*1.2)
  plt.legend(loc='upper left')
  plt.grid()

  print('NM fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' % tuple(popt))
  # ndof=fa.size-3
  # chi2,prob=chisquare(fa,pol2(pshift, *popt),ddof=ndof)
  # print('NM fit quality: chi2=%1.3f  p=%1.5f'%(chi2,prob))

  print('EZ-flow fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' % tuple(ezcal))
  #print('EZ-flow fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(ezcal))
  # chi2,prob=chisquare(fez,pol4(pez, *ezcal),ddof=fez.size-5)
  # print('EZ-flow fit quality: chi2=%1.3f  p=%1.5f'%(chi2,prob))
  
  ##########


  ###### Volume Calculation ######

  #volNM=np.append(volNM,flow*ttot/60.) # flow[slpm]*t[s]/60=vol[l]
  volNM=fa*ta/60.
  volNMerr=faerr*ta/60.
  
  feerr=np.array([])
  volEZ=np.array([])
  volEZerr=np.array([])
  for i,p in enumerate(pshift):
    fff=pol4(p, *ezcal)
    #fff=pol2(pmean, *ezcal)
    ffferr=0.05*fff
    if ffferr < 0.5: ffferr=0.5
    feerr=np.append(feerr,ffferr)
    volEZ=np.append(volEZ,fff*ta[i]/60.)
    volEZerr=np.append(volEZerr,ffferr*ta[i]/60.)

  #########################

  ##### Debug #####

  fe=pol4(pshift, *ezcal)
  #fe=pol2(pshift, *ezcal)
  plt.subplot(222)
  plt.title('Flow Comparison')
  plt.errorbar(idx,fa,yerr=faerr,label='Matheson')
  plt.errorbar(idx,fe,yerr=feerr,label='EZcalib')
  plt.xlabel('Test ID')
  plt.ylabel('Flow [slpm]')
  plt.xticks(idx)
  plt.grid()
  plt.legend()

  plt.subplot(224)
  plt.title('Volume Comparison')
  plt.errorbar(idx,volNM,yerr=volNMerr,label='Matheson')
  plt.errorbar(idx,volEZ,yerr=volEZerr,label='EZcalib')
  plt.xlabel('Test ID')
  plt.ylabel('Volume [litres]')
  plt.xticks(idx)
  plt.grid()
  plt.legend()

  ##########

  ##### Final result #####

  plt.subplot(223)
  plt.title('Volume Measurement Comparison')
  plt.errorbar(volNM*1.e3,volEZ*1.e3,xerr=volNMerr*1.e3,yerr=volEZerr*1.e3,c='k',marker='.',ls='none',label='Vol. comparison')
  plt.ylabel('EZflow [ml]')
  plt.xlabel('Matheson [ml]')
  xpoints=np.linspace(0.0,(volNM*1.e3).max(),10)
  #plt.plot(xpoints,bisec(xpoints),'--',c='Gray',label='not the fit')
  plt.grid()

  par,_=curve_fit( pol1,volNM*1.e3,volEZ*1.e3,sigma=volEZerr*1.e3,absolute_sigma=True, )
  plt.plot(xpoints, pol1(xpoints, *par), 'r--', 
            label='fit: m=%5.3f, q=%5.3f' % tuple(par))

  maximum_bias_error_volume = 4          # A in ml
  maximum_linearity_error_volume = 0.15  # B/100 for tidal volume
  
  maximum_error_string = f'$\pm$({maximum_bias_error_volume:.1f}+({(maximum_linearity_error_volume*100):.1f}% of Matheson))'
  print(maximum_error_string)
  x_limit = np.arange(0.0, (volNM*1.e3).max(), 0.01)
  max_limit = maximum_bias_error_volume + (1 + maximum_linearity_error_volume) * x_limit
  min_limit = -maximum_bias_error_volume + (1 - maximum_linearity_error_volume) * x_limit
  plt.fill_between(x_limit, min_limit, max_limit, facecolor='green', alpha=0.2, label=maximum_error_string)

  plt.legend(loc='upper left')

  succ=0
  fail=0
  for i,v in enumerate(volNM):
    nominal_volume=v*1.e3 #l->ml
    if nominal_volume < 50: continue

    nominal_volume_low=nominal_volume-maximum_bias_error_volume-maximum_linearity_error_volume*nominal_volume
    nominal_volume_wid=2*(maximum_bias_error_volume+maximum_linearity_error_volume*nominal_volume)

    min_volume=(volEZ[i]-volEZerr[i])*1.e3
    max_volume=(volEZ[i]+volEZerr[i])*1.e3

    if min_volume > nominal_volume_low and max_volume < nominal_volume_low + nominal_volume_wid:
      print("SUCCESS: Volume all values within maximum errors wrt set value")
      succ+=1
    else:
      print("FAILURE: Volume outside maximum errors wrt set value")
      fail+=1

  print(f'SUCCESS: {succ:d}   FAILURE:{fail:d}')





# # convert pressure differential to flow 
# # using calibration of Venturi flowmeter
# # flow is in slpm
# df['flux']=df['p_patient'].apply(pol4,args=par)
# # s->min
# df['dt']=df['dt']/60.
# # calculate instanteous volume
# df['flux_x_dt']=df['flux']*df['dt'].diff()
# # integrate (sum) to get total volume
# # and l->cl
# df['volume']=df['flux_x_dt'].cumsum()*100.




if __name__=='__main__':

  dirs=["C:/Users/andre/Documents/MVM/breathing_simulator_test/data/pressureTest/",
        "C:/Users/andre/Documents/MVM/breathing_simulator_test/data/flowTest/",]
 
  fig, axs = plt.subplots(3)
  PressureTest(dirs[0])
  fig.set_size_inches(10,9)
  fig.tight_layout()
  fig.savefig('PIprestest.pdf')

  plt.figure(2)
  FlowTest(dirs[1])
  fig2=plt.gcf()
  fig2.set_size_inches(18,9)
  fig2.tight_layout()
  fig2.savefig('PIflowtest_v0.pdf')
  plt.show()
