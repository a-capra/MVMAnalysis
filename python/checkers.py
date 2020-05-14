import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import glob, os

from plotO2 import bisec

def read_jsonfile(fname):
  with open(fname) as f:
    data = json.load(f)
    df = pd.DataFrame.from_dict(data['data'])
  return df


def calculate_cycle_info(df):
  df['out_diff'] = df['pv2_ctrl'].diff().fillna(0)
  df['out_status'] = np.where(df['out_diff'] == 0, 'steady', np.where(df['out_diff'] < 0, 'closing', 'opening'))
  start_times = df[df['out_status'] == 'closing']['time'].unique()

  df['start']  = 0
  df['ncycle'] = 0
  for i,s in enumerate(start_times) :
    df.loc[df.time>s,  'start' ] = s
    df.loc[df.time>s,  'ncycle'] = i

  # determine inspiration end times
  inspiration_end_times   = df[df['out_status'] == 'opening']['time'].unique()
  inspiration_start_times = start_times
  if inspiration_end_times[0]< inspiration_start_times[0] :
    inspiration_end_times = inspiration_end_times[1:]

  df['is_inspiration']=0
  for i,(s,v) in enumerate(zip(inspiration_start_times, inspiration_end_times)):
    if i>=len(inspiration_start_times)-1 : continue
    this_inspiration   = (df.time>s) & (df.time<v) # all samples
    df.loc[this_inspiration, 'is_inspiration'] = 1

  return start_times



def timers(df):
  df['Dtime']=df['time'].diff().fillna(0)
  df['Dts']=df['ts'].diff().fillna(0)
  df['Deltats_ms']=df['Dtime']*1.e3


def volumers(df):
  df['tidal_volume'] = 0

  for c in df['start'].unique():
    cycle_data = df[df['start']==c]
    #cumul = cycle_data['flux_inhale'].cumsum()
    cumul = cycle_data['f_total'].cumsum()
    df.loc[df.start == c, 'tidal_volume'] = cumul

  DeltaT=df['Dtime'].mean()
  print('Average Delta t = {:0.3f}s'.format(DeltaT))
  # correct flow sum into volume (aka flow integral)
  df['tidal_volume'] *= DeltaT/60.*100
  # set inspiration-only variables to zero outside the inspiratory phase
  df['tidal_volume'] *= df['is_inspiration']
  maxV=df['tidal_volume'].max()

  return maxV



if __name__=='__main__':

  frames=[]

  #dirs=["C:/Users/andre/Documents/MVM/breathing_simulator_test/data/20200424/","C:/Users/andre/Documents/MVM/breathing_simulator_test/data/20200426/","C:/Users/andre/Documents/MVM/breathing_simulator_test/data/20200428/","C:/Users/andre/Documents/MVM/breathing_simulator_test/data/20200430/","C:/Users/andre/Documents/MVM/breathing_simulator_test/data/20200501/"]
  dirs=["C:/Users/andre/Documents/MVM/breathing_simulator_test/data/test/"]
  n=len(dirs)
  fig0,ax0=plt.subplots(2,2)
  ax0=ax0.flatten()
  fig0.set_size_inches(15,11)
  maxV=0.0
  for i,d in enumerate(dirs):
    fin=glob.iglob(os.path.join(d, '*.json'))
    for j,f in enumerate(fin):
      if os.path.isfile(f):
        idx=i+j*n
        print(f'Reading {f} as {idx}')
        df=read_jsonfile(fname=f)
        stts=calculate_cycle_info(df)
        timers(df)
        maxV_temp=volumers(df)
        if maxV_temp>maxV: maxV=maxV_temp
        df['fileindex']=idx
        df.plot.line(ax=ax0[idx],x='time',y=['tidal_volume','v_total'],label=['calculatated','readout'])
        #df.plot.line(ax=ax0[idx],x='time',y=['tidal_volume','Deltats_ms_ck'],label=['vol','Dt'])
        ax0[idx].set_xlim(stts[5],stts[5]+30.)
        ax0[idx].set_title(os.path.basename(f))
        frames.append(df)
      else:
        print(f'{f} not a file')
  fig0.tight_layout()
  fig0.savefig('checker_volumes_time.png')

  dfc=pd.concat(frames)

  fig1,ax1=plt.subplots(2,1)
  ax1=ax1.flatten()
  fig1.set_size_inches(15,7)
  dfc.plot.scatter(ax=ax1[0], x='time',y='Dtime')
  ax1[0].set_xlabel('Elapsed Time/s')
  ax1[0].set_ylabel('Delta/s')
  ax1[0].set_title('Readout Period')
  dfc['Dtime'].plot.hist(ax=ax1[1],bins=20)
  ax1[1].set_yscale('log')
  ax1[1].set_xlabel('Delta/s')
  fig1.tight_layout()
  fig1.savefig('checker_Delta_time.png')


  fig2,ax2=plt.subplots(1,1)
  #ax2=ax2.flatten()
  fig2.set_size_inches(15,7)
  #ax2[0]=dfc.plot.scatter(ax=ax2[0],x='time',y='Dts',c='k',label='Delta ts')
  #ax2[0].set_xlabel('Elapsed Time/s')
  #ax2[0].set_ylabel('Delta/ms')

  dfc.plot.scatter(ax=ax2,x='tidal_volume',y='v_total',c='fileindex', cmap='rainbow')
  xpoints=np.linspace(0.0,maxV,10)
  ax2.plot(xpoints,bisec(xpoints),'--',c='Gray')
  ax2.set_xlabel('Calculated Tidal Volume/cl')
  #ax2.set_xlim(0.0,maxV)
  ax2.set_ylabel('Readout Volume/cl')
  fig2.tight_layout()
  fig2.savefig('checker_volVsVol.png')


  plt.show()
