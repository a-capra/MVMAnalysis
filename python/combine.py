import numpy as np
import pandas as pd
import logging as log
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib import colors
from scipy.interpolate import interp1d
import matplotlib.patches as patches
from scipy.signal import find_peaks
import json

from db import *
from mvmio import *
from combine_plot_service_canvases import *
from combine_plot_arXiv_canvases import *
from combine_plot_summary_canvases import *
from combine_plot_mvm_only_canvases import *

# usage
# py combine.py ../Data -p -f VENTILATOR_12042020_CONTROLLED_FR20_PEEP5_PINSP30_C50_R5_RATIO050.txt  --mvm-col='mvm_col_arduino' -d plots_iso_12Apr

def add_timestamp(df, timecol='dt'):
  ''' Add timestamp column assuming constant sampling in time '''
  df['timestamp'] = df['dt'] #np.linspace( df.iloc[0,:][timecol] ,  df.iloc[-1,:][timecol] , len(df) )
  ''' Based on discussions at 2020-04-26 analysis call, check to see of there really is a
  problem with the time stamps and to see how big the shift is. CJJ - 2020-04-26'''
  df['dtcheck'] = df['timestamp']-df['dt']
  max_time_off = df['dtcheck'].max()
  min_time_off = df['dtcheck'].min()
  print("The maximum shifts in timestamp are... ", max_time_off, min_time_off)
  return df

def get_deltat(df, timestampcol='timestamp', timecol='dt'):
  ''' Retrieve sampling time '''
  return df[timestampcol].iloc[2] - df[timestampcol].iloc[1]

def correct_sim_df(df):
  ''' Apply corrections to simulator data '''
  df['total_vol'] = df['total_vol'] * 0.1

def correct_mvm_df(df, pressure_offset=0, pv2_thr=50):
  ''' Apply corrections to MVM data '''
  # correct for miscalibration of pressure sensor if necessary
  df['pressure_pv1'] = df['pressure_pv1']  + pressure_offset
  df['airway_pressure'] = df['airway_pressure']  + pressure_offset

  #set MVM flux to 0 when out valve is open AND when the flux is negative
  #df['flux'] = np.where ( df['out'] > pv2_thr , 0 , df['flux'] )
  #df['flux'] = np.where ( df['flux'] < 0 , 0 , df['flux'] )

def synchronize_first_signals(df, dfhd, threshold_sim, threshold_mvm, diagnostic_plots=False) :
  ''' This is an automated system to give a close syncronization between the MVM and the simulator
  The idea is that the simulator will be at atmosphere until the first time the MVM lets air
  through. Then it will (at least) go to the PEEP value.
  As long as the simulator is turned on before the MVM, this will work.
  Apparently, proper  use of the simulator requires this.
  Chris.Jillings 2020-04-24
  '''
  # Find the times in the dt column of the simulator DataFrame that the pressure is higher than thershold_sim
  dftmp = df[ ( df['airway_pressure']>threshold_sim ) ]
  this_shape = dftmp.shape
  if this_shape[0] <1 :
    log.warning("In synchronize_first_signals no threshold cross was found for simulator. Returning 0. ")
    return 0
  sim_threshold_row = dftmp.iloc[0]
  simulator_time = sim_threshold_row['dt']

  #Now find the MVM time
  dfhdtmp = dfhd[ ( dfhd['airway_pressure']>threshold_mvm ) ]
  this_shape = dfhdtmp.shape
  if this_shape[0] <1 :
    log.warning("In synchronize_first_signals no threshold cross was found for MVM. Returning 0. ")
    return 0
  mvm_threshold_row = dfhdtmp.iloc[0]
  mvm_time = mvm_threshold_row['dt']
  print("The auto-calculated synchronization time shift between MVM and simulator is %.3f seconds"%(simulator_time - mvm_time))
  if diagnostic_plots  :
    dfhd['dtshifted'] = dfhd['dt'] + (simulator_time - mvm_time)
    figdiag, axdiag = plt.subplots(2)
    figdiag.set_size_inches(12,6)
    axdiag[0].set_xlim(-10, simulator_time+20)
    axdiag[1].set_xlim(-10, simulator_time+20)
    dfhd.plot(ax=axdiag[0], kind='line', x='dtshifted', y='airway_pressure', label='MVM pressure', color='red')
    df.plot(ax=axdiag[1], kind='line', x='dt', y='airway_pressure', label='Sim pressure', color='black')
    axdiag[0].grid(True,which='major',axis='x')
    axdiag[1].grid(True,which='major',axis='x')
    plt.show()
  return (simulator_time - mvm_time)


def apply_rough_shift(sim, mvm, manual_offset):
  mvm['dt'] = mvm['timestamp']
  imax1 = mvm[ ( mvm['dt']<8 ) & (mvm['dt']>2) ] ['flux'].idxmax()
  tmax1 = mvm['dt'].iloc[imax1]
  imax2 = sim[  (sim['dt']<8) & (sim['dt']>2)   ] ['total_flow'].idxmax()
  tmax2 = sim['dt'].iloc[imax2]
  shift  = tmax2 - tmax1
  if (manual_offset > 0 ) : print ("...Adding additional manual shift by [s]: ", manual_offset)
  mvm['dt'] = mvm['timestamp'] + shift  + manual_offset # add arbitrary shift to better match data

def apply_manual_shift(sim, mvm, manual_offset):
  mvm['dt'] = mvm['timestamp'] + manual_offset

def apply_good_shift(sim, mvm, resp_rate, manual_offset):
  resp_period = 60./resp_rate

  sim_variable = 'total_flow' #'airway_pressure' #total_flow
  mvm_variable = 'flux'       #'airway_pressure' #flux
  sim['flux'] = np.where(sim[sim_variable]>0, sim[sim_variable], 0)
  sec = sim.dt[1]-sim.dt[0]
  first = sim.dt.to_list()[0]
  last  = sim.dt.to_list()[-1]
  sim_peaks = sim[(sim['dt']>first+5)&(sim['dt']<last-5)]#.rolling(1).mean()
  peak_rows, _ = find_peaks(sim_peaks['flux'].to_list(), prominence=sim_peaks['flux'].max()*0.5, distance=resp_period/sec*0.8)
  if (len(peak_rows)<2) :
    peak_rows, _ = find_peaks(sim_peaks['flux'].to_list(), prominence=sim_peaks['flux'].max()*0.22, distance=resp_period/sec*0.8)

  sim_peaks = sim_peaks.iloc[peak_rows]
  sim_peaks.sort_values(by=['dt'], inplace=True)
  sim_intervals = sim_peaks['dt'].diff().dropna().to_list()

  mvm['dt'] = mvm['timestamp']
  sec = mvm.dt[1]-mvm.dt[0]

  first = mvm.dt.to_list()[0]
  last  = mvm.dt.to_list()[-1]
  mvm_peaks = mvm[(mvm['dt']>first+5)&(mvm['dt']<last-5)]#.rolling(1).mean()
  peak_rows, _ = find_peaks(mvm_peaks[mvm_variable].to_list(), height=mvm_peaks[mvm_variable].max()*0.5, distance=resp_period/sec*0.8)
  if (len(peak_rows)<2) :
    peak_rows, _ = find_peaks(mvm_peaks[mvm_variable].to_list(), height=mvm_peaks[mvm_variable].max()*0.3, distance=resp_period/sec*0.8)
  mvm_peaks = mvm_peaks.iloc[peak_rows]
  mvm_peaks.sort_values(by=['dt'], inplace=True)
  mvm_intervals = mvm_peaks['dt'].diff().dropna().to_list()

  #print ("MVM PEAKS", mvm_peaks['dt'])
  #print ("SIM PEAKS", sim_peaks['dt'])

  mvm_peak_times = mvm_peaks['dt'].to_list()
  sim_peak_times = sim_peaks['dt'].to_list()
  mvm_peak_hgts = mvm_peaks[mvm_variable].to_list()
  sim_peak_hgts = sim_peaks['flux'].to_list()
  print ("I have identified: ", len(mvm_peak_times), len(sim_peak_times))

  central_idx    = 20
  min_difference = 1e7
  tdiff          = 0
  for i in range (9) :
    offset = -4 + i
    subset_sim   = sim_peak_hgts[central_idx-6:central_idx+6]
    subset_mvm   = mvm_peak_hgts[central_idx-6 + offset :central_idx+6 + offset ]
    pdiff  = sum ( [ (x-y)*( x-y) for (x,y) in zip (subset_mvm, subset_sim)  ] )
    if pdiff < min_difference :
      min_difference = pdiff
      print ("minimisig distance: ",i, min_difference, mvm_peak_times[central_idx] - sim_peak_times [central_idx] )
      tdiff =  - np.mean( [ (x-y) for (x,y) in zip (mvm_peak_times, sim_peak_times)  ] )

  mvm_mean = np.nanmean(mvm_intervals)
  sim_mean = np.nanmean(sim_intervals)
  print(np.mean(mvm_intervals), np.mean(sim_intervals))

  interval = mvm_peaks.dt.to_list()[10]-sim_peaks.dt.to_list()[5]

  delay =  tdiff
  #delay = - ( interval - int(interval/mvm_mean)*mvm_mean )

  print('inspiratory rate ',60./mvm_mean, 'cycle / min')
  print('delay ',delay, 's')

  if (abs(manual_offset) > 0 ) : print (".. adding adidtional shift [s]", manual_offset )
  mvm['dt'] += delay + manual_offset

  """
  ax = mvm.plot( x='dt',y='flux')
  sim.plot(ax=ax, x='dt',y='flux')
  ax.plot(sim_peaks['dt'].to_list(),sim_peaks['flux'].to_list(),'x')
  ax.plot(mvm_peaks['dt'].to_list(),mvm_peaks['flux'].to_list(),'x')
  if args.show:
    plt.show()
  """

def add_pv2_status(df):
  df['out_diff'] = df['out'].diff().fillna(0)
  df['out_status'] = np.where(df['out_diff'] == 0, 'steady', np.where(df['out_diff'] < 0, 'closing', 'opening'))

def get_start_times(df, thr=50, dist=0.1, quantity='out', timecol='dt'):
  ''' Get times where a given quantity starts being above threshold
  times_open = df[ ( df[quantity]>thr ) ][timecol]
  start_times = [ float(times_open.iloc[i]) for i in range(0, len(times_open)-1) if times_open.iloc[i+1]-times_open.iloc[i] > 0.1  or i == 0  ]
'''
  start_times = df[df['out_status'] == 'closing']['dt'].unique()
  return start_times

def get_muscle_start_times(df, quantity='muscle_pressure', timecol='dt'):
  ''' True start time of breaths based on muscle_pressure (first negative time) '''
  negative_times = df[ ( df[quantity]<0 ) ][dt]    #array of negative times
  start_times = [ float (negative_times.iloc[i])  for i in range(0,len(negative_times)) if negative_times.iloc[i]-negative_times.iloc[i-1] > 1.  or i == 0  ]   #select only the first negative per bunch
  return start_times

def get_reaction_times(df, start_times, quantity='total_flow', timecol='dt', thr=5):
  ''' Reaction times based on total flow '''
  positive_flow_times = df[ ( df[quantity]>thr ) ][timecol]   # constant threshold to avoid artifact
  #flow_start_time = [ float (positive_flow_times.iloc[i])  for i in range(1,len(positive_flow_times)) if positive_flow_times.iloc[i]-positive_flow_times.iloc[i-1] > 1. ]   #select only the first negative per bunch
  flow_start_time = []

  for st in start_times :
    isFound = False
    for pf in positive_flow_times :
      if pf > st and isFound == False:
        isFound = True
        flow_start_time.append(pf)
    if isFound == False :
      flow_start_time.append(0)

  reaction_times = np.array ([100 * ( ft - st ) for ft,st in zip (flow_start_time,start_times) ])
  reaction_times = np.where ( abs(reaction_times)>1000, 0,reaction_times  )
  return reaction_times

def add_cycle_info(sim, mvm, start_times, reaction_times):
  ''' Add cycle start, reaction time, time-in-breath '''
  '''
  Add integer index information for easier lining up of
  stats at given bin on cycle overlay plots.
  Chris.Jillings@snolab.ca 2020-04-20
  '''
  sim['start'] = 0
  sim['tbreath'] = 0
  sim['reaction_time'] = 0
  for s,f in zip (start_times,reaction_times) :
    times_open = sim[ ( sim.dt>s ) ]['iindex']
    if len(times_open) > 0 :
      this_iindex = times_open.iloc[0]  #The sim data not synchronous to the start_times
    else :  this_iindex = -1
    sim.loc[sim.dt>s,'start']   = s
    sim.loc[sim.dt>s,'tbreath'] = sim.dt - s
    sim.loc[sim.dt>s,'reaction_time'] = f
    sim.loc[sim.dt>s,'siindex'] = this_iindex

  mvm['start'] = 0
  mvm['ncycle']= 0
  for i,s in enumerate(start_times) :
    mvm.loc[mvm.dt>s,  'start' ]   = s
    mvm.loc[mvm.dt>s,  'ncycle']   = i

def add_chunk_info(df):
  ''' Add mean values computed on simulator dataframe chunk '''
  cycle = df.groupby(['start']).mean()

  df['mean_pressure'] = 0

  for i,r in cycle.iterrows():
    df.loc[df.start == i, 'mean_pressure'] = r.total_flow

  df['norm_pressure']  = df['airway_pressure'] - df['mean_pressure']

  df['max_pressure'] = 0
  cycle = df.groupby(['start']).max()
  for i,r in cycle.iterrows():
    df.loc[df.start == i, 'max_pressure'] = r.total_flow

def stats_for_repeated_cycles(adf, variable='total_flow') :
    '''
    This function assumed that the simulator DataFrame has been pre-processed
    to include integer indexing and that the start times for each cycle have been
    calculated. This loops through the given DataFrame and
    1: Checks that there really is a one-to-one correspondence between dtc and
    the integer indexing
    2: Finds the series for the variable in question for a given integer index
    since start of cycle
    3: calculates some basic stats that can be used in plotting
    4: Returns a DataFrame for plotting.
    N.B. This function must be called once for each variable to be plotted.
    ---
    Chris.Jillings@snolab.ca 2020-04-19
    '''
    nstats = 8
    di_series = adf['diindex']
    length = di_series.max() - di_series.min()
    local_stats_array = np.zeros((int(length), nstats),dtype='float64')
    di_arr = np.arange(di_series.min(), di_series.max())
    for i in di_arr:
      this_series = adf.loc[adf.diindex==i, variable]
      # Do some sanity checking here that diindex and dtc track perfectly
      dtcmin = adf[adf.diindex==i]['dtc'].min()
      dtcmax = adf[adf.diindex==i]['dtc'].max()
      if ( (dtcmax-dtcmin)>0.1 ) :
        log.warning("In stats_for_repeated_cycles() the integer step counting and floating-point times are out of sync. Overlay plots may be affected.")
      local_stats_array[int(i-di_series.min())] = [1.0*i, (dtcmax+dtcmin)/2.0, this_series.mean(), this_series.median(), this_series.min(), this_series.max(), this_series.std(), this_series.count()]
    answer = pd.DataFrame(local_stats_array, columns=['diiindex','dtc','mean', 'median', 'min','max', 'std', 'N'] )
    return answer


def add_clinical_values (df, max_R=250, max_C=100) :
  deltaT = get_deltat(df, timestampcol='dt')
  """Add for reference the measurement of "TRUE" clinical values as measured using the simulator"""

  #true resistance
  df['delta_pout_pin']        =  df['airway_pressure'] - df['chamber1_pressure']                  # cmH2O
  df['delta_vol']             = ( df['chamber1_vol'] * 2. ) .diff()                               # ml
  df['airway_resistance']     =  df['delta_pout_pin'] / ( df['delta_vol'] / deltaT/1000. )                 # cmH2O/(l/s)
  df.loc[ (abs(df.airway_resistance)>max_R) | (df.airway_resistance<0) ,"airway_resistance"] = 0

  #true compliance
  df['deltapin']     =  df['chamber1_pressure'].diff()
  df['delta_volin']  =  ( df['chamber1_vol']  + df['chamber2_vol']) . diff()
  df['compliance']   =  df['delta_volin']/df['deltapin']
  df.loc[abs(df.compliance)>max_C,"compliance"] = 0

  #integral of flux - double checks only
  #df['int_vol']      =  1 + df['total_flow'].cumsum() * deltaT / 60. * 100
  #df['ch_vol']       =  -1 + (df['chamber1_vol']  + df['chamber2_vol']) * 0.1
  #df['comp_vol']     =  (df['compressed_vol'] / df['airway_pressure']) * 0.1 * 1000
  #df['ch_vol']       -=  df['ch_vol'] . min()
  #df['comp_vol']     -=  df['comp_vol'] . min()


def measure_clinical_values(df, start_times):
  ''' Compute tidal volume and other clinical quantities for MVM data '''
  # TODO: "tolerances", i.e. pre-t0 and post-t1, are currently hardcoded
  deltaT = get_deltat(df)
  #substitute the fixed deltaT with sample-dependent dt after removal of linspace
  df['flux_x_dt']  = df['flux'] * df['dt'].diff()

  # determine inspiration end times
  inspiration_end_times   = df[df['out_status'] == 'opening']['dt'].unique()
  inspiration_start_times = start_times

  if (inspiration_end_times[0]< inspiration_start_times[0]) :
    inspiration_end_times = inspiration_end_times[1:]

# if df['out'].iloc[0] > 0: # if PV2 is open at beginning of considered dataset
#   tmp = [0]
#   for x in inspiration_end_times: tmp.append(x)
#   inspiration_end_times = tmp

  # propagate variables to all samples in each inspiration
  # i.e. assign to all samples in each cycle the same value of something
  df['is_inspiration'] = 0
  df['cycle_tidal_volume'] = 0
  df['cycle_peak_pressure'] = 0
  df['cycle_plateau_pressure'] = 0
  df['cycle_PEEP'] = 0
  df['cycle_Cdyn'] = 0
  df['cycle_Cstat'] = 0
  inspiration_durations = []

  for i,(s,v) in enumerate(zip(inspiration_start_times, inspiration_end_times)):

    if i>=len(inspiration_start_times)-1 : continue

    this_inspiration   = (df.dt>s) & (df.dt<v) # all samples
    first_sample       = (df.dt == s) # start of inspiration
    last_sample        = (df.dt == v) # beginning of expiration
    next_inspiration_t = inspiration_start_times[i+1]
    end_of_inspiration_t = v
    inspiration_durations.append (v-s)

    df.loc[this_inspiration, 'is_inspiration'] = 1
    #measured inspiration
    df.loc[ ( df.dt >  s ) & ( df.dt < next_inspiration_t ), 'cycle_tidal_volume']     = df[  ( df.dt >  s ) & ( df.dt < end_of_inspiration_t) ]['flux_x_dt'].sum() / 60. * 100
    df.loc[this_inspiration, 'cycle_peak_pressure']    = df[ this_inspiration ]['airway_pressure'].max()
    df.loc[this_inspiration, 'cycle_plateau_pressure'] = df[ this_inspiration & ( df.dt > v - 51e-3 ) & ( df.dt < v-10e-3 ) ]['airway_pressure'].mean()
    #not necessarily measured during inspiration
    df.loc[this_inspiration, 'cycle_PEEP']             = df[ ( df.dt > next_inspiration_t - 51e-3 ) & ( df.dt < next_inspiration_t) ] ['airway_pressure'].mean()
    #print ("cycle_peak_pressure: " , df[ this_inspiration &( df.dt > v - 20e-3 ) & ( df.dt < v-10e-3 ) ]['airway_pressure'].mean() )

  respiration_rate     = 60/np.mean(np.gradient (start_times))
  inspiration_duration = np.mean(inspiration_durations)
  # create cumulative variables (which get accumulated during a cycle)
  df['tidal_volume'  ] = 0

  for c in df['start'].unique():
    cycle_data = df[df['start']==c]
    cum = cycle_data['flux_x_dt'].cumsum()
    df.loc[df.start == c, 'tidal_volume'] = cum

  # correct flow sum into volume (aka flow integral)
  df['tidal_volume']   *= 1./60.*100

  # set inspiration-only variables to zero outside the inspiratory phase
  df['tidal_volume'] *= df['is_inspiration']

  # calculate compliance and residence
  df['compliance'] = 0
  df['resistance'] = 0

  for i,(s,v) in enumerate(zip(inspiration_start_times, inspiration_end_times)):

    this_inspiration   = (df.dt>s + 0.5 * ( v-s) ) & (df.dt<v)     #
    df.loc[this_inspiration, 'cycle_peak_pressure']

    df.loc[this_inspiration, 'vol_over_flux']           =  df['tidal_volume']*10. / df['flux']                                         # in [ml/(l/min)]
    df.loc[this_inspiration,'delta_p_over_flux']       =  ( df['airway_pressure']-df['cycle_PEEP'])/ df['flux']                       # in [cmH2O/(l/min)]
    df.loc[this_inspiration,'delta_vol_over_flux']     =  df.loc[this_inspiration,'vol_over_flux'].diff(periods=4).rolling(4).mean()                                         # in [ml/(l/min)]
    df.loc[this_inspiration,'delta_delta_p_over_flux'] =  df['delta_p_over_flux'].diff(periods=4).rolling(4).mean()                                    # in [cmH2O/(l/min)]

    df.loc[this_inspiration,'compliance']   =  df['delta_vol_over_flux']/ df['delta_delta_p_over_flux']                               # in ml/cmH2O
    df.loc[this_inspiration,'resistance']   =  (df['delta_p_over_flux']  - df['vol_over_flux']/df['compliance'])*60.                  # in cmH2O/(l/s)

    df.loc[this_inspiration,'compliance'] = np.where( abs ( df[(this_inspiration)]['compliance'] )  > 300, 0, df[(this_inspiration)]['compliance'])
    df.loc[this_inspiration,'resistance'] = np.where( abs ( df[(this_inspiration)]['resistance'] )  > 300, 0, df[(this_inspiration)]['resistance'])

    df.loc[this_inspiration,'compliance'] = np.mean(df.loc[this_inspiration]['compliance'])
    df.loc[this_inspiration,'resistance'] = np.mean(df.loc[this_inspiration]['resistance'])

    df.loc[this_inspiration,'compliance'] *= df['is_inspiration']
    df.loc[this_inspiration,'resistance'] *= df['is_inspiration']

  return respiration_rate, inspiration_duration

def get_IoverEAndFrequency (dftest, quantity, inhaleTr, exhaleTr, inverted=False):
  '''
  Computer I:E and 1/periode for every breath cycle, for summary plot
  We should look for
  1) First up-going, positive flow record, start of inhalation
    1.1) Because of some small wiggles in the data (mostly the flux from the MVM), we are using a 2-threshold system
    1.2) We require that we pass a high trigger  set to be close to the maximum flow
    1.3) Then we backtrack to see when we passed a low triggger, which will be the start of the inhalation.
  2) First down going, negative flow, end of inhalation, start of exhalation
  3) Flow getting back to zero

  I am not sure if that is better to work on the flow itself or its derivative. Let's try with the flow, we will used the derivative if it does not work.
  '''
  tFlow =  dftest[quantity]
  time = dftest['dt']

  startIn = 0.0
  stopIn = 0.0
  startEx = 0.0
  stopEx = 0.0

  dtInhalate = 0.0
  dtExhalate = 0.0
  frequency = 0.0
  IoverE = 0.0

  mIoverE = []
  mFrequency = []

  inInhalation = False
  inExhalation = False

  correction=1.0  # Some signals like "out" need to be inverted to be used with this treshold search
  if inverted:
    correction = -1.0

  for i,(f,t) in enumerate(zip(tFlow, time)):
    if inInhalation == False: # if we are not inhaling, we look for the start
      if f*correction > inhaleTr:    # Passed the threshold, we are now inhaling
        if inExhalation == True: # if we were exhaling previously, that's the end of it, as well as the end of the breath
          stopEx = t
          inExhalation = False
          # We can calculate the variables
          dtInhalate = stopIn - startIn
          dtExhalate = stopEx - startEx # It cannot be zero as stopEx will always come 1 iteration later.
          frequency = 1./(dtInhalate + dtExhalate)*60. # We want breath/min and the units are in seconds
          IoverE= dtInhalate/dtExhalate
          mIoverE.append(IoverE)
          mFrequency.append(frequency)
        inInhalation = True
        startIn = t
    else:
      if f*correction< exhaleTr:  # Passed the threshold, we are now exhaling. Is it possible that we were not inhaling before?.
        inInhalation = False
        stopIn = t
        inExhalation = True
        startEx = t

  return mIoverE, mFrequency

def add_run_info(df, dist=25):
  ''' Add run info based on max pressure '''
  df['mean_max_pressure'] = df['max_pressure'].rolling(4).mean()

  df['run'] = abs(df['mean_max_pressure'].diff())
  df['run'] = np.where(df['run'] > 2, 100, 0)

  starts  = df[df.run > 0]['dt'].values

  # add run number
  df['run'] = -1
  cc = 1
  for i in range(0,len(starts)-1):
    if starts[i+1] - starts[i] > dist:
      df.loc[(df['dt'] >= starts[i]) & (df['dt'] < starts[i+1]), 'run'] = cc
      cc += 1

  df['run'] = df['run']*10


def process_run(conf, ignore_sim=False, auto_sync_debug=False):
  objname = conf["objname"]
  meta = conf["meta"]

  # retrieve simulator data
  if not ignore_sim:
    df = get_simulator_df(conf["fullpath_rwa"], conf["fullpath_dta"])
    # Error checking in case rwa and dta data do not jive
    if df.shape[0] == 0 :
      log.error("Simulator data is empty for this test. Abort process_run.")
      return {}
  else:
    print ("I am ignoring the simulator")

  # retrieve MVM data
  if conf["json"]:
    dfhd = get_mvm_df_json(fname=conf["fullpath_mvm"])
  else:
    dfhd = get_mvm_df(fname=conf["fullpath_mvm"], sep=conf["mvm_sep"], configuration=conf["mvm_col"])

  add_timestamp(dfhd)

  # apply corrections
  correct_mvm_df(dfhd, conf["pressure_offset"])

  auto_sync = True # default to auto-synchronization
  if not ignore_sim :
    correct_sim_df(df)
    if( conf["offset"]!=0. ) :
      auto_sync = False

    if auto_sync:
      time_shift = synchronize_first_signals(df, dfhd, 4, 4, auto_sync_debug)
      apply_manual_shift(sim=df, mvm=dfhd, manual_offset=time_shift)   #manual version, -o option from command line
    else:
      #add time shift
      apply_manual_shift(sim=df, mvm=dfhd, manual_offset=conf["offset"])   #manual version, -o option from command line
    #apply_rough_shift(sim=df, mvm=dfhd, manual_offset=conf["offset"])   #rough version, based on one pair of local maxima of flux
    #apply_good_shift(sim=df, mvm=dfhd, resp_rate=meta[objname]["Rate respiratio"], manual_offset=conf["offset"])  #more elaborate alg, based on matching several maxima

  ##################################
  # cycles
  ##################################

  # add PV2 status info
  add_pv2_status(dfhd)

  # compute cycle start
  # start_times = get_muscle_start_times(df) # based on muscle pressure
  start_times    = get_start_times(dfhd) # based on PV2
  #  start_times    = get_start_times(df, thr=8, quantity='total_flow', timecol='dt')

  if ignore_sim :
    # stop here if sim is ignored
    return {
        "mvm" : dfhd,
        "start_times" : start_times
        }

  reaction_times = get_reaction_times(df, start_times)
  # Add some integer indexing fields for convenience in stats summary for
  # overlap plots
  # iindex is an integer index corresponding to a "bin number" for dt
  # after add_cylce_index is called siindex will correspond to df.start
  # diindex will correspond to dtc
  # Chris.Jillings
  this_shape = df.shape
  df['iindex'] = np.arange(this_shape[0])
  df['siindex'] = np.zeros(this_shape[0])
  df['diindex'] = np.zeros(this_shape[0])


  # add info
  add_cycle_info(sim=df, mvm=dfhd, start_times=start_times, reaction_times=reaction_times)
  dfhd['dtc'] = dfhd['dt'] - dfhd['start']
  df['dtc'] = df['dt'] - df['start']
  df['diindex'] = df['iindex'] - df['siindex']

  ##################################
  # chunks
  ##################################

  add_chunk_info(df)

  # compute tidal volume etc
  add_clinical_values(df)
  respiration_rate, inspiration_duration = measure_clinical_values(dfhd, start_times=start_times)

  #thispeep  = [ dfhd[dfhd.ncycle==i]['cycle_PEEP'].iloc[0] for
  measured_peeps      = []
  measured_volumes    = []
  measured_peaks      = []
  measured_plateaus   = []
  real_tidal_volumes  = []
  real_plateaus       = []

  # computer the duration of the inhalation over the duration of the exhalation for every breath, as well as the frequency of everybreath (1/period)
  # first for the MVM
  measured_IoverE, measured_Frequency = get_IoverEAndFrequency(dfhd, 'out', -5., -5, True) # "out" needs to be read in inverted logic. The threshold low is -5 for inhalation, -5 for exhalation (going the other way)
  # The first cycle is always bad, removew it
  if (len(measured_IoverE)):
    del measured_IoverE[0]
  if (len(measured_Frequency)):
    del measured_Frequency[0]

  # second for the simulator
  real_IoverE, real_Frequency = get_IoverEAndFrequency(df, 'total_flow', -0.5, -5) # the threshold is -0.5 for the total flow in inhalation (catching the small step), -5 in exhalation (quick inversion of flow)
  # The first cycle is always bad, removew it
  if (len(real_IoverE)):
    del real_IoverE[0]
  if (len(real_Frequency)):
    del real_Frequency[0]

  for i,nc in enumerate(dfhd['ncycle'].unique()) :

    this_cycle              = dfhd[ dfhd.ncycle==nc ]
    this_cycle_insp         = this_cycle[this_cycle.is_inspiration==1]

    if len(this_cycle_insp)<1 : continue
    cycle_inspiration_end   = this_cycle_insp['dt'].iloc[-1]

    if i > len(dfhd['ncycle'].unique()) -3 : continue
    #compute tidal volume in simulator df
    subdf             = df[ (df.dt>start_times[i]) & (df.dt<start_times[i+1]) ]
    real_tidal_volume = ( subdf['total_vol'] - subdf['total_vol'].min() ).max()
    #compute plateau in simulator
    subdf             = df[ (df.dt>start_times[i]) & (df.dt<cycle_inspiration_end) ]
    real_plateau      = subdf[ (subdf.dt > cycle_inspiration_end - 20e-3) ]['airway_pressure'].mean()
    #this_cycle_insp[(this_cycle_insp['dt'] > start_times[i] + inspiration_duration - 20e-3) & (this_cycle_insp['dt'] < start_times[i] + inspiration_duration - 10e-3)]['airway_pressure'].mean()
    real_tidal_volumes.append(real_tidal_volume)
    real_plateaus.append (real_plateau)

    measured_peeps.append(  this_cycle['cycle_PEEP'].iloc[0])
    measured_volumes.append(this_cycle['cycle_tidal_volume'].iloc[0])
    measured_peaks.append(   this_cycle['cycle_peak_pressure'].iloc[0])
    measured_plateaus.append(this_cycle['cycle_plateau_pressure'].iloc[0])

  """
  print ("measured_peeps", measured_peeps)
  print ("measured_volumes",measured_volumes)
  print ("measured_peaks",measured_peaks)
  print ("measured_plateau",measured_plateau)

  #plt.plot(dfhd['dt'], dfhd['out'], label='control out')
  plt.plot(dfhd['dt'], dfhd['airway_pressure'], label='pressure')
  plt.plot(dfhd['dt'], dfhd['is_inspiration'], label='is_inspiration')
  #plt.plot(dfhd['dt'], dfhd['cycle_tidal_volume'], label='cycle_tidal_volume')
  #plt.plot(dfhd['dt'], dfhd['cycle_PEEP'], label='cycle_PEEP')
  #plt.plot(dfhd['dt'], dfhd['tidal_volume'], label='tidal_volume')
  #plt.plot(dfhd['dt'], dfhd['cycle_tidal_volume'], label='cycle_tidal_volume')

  #plt.plot(df['dt'], df[''], label='')
  #plt.plot(df['dt'], df['tracheal_pressure'], label='true tracheal_pressure')
  #plt.plot(df['dt'], df['airway_pressure'], label='true airway_pressure')
  #plt.plot(df['dt'], df['chamber1_pressure'], label='true ch1_pressure')
  #plt.plot(df['dt'], df['chamber2_pressure'], label='true ch2_pressure')
  plt.plot(df['dt'], df['airway_resistance'], label='true airway_resistance')
  plt.plot(df['dt'], df['compliance'], label='true compliance')
  #plt.plot(dfhd['dt'], dfhd['compliance'], label='meas compliance')

  plt.legend()
  if args.show:
    plt.show()
  print('Press ENTER to continue to real plots')
  input()
  """

  #raise RuntimeError('Valerio')

  ##################################
  # find runs
  ##################################

  add_run_info(df)

  ##################################
  # Make data frames for statistics on overlayed cycles
  ##################################
  my_selected_cycle = meta[objname]['cycle_index']
  cycles_to_show = 30
  dftmp = df[ (df['start'] >= start_times[ my_selected_cycle ] ) & ( df['start'] < start_times[ min ([my_selected_cycle + cycles_to_show, len(start_times)-2] )  ])]
  stats_total_vol = stats_for_repeated_cycles(dftmp, 'total_vol')
  stats_total_flow = stats_for_repeated_cycles(dftmp, 'total_flow')
  stats_airway_pressure = stats_for_repeated_cycles(dftmp, 'airway_pressure')

  # keep the following in sync wiht the data dict read by plot_run
  return {
      "sim" : df,
      "sim_trunc" : dftmp,
      "mvm" : dfhd,
      "start_times" : start_times,
      "reaction_times" : reaction_times,
      "respiration_rate" : respiration_rate,
      "inspiration_duration" : inspiration_duration,
      "measured_peeps" : measured_peeps,
      "measured_volumes" : measured_volumes,
      "measured_peaks" : measured_peaks,
      "measured_plateaus" : measured_plateaus,
      "measured_IoverE" : measured_IoverE,
      "measured_Frequency" : measured_Frequency,
      "real_tidal_volumes" : real_tidal_volumes,
      "real_plateaus" : real_plateaus,
      "real_IoverE" : real_IoverE,
      "real_Frequency" : real_Frequency,
      "stats_total_vol" : stats_total_vol,
      "stats_total_flow" : stats_total_flow,
      "stats_airway_pressure" : stats_airway_pressure
      }


def plot_run(data, conf, args):
  colors = {  "muscle_pressure": "#009933"  , #green
    "sim_airway_pressure": "#cc3300" ,# red
    "total_flow":"#ffb84d" , #
    "tidal_volume":"#ddccff" , #purple
    "total_vol":"pink" , #
    "reaction_time" : "#999999", #
    "pressure" : "black" , #  blue
    "vent_airway_pressure": "#003399" ,# blue
    "flux" : "#3399ff" #light blue
  }

  meta = conf["meta"]
  objname = conf["objname"]

  # stop here if sim is ignored
  if args.ignore_sim :
    if args.plot :
      plot_mvm_only_canvases(data["mvm"], meta, objname, args.output_directory, data["start_times"], colors, args.figure_format, args.web)
      print ("Quitting due to ignore_sim")
      if args.show:
        plt.show()
    return

  # keep the following in sync with the dict returned by process_run
  try:
    df = data["sim"]
  except KeyError:
    log.error("Simulator data frame not available in plot_run for this test, exit")
    return
  dftmp = data["sim_trunc"]
  dfhd = data["mvm"]
  start_times = data["start_times"]
  reaction_times = data["reaction_times"]
  respiration_rate = data["respiration_rate"]
  inspiration_duration = data["inspiration_duration"]
  measured_peeps = data["measured_peeps"]
  measured_volumes = data["measured_volumes"]
  measured_peaks = data["measured_peaks"]
  measured_plateaus = data["measured_plateaus"]
  measured_IoverE = data["measured_IoverE"]
  measured_Frequency = data["measured_Frequency"]
  real_tidal_volumes = data["real_tidal_volumes"]
  real_plateaus = data["real_plateaus"]
  real_IoverE = data["real_IoverE"]
  real_Frequency = data["real_Frequency"]
  stats_total_vol = data["stats_total_vol"]
  stats_total_flow = data["stats_total_flow"]
  stats_airway_pressure = data["stats_airway_pressure"]

  ##################################
  # saving and plotting
  ##################################
  if args.save:
    df.to_hdf(f'{objname}.sim.h5', key='simulator')
    dftmp.to_hdf(f'{objname}.sim.h5', key='simulator_truncated')
    dfhd.to_hdf(f'{objname}.mvm.h5', key='MVM')
    stats_total_vol.to_hdf(f'{objname}.sim.h5', key='stats_total_vol')
    stats_total_flow.to_hdf(f'{objname}.sim.h5', key='stats_total_flow')
    stats_airway_pressure.to_hdf(f'{objname}.sim.h5', key='stats_airway_pressure')

  if args.plot :
    ####################################################
    '''choose here the name of the MVM flux variable to be shown in arXiv plots'''
    ####################################################
    dfhd['display_flux'] = dfhd['flux']

    ####################################################
    '''general service canavas'''
    ####################################################
    plot_service_canvases (df, dfhd, meta, objname, args.output_directory, start_times, colors, args.figure_format, args.web, respiration_rate, inspiration_duration)

    ####################################################
    '''formatted plots for ISO std / arXiv. Includes 3 view plot and 30 cycles view'''
    ####################################################
    plot_arXiv_canvases (df, dfhd, meta, objname, args.output_directory, start_times, colors, args.figure_format, args.web)

    ## For the moment only one test per file is supported here
    ## correct for outliers ?  no, we need to see them
    measured_peeps      = measured_peeps[3:-3]
    measured_plateaus   = measured_plateaus[3:-3]
    measured_peaks      = measured_peaks[3:-3]
    measured_volumes    = measured_volumes[3:-3]
    real_plateaus       = real_plateaus [3:-3]
    real_tidal_volumes  = real_tidal_volumes[3:-3]

    mean_peep      = np.mean(measured_peeps)
    mean_plateau   = np.mean(measured_plateaus)
    mean_peak      = np.mean(measured_peaks)
    mean_volume    = np.mean(measured_volumes)
    mean_iovere    = np.mean(measured_IoverE)
    mean_frequency = np.mean(measured_Frequency)
    rms_peep       = np.std(measured_peeps)
    rms_plateau    = np.std(measured_plateaus)
    rms_peak       = np.std(measured_peaks)
    rms_volume     = np.std(measured_volumes)
    rms_iovere     = np.std(measured_IoverE)
    rms_frequency  = np.std(measured_Frequency)
    max_peep       = np.max(measured_peeps)
    max_plateau    = np.max(measured_plateaus)
    max_peak       = np.max(measured_peaks)
    max_volume     = np.max(measured_volumes)
    max_iovere     = np.max(measured_IoverE)
    max_frequency  = np.max(measured_Frequency)
    min_peep       = np.min(measured_peeps)
    min_plateau    = np.min(measured_plateaus)
    min_peak       = np.min(measured_peaks)
    min_volume     = np.min(measured_volumes)
    min_iovere     = np.min(measured_IoverE)
    min_frequency  = np.min(measured_Frequency)

    #simulator values
    simulator_plateaus = np.array(real_plateaus)
    simulator_plateaus = simulator_plateaus[~np.isnan(simulator_plateaus)]
    simulator_plateau  = np.mean(simulator_plateaus)

    simulator_volumes = np.array(real_tidal_volumes)
    simulator_volumes = simulator_volumes[~np.isnan(simulator_volumes)]
    simulator_volume  = np.mean(simulator_volumes)

    simulator_ioveres = np.array(real_IoverE)
    simulator_ioveres = simulator_ioveres[~np.isnan(simulator_ioveres)]
    simulator_iovere  = np.mean(simulator_ioveres)

    simulator_frequencys = np.array(real_Frequency)
    simulator_frequencys = simulator_frequencys[~np.isnan(simulator_frequencys)]
    simulator_frequency = np.mean(simulator_frequencys)

    meta[objname]["mean_peep"]         =  mean_peep
    meta[objname]["rms_peep"]          =  rms_peep
    meta[objname]["max_peep"]          =  max_peep
    meta[objname]["min_peep"]          =  min_peep
    meta[objname]["mean_plateau"]      =  mean_plateau
    meta[objname]["rms_plateau"]       =  rms_plateau
    meta[objname]["max_plateau"]       =  max_plateau
    meta[objname]["min_plateau"]       =  min_plateau
    meta[objname]["mean_peak"]         =  mean_peak
    meta[objname]["rms_peak"]          =  rms_peak
    meta[objname]["max_peak"]          =  max_peak
    meta[objname]["min_peak"]          =  min_peak
    meta[objname]["mean_volume"]       =  mean_volume
    meta[objname]["rms_volume"]        =  rms_volume
    meta[objname]["max_volume"]        =  max_volume
    meta[objname]["min_volume"]        =  min_volume
    meta[objname]["mean_iovere"]       =  mean_iovere
    meta[objname]["rms_iovere"]        =  rms_iovere
    meta[objname]["max_iovere"]        =  max_iovere
    meta[objname]["min_iovere"]        =  min_iovere
    meta[objname]["mean_frequency"]    =  mean_frequency
    meta[objname]["rms_frequency"]     =  rms_frequency
    meta[objname]["max_frequency"]     =  max_frequency
    meta[objname]["min_frequency"]     =  min_frequency
    meta[objname]["simulator_volume"]  =  simulator_volume
    meta[objname]["simulator_plateau"] =  simulator_plateau
    meta[objname]["simulator_iovere"]  =  simulator_iovere
    meta[objname]["simulator_frequency"]  =  simulator_frequency

    ####################################################
    '''summary plots of measured quantities and avg wfs'''
    ####################################################
    plot_summary_canvases (df, dfhd, meta, objname, args.output_directory, start_times, colors, args.figure_format, args.web, measured_peeps, measured_plateaus, real_plateaus, measured_peaks, measured_volumes, measured_IoverE, measured_Frequency, real_tidal_volumes, real_IoverE, real_Frequency)

    ####################################################
    '''overlay the cycles and shows consistency of simulator readings from cycle to cycle'''
    ####################################################
    plot_overlay_canvases (dftmp, dfhd, meta, objname, args.output_directory, start_times, colors, args.figure_format, args.web, stats_total_vol, stats_total_flow, stats_airway_pressure)

    ####################################################
    '''dump summary data in json file, for get_tables'''
    ####################################################
    filepath = "%s/summary_%s_%s.json" % (args.output_directory, meta[objname]['Campaign'],objname.replace('.txt', '')) # TODO: make sure it is correct, or will overwrite!
    json.dump( meta[objname], open(filepath , 'w' ) )

    ####################################################
    '''show then close figures for this run'''
    ####################################################
    if args.show:
      plt.show()
    plt.close('all')


if __name__ == '__main__':
  import argparse
  import matplotlib
  import style

  parser = argparse.ArgumentParser(description='repack data taken in continuous mode')
  parser.add_argument("input", help="name of the MVM input files (.txt)", nargs='+')
  parser.add_argument("-d", "--output-directory", type=str, help="name of the output directory for plots", default="plots_iso")
  parser.add_argument("-i", "--ignore_sim", action='store_true',  help="ignore simulator")
  parser.add_argument("-skip", "--skip_files", type=str,  help="skip files", nargs='+', default="")
  parser.add_argument("-p", "--plot", action='store_true', help="make and save plots")
  parser.add_argument("-show", "--show", action='store_true', help="show plots")
  parser.add_argument("-s", "--save", action='store_true', help="save HDF")
  parser.add_argument("-w", "--web", action='store_true', help="enable web display figure path")
  parser.add_argument("-f", "--filename", type=str, help="single file to be processed", default='.')
  parser.add_argument("-c", "--campaign", type=str, help="single campaign to be processed", default="")
  parser.add_argument("-json", action='store_true', help="read json instead of csv")
  parser.add_argument("-o", "--offset", type=float, help="offset between vent/sim", default='0.')
  parser.add_argument("-a", "--automatic_sync", action='store_true', help="displays auto-sync diagnostics plot")
  parser.add_argument("--pressure-offset", type=float, help="pressure offset", default='0.')
  parser.add_argument("--figure-format", type=str, help="format for output figures", default='png')
  parser.add_argument("--cnaf", action='store_true', help="overrides db-google-id to use the CNAF spreadsheet")
  parser.add_argument("--db-google-id", type=str, help="name of the Google spreadsheet ID for metadata", default=default_db_google_id)
  parser.add_argument("--db-range-name", type=str, help="name of the Google spreadsheet range for metadata", default="20200412 ISO!A2:AZ")
  parser.add_argument("--mvm-sep", type=str, help="separator between datetime and the rest in the MVM filename", default="->")
  parser.add_argument("--mvm-col", type=str, help="columns configuration for MVM acquisition, see mvmio.py", default="mvm_col_arduino")
  args = parser.parse_args()

  if args.cnaf:
    args.db_google_id = cnaf_db_google_id
    print ("Using the CNAF metadata spreadsheet")
  elif args.db_google_id == default_db_google_id:
    print ("Using the default metadata spreadsheet")
  else:
    print (f"Using the metadata spreadsheet {args.db_google_id}")

  conf = {
      "json" : args.json,
      "offset" : args.offset,
      "pressure_offset" : args.pressure_offset,
      "mvm_sep" : args.mvm_sep,
      "mvm_col" : args.mvm_col
      }

  # determine site name from spreadsheet tab name
  sitename = args.db_range_name.split('!')[0]
  #FIXME in spreadsheet: workaround for Elemaster data, assuming no tab name from another site contains 'ISO'
  if 'ISO' in sitename:
    sitename = "Elemaster"
  if not sitename:
    sitename = "UnknownSite"
  print(f'Analyzing data from {sitename}')

  filenames = []  #if the main argument is a json, skip the direct spreadsheet reader
  if args.input[0].split('.')[-1]== 'json' :

    for input in args.input :
      meta  = read_meta_from_spreadsheet_json (input)
      objname = list ( meta.keys()) [0]
      validate_meta(meta[objname])
      meta[objname]['SiteName'] = sitename
      basedir = '/'.join ( input.split('/')[0:-1] )
      fullpath_rwa = "%s/%s"%( basedir,meta[objname]['RwaFileName'] )
      fullpath_dta = "%s/%s"%( basedir,meta[objname]['DtaFileName'] )
      fname        = "%s/%s"%( basedir,meta[objname]['MVM_filename'] )
      filenames.append(fname)
      conf["meta"] = meta
      conf["objname"] = objname
      conf["fullpath_mvm"] = fname
      conf["fullpath_rwa"] = fullpath_rwa
      conf["fullpath_dta"] = fullpath_dta
      data = process_run(conf=conf, ignore_sim=args.ignore_sim, auto_sync_debug=args.automatic_sync)
      plot_run(data, conf, args)


  else :
    # take only the first input as data folder path
    input = args.input[0]
    # else read metadata spreadsheet
    print("About to read spreadsheet...")
    df_spreadsheet = read_online_spreadsheet(spreadsheet_id=args.db_google_id, range_name=args.db_range_name)

    # if option -n, select only one test
    if len ( args.filename )  > 2 :
      unreduced_filename = args.filename.split("/")[-1]
      reduced_filename = '.'.join(unreduced_filename.split('.')[:])

      print ( "Selecting only: ", reduced_filename )
      df_spreadsheet = df_spreadsheet[ ( df_spreadsheet["MVM_filename"] == unreduced_filename )  ]

    filenames = df_spreadsheet['MVM_filename'].unique()
    if not filenames.size > 0:
      log.error("No valid file name found in selected metadata spreadsheet range")

    for filename in filenames:
      # continue if there is no filename
      if not filename:
        continue

      # read the metadata and create a dictionary with relevant info
      meta  = read_meta_from_spreadsheet (df_spreadsheet, filename)

      objname = f'{filename}_0'   # at least first element is always there
      meta[objname]['SiteName'] = sitename

      # compute the file location: local folder to the data repository + compaign folder + filename
      fname = f'{input}/{meta[objname]["Campaign"]}/{meta[objname]["MVM_filename"]}'

      # detect whether input file is txt or json
      print()
      if fname.endswith(".txt"):
        # here json argument should be False
        if args.json:
          print("txt input file detected, setting args.json = False")
          args.json = False
      elif fname.endswith(".json"):
        # here json argument should be True
        if not args.json:
          print("json input file detected, setting args.json = True")
          args.json = True
      else:
        # if the file name does not end in .txt or .json, try adding an extension based on argument json
        if args.json:
          print("args.json is True, adding extra .json to file name")
          fname = f'{fname}.json'
        else:
          print("args.json is False, adding extra .txt to file name")
          fname = f'{fname}.txt'

      # print file name, then check whether it should be skipped
      print(f'File name {fname}')
      if fname.split('/')[-1] in args.skip_files:
        print('    ... skipped')
        continue

      if args.campaign:
        if args.campaign not in fname:
          print(f'    ... not in selected campaign {args.campaign}')
          continue

      # validate metadata for this run
      validate_meta(meta[objname])

      # determine RWA and DTA data locations
      fullpath_rwa = f'{input}/{meta[objname]["Campaign"]}/{meta[objname]["SimulatorFileName"]}'

      if fullpath_rwa.endswith('.dta'):
        fullpath_rwa =  fullpath_rwa[:-4]      #remove extension if dta
      if not fullpath_rwa.endswith('.rwa'):
        fullpath_rwa =  f'{fullpath_rwa}.rwa'  #if .rwa extension not present, add it

      fullpath_dta = fullpath_rwa.replace('rwa', 'dta')
      print(f'will retrieve RWA and DTA simulator data from {fullpath_rwa} and {fullpath_dta}')

      # run
      conf["meta"] = meta
      conf["objname"] = objname
      conf["fullpath_mvm"] = fname
      conf["fullpath_rwa"] = fullpath_rwa
      conf["fullpath_dta"] = fullpath_dta
      data = process_run(conf=conf, ignore_sim=args.ignore_sim, auto_sync_debug=args.automatic_sync)
      plot_run(data, conf, args)
