import pandas as pd
import matplotlib.pyplot as plt

from mvmio import *
from combine import *
from simple_plots import *

columns_rwa = ['dt',
    'airway_pressure',
    'muscle_pressure',
    'tracheal_pressure',
    'chamber1_vol',
    'chamber2_vol',
    'total_vol',
    'chamber1_pressure',
    'chamber2_pressure',
    'breath_fileno',
    'aux1',
    'aux2',
    'oxygen'
  ]
  
columns_dta = ['breath_no',
    'compressed_vol',
    'airway_pressure',
    'muscle_pressure',
    'total_vol',
    'total_flow',
    'chamber1_pressure',
    'chamber2_pressure',
    'chamber1_vol',
    'chamber2_vol',
    'chamber1_flow',
    'chamber2_flow',
    'tracheal_pressure',
    'ventilator_vol',
    'ventilator_flow',
    'ventilator_pressure',
  ]

fullpath_rwa = "C:/Users/andre/Documents/MVM/breathing_simulator_test/20200408/rawdata/passive_C20R6_RR15_Pins30_It1.5_PEEP7.rwa"
fullpath_dta = fullpath_rwa.replace('rwa', 'dta')

fname="VENTILATOR_CONTROLLED_FR20_PEEP5_PINSP30_C20_R5_RATIO0.50.txt"
input_mvm = "C:/Users/andre/Documents/MVM/breathing_simulator_test/data_analysis/data/Run_12_Apr_8/"+fname
mvm_sep = " -> "
mvm_columns = "mvm_col_arduino"

pressure_offset=0
manual_offset=0

output_directory="C:/Users/andre/Documents/MVM/breathing_simulator_test/data_analysis/simple_plots"
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

sett = { 'C':20,
'R':6,
'RR':15,
'PIP':30,
'PEEP':7,
'RF':0.5}


if __name__ == '__main__':

  # retrieve simulator data
  df = get_simulator_df(fullpath_rwa, fullpath_dta, columns_rwa, columns_dta)

  # retrieve MVM data
  dfhd = get_mvm_df(fname=input_mvm, sep=mvm_sep, configuration=mvm_columns)
  add_timestamp(dfhd)

  # apply corrections
  correct_mvm_df(dfhd, pressure_offset)
  correct_sim_df(df)

  apply_good_shift(sim=df, mvm=dfhd, resp_rate=sett['RR'], manual_offset=manual_offset)

  ##################################
  # cycles
  ##################################

  # add PV2 status info
  add_pv2_status(dfhd)

  # compute cycle start
  # start_times = get_muscle_start_times(df) # based on muscle pressure
  start_times    = get_start_times(dfhd) # based on PV2
  reaction_times = get_reaction_times(df, start_times)

  # add info
  add_cycle_info(sim=df, mvm=dfhd, start_times=start_times, reaction_times=reaction_times)

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

  for i,nc in enumerate(dfhd['ncycle'].unique()) :

    this_cycle              = dfhd[ dfhd.ncycle==nc ]
    this_cycle_insp         = this_cycle[this_cycle.is_inspiration==1]

    if len(this_cycle_insp)<1 : continue
    cycle_inspiration_end   = this_cycle_insp['dt'].iloc[-1]

    if i > len(dfhd['ncycle'].unique()) -2 : continue
    #compute tidal volume in simulator df
    subdf             = df[ (df.dt>start_times[i]) & (df.dt<start_times[i+1]) ]

    subdf['total_vol_subtracted'] = subdf['total_vol'] - subdf['total_vol'].min()
    real_tidal_volume = subdf['total_vol_subtracted' ] .max()
    #compute plateau in simulator
    subdf             = df[ (df.dt>start_times[i]) & (df.dt<cycle_inspiration_end) ]
    real_plateau      = subdf[ (subdf.dt > cycle_inspiration_end - 20e-3) ]['airway_pressure'].mean()
    #this_cycle_insp[(this_cycle_insp['dt'] > start_times[i] + inspiration_duration - 20e-3) & (this_cycle_insp['dt'] < start_times[i] + inspiration_duration - 10e-3)]['airway_pressure'].mean()
    real_tidal_volumes.append(  real_tidal_volume   )
    real_plateaus.append (real_plateau)


    measured_peeps.append(  this_cycle['cycle_PEEP'].iloc[0])
    measured_volumes.append(this_cycle['cycle_tidal_volume'].iloc[0])
    measured_peaks.append(   this_cycle['cycle_peak_pressure'].iloc[0])
    measured_plateaus.append(this_cycle['cycle_plateau_pressure'].iloc[0])

  print ("measured_peeps", measured_peeps)
  print ("measured_volumes",measured_volumes)
  print ("measured_peaks",measured_peaks)
  #print ("measured_plateau",measured_plateau)


  ##################################
  # find runs
  ##################################
  add_run_info(df)

  ####################################################
  # plot simple canavas
  ####################################################
  plot_general_canvases(df, dfhd, fname, output_directory, start_times, colors, respiration_rate, inspiration_duration)

  # 'choose here the name of the MVM flux variable to be shown in arXiv plots
  dfhd['display_flux'] = dfhd['flux_3'] # why???
  plot_arXiv_style(df, dfhd, fname, output_directory, start_times, colors, sett)

  '''
  ax1 = plt.subplot(311)
  plt.plot(df['dt'], df['airway_pressure'], label='true airway_pressure')
  plt.plot(dfhd['dt'], dfhd['airway_pressure'], label='pressure')

  ax2 = plt.subplot(312, sharex=ax1)
  plt.plot(df['dt'], df['total_vol'], label='true total_vol')
  plt.plot(dfhd['dt'], dfhd['tidal_volume'], label='tidal volume')

  ax3 = plt.subplot(313, sharex=ax1)
  plt.plot(df['dt'], df['total_flow'], label='true total_flow')
  plt.plot(dfh['dt'], df['display_flux'], label='flow')

  #plt.legend()
  plt.show()
  '''
