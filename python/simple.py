import pandas as pd
import matplotlib.pyplot as plt
import argparse

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

output_directory="C:/Users/andre/Documents/MVM/breathing_simulator_test/data_analysis/online_plots"
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

#fullpath = "C:/Users/andre/Documents/MVM/breathing_simulator_test/20200408/rawdata/"
#fullpath = "C:/Users/andre/Documents/MVM/breathing_simulator_test/data_analysis/data/Run_12_Apr_8/"
fullpath = "C:/Users/andre/Documents/MVM/breathing_simulator_test/data_analysis/data/Napoli/"

#simname='passive_C20R6_RR15_Pins30_It1.5_PEEP7.rwa'
#simname='08042020_CONTROLLED_FR20_PEEP5_PINSP30_C20_R5_RATIO05.rwa'
simname='run_035.rwa'
fullpath_rwa = fullpath+simname

#fname="VENTILATOR_CONTROLLED_FR20_PEEP5_PINSP30_C20_R5_RATIO0.50.txt"
fname="run035_MVM_NA_Arxiv6a_O2_wSIM_C25R20.json"
input_mvm = fullpath+fname

mvm_sep = " -> "
mvm_columns = "mvm_col_arduino"

sett = { 'C':25,
'R':20,
'RR':10,
'P':15,
'PEEP':5,
'RF':0.5}


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='MVM online analysis')
  parser.add_argument("input_mvm", help="name of the MVM input file (.json)",default='run035_MVM_NA_Arxiv6a_O2_wSIM_C25R20.json')
  parser.add_argument("input_asl", help="name of the ASL5000 simulation file (.rwa)",default='run_035.rwa')

  parser.add_argument("-i", "--input_directory", type=str, help="name of the output directory for plots", default='./data/')
  parser.add_argument("-d", "--output_directory", type=str, help="name of the output directory for plots", default='./online_plots')
  parser.add_argument("-s", "--skip_sim", action='store_true',  help="skip_sim")
  #parser.add_argument("-json", action='store_true', help="read json instead of csv")
  parser.add_argument("-txt", action='store_true', help="read csv instead of json")
  parser.add_argument("-t", "--time_offset", type=float, help="time offset between vent/sim", default=0)
  parser.add_argument("-o", "--pressure_offset", type=float, help="pressure offset between vent/sim", default=0)
  parser.add_argument("->","--tag", type=str, help="global label", default="test")

  parser.add_argument("-C", "--compliance", type=float, help="Compliance of the respiratory system  [ml/hPa]", default=0)
  parser.add_argument("-R", "--resistance", type=float, help="Resistance of the respiratory system  [hPa/l/s]", default=0)
  parser.add_argument("-P", "--inspiratory_pressure", type=float, help="Targeted Pressure [cmH2O]", default=0)
  parser.add_argument("-Q", "--PEEP",type=float, help="Positive end-expiratory pressure  [cmH2O]", default=0)
  parser.add_argument("-r", "--respiratory_rate", help='Number of breath per minute',default=1)
  parser.add_argument("-f", "--respiratory_fraction", help='I:E',default=0)

  args = parser.parse_args()

  fullpath=args.input_directory
  input_mvm = fullpath+args.input_mvm
  print('Reading MVM data from',input_mvm)
  fullpath_rwa = fullpath+args.input_asl
  print('Reading ASL5000 data from',fullpath_rwa)
  fullpath_dta = fullpath_rwa.replace('rwa', 'dta')

  pressure_offset = args.pressure_offset
  manual_offset = args.time_offset

  sett = { 
          'C':float(args.compliance),
          'R':float(args.resistance),
          'RR':float(args.respiratory_rate),
          'P':float(args.inspiratory_pressure),
          'PEEP':float(args.PEEP),
          'RF':float(args.respiratory_fraction)
          }

  # retrieve simulator data
  if not args.skip_sim:
    df = get_simulator_df(fullpath_rwa, fullpath_dta, columns_rwa, columns_dta)
  else:
    print('Ignore ASL5000 data')

  mvm_json=True
  if args.txt: 
    mvm_json = False
  # retrieve MVM data
  if mvm_json:    dfhd = get_mvm_df_json (fname=input_mvm)
  else: dfhd = get_mvm_df(fname=input_mvm, sep=mvm_sep, configuration=mvm_columns)
  add_timestamp(dfhd)

  # apply corrections
  correct_mvm_df(dfhd, pressure_offset)
  if not args.skip_sim:
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

  if args.skip_sim:
      plot_mvm_only_canvases(dfhd, {'dummy':{'test_name':'test'}}, 'dummy', start_times, colors)
      exit #stop here if sim is ignored

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

  """   
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
  print ("measured_plateau",measured_plateau)
  """

  ##################################
  # find runs
  ##################################
  add_run_info(df)

  ####################################################
  # plot simple canavas
  ####################################################
  plot_general_canvases(df, dfhd, fname, args.output_directory, start_times, colors, respiration_rate, inspiration_duration, args.tag)

  # 'choose here the name of the MVM flux variable to be shown in arXiv plots
  dfhd['display_flux'] = dfhd['flux_3'] # why???
  plot_arXiv_style(df, dfhd, fname, args.output_directory, start_times, colors, sett, args.tag)


  fig=plt.figure(figsize=(13,8))
  
  ax1 = plt.subplot(311)
  plt.title(' '.join([k+str(sett[k]) for k in sett]))
  plt.plot(df['dt'], df['airway_pressure'], label='ASL5000 airway p')
  plt.plot(dfhd['dt'], dfhd['airway_pressure'], label='MVM pressure')
  plt.setp(ax1.get_xticklabels(), visible=False)
  plt.ylabel('[cmH2O]')
  plt.legend()


  ax2 = plt.subplot(312, sharex=ax1)
  plt.plot(df['dt'], df['total_vol'], label='ASL5000 tot vol')
  plt.plot(dfhd['dt'], dfhd['tidal_volume'], label='MVM tidal vol')
  plt.setp(ax2.get_xticklabels(), visible=False)
  plt.ylabel('[cL]')
  plt.legend()

  ax3 = plt.subplot(313, sharex=ax1)
  plt.plot(df['dt'], df['total_flow'], label='ASL5000 tot flow')
  plt.plot(dfhd['dt'], dfhd['display_flux'], label='MVM flow')
  plt.xlabel('[sec]')
  plt.ylabel('[L/min]')
  plt.legend()

  plt.xlim(10, 60)

  fig.tight_layout()
  plt.show()
  fig.savefig('{:s}/C{:1.0f}R{:1.0f}_RR{:1.0f}_Pins{:1.0f}_PEEP{:1.0f}.png'.format(args.output_directory,sett['C'],sett['R'],sett['RR'],sett['P'],sett['PEEP']))