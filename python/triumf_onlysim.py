'''
Started from combine.py at 8a9574d9ea3f3454a85669d60824df9360104c86
    Trimmed in order to analyze TRIUMF simulator data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors
from scipy.interpolate import interp1d
import matplotlib.patches as patches
from scipy.signal import find_peaks
import json

from db import *
from mvmio import *

def add_timestamp(df, timecol='dt'):
  ''' add timestamp column assuming constant sampling in time '''
  df['timestamp'] = np.linspace( df.iloc[0,:][timecol] ,  df.iloc[-1,:][timecol] , len(df) )
  return df

def get_deltat(df, timestampcol='timestamp', timecol='dt'):
  ''' retrieve sampling time '''
  return df[timestampcol].iloc[2] - df[timestampcol].iloc[1]

def correct_sim_df(df):
  ''' Apply corrections to simulator data '''
  df['total_vol'] = df['total_vol'] * 0.1

def get_start_times(df, thr=50, quantity='out', timecol='dt'):
  ''' Get time where a given quantity starts being above threshold '''
  times_open = df[ ( df[quantity]>thr ) ][timecol]
  start_times = [ float(times_open.iloc[i]) for i in range(0, len(times_open)-1) if times_open.iloc[i+1]-times_open.iloc[i] > 0.1  or i == 0  ]
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

def add_cycle_info(sim, start_times, reaction_times):
  ''' Add cycle start, reaction time, time-in-breath '''
  sim['start'] = 0
  sim['tbreath'] = 0
  sim['reaction_time'] = 0
  for s,f in zip (start_times,reaction_times) :
    sim.loc[sim.dt>s,'start']   = s
    sim.loc[sim.dt>s,'tbreath'] = sim.dt - s
    sim.loc[sim.dt>s,'reaction_time'] = f

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

def add_clinical_values (df, max_R=250, max_C=100) :
  deltaT = get_deltat(df, timestampcol='dt')
  """Add for reference the measurement of "TRUE" clinical values as measured using the simulator"""

  #true resistance
  df['delta_pout_pin']        =  df['airway_pressure'] - df['chamber1_pressure']                  # cmH2O
  df['delta_vol']             = ( df['chamber1_vol'] * 2. ) .diff()                               # ml
  df['airway_resistance']     =  df['delta_pout_pin'] / ( df['delta_vol'] / deltaT/1000. )        # cmH2O/(l/s)
  df.loc[ (abs(df.airway_resistance)>max_R) | (df.airway_resistance<0) ,"airway_resistance"] = 0

  #true compliance
  df['deltapin']     =  df['chamber1_pressure'].diff()
  df['delta_volin']  =  ( df['chamber1_vol']  + df['chamber2_vol']) . diff()
  df['compliance']   =  df['delta_volin']/df['deltapin']
  df.loc[abs(df.compliance)>max_C,"compliance"] = 0

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

def process_run(meta, objname, fullpath_rwa, fullpath_dta, columns_rwa, columns_dta, manual_offset=0., save=False, mhracsv=None, pressure_offset=0, output_directory='plots_tmp'):
  # retrieve simulator data
  df = get_simulator_df(fullpath_rwa, fullpath_dta, columns_rwa, columns_dta)

  # apply corrections
  correct_sim_df(df)

  ##################################
  # cycles
  ##################################

  # compute cycle start
  start_times    = get_start_times(df, thr=8, quantity='airway_pressure', timecol='dt')
  reaction_times = get_reaction_times(df, start_times)

  # add info
  add_cycle_info(sim=df, start_times=start_times, reaction_times=reaction_times)

  ##################################
  # chunks
  ##################################

  add_chunk_info(df)

  # compute tidal volume etc
  add_clinical_values(df)

  #plt.plot(df['dt'], df['tracheal_pressure'], label='true tracheal_pressure')
  #plt.plot(df['dt'], df['airway_pressure'], label='true airway_pressure')
  #plt.plot(df['dt'], df['chamber1_pressure'], label='true ch1_pressure')
  #plt.plot(df['dt'], df['chamber2_pressure'], label='true ch2_pressure')
  #plt.plot(df['dt'], df['airway_resistance'], label='true airway_resistance')
  #plt.plot(df['dt'], df['compliance'], label='true compliance')
  #plt.legend()
  #plt.show()

  ##################################
  # find runs
  ##################################

  add_run_info(df)

  ##################################
  # saving and plotting
  ##################################
  if save:
    df.to_hdf(f'{objname}.sim.h5', key='simulator')

  if args.plot:
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

    ####################################################
    '''general service canvas number 1'''
    ####################################################
    ax = df.plot(x='dt', y='airway_pressure', label='airway_pressure [cmH2O]', c=colors['sim_airway_pressure'])
    df.plot(ax=ax, x='dt', y='total_flow',    label='total_flow      [l/min]', c=colors['total_flow'])
    df.plot(ax=ax, x='dt', y='run', label='run index')
    df.plot(ax=ax, x='dt', y='total_vol',          label='sim volume            [l/min]', c='red' )
    plt.plot(start_times, [0]*len(start_times), 'bo', label='real cycle start time')
    #df.plot(ax=ax , x='dt', y='muscle_pressure', label='muscle_pressure [cmH2O]', c=colors['muscle_pressure'])
    df.plot(ax=ax, x='dt', y='total_vol',         label='SIM tidal volume       [cl]', c=colors['total_vol'] , alpha=0.4)

    #df.plot(ax=ax, x='dt', y='breath_no', label='breath_no', marker='.')
    #df.plot(ax=ax, x='dt', y='tracheal_pressure', label='tracheal_pressure')
    #df.plot(ax=ax, x='dt', y='total_vol', label='total_vol')
    #plt.plot(df['dt'], df['total_vol']/10., label='total_vol [cl]')
    #df.plot(ax=ax, x='dt', y='deriv_total_vol', label='deriv_total_vol [l/min]')

    #plt.gcf().suptitle(objname)
    #plt.plot(start_times, [0]*len(start_times), 'bo', label='real cycle start time')
    #df.plot(ax=ax, x='dt', y='reaction_time', label='reaction time [10 ms]', c=colors['reaction_time'])
    #plt.plot(   ,  100 *reaction_times,      label='reaction time ', marker='o', markersize=1, linewidth=0, c='red')

    for i,t in enumerate(start_times) :
      ax.text(t, 0.5, "%i"%i, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)

    ax.set_xlabel("Time [sec]")
    ax.legend(loc='upper center', ncol=2)

    ax.set_title ("TRIUMF Test n %s"%meta[objname]['test_name'], weight='heavy')
    figpath = "%s/%s_service_%s.png" % (output_directory, meta[objname]['Campaign'],  objname.replace('.txt', ''))
    print(f'Saving figure to {figpath}')
    plt.savefig(figpath)


    ####################################################
    '''general service canvas number 2, measured simulation parameters'''
    ####################################################
    figbis, axbis = plt.subplots()

    df.plot(ax=axbis, x='dt', y='airway_pressure', label='airway_pressure [cmH2O]', c=colors['sim_airway_pressure'])
    df.plot(ax=axbis, x='dt', y='total_flow',    label='total_flow      [l/min]', c=colors['total_flow'])
    plt.plot(start_times, [0]*len(start_times), 'bo', label='real cycle start time')
    #df.plot(ax=axbis, x='dt', y='compliance',   label='SIM compliance', c='blue')
    #df.plot(ax=axbis, x='dt', y='airway_resistance',   label='SIM resistance', c='black', linestyle="--")

    xmin, xmax = axbis.get_xlim()
    ymin, ymax = axbis.get_ylim()
    mytext = ""  # text label to write on the plot
    axbis.text((xmax-xmin)/2.+xmin, 0.08*(ymax-ymin) + ymin, mytext, verticalalignment='bottom', horizontalalignment='center', color='#7697c4')

    axbis.set_xlabel("Time [sec]")
    axbis.legend(loc='upper center', ncol=2)

    axbis.set_title ("TRIUMF Test n %s"%meta[objname]['test_name'], weight='heavy')
    figpath = "%s/%s_service2_%s.png" % (output_directory, meta[objname]['Campaign'],  objname.replace('.txt', ''))
    print(f'Saving figure to {figpath}')
    figbis.savefig(figpath)


    ####################################################
    '''formatted plots for ISO std'''
    ####################################################
    for i in range (len(meta)) :

      local_objname = "%s_%i"% ( objname[:-2] , i )

      PE = meta[local_objname]["Peep"]
      PI = meta[local_objname]["Pinspiratia"]
      RR = meta[local_objname]["Rate respiratio"]
      RT = meta[local_objname]["Resistance"]
      CM = meta[local_objname]["Compliance"]

      print ("Looking for R=%s, C=%s, RR=%s, PEEP=%s, PINSP=%s"%(RT,CM,RR,PE,PI) )

      my_selected_cycle = meta[local_objname]["cycle_index"]
      print ("\nFor test [ %s ]  I am selecting cycle %i, starting at %f \n"%(meta[local_objname]["test_name"], my_selected_cycle , start_times[ my_selected_cycle ]))

      fig11, ax11 = plt.subplots()

      print (start_times)
      ## make a subset dataframe for simulator
      dftmp = df[ (df['start'] >= start_times[ my_selected_cycle ] ) & ( df['start'] < start_times[ my_selected_cycle + 6])  ]
      ## the (redundant) line below avoids the annoying warning
      dftmp = dftmp[ (dftmp['start'] >= start_times[ my_selected_cycle ] ) & ( dftmp['start'] < start_times[ my_selected_cycle + 6])  ]

      dftmp.loc[:, 'total_vol'] = dftmp['total_vol'] - dftmp['total_vol'].min()

      dftmp.plot(ax=ax11, x='dt', y='total_vol',         label='SIM tidal volume       [cl]', c=colors['total_vol'] , alpha=0.4)
      dftmp.plot(ax=ax11, x='dt', y='total_flow',        label='SIM flux            [l/min]', c=colors['total_flow'])
      dftmp.plot(ax=ax11, x='dt', y='airway_pressure',   label='SIM airway pressure [cmH2O]', c=colors['sim_airway_pressure'])
      #dftmp.plot(ax=ax11 , x='dt', y='muscle_pressure',  label='SIM muscle_pressure [cmH2O]', c=colors['muscle_pressure'])

      ymin, ymax = ax11.get_ylim()
      ax11.set_ylim(ymin*1.45, ymax*1.55)
      ax11.legend(loc='upper center', ncol=2)
      title1="R = %i [cmH2O/l/s]         C = %2.1f [ml/cmH20]         PEEP = %s [cmH20]"%(RT,CM,PE )
      title2="Inspiration Pressure = %s [cmH20]       Frequency = %s [breath/min]"%(PI,RR)

      ax11.set_xlabel("Time [s]")

      xmin, xmax = ax11.get_xlim()
      ymin, ymax = ax11.get_ylim()
      ax11.text((xmax-xmin)/2.+xmin, 0.08*(ymax-ymin) + ymin,   title2, verticalalignment='bottom', horizontalalignment='center', color='#7697c4')
      ax11.text((xmax-xmin)/2.+xmin, 0.026*(ymax-ymin) + ymin,  title1, verticalalignment='bottom', horizontalalignment='center', color='#7697c4')
      nom_pressure = float(meta[local_objname]["Pinspiratia"])
      rect = patches.Rectangle((xmin,nom_pressure-2),xmax-xmin,4,edgecolor='None',facecolor='green', alpha=0.2)
      #add / remove nominal pressure band
      #ax11.add_patch(rect)

      nom_peep = float(meta[local_objname]["Peep"])
      rect = patches.Rectangle((xmin,nom_peep-0.1),xmax-xmin,0.5,edgecolor='None',facecolor='grey', alpha=0.3)
      #add / remove PEEP line
      #ax11.add_patch(rect)

      ax11.set_title ("TRIUMF Test n %s"%meta[objname]['test_name'], weight='heavy')
      figpath = "%s/%s_%s.pdf" % (output_directory, meta[objname]['Campaign'],  objname.replace('.txt', ''))
      print(f'Saving figure to {figpath}')
      fig11.savefig(figpath)

      ## Same plot, with SIM compilance and SIM resistance
      #dftmp.plot(ax=ax11, x='dt', y='compliance',   label='SIM compliance', c='blue')
      #dftmp.plot(ax=ax11, x='dt', y='airway_resistance',   label='SIM resistance', c='black', linestyle="--")
      #figpath = "%s/%s_extrainfo_%s.pdf" % (output_directory, meta[objname]['Campaign'],  objname.replace('.txt', ''))
      #ax11.legend(loc='upper center', ncol=2)
      #print(f'Saving figure to {figpath}')
      #fig11.savefig(figpath)

      ####################################################
      '''plot the avg wfs'''
      ####################################################

      fig2, ax2 = plt.subplots()
      ## make a subset dataframe for simulator
      #dftmp = df[ (df['start'] >= start_times[ 4 ] ) & ( df['start'] < start_times[ min ([35,len(start_times)-1] )  ])]
      dftmp = df[ (df['start'] >= start_times[ my_selected_cycle ] ) & ( df['start'] < start_times[ len(start_times)-1 ])]
      ## the (redundant) line below avoids the annoying warning
      dftmp = dftmp[ (dftmp['start'] >= start_times[ my_selected_cycle ] ) & ( dftmp['start'] < start_times[ len(start_times)-1 ])  ]
      dftmp['dtc'] = df['dt'] - df['start']

      dftmp.loc[:, 'total_vol'] = dftmp['total_vol'] - dftmp['total_vol'].min()

      dftmp.plot(ax=ax2, x='dtc', y='total_vol',         label='SIM tidal volume       [cl]', c=colors['total_vol'] ,          marker='o', markersize=0.3, linewidth=0)
      dftmp.plot(ax=ax2, x='dtc', y='total_flow',        label='SIM flux            [l/min]', c=colors['total_flow'],          marker='o', markersize=0.3, linewidth=0)
      dftmp.plot(ax=ax2, x='dtc', y='airway_pressure',   label='SIM airway pressure [cmH2O]', c=colors['sim_airway_pressure'], marker='o', markersize=0.3, linewidth=0)

      ymin, ymax = ax2.get_ylim()
      ax2.set_ylim(ymin*1.4, ymax*1.5)
      ax2legend = ax2.legend(loc='upper center', ncol=2)
      ## hack to set larger marker size in legend only, one line per data series
      legmarkersize = 10
      ax2legend.legendHandles[0]._legmarker.set_markersize(legmarkersize)
      ax2legend.legendHandles[1]._legmarker.set_markersize(legmarkersize)
      ax2legend.legendHandles[2]._legmarker.set_markersize(legmarkersize)

      title1="R = %i [cmH2O/l/s]         C = %2.1f [ml/cmH20]        PEEP = %s [cmH20]"%(RT,CM,PE )
      title2="Inspiration Pressure = %s [cmH20]       Frequency = %s [breath/min]"%(PI,RR)

      ax2.set_xlabel("Time [s]")

      xmin, xmax = ax2.get_xlim()
      ymin, ymax = ax2.get_ylim()
      ax2.text((xmax-xmin)/2.+xmin, 0.08*(ymax-ymin) + ymin,   title2, verticalalignment='bottom', horizontalalignment='center', color='#7697c4')
      ax2.text((xmax-xmin)/2.+xmin, 0.026*(ymax-ymin) + ymin,  title1, verticalalignment='bottom', horizontalalignment='center', color='#7697c4')
      nom_pressure = float(meta[local_objname]["Pinspiratia"])
      rect = patches.Rectangle((xmin,nom_pressure-2),xmax-xmin,4,edgecolor='None',facecolor='green', alpha=0.2)
      ax2.add_patch(rect)

      nom_peep = float(meta[local_objname]["Peep"])
      rect = patches.Rectangle((xmin,nom_peep-0.1),xmax-xmin,0.5,edgecolor='None',facecolor='grey', alpha=0.3)
      ax2.add_patch(rect)

      ax2.set_title ("TRIUMF Test n %s"%meta[objname]['test_name'])
      figpath = "%s/%s_avg_%s.png" % (output_directory, meta[objname]['Campaign'] , objname.replace('.txt', ''))
      print(f'Saving figure to {figpath}')
      fig2.savefig(figpath)


if __name__ == '__main__':
  import argparse
  import matplotlib
  import style

  parser = argparse.ArgumentParser(description='repack data taken in continuous mode')
  parser.add_argument("input", help="name of the MVM input file (.txt)")
  parser.add_argument("-d", "--output-directory", type=str, help="name of the output directory for plots", default="plots_iso")
  parser.add_argument("-skip", "--skip_files", type=str,  help="skip files", nargs='+', default="")
  parser.add_argument("-p", "--plot", action='store_true', help="show plots")
  parser.add_argument("-s", "--save", action='store_true', help="save HDF")
  parser.add_argument("-f", "--filename", type=str, help="single file to be processed", default='.')
  parser.add_argument("-c", "--campaign", type=str, help="single campaign to be processed", default="")
  parser.add_argument("-o", "--offset", type=float, help="offset between vent/sim", default='.0')
  parser.add_argument("--db-google-id", type=str, help="name of the Google spreadsheet ID for metadata", default="1aQjGTREc9e7ScwrTQEqHD2gmRy9LhDiVatWznZJdlqM")
  parser.add_argument("--db-range-name", type=str, help="name of the Google spreadsheet range for metadata", default="TRIUMF!A2:AP")
  args = parser.parse_args()

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
  columns_dta = [#'dt',
    'breath_no',
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

  print ("INPUT : : ",  args.input )

  # read metadata spreadsheet
  df_spreadsheet = read_online_spreadsheet(spreadsheet_id=args.db_google_id, range_name=args.db_range_name)
  filenames = df_spreadsheet['MVM_filename'].unique()

  for filename in filenames:
    # continue if there is no filename
    if not filename: continue

    # read the metadata and create a dictionary with relevant info
    meta  = read_meta_from_spreadsheet (df_spreadsheet, filename)

    objname = f'{filename}_0'   #at least first element is always there

    # compute the file location: local folder to the data repository + compaign folder + filename
    fname = f'{args.input}/{meta[objname]["Campaign"]}/{meta[objname]["MVM_filename"]}'

    print(f'\nFile name {fname}')
    if fname.split('/')[-1] in args.skip_files:
      print('    ... skipped')
      continue

    if args.campaign:
      if args.campaign not in fname:
        print(f'    ... not in selected campaign {args.campaign}')
        continue

    # determine RWA and DTA data locations
    fullpath_rwa = f'{fname}/{meta[objname]["SimulatorFileName"]}'

    if fullpath_rwa.endswith('.dta'):
      fullpath_rwa =  fullpath_rwa[:-4]      #remove extension if dta
    if not fullpath_rwa.endswith('.rwa'):
      fullpath_rwa =  f'{fullpath_rwa}.rwa'  #if .rwa extension not present, add it

    fullpath_dta = fullpath_rwa.replace('rwa', 'dta')
    print(f'will retrieve RWA and DTA simulator data from {fullpath_rwa} and {fullpath_dta}')

    # run
    process_run(meta, objname=objname, fullpath_rwa=fullpath_rwa, fullpath_dta=fullpath_dta, columns_rwa=columns_rwa, columns_dta=columns_dta, save=args.save, manual_offset=args.offset, output_directory=args.output_directory)

  if args.plot:
    if ( len (filenames) < 2 ) :
      plt.show()
    else :
      answer = input("plot all the files? (return: yes, Ctrl-D: no)")
      plt.show()
