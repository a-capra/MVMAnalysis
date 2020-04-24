import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib import colors
from scipy.interpolate import interp1d
import matplotlib.patches as patches

def plot_all(df, dfhd, objname, output_directory, start_times, colors, meta, tag='TRIUMF_test'):

  ####################################################
  '''general service canavas number 1'''
  ####################################################

  fig31,ax31 = plt.subplots(3,1,sharex=True)
  ax31 = ax31.flatten()

  if df is not None:
    df.plot(ax=ax31[0], x='dt', y='total_flow',    label='total_flow      [l/min]', c=colors['total_flow'])
    df.plot(ax=ax31[1], x='dt', y='airway_pressure', label='airway_pressure [cmH2O]', c=colors['sim_airway_pressure'])
    df.plot(ax=ax31[2], x='dt', y='total_vol',         label='SIM tidal volume       [cl]', c=colors['total_vol'] , alpha=0.4)

  if 'flux_2' in dfhd:
    dfhd.plot(ax=ax31[0], x='dt', y='flux_2', label='flux_2', c='r', linestyle="--")
  if 'flux_3' in dfhd:
    dfhd.plot(ax=ax31[0], x='dt', y='flux_3',  label='flux_3', c='r')
  dfhd.plot(ax=ax31[0], x='dt', y='flux',            label='ventilator flux            [l/min]', c=colors['flux'] )

  dfhd.plot(ax=ax31[1], x='dt', y='airway_pressure', label='ventilator airway pressure [cmH2O]', c=colors['vent_airway_pressure'])
  #plt.plot(start_times, [0]*len(start_times), 'bo', label='real cycle start time')

  dfhd.plot(ax=ax31[2],  x='dt', y='tidal_volume',    label='MVM tidal volume       [cl]',        c='b')

  for ax in ax31:
    for i,t in enumerate(start_times) :
      ax.text(t, 0.5, "%i"%i, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=8)
    ax.set_xlabel("Time [s]")
    ax.plot(start_times, [0]*len(start_times), 'bo', label='real cycle start time')
    ax.legend(loc='upper right', ncol=2)
  
  fig31.set_size_inches(13,10)
  ttl=' '.join([k+str(meta[k]) for k in meta])
  fig31.suptitle(tag.replace('_',' ') + ' ' + ttl, weight='heavy')
  figpath = "%s/%s_all_%s.png" % (output_directory, tag,  objname.replace('.json', '')) # TODO: make sure it is correct, or will overwrite!
  print(f'Saving figure to {figpath}')
  fig31.tight_layout()
  fig31.savefig(figpath)



def plot_3views(df, dfhd, objname, output_directory, start_times, colors, meta, my_selected_cycle, tag='TRIUMF_test'):

  ####################################################
  '''plot 3 views in a single canvas'''
  ####################################################

  Nbreath=6
  if (my_selected_cycle + Nbreath) >= len(start_times):
    my_selected_cycle = len(start_times) - 1 - Nbreath
    print('move beginning of selected cycle to',my_selected_cycle)
  if my_selected_cycle < 0: my_selected_cycle=0
  print ("For test I am selecting cycle %i, starting at %f \n"%( my_selected_cycle , start_times[ my_selected_cycle ]))

  fig31,ax31 = plt.subplots(3,1)
  ax31 = ax31.flatten()

  #print (start_times)
  #make a subset dataframe for simulator
  dftmp = df[ (df['start'] >= start_times[ my_selected_cycle ] ) & ( df['start'] < start_times[ my_selected_cycle + Nbreath])  ]
  #the (redundant) line below avoids the annoying warning
  dftmp = dftmp[ (dftmp['start'] >= start_times[ my_selected_cycle ] ) & ( dftmp['start'] < start_times[ my_selected_cycle + Nbreath])  ]

  #make a subset dataframe for ventilator
  first_time_bin  = dftmp['dt'].iloc[0]
  last_time_bin   = dftmp['dt'].iloc[len(dftmp)-1]
  dfvent = dfhd[ (dfhd['dt']>first_time_bin) & (dfhd['dt']<last_time_bin) ]

  dftmp.loc[:, 'total_vol'] = dftmp['total_vol'] - dftmp['total_vol'].min()

  dftmp.plot(ax=ax31[2], x='dt', y='total_vol',         label='SIM tidal volume       [cl]',        c='r')
  dftmp.plot(ax=ax31[0], x='dt', y='total_flow',        label='SIM flux            [l/min]',        c='r')
  dftmp.plot(ax=ax31[1], x='dt', y='airway_pressure',   label='SIM airway pressure [cmH2O]',        c='r')

  dfvent.plot(ax=ax31[2],  x='dt', y='tidal_volume',    label='MVM tidal volume       [cl]',        c='b')
  dfvent.plot(ax=ax31[0],  x='dt', y='display_flux',    label='MVM flux            [l/min]',        c='b')
  dfvent.plot(ax=ax31[1],  x='dt', y='airway_pressure', label='MVM airway pressure [cmH2O]',        c='b')

  ax31[0].set_xlabel("Time [s]")
  ax31[1].set_xlabel("Time [s]")
  ax31[2].set_xlabel("Time [s]")

  fig31.set_size_inches(15,8)
  ttl=' '.join([k+str(meta[k]) for k in meta])
  fig31.suptitle(tag.replace('_',' ') + ' ' + ttl, weight='heavy')
  figpath = "%s/%s_3views_%s.png" % (output_directory, tag,  objname.replace('.json', '')) # TODO: make sure it is correct, or will overwrite!
  print(f'Saving figure to {figpath}')
  fig31.tight_layout()
  fig31.savefig(figpath)
  


def basic_plotting(df,dfhd,sett,output_directory):
  fig=plt.figure(figsize=(15,8))

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

  #plt.xlim(10, 60)

  fig.tight_layout()
  fig.savefig('{:s}/C{:1.0f}R{:1.0f}_RR{:1.0f}_Pins{:1.0f}_PEEP{:1.0f}.png'.format(output_directory,sett['C'],sett['R'],sett['RR'],sett['P'],sett['PEEP']))
