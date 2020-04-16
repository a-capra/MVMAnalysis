import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib import colors
from scipy.interpolate import interp1d
import matplotlib.patches as patches


def plot_general_canvases (df, dfhd, objname, output_directory, start_times, colors, respiration_rate, inspiration_duration):

  ####################################################
  '''general service canavas number 1'''
  ####################################################
  ax = df.plot(x='dt', y='airway_pressure', label='airway_pressure [cmH2O]', c=colors['sim_airway_pressure'])
  df.plot(ax=ax, x='dt', y='total_flow',    label='total_flow      [l/min]', c=colors['total_flow'])
  #df.plot(ax=ax, x='dt', y='run', label='run index')
  plt.plot(start_times, [0]*len(start_times), 'bo', label='real cycle start time')
  #df.plot(ax=ax , x='dt', y='muscle_pressure', label='muscle_pressure [cmH2O]', c=colors['muscle_pressure'])
  df.plot(ax=ax, x='dt', y='total_vol',         label='SIM tidal volume       [cl]', c=colors['total_vol'] , alpha=0.4)
  #dfhd.plot(ax=ax,  x='dt', y='tidal_volume',    label='MVM tidal volume       [cl]', c=colors['tidal_volume'])

  #df.plot(ax=ax, x='dt', y='breath_no', label='breath_no', marker='.')
  #df.plot(ax=ax, x='dt', y='tracheal_pressure', label='tracheal_pressure')
  #df.plot(ax=ax, x='dt', y='total_vol', label='total_vol')
  #plt.plot(df['dt'], df['total_vol']/10., label='total_vol [cl]')
  #dfhd.plot(ax=ax, x='dt', y='pressure', label='ventilator pressure [cmH2O]', c=colors['pressure'], linewidth = linw)
  dfhd.plot(ax=ax, x='dt', y='airway_pressure', label='ventilator airway pressure [cmH2O]', c=colors['vent_airway_pressure'])
  dfhd.plot(ax=ax, x='dt', y='flux',            label='ventilator flux            [l/min]', c=colors['flux'] )
  #dfhd.plot(ax=ax, x='dt', y='volume',          label='volume            [l/min]', c=colors['flux'] )
#  dfhd.plot(ax=ax, x='dt', y='out', label='out')
  #dfhd.plot(ax=ax, x='dt', y='in', label='in')
  #dfhd.plot(ax=ax, x='dt', y='service_1', label='service 2', c='black', linestyle="--")
  #dfhd.plot(ax=ax, x='dt', y='service_2', label='service 1', c='black')
  dfhd.plot(ax=ax, x='dt', y='flux_2', label='flux_2', c='r', linestyle="--")
  dfhd.plot(ax=ax, x='dt', y='flux_3',  label='flux_3', c='r')
  #dfhd.plot(ax=ax, x='dt', y='derivative',  label='derivative', c='gray')
  #df.plot(ax=ax, x='dt', y='deriv_total_vol', label='deriv_total_vol [l/min]')

  #plt.gcf().suptitle(objname)
  #plt.plot(start_times, [0]*len(start_times), 'bo', label='real cycle start time')
  #df.plot(ax=ax, x='dt', y='reaction_time', label='reaction time [10 ms]', c=colors['reaction_time'])
  #plt.plot(   ,  100 *reaction_times,      label='reaction time ', marker='o', markersize=1, linewidth=0, c='red')
  #ax.
  for i,t in enumerate(start_times) :
    ax.text(t, 0.5, "%i"%i, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)

  ax.set_xlabel("Time [sec]")
  ax.legend(loc='upper center', ncol=2)

  ax.set_title ("Test BCIT", weight='heavy')
  figpath = "%s/%s_service_%s.png" % (output_directory, "BCIT test",  objname.replace('.txt', ''))
  print(f'Saving figure to {figpath}')
  plt.savefig(figpath)


  ####################################################
  '''general service canavas number 2, measured simulation parameters'''
  ####################################################

  figbis = plt.figure()
  figbis.suptitle ("Test BCIT", weight='heavy')
  gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[.7,1.3], width_ratios = [1,1])

  axbis0 = figbis.add_subplot(gs[0, 0])
  axbis1 = figbis.add_subplot(gs[0, 1])
  axbis2 = figbis.add_subplot(gs[1:, :])

  df.plot(ax=axbis2, x='dt', y='airway_pressure', label='airway_pressure [cmH2O]', c=colors['sim_airway_pressure'])
  df.plot(ax=axbis2, x='dt', y='total_flow',    label='total_flow      [l/min]', c=colors['total_flow'])
  plt.plot(start_times, [0]*len(start_times), 'bo', label='real cycle start time')
  df.plot(ax=axbis2, x='dt', y='compliance',   label='SIM compliance', c='black')
  df.plot(ax=axbis2, x='dt', y='airway_resistance',   label='SIM resistance', c='black', linestyle="--")

  #dfhd.plot(axbis=axbis, x='dt', y='pressure', label='ventilator pressure [cmH2O]', c=colors['pressure'], linewidth = linw)
  dfhd.plot(ax=axbis2, x='dt', y='airway_pressure', label='ventilator airway pressure [cmH2O]', c=colors['vent_airway_pressure'])
  dfhd.plot(ax=axbis2, x='dt', y='pressure_pv1',    label='ventilator PV1 pressure    [cmH2O]', c=colors['vent_airway_pressure'], linestyle="--")
  dfhd.plot(ax=axbis2, x='dt', y='flux',            label='ventilator flux            [l/min]', c=colors['flux'] )
  dfhd.plot(ax=axbis2, x='dt', y='resistance',      label='ventilator resistance  [cmH2O/l/s]', c='pink' )
  dfhd.plot(ax=axbis2, x='dt', y='compliance',      label='ventilator compliance   [ml/cmH2O]', c='purple' )

  xmin, xmax = axbis2.get_xlim()
  ymin, ymax = axbis2.get_ylim()
  mytext = "Measured respiration rate: %1.1f br/min, inspiration duration: %1.1f s"%(respiration_rate, inspiration_duration)
  axbis2.text((xmax-xmin)/2.+xmin, 0.08*(ymax-ymin) + ymin,   mytext, verticalalignment='bottom', horizontalalignment='center', color='black')

  axbis2.set_xlabel("Time [sec]")
  axbis2.legend(loc='upper center', ncol=2)

  axbis0.hist ( dfhd[( dfhd['compliance']>0) ]['compliance'].unique()  , bins=50)
  axbis0.set_xlabel("Measured compliance [ml/cmH2O]")
  axbis1.hist ( dfhd[( dfhd['resistance']>0)]['resistance'].unique() , bins=50 )
  axbis1.set_xlabel("Measured resistance [cmH2O/l/s]")

  figpath = "%s/%s_service2_%s.png" % (output_directory, "BCIT test",  objname.replace('.txt', '')) # TODO: make sure it is correct, or will overwrite!
  print(f'Saving figure to {figpath}')
  figbis.savefig(figpath)


def plot_arXiv_style(df, dfhd, objname, output_directory, start_times, colors, meta):

  ####################################################
  '''formatted plots for ISO std / arXiv'''
  ####################################################

  local_objname = "%s_%i"% ( objname[:-2] , 1 )

  PE = meta["PEEP"]
  PI = meta["PIP"]
  RR = meta["RR"]
  RT = meta["R"]
  CM = meta["C"]

  print ("Looking for R=%s, C=%s, RR=%s, PEEP=%s, PINSP=%s"%(RT,CM,RR,PE,PI) )

  my_selected_cycle = 0

  print ("\nFor test BCIT  I am selecting cycle %i, starting at %f \n"%( my_selected_cycle , start_times[ my_selected_cycle ]))

  fig11,ax11 = plt.subplots()

  print (start_times)
  #make a subset dataframe for simulator
  dftmp = df[ (df['start'] >= start_times[ my_selected_cycle ] ) & ( df['start'] < start_times[ my_selected_cycle + 6])  ]
  #the (redundant) line below avoids the annoying warning
  dftmp = dftmp[ (dftmp['start'] >= start_times[ my_selected_cycle ] ) & ( dftmp['start'] < start_times[ my_selected_cycle + 6])  ]

  #make a subset dataframe for ventilator
  first_time_bin  = dftmp['dt'].iloc[0]
  last_time_bin   = dftmp['dt'].iloc[len(dftmp)-1]
  dfvent = dfhd[ (dfhd['dt']>first_time_bin) & (dfhd['dt']<last_time_bin) ]

  dftmp.loc[:, 'total_vol'] = dftmp['total_vol'] - dftmp['total_vol'].min()

  dftmp.plot(ax=ax11, x='dt', y='total_vol',         label='SIM tidal volume       [cl]', c=colors['total_vol'] , alpha=0.4, linestyle="--")
  dftmp.plot(ax=ax11, x='dt', y='total_flow',        label='SIM flux            [l/min]', c=colors['total_flow'])
  dftmp.plot(ax=ax11, x='dt', y='airway_pressure',   label='SIM airway pressure [cmH2O]', c=colors['sim_airway_pressure'])

  dfvent.plot(ax=ax11,  x='dt', y='tidal_volume',    label='MVM tidal volume       [cl]', c=colors['tidal_volume'])
  dfvent.plot(ax=ax11,  x='dt', y='display_flux',    label='MVM flux            [l/min]', c=colors['flux'])
  #dfvent.plot(ax=ax11,  x='dt', y='flux',            label='MVM SP1+SP2 flux    [l/min]', c=colors['flux'], linestyle='--')
  dfvent.plot(ax=ax11,  x='dt', y='airway_pressure', label='MVM airway pressure [cmH2O]', c=colors['vent_airway_pressure'])

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
  nom_pressure = float(PI)
  rect = patches.Rectangle((xmin,nom_pressure-2),xmax-xmin,4,edgecolor='None',facecolor='green', alpha=0.2)
  #add / remove nominal pressure band
  #ax11.add_patch(rect)

  nom_peep = float(PE)
  rect = patches.Rectangle((xmin,nom_peep-0.1),xmax-xmin,0.5,edgecolor='None',facecolor='grey', alpha=0.3)
  #add / remove PEEP line
  #ax11.add_patch(rect)

  ax11.set_title ("Test BCIT ", weight='heavy')
  figpath = "%s/%s_%s.pdf" % (output_directory, "BCIT test",  objname.replace('.txt', '')) # TODO: make sure it is correct, or will overwrite!
  print(f'Saving figure to {figpath}')
  fig11.savefig(figpath)




  ####################################################
  '''plot 3 views in a single canvas'''
  ####################################################

  fig31,ax31 = plt.subplots(3,1)
  ax31 = ax31.flatten()

  dftmp.plot(ax=ax31[2], x='dt', y='total_vol',         label='SIM tidal volume       [cl]',        c='r')
  dftmp.plot(ax=ax31[0], x='dt', y='total_flow',        label='SIM flux            [l/min]',        c='r')
  dftmp.plot(ax=ax31[1], x='dt', y='airway_pressure',   label='SIM airway pressure [cmH2O]',        c='r')

  dfvent.plot(ax=ax31[2],  x='dt', y='tidal_volume',    label='MVM tidal volume       [cl]',        c='b')
  dfvent.plot(ax=ax31[0],  x='dt', y='display_flux',            label='MVM flux            [l/min]', c='b')
  dfvent.plot(ax=ax31[1],  x='dt', y='airway_pressure', label='MVM airway pressure [cmH2O]',        c='b')

  ax31[0].set_xlabel("Time [s]")
  ax31[1].set_xlabel("Time [s]")
  ax31[2].set_xlabel("Time [s]")

  fig31.suptitle ("Test BCIT", weight='heavy')
  figpath = "%s/%s_3views_%s.png" % (output_directory, "BCIT test",  objname.replace('.txt', '')) # TODO: make sure it is correct, or will overwrite!
  print(f'Saving figure to {figpath}')
  fig31.savefig(figpath)

