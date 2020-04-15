import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib import colors
from scipy.interpolate import interp1d
import matplotlib.patches as patches


def plot_service_canvases (df, dfhd, meta, objname, output_directory, start_times, colors, respiration_rate, inspiration_duration) :

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

  ax.set_title ("Test n %s"%meta[objname]['test_name'], weight='heavy')
  figpath = "%s/%s_service_%s.png" % (output_directory, meta[objname]['Campaign'],  objname.replace('.txt', ''))
  print(f'Saving figure to {figpath}')
  plt.savefig(figpath)


  ####################################################
  '''general service canavas number 2, measured simulation parameters'''
  ####################################################

  figbis = plt.figure()
  figbis.suptitle ("Test n %s"%meta[objname]['test_name'], weight='heavy')
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

  figpath = "%s/%s_service2_%s.png" % (output_directory, meta[objname]['Campaign'],  objname.replace('.txt', '')) # TODO: make sure it is correct, or will overwrite!
  print(f'Saving figure to {figpath}')
  figbis.savefig(figpath)
