import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib import colors
from scipy.interpolate import interp1d
import matplotlib.patches as patches

from combine_plot_utils import *

def plot_arXiv_canvases (df, dfhd, meta, objname, output_directory, start_times, colors, figure_format, web) :
  ####################################################
  '''formatted plots for ISO std / arXiv'''
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

    fig11,ax11 = plt.subplots()

    #make a subset dataframe for simulator
    print (start_times)
    cycles_to_show = 6
    dftmp = df[ (df['start'] >= start_times[ my_selected_cycle ] ) & ( df['start'] < start_times[ my_selected_cycle + cycles_to_show])  ].copy()
    dftmp.loc[:, 'total_vol'] = dftmp['total_vol'] - dftmp['total_vol'].min()

    #make a subset dataframe for ventilator
    first_time_bin  = dftmp['dt'].iloc[0]
    last_time_bin   = dftmp['dt'].iloc[len(dftmp)-1]
    dfvent = dfhd[ (dfhd['dt']>first_time_bin) & (dfhd['dt']<last_time_bin) ]

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
    title1="R = %i [cmH2O/l/s]         C = %2.1f [ml/cmH2O]         PEEP = %s [cmH2O]"%(RT,CM,PE )
    title2="Inspiration Pressure = %s [cmH2O]       Frequency = %s [breath/min]"%(PI,RR)

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

    set_plot_title(ax11, meta, objname)
    save_figure(fig11, '%icycles'%(cycles_to_show), meta, objname, output_directory, figure_format, web)


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

    set_plot_suptitle(fig31, meta, objname)
    save_figure(fig31, '3views', meta, objname, output_directory, figure_format, web)


    ####################################################
    '''plot 30 cycles in a single canvas'''
    ####################################################
    """
    fig30c,ax30cycles = plt.subplots(3,1)
    ax30cycles = ax30cycles.flatten()
    my_selected_cycle = 10

    print (len(start_times) )
    for ii in range (3) :
      if len (start_times) < (ii+1) *10 : continue

      #make a subset dataframe for simulator
      print (    start_times[ my_selected_cycle + ii*10 ] , start_times[ my_selected_cycle + ii*10 +11 ] )
      dftmp = df[ (df['start'] >= start_times[ my_selected_cycle + ii*10 ] ) & ( df['start'] < start_times[ my_selected_cycle + 11 + ii*10 ])  ].copy()
      print (len (dftmp) , dftmp, dftmp['dt'].iloc[len(dftmp)-1] )
      #make a subset dataframe for ventilator
      first_time_bin  = dftmp['dt'].iloc[0]
      last_time_bin   = dftmp['dt'].iloc[len(dftmp)-1]
      dfvent = dfhd[ (dfhd['dt']>first_time_bin) & (dfhd['dt']<last_time_bin) ]

      dftmp.loc[:, 'total_vol'] = dftmp['total_vol'] - dftmp['total_vol'].min()
      dftmp.plot(ax=ax30cycles[ii], x='dt', y='total_flow',        label='SIM flux            [l/min]', c=colors['total_flow'] ,  legend=False)
      dftmp.plot(ax=ax30cycles[ii], x='dt', y='airway_pressure',   label='SIM airway pressure [cmH2O]', c=colors['sim_airway_pressure'] ,  legend=False)
      dfvent.plot(ax=ax30cycles[ii],  x='dt', y='flux',            label='MVM flux            [l/min]', c=colors['flux']  ,  legend=False)
      dfvent.plot(ax=ax30cycles[ii],  x='dt', y='airway_pressure', label='MVM airway pressure [cmH2O]', c=colors['vent_airway_pressure']   ,  legend=False )
      ax30cycles[ii].set_xlabel("Time [s]")

    '''
    xmin, xmax = ax30cycles.get_xlim()
    ymin, ymax = ax30cycles.get_ylim()
    ax30cycles.text((xmax-xmin)/2.+xmin, 0.08*(ymax-ymin) + ymin,   title2, verticalalignment='bottom', horizontalalignment='center', color='#7697c4')
    ax30cycles.text((xmax-xmin)/2.+xmin, 0.026*(ymax-ymin) + ymin,  title1, verticalalignment='bottom', horizontalalignment='center', color='#7697c4')
    nom_pressure = float(meta[local_objname]["Pinspiratia"])
    rect = patches.Rectangle((xmin,nom_pressure-2),xmax-xmin,4,edgecolor='None',facecolor='green', alpha=0.2)
    ax30cycles.add_patch(rect)

    nom_peep = float(meta[local_objname]["Peep"])
    rect = patches.Rectangle((xmin,nom_peep-0.1),xmax-xmin,0.5,edgecolor='None',facecolor='grey', alpha=0.3)
    ax30cycles.add_patch(rect)
    '''
    set_plot_suptitle(fig30c, meta, objname)
    save_figure(fig30c, '30cycles', meta, objname, output_directory, figure_format, web)
    """
