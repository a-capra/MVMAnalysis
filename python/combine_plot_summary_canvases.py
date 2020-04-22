import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib import colors
from scipy.interpolate import interp1d
import matplotlib.patches as patches

from mvmconstants import *


def plot_summary_canvases (df, dfhd, meta, objname, output_directory, start_times, colors,  measured_peeps, measured_plateaus, real_plateaus, measured_peak, measured_volumes, real_tidal_volumes) :

  for i in range (len(meta)) :

    ####################################################
    '''plot the avg wfs'''
    ####################################################
    local_objname = "%s_%i"% ( objname[:-2] , i )

    PE = meta[local_objname]["Peep"]
    PI = meta[local_objname]["Pinspiratia"]
    RR = meta[local_objname]["Rate respiratio"]
    RT = meta[local_objname]["Resistance"]
    CM = meta[local_objname]["Compliance"]

    nom_peep = float(meta[local_objname]["Peep"])


    
    fig2, ax2 = plt.subplots()
    #make a subset dataframe for simulator
    # This is hardcoded - should it be?
    dftmp = df[ (df['start'] >= start_times[ 4 ] ) & ( df['start'] < start_times[ min ([35,len(start_times)-1] )  ])]
    #dftmp['dtc'] = df['dt'] - df['start']  #HTJI Done in the main program

    #make a subset dataframe for ventilator
    first_time_bin  = dftmp['dt'].iloc[0]
    last_time_bin   = dftmp['dt'].iloc[len(dftmp)-1]
    dfvent = dfhd[ (dfhd['dt']>first_time_bin) & (dfhd['dt']<last_time_bin) ]
    dfvent['dtc'] = dfvent['dt'] - dfvent['start']
    dfvent = dfvent.sort_values('dtc')

    dftmp.loc[:, 'total_vol'] = dftmp['total_vol'] - dftmp['total_vol'].min()

    dftmp.plot(ax=ax2, x='dtc', y='total_vol',         label='SIM tidal volume       [cl]', c=colors['total_vol'] ,          marker='o', markersize=0.3, linewidth=0)
    dftmp.plot(ax=ax2, x='dtc', y='total_flow',        label='SIM flux            [l/min]', c=colors['total_flow'],          marker='o', markersize=0.3, linewidth=0)
    dftmp.plot(ax=ax2, x='dtc', y='airway_pressure',   label='SIM airway pressure [cmH2O]', c=colors['sim_airway_pressure'], marker='o', markersize=0.3, linewidth=0)

    dfvent.plot(ax=ax2,  x='dtc', y='tidal_volume',    label='MVM tidal volume       [cl]', c=colors['tidal_volume'],         marker='o', markersize=0.3, linewidth=0.2)
    dfvent.plot(ax=ax2,  x='dtc', y='display_flux',    label='MVM flux            [l/min]', c=colors['flux'],                 marker='o', markersize=0.3, linewidth=0.2)
    dfvent.plot(ax=ax2,  x='dtc', y='airway_pressure', label='MVM airway pressure [cmH2O]', c=colors['vent_airway_pressure'], marker='o', markersize=0.3, linewidth=0.2)

    
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymin*1.4, ymax*1.5)
    ax2.legend(loc='upper center', ncol=2)
    title1="R = %i [cmH2O/l/s]         C = %2.1f [ml/cmH2O]        PEEP = %s [cmH2O]"%(RT,CM,PE )
    title2="Inspiration Pressure = %s [cmH2O]       Frequency = %s [breath/min]"%(PI,RR)

    ax2.set_xlabel("Time [s]")

    xmin, xmax = ax2.get_xlim()
    ymin, ymax = ax2.get_ylim()
    ax2.text((xmax-xmin)/2.+xmin, 0.08*(ymax-ymin) + ymin,   title2, verticalalignment='bottom', horizontalalignment='center', color='#7697c4')
    ax2.text((xmax-xmin)/2.+xmin, 0.026*(ymax-ymin) + ymin,  title1, verticalalignment='bottom', horizontalalignment='center', color='#7697c4')
    nom_pressure = float(meta[local_objname]["Pinspiratia"])
    rect = patches.Rectangle((xmin,nom_pressure-2),xmax-xmin,4,edgecolor='None',facecolor='green', alpha=0.2)
    ax2.add_patch(rect)

    rect = patches.Rectangle((xmin,nom_peep-0.1),xmax-xmin,0.5,edgecolor='None',facecolor='grey', alpha=0.3)
    ax2.add_patch(rect)

    ax2.set_title ("Test n %s"%meta[objname]['test_name'])
    figpath = "%s/%s_avg_%s.png" % (output_directory, meta[objname]['Campaign'] , objname.replace('.txt', '')) # TODO: make sure it is correct, or will overwrite!
    print(f'Saving figure to {figpath}')
    fig2.savefig(figpath)
    
    mean_peep    =   meta[objname]["mean_peep"]
    mean_plateau =   meta[objname]["mean_plateau"]
    mean_peak    =   meta[objname]["mean_peak"]
    mean_volume  =   meta[objname]["mean_volume"]
    rms_peep     =   meta[objname]["rms_peep"]
    rms_plateau  =   meta[objname]["rms_plateau"]
    rms_peak     =   meta[objname]["rms_peak"]
    rms_volume   =   meta[objname]["rms_volume"]
    max_peep     =   meta[objname]["max_peep"]
    max_plateau  =   meta[objname]["max_plateau"]
    max_peak     =   meta[objname]["max_peak"]
    max_volume   =   meta[objname]["max_volume"]
    min_peep     =   meta[objname]["min_peep"]
    min_plateau  =   meta[objname]["min_plateau"]
    min_peak     =   meta[objname]["min_peak"]
    min_volume   =   meta[objname]["min_volume"]
    simulator_volume  = meta[objname]["simulator_volume"]
    simulator_plateau = meta[objname]["simulator_plateau"]


    ####################################################
    '''summary plots'''
    ####################################################

    figs,axes = plt.subplots(2,2)
    figs.suptitle ("Test n %s"%meta[objname]['test_name'], weight='heavy')
    axs = axes.flatten()

    ## MVM PEEP compared with set value
    nom_peep_low = nom_peep - MVM.maximum_bias_error_peep - MVM.maximum_linearity_error_peep * nom_peep
    nom_peep_wid = 2 * (MVM.maximum_bias_error_peep + MVM.maximum_linearity_error_peep * nom_peep)
    axs[0].hist ( measured_peeps  , bins=50,  range=(  min([ mean_peep,nom_peep] )*0.6 , max( [mean_peep,nom_peep] ) *1.4  )   )
    aa = patches.Rectangle( (nom_peep_low, axs[0].get_ylim()[0]  ) , nom_peep_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[0].add_patch(aa)
    axs[0].set_title("PEEP [cmH2O], nominal: %2.1f [cmH2O]"%nom_peep, weight='heavy', fontsize=10)

    ## MVM Pinsp compared with set value
    nominal_plateau = meta[objname]["Pinspiratia"]
    nominal_plateau_low = nominal_plateau - MVM.maximum_bias_error_pinsp - MVM.maximum_linearity_error_pinsp * nominal_plateau
    nominal_plateau_wid = 2 * (MVM.maximum_bias_error_pinsp + MVM.maximum_linearity_error_pinsp * nominal_plateau)
    _range = (   min([ mean_plateau,nominal_plateau] )*0.8 , max( [mean_plateau,nominal_plateau] ) *1.3  )
    axs[1].hist ( measured_plateaus, bins=100, range=_range   )
    aa = patches.Rectangle( (nominal_plateau_low, axs[0].get_ylim()[0]  ) , nominal_plateau_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[1].add_patch(aa)
    axs[1].set_title("plateau [cmH2O], nominal: %s [cmH2O]"%nominal_plateau, weight='heavy', fontsize=10)

    ## MVM Pinsp compared with simulator values
    simulator_plateau_low = simulator_plateau - MVM.maximum_bias_error_pinsp - MVM.maximum_linearity_error_pinsp * simulator_plateau
    simulator_plateau_wid = 2 * (MVM.maximum_bias_error_pinsp + MVM.maximum_linearity_error_pinsp * simulator_plateau)
    #print (measured_plateaus, mean_plateau, simulator_plateau )
    _range = ( min([ mean_plateau,simulator_plateau] )*0.7 , max( [mean_plateau,simulator_plateau] ) *1.4    )
    axs[2].hist (   measured_plateaus, bins=100, range=_range, label='MVM')
    axs[2].hist (  real_plateaus , bins=100, range=_range,  label='SIM', alpha=0.7)
    aa = patches.Rectangle( (simulator_plateau_low, axs[0].get_ylim()[0]  ) , simulator_plateau_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[2].set_title("plateau [cmH2O], <SIM>: %2.1f [cmH2O]"%(simulator_plateau), weight='heavy', fontsize=10 )
    axs[2].legend(loc='upper left')
    axs[2].add_patch(aa)

    ## MVM tidal volumes compared with simulator values
    MVM_maximum_bias_error_volume_cl = MVM.maximum_bias_error_volume * 0.1   # ml to cl
    simulator_volume_low = simulator_volume - MVM_maximum_bias_error_volume_cl - MVM.maximum_linearity_error_volume * simulator_volume
    simulator_volume_wid = 2 * (MVM_maximum_bias_error_volume_cl + MVM.maximum_linearity_error_volume * simulator_volume)
    _range = ( min([ mean_volume,simulator_volume] )*0.7 , max( [mean_volume,simulator_volume] ) *1.4    )
    axs[3].hist ( measured_volumes  , bins=100, range=_range, label='MVM')
    axs[3].hist ( real_tidal_volumes , range=_range, bins= 100 , label='SIM', alpha=0.7)
    aa = patches.Rectangle( (simulator_volume_low, axs[0].get_ylim()[0]  ) , simulator_volume_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[3].set_title("Tidal Volume [cl], <SIM>: %2.1f [cl], nominal %2.1f [cl]"%(simulator_volume, float( meta[objname]['Tidal Volume'])/10), weight='heavy', fontsize=10)
    axs[3].legend(loc='upper left')
    axs[3].add_patch(aa)

    figpath = "%s/%s_summary_%s.png" % (output_directory, meta[objname]['Campaign'], objname.replace('.txt', '')) # TODO: make sure it is correct, or will overwrite!
    print(f'Saving figure to {figpath}')
    figs.savefig(figpath)

    ## Debug output
    #print("measured_peeps:", measured_peeps)
    #print("measured_plateaus:", measured_plateaus)
    #print("real_plateaus:", real_plateaus)
    #print("measured_volumes:", measured_volumes)
    #print("real_tidal_volumes:", real_tidal_volumes)

    ## Print test result, based on comparisons with maximum errors
    if min_peep > nom_peep_low and max_peep < nom_peep_low + nom_peep_wid:
      print("SUCCESS: PEEP all values within maximum errors")
    else:
      print("FAILURE: PEEP outside maximum errors")
    if min_plateau > nominal_plateau_low and max_plateau < nominal_plateau_low + nominal_plateau_wid:
      print("SUCCESS: Pinsp all values within maximum errors")
    else:
      print("FAILURE: Pinsp outside maximum errors")



def plot_overlay_canvases (dftmp, dfhd, meta, objname, output_directory, start_times, colors, stats_total_vol, stats_total_flow, stats_airway_pressure ) :

    figoverlay, axoverlay = plt.subplots(6)
    figoverlay.set_size_inches(7,9)
    figoverlay.suptitle ("Test n %s Consistency of Cycles"%meta[objname]['test_name'], weight='heavy', fontsize=14)

    axoverlay[4].set_ylabel('Total Vol',fontsize=10)
    axoverlay[4].set_xlim(0,4)
    dftmp.plot(ax=axoverlay[4], kind='scatter', x='dtc', y='total_vol', color = colors['total_vol'],fontsize=10)
    axoverlay[5].set_xlabel('Time since start of cycle (s)',fontsize=14)
    axoverlay[5].set_xlim(0,4)
    axoverlay[5].set_ylim(-0.2,0.2)
#    axoverlay[5].legend(loc='upper right', title_fontsize=10, fontsize=10, title='Frac diff from median')
    stats_total_vol['max_minus_median']=  (stats_total_vol['max'] - stats_total_vol['median'])/stats_total_vol['median']
    stats_total_vol.plot(ax=axoverlay[5], kind='line', x='dtc', y='max_minus_median', label='_a', color = colors['total_vol'], linewidth=1,fontsize=10)
    stats_total_vol['min_minus_median']=  (stats_total_vol['min'] - stats_total_vol['median'])/stats_total_vol['median']
    stats_total_vol.plot(ax=axoverlay[5], kind='line', x='dtc', y='min_minus_median', label='_b', color = colors['total_vol'], linewidth=1)

    #
    axoverlay[0].set_ylabel('Total Flow',fontsize=10)
    axoverlay[0].set_xlim(0,4)
    dftmp.plot(ax=axoverlay[0], kind='scatter', x='dtc', y='total_flow', color = colors['total_flow'],fontsize=10)
    axoverlay[1].set_xlim(0,4)
    axoverlay[1].set_ylim(-0.2,0.2)
    stats_total_flow['max_minus_median']=  (stats_total_flow['max'] - stats_total_flow['median'])/stats_total_flow['median']
    stats_total_flow.plot(ax=axoverlay[1], kind='line', x='dtc', y='max_minus_median', label='_a',color = colors['total_flow'], linewidth=1,fontsize=10)
    stats_total_flow['min_minus_median']=  (stats_total_flow['min'] - stats_total_flow['median'])/stats_total_flow['median']
    stats_total_flow.plot(ax=axoverlay[1], kind='line', x='dtc', y='min_minus_median', label='_b',color = colors['total_flow'], linewidth=1)
#
    axoverlay[2].set_ylabel('Pressure',fontsize=10)
    axoverlay[2].set_xlim(0,4)
    dftmp.plot(ax=axoverlay[2], kind='scatter', x='dtc', y='airway_pressure', color = colors['pressure'],fontsize=10)
    axoverlay[3].set_xlim(0,4)
    axoverlay[3].set_ylim(-0.2,0.2)
    stats_airway_pressure['max_minus_median']=  (stats_airway_pressure['max'] - stats_airway_pressure['median'])/stats_airway_pressure['median']
    stats_airway_pressure.plot(ax=axoverlay[3], kind='line', x='dtc', y='max_minus_median', label='_a',color = colors['pressure'], linewidth=1,fontsize=10)
    stats_airway_pressure['min_minus_median']=  (stats_airway_pressure['min'] - stats_airway_pressure['median'])/stats_airway_pressure['median']
    stats_airway_pressure.plot(ax=axoverlay[3], kind='line', x='dtc', y='min_minus_median', label='_b', color = colors['pressure'], linewidth=1)



    figpath = "%s/%s_overlay_%s.png" % (output_directory, meta[objname]['Campaign'], objname.replace('.txt', '')) # TODO: make sure it is correct, or will overwrite!
    figoverlay.savefig(figpath)

