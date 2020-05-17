import numpy as np
import pandas as pd
import logging as log
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib import colors
from scipy.interpolate import interp1d
import matplotlib.patches as patches

from mvmconstants import *
from combine_plot_utils import *

def plot_summary_canvases (df, dfhd, meta, objname, output_directory, start_times, colors, figure_format, web, measured_peeps, measured_plateaus, real_plateaus, measured_peaks, measured_volumes, measured_IoverE, measured_Frequency, real_tidal_volumes, real_IoverE, real_Frequency) :

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
    IE = meta[local_objname]["I:E"]

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
    ax2legend = ax2.legend(loc='upper center', ncol=2)
    ## hack to set larger marker size in legend only, one line per data series
    legmarkersize = 10
    for iplot in range(6):
      ax2legend.legendHandles[iplot]._legmarker.set_markersize(legmarkersize)

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

    set_plot_title(ax2, meta, objname)
    save_figure(fig2, 'avg', meta, objname, output_directory, figure_format, web)

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
    simulator_iovere  = meta[objname]["simulator_iovere"]
    simulator_frequency = meta[objname]["simulator_frequency"]


    ####################################################
    '''summary plots'''
    ####################################################

    figs,axes = plt.subplots(3,2)
    plt.subplots_adjust(hspace=0.3)
    axs = axes.flatten()

    ## MVM PEEP compared with set value
    nom_peep_low = nom_peep - MVM.maximum_bias_error_peep - MVM.maximum_linearity_error_peep * nom_peep
    nom_peep_wid = 2 * (MVM.maximum_bias_error_peep + MVM.maximum_linearity_error_peep * nom_peep)
    axs[0].hist ( measured_peeps  , bins=50,  range=(  min([ mean_peep,nom_peep] )*0.6 , max( [mean_peep,nom_peep] ) *1.4  )   )
    axs[0].tick_params(axis='both', which='major', labelsize=8)
    axs[0].tick_params(axis='both', which='minor', labelsize=8)
    aa = patches.Rectangle( (nom_peep_low, axs[0].get_ylim()[0]  ) , nom_peep_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[0].add_patch(aa)
    axs[0].set_title("PEEP [cmH2O], nominal: %2.1f [cmH2O]"%nom_peep, weight='heavy', fontsize=10)

    ## MVM Pinsp compared with set value
    nominal_plateau = meta[objname]["Pinspiratia"]
    nominal_plateau_low = nominal_plateau - MVM.maximum_bias_error_pinsp - MVM.maximum_linearity_error_pinsp * nominal_plateau
    nominal_plateau_wid = 2 * (MVM.maximum_bias_error_pinsp + MVM.maximum_linearity_error_pinsp * nominal_plateau)
    _range = (   min([ mean_plateau,nominal_plateau] )*0.8 , max( [mean_plateau,nominal_plateau] ) *1.3  )
    axs[1].hist ( measured_plateaus, bins=100, range=_range   )
    axs[1].tick_params(axis='both', which='major', labelsize=8)
    axs[1].tick_params(axis='both', which='minor', labelsize=8)
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
    axs[2].tick_params(axis='both', which='major', labelsize=8)
    axs[2].tick_params(axis='both', which='minor', labelsize=8)
    aa = patches.Rectangle( (simulator_plateau_low, axs[0].get_ylim()[0]  ) , simulator_plateau_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[2].set_title("plateau [cmH2O], <SIM>: %2.1f [cmH2O]"%(simulator_plateau), weight='heavy', fontsize=10 )
    axs[2].legend(loc='upper left', fontsize=10)
    axs[2].add_patch(aa)

    ## MVM tidal volumes compared with simulator values
    MVM_maximum_bias_error_volume_cl = MVM.maximum_bias_error_volume * 0.1   # ml to cl
    simulator_volume_low = simulator_volume - MVM_maximum_bias_error_volume_cl - MVM.maximum_linearity_error_volume * simulator_volume
    simulator_volume_wid = 2 * (MVM_maximum_bias_error_volume_cl + MVM.maximum_linearity_error_volume * simulator_volume)
    _range = ( min([ mean_volume,simulator_volume] )*0.7 , max( [mean_volume,simulator_volume] ) *1.4    )
    axs[3].hist ( measured_volumes  , bins=100, range=_range, label='MVM')
    axs[3].hist ( real_tidal_volumes , range=_range, bins= 100 , label='SIM', alpha=0.7)
    axs[3].tick_params(axis='both', which='major', labelsize=8)
    axs[3].tick_params(axis='both', which='minor', labelsize=8)
    aa = patches.Rectangle( (simulator_volume_low, axs[0].get_ylim()[0]  ) , simulator_volume_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[3].set_title("Tidal Volume [cl], <SIM>: %2.1f [cl], nominal %2.1f [cl]"%(simulator_volume, float( meta[objname]['Tidal Volume'])/10), weight='heavy', fontsize=10)
    axs[3].legend(loc='upper left', fontsize=10)
    axs[3].add_patch(aa)

    ## MVM E:I compared compared to simulator values
    simulator_eoveri_low = 0.
    simulator_eoveri_wid = IE * 2.
    _range = ( simulator_eoveri_low , simulator_eoveri_wid )
    axs[4].hist ( measured_IoverE  , bins=100, range=_range, label='MVM')
    axs[4].hist ( real_IoverE , range=_range, bins= 100 , label='SIM', alpha=0.7)
    axs[4].tick_params(axis='both', which='major', labelsize=8)
    axs[4].tick_params(axis='both', which='minor', labelsize=8)
    axs[4].legend(loc='upper left', fontsize=10)
    axs[4].set_title("I:E , <SIM>: %2.2f, nominal %2.2f "%(simulator_iovere, float(IE)), weight='heavy', fontsize=10)
    aa = patches.Rectangle( (IE*0.9, axs[4].get_ylim()[0] ), IE*0.2 , axs[4].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[4].add_patch(aa)

    ## MVM frequency compared to simulator values
    simulator_RR_low = RR - 5
    simulator_RR_wid = RR + 5
    _range = ( simulator_RR_low, simulator_RR_wid )
    axs[5].hist ( measured_Frequency  , bins=100, range=_range, label='MVM')
    axs[5].hist ( real_Frequency , range=_range, bins= 100 , label='SIM', alpha=0.7)
    axs[5].tick_params(axis='both', which='major', labelsize=8)
    axs[5].tick_params(axis='both', which='minor', labelsize=8)
    axs[5].set_title("Frequency [breaths/min], <SIM>: %2.1f [breaths/min], nominal %2.1f [breaths/min]"%(simulator_frequency, float( RR )), weight='heavy', fontsize=10)
    axs[5].legend(loc='upper left', fontsize=10)
    aa = patches.Rectangle( (RR*0.9, axs[5].get_ylim()[0]  ) , RR*0.2 , axs[5].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[5].add_patch(aa)

    set_plot_suptitle(figs, meta, objname)
    save_figure(figs, 'summary', meta, objname, output_directory, figure_format, web)

    ## Debug output
    #print("measured_peeps:", measured_peeps)
    #print("measured_plateaus:", measured_plateaus)
    #print("real_plateaus:", real_plateaus)
    #print("measured_volumes:", measured_volumes)
    #print("real_tidal_volumes:", real_tidal_volumes)

    ## Print test results, based on comparisons with maximum errors
    if min_peep > nom_peep_low and max_peep < nom_peep_low + nom_peep_wid:
      print("SUCCESS: PEEP all values within maximum errors")
    else:
      print("FAILURE: PEEP outside maximum errors")
    if min_plateau > nominal_plateau_low and max_plateau < nominal_plateau_low + nominal_plateau_wid:
      print("SUCCESS: Pinsp all values within maximum errors")
    else:
      print("FAILURE: Pinsp outside maximum errors")
    if min_volume > simulator_volume_low and max_volume < simulator_volume_low + simulator_volume_wid:
      print("SUCCESS: Volume all values within maximum errors wrt simulator")
    else:
      print("FAILURE: Volume outside maximum errors wrt simulator")


def plot_overlay_canvases (dftmp, dfhd, meta, objname, output_directory, start_times, colors, figure_format, web, stats_total_vol, stats_total_flow, stats_airway_pressure ) :

  ## For the moment only one test per file is supported here
  if len(meta) != 1 :
    log.warning("The length of the meta array is not 1. Assumption made in plot_overlay_canvases is invalid.")

  local_objname = "%s_%i"% ( objname[:-2] , 0 )  # i = 0

  PE = meta[local_objname]["Peep"]
  PI = meta[local_objname]["Pinspiratia"]
  RR = meta[local_objname]["Rate respiratio"]
  RT = meta[local_objname]["Resistance"]
  CM = meta[local_objname]["Compliance"]

  n_cycles = 0
  temp_shape = stats_total_vol.shape
  if temp_shape[0] > 0:
    this_series = stats_total_vol.iloc[0]
    n_cycles = this_series['N']

    figoverlay, axoverlay = plt.subplots(6)
    figoverlay.set_size_inches(7,9)

    title1="R = %i [cmH2O/l/s]         C = %2.1f [ml/cmH2O]         PEEP = %s [cmH2O]"%(RT,CM,PE )
    title2="Inspiration Pressure = %s [cmH2O]       Frequency = %s [breath/min]"%(PI,RR)
    figoverlay.text(0.5, 0.93, title1, color='#7697c4', fontsize=10, ha='center')
    figoverlay.text(0.5, 0.90, title2, color='#7697c4', fontsize=10, ha='center')

    axoverlay[4].set_ylabel('Total Vol',fontsize=10)
    axoverlay[4].set_xlim(0,5)
    dftmp.plot(ax=axoverlay[4], kind='scatter', x='dtc', y='total_vol', color = colors['total_vol'],fontsize=10,marker='+', s=2.0)
    axoverlay[5].set_xlabel('Time since start of cycle (s)',fontsize=14)
    axoverlay[5].set_xlim(0,5)
    axoverlay[5].set_ylim(-0.2,0.2)
    stats_total_vol['max_minus_median']=  (stats_total_vol['max'] - stats_total_vol['median'])/stats_total_vol['median']
    stats_total_vol.plot(ax=axoverlay[5], kind='line', x='dtc', y='max_minus_median', color = colors['total_vol'], linewidth=1,fontsize=10)
    stats_total_vol['min_minus_median']=  (stats_total_vol['min'] - stats_total_vol['median'])/stats_total_vol['median']
    stats_total_vol.plot(ax=axoverlay[5], kind='line', x='dtc', y='min_minus_median', color = colors['total_vol'], linewidth=1)
    #axoverlay[5].legend(loc='upper right', title_fontsize=10, fontsize=10, title='Frac diff from median')
    axoverlay[5].get_legend().remove()
    axoverlay[5].text(4.95,0.1, "Max/min frac", ha='right', fontsize=8)
    axoverlay[5].text(4.95,0.0, "deviation", ha='right', fontsize=8)
    axoverlay[5].text(4.95,-0.1, "from median", ha='right', fontsize=8)
    axoverlay[5].set_xlabel('Time from start of cycle [s]', fontsize=10)

    axoverlay[0].set_ylabel('Total Flow',fontsize=10)
    axoverlay[0].set_xlim(0,5)
    dftmp.plot(ax=axoverlay[0], kind='scatter', x='dtc', y='total_flow', color = colors['total_flow'],fontsize=10,marker='+',s=4.0)
    axoverlay[1].set_xlim(0,5)
    axoverlay[1].set_ylim(-0.2,0.2)
    stats_total_flow['max_minus_median']=  (stats_total_flow['max'] - stats_total_flow['median'])/stats_total_flow['median']
    stats_total_flow.plot(ax=axoverlay[1], kind='line', x='dtc', y='max_minus_median', color = colors['total_flow'], linewidth=1,fontsize=10)
    stats_total_flow['min_minus_median']=  (stats_total_flow['min'] - stats_total_flow['median'])/stats_total_flow['median']
    stats_total_flow.plot(ax=axoverlay[1], kind='line', x='dtc', y='min_minus_median', color = colors['total_flow'], linewidth=1)
    axoverlay[1].get_legend().remove()
    axoverlay[1].text(4.95,0.1, "Max/min frac", ha='right', fontsize=8)
    axoverlay[1].text(4.95,0.0, "deviation", ha='right', fontsize=8)
    axoverlay[1].text(4.95,-0.1, "from median", ha='right', fontsize=8)


    axoverlay[2].set_ylabel('Pressure',fontsize=10)
    axoverlay[2].set_xlim(0,5)
    dftmp.plot(ax=axoverlay[2], kind='scatter', x='dtc', y='airway_pressure', color = colors['pressure'],fontsize=10,marker='+',s=4.0)
    axoverlay[3].set_xlim(0,5)
    axoverlay[3].set_ylim(-0.2,0.2)
    stats_airway_pressure['max_minus_median']=  (stats_airway_pressure['max'] - stats_airway_pressure['median'])/stats_airway_pressure['median']
    stats_airway_pressure.plot(ax=axoverlay[3], kind='line', x='dtc', y='max_minus_median', color = colors['pressure'], linewidth=1,fontsize=10)
    stats_airway_pressure['min_minus_median']=  (stats_airway_pressure['min'] - stats_airway_pressure['median'])/stats_airway_pressure['median']
    stats_airway_pressure.plot(ax=axoverlay[3], kind='line', x='dtc', y='min_minus_median', color = colors['pressure'], linewidth=1)
    axoverlay[3].get_legend().remove()
    axoverlay[3].text(4.95,0.1, "Max/min frac", ha='right', fontsize=8)
    axoverlay[3].text(4.95,0.0, "deviation", ha='right', fontsize=8)
    axoverlay[3].text(4.95,-0.1, "from median", ha='right', fontsize=8)

    figoverlay.suptitle ("Test n %s Consistency of %s Cycles"%(meta[objname]['test_name'],n_cycles), weight='heavy', fontsize=12)
    #set_plot_suptitle(figoverlay, meta, objname)  #FIXME need to show "Consistency of Cycles" as well
    save_figure(figoverlay, 'overlay', meta, objname, output_directory, figure_format, web)
