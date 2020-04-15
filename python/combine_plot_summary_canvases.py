import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib import colors
from scipy.interpolate import interp1d
import matplotlib.patches as patches


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

    fig2, ax2 = plt.subplots()
    #make a subset dataframe for simulator
    dftmp = df[ (df['start'] >= start_times[ 4 ] ) & ( df['start'] < start_times[ min ([35,len(start_times)-1] )  ])]
    dftmp['dtc'] = df['dt'] - df['start']

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
    simulator_volume  = meta[objname]["simulator_volume"]
    simulator_plateau = meta[objname]["simulator_plateau"]


    ####################################################
    '''summary plots'''
    ####################################################

    figs,axes = plt.subplots(2,2)
    figs.suptitle ("Test n %s"%meta[objname]['test_name'], weight='heavy')

    axs = axes.flatten()
    #axs.set_title("PEEP", "", "a", "")
    nom_peep_low = nom_peep - 2 - 0.04 * nom_peep
    nom_peep_wid = 4 + 0.08 * nom_peep
    axs[0].hist ( measured_peeps  , bins=50,  range=(  min([ mean_peep,nom_peep] )*0.6 , max( [mean_peep,nom_peep] ) *1.4  )   )
    aa = patches.Rectangle( (nom_peep_low, axs[0].get_ylim()[0]  ) , nom_peep_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[0].add_patch(aa)
    axs[0].set_title("PEEP [cmH20], nominal: %i [cmH20]"%nom_peep,weight='heavy', fontsize=10)

    nominal_plateau = meta[objname]["Pinspiratia"]
    nominal_plateau_low = nominal_plateau - 2 - 0.04 * nominal_plateau
    nominal_plateau_wid = 4 + 0.08 * nominal_plateau
    _range = (   min([ mean_plateau,nominal_plateau] )*0.8 , max( [mean_plateau,nominal_plateau] ) *1.3  )
    axs[1].hist ( measured_plateaus, bins=100, range=_range   )
    aa = patches.Rectangle( (nominal_plateau_low, axs[0].get_ylim()[0]  ) , nominal_plateau_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[1].add_patch(aa)
    axs[1].set_title("plateau [cmH20], nominal: %s [cmH20]"%nominal_plateau, weight='heavy', fontsize=10)


    nominal_plateau     = simulator_plateau
    nominal_plateau_low = nominal_plateau - 2 - 0.04 * nominal_plateau
    nominal_plateau_wid = 4 + 0.08 * nominal_plateau
    #print (measured_plateaus, mean_plateau, nominal_plateau )
    _range = ( min([ mean_plateau,nominal_plateau] )*0.7 , max( [mean_plateau,nominal_plateau] ) *1.4    )
    axs[2].hist (   measured_plateaus, bins=100, range=_range, label='MVM')
    axs[2].hist (  real_plateaus , bins=100, range=_range,  label='SIM', alpha=0.7)
    aa = patches.Rectangle( (nominal_plateau_low, axs[0].get_ylim()[0]  ) , nominal_plateau_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[2].set_title("plateau [cmH2O], <SIM>: %2.1f [cmH2O]"%(nominal_plateau), weight='heavy', fontsize=10 )
    axs[2].legend(loc='upper left')
    axs[2].add_patch(aa)

    nominal_volume     =  simulator_volume
    #print (nominal_volume)
    nominal_volume_low = nominal_volume - 4 - 0.15 * nominal_volume
    nominal_volume_wid = 8 + 0.3 * nominal_volume
    _range = ( min([ mean_volume,nominal_volume] )*0.7 , max( [mean_volume,nominal_volume] ) *1.4    )
    axs[3].hist ( measured_volumes  , bins=100, range=_range, label='MVM')
    axs[3].hist ( real_tidal_volumes , range=_range, bins= 100 , label='SIM', alpha=0.7)
    aa = patches.Rectangle( (nominal_volume_low, axs[0].get_ylim()[0]  ) , nominal_volume_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
    axs[3].set_title("TV [cl], <SIM>: %2.1f [cl], nominal %i [cl]"%(nominal_volume,int ( meta[objname]['Tidal Volume'])/10), weight='heavy', fontsize=10)
    axs[3].legend(loc='upper left')
    axs[3].add_patch(aa)

    figpath = "%s/%s_summary_%s.png" % (output_directory, meta[objname]['Campaign'], objname.replace('.txt', '')) # TODO: make sure it is correct, or will overwrite!
    figs.savefig(figpath)
