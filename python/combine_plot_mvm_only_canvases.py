import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib import colors
from scipy.interpolate import interp1d
import matplotlib.patches as patches


def plot_mvm_only_canvases (dfhd, meta, objname, output_directory, start_times, colors) :

    ####################################################
    '''general service canavas number 1'''
    ####################################################
    ax = dfhd.plot(x='dt', y='out',         label='control out', c=colors['total_vol'] , alpha=0.4)
    plt.plot(start_times, [0]*len(start_times), 'bo', label='real cycle start time')
    #dfhd.plot(ax=ax,  x='dt', y='tidal_volume',    label='MVM tidal volume       [cl]', c=colors['tidal_volume'])

    #dfhd.plot(ax=ax, x='dt', y='pressure', label='ventilator pressure [cmH2O]', c=colors['pressure'], linewidth = linw)
    dfhd.plot(ax=ax, x='dt', y='airway_pressure', label='ventilator airway pressure [cmH2O]', c=colors['vent_airway_pressure'])
    dfhd.plot(ax=ax, x='dt', y='flux',            label='ventilator flux (SP1)      [l/min]', c=colors['flux'] )
    #dfhd.plot(ax=ax, x='dt', y='volume',          label='volume            [l/min]', c=colors['flux'] )
  #  dfhd.plot(ax=ax, x='dt', y='out', label='out')
    dfhd.plot(ax=ax, x='dt', y='in', label='control in', c='green')
    #dfhd.plot(ax=ax, x='dt', y='service_1', label='service 2', c='black', linestyle="--")
    #dfhd.plot(ax=ax, x='dt', y='service_2', label='service 1', c='black')
    #dfhd.plot(ax=ax, x='dt', y='flux_2', label='flux_2', c='r', linestyle="--")
    #dfhd.plot(ax=ax, x='dt', y='flux_3',  label='flux_3', c='r')
    #dfhd.plot(ax=ax, x='dt', y='derivative',  label='derivative', c='gray')
    #df.plot(ax=ax, x='dt', y='deriv_total_vol', label='deriv_total_vol [l/min]')

    ax.set_title ("Filename: %s"%meta[objname]['test_name'], weight='heavy')

    for i,t in enumerate(start_times) :
      ax.text(t, 0.5, "%i"%i, verticalalignment='bottom', horizontalalignment='center', color='black', fontsize=14)
    ax.legend(loc='upper center', ncol=2)

    ax.set_title ("Test n %s"%meta[objname]['test_name'], weight='heavy')
    figpath = "%s/%s_mvmonly_%s.png" % (output_directory, meta[objname]['Campaign'],  objname.replace('.txt', ''))
    print(f'Saving figure to {figpath}')
    plt.savefig(figpath)
