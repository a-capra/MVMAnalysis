import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib import colors
from scipy.interpolate import interp1d
import matplotlib.patches as patches

from mvmconstants import *

def plot_histo (dfhd, objname, output_directory, meta, tag,
                respiration_rate, inspiration_duration, 
                measured_peeps, measured_plateaus, real_plateaus, measured_peak, measured_volumes, real_tidal_volumes) :
  
  nom_peep = meta["PEEP"]
  nominal_plateau = meta["P"]
  RR = meta["RR"]
  RT = meta["R"]
  CM = meta["C"]

  mean_peep    = np.mean(measured_peeps)

  figs,axes = plt.subplots(2,2)
  axs = axes.flatten()

  ## MVM PEEP compared with set value
  nom_peep_low = nom_peep - MVM.maximum_bias_error_peep - MVM.maximum_linearity_error_peep * nom_peep
  nom_peep_wid = 2 * (MVM.maximum_bias_error_peep + MVM.maximum_linearity_error_peep * nom_peep)
  axs[0].hist ( measured_peeps  , bins=50,  range=(  min([ mean_peep,nom_peep] )*0.6 , max( [mean_peep,nom_peep] ) *1.4  )   )
  aa = patches.Rectangle( (nom_peep_low, axs[0].get_ylim()[0]  ) , nom_peep_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
  axs[0].add_patch(aa)
  axs[0].set_title("PEEP [cmH2O], nominal: %2.1f [cmH2O]"%nom_peep, weight='heavy', fontsize=10)

  ## MVM Pinsp compared with set value
  __range = (0.,40.)
  axs[1].hist ( measured_plateaus, bins=100, range=__range   )
  nominal_plateau_low = nominal_plateau - MVM.maximum_bias_error_pinsp - MVM.maximum_linearity_error_pinsp * nominal_plateau
  nominal_plateau_wid = 2 * (MVM.maximum_bias_error_pinsp + MVM.maximum_linearity_error_pinsp * nominal_plateau)
  aa = patches.Rectangle( (nominal_plateau_low, axs[0].get_ylim()[0]  ) , nominal_plateau_wid , axs[0].get_ylim()[1] , edgecolor='red' , facecolor='green' , alpha=0.2)
  axs[1].add_patch(aa)
  axs[1].set_title("plateau [cmH2O], nominal: %s [cmH2O]"%nominal_plateau, weight='heavy', fontsize=10)

  xmin, xmax = axs[1].get_xlim()
  ymin, ymax = axs[1].get_ylim()
  mytext = "Measured respiration rate: %1.1f br/min, inspiration duration: %1.1f s\nNominal Respiratory rate: %1.1f"%(respiration_rate, inspiration_duration,RR)
  axs[1].text((xmax-xmin)/2.+xmin, 0.08*(ymax-ymin) + ymin,   mytext, verticalalignment='bottom', horizontalalignment='center', color='black')

  axs[2].set_title("compliance [mL/cmH2O], nominal: %s [mL/cmH2O]"%CM, weight='heavy', fontsize=10)
  axs[2].hist ( dfhd[( dfhd['compliance']>0) ]['compliance'].unique()  , bins=50)
  axs[2].set_xlabel("Measured compliance [ml/cmH2O]")

  axs[3].set_title("resistance [cmH2O/L/s], nominal: %s [cmH2O/L/s]"%RT, weight='heavy', fontsize=10)
  axs[3].hist ( dfhd[( dfhd['resistance']>0)]['resistance'].unique() , bins=50 )
  axs[3].set_xlabel("Measured resistance [cmH2O/l/s]")

  figs.set_size_inches(15,8)
  #ttl=' '.join([k+str(meta[k]) for k in meta])
  figs.suptitle(objname.replace('.json',''), weight='heavy')
  figpath = "%s/%s_histo_%s.png" % (output_directory, tag,  objname.replace('.json', ''))
  print(f'Saving figure to {figpath}')
  figs.tight_layout()
  figs.savefig(figpath)
