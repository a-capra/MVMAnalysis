import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mvmconstants import *

def bisec(x):
  return x

def compare_sim_pressures(df):
  fig,ax=plt.subplots(2,1)
  ax=ax.flatten()
  fig.set_size_inches(13,7)

  testname=df['test_name'][0]
  testseries=testname[0:2]+'xx'

  min_plateau = df['mean_plateau'] - df['min_plateau']
  max_plateau = df['max_plateau'] - df['mean_plateau']
  df.plot(kind='scatter', ax=ax[0], x='simulator_plateau', y='mean_plateau', yerr=[min_plateau, max_plateau], c='O2', cmap='winter')
  ax[0].set_ylabel('Plateau Pressure [cmH2O]')
  ax[0].set_xlabel('Simulator Pressure [cmH2O]')

  xpoints=np.linspace(df['simulator_plateau'].min(),df['simulator_plateau'].max(),10)
  ax[0].plot(xpoints,bisec(xpoints),'--',c='Gray')

  min_peak = df['mean_peak'] - df['min_peak']
  max_peak = df['max_peak'] - df['mean_peak']
  df.plot(kind='scatter', ax=ax[1], x='simulator_plateau', y='mean_peak', yerr=[min_peak, max_peak], c='O2', cmap='winter')
  ax[1].set_ylabel('Peak Pressure [cmH2O]')
  ax[1].set_xlabel('Simulator Pressure [cmH2O]')

  fig.tight_layout()
  fig.savefig(f'FiO2test_simpressures_Series{testseries}.png')



def compare_set_pressures(df):
  fig,ax=plt.subplots(3,1)
  ax=ax.flatten()
  fig.set_size_inches(13,10)

  testname=df['test_name'][0]
  testseries=testname[0:2]+'xx'

  min_plateau = df['mean_plateau'] - df['min_plateau']
  max_plateau = df['max_plateau'] - df['mean_plateau']
  df.plot(kind='scatter', ax=ax[0], x='Pinspiratia', y='mean_plateau', yerr=[min_plateau, max_plateau], c='O2', cmap='winter',label='P measured, test'+testseries)
  ax[0].set_ylabel('Plateau Pressure [cmH2O]')

  x_limit = np.arange(df['Pinspiratia'].min()*0.9, df['Pinspiratia'].max()*1.1, 0.01)
  max_limit = MVM.maximum_bias_error_pinsp + (1 + MVM.maximum_linearity_error_pinsp) * x_limit
  min_limit = -MVM.maximum_bias_error_pinsp + (1 - MVM.maximum_linearity_error_pinsp) * x_limit
  ax[0].fill_between(x_limit, min_limit, max_limit, facecolor='gray', alpha=0.2, label='ISO compliance band')

  ax[0].set_xlabel('SetPoint Pressure [cmH2O]')
  ax[0].legend(loc='upper left')

  min_peak = df['mean_peak'] - df['min_peak']
  max_peak = df['max_peak'] - df['mean_peak']
  df.plot(kind='scatter', ax=ax[1], x='Pinspiratia', y='mean_peak', yerr=[min_peak, max_peak], c='O2', cmap='winter',label='P peak, test'+testseries)
  ax[1].set_ylabel('Peak Pressure [cmH2O]')

  x_limit = np.arange(df['Pinspiratia'].min()*0.9, df['Pinspiratia'].max()*1.1, 0.01)
  max_limit = MVM.maximum_bias_error_pinsp + (1 + MVM.maximum_linearity_error_pinsp) * x_limit
  min_limit = -MVM.maximum_bias_error_pinsp + (1 - MVM.maximum_linearity_error_pinsp) * x_limit
  ax[1].fill_between(x_limit, min_limit, max_limit, facecolor='gray', alpha=0.2, label='ISO compliance band')

  ax[1].set_xlabel('SetPoint Pressure [cmH2O]')
  ax[1].legend(loc='upper left')


  min_peep = df['mean_peep'] - df['min_peep']
  max_peep = df['max_peep'] - df['mean_peep']
  df.plot(kind='scatter', ax=ax[2], x='Peep', y='mean_peep', yerr=[min_peep, max_peep], c='O2', cmap='winter',label='PEEP, test'+testseries)
  ax[2].set_ylabel('PEEP [cmH2O]')

  x_limit = np.arange(df['Peep'].min()*0.9, df['Peep'].max()*1.1, 0.01)
  max_limit = MVM.maximum_bias_error_peep + (1 + MVM.maximum_linearity_error_peep) * x_limit
  min_limit = -MVM.maximum_bias_error_peep + (1 - MVM.maximum_linearity_error_peep) * x_limit
  ax[2].fill_between(x_limit, min_limit, max_limit, facecolor='gray', alpha=0.2, label='ISO compliance band')

  ax[2].set_xlabel('PEEP [cmH2O]')
  ax[2].legend(loc='upper left')

  fig.tight_layout()
  fig.savefig(f'FiO2test_setpressures_Series{testseries}.png')



def compare_volumes(df):
  fig,ax=plt.subplots(2,1)
  ax=ax.flatten()
  fig.set_size_inches(13,7)
  
  testname=df['test_name'][0]
  testseries=testname[0:2]+'xx'

  min_volume = df['mean_volume'] - df['min_volume']
  max_volume = df['max_volume'] - df['mean_volume']

  df.plot(kind='scatter', ax=ax[0], x='simulator_volume', y='mean_volume', yerr=[min_volume, max_volume], c='O2', cmap='winter')
  ax[0].set_xlabel('Simulator Volume [cl]')
  ax[0].set_ylabel('Measured Volume [cl]')

  xpoints=np.linspace(df['simulator_volume'].min(),df['simulator_volume'].max(),10)
  ax[0].plot(xpoints,bisec(xpoints),'--',c='Gray')

  df['TVml']=df['Tidal Volume']*0.1
  df.plot(kind='scatter', ax=ax[1], x='TVml', y='mean_volume', yerr=[min_volume, max_volume], c='O2', cmap='winter', label='V measured, test'+testseries)
  ax[1].set_xlabel('Intended Volume [cl]')
  ax[1].set_ylabel('Measured Volume [cl]')
  #x_limit = np.arange(df['simulator_volume'].min()*0.9, df['simulator_volume'].max()*1.1, 0.01)
  x_limit = np.arange(0.0, df['simulator_volume'].max()*1.1, 0.01)
  max_limit = MVM.maximum_bias_error_volume + (1 + MVM.maximum_linearity_error_volume) * x_limit
  min_limit = -MVM.maximum_bias_error_volume + (1 - MVM.maximum_linearity_error_volume) * x_limit
  ax[1].fill_between(x_limit, min_limit, max_limit, facecolor='gray', alpha=0.2, label='ISO compliance band')
  ax[1].legend(loc='upper left')

  fig.tight_layout()
  fig.savefig(f'FiO2test_volume_Series{testseries}.png')



from distances import *

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='FiO2 variation')
  parser.add_argument("input", help="summary file(s) ", nargs='+')
  parser.add_argument("-v", "--no_volume", action='store_true', help="indicate that the intented tidal volume is not specified")

  args=parser.parse_args()

  dataframe=process_files(args.input)
  compare_sim_pressures(dataframe)
  compare_set_pressures(dataframe)

  compare_volumes(dataframe)

  plt.show()
