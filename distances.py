import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


index = ['Compliance', 'Resistance', 'Rate respiratio', 'I:E', 'Peep', 'Date', 'Run', 'Pinspiratia', 'SimulatorFileName', 'Campaign', 'MVM_filename', 'test_name', 'Tidal Volume', 'leakage', 'cycle_index', 'SiteName', 'mean_peep', 'rms_peep', 'max_peep', 'min_peep', 'mean_plateau', 'rms_plateau', 'max_plateau', 'min_plateau', 'mean_peak', 'rms_peak', 'max_peak', 'min_peak', 'mean_volume', 'rms_volume', 'max_volume', 'min_volume', 'simulator_volume', 'simulator_plateau']

def process_files(jfiles):
  dfs=[]
  for i, fname in enumerate(jfiles):
    dfs.append(pd.DataFrame(json.loads(open(fname).read()), index=[i]))
    df=pd.concat(dfs)
  return df

def calculate_distances(df):
  df['peep_diff'] = df['Peep'] - df['mean_peep']
  df['p_diff']    = df['Pinspiratia'] - df['mean_plateau']
  df['pk_diff']   = df['Pinspiratia'] - df['mean_peak']
  df['v_diff']    = df['Tidal Volume'] - df['mean_volume']*10.0

  df['peep_maxdiff'] = df['Peep'] - df['max_peep']
  df['p_maxdiff']    = df['Pinspiratia'] - df['max_plateau']
  df['pk_maxdiff']   = df['Pinspiratia'] - df['max_peak']
  df['v_maxdiff']    = df['Tidal Volume'] - df['max_volume']*10.0
  return df

def show_distances(df,display_volume):
  if display_volume:
    fig1,ax1=plt.subplots(4,1,sharex=True)
  else:
    fig1,ax1=plt.subplots(3,1,sharex=True)
  fig1.set_size_inches(13,10)
  ax1=ax1.flatten()

  testname=df['test_name'][0]
  testseries=testname[0:2]+'xx'

  df.plot(kind='scatter', ax=ax1[0], x='test_name', y='peep_diff', yerr='rms_peep', c='r', grid=True)
  ax1[0].set_title(f'Set PEEP - Mean PEEP for Series {testseries}')
  ax1[0].set_ylabel('Pressure [cmH2O]')
  df.plot(kind='scatter', ax=ax1[1], x='test_name', y='p_diff', yerr='rms_plateau', c='g', grid=True)
  ax1[1].set_title('Target Pressure - Mean Plateau* Pressure')
  ax1[1].set_ylabel('Pressure [cmH2O]')
  df.plot(kind='scatter', ax=ax1[2], x='test_name', y='pk_diff', yerr='rms_peak', c='k', grid=True)
  ax1[2].set_title('Target Pressure - Mean Peak Pressure')
  ax1[2].set_ylabel('Pressure [cmH2O]')
  if display_volume:
    df.plot(kind='scatter', ax=ax1[3], x='test_name', y='v_diff', yerr='rms_volume', c='b', grid=True)
    ax1[3].set_title('Intended Tidal Volume - Mean Tidal Volume')
    ax1[3].set_ylabel('vol [ml]')
    ax1[3].set_xlabel('TEST ID')
  else:
    ax1[2].set_xlabel('TEST ID')

  fig1.tight_layout()
  testname=df['test_name'][0]
  fig1.savefig(f'diff_Series{testseries}.png')


def show_range(df):
  fig2,ax2=plt.subplots(3,1,sharex=True)
  fig2.set_size_inches(13,10)
  ax2=ax2.flatten()

  testname=df['test_name'][0]
  testseries=testname[0:2]+'xx'

  df.plot(kind='scatter', ax=ax2[0], x='test_name', y='simulator_plateau', label='SIM Pressure', c='r', grid=True)
  df.plot(kind='scatter', ax=ax2[0], x='test_name', y='min_plateau', label='MVM Range', c='b', grid=True)
  df.plot(kind='scatter', ax=ax2[0], x='test_name', y='max_plateau', c='b', grid=True)
  ax2[0].set_title(f'Plateau* Pressure Range for Series {testseries}')
  ax2[0].set_ylabel('Pressure [cmH2O]')

  df.plot(kind='scatter', ax=ax2[1], x='test_name', y='simulator_plateau', label='SIM Pressure', c='r', grid=True)
  df.plot(kind='scatter', ax=ax2[1], x='test_name', y='min_peak', label='MVM Range', c='b', grid=True)
  df.plot(kind='scatter', ax=ax2[1], x='test_name', y='max_peak', c='b', grid=True)
  ax2[1].set_title('Peak Pressure Range')
  ax2[1].set_ylabel('Pressure [cmH2O]')

  df.plot(kind='scatter', ax=ax2[2], x='test_name', y='simulator_volume', label='SIM Volume', c='r', grid=True)
  df.plot(kind='scatter', ax=ax2[2], x='test_name', y='min_volume', label='MVM Range', c='b', grid=True)
  df.plot(kind='scatter', ax=ax2[2], x='test_name', y='max_volume', c='b', grid=True)
  ax2[2].set_title('Tidal Volume Range')
  ax2[2].set_ylabel('Volume [ml]')
  ax2[2].set_xlabel('TEST ID')

  fig2.tight_layout()
  fig2.savefig(f'range_Series{testseries}.png')


def show_maxdeviation(df,display_volume):
  if display_volume:
    fig3,ax3=plt.subplots(4,1,sharex=True)
  else:
    fig3,ax3=plt.subplots(3,1,sharex=True)

  fig3.set_size_inches(13,10)
  ax3=ax3.flatten()

  testname=df['test_name'][0]
  testseries=testname[0:2]+'xx'

  df.plot(kind='scatter', ax=ax3[0], x='test_name', y='peep_maxdiff', c='r', grid=True)
  ax3[0].set_title(f'Set PEEP - Max PEEP for Series {testseries}')
  ax3[0].set_ylabel('Pressure [cmH2O]')
  df.plot(kind='scatter', ax=ax3[1], x='test_name', y='p_maxdiff', c='g', grid=True)
  ax3[1].set_title('Target Pressure - Max Plateau* Pressure')
  ax3[1].set_ylabel('Pressure [cmH2O]')
  df.plot(kind='scatter', ax=ax3[2], x='test_name', y='pk_maxdiff', c='k', grid=True)
  ax3[2].set_title('Target Pressure - Max Peak Pressure')
  ax3[2].set_ylabel('Pressure [cmH2O]')
  if display_volume:
    df.plot(kind='scatter', ax=ax3[3], x='test_name', y='v_maxdiff', c='b', grid=True)
    ax3[3].set_title('Intended Tidal Volume - Max Tidal Volume')
    ax3[3].set_ylabel('vol [ml]')
    ax3[3].set_xlabel('TEST ID')
  else:
    ax3[2].set_xlabel('TEST ID')

  fig3.tight_layout()
  testname=df['test_name'][0]
  fig3.savefig(f'maxdiff_Series{testseries}.png')


if __name__=='__main__':

  parser = argparse.ArgumentParser(description='Test Quality by AC')
  parser.add_argument("input", help="summary file(s) ", nargs='+')
  parser.add_argument("-v", "--no_volume", action='store_true', help="indicate that the intented tidal volume is not specified")

  args=parser.parse_args()

  dataframe=process_files(args.input)

  calculate_distances(dataframe)
  show_distances(dataframe,not args.no_volume)
  show_range(dataframe)
  show_maxdeviation(dataframe,not args.no_volume)

  plt.show()
  
