import numpy as np
import pandas as pd
import json

mapping = {
  'default': [
    'date',
    'flux',
    'pressure_pv1',
    'airway_pressure',
    'in',
    'service_1',
    'out',
    'service_2'
  ],
  'mvm_col_arduino' : [
    'date',
    'time_arduino',
    'flux_2',
    'pressure_pv1',
    'airway_pressure',
    'in',
    'service_1',
    'out',
    'flux_3',
    'flux',
    'volume',
    'service_2'
  ],

  'mvm_col_no_time' : [
    'pressure_pv1' ,
    'airway_pressure',
    'in',
    'service_1',
    'out',
    'flux_2',
    'flux',
    'volume',
    'service_2'
  ],
  'mvm_col_no_time_2': [
    'flux_3',
    'pressure_pv1' ,
    'airway_pressure',
    'in',
    'service_1',
    'out',
    'flux_2',
    'flux',
    'service_2',
    'derivative'
  ],

  'mvm_triumf_1':[
    'date',#'ts',
    'mode',
    'phase',
    'airway_pressure',#'ps1_p',
    'ps1_t',
    'flux_2',#'fs1_r',
    'flux',
    'in',#'inlet',
    'out'#'outlet'
  ]
}

map_arduino_bs={
  'mvm_triumf_1':{
    'ts':'date',
    'mode':'mode',
    'phase':'phase',
    'ps1_p':'airway_pressure',
    'ps1_t':'ps1_t',
    'fs1_r':'flux_2',
    'inlet':'in',
    'outlet':'out'
  }
}


def get_raw_df(fname, columns, columns_to_deriv, timecol='dt'):
  df = pd.read_csv(fname, skiprows=4, names=columns, sep='\t', engine='python')
  for column in columns_to_deriv:
    df[f'deriv_{column}'] = df[column].diff() / df[timecol].diff() * 60. # TODO
  return df

def get_simulator_df(fullpath_rwa, fullpath_dta, columns_rwa, columns_dta):
  df_rwa = get_raw_df(fullpath_rwa, columns=columns_rwa, columns_to_deriv=['total_vol'])
  df_dta = get_raw_df(fullpath_dta, columns=columns_dta, columns_to_deriv=[])
  df0 = df_dta.join(df_rwa['dt'])
  df_rwa['oxygen'] = df_rwa['oxygen']  /  df_rwa['airway_pressure']
  df  = df0.join(df_rwa['oxygen'] )
  df['dt'] = np.linspace( df.iloc[0,:]['dt'] ,  df.iloc[-1,:]['dt'] , len(df) ) # correct for duplicate times
  return df

def get_csv(fname, sep, conf):
  #data from the ventilator
  data = []

  columns = mapping[conf]
  print ("This is the chosen column mapping for the MVM file: ", columns)

  with open(fname) as f:
    lines = f.readlines()
    for iline, line in enumerate(lines):
      if iline == 0: continue # skip first line

      line = line.strip()
      line = line.strip('\n')
      if not line: continue
      l = line.split(sep)
      try:
        par = ''.join(l[1:]).split(',')
      except :
        print (line)
        continue
      # remove unformatted lines
      try:
        for x in par: float(x)
      except ValueError:
        continue

      t = l[0]
      if ':' not in l[0]:
        t = float(l[0]) # in this way, t is either a string (if HHMMSS) or a float

      if "mvm_col_no_time" in conf:
        dataline = dict ( zip ( columns[0:10], [float(i) for i in par[0:10]]  )   )
        step = 0.012 #s
        dataline['date']  = iline * step
        data.append(  dataline )
      elif conf == "mvm_col_arduino":
        dataline = dict ( zip ( columns[2:11], [float(i) for i in par[1:10]]  )   )
        dataline['date'] = float ( par[0] )  * 1e-3
        #dataline['date'] = t
        data.append(  dataline )
      else:  #default
        dataline = dict (   zip ( columns[1:8], [float(i) for i in par[0:7]]  )   )
        dataline['date'] = t
        data.append( dataline )

  return data

def get_json(fname, conf):
  #data from the ventilator
  data = []
  with open(fname) as f:
    log_data = json.load(f)["data"]

  offset = log_data[0]["ts"]

  for rec in log_data:
    for key, val in map_arduino_bs[conf].items():
      rec[val]=rec.pop(key)
    rec['date']-=offset
    rec['flux']=rec['flux_2']
    if rec['out'] == 'OPEN':
      rec['out'] = 1
    else:
      rec['out'] = 0
    data.append(rec)

  return data

def get_mvm_df(fname, sep=' -> ', configuration='default'):
  #data from the ventilator
  print ("This is the chosen column mapping for the MVM file: ", mapping[configuration])

  is_unix = False
  if configuration=="mvm_col_arduino":
    is_unix = True

  print(f'MVM datafile: {fname}')
  if fname[-5:]=='.json':
    is_unix = True
    data = get_json(fname,configuration)
  else:
    data = get_csv(fname,sep,configuration)

  #is_manual = False
  df = pd.DataFrame(data)
  if not is_unix: # text timestamp
    df['dt'] = ( pd.to_datetime(df['date']) - pd.to_datetime(df['date'][0]) )/np.timedelta64(1,'s')
  else: # unix timestamp in seconds
    #print (df['date'])
    df['dt'] = ( pd.to_datetime(df['date'], unit='s') - pd.to_datetime(df['date'][0], unit='s') )/np.timedelta64(1,'s')

  #print (df.head())
  #dtmax = df.iloc[-1,:]['dt']
  #timestamp = np.linspace( df.iloc[0,:]['dt'] ,  df.iloc[-1,:]['dt']*(dtmax-0.08)/dtmax , len(df) )   #use this line if you want to stretch the x axis of MVM data

  return df
