import numpy as np
import pandas as pd
import logging as log
import json

columns_rwa = ['dt',
  'airway_pressure',
  'muscle_pressure',
  'tracheal_pressure',
  'chamber1_vol',
  'chamber2_vol',
  'total_vol',
  'chamber1_pressure',
  'chamber2_pressure',
  'breath_fileno',
  'aux1',
  'aux2',
  'oxygen'
]

columns_dta = [#'dt',
  'breath_no',
  'compressed_vol',
  'airway_pressure',
  'muscle_pressure',
  'total_vol',
  'total_flow',
  'chamber1_pressure',
  'chamber2_pressure',
  'chamber1_vol',
  'chamber2_vol',
  'chamber1_flow',
  'chamber2_flow',
  'tracheal_pressure',
  'ventilator_vol',
  'ventilator_flow',
  'ventilator_pressure',
]

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
    'flux',
    'pressure_pv1',
    'airway_pressure',
    'in',
    'service_1',
    'out',
    'flux_2',
    'flux_3',
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

map_arduino2analysis={
    'mvm_triumf_1':{
                    'date':'ts',
                    'airway_pressure':'ps1_p',
                    'flux_2':'fs1_r',
                    'in':'inlet',
                    'out':'outlet',
                    'flux':'fs1_r',
                    'flux_3':'fs1_r'
                    },
    'mvm_control':{
                  'date':'time',
                  'time_arduino':'ts',
                  'flux':'flux_inhale',
                  'pressure_pv1':'p_valve',
                  'airway_pressure':'p_patient',
                  'in':'pv1_ctrl',
                  'service_1':'p_slow',
                  'out':'pv2_ctrl',
                  'flux_2':'f_vent_raw',
                  'flux_3':'f_total',
                  'volume':'v_total',
                  'service_2':'p_patient_dv2'
                  }
      }

def get_raw_df(fname, columns, columns_to_deriv, timecol='dt'):
  df = pd.read_csv(fname, skiprows=4, names=columns, sep='\t', engine='python')
  for column in columns_to_deriv:
    df[f'deriv_{column}'] = df[column].diff() / df[timecol].diff() * 60. # TODO
  return df

def get_simulator_df(fullpath_rwa, fullpath_dta, df_columns_rwa=columns_rwa, df_columns_dta=columns_dta):
  ''' Load the simulator data frame from the specified paths '''
  df_rwa = get_raw_df(fullpath_rwa, columns=df_columns_rwa, columns_to_deriv=['total_vol'])
  df_dta = get_raw_df(fullpath_dta, columns=df_columns_dta, columns_to_deriv=[])

  ''' There have been examples when the dta file has a different number of lines than the rwa file.
  This has led to bad results: i.e. results that superficially look like the analysis is working but are wrong.
  Impose check on the length of the data frames.
  Chris Jillings 2020-05-17 '''
  length_rwa = df_rwa.shape[0]
  length_dta = df_dta.shape[0]
  if length_rwa != length_dta :
    log.error(f"""
      Error: rwa and dta files have a different number of lines, respectively {length_rwa} vs {length_dta}
      File at path '{fullpath_rwa}' is not compatible with '{fullpath_dta}'
      Returning empty simulator data frame""")
    df = pd.DataFrame()
  else :
    df0 = df_dta.join(df_rwa['dt'])
    df_rwa['oxygen'] = df_rwa['oxygen'] / df_rwa['airway_pressure']
    df  = df0.join(df_rwa['oxygen'] )
    df['dt'] = np.linspace( df.iloc[0,:]['dt'] ,  df.iloc[-1,:]['dt'] , len(df) ) # correct for duplicate times

  return df


def get_mvm_df(fname, sep=' -> ', configuration='default'):
  #data from the ventilator
  data = []

  is_unix = False
  columns = mapping[configuration]
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
        par = sep.join(l[1:]).split(',')
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
        is_unix = True

      if ( "mvm_col_no_time" in configuration  ) :
        dataline = dict ( zip ( columns[0:10], [float(i) for i in par[0:10]]  )   )
        step = 0.012 #s
        dataline['date']  = iline * step
        data.append(  dataline )
        is_unix = True
      elif (configuration == "mvm_col_arduino") :
        dataline = dict ( zip ( columns[2:11], [float(i) for i in par[1:10]]  )   )
        dataline['date'] = float ( par[0] )  * 1e-3
        is_unix = True
        #dataline['date'] = t
        data.append(  dataline )
      else :  #default
        dataline = dict (   zip ( columns[1:8], [float(i) for i in par[0:7]]  )   )
        dataline['date'] = t
        data.append( dataline )

  is_manual = False
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


def get_mvm_df_json(fname, map='mvm_control') :

  mydict = json.loads(open(fname).read())
  column_names = []

  if isinstance( mydict['data'][0] , dict )   == False :
    column_names = mydict['format']
    data = []
    for mylist in mydict['data'] :
      data.append (dict ( zip (column_names , mylist ) ))
    print ("mvmio.py: Reading compact dictionary format")
    df = pd.DataFrame.from_dict(data)
  else :
    print ("mvmio.py: Reading full dictionary format")
    df = pd.DataFrame.from_dict(mydict['data'])
    column_names = [x for x in mydict['data'][0].keys()]

  #print(column_names)
  #temporary: convert arduino variable names into the analysis names
  for key,val in map_arduino2analysis[map].items():
    try:
      df[key]=df[val]
      #print(f'{key}:{val}')
    except KeyError as ky:
      print(ky,'is not recognized')
      pass

  try:
    df.replace({"CLOSED":0.0, "OPEN":1.0},inplace=True)
  except TypeError:
    pass
  df['dt'] = ( pd.to_datetime(df['date'], unit='s') - pd.to_datetime(df['date'][0], unit='s') )/np.timedelta64(1,'s')
 
  return df
