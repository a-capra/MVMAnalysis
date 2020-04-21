from __future__ import print_function
import csv
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os
import pandas as pd

""" Converts Google sheet data to a Pandas DataFrame.
Note: This script assumes that your data contains a header file on the first row!
Also note that the Google API returns 'none' from empty cells - in order for the code
below to work, you'll need to make sure your sheet doesn't contain empty cells,
or update the code to account for such instances.
"""
def gsheet2df(gsheet):
  header = gsheet[1]   # Assumes first line is header!
  values = gsheet[2:]  # Everything else is data.

  if not values:
    print('No data found.')
  else:
    all_data = []
    for col_id, col_name in enumerate(header):
      column_data = []
      for irow, row in enumerate(values):
        column_data.append(row[col_id])
      ds = pd.Series(data=column_data, name=col_name)
      all_data.append(ds)
    df = pd.concat(all_data, axis=1)
  return df

def read_meta_csv(fname):
  ''' get a dict containing run metadata '''
  meta = {}
  with open(fname) as f:
    reader = csv.DictReader(f)

""" Read the online spreadsheet
From in https://developers.google.com/sheets/api/quickstart/python
"""
def read_online_spreadsheet (spreadsheet_id, range_name) :
  # If modifying these scopes, delete the file token.pickle.
  SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

  # The ID and range of a sample spreadsheet.
  SAMPLE_SPREADSHEET_ID = spreadsheet_id
  SAMPLE_RANGE_NAME = range_name
  creds = None
  # The file token.pickle stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists('token.pickle') :
    with open('token.pickle', 'rb') as token:
      creds = pickle.load(token)

  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
      'mvm_credentials.json', SCOPES)
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.pickle', 'wb') as token:
      pickle.dump(creds, token)

  service = build('sheets', 'v4', credentials=creds)

  # Call the Sheets API
  sheet = service.spreadsheets()
  result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,range=SAMPLE_RANGE_NAME).execute()
  values = gsheet2df ( result.get( 'values' , []) )

  return values

"""
End of the spreadsheet reader
"""

def read_meta_from_spreadsheet (df, filename) :
  df = df[(df["MVM_filename"]==filename)]

  meta = {}
  for idx in range(len(df))  :
    compliance = df["C"].iloc[idx]
    resistance = df["R"].iloc[idx]
    key  = f'{filename}_{idx}'
    meta[ key ] = {
      'Compliance': float ( compliance ) ,
      'Resistance': float ( resistance )  ,
      'Rate respiratio':  float ( df["rate"].iloc[idx] )   ,
      'I:E': df["ratio"].iloc[idx],
      'Peep':   float (df["PEEP"].iloc[idx] ) ,
      'Run' : df['run'].iloc[idx] ,
      'Pinspiratia': float ( df["plateau"].iloc[idx] ) ,
      'SimulatorFileName': df["simulator_filename"].iloc[idx] ,
      'Campaign': df["campaign"].iloc[idx] ,
      'MVM_filename' : df["MVM_filename"].iloc[idx],
      'test_name' : df["N"].iloc[idx],
      'Tidal Volume' : df["TV"].iloc[idx],
      'leakage' : df["leakage"].iloc[idx],
      'cycle_index' : int ( df["cycle_index"].iloc[idx]) ,
    }
  return meta

def read_mhra_csv(fname):
  dfmhra = pd.read_csv(fname, sep=',')
  dfmhra['plot'] = 0
  return dfmhra
