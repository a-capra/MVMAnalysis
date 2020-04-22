from db import *

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Compare two datasets.")
  parser.add_argument("db-range-name-1", help="Name and range of the metadata spreadsheet for the first dataset")
  parser.add_argument("data-location-1", help="Path to the first dataset.")
  parser.add_argument("db-range-name-2", help="Name and range of the metadata spreadsheet for the second dataset")
  parser.add_argument("data-location_2", help="Path to the second dataset.")
  parser.add_argument("-d", "--output-directory", type=str, help="Plot output directory.", default="plots_iso")
  parser.add_argument("-i", "--ignore_sim", action="store_true", help="Ignore simulation.")
  parser.add_argument("-S", "--skip_files", type=str, help="Skip listed files in both datasets.", nargs="+", default="")
  parser.add_argument("--campaign-1", type=str, help="Process only a single campaign of first dataset.", default="")
  parser.add_argument("--campaign-2", type=str, help="Process only a single campaign of second dataset.", default="")
  parser.add_argument("--j1", action="store_true", help="Try to read first dataste as JSON instead of CSV.")
  parser.add_argument("--j2", action="store_true", help="Try to read second dataste as JSON instead of CSV.")
  parser.add_argument("--offset-1", type=float, help="Time offset between first vent and sim datasets.")
  parser.add_argument("--offset-2", type=float, help="Time offset between second vent and sim datasets.")
  parser.add_argument("--db-google-id-1", type=str, help="First datset metadata spreadsheet ID.", default="1aQjGTREc9e7ScwrTQEqHD2gmRy9LhDiVatWznZJdlqM")
  parser.add_argument("--db-google-id-2", type=str, help="Second datset metadata spreadsheet ID.", default="1aQjGTREc9e7ScwrTQEqHD2gmRy9LhDiVatWznZJdlqM")

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

  print ("First dataset location: ", args.data_location_1)
  print ("Second dataset location: ", args.data_location_1)

  # read metadata spreadsheet
  df_spreadsheet_1 = read_online_spreadsheet(spreadsheet_id=args.dg_google_id_1, range_name=db_range_name_1)
  df_spreadsheet_2 = read_online_spreadsheet(spreadsheet_id=args.dg_google_id_2, range_name=db_range_name_2)

  # Check for tests only present in one of the spreadsheets
  df_spreadsheet_1_only = df_spreadsheet_1[~df_spreadsheet_1["test_name"].isin(df_spreadsheet_2["test_name"])]
  if not df_spreadsheet_1_only.empty:
    print("WARNING: The following tests are only present in the first dataset. Skipping...")
    print(df_spreadsheet_1_only)
  df_spreadsheet_2_only = df_spreadsheet_2[~df_spreadsheet_2["test_name"].isin(df_spreadsheet_1["test_name"])]
  if not df_spreadsheet_2_only.empty:
    print("WARNING: The following tests are only present in the second dataset. Skipping...")
    print(df_spreadsheet_2_only)

  test_names = df_spreadsheet_1["test_name"].unique()
  for test_name in test_names:
    # Warn about duplicate tests
    cur_tests_1 = df_spreadsheet_1[df_spreadsheet_1["test_name"] == test_name]
    if len(cur_tests_1) > 1:
      print(f"WARNING: More than one test {test_name} found in first dataset. Using first one...")
    cur_tests_2 = df_spreadsheet_2[df_spreadsheet_2["test_name"] == test_name]
    # Skip tests not present in second dataset. We've already warned about this above.
    if cur_tests_2.empty:
      continue
    elif len(cur_tests_2) > 1:
      print(f"WARNING: More than one test {test_name} found in second dataset. Using first one...")

    # Read meta data from spreadsheets
    filename_1 = cur_test_1.iloc[0]["MVM_filename"]
    filename_2 = cur_test_2.iloc[2]["MVM_filename"]
    meta_1 = read_meta_from_spreadsheet(df_spreadsheet_1, filename_1)
    meta_2 = read_meta_from_spreadsheet(df_spreadsheet_2, filename_2)

    # Only use first element
    objname_1 = f"{filename_1}_0"
    objname_2 = f"{filename_2}_0"

    # Only process selected campaigns
    if args.campaign_1 and (meta_1[objname_1]["Campaign"] != args.campaign_1):
      print(f"Test {test_name} not in selected campaign {args.campaign_1}. Skipping...")
      continue
    if args.campaign_2 and (meta_2[objname_2]["Campaign"] != args.campaign_2):
      print(f"Test {test_name} not in selected campaign {args.campaign_2}. Skipping...")
      continue

    # Build MVM paths and skip user requested files
    fullpath_mvm_1 = f"{args.data_location_1}/{meta_1[objname_1]['Campaign']}/{meta_1[objname_1]['MVM_filename']}"
    if not fullpath_mvm_1.endswith(".txt"):
      fullpath_mvm_1 += ".txt"
    print(f"\nFirst MVM file: {fullpath_mvm_1}")
    if fullpath_mvm_1.split("/")[-1] in args.skip_files:
      print("\tskipping per user request...")
      continue
    fullpath_mvm_2 = f"{args.data_location_2}/{meta_2[objname_2]['Campaign']}/{meta_2[objname_2]['MVM_filename']}"
    if not fullpath_mvm_2.endswith(".txt"):
      fullpath_mvm_2 += ".txt"
    print(f"\nSecond MVM file: {fullpath_mvm_2}")
    if fullpath_mvm_2.split("/")[-1] in args.skip_files:
      print("\tskipping per user request...")
      continue

    # Build simulations paths
    fullpath_rwa_1 = f"{args.data_location_1}/{meta_1[objname_1]['Campaign']}/{meta[objname_1]['SimulatorFileName']}"
    # Fix extensions
    if fullpath_rwa_1.endswith(".dta"):
      fullpath_rwa_1 = fullpath_rwa_1[:-4]
    if not fullpath_rwa_1.endswith(".rwa"):
      fullpath_rwa_1 += ".rwa"
    fullpath_dta_1 = fullpath_dta.replace("rwa", "dta")
    print(f"First simulation files: {fullpath_rwa_1}, {fullpath_dta_1}")
    fullpath_rwa_2 = f"{args.data_location_2}/{meta_2[objname_2]['Campaign']}/{meta[objname_2]['SimulatorFileName']}"
    # Fix extensions
    if fullpath_rwa_2.endswith(".dta"):
      fullpath_rwa_2 = fullpath_rwa_2[:-4]
    if not fullpath_rwa_2.endswith(".rwa"):
      fullpath_rwa_2 += ".rwa"
    fullpath_dta_2 = fullpath_dta.replace("rwa", "dta")
    print(f"First simulation files: {fullpath_rwa_2}, {fullpath_dta_2}")
