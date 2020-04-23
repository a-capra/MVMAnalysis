import numpy as np
import db
import combine as cb
import mvmio as io
import matplotlib.pyplot as plt


# Adapted from plot_arxXiv_canvases
def plot_3views(data, run_config, output_directory):
  fig31, ax31 = plt.subplots(3, 1)
  ax31 = ax31.flatten()
  run_config[0]["linestyle"] = "--"
  run_config[1]["linestyle"] = ":"

  for idx, rc, d in zip(range(1, (len(run_config) + 1)), run_config, data):
    my_selected_cycle = rc["meta"]["cycle_index"]

    # Simulator subset
    d["sim_sel"] = d["sim"][(d["sim"]["start"] >= d["start_times"][my_selected_cycle]) & (d["sim"]["start"] < (d["start_times"][my_selected_cycle + 6]))].copy()

    # Ventilator subset
    first_time_bin = d["sim_sel"]["dt"].iloc[0]
    last_time_bin = d["sim_sel"]["dt"].iloc[-1]
    d["mvm_sel"] = d["mvm"][(d["mvm"]["dt"] > first_time_bin) & (d["mvm"]["dt"] < last_time_bin)]

    d["sim_sel"].loc[:, "total_vol"] = d["sim_sel"]["total_vol"] - d["sim_sel"]["total_vol"].min()

    d["sim_sel"].plot(ax=ax31[0], x="dt", y="total_flow", label=f"SIM {idx}", c="r", linestyle=rc["linestyle"])
    d["sim_sel"].plot(ax=ax31[1], x="dt", y="airway_pressure", label=f"SIM {idx}", c="r", linestyle=rc["linestyle"])
    d["sim_sel"].plot(ax=ax31[2], x="dt", y="total_vol", label=f"SIM {idx}", c="r", linestyle=rc["linestyle"])
    d["mvm_sel"].plot(ax=ax31[0], x="dt", y="display_flux", label=f"MVM {idx}", c="b", linestyle=rc["linestyle"])
    d["mvm_sel"].plot(ax=ax31[1], x="dt", y="airway_pressure", label=f"MVM {idx}", c="b", linestyle=rc["linestyle"])
    d["mvm_sel"].plot(ax=ax31[2], x="dt", y="tidal_volume", label=f"MVM {idx}", c="b", linestyle=rc["linestyle"])

  ax31[0].set_xlabel("")
  ax31[0].set_ylabel("Flux [l/min]")
  ax31[1].set_xlabel("")
  ax31[1].set_ylabel("Airway pressure [cmH2O]")
  ax31[2].set_xlabel("Time [s]")
  ax31[2].set_ylabel("Tidal volume [cl]")

  title = f"{run_config[0]['dataset_name']} test {run_config[0]['meta']['test_name']} (1) vs {run_config[1]['dataset_name']} test {run_config[1]['meta']['test_name']} (2)"
  fig31.suptitle(title, weight="heavy")
  figpath = f"{output_directory}/{run_config[0]['dataset_name']}_{run_config[0]['meta']['test_name']}_vs_{run_config[1]['dataset_name']}_{run_config[1]['meta']['test_name']}.png"
  print(f"Saving figure to {figpath}...")
  fig31.savefig(figpath)


def process_run(run_config, output_directory):
  colors = {
    "muscle_pressure": "#009933"  , #green
    "sim_airway_pressure": "#cc3300" ,# red
    "total_flow":"#ffb84d" , #
    "tidal_volume":"#ddccff" , #purple
    "total_vol":"pink" , #
    "reaction_time" : "#999999", #
    "pressure" : "black" , #  blue
    "vent_airway_pressure": "#003399" ,# blue
    "flux" : "#3399ff" #light blue
  }

  data = [{}, {}]
  for rc, d in zip(run_config, data):
    # Simulator data
    d["sim"] = io.get_simulator_df(rc["fullpath_rwa"], rc["fullpath_dta"], columns_rwa, columns_dta)

    # MVM data
    d["mvm"] = io.get_mvm_df_json(rc["fullpath_mvm"]) if rc["json"] else io.get_mvm_df(fname=rc["fullpath_mvm"], sep=rc["mem_sep"], configuration=rc["mvm_col"])

    cb.add_timestamp(d["mvm"])

    cb.correct_sim_df(d["sim"])
    # Add time shift
    cb.apply_manual_shift(sim=d["sim"], mvm=d["mvm"], manual_offset=rc["offset"])

    cb.add_pv2_status(d["mvm"])

    # Compute cycle start based on PV2
    d["start_times"] = cb.get_start_times(d["mvm"])

    d["reaction_times"] = cb.get_reaction_times(d["sim"], d["start_times"])

    this_shape = d["sim"].shape
    d["sim"]["iindex"] = np.arange(this_shape[0])
    d["sim"]["siindex"] = np.zeros(this_shape[0])
    d["sim"]["diindex"] = np.zeros(this_shape[0])

    cb.add_cycle_info(sim=d["sim"], mvm=d["mvm"], start_times=d["start_times"], reaction_times=d["reaction_times"])
    d["sim"]["dtc"] = d["sim"]["dt"] - d["sim"]["start"]
    d["sim"]["diindex"] = d["sim"]["diindex"] - d["sim"]["siindex"]

    cb.add_chunk_info(d["sim"])

    # Compute tidal volume etc.
    cb.add_clinical_values(d["sim"])
    d["respiration_rate"], d["inspiration_duration"] = cb.measure_clinical_values(d["mvm"], start_times=d["start_times"])

    cb.add_run_info(d["sim"])

    # Choose the flux to plot
    d["mvm"]["display_flux"] = d["mvm"]["flux"]

  plot_3views(data, run_config, output_directory)


if __name__ == "__main__":
  import argparse
  import style
  import sys

  parser = argparse.ArgumentParser(description="Compare two datasets.")
  parser.add_argument("db_range_name_1", help="Name and range of the metadata spreadsheet for the first dataset")
  parser.add_argument("data_location_1", help="Path to the first dataset.")
  parser.add_argument("db_range_name_2", help="Name and range of the metadata spreadsheet for the second dataset")
  parser.add_argument("data_location_2", help="Path to the second dataset.")
  parser.add_argument("-d", "--output-directory", type=str, help="Plot output directory.", default="plots_iso")
  parser.add_argument("-t", "--test-names", type=str, help="Only process listed test pair.", nargs="2", default="")
  parser.add_argument("--campaign-1", type=str, help="Process only a single campaign of first dataset.", default="")
  parser.add_argument("--campaign-2", type=str, help="Process only a single campaign of second dataset.", default="")
  parser.add_argument("--j1", action="store_true", help="Try to read first dataste as JSON instead of CSV.")
  parser.add_argument("--j2", action="store_true", help="Try to read second dataste as JSON instead of CSV.")
  parser.add_argument("--offset-1", type=float, help="Time offset between first vent and sim datasets.", default=0.)
  parser.add_argument("--offset-2", type=float, help="Time offset between second vent and sim datasets.", default=0.)
  parser.add_argument("--db-google-id-1", type=str, help="First datset metadata spreadsheet ID.", default="1aQjGTREc9e7ScwrTQEqHD2gmRy9LhDiVatWznZJdlqM")
  parser.add_argument("--db-google-id-2", type=str, help="Second datset metadata spreadsheet ID.", default="1aQjGTREc9e7ScwrTQEqHD2gmRy9LhDiVatWznZJdlqM")
  parser.add_argument("--mvm-sep-1", type=str, help="Separator between datetime and the rest in the first MVM file", default=" -> ")
  parser.add_argument("--mvm-sep-2", type=str, help="Separator between datetime and the rest in the second MVM file", default=" -> ")
  parser.add_argument("--mvm-col-1", type=str, help="Columns configuration for first MVM acquisition, see mvmio.py", default="default")
  parser.add_argument("--mvm-col-2", type=str, help="Columns configuration for second MVM acquisition, see mvmio.py", default="default")
  args = parser.parse_args()

  run_config = [
      {
        "db_range_name" : args.db_range_name_1,
        "data_location" : args.data_location_1,
        "single_campaign" : args.campaign_1,
        "json" : args.j1,
        "offset" : args.offset_1,
        "db_google_id" : args.db_google_id_1,
        "mvm_sep" : args.mvm_sep_1,
        "mcm_col" : args.mvm_col_1
      },
      {
        "db_range_name" : args.db_range_name_2,
        "data_location" : args.data_location_2,
        "single_campaign" : args.campaign_2,
        "json" : args.j2,
        "offset" : args.offset_2,
        "db_google_id" : args.db_google_id_2,
        "mvm_sep" : args.mvm_sep_2,
        "mcm_col" : args.mvm_col_2
      }
  ]

  columns_rwa = [
    'dt',
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
  columns_dta = [
    #'dt',
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

  print ("First dataset location: ", run_config[0]["data_location"])
  print ("Second dataset location: ", run_config[1]["data_location"])

  # read metadata spreadsheet
  df_spreadsheet = [db.read_online_spreadsheet(rc["db_google_id"], rc["db_range_name"]) for rc in run_config]

  if args.test_names:
    test_names = [args.test_names]
    for tn, rc, ss in zip (test_names[0], run_config, df_spreadsheet):
      if not ss["N"].isin(tn).any():
        print(f"ERROR: Failed to find {tn[0]} in {rc['db_range_name']}!")
        sys.exit(1)
  else:
    # Check for tests only present in one of the spreadsheets
    df_spreadsheet_1_only = df_spreadsheet[0][~df_spreadsheet[0]["N"].isin(df_spreadsheet[1]["N"])]
    if not df_spreadsheet_1_only.empty:
      print("WARNING: The following tests are only present in the first dataset. Skipping...")
      print(df_spreadsheet_1_only)
    df_spreadsheet_2_only = df_spreadsheet[1][~df_spreadsheet[1]["N"].isin(df_spreadsheet[0]["N"])]
    if not df_spreadsheet_2_only.empty:
      print("WARNING: The following tests are only present in the second dataset. Skipping...")
      print(df_spreadsheet_2_only)
    # Just duplicate the test names
    test_names = [[tn, tn] for tn in df_spreadsheet[0][df_spreadsheet[0]["N"].isin(df_spreadsheet[1]["N"])]["N"]]

  for test_pair in test_names:
    success = True
    for tn, rc, ss in zip(test_pair, run_config, df_spreadsheet):
      if rc["single_campaign"]:
        print(f"Only processing tests from {rc['db_range_name']} {rc['single_campaign']}...")
        cur_test = ss[(ss["N"] == tn) & (ss["Campaign"] == rc["single_campaign"])]
      else:
        cur_test = ss[ss["N"] == tn]
      if len(cur_test) > 1:
        print(f"WARNING: More than one test {test_name} found in rc['db_range_name']. Using first one...")

      # Read meta data from spreadsheets
      filename = cur_test.iloc[0]["MVM_filename"]
      if not filename:
        success = False
        break
      rc["meta"] = db.read_meta_from_spreadsheet(cur_test, filename)

      # Only use first element
      rc["objname"] = f"{filename}_0"
      rc["meta"] = rc["meta"][rc["objname"]]

      # Build MVM paths and skip user requested files
      rc["fullpath_mvm"] = f"{rc['data_location']}/{rc['meta']['Campaign']}/{rc['meta']['MVM_filename']}"
      if not (rc["json"] or rc["fullpath_mvm"].endswith(".txt")):
        print("Adding missing txt extension to MVM path.")
        rc["fullpath_mvm"] += ".txt"
      print(f"\nMVM file {i+1}: {rc['fullpath_mvm']}")

      # Build simulations paths
      rc["fullpath_rwa"] = f"{rc['data_location']}/{rc['meta']['Campaign']}/{rc['meta']['SimulatorFileName']}"
      # Fix extensions
      if rc["fullpath_rwa"].endswith(".dta"):
        rc["fullpath_rwa"] = rc["fullpath_rwa"][:-4]
      if not rc["fullpath_rwa"].endswith(".rwa"):
        rc["fullpath_rwa"] += ".rwa"
      rc["fullpath_dta"] = rc["fullpath_rwa"].replace("rwa", "dta")
      print(f"Files of simulation {i+1}: {rc['fullpath_rwa']}, {rc['fullpath_dta']}")

      rc["dataset_name"] = f"{rc['db_range_name'].split('!')[0]}_{rc['meta']['Campaign']}"

    if success:
      process_run(run_config, args.output_directory)
