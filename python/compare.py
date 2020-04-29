import numpy as np
import db
import combine as cb
import mvmio as io
import matplotlib.pyplot as plt
import style
import combine_plot_utils as cbpu


# Adapted from plot_arxXiv_canvases
def plot_3view_comparison(data, run_config, output_directory):
  # We handle duplicates in the main function, and only use the first element here
  meta = [rc["meta"][rc["objname"]] for rc in run_config]

  fig31, ax31 = plt.subplots(3, 1)
  ax31 = ax31.flatten()
  run_config[0]["linestyle"] = "--"
  run_config[1]["linestyle"] = ":"

  title = []
  for idx, rc, m, d in zip(range(1, (len(run_config) + 1)), run_config, meta, data):
    # Select MVM flux to plot
    d["mvm"]["display_flux"] = d["mvm"]["flux"]

    my_selected_cycle = m["cycle_index"]
    cycles_to_display = 6

    # Simulator subset
    d["sim_sel"] = d["sim"][(d["sim"]["start"] >= d["start_times"][my_selected_cycle]) & (d["sim"]["start"] < (d["start_times"][my_selected_cycle + cycles_to_display]))].copy()

    # Ventilator subset
    first_time_bin = d["sim_sel"]["dt"].iloc[0]
    last_time_bin = d["sim_sel"]["dt"].iloc[-1]
    d["mvm_sel"] = d["mvm"][(d["mvm"]["dt"] > first_time_bin) & (d["mvm"]["dt"] < last_time_bin)].copy()

    d["sim_sel"].loc[:, "total_vol"] = d["sim_sel"]["total_vol"] - d["sim_sel"]["total_vol"].min()

    # Align timestamps
    d["mvm_sel"].loc[:, "dt"] = d["mvm_sel"]["dt"] - d["mvm_sel"]["dt"].iloc[0]
    d["sim_sel"].loc[:, "dt"] = d["sim_sel"]["dt"] - d["sim_sel"]["dt"].iloc[0]

    d["sim_sel"].plot(ax=ax31[0], x="dt", y="total_flow", label=f"SIM {idx}", c="r", linestyle=rc["linestyle"])
    d["sim_sel"].plot(ax=ax31[1], x="dt", y="airway_pressure", label=f"SIM {idx}", c="r", linestyle=rc["linestyle"])
    d["sim_sel"].plot(ax=ax31[2], x="dt", y="total_vol", label=f"SIM {idx}", c="r", linestyle=rc["linestyle"])
    d["mvm_sel"].plot(ax=ax31[0], x="dt", y="display_flux", label=f"MVM {idx}", c="b", linestyle=rc["linestyle"])
    d["mvm_sel"].plot(ax=ax31[1], x="dt", y="airway_pressure", label=f"MVM {idx}", c="b", linestyle=rc["linestyle"])
    d["mvm_sel"].plot(ax=ax31[2], x="dt", y="tidal_volume", label=f"MVM {idx}", c="b", linestyle=rc["linestyle"])

    title.append(cbpu.form_title(rc["meta"], rc["objname"]))

  ax31[0].set_xlabel("")
  ax31[0].set_ylabel("F [l/min]")
  ax31[1].set_xlabel("")
  ax31[1].set_ylabel("AWP [cmH2O]")
  ax31[2].set_xlabel("Time [s]")
  ax31[2].set_ylabel("TV [cl]")

  fig31.suptitle(f"{title[0]} (1) vs {title[1]} (2)", weight="heavy")
  figpath = f"{output_directory}/{title[0].replace(' ', '_')}_vs_{title[1].replace(' ', '_')}.png"
  print(f"Saving figure to {figpath}...")
  fig31.savefig(figpath)


if __name__ == "__main__":
  import argparse
  import sys

  parser = argparse.ArgumentParser(description="Compare two datasets.")
  parser.add_argument("db_range_name_1", help="Name and range of the metadata spreadsheet for the first dataset")
  parser.add_argument("data_location_1", help="Path to the first dataset.")
  parser.add_argument("db_range_name_2", help="Name and range of the metadata spreadsheet for the second dataset")
  parser.add_argument("data_location_2", help="Path to the second dataset.")
  parser.add_argument("-d", "--output-directory", type=str, help="Plot output directory.", default="comparison_plots")
  parser.add_argument("-t", "--test-names", type=str, help="Only process listed test pair.", nargs=2, default="")
  parser.add_argument("--campaign-1", type=str, help="Process only a single campaign of first dataset.", default="")
  parser.add_argument("--campaign-2", type=str, help="Process only a single campaign of second dataset.", default="")
  parser.add_argument("--j1", action="store_true", help="Try to read first dataste as JSON instead of CSV.")
  parser.add_argument("--j2", action="store_true", help="Try to read second dataste as JSON instead of CSV.")
  parser.add_argument("--offset-1", type=float, help="Time offset between first vent and sim datasets.", default=0.)
  parser.add_argument("--offset-2", type=float, help="Time offset between second vent and sim datasets.", default=0.)
  parser.add_argument("--pressure-offset-1", type=float, help="Pressure offset for first MVM dataset.", default=0.)
  parser.add_argument("--pressure-offset-2", type=float, help="Pressure offset for second MVM dataset.", default=0.)
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
        "pressure_offset" : args.pressure_offset_1,
        "db_google_id" : args.db_google_id_1,
        "mvm_sep" : args.mvm_sep_1,
        "mvm_col" : args.mvm_col_1
      },
      {
        "db_range_name" : args.db_range_name_2,
        "data_location" : args.data_location_2,
        "single_campaign" : args.campaign_2,
        "json" : args.j2,
        "offset" : args.offset_2,
        "pressure_offset" : args.pressure_offset_2,
        "db_google_id" : args.db_google_id_2,
        "mvm_sep" : args.mvm_sep_2,
        "mvm_col" : args.mvm_col_2
      }
  ]

  for idx, rc in enumerate(run_config):
    print(f"Meta data {idx}: {rc['db_range_name']} Data location {idx}: {rc['data_location']}")

  # read metadata spreadsheet
  df_spreadsheet = [db.read_online_spreadsheet(rc["db_google_id"], rc["db_range_name"]) for rc in run_config]

  if args.test_names:
    test_names = [args.test_names]
    for tn, rc, ss in zip (test_names[0], run_config, df_spreadsheet):
      if not ss["N"].isin([tn]).any():
        print(f"ERROR: Failed to find {tn} in {rc['db_range_name']}!")
        sys.exit(1)
  else:
    # Check for tests only present in one of the spreadsheets
    df_spreadsheet_0_only = df_spreadsheet[0][~df_spreadsheet[0]["N"].isin(df_spreadsheet[1]["N"])]
    if not df_spreadsheet_0_only.empty:
      print("WARNING: The following tests are only present in {run_config[0]['db_range_name']}. Skipping...")
      print(df_spreadsheet_0_only)
    df_spreadsheet_1_only = df_spreadsheet[1][~df_spreadsheet[1]["N"].isin(df_spreadsheet[0]["N"])]
    if not df_spreadsheet_1_only.empty:
      print("WARNING: The following tests are only present in {run_config[1]['db_range_name']}. Skipping...")
      print(df_spreadsheet_1_only)
    # Just duplicate the test names
    test_names = [[tn, tn] for tn in df_spreadsheet[0][df_spreadsheet[0]["N"].isin(df_spreadsheet[1]["N"])]["N"]]

  for test_pair in test_names:
    success = True
    data = []
    for tn, rc, ss in zip(test_pair, run_config, df_spreadsheet):
      print(f"Looking for {tn} in {rc['db_range_name']}...")
      if rc["single_campaign"]:
        cur_test = ss[(ss["N"] == tn) & (ss["campaign"] == rc["single_campaign"])]
        if cur_test.empty:
          print(f"Test {tn} not found in {rc['db_range_name']} {rc['single_campaign']}. Skipping...")
          success = False
          break
      else:
        cur_test = ss[ss["N"] == tn]
      if len(cur_test) > 1:
        print(f"WARNING: More than one test {tn} found in {rc['db_range_name']}. Using first one...")

      # Read meta data from spreadsheets
      filename = cur_test.iloc[0]["MVM_filename"]
      if not filename:
        print(f"WARNING: Test {tn} in {rc['db_range_name']} has emtpy filename. Skipping...")
        success = False
        break
      rc["meta"] = db.read_meta_from_spreadsheet(cur_test, filename)

      # We've already checked and warned about duplicate tests above, so we just use the first element here
      rc["objname"] = f"{filename}_0"
      meta = rc["meta"][rc["objname"]]

      # Build MVM paths and skip user requested files
      rc["fullpath_mvm"] = f"{rc['data_location']}/{meta['Campaign']}/{meta['MVM_filename']}"
      if not (rc["json"] or rc["fullpath_mvm"].endswith(".txt")):
        print("Adding missing txt extension to MVM path.")
        rc["fullpath_mvm"] += ".txt"
      if rc["json"] and not rc["fullpath_mvm"].endswith(".json"):
        print("Adding missing json extension to MVM path.")
        rc["fullpath_mvm"] += ".json"
      print(f"\nMVM file: {rc['fullpath_mvm']}")

      # Build simulations paths
      rc["fullpath_rwa"] = f"{rc['data_location']}/{meta['Campaign']}/{meta['SimulatorFileName']}"
      # Fix extensions
      if rc["fullpath_rwa"].endswith(".dta"):
        rc["fullpath_rwa"] = rc["fullpath_rwa"][:-4]
      if not rc["fullpath_rwa"].endswith(".rwa"):
        rc["fullpath_rwa"] += ".rwa"
      rc["fullpath_dta"] = rc["fullpath_rwa"].replace("rwa", "dta")
      print(f"Files of simulation: {rc['fullpath_rwa']}, {rc['fullpath_dta']}")

      rc["dataset_name"] = f"{rc['db_range_name'].split('!')[0]}_{meta['Campaign']}"
      print(f"Processing {tn} in {rc['dataset_name']}...")
      data.append(cb.process_run(rc))

    if success:
      plot_3view_comparison(data, run_config, args.output_directory)
