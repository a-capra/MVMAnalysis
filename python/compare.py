import db
import combine as cb
import mvmio as io
import matplotlib.pyplot as plt


# Adapted from plot_arxXiv_canvases
def plot_3views(data, run_config, output_directory):
  # Test name is the same for both datasets by construction
  test_name = run_config[0]["meta"]["test_name"]

  fig31, ax31 = plt.subplots(3, 1)
  ax31 = ax31.flatten()
  run_config[0]["marker"] = "+"
  run_config[1]["marker"] = "x"

  for rc, d in zip(run_config, data):
    my_selected_cycle = rc["meta"]["cycle_index"]

    # Simulator subset
    d["sim_sel"] = d["sim"][(d["sim"]["start"] >= d["start_times"][my_selected_cycle]) & (d["sim"]["start"] < (d["start_times"][my_selected_cycle] + 6))]

    # Ventilator subset
    first_time_bin = d["sim_sel"]["dt"].iloc[0]
    last_time_bin = d["sim_sel"]["dt"].iloc[-1]
    d["mvm_sel"] = d["mvm"][(d["mvm"]["dt"] > first_time_bin) & (d["mvm"]["dt"] < last_time_bin)]

    d["sim_sel"].loc[:, "total_vol"] = d["sim_sel"]["total_vol"] - d["sim_sel"]["total_vol"].min()

    dataset_name = rc["db_range_name"].split("!")[0]
    d["sim_sel"].plot(ax=ax31[0], x="dt", y="total_flow", label=f"SIM {dataset_name} flux [l/min]", c="r", marker=rc["marker"])
    d["sim_sel"].plot(ax=ax31[1], x="dt", y="airway_pressure", label=f"SIM {dataset_name} airway pressure [cmH2O]", c="r", marker=rc["marker"])
    d["sim_sel"].plot(ax=ax31[2], x="dt", y="total_vol", label=f"SIM {dataset_name} tidal volume [cl]", c="r", marker=rc["marker"])
    d["mvm_sel"].plot(ax=ax31[0], x="dt", y="display_flux", label=f"MVM {dataset_name} flux [l/min]", c="b", marker=rc["marker"])
    d["mvm_sel"].plot(ax=ax31[1], x="dt", y="airway_pressure", label=f"MVM {dataset_name} airway pressure [cmH2O]", c="b", marker=rc["marker"])
    d["mvm_sel"].plot(ax=ax31[2], x="dt", y="tidal_volume", label=f"MVM {dataset_name} tidal volume [cl]", c="b", marker=rc["marker"])

  ax31[0].set_xlabel("Time [s]")
  ax31[1].set_xlabel("Time [s]")
  ax31[2].set_xlabel("Time [s]")

  fig31.suptitle(f"Test nr. {test_name}", weight="heavy")
  figpath = f"{output_directory}/{test_name}.png"
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
    d["mvm"] = io.get_mvm_json(rc["fullpath_mvm"]) if rc["json"] else io.get_mvm_df(fname=rc["fullpath_mvm"], sep=rc["mem_sep"], configuration=rc["mvm_col"])

    cb.add_timestamp(d["mvm"])

    cb.correct_sim(d["sim"])
    # Add time shift
    cb.apply_manual_shift(sim=d["sim"], mvm=d["mvm"], manual_offset=rc["offset"])

    cb.add_pv2_status(d["mvm"])

    # Compute cycle start based on PV2
    d["start_times"] = cb.get_start_times(d["mvm"])

    d["reaction_times"] = cb.get_reaction_times(d["sim"], d["start_times"])

    cb.add_cycle_info(sim=d["sim"], mvm=d["mvm"], start_times=d["start_times"], reaction_times=d["reaction_times"])

    cb.add_chunk_info(d["sim"])

    # Compute tidal volume etc.
    add_clinical_values(d["sim"])
    d["respiration_rate"], d["inspiration_duration"] = cb.measure_clinical_values(d["mvm"], start_times=d["start_times"])

    cb.add_run_info(d["sim"])

    # Choose the flux to plot
    d["mvm"]["display_flux"] = d["mvm"]["flux"]


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Compare two datasets.")
  parser.add_argument("db_range_name_1", help="Name and range of the metadata spreadsheet for the first dataset")
  parser.add_argument("data_location_1", help="Path to the first dataset.")
  parser.add_argument("db_range_name_2", help="Name and range of the metadata spreadsheet for the second dataset")
  parser.add_argument("data_location_2", help="Path to the second dataset.")
  parser.add_argument("-d", "--output-directory", type=str, help="Plot output directory.", default="plots_iso")
  parser.add_argument("-S", "--skip-files", type=str, help="Skip listed files in both datasets.", nargs="+", default="")
  parser.add_argument("--campaign-1", type=str, help="Process only a single campaign of first dataset.", default="")
  parser.add_argument("--campaign-2", type=str, help="Process only a single campaign of second dataset.", default="")
  parser.add_argument("--j1", action="store_true", help="Try to read first dataste as JSON instead of CSV.")
  parser.add_argument("--j2", action="store_true", help="Try to read second dataste as JSON instead of CSV.")
  parser.add_argument("--offset-1", type=float, help="Time offset between first vent and sim datasets.")
  parser.add_argument("--offset-2", type=float, help="Time offset between second vent and sim datasets.")
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

  # Check for tests only present in one of the spreadsheets
  df_spreadsheet_1_only = df_spreadsheet[0][~df_spreadsheet[0]["N"].isin(df_spreadsheet[1]["N"])]
  if not df_spreadsheet_1_only.empty:
    print("WARNING: The following tests are only present in the first dataset. Skipping...")
    print(df_spreadsheet_1_only)
  df_spreadsheet_2_only = df_spreadsheet[1][~df_spreadsheet[1]["N"].isin(df_spreadsheet[0]["N"])]
  if not df_spreadsheet_2_only.empty:
    print("WARNING: The following tests are only present in the second dataset. Skipping...")
    print(df_spreadsheet_2_only)

  test_names = df_spreadsheet[0]["N"].unique()
  for test_name in test_names:
    cur_tests = []
    # Warn about duplicate tests
    cur_tests.append(df_spreadsheet[0][df_spreadsheet[0]["N"] == test_name])
    if len(cur_tests[0]) > 1:
      print(f"WARNING: More than one test {test_name} found in first dataset. Using first one...")
    cur_tests.append(df_spreadsheet[1][df_spreadsheet[1]["N"] == test_name])
    # Skip tests not present in second dataset. We've already warned about this above.
    if cur_tests[1].empty:
      continue
    elif len(cur_tests[1]) > 1:
      print(f"WARNING: More than one test {test_name} found in second dataset. Using first one...")

    for i, rc in enumerate(run_config):
      # Read meta data from spreadsheets
      filename = cur_tests[i].iloc[0]["MVM_filename"]
      rc["meta"] = db.read_meta_from_spreadsheet(cur_tests[i], filename)

      # Only use first element
      rc["objname"] = f"{filename}_0"
      rc["meta"] = rc["meta"][rc["objname"]]

      # Only process selected campaigns
      if rc["single_campaign"] and (rc["meta"]["Campaign"] != rc["single_campaign"]):
        print(f"Test {test_name} not in selected campaign {rc['campaign']}. Skipping...")
        continue

      # Build MVM paths and skip user requested files
      rc["fullpath_mvm"] = f"{rc['data_location']}/{rc['meta']['Campaign']}/{rc['meta']['MVM_filename']}"
      if not rc["fullpath_mvm"].endswith(".txt"):
        rc["fullpath_mvm"] += ".txt"
      print(f"\nMVM file {i+1}: {rc['fullpath_mvm']}")
      if rc["fullpath_mvm"].split("/")[-1] in args.skip_files:
        print("\tskipping per user request...")
        continue

      # Build simulations paths
      rc["fullpath_rwa"] = f"{rc['data_location']}/{rc['meta']['Campaign']}/{rc['meta']['SimulatorFileName']}"
      # Fix extensions
      if rc["fullpath_rwa"].endswith(".dta"):
        rc["fullpath_rwa"] = rc["fullpath_rwa"][:-4]
      if not rc["fullpath_rwa"].endswith(".rwa"):
        rc["fullpath_rwa"] += ".rwa"
      rc["fullpath_dta"] = rc["fullpath_rwa"].replace("rwa", "dta")
      print(f"Files of simulation {i+1}: {rc['fullpath_rwa']}, {rc['fullpath_dta']}")

    process_run(rc, output_directory)
