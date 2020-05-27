''' Read JSON output from combine.py and prepare summary plots '''

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import lmfit

from mvmconstants import *


def get_table(df):
  toprint = []

  for i, row in df.iterrows():
    what = [
      # ISO test details
      ('$V_{tidal}$ [ml]', f'${float(row["Tidal Volume"]):.0f}$'),
      ('$C$ [ml/cmH2O]', f'${float(row["Compliance"]):.0f}$'),
      ('$R$ [cmH2O/l/s]', f'${float(row["Resistance"]):.0f}$'),
      ('rate [breaths/min]', f'${float(row["Rate respiratio"]):.0f}$'),
      ('I:E', f'${float(row["I:E"]):.2f}$'),
      ('$P_{insp}$ [cmH2O]', f'${float(row["Pinspiratia"]):.0f}$'),
      ('$O_{2}$', f'$21\%$'), # TODO add oxygen to json
      ('BAP [cmH2O]', f'${float(row["Peep"]):.0f}$'),
      # SIM measurements
      ('simulator $V_{tidal}$ [ml]', f'${float(row["simulator_volume_ml"]):.0f}$'),
      ('simulator $P_{plateau}$ [cmH2O]', f'${float(row["simulator_plateau"]):.0f}$'),
      ('simulator I:E', f'${float(row["simulator_iovere"]):.0f}$'),
      ('simulator Frequenct', f'${float(row["simulator_frequency"]):.0f}$'),
      # MVM measurements
      ('measured $V_{tidal}$ [ml]', f'${float(row["mean_volume_ml"]):.0f} \pm {float(row["rms_volume_ml"]):.0f}$'),
      ('measured $P_{plateau}$ [cmH2O]', f'${float(row["mean_plateau"]):.0f} \pm {float(row["rms_plateau"]):.0f}$'),
      ('measured $P_{peak}$ [cmH2O]', f'${float(row["mean_peak"]):.0f} \pm {float(row["rms_peak"]):.0f}$'),
      ('measured PEEP [cmH2O]', f'${float(row["mean_peep"]):.0f} \pm {float(row["rms_peep"]):.0f}$'),
      ('measured I:E', f'${float(row["mean_iovere"]):.0f}$'),
      ('measured Frequency', f'${float(row["mean_frequency"]):.0f}$'),
    ]

    if i == 0:
      columns = [ title for title, _ in what ]
      toprint.append(r'''
        \documentclass[a4paper]{article}
        \usepackage[margin=1.0cm]{geometry}
        \usepackage{rotating}
        \title{ISO test table}
        \begin{document}
        \maketitle
        \begin{sidewaystable}
        \tiny
        \begin{tabular}{''')
      cstring = 'c'*(len(what)+1)
      toprint[-1] = f'{toprint[-1]}{cstring}}}'
      toprint.append(' & '.join(columns))
      toprint[-1] = f'{toprint[-1]}\\\\\\hline'# horizontal bar and new line in LaTeX

    content = [ cont for _, cont in what ]
    toprint.append(' & '.join(content))
    toprint[-1] = f'{toprint[-1]}\\\\' # new line in LaTeX

  toprint.append(r'''
    \end{tabular}
    \end{sidewaystable}
    \end{document}
  ''')
  output = '\n'.join(toprint)
  print(output)
  return output


def process_files(files, output_dir, save_h5=False):
  dfs = []
  for i, fname in enumerate(files):
    dfs.append(pd.DataFrame(json.loads(open(fname).read()), index=[i]))
  df = pd.concat(dfs)
  if save_h5:
    df.to_hdf(f'{output_dir}/summary.h5', 'MVM')

  df['Tidal Volume'] = df['Tidal Volume'].astype(int)
  df['simulator_volume_ml'] = df['simulator_volume'] * 10 # TODO patch conversion from ml to cl, remove when appropriate
  df['mean_volume_ml'] = df['mean_volume'] * 10
  df['rms_volume_ml'] = df['rms_volume'] * 10
  df['max_volume_ml'] = df['max_volume'] * 10
  df['min_volume_ml'] = df['min_volume'] * 10

 # Index(['Campaign', 'Compliance', 'I:E', 'MVM_filename', 'Peep', 'Pinspiratia',
 #      'Rate respiratio', 'Resistance', 'Run', 'SimulatorFileName',
 #      'Tidal Volume', 'cycle_index', 'mean_peak', 'mean_peep', 'mean_plateau',
 #      'mean_volume', 'rms_peak', 'rms_peep', 'rms_plateau', 'rms_volume',
 #      'simulator_volume', 'test_name'],

  # TABLE
  table = get_table(df)
  with open(f'{output_dir}/isotable.tex', 'w') as isotable:
    isotable.write(table)

  ## Plot different series by tidal volume as per ISO
  TV = {
    '$V_{tidal}$ >= 300 ml': df[df['Tidal Volume'] >= 300],
    '300 ml > $V_{tidal}$ >= 50 ml': df[(300 > df['Tidal Volume']) & (df['Tidal Volume'] >= 50)],
    '$V_{tidal}$ < 50 ml': df[df['Tidal Volume'] < 50]
  }

  ## Variables to plot.  See start of for loop below for convention
  ## x-axis and y-axis must use the same unit
  variables = [
    ('BAP', 'PEEP', '[$cmH_{2}O$]', 'Peep', 'mean_peep', 'rms_peep', 'max_peep', 'min_peep'),
    ('$P_{plateau}$ from lung simulator', '$P_{plateau}$ from MVM', '[$cmH_{2}O$]', 'simulator_plateau', 'mean_plateau', 'rms_plateau', 'max_plateau', 'min_plateau'),
    ('set $P_{insp}$', '$P_{plateau}$ from MVM', '[$cmH_{2}O$]', 'Pinspiratia', 'mean_plateau', 'rms_plateau', 'max_plateau', 'min_plateau'),
#   ('set $P_{insp}$', '$P_{peak}$ from MVM', '[$cmH_{2}O$]', 'Pinspiratia', 'mean_peak', 'rms_peak', 'max_peak', 'min_peak'),
    ('target $V_{tidal}$', '$V_{tidal}$ from MVM', '[ml]', 'Tidal Volume', 'mean_volume_ml', 'rms_volume_ml', 'max_volume_ml', 'min_volume_ml'),
    ('$V_{tidal}$ from lung simulator', '$V_{tidal}$ from MVM', '[ml]', 'simulator_volume_ml', 'mean_volume_ml', 'rms_volume_ml', 'max_volume_ml', 'min_volume_ml'),
    ('I:E from lung simulator', 'I:E from MVM', '', 'simulator_iovere', 'mean_iovere', 'rms_iovere', 'max_iovere', 'min_iovere'),
    ('set I:E', 'I:E from MVM', '', 'I:E', 'mean_iovere', 'rms_iovere', 'max_iovere', 'min_iovere'),
    ('breath rate from lung simulator', 'breath rate from MVM', '[breaths/min]', 'simulator_frequency', 'mean_frequency', 'rms_frequency', 'max_frequency', 'min_frequency'),
    ('set breath rate', 'breath rate from MVM', '[breaths/min]', 'Rate respiratio', 'mean_frequency', 'rms_frequency','max_frequency', 'min_frequency'),
  ]

  ## Retrieve maximum errors, for use in loop
  maximum_bias_error = {
    'mean_peep' : MVM.maximum_bias_error_peep,
    'mean_plateau' : MVM.maximum_bias_error_pinsp,
    'mean_volume_ml' : MVM.maximum_bias_error_volume,
    'mean_iovere' : MVM.maximum_bias_error_iovere,
    'mean_frequency' : MVM.maximum_bias_error_frequency
  }
  maximum_linearity_error = {
    'mean_peep' : MVM.maximum_linearity_error_peep,
    'mean_plateau' : MVM.maximum_linearity_error_pinsp,
    'mean_volume_ml' : MVM.maximum_linearity_error_volume,
    'mean_iovere' : MVM.maximum_linearity_error_iovere,
    'mean_frequency' : MVM.maximum_linearity_error_frequency
  }

  line = lmfit.models.LinearModel()
  for xname, yname, unit, setval, mean, rms, max, min in variables:
    print(f'\nMaking compliance summary plot for {yname} vs {xname}')
    fig, ax = plt.subplots(1, 1)
    fig.canvas.set_window_title(f'x={setval}, y={mean}, yerr=Full range of measured values')

    for setname, data in TV.items():
#     params = line.guess(data[mean], x=data[setval])
#     res = line.fit(data[mean], params, x=data[setval], weights=1./data[rms])

      # provide (min, max) asymmetric "error bars" to show full range
      min_values_subtr = data[mean] - data[min]
      max_values_subtr = data[max] - data[mean]
      ax.errorbar(data[setval], data[mean], yerr=[min_values_subtr, max_values_subtr], fmt='o', label=setname)

    # define data frame to consider for compliance tests
    df_to_fit = df
    if setval == 'simulator_volume_ml':
      df_to_fit = df_to_fit[df_to_fit[setval] > 50] # 201.12.1.104 from ISO

    # linear fit
    params = line.guess(df_to_fit[mean], x=df_to_fit[setval])
    res = line.fit(df_to_fit[mean], params, x=df_to_fit[setval], weights=1./df_to_fit[rms])
    print(res.fit_report())
    #fitstring = f'${res.best_values["intercept"]} \pm {(res.best_values["slope"]-1)*100}$%'
    fitstring = f'Best fit: y = {res.best_values["intercept"]:.1f} + {res.best_values["slope"]:.2f}x'
    ax.plot(df_to_fit[setval], res.best_fit, '-', label=fitstring)

    # show compliance requirement band
    maximum_error_string = f'$\pm$({maximum_bias_error[mean]:.1f} +({(maximum_linearity_error[mean]*100):.1f}% of {xname})) {unit}'
    x_limit = np.arange(0.0, df_to_fit[setval].max()*1.2, 0.01)
    max_limit = maximum_bias_error[mean] + (1 + maximum_linearity_error[mean]) * x_limit
    min_limit = -maximum_bias_error[mean] + (1 - maximum_linearity_error[mean]) * x_limit
    ax.fill_between(x_limit, min_limit, max_limit, facecolor='green', alpha=0.2, label=maximum_error_string)

    ax.legend()
    ax.set_xlabel(f'{xname} {unit}')
    ax.set_ylabel(f'{yname} {unit}')
    ax.set_xlim(0, df_to_fit[setval].max()*1.2)
    ax.set_ylim(0, df_to_fit[mean].max()*1.2)
    fig.savefig(f'{output_dir}/isoplot_{setval}.pdf')
    #fig.show()
  plt.show()


if __name__ == '__main__':
  import argparse
  import style

  parser = argparse.ArgumentParser(description='prepare run summary tables from JSON')
  parser.add_argument("input", help="name of the input file (.txt)", nargs='+')
  parser.add_argument("-p", "--plot", action='store_true', help="show plots")
  parser.add_argument("-o", "--output-dir", type=str, help="output folder for images and LaTeX", default='.')
  parser.add_argument("-5", "--save-h5", action='store_true', help="save h5 file with summary DataFrame")
  args = parser.parse_args()

  process_files(args.input, output_dir=args.output_dir, save_h5=args.save_h5)
