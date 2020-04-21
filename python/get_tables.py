''' Read JSON output from combine.py and prepare summary plots '''

import pandas as pd
import json
import matplotlib.pyplot as plt
import lmfit

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
      # MVM measurements
      ('measured $V_{tidal}$ [ml]', f'${float(row["mean_volume_ml"]):.0f} \pm {float(row["rms_volume_ml"]):.0f}$'),
      ('measured $P_{plateau}$ [cmH2O]', f'${float(row["mean_plateau"]):.0f} \pm {float(row["rms_plateau"]):.0f}$'),
      ('measured $P_{peak}$ [cmH2O]', f'${float(row["mean_peak"]):.0f} \pm {float(row["rms_peak"]):.0f}$'),
      ('measured PEEP [cmH2O]', f'${float(row["mean_peep"]):.0f} \pm {float(row["rms_peep"]):.0f}$'),
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

def process_files(files, output_dir):
  dfs = []
  for i, fname in enumerate(files):
    dfs.append(pd.DataFrame(json.loads(open(fname).read()), index=[i]))
  df = pd.concat(dfs)

  df['Tidal Volume'] = df['Tidal Volume'].astype(int)
  df['simulator_volume_ml'] = df['simulator_volume'] * 10 # TODO patch conversion from ml to cl, remove when appropriate
  df['mean_volume_ml'] = df['mean_volume'] * 10
  df['rms_volume_ml'] = df['rms_volume'] * 10

 # Index(['Campaign', 'Compliance', 'I:E', 'MVM_filename', 'Peep', 'Pinspiratia',
 #      'Rate respiratio', 'Resistance', 'Run', 'SimulatorFileName',
 #      'Tidal Volume', 'cycle_index', 'mean_peak', 'mean_peep', 'mean_plateau',
 #      'mean_volume', 'rms_peak', 'rms_peep', 'rms_plateau', 'rms_volume',
 #      'simulator_volume', 'test_name'],

  # TABLE
  table = get_table(df)
  with open(f'{output_dir}/isotable.tex', 'w') as isotable:
    isotable.write(table)

  # PLOTS
  TV = {
    '$V_{tidal}$ >= 300 ml': df[df['Tidal Volume'] >= 300],
    '300 ml > $V_{tidal}$ >= 50 ml': df[(300 > df['Tidal Volume']) & (df['Tidal Volume'] >= 50)],
    '$V_{tidal}$ < 50 ml': df[df['Tidal Volume'] < 50]
  }

  variables = [
    ('BAP [$cmH_{2}O$]', 'PEEP [$cmH_{2}O$]', 'Peep', 'mean_peep', 'rms_peep'),
    ('$P_{plateau}$ from simulator [$cmH_{2}O$]', 'measured $P_{plateau}$ [$cmH_{2}O$]', 'simulator_plateau', 'mean_plateau', 'rms_plateau'),
    ('set $P_{insp}$ [$cmH_{2}O$]', 'measured $P_{plateau}$ [$cmH_{2}O$]', 'Pinspiratia', 'mean_plateau', 'rms_plateau'),
#   ('set $P_{insp}$ [$cmH_{2}O$]', 'measured $P_{peak}$ [$cmH_{2}O$]', 'Pinspiratia', 'mean_peak', 'rms_peak'),
    ('set $V_{tidal}$ [ml]', 'measured $V_{tidal}$ [ml]', 'Tidal Volume', 'mean_volume_ml', 'rms_volume_ml'),
    ('$V_{tidal}$ from simulator [ml]', 'measured $V_{tidal}$ [ml]', 'simulator_volume_ml', 'mean_volume_ml', 'rms_volume_ml'),
  ]

  line = lmfit.models.LinearModel()
  for xname, yname, setval, mean, rms in variables:
    fig, ax = plt.subplots(1, 1)
    fig.canvas.set_window_title(f'x={setval}, y={mean}, yerr={rms}')

    for setname, data in TV.items():
#     params = line.guess(data[mean], x=data[setval])
#     res = line.fit(data[mean], params, x=data[setval], weights=1./data[rms])

      ax.errorbar(data[setval], data[mean], yerr=data[rms], fmt='o', label=setname)

    # linear fit
    df_to_fit = df
    if setval == 'simulator_volume_ml':
      df_to_fit = df_to_fit[df_to_fit[setval] > 50] # 201.12.1.104 from ISO
    params = line.guess(df_to_fit[mean], x=df_to_fit[setval])
    res = line.fit(df_to_fit[mean], params, x=df_to_fit[setval], weights=1./df_to_fit[rms])

    print(res.fit_report())
   #fitstring = f'${res.best_values["intercept"]} \pm {(res.best_values["slope"]-1)*100}$%'
    fitstring = f'$\pm$({abs(res.best_values["intercept"]):.1f} +({abs((res.best_values["slope"]-1)*100):.1f}% of the value))'
    ax.plot(df_to_fit[setval], res.best_fit, '-', label=fitstring)
    ax.legend()
    ax.set_xlabel(f'{xname}')
    ax.set_ylabel(f'{yname}')
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
  args = parser.parse_args()

  process_files(args.input, output_dir=args.output_dir)
