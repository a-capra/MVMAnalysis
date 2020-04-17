import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import argparse

parser = argparse.ArgumentParser(prog='plot_log')
parser.add_argument('log', help="Log file to read from")
parser.add_argument('-i', '--interactive',
                    help="Open interactive plot window", action="store_true", default=False)
parser.add_argument('-t', '--timerange', help="Zoom time axis to range <t1> <t2>",
                    nargs=2, type=float, default=[0, -1])
args = parser.parse_args()

# Load in JSON file
with open(args.log) as log_file:
    log_data = json.load(log_file)["data"]

offset = log_data[0]["ts"]

ts = []
phase = []
ps1_p = []
ps1_t = []
fs1_r = []
inlet = []
outlet = []
phase_converted = []
inlet_converted = []
outlet_converted = []

for rec in log_data:
    ts.append((rec["ts"] - offset))
    phase.append(rec["phase"])
    ps1_p.append(rec["ps1_p"])
    ps1_t.append(rec["ps1_t"])
    fs1_r.append(rec["fs1_r"])
    inlet.append(rec["inlet"])
    outlet.append(rec["outlet"])

    # Modify the text fields to data points for easy viewing in the combined plot
    if rec["phase"] == "EXHALE":
        phase_value = -50
    elif rec["phase"] == "HOLD":
        phase_value = -60
    elif rec["phase"] == "INHALE":
        phase_value = -70
    elif rec["phase"] == "OFF":
        phase_value = -80
    elif rec["phase"] == "WAIT":
        phase_value = -90
    elif rec["phase"] == "UNKNOWN":
        phase_value = -100
    else:
        phase_value = -110

    phase_converted.append(phase_value)
    inlet_converted.append(-10 if rec["inlet"] == "OPEN" else -20)
    outlet_converted.append(-30 if rec["outlet"] == "OPEN" else -40)

t_last = -1
tidalV = np.array([])
tsv = np.array([])
tv = 0.

for i in range(1, len(ts)):
    if(phase[i] == "INHALE" and phase[i-1] != "INHALE"):
        if(t_last >= 0.0):
            tidalV = np.append(tidalV, tv)
            tsv = np.append(tsv,ts[i])
            tv = 0.
        t_last = ts[i]
    elif(phase[i] == "WAIT" and phase[i-1] == "EXHALE"):
        continue
    elif(phase[i] == "OFF"):
        continue
    else:
        tv = tv + fs1_r[i]*(ts[i]-ts[i-1])/60.


fig = plt.figure(figsize=(12, 8))
# , (plot_phase, plot_fs1_rate), (plot_ps1_pres, plot_ps1_temp), (plot_pv1, plot_pv2)) = plt.subplots(2, 3)
#gs = fig.add_gridspec(3, 3)
# ax = plt.subplots(7)
#plot_combined = fig.add_subplot(gs[0:,0])
plot_combined = plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=3)
line1, = plot_combined.plot(ts, phase_converted, label='Cycle Phase')
line2, = plot_combined.plot(ts, ps1_p, label='PS1 Pressure')
line3, = plot_combined.plot(tsv, tidalV, label='Tidal Volume')
#line3, = plot_combined.plot(ts, ps1_t, label='PS1 Temp')
line4, = plot_combined.plot(ts, fs1_r, label='FS1 Flow Rate')
line5, = plot_combined.plot(ts, inlet_converted, label='PV1 Inlet Valve')
line6, = plot_combined.plot(ts, outlet_converted, label='PV2 Outlet Valve')
plot_combined.set_title("Combined Telemetry")
plot_combined.set_xlabel("Time (s)")
plot_combined.set_ylabel("Various")
plot_combined.set_yticks([])
leg = plot_combined.legend(loc='upper right', shadow=True)
leg.get_frame().set_alpha(0.4)
if(args.timerange[1] > args.timerange[0]):
    plot_combined.set_xlim(args.timerange)

plot_phase = plt.subplot2grid((3, 3), (1, 0), rowspan=1, colspan=1)
plot_phase.plot(ts, phase, label='Control Loop Cycle Phase')
plot_phase.set_title("Cycle Phase")
plot_phase.set_xlabel("Time (s)")
plot_phase.set_ylabel("Cycle")
if(args.timerange[1] > args.timerange[0]):
    plot_phase.set_xlim(args.timerange)

plot_fs1_rate = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=1)
plot_fs1_rate.plot(ts, fs1_r, label='FS1 Flow Rate')
plot_fs1_rate.set_title("FS1 Flow Rate")
plot_fs1_rate.set_xlabel("Time (s)")
plot_fs1_rate.set_ylabel("Standard Litre per minute (slm)")
if(args.timerange[1] > args.timerange[0]):
    plot_fs1_rate.set_xlim(args.timerange)

plot_ps1_pres = plt.subplot2grid((3, 3), (1, 1), rowspan=1, colspan=1)
plot_ps1_pres.plot(ts, ps1_p, label='PS1 Pressure')
plot_ps1_pres.set_title("PS1 Pressure")
plot_ps1_pres.set_xlabel("Time (s)")
plot_ps1_pres.set_ylabel("$cmH_2O$")
if(args.timerange[1] > args.timerange[0]):
    plot_ps1_pres.set_xlim(args.timerange)

plot_tidal_vol = plt.subplot2grid((3, 3), (2, 1), rowspan=1, colspan=1)
plot_tidal_vol.plot(tsv, tidalV, label='Tidal Volume')
plot_tidal_vol.set_title("Tidal Volume")
plot_tidal_vol.set_xlabel("Time (s)")
plot_tidal_vol.set_ylabel("$L$")
if(args.timerange[1] > args.timerange[0]):
    plot_tidal_vol.set_xlim(args.timerange)
# plot_ps1_temp = plt.subplot2grid((3, 3), (2, 1), rowspan=1, colspan=1)
# plot_ps1_temp.plot(ts, ps1_t, label='PS1 Temp')
# plot_ps1_temp.set_title("PS1 Temperature")
# plot_ps1_temp.set_xlabel("Time (s)")
# plot_ps1_temp.set_ylabel("Celsius")
# if(args.timerange[1] > args.timerange[0]):
#     plot_ps1_temp.set_xlim(args.timerange)

plot_pv1 = plt.subplot2grid((3, 3), (1, 2), rowspan=1, colspan=1)
plot_pv1.plot(ts, inlet, label='PV1 Inlet Valve')
plot_pv1.set_title("PV1 Inlet Valve")
plot_pv1.set_xlabel("Time (s)")
if(args.timerange[1] > args.timerange[0]):
    plot_pv1.set_xlim(args.timerange)

plot_pv2 = plt.subplot2grid((3, 3), (2, 2), rowspan=1, colspan=1)
plot_pv2.plot(ts, outlet, label='PV2 Outlet Valve')
plot_pv2.set_title("PV2 Outlet Valve")
plot_pv2.set_xlabel("Time (s)")
if(args.timerange[1] > args.timerange[0]):
    plot_pv2.set_xlim(args.timerange)

plt.subplots_adjust(wspace=0.2, hspace=0.3)

lines = [line1, line2, line3, line4, line5, line6]
lined = dict()
for legline, origline in zip(leg.get_lines(), lines):
    legline.set_picker(10)  # 5 pts tolerance
    lined[legline] = origline


def onpick(event):
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
    legline = event.artist
    origline = lined[legline]
    vis = not origline.get_visible()
    origline.set_visible(vis)
    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled
    if vis:
        legline.set_alpha(1.0)
    else:
        legline.set_alpha(0.2)
    fig.canvas.draw()


fig.canvas.mpl_connect('pick_event', onpick)

plt.tight_layout()
if(args.interactive):
    plt.show()
else:
    plt.savefig("plots/" + args.log + ".png", bbox_inches='tight')
