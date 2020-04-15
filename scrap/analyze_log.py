#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns

import json
import argparse

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]

layout = (3,3)

tv_correction = 310./273.15*1013. # correct tidal volume from STP to BTP

'''
parser = argparse.ArgumentParser(prog='analyze_log')
parser.add_argument('log', help="Log file to write to")
parser.add_argument('-b', '--btp', help="Convert tidal volume to BTP", action="store_true", default=False)
args = parser.parse_args()
'''

fname0='./log_20200410_pin15_peep5_bpm20_with_lung_c10_r50.json'
# Load in JSON file
with open(fname0) as log_file:
    log_data = json.load(log_file)["data"]

offset = log_data[0]["ts"]

ts = []
phase = []
ps1_p = []
ps1_t = []
fs1_r = []
inlet = []
outlet = []

for rec in log_data:
    ts.append((rec["ts"] - offset))
    phase.append(rec["phase"])
    ps1_p.append(rec["ps1_p"])
    ps1_t.append(rec["ps1_t"])
    fs1_r.append(rec["fs1_r"])
    inlet.append(rec["inlet"])
    outlet.append(rec["outlet"])

ts = np.array(ts)
phase = np.array(phase)
ps1_p = np.array(ps1_p)
ps1_t = np.array(ps1_t)
fs1_r = np.array(fs1_r)
inlet = np.array(inlet)
outlet = np.array(outlet)

dupes = []
for i in range(1, len(ts)):
    if(ts[i] == ts[i-1]):
        if(ps1_p[i] != ps1_p[i-1]):
            print(ts[i], ": p:", ps1_p[i], "!=", ps1_p[i-1])
        if(ps1_t[i] != ps1_t[i-1]):
            print(ts[i], ": t:", ps1_t[i], "!=", ps1_t[i-1])
        if(fs1_r[i] != fs1_r[i-1]):
            print(ts[i], ": f:", fs1_r[i], "!=", fs1_r[i-1])
        # if(phase[i] != phase[i-1]):
        #     print(ts[i], ": ph:", phase[i], "!=", phase[i-1])
        # if(inlet[i] != inlet[i-1]):
        #     print(ts[i], ": i:", inlet[i], "!=", inlet[i-1])
        # if(outlet[i] != outlet[i-1]):
        #     print(ts[i], ": o:", outlet[i], "!=", outlet[i-1])
        dupes.append(i-1)

ts = np.delete(ts,dupes)
ps1_p = np.delete(ps1_p,dupes)
ps1_t = np.delete(ps1_t,dupes)
fs1_r = np.delete(fs1_r,dupes)
phase = np.delete(phase,dupes)
inlet = np.delete(inlet,dupes)
outlet = np.delete(outlet,dupes)

dts = np.diff(ts)
fig = plt.figure(figsize=layout)

plotdt = plt.subplot2grid(layout, (2, 0))
plotdt.plot(ts[1:], dts)
# plotdt.set_title("Timing")
plotdt.set_xlabel("Time (s)")
plotdt.set_ylabel("Time difference (s)")

n = np.count_nonzero(dts)
if(n != len(dts)):
    print("Found",len(dts)-np.count_nonzero(dts), "timestamp duplications in", len(ts), "readings. -> ratio", float(len(dts)-np.count_nonzero(dts))/float(len(ts)))


# ts_intp = np.arange(np.min(ts), np.max(ts), 0.01)
# p_intp = np.interp(ts_intp, ts, ps1_p)
# plot_acf(p_intp, lags=5000)

t_last = -1
period = np.array([])
t_inh = np.array([])
p_max = np.array([])
f_max = np.array([])
tidalV = np.array([])
maxp = 0.
maxf = 0.
tv = 0.

for i in range(1, len(ts)):
    if(phase[i] == "INHALE" and phase[i-1] != "INHALE"):
        if(t_last >= 0):
            period = np.append(period, ts[i]-t_last)
            tidalV = np.append(tidalV, tv)
            tv = 0.
        t_last = ts[i]
    elif(phase[i] == "HOLD" and phase[i-1] == "INHALE"):
        t_inh = np.append(t_inh, ts[i] - t_last)
    if(phase[i] == "WAIT" and phase[i-1] == "EXHALE"):
        p_max = np.append(p_max, maxp)
        maxp = 0.
        f_max = np.append(f_max, maxf)
        maxf = 0.
    elif(phase[i] == "OFF"):
        maxp = 0.
        maxf = 0.
    else:
        if(ps1_p[i] > maxp):
            maxp = ps1_p[i]
        if(fs1_r[i] > maxf):
            maxf = fs1_r[i]
        #tv_fac = tv_correction/(1013.+ps1_p[i]) if args.btp else 1.
        tv_fac=1.
        tv = tv + fs1_r[i]*(ts[i]-ts[i-1])/60. * tv_fac

plot_period = plt.subplot2grid(layout,(0,0))
plot_period.plot(range(len(period)),period)
plot_period.set_title("Timing")
plot_period.set_xlabel("Cycle number")
plot_period.set_ylabel("Cycle length (s)")

plot_t_inh = plt.subplot2grid(layout,(1,0))
plot_t_inh.plot(range(len(t_inh)),t_inh)
# plot_t_inh.set_title("Cycle length")
plot_t_inh.set_xlabel("Cycle number")
plot_t_inh.set_ylabel("Inhale time (s)")

plot_p_max = plt.subplot2grid(layout,(0,1))
plot_p_max.plot(range(len(p_max)),p_max)
plot_p_max.set_title("Values")
plot_p_max.set_xlabel("Cycle number")
plot_p_max.set_ylabel("Max. pressure")

plot_f_max = plt.subplot2grid(layout,(1,1))
plot_f_max.plot(range(len(f_max)),f_max)
plot_f_max.set_xlabel("Cycle number")
plot_f_max.set_ylabel("Max. flow")

plot_tidal = plt.subplot2grid(layout,(2,1))
plot_tidal.plot(range(len(tidalV)),tidalV)
plot_tidal.set_xlabel("Cycle number")
plot_tidal.set_ylabel("Tidal Volume")

plot_prange = plt.subplot2grid(layout,(0,2))
sns.boxplot(x=ps1_p)
plot_prange.set_xlabel("Pressure")
plot_prange.set_title("Ranges/Outliers")

plot_frange = plt.subplot2grid(layout,(1,2))
sns.boxplot(x=fs1_r)
plot_frange.set_xlabel("Flow")

plot_trange = plt.subplot2grid(layout,(2,2))
sns.boxplot(x=ps1_t)
plot_trange.set_xlabel("Temperature")


plt.show()
