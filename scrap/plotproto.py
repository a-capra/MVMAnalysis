import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from pathlib import Path
import json


compliance=50 # mL/cmH2O
resistance=20 # cmH20/L/s 
respiratory_rate=20 # bpm
inspiratory_pressure=15 # cmH2O
inspiratory_time=1.5 # s
PEEP=5 # cmH2O

#==============================================================================


fname0='./data/log_20200410_pin15_peep5_bpm20_with_lung_c{}_r{}.json'.format(compliance,resistance)
# Load in JSON file
with open(fname0) as log_file:
    log_data = json.load(log_file)["data"]

offset = log_data[0]["ts"]
print('json offset is',offset)

ts = np.array([])
phase = []
ps1_p = np.array([])
fs1_r = np.array([])

t_last = -1
tidalV = np.array([])
tsv = np.array([])
tv = 0.

for rec in log_data:
    ts=np.append(ts,(rec["ts"] - offset))
    phase.append(rec["phase"])
    ps1_p=np.append(ps1_p,rec["ps1_p"])
    fs1_r=np.append(fs1_r,rec["fs1_r"])
    
dupes = []
for i in range(1, len(ts)):
    if(ts[i] == ts[i-1]):
        #if(ps1_p[i] != ps1_p[i-1]):
        #    print(ts[i], ": p:", ps1_p[i], "!=", ps1_p[i-1])
        #if(fs1_r[i] != fs1_r[i-1]):
        #    print(ts[i], ": f:", fs1_r[i], "!=", fs1_r[i-1])
        dupes.append(i-1)

ts = np.delete(ts,dupes)
ps1_p = np.delete(ps1_p,dupes)
fs1_r = np.delete(fs1_r,dupes)
phase = np.delete(phase,dupes)

   
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
    
    
#==============================================================================
lab1=['Time (sec)',	'Airway Pressure (cmH2O)',	'Muscle Pressure (cmH2O)',	'Tracheal Pressure (cmH2O)',	'Chamber 1 Volume (L)',	'Chamber 2 Volume (L)',	'Total Volume (L)',	'Chamber 1 Pressure (cmH2O)',	'Chamber 2 Pressure (cmH2O)',	'Breath File Number (#)',	'Aux 1 (V)',	'Aux 2 (V)',	'Oxygen Sensor (V)'] # rwa file

lab2=['Breath Number (#)','Compressed Volume (mL)','Airway Pressure (cmH2O)','Muscle Pressure (cmH2O)','Total Volume (mL)', 'Total Flow (L/min)','Chamber 1 Pressure (cmH2O)', 'Chamber 2 Pressure (cmH2O)', 'Chamber 1 Volume (mL)','Chamber 2 Volume (mL)','Chamber 1 Flow (L/min)','Chamber 2 Flow (L/min)','Tracheal Pressure (cmH2O)','Ventilator Volume (mL)','Ventilator Flow (L/min)','Ventilator Pressure (cmH2O)'] # dta file

fname1="./data/breathC{}R{}.rwa".format(compliance,resistance)
my_file = Path(fname1)
if not my_file.is_file():
    print('Error!')
print("Reading from:",fname1)
data1=np.loadtxt(fname1,skiprows=4)

fname2="./data/breathC{}R{}.dta".format(compliance,resistance)
print("Reading from:",fname2)
data2=np.loadtxt(fname2,skiprows=4)

print("Number of entries:",len(data1),len(data2))

Time=data1[:,0]
#Paw1=data1[:,1]
Paw2=data2[:,2]
Vtot1=data1[:,6]
#Vtot2=data2[:,4]
Ftot=data2[:,5]
Breath1=data1[:,-4]
Breath2=data2[:,0]

print("Time range:", data1[0][0],data1[-1][0])
print('Time range:', Time[0],Time[-1])
print("Number of breath:",data1[-1][-4],data2[-1][0],Breath1[-1],Breath2[-1])

print('Quantities of Interest:',lab1[6],lab2[5],"and",lab2[2])

breath_cycle_length=60./respiratory_rate
respiratory_ratio=inspiratory_time/(breath_cycle_length-inspiratory_time)
print("respiratory ratio",respiratory_ratio,'=',Fraction(str(respiratory_ratio)))

#==============================================================================
ntemp=0
tbreath=[]
for i, n in enumerate(Breath1):
    if n != ntemp:
        tbreath.append(Time[i])
        ntemp=n
print('N breath',len(tbreath))
print('Breath cycle length:',tbreath[-2]-tbreath[-3], breath_cycle_length)
#==============================================================================

#==============================================================================
# TIME offsets

#maxt_sim=np.argmax(Paw2)
maxt_sim=-1
for i, p in enumerate(Paw2):
    if p > 0.:
        maxt_sim=i
print(Time[maxt_sim])
#maxt_vent=np.argmax(ps1_p)
maxt_vent=-1
for i, p in enumerate(ps1_p):
    if p > 0.:
        maxt_vent=i
print(ts[maxt_vent])
offset=Time[maxt_sim]-ts[maxt_vent]
offset=24
#offset=22.2
#offset=23.
print('Time offset:',offset, 's')
Time=Time-offset

#==============================================================================

fig=plt.figure(figsize=(13,8))

ax1 = plt.subplot(311)
plt.title('Lungs: C={:1.0f}mL/cmH2O, R={:1.0f}cmH20/L/s  Ventilator: Rate={:1.0f}bpm, PIP={:1.0f}cmH2O Insp={:1.1f}s, PEEP={:1.0f}cmH2O'.format(compliance,resistance,respiratory_rate,inspiratory_pressure,inspiratory_time,PEEP))
plt.plot(Time, Paw2, label='Pressure Sim')
plt.plot(ts,ps1_p,label='PS1')
plt.ylabel('Pressure [cmH2O]')
#y1,y2=plt.ylim()
#plt.vlines(tbreath, y1,y2, label="breath")
plt.setp(ax1.get_xticklabels(), visible=False)
plt.grid(axis='y')
plt.ylim(-0.5,30.)
plt.legend(loc='upper right')
plt.xticks(np.arange(0, Time[-1], step=1.))

ax2 = plt.subplot(312, sharex=ax1)
plt.plot(Time, Vtot1, label='Sim Tidal Vol.')
plt.plot(tsv, tidalV, 'r', label="Calc. Vol")
plt.ylabel('Volume [L]')
#plt.ylim(-0.4,0.75)
#plt.vlines(tbreath, y1,y2, label="breath")
plt.grid(axis='y')
plt.setp(ax2.get_xticklabels(), visible=False)
plt.legend(loc='upper right')

ax3 = plt.subplot(313, sharex=ax1)
plt.plot(Time, Ftot/60.,label="Flow Sim")
plt.plot(ts,fs1_r/60., 'g',label='SP1')
plt.ylabel('Flow Rate [L/sec]')
#plt.ylim(-1.0,1.0)
#plt.ylim(-0.5,1.5)
plt.ylim(-0.5,1.)
#plt.vlines(tbreath, y1,y2, label="breath")
plt.grid(axis='y')
plt.legend(loc='upper right')

plt.xlabel('sec')
plt.xlim(Time[-1]-20.*breath_cycle_length, Time[-1]-15.*breath_cycle_length)

fig.tight_layout()
plt.show()
fig.savefig('./plots/sim_C{:1.0f}R{:1.0f}_RR{:1.0f}_Pins{:1.0f}_It{:1.1f}_PEEP{:1.0f}.png'.format(compliance,resistance,respiratory_rate,inspiratory_pressure,inspiratory_time,PEEP))
