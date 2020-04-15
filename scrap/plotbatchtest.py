import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction


compliance=10 # mL/cmH2O
resistance=50 # cmH20/L/s 
respiratory_rate=20 # bpm
inspiratory_pressure=15 # cmH2O
inspiratory_time=1.5 # s
PEEP=5 # cmH2O

breath_cycle_length=60./respiratory_rate
respiratory_ratio=inspiratory_time/(breath_cycle_length-inspiratory_time)
print("respiratory ratio",Fraction(str(respiratory_ratio)))

#passive_C20R6_RR15_Pins30_It1.5_PEEP7.rwa
#fname1="./rawdata/passive_C{:1.0f}R{:1.0f}_RR{:1.0f}_Pins{:1.0f}_It{:1.1f}_PEEP{:1.0f}.rwa".format(compliance,resistance,respiratory_rate,inspiratory_pressure,inspiratory_time,PEEP)
fname1="./1000breathC10R50/1000breathC10R50.rwa"
print("Reading from:",fname1)
data1=np.loadtxt(fname1,skiprows=4)

lab1=['Time (sec)',	'Airway Pressure (cmH2O)',	'Muscle Pressure (cmH2O)',	'Tracheal Pressure (cmH2O)',	'Chamber 1 Volume (L)',	'Chamber 2 Volume (L)',	'Total Volume (L)',	'Chamber 1 Pressure (cmH2O)',	'Chamber 2 Pressure (cmH2O)',	'Breath File Number (#)',	'Aux 1 (V)',	'Aux 2 (V)',	'Oxygen Sensor (V)']


#fname2="./rawdata/passive_C{:1.0f}R{:1.0f}_RR{:1.0f}_Pins{:1.0f}_It{:1.1f}_PEEP{:1.0f}.dta".format(compliance,resistance,respiratory_rate,inspiratory_pressure,inspiratory_time,PEEP)
fname2="./1000breathC10R50/1000breathC10R50.dta"
print("Reading from:",fname2)
data2=np.loadtxt(fname2,skiprows=4)

lab2=['Breath Number (#)','Compressed Volume (mL)','Airway Pressure (cmH2O)','Muscle Pressure (cmH2O)','Total Volume (mL)', 'Total Flow (L/min)','Chamber 1 Pressure (cmH2O)', 'Chamber 2 Pressure (cmH2O)', 'Chamber 1 Volume (mL)','Chamber 2 Volume (mL)','Chamber 1 Flow (L/min)','Chamber 2 Flow (L/min)','Tracheal Pressure (cmH2O)','Ventilator Volume (mL)','Ventilator Flow (L/min)','Ventilator Pressure (cmH2O)']

print("Number of entries:",len(data1),len(data2))

Time=data1[:,0]
Breath1=data1[:,-4]
Breath2=data2[:,0]

Paw1=data1[:,1]
Paw2=data2[:,2]
Pvent=data2[:,-1]

Vtot1=data1[:,6]
Vtot2=data2[:,4]
Vcomp=data2[:,1]
Vvent=data2[:,-3]

Ftot=data2[:,5]
Fvent=data2[:,-2]

print("Time range:", data1[0][0],data1[-1][0])
print('Time range:', Time[0],Time[-1])
print("Number of breath:",data1[-1][-4],data2[-1][0],Breath1[-1],Breath2[-1])

fig=plt.figure(figsize=(13,8))

ax1 = plt.subplot(421)
#plt.title("airway pressure")
plt.plot(Time, Paw1, label='Airway Pressure 1')
plt.plot(Time, Paw2, label='Airway Pressure 2')
plt.plot(Time, Pvent, label='Ventilator Pressure')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('cmH20')
plt.legend()


ax2 = plt.subplot(423, sharex=ax1)
plt.plot(Time, Vtot1*1.e3, label='Total Volume 1')
plt.plot(Time, Vtot2, label='Total Volume 2')
#plt.plot(Time, Vcomp, label='Compressed Volume')
plt.plot(Time, Vvent, label='Ventilator Volume')
plt.plot(Time, Vtot2-Vvent, label='Volume Difference')
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('mL')
plt.legend()


ax3 = plt.subplot(425, sharex=ax1)
plt.plot(Time,Ftot,label="Total Flow")
plt.plot(Time,Fvent,label="Ventilator Flow")
plt.ylabel('L/min')
plt.setp(ax3.get_xticklabels(), visible=False)
plt.ylim(-150.,150.)
plt.legend()


ax4 = plt.subplot(427, sharex=ax1)
plt.plot(Time, Breath1, label='1')
plt.plot(Time, Breath2+1, label='2+1')
#plt.setp(ax4.get_xticklabels(), visible=False)
plt.legend()


plt.xlabel('sec')
#plt.xlim(Time[-1]-10.*breath_cycle_length, Time[-1])


P1=data2[:,6]
P2=data2[:,7]
Pmusc=data2[:,3]
Ptrach=data2[:,-4]

ax12 = plt.subplot(422, sharex=ax1, sharey=ax1)
plt.plot(Time,Paw2, label='Airways 2')
plt.plot(Time,P1,label="P1")
plt.plot(Time,P2,label="P2")
plt.plot(Time,Pmusc,label="Muscle")
plt.plot(Time,Ptrach,label="Tracheal")

plt.setp(ax12.get_xticklabels(), visible=False)
plt.setp(ax12.get_yticklabels(), visible=False)
plt.legend()


V1=data2[:,8]
V2=data2[:,9]

ax22 = plt.subplot(424, sharex=ax1, sharey=ax2)
plt.plot(Time, Vtot2, label='Total Volume 2')
plt.plot(Time,V1,label="V1")
plt.plot(Time,V2,label="V2")
plt.setp(ax22.get_xticklabels(), visible=False)
plt.setp(ax22.get_yticklabels(), visible=False)
plt.legend()


F1=data2[:,10]
F2=data2[:,11]

ax32 = plt.subplot(426, sharex=ax1, sharey=ax3)
plt.plot(Time, Ftot, label='Total Flow')
plt.plot(Time,F1,label="F1")
plt.plot(Time,F2,label="F2")
plt.setp(ax22.get_xticklabels(), visible=False)
plt.setp(ax22.get_yticklabels(), visible=False)
plt.legend()


ntemp=0
tbreath=[]
for i, n in enumerate(Breath2):
    if n != ntemp:
        tbreath.append(Time[i])
        ntemp=n
        
print('N breath',len(tbreath))

ax42 = plt.subplot(428, sharex=ax1)
plt.vlines(tbreath, 0., 1.)
plt.setp(ax42.get_yticklabels(), visible=False)
plt.xlabel('sec')

xmin=Time[-1]-15.*breath_cycle_length
xmax=Time[-1]-5.*breath_cycle_length
print("limits=[",xmin,xmax,"]s")
plt.xlim(xmin, xmax)

fig.tight_layout()

plt.show()