#####################################
# Written by: Flor Sanders          #
# For: Vaardigheidstest MoReDyS     #
# Version: 1.0                      #
# Last edited: 18/05/2019           #
#####################################

# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Defining the step response of the model as a function
def modelstep(t, K, L, tau):
    return K * (1 - np.exp(-(t-L)/tau))

# Defining the parameters for the program
# Giving the correct file path, to obtain the measurement data
path = "data.csv"
# The minimum voltage for which we will consider the measurement started
Vmin = 0.005
# Battery voltage
Vbat = 1.2718624179251492

# Using pandas to read the csv data in as a matrix
data = pd.read_csv(path,sep=';',decimal=',').values

# Obtaining the end voltage V0 of the system
Vinf = data[-1][1]
print("Vinf = %.3f V" %(Vinf))

# Getting the time steps used for the measurement
dt = data[1][0] - data[0][0]

# We need to find the time for which the measurement starts (When the button has been pressed)
t0 = (data[:, 0][np.abs(data[:, 1]) > Vmin][0]) - dt
print("Start time: %.2f s" %(t0))

# We need to find the point of maximum ascent
m = 0
for i,point in enumerate(data):
    if i >= 6 and i <= len(data) - 7:
        #ascent = np.average(data[i-5:i+5, 1] - data[i-6:i+4, 1])/dt
        ascent = (data[i+1, 1] - data[i, 1])/dt
        if ascent > m:
            m = ascent
            tm = point[0]
            Vm = point[1]
            im = i
print("Maximum rise: %.3f V/s at time %.2f s" %(m, tm))
print("Amplitude at the maximum rise: %.3f" %(Vm))

# Calculating parameters for the model
K = Vinf/Vbat
ta = tm - Vm/m
tc = tm + (Vinf-Vm)/m
tau = tc-ta
L = ta-t0
print("The parameters are: \nK = %.3f\nL = %.3f s\ntau = %.3f s" %(K, L, tau))

# Calculating points to plot steepest tangent
tangent = np.matrix([[ta, 0], [tm, Vm], [tc, Vinf]])

# Calculating points to plot the model
t = np.arange(ta, data[-1, 0], dt)
stepresponse = np.matrix([t, modelstep(t-t0, Vinf, L, tau)]).transpose()

# Design rules Ziegler-Nichols - Linear
R = K/tau
print("Ziegler-Nichols: LINEAR")
print("Regelaar\tKc\t\tTi [s]\t\tTd [s]")
print("P\t\t%.3f" %(1/(R*L)))
print("PI\t\t%.3f\t\t%.3f" %(0.9/(R*L), L/0.3))
print("PID\t\t%.3f\t\t%.3f\t\t%.3f" %(1.2/(R*L), 2*L, 0.5*L))

# Design rules Ziegler-Nichols - oscillatie (1e orde padebenadering)
Ku = (L/2 + tau)*2/(K*L)
omegau = np.sqrt((K*Ku+1)/(tau*L/2))
Tu = 2*np.pi/omegau
print("Ziegler-Nichols: OSCILLATION (1st ORDER PADE)")
print("Regelaar\tKc\t\tTi [s]\t\tTd [s]")
print("P\t\t%.3f" %(0.5*Ku))
print("PI\t\t%.3f\t\t%.3f" %(0.45*Ku, Tu/1.2))
print("PID\t\t%.3f\t\t%.3f\t\t%.3f" %(0.6*Ku, 0.5*Tu, Tu/8))

# Plotting everything!
fig, ax = plt.subplots()

# Data (Measurement)
datatime = data[:,0].tolist()
datadata = data[:,1].tolist()
ax.plot(datatime,datadata, label = "Measurement")

# Tangent line
tangenttime = tangent[:,0].tolist()
tangentdata = tangent[:,1].tolist()
ax.plot(tangenttime, tangentdata, label = "Tangent line")

# Model
modeltime = stepresponse[:,0].tolist()
modeldata = stepresponse[:,1].tolist()
ax.plot(modeltime, modeldata, label = "Model step response")

# Labels
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("System step response")
legend = ax.legend(loc = 'lower right')

plt.show()