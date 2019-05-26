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
def modelstep(t, K, omegan, zeta):
    omegap = omegan*np.sqrt(1-zeta**2)
    return K*(1 - np.exp(-zeta*omegan*t)*(np.cos(omegap*t) + zeta*omegan/omegap*np.sin(omegap*t)))

# Defining the parameters for the program
# Giving the correct file path, to obtain the measurement data
path = "data.csv"
# The minimum voltage for which we will consider the measurement started
Vmin = 0.05
# The amount of elemets looked back and ahead to consider the maximum
averaging = 0
#Battery voltage
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
print("Starting time of the measurement: %.2f s" %(t0))

# We need to find the absolute maximum in the voltage measurement data
datamax = np.array([i for i in data[:, 1][:-1-averaging]])
for i in range(1,averaging+1):
    datamax += data[:,1][i:-1-averaging+i]
imax = np.argmax(datamax) + averaging + 1
Vmax = data[imax, 1]
tmax = data[imax, 0]
print(imax)
print(Vmax)

# We also need the maximum following the previous one
# Continuing to the first deepest minimum from where we can find the next maximum
imin = np.argmin(datamax[imax:]) + imax
print(imin)
# Getting the next maximum
imax2 = np.argmax(datamax[imin:-1]) + imin + averaging + 1
Vmax2 = data[imax2, 1]
tmax2 = data[imax2, 0]
print(Vmax2)

# Calculating the parameters for the model
ld = np.log((Vmax-Vinf)/(Vmax2-Vinf))
zeta = np.abs(ld) / np.sqrt((2*np.pi)**2 + ld**2)
Tp = tmax2 - tmax
omegap = 2*np.pi/Tp
omegan = omegap/np.sqrt(1-zeta**2)
K = Vinf/Vbat
print("The parameters are: \nK = %.3f\nomega_n = %.3f rad/s\nzeta = %.3f" %(K, omegan, zeta))
print(-zeta*omegan)
print(omegan*np.sqrt(1-zeta**2))

# Calculating points to plot the model
t = np.arange(t0, data[-1, 0], dt)
stepresponse = np.matrix([t, modelstep(t-t0, Vinf, omegan, zeta)]).transpose()

# Plotting everything!
fig, ax = plt.subplots()

# Data (Measurement)
datatime = data[:,0].tolist()
datadata = data[:,1].tolist()
ax.plot(datatime,datadata, label = "Measurement")

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