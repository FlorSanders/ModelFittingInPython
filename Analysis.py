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

# Defining the parameters for the program
# Giving the correct file path, to obtain the measurement data
path = "data.csv"
# The minimum voltage for which we will consider the measurement started
Vmin = 0.03
# Battery voltage
Vbat = 1.2718624179251492
# The amount of elemets looked back and ahead to consider the maximum
averaging = 5

# Using pandas to read the csv data in as a matrix
data = pd.read_csv(path,sep=';',decimal=',').values

# Obtaining the end voltage V0 of the system
Vinf = data[-1][1]
print("Vinf = %.3f V" %(Vinf))

# Getting the time steps used for the measurement
dt = data[1][0] - data[0][0]

# We need to find the time for which the measurement starts (When the button has been pressed)
t0 = (data[:, 0][np.abs(data[:, 1]) > Vmin][0]) - dt
print("Start time of the measurement: %.2f s" %(t0))

# Get the simple time parameters
tau = data[:, 0][data[:,1] > (1-np.exp(-1))*Vinf][0] - t0
Tr = data[:, 0][data[:,1] > 0.9*Vinf][0] - data[:, 0][data[:,1] > 0.1*Vinf][0]
K = Vinf/Vbat
print("tau (1e orde): %.3f s" %tau)
print("Rising time: %.3f s" %Tr)
print("Amplitude: %.3f" %K)

# Getting the max voltage as wel as the overshoot
datamax = np.array([i for i in data[:, 1][:-1-averaging]])
for i in range(1,averaging+1):
    datamax += data[:,1][i:-1-averaging+i]
imax = np.argmax(datamax) + averaging + 1
Vmax = data[imax, 1]
tmax = data[imax, 0]
print("Overshoot: %.3f" %((Vmax-Vinf)/Vinf))

# Getting the remaining error in regime
print("Remaining error = %.3f V = %.3f percent" %(Vbat-Vinf, (Vbat-Vinf)/Vinf*100))

# Getting the 5 % off time from the measurement
for i, point in enumerate(data):
    if(np.abs(point[1]-Vinf)/Vinf > 0.05):
        t5p = point[0] - t0
t5p = t5p + dt
print("The 5 percent off time of the system is %.3f s" %(t5p))


# Plotting everything!
fig, ax = plt.subplots()

# Data (Measurement)
datatime = data[:,0].tolist()
datadata = data[:,1].tolist()
ax.plot(datatime,datadata, label = "Measurement")

# Labels
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Systems step response")
legend = ax.legend(loc = 'lower right')

plt.show()