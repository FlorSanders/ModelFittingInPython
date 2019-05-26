# First order and oscillating model fitting

This is some code I've written for a university course called "Designing and adjusting of dynamic systems".
This code can fit data from measurements of the systems step response saved in a CSV with `;` as delimiter and `,` as comma sign, with the structure `time stamp;data`, though this can obviously be edited in the code to a model.

The models which can be fitted are:
A first order model with amplitude A, time constant tau and delay L.
An oscillating second order model with amplitude A, damping ratio zeta and natural frequency omega_n.

Further, the program will produce a plot of the measurement along with the step response of the fitted model, while printing a whole bunch of information about the measurement  in the terminal.

Finally, there's a program which simply analyses basic statistics about the measurement, like rising time, overshoot, amplitude and others.

I tried automating as much of the process as possible, which resulted in way less time and effort spent going through data and trying to read of everything myself.
Also: I've added two example files of measurement data for both the first order and oscillating model to test the program for yourself before applying it to your own data.