This repository contains publicly available code for experiments published in academic journals.

The code is written entirely in Python and depends on the following libraries:
- NumPy http://www.numpy.org/
- SciPy http://www.scipy.org/
- Matplotlib http://matplotlib.org/
- Pandas http://pandas.pydata.org/
- Brian Simulator http://www.briansimulator.org/
- Plot.ly python bindings: http://plot.ly

A number of the experiments are carried out in iPython Notebooks http://ipython.org/ and import "pylab" to create a
Matlab like environment by importing from Numpy, Matplotlib etc.  See this discussion for more on pylab
http://stackoverflow.com/questions/12987624/confusion-between-numpy-scipy-matplotlib-and-pylab

It is highly recommended that you install the Anaconda Python distribution from continuum analytics
for scientific computing which packages with most of the aforementioned libraries. http://www.continuum.io/.  Academic
licenses are free.

To install brian simulator, run "pip install brian" from the command line on a Unix-based system if you have pip
installed.

run "ipython notebook --pylab=inline" from the command line in the '/MLI_PKJ_net/notebooks/' directory to start iPython
notebook.

The experiments make heavy use of the Brian Simulator, a framework for simulating spiking neuron models.  You should be
acquainted with it in order to understand the code here.
