This code allows to generate a model network of 160 Molecular Layer Interneurons inspired from Lennon et al., and implemented in the BRIAN simulator (Stimberg et al., 2014), and to study the effects of its local topology on its dynamics through the use of TOPALG.

TOPALG is an algorithm implemented in python 2.7, whose function is to push a neural networks triadic connectivity to converge towards a chosen distribution. It is designed to work with networks implemented in BRIAN. It has been written in 2016 by Maxime Beau and Arnd Roth.This repository contains publicly available code for experiments published in academic journals.

The code is written entirely in Python and depends on the following libraries:
- NumPy http://www.numpy.org/
- SciPy http://www.scipy.org/
- Matplotlib http://matplotlib.org/
- Pandas http://pandas.pydata.org/
- Brian Simulator http://www.briansimulator.org/
- Plot.ly python bindings: http://plot.ly
- NetworkX https://networkx.github.io/


Description of the directories:
- MLI_net_personnal: contains the description of the model inspired from Lennon et al., 2014, as well as TOPALG.
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
