from __future__ import print_function
import datetime
import os
from os import chdir
import brian
from brian import *

import copy as COPY
import sys
sys.path.append('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/model_PKJ_MLI - Lennon et al., computneur2014')
from MLI_PKJ_net import *
sys.path.append('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project')
from MLI_net_personal import *
sys.path.append('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/MLI_net_personal')
from molecular_layer_geometric import *
from sampling_samplingAnalysis import *
from sampling_samplingAnalysis import Nsample_sampling__Ntriads_patchings__triadsAnalysis

import cPickle
import time
set_global_preferences(useweave=True, usenewpropagate=True, usecodegen=True, usecodegenweave=True)
defaultclock.dt = .1*ms # not .25*ms
from statsmodels.tsa.stattools import acf
from pandas import *
DEBUG=True
''' This file helped me to solve the mystery of running several simulations in one single script in Brian.
    - Problem of the stimuli : if you don't set them at zero during the second run thanks to a boolean statement, they may mess your network up
    - Problem of the synapses which cannot be modified once they have been run : use the python module copy
'''

######################### MAKE THE NETWORK NEURONS ###################################


N_MLI_geometric_OX=16 # MLItable "columns" : len(MLItable[0])
N_MLI_geometric_OY=10 # MLItable "raws" : len(MLItable)
N_MLI = N_MLI_geometric_OX*N_MLI_geometric_OY
MLI = Geometric_MLI(N_MLI)
MLItable_output = create_MLItable(MLI, N_MLI_geometric_OX, N_MLI_geometric_OY)
MLItable = MLItable_output[0]
MLI = MLItable_output[1]


########################### CONNECT THE NETWORK #################################

# synaptic weights
w_mli_mli = 1.


# Synapses
S_MLI_MLI = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')

# Connections
connect_mli_mli_personal(MLItable, S_MLI_MLI, syn_prob=0.19) # according to experimental prob : 0.2
S_MLI_MLI.w[:,:] = 'rand()*w_mli_mli'

# Copy of the Synapses : S is to be run immediatly, S_MLI_MLI to be modified then run
S = COPY.copy(S_MLI_MLI)

print('Convergences')
# print 'MLI->PKJ convergence: ', len(S_MLI_PKJ)/float(N_PKJ)
print ('MLI->MLI convergence: ', len(S_MLI_MLI)/float(N_MLI))
# print 'PKJ->MLI-BS convergence: ', len(S_PKJ_MLI)/(float(N_MLI))
print ('\nDivergences')
# print 'MLI->PKJ divergence: ', len(S_MLI_PKJ)/float(N_MLI)
print ('MLI->MLI divergence: ', len(S_MLI_MLI)/float(N_MLI))
# print 'PKJ->MLI-BS divergence: ', len(S_PKJ_MLI)/float(N_PKJ)

##########################  RUNNING caracteristics   ##################################

# Monitors
if DEBUG==True :
    MS_MLI = SpikeMonitor(MLI[0:3])
    MR_MLI = PopulationRateMonitor(MLI[0:3],bin=1*ms)
    MISI_MLI = ISIHistogramMonitor(MLI[0:3],bins=arange(0,162,2)*ms)
    MV_MLI = StateMonitor(MLI[0:3], 'V', record=range(len(MLI[0:3])))
else :
    MS_MLI = SpikeMonitor(MLI)
    MR_MLI = PopulationRateMonitor(MLI,bin=1*ms)
    MISI_MLI = ISIHistogramMonitor(MLI,bins=arange(0,162,2)*ms)
    MV_MLI = StateMonitor(MLI, 'V', record=range(len(MLI)))
# Runtime
runtime = 2*second

# Deterministic input
EXC_INPUT=False
START = 0*second
EXC_PARAMS = [0.06, 0.5, 0.005*second] # in nA, (period) seconds, seconds
@network_operation(Clock(dt=defaultclock.dt))#Decorator to make a function into a NetworkOperation = a callable class which is called every time step by the Network run method
def exc_input():
    global EXC_INPUT
    global DEBUG
    global START
    global EXC_PARAMS
    EXC_AMP, EXC_DEL, EXC_DUR = EXC_PARAMS[0], EXC_PARAMS[1], EXC_PARAMS[2]
    if EXC_INPUT==True :
        # current caracs : amp, del, dur
        #Ione = 0.06 # amplitude in nA amp
        #Itwo = 0 # amplitude in nA amp
        #Ithree = 0
        Iarray=array([EXC_AMP for i in range(len(MLI))]) # in nA
        MLI.Istim = zeros(len(MLI))*nA # default : no current
        # function
        if (float(defaultclock.t)*10000)%(EXC_DEL*10000)==0: # start of the stim every EXC_DEL*second. The multiplication by 10000 is made to avoid modulo python aberrations.
            START = defaultclock.t
            if DEBUG==True:
                print('\n\n START updated', START, end=' - ')
                print('START+EXC_DUR :',START+EXC_DUR, end = '\n\n')
        if defaultclock.t<=START+EXC_DUR : # duration of the stim : 5ms
            #MLI.Istim = ((zeros(len(MLI))+1)*array([Ione,Itwo])) *nA # amp of the stim in nA
            MLI.Istim = Iarray*nA
            if DEBUG==True:
                print(Iarray[0], 'nA at time ', defaultclock.t, ' in ', len(MLI),' MLIs.', end=' - ')
    elif EXC_INPUT==False:
            MLI.Istim = zeros(len(MLI))*nA # reset : no current

# Noisy Lennon input
RANDOM_CURRENT=False
@network_operation(Clock(dt=defaultclock.dt)) #Decorator to make a function into a NetworkOperation = a callable class which is called every time step by the Network run method
def random_current():
    global RANDOM_CURRENT
    global DEBUG
    if RANDOM_CURRENT==True:
        MLI.I = gamma(3.966333,0.006653,size=len(MLI)) * nA
    elif RANDOM_CURRENT==False:
        MLI.I = zeros(len(MLI))*nA

# Debug function
@network_operation(Clock(dt=defaultclock.dt))
def debug():
    if DEBUG==True and (float(defaultclock.t)*10000)%(0.05*10000)==0:# every 50ms, 10Hz
        print('MLI[0:3].I = ', MLI[0:3].I)
        print('MLI[0:3].Istim = ', MLI[0:3].Istim)


# networks
network_genuine = Network(MLI, S, MS_MLI, MR_MLI, MISI_MLI, MV_MLI, exc_input, random_current, debug)
network_TOPALG = Network(MLI, S_MLI_MLI, MS_MLI, MR_MLI, MISI_MLI, MV_MLI, exc_input, random_current, debug)

##########################  RUN - genuine connectivity, nostim   ##################################

# Which stimuli
EXC_INPUT=False
RANDOM_CURRENT=True

# Run
network_genuine.run(runtime)

# Plot
if DEBUG==True :
    MV_MLI.insert_spikes(MS_MLI)
    ax = MV_MLI.plot()
    show(ax)
rp = raster_plot(MS_MLI)
show(rp)

##########################  RUN - genuine connectivity, stim   ##################################

# restore the network's monitors and clock + the neurongroup without the synapse
network_genuine.reinit(states=False)
START = 0*second # not needed, just in case

# Which stimuli
EXC_INPUT, EXC_PARAMS[0] = True, 0.6 #Higher current amp, in nA
RANDOM_CURRENT = True

# Run
network_genuine.run(runtime)

# Plot
if DEBUG==True :
    MV_MLI.insert_spikes(MS_MLI)
    ax = MV_MLI.plot()
    show(ax)
rps = raster_plot(MS_MLI)
show(rps)

### S_MLI_MLI modification : since S is the one which has been ran, it can still be mofdified -- mimics TOPALG ###
S_MLI_MLI[0,0]=True
S_MLI_MLI.w[0,0]=1

##########################  RUN - TOPALG connectivity, nostim   ##################################

# restore the network's monitors and clock
network_genuine.reinit(states=False) # reinit genuine's monitors and clock since they are the same as TOPALG's
START = 0*second # not needed, just in case

# Which stimuli
EXC_INPUT=False
RANDOM_CURRENT=True

# Run
network_TOPALG.run(runtime)

# Plot
if DEBUG==True :
    MV_MLI.insert_spikes(MS_MLI)
    ax = MV_MLI.plot()
    show(ax)
rp = raster_plot(MS_MLI)
show(rp)

##########################  RUN - TOPALG connectivity, stim   ##################################

# restore the network's monitors and clock
network_TOPALG.reinit(states=False)
START = 0*second # global variable of the stim which has to be reset (was unset in genuine connectivity - stim)

# Which stimuli
EXC_INPUT, EXC_PARAMS[0] = True, 0.6 # Higher current amp, in nA
RANDOM_CURRENT = True

# Run
network_TOPALG.run(runtime)

# Plot
if DEBUG==True :
    MV_MLI.insert_spikes(MS_MLI)
    ax = MV_MLI.plot()
    show(ax)
rps = raster_plot(MS_MLI)
show(rps)