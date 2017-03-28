
from __future__ import print_function
import brian
import pylab
from os import chdir
chdir('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project')
from MLI_net_personal import *
chdir('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/MLI_net_personal')
from molecular_layer_geometric import *
from sampling_samplingAnalysis import *
from sampling_samplingAnalysis import Nsample_sampling__Ntriads_patchings__triadsAnalysis

'''This file contains 
    - a MLIgroup group of 2 MLIs
    - a Synapse object S_MLI_MLI connecting the 0 to the 1 with a weight of 1
    - 2 run profiles : one without any stimulus -- one with a 0.055-0.06nA current injected in the neuron 1
    '''
defaultclock.dt = .25*ms

#### create the objects : neurongroup, synapses
MLI = Geometric_MLI(2)
w_mli_mli = 1.
S_MLI_MLI = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')
S_MLI_MLI[0,1] = True
S_MLI_MLI.w[0,1] =  1

#### monitors and network : prepare the objects for the simulations run
# monitors
MS_MLI = SpikeMonitor(MLI)
MV_MLI = StateMonitor(MLI, 'V', record=range(len(MLI)))

# network
network = Network(MLI,
                  S_MLI_MLI,
                  MS_MLI,
                  MV_MLI
                  )
#### Run caracteristics ; stim def
runtime = 2*second
START = 0*second
def exc_input():
    global START
    # current caracs : amp, del, dur
    #Ione = 0.06 # amplitude in nA amp
    #Itwo = 0 # amplitude in nA amp
    #Ithree = 0
    Iarray=array([0.06 for i in range(len(MLI))])
    period = 0.5 # 'multiple delays' in seconds del
    duration = 0.005*second # duration dur
    # function
    MLI.Istim = zeros(len(MLI))*nA # init : no current
    if float(defaultclock.t)%period==0: # start of the stim every period*second (NB: first value of defaultclock.t=defaultclock,dt, not 0)
        START = defaultclock.t
        print('\n\n START updated', START, end=' - ')
        print('START+duration :',START+duration, end = '\n\n')
    if defaultclock.t<=START+duration : # duration of the stim : 5ms
        #MLI.Istim = ((zeros(len(MLI))+1)*array([Ione,Itwo])) *nA # amp of the stim in nA
        MLI.Istim = Iarray*nA
        print('1nA at time :', defaultclock.t, end=' - ')
Stim = NetworkOperation(exc_input, Clock(dt=defaultclock.dt))



#### Run without stim + plotting
network.run(runtime)

print('spikes when no stim :', MS_MLI.nspikes)
MV_MLI.insert_spikes(MS_MLI) # Ils rajoutent des spikes les tricheurs !! Sur la raw data, une spike ne se voit qu'au fait que Vm retourne soudainement à Vreset
ax = MV_MLI.plot()
show(ax)
rp = raster_plot(MS_MLI)
show(rp)

#### Run with stim + plotting

# restore the network as before the first run ; add the stimulus
network.reinit(states=False) # doesn't reinitialise Synapses and NeuronGroup caracs, only the monitors/clocks
network.add(Stim)

# Run ; plot
network.run(runtime)

print('spikes when stim :', MS_MLI.nspikes)
MV_MLI.insert_spikes(MS_MLI) # Ils rajoutent des spikes les tricheurs !! Sur la raw data, une spike ne se voit qu'au fait que Vm retourne soudainement à Vreset
ax = MV_MLI.plot()
show(ax)
rp = raster_plot(MS_MLI)
show(rp)
