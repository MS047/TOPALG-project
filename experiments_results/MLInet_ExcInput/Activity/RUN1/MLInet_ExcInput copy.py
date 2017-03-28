from __future__ import print_function
import datetime
import os
from os import chdir
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

FUNCTION = 'untransitivity'

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
MS_MLI = SpikeMonitor(MLI)
MR_MLI = PopulationRateMonitor(MLI,bin=1*ms)
MISI_MLI = ISIHistogramMonitor(MLI,bins=arange(0,162,2)*ms)
MV_MLI = StateMonitor(MLI, 'V', record=range(N_MLI))

# Runtime
runtime = 300*second

# Deterministic input
EXC_INPUT=False
START = 0*second
EXC_PARAMS = [0.06, 1, 0.01*second] # ampin nA, period ("delay") in seconds, duration in seconds
@network_operation(Clock(dt=defaultclock.dt))#Decorator to make a function into a NetworkOperation = a callable class which is called every time step by the Network run method
def exc_input():
    global EXC_INPUT
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
        if defaultclock.t<=START+EXC_DUR : # duration of the stim : 5ms
            #MLI.Istim = ((zeros(len(MLI))+1)*array([Ione,Itwo])) *nA # amp of the stim in nA
            MLI.Istim = Iarray*nA
    elif EXC_INPUT==False:
        MLI.Istim = zeros(len(MLI))*nA # reset : no current

# Noisy Lennon input
RANDOM_CURRENT=False
@network_operation(Clock(dt=defaultclock.dt)) #Decorator to make a function into a NetworkOperation = a callable class which is called every time step by the Network run method
def random_current():
    global RANDOM_CURRENT
    if RANDOM_CURRENT==True:
        MLI.I = gamma(3.966333,0.006653,size=len(MLI)) * nA
    elif RANDOM_CURRENT==False:
        MLI.I = zeros(len(MLI))*nA

# networks
network_genuine = Network(MLI, S, MS_MLI, MR_MLI, MISI_MLI, MV_MLI, exc_input, random_current)
network_TOPALG = Network(MLI, S_MLI_MLI, MS_MLI, MR_MLI, MISI_MLI, MV_MLI, exc_input, random_current)

##########################  RUN - genuine connectivity, nostim   ##################################

# Which stimuli
EXC_INPUT=False
RANDOM_CURRENT=True

# Run
TIME_BEFORE_RUN = time.time()
network_genuine.run(runtime)
print(time.time()-TIME_BEFORE_RUN, ' seconds of run - realtime.')
##########################  PLOT ACTIVITY - genuine connectivity, nostim ##################################

# Plot raster plot
rp = raster_plot(MS_MLI, newfigure=True)
rp.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/GenuinenoStim/rasterplot_genuine-connectivity_nostim.png',dpi=600*6.93/25)

ind, mean_fr, isi_cv, err = find_closest_match_neuron(MS_MLI, 15., .40)

# Plot ISI
fig = figure(figsize=(25,5))
ax = fig.add_subplot(131)
MV_MLI.insert_spikes(MS_MLI)
ax.plot(MV_MLI.times[:6000],(MV_MLI.values[ind,:6000]), color='#8C2318')
xlim([-.1,1.5])
add_scalebar(ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax = fig.add_subplot(132)
plot_neuron_ISI_histogram(MS_MLI, ind, ax, xytext=(100,400),nbins=80, color='#8C2318', edgecolor='w')
xlim([0,200])
simpleaxis(ax)
tick_params(labelsize=18)
xlabel('ISI (ms)', fontsize=20)
ylabel('Count', fontsize=20)

# Plot spike autocorrelation
ax = fig.add_subplot(133)
plot_spike_correlogram(MS_MLI.spiketimes[ind],MS_MLI.spiketimes[ind], bin=1*ms, width=200*ms,ax=ax, color='#8C2318')
simpleaxis(ax)
tick_params(labelsize=18)

tight_layout()
fig.subplots_adjust(wspace=.3)
fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/GenuinenoStim/autocorrelation_genuine-connectivity_nostim.png',dpi=600*6.93/25)

# Plotting functions/imports
from scipy.stats import spearmanr

def population_fr_stats(spike_monitor):
    mean_frs = []
    for ind in range(len(spike_monitor.spiketimes)):
        if mean(diff(spike_monitor.spiketimes[ind]))**-1 != nan: #????
            mean_frs.append(mean(diff(spike_monitor.spiketimes[ind]))**-1)
    return mean(mean_frs), std(mean_frs), max(mean_frs), min(mean_frs), mean_frs

def population_isi_cv_stats(spike_monitor):
    cvs = []
    for ind in range(len(spike_monitor.spiketimes)):
        isi_mean, isi_std = isi_mean_and_std(spike_monitor, ind)
        if isi_std/isi_mean != nan: #????
            cvs.append(isi_std/isi_mean)
    return mean(cvs), std(cvs), max(cvs), min(cvs), cvs

m,s,ma,mi,frs = population_fr_stats(MS_MLI)
print('Mean MLI FR: %s, Std: %s, Max: %s, Min: %s'%(m,s,ma,mi))
m,s,ma,mi,cvs = population_isi_cv_stats(MS_MLI)
print('Mean MLI CV: %s, Std: %s, Max: %s, Min: %s'%(m,s,ma,mi))
print("MLI FR-CV correlation.  Spearman's R: %s, p = %s" % spearmanr(frs,cvs))

# plot histogram
try :
    fig = figure(figsize=(6,7))
    
    ax = fig.add_subplot(211)
    ax.hist([mean(diff(MS_MLI.spiketimes[i]))**-1 for i in range(N_MLI)],15,color='#8C2318', edgecolor='w')
    simpleaxis(ax)
    tick_params(labelsize=20)
    xlabel('Mean firing rate (Hz)', fontsize=20)
    ylabel('Number of cells', fontsize=20, labelpad=10)
    title('MLI mean firing rates', fontsize=20)
    yticks(arange(0,19,3))
    
    tight_layout()
    fig.subplots_adjust(hspace=.3)
    fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/GenuinenoStim/population_rate_histograms_genuine-connectivity_nostim.png',dpi=600*3.35/6)
except:
    pass

# plot mean firing rate = f(t)
times = array([])
for i in range(len(MLI)):
    x = np.hstack([times, MS_MLI.spiketimes[i]])
    times = x
times = np.fmod(times, EXC_PARAMS[1]) # stack together all the periods
bin_mfr=0.005 # in seconds. Divide the weights by this to have a result per second = in Hz
fig = figure(figsize=(6,7))
ax = fig.add_subplot(211)
ax.hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
simpleaxis(ax)
tick_params(labelsize=18)
ylabel('Mean firing rate (Hz)', fontsize=18)
xlabel('time (s)', fontsize=18, labelpad=10)
title('MLI mean firing rates over time', fontsize=18)
yticks(arange(0,10,2))

tight_layout()
fig.subplots_adjust(hspace=.3)
fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/GenuinenoStim/mean_firing_rate_genuine-connectivity_nostim.png',dpi=600*3.35/6)
data = hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
trg=open('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/GenuinenoStim/mean_firing_rate_genuine-connectivity_nostim.txt', 'w')
trg.write(str(data))
trg.close()



##########################  RUN - genuine connectivity, stim   ##################################

# restore the network's monitors and clock + the neurongroup without the synapse
network_genuine.reinit(states=False)
START = 0*second # not needed, just in case

# Which stimuli
EXC_INPUT, EXC_PARAMS[0] = True, 0.6 #Higher current amp, in nA
RANDOM_CURRENT = True

# Run
TIME_BEFORE_RUN = time.time()
network_genuine.run(runtime)
print(time.time()-TIME_BEFORE_RUN, ' seconds of run - realtime.')

##########################  PLOT ACTIVITY - genuine connectivity, stim ##################################

# Plot raster plot
rp = raster_plot(MS_MLI, newfigure=True)
rp.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/GenuineStim/rasterplot_genuine-connectivity_stim.png',dpi=600*6.93/25)

ind, mean_fr, isi_cv, err = find_closest_match_neuron(MS_MLI, 15., .40)

ind, mean_fr, isi_cv, err = find_closest_match_neuron(MS_MLI, 15., .40)

# Plot ISI
fig = figure(figsize=(25,5))
ax = fig.add_subplot(131)
MV_MLI.insert_spikes(MS_MLI)
ax.plot(MV_MLI.times[:6000],(MV_MLI.values[ind,:6000]), color='#8C2318')
xlim([-.1,1.5])
add_scalebar(ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax = fig.add_subplot(132)
plot_neuron_ISI_histogram(MS_MLI, ind, ax, xytext=(100,400),nbins=80, color='#8C2318', edgecolor='w')
xlim([0,200])
simpleaxis(ax)
tick_params(labelsize=18)
xlabel('ISI (ms)', fontsize=20)
ylabel('Count', fontsize=20)

# Plot spike autocorrelation
ax = fig.add_subplot(133)
plot_spike_correlogram(MS_MLI.spiketimes[ind],MS_MLI.spiketimes[ind], bin=1*ms, width=200*ms,ax=ax, color='#8C2318')
simpleaxis(ax)
tick_params(labelsize=18)

tight_layout()
fig.subplots_adjust(wspace=.3)
fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/GenuineStim/autocorrelation_genuine-connectivity_stim.png',dpi=600*6.93/25)

# Plotting functions/imports
from scipy.stats import spearmanr

def population_fr_stats(spike_monitor):
    mean_frs = []
    for ind in range(len(spike_monitor.spiketimes)):
        if mean(diff(spike_monitor.spiketimes[ind]))**-1 != nan: #????
            mean_frs.append(mean(diff(spike_monitor.spiketimes[ind]))**-1)
    return mean(mean_frs), std(mean_frs), max(mean_frs), min(mean_frs), mean_frs

def population_isi_cv_stats(spike_monitor):
    cvs = []
    for ind in range(len(spike_monitor.spiketimes)):
        isi_mean, isi_std = isi_mean_and_std(spike_monitor, ind)
        if isi_std/isi_mean != nan: #????
            cvs.append(isi_std/isi_mean)
    return mean(cvs), std(cvs), max(cvs), min(cvs), cvs

m,s,ma,mi,frs = population_fr_stats(MS_MLI)
print('Mean MLI FR: %s, Std: %s, Max: %s, Min: %s'%(m,s,ma,mi))
m,s,ma,mi,cvs = population_isi_cv_stats(MS_MLI)
print('Mean MLI CV: %s, Std: %s, Max: %s, Min: %s'%(m,s,ma,mi))
print("MLI FR-CV correlation.  Spearman's R: %s, p = %s" % spearmanr(frs,cvs))

# plot histogram
try :
    fig = figure(figsize=(6,7))
    
    ax = fig.add_subplot(211)
    ax.hist([mean(diff(MS_MLI.spiketimes[i]))**-1 for i in range(N_MLI)],15,color='#8C2318', edgecolor='w')
    simpleaxis(ax)
    tick_params(labelsize=20)
    xlabel('Mean firing rate (Hz)', fontsize=20)
    ylabel('Number of cells', fontsize=20, labelpad=10)
    title('MLI mean firing rates', fontsize=20)
    yticks(arange(0,19,3))
    
    tight_layout()
    fig.subplots_adjust(hspace=.3)
    fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/GenuineStim/population_rate_histograms_genuine-connectivity_stim.png',dpi=600*3.35/6)
except:
    pass

# plot mean firing rate = f(t)
times = array([])
for i in range(len(MLI)):
    x = np.hstack([times, MS_MLI.spiketimes[i]])
    times = x
times = np.fmod(times, EXC_PARAMS[1]) # stack together all the periods
bin_mfr=0.005 # in seconds. Divide the weights by this to have a result per second = in Hz
fig = figure(figsize=(6,7))
ax = fig.add_subplot(211)
ax.hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
simpleaxis(ax)
tick_params(labelsize=18)
ylabel('Mean firing rate (Hz)', fontsize=18)
xlabel('time (s)', fontsize=18, labelpad=10)
title('MLI mean firing rates over time', fontsize=18)
yticks(arange(0,10,2))

tight_layout()
fig.subplots_adjust(hspace=.3)
fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/GenuineStim/mean_firing_rate_genuine-connectivity_stim.png',dpi=600*3.35/6)
data = hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
trg=open('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/GenuineStim/mean_firing_rate_genuine-connectivity_stim.txt', 'w')
trg.write(str(data))
trg.close()


##################### TOPALG : PRE CORRECTION ANALYSIS - CORRECTION - POST CORRECTION ANALYSIS ##############################

# save the genuine connectivity
syn_dir = '/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Connectivity/Synapses/'
save_synapses(S, 'genuine', syn_dir)

# Choose a functionnally different aimed connection
Analysis_src = Nsample_sampling__Ntriads_patchings__triadsAnalysis(Nsample=1000, Ntriads=1, SynapseObj=S_MLI_MLI, MLItable=MLItable, strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100)
src_dis = Analysis_src[5]

if FUNCTION == 'transitivity':
    # 1) winner takes all between untransitive including loops (030C) and transitive including  feedforward (030T)motifs !
    src_dis['030T']+=src_dis['030C'] # percentage of FF = FF+LP
    src_dis['030C']=0 # no LP left
    src_dis['120D']+=src_dis['120C']
    src_dis['120C']=0
    src_dis['300']+=src_dis['201']
    src_dis['300']+=src_dis['021C']
    src_dis['201']=0
    src_dis['021C']=0
    # 2) Carefully use some of the 012 to generate transitive motifs
    src_dis['030T']+=(src_dis['012']*0.05)*1./3 # three 1 edge motifs turned in one 3 edges motif and two no edges motifs
    src_dis['003']+=(src_dis['012']*0.05)*2./3
    src_dis['012']*=0.95

if FUNCTION == 'untransitivity':
    # 1) winner takes all between untransitive including loops (030C) and transitive including  feedforward (030T)motifs !
    src_dis['030C']+=src_dis['030T'] # percentage of LP = FF+LP
    src_dis['030T']=0 # no FF left
    src_dis['120C']+=src_dis['120D']
    src_dis['120D']=0
    src_dis['201']+=src_dis['300']/2
    src_dis['021C']+=src_dis['300']/2
    src_dis['300']=0
    # 2) Carefully use some of the 012 to generate transitive motifs
    src_dis['030C']+=(src_dis['012']*0.05)*1./3 # three 1 edge motifs turned in one 3 edges motif and two no edges motifs
    src_dis['003']+=(src_dis['012']*0.05)*2./3
    src_dis['012']*=0.95

# Check there is no mistake
sum_src_dis = 0
for val in src_dis.values():
    sum_src_dis+=val
if sum_src_dis<=0.99 or sum_src_dis>=1.01:
    print ('/!\ WARNING DISTRIBUTIONS SUM FAR FROM 1 : ', sum_src_dis)
elif 0.99<=sum_src_dis<=1.01:
    print('OK, distributions sum between 0.99 and 1.01 : ', sum_src_dis)

# Make S_MLI_MLI converge toward this functionnaly different connectivity
iterative_corrections_output = iterative_corrections(Niterations=10, Ncorrection=500, NtriadsCorrection=1, Nanalysis=2000, NtriadsAnalysis=1, SynapseObj_real=S_MLI_MLI,
                                                     MLItable=MLItable,
                                                     TRY=10, prev_aim_diff=0.003,
                                                     strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100,
                                                     Dis003=src_dis['003'], Dis012=src_dis['012'], Dis021C=src_dis['021C'],
                                                     Dis021D=src_dis['021D'],
                                                     Dis021U=src_dis['021U'], Dis030C=src_dis['030C'], Dis030T=src_dis['030T'],
                                                     Dis102=src_dis['102'],
                                                     Dis111D=src_dis['111D'], Dis111U=src_dis['111U'], Dis120C=src_dis['120C'],
                                                     Dis120D=src_dis['120D'],
                                                     Dis120U=src_dis['120U'], Dis201=src_dis['201'], Dis210=src_dis['210'],
                                                     Dis300=src_dis['300'], figsDirectory='/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Connectivity')

S_MLI_MLI = iterative_corrections_output[0]
print("(iteration no (0 before the first correction), meanDiff) : ")
print(iterative_corrections_output[1])

# save the TOPALG connectivity
syn_dir = '/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Connectivity/Synapses/'
save_synapses(S_MLI_MLI, 'TOPALG', syn_dir)

##########################  RUN - TOPALG connectivity, nostim   ##################################

# restore the network's monitors and clock
network_genuine.reinit(states=False) # reinit genuine's monitors and clock since they are the same as TOPALG's
START = 0*second # not needed, just in case

# Which stimuli
EXC_INPUT=False
RANDOM_CURRENT=True

# Run
TIME_BEFORE_RUN = time.time()
network_TOPALG.run(runtime)
print(time.time()-TIME_BEFORE_RUN, ' seconds of run - realtime.')

##########################  PLOT ACTIVITY - TOPALG connectivity, nostim ##################################

# Plot raster plot
rp = raster_plot(MS_MLI, newfigure=True)
rp.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/TOPALGnoStim/rasterplot_TOPALG-connectivity_nostim.png',dpi=600*6.93/25)

ind, mean_fr, isi_cv, err = find_closest_match_neuron(MS_MLI, 15., .40)
ind, mean_fr, isi_cv, err = find_closest_match_neuron(MS_MLI, 15., .40)

# Plot ISI
fig = figure(figsize=(25,5))
ax = fig.add_subplot(131)
MV_MLI.insert_spikes(MS_MLI)
ax.plot(MV_MLI.times[:6000],(MV_MLI.values[ind,:6000]), color='#8C2318')
xlim([-.1,1.5])
add_scalebar(ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax = fig.add_subplot(132)
plot_neuron_ISI_histogram(MS_MLI, ind, ax, xytext=(100,400),nbins=80, color='#8C2318', edgecolor='w')
xlim([0,200])
simpleaxis(ax)
tick_params(labelsize=18)
xlabel('ISI (ms)', fontsize=20)
ylabel('Count', fontsize=20)

# Plot spike autocorrelation
ax = fig.add_subplot(133)
plot_spike_correlogram(MS_MLI.spiketimes[ind],MS_MLI.spiketimes[ind], bin=1*ms, width=200*ms,ax=ax, color='#8C2318')
simpleaxis(ax)
tick_params(labelsize=18)

tight_layout()
fig.subplots_adjust(wspace=.3)
fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/TOPALGnoStim/autocorrelation_TOPALG-connectivity_nostim.png',dpi=600*6.93/25)

# Plotting functions/imports
from scipy.stats import spearmanr

def population_fr_stats(spike_monitor):
    mean_frs = []
    for ind in range(len(spike_monitor.spiketimes)):
        if mean(diff(spike_monitor.spiketimes[ind]))**-1 != nan: #????
            mean_frs.append(mean(diff(spike_monitor.spiketimes[ind]))**-1)
    return mean(mean_frs), std(mean_frs), max(mean_frs), min(mean_frs), mean_frs

def population_isi_cv_stats(spike_monitor):
    cvs = []
    for ind in range(len(spike_monitor.spiketimes)):
        isi_mean, isi_std = isi_mean_and_std(spike_monitor, ind)
        if isi_std/isi_mean != nan: #????
            cvs.append(isi_std/isi_mean)
    return mean(cvs), std(cvs), max(cvs), min(cvs), cvs

m,s,ma,mi,frs = population_fr_stats(MS_MLI)
print('Mean MLI FR: %s, Std: %s, Max: %s, Min: %s'%(m,s,ma,mi))
m,s,ma,mi,cvs = population_isi_cv_stats(MS_MLI)
print('Mean MLI CV: %s, Std: %s, Max: %s, Min: %s'%(m,s,ma,mi))
print("MLI FR-CV correlation.  Spearman's R: %s, p = %s" % spearmanr(frs,cvs))

# plot histogram
try :
    fig = figure(figsize=(6,7))
    
    ax = fig.add_subplot(211)
    ax.hist([mean(diff(MS_MLI.spiketimes[i]))**-1 for i in range(N_MLI)],15,color='#8C2318', edgecolor='w')
    simpleaxis(ax)
    tick_params(labelsize=20)
    xlabel('Mean firing rate (Hz)', fontsize=20)
    ylabel('Number of cells', fontsize=20, labelpad=10)
    title('MLI mean firing rates', fontsize=20)
    yticks(arange(0,19,3))
    
    tight_layout()
    fig.subplots_adjust(hspace=.3)
    fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/TOPALGnoStim/population_rate_histograms_TOPALG-connectivity_nostim.png',dpi=600*3.35/6)
except:
    pass

# plot mean firing rate = f(t)
times = array([])
for i in range(len(MLI)):
    x = np.hstack([times, MS_MLI.spiketimes[i]])
    times = x
times = np.fmod(times, EXC_PARAMS[1]) # stack together all the periods
bin_mfr=0.005 # in seconds. Divide the weights by this to have a result per second = in Hz
fig = figure(figsize=(6,7))
ax = fig.add_subplot(211)
ax.hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
simpleaxis(ax)
tick_params(labelsize=18)
ylabel('Mean firing rate (Hz)', fontsize=18)
xlabel('time (s)', fontsize=18, labelpad=10)
title('MLI mean firing rates over time', fontsize=18)
yticks(arange(0,10,2))

tight_layout()
fig.subplots_adjust(hspace=.3)
fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/TOPALGnoStim/mean_firing_rate_TOPALG-connectivity_nostim.png',dpi=600*3.35/6)
data = hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
trg=open('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/TOPALGnoStim/mean_firing_rate_TOPALG-connectivity_nostim.txt', 'w')
trg.write(str(data))
trg.close()


##########################  RUN - TOPALG connectivity, stim   ##################################

# restore the network's monitors and clock
network_TOPALG.reinit(states=False)
START = 0*second # global variable of the stim which has to be reset (was unset in genuine connectivity - stim)

# Which stimuli
EXC_INPUT, EXC_PARAMS[0] = True, 0.6 # Higher current amp, in nA
RANDOM_CURRENT = True

# Run
TIME_BEFORE_RUN = time.time()
network_TOPALG.run(runtime)
print(time.time()-TIME_BEFORE_RUN, ' seconds of run - realtime.')

##########################  PLOT ACTIVITY - TOPALG connectivity, stim ##################################

# Plot raster plot
rp = raster_plot(MS_MLI, newfigure=True)
rp.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/TOPALGStim/rasterplot_TOPALG-connectivity_stim.png',dpi=600*6.93/25)

ind, mean_fr, isi_cv, err = find_closest_match_neuron(MS_MLI, 15., .40)

ind, mean_fr, isi_cv, err = find_closest_match_neuron(MS_MLI, 15., .40)

# Plot ISI
fig = figure(figsize=(25,5))
ax = fig.add_subplot(131)
MV_MLI.insert_spikes(MS_MLI)
ax.plot(MV_MLI.times[:6000],(MV_MLI.values[ind,:6000]), color='#8C2318')
xlim([-.1,1.5])
add_scalebar(ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax = fig.add_subplot(132)
plot_neuron_ISI_histogram(MS_MLI, ind, ax, xytext=(100,400),nbins=80, color='#8C2318', edgecolor='w')
xlim([0,200])
simpleaxis(ax)
tick_params(labelsize=18)
xlabel('ISI (ms)', fontsize=20)
ylabel('Count', fontsize=20)

# Plot spike autocorrelation
ax = fig.add_subplot(133)
plot_spike_correlogram(MS_MLI.spiketimes[ind],MS_MLI.spiketimes[ind], bin=1*ms, width=200*ms,ax=ax, color='#8C2318')
simpleaxis(ax)
tick_params(labelsize=18)

tight_layout()
fig.subplots_adjust(wspace=.3)
fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/TOPALGStim/autocorrelation_TOPALG-connectivity_stim.png',dpi=600*6.93/25)

# Plotting functions/imports
from scipy.stats import spearmanr

def population_fr_stats(spike_monitor):
    mean_frs = []
    for ind in range(len(spike_monitor.spiketimes)):
        if mean(diff(spike_monitor.spiketimes[ind]))**-1 != nan: #????
            mean_frs.append(mean(diff(spike_monitor.spiketimes[ind]))**-1)
    return mean(mean_frs), std(mean_frs), max(mean_frs), min(mean_frs), mean_frs

def population_isi_cv_stats(spike_monitor):
    cvs = []
    for ind in range(len(spike_monitor.spiketimes)):
        isi_mean, isi_std = isi_mean_and_std(spike_monitor, ind)
        if isi_std != nan and isi_mean != nan : #????
            cvs.append(isi_std/isi_mean)
    return mean(cvs), std(cvs), max(cvs), min(cvs), cvs

m,s,ma,mi,frs = population_fr_stats(MS_MLI)
print('Mean MLI FR: %s, Std: %s, Max: %s, Min: %s'%(m,s,ma,mi))
m,s,ma,mi,cvs = population_isi_cv_stats(MS_MLI)
print('Mean MLI CV: %s, Std: %s, Max: %s, Min: %s'%(m,s,ma,mi))
print("MLI FR-CV correlation.  Spearman's R: %s, p = %s" % spearmanr(frs,cvs))

# plot histogram
try :
    fig = figure(figsize=(6,7))

    ax = fig.add_subplot(211)
    ax.hist([mean(diff(MS_MLI.spiketimes[i]))**-1 for i in range(N_MLI)],15,color='#8C2318', edgecolor='w')
    simpleaxis(ax)
    tick_params(labelsize=20)
    xlabel('Mean firing rate (Hz)', fontsize=20)
    ylabel('Number of cells', fontsize=20, labelpad=10)
    title('MLI mean firing rates', fontsize=20)
    yticks(arange(0,19,3))

    tight_layout()
    fig.subplots_adjust(hspace=.3)
    fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/TOPALGStim/population_rate_histograms_TOPALG-connectivity_stim.png',dpi=600*3.35/6)
except:
    pass
# plot mean firing rate = f(t)
times = array([])
for i in range(len(MLI)):
    x = np.hstack([times, MS_MLI.spiketimes[i]])
    times = x
times = np.fmod(times, EXC_PARAMS[1]) # stack together all the periods
bin_mfr=0.005 # in seconds. Divide the weights by this to have a result per second = in Hz
fig = figure(figsize=(6,7))
ax = fig.add_subplot(211)
ax.hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
simpleaxis(ax)
tick_params(labelsize=18)
ylabel('Mean firing rate (Hz)', fontsize=18)
xlabel('time (s)', fontsize=18, labelpad=10)
title('MLI mean firing rates over time', fontsize=18)
yticks(arange(0,10,2))

tight_layout()
fig.subplots_adjust(hspace=.3)
fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/TOPALGStim/mean_firing_rate_TOPALG-connectivity_stim.png',dpi=600*3.35/6)
data = hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
trg=open('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/MLInet_ExcInput/Activity/TOPALGStim/mean_firing_rate_TOPALG-connectivity_stim.txt', 'w')
trg.write(str(data))
trg.close()