''' Let's try in this script to select the relevant patterns to switch = the ones 
with a huge difference (loops, FF, 003 and 012), by being less demanding with the 
difference between the aimed and the actual distribution of a single pattern 0.01 
(the diff which includes it into the dec or inc lists, prev_aim_diff) while being 
more demanding with the mean difference netween patterns 0.001 (prev_aim_diff_out, 
which chooses the loops breaks in TOPALG. If there are only differences in a few 
patterns, the meandiff will be very little despite all > more demanding). 
HARD_TOPALG=False so the only differences made will be big ones. No need to change 
distributions which cannot be reached by the TOPALG anyway (weak differences).'''

from __future__ import print_function
import datetime
import os
from os import chdir
from brian import *

import copy as COPY
import sys
sys.path.append('../model_PKJ_MLI - Lennon et al., computneur2014')
from MLI_PKJ_net import *
sys.path.append('./')
from MLI_network-TOPALG import *
sys.path.append('./MLI_net_personal')
from connection import *
from molecular_layer_geometric import *
from sampling_samplingAnalysis import *
from sampling_samplingAnalysis import Nsample_sampling__Ntriads_patchings__triadsAnalysis

import cPickle
import time
set_global_preferences(useweave=True, usenewpropagate=True, usecodegen=True, usecodegenweave=True)
defaultclock.dt = .1*ms # not .25*ms
from statsmodels.tsa.stattools import acf
from pandas import *
HARD_TOPALG=False
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
w_mli_mli = 2. #instead of 1


# Synapses
S_transitive = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')
S_untransitive = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')

# Connections
connect_mli_mli_personal(MLItable, S_transitive, syn_prob=0.19) # according to experimental prob : 0.2
S_transitive.w[:,:] = 'rand()*w_mli_mli'

# save the genuine connectivity
syn_dir = './experiments_results/MLInet_ExcInput/Connectivity/RUN9/Transitive/Synapses/'
save_synapses(S_transitive, 'origin', syn_dir)

# Give the genuine connectivity (the one S_transitive already has it) to the second synapse
S_untransitive = copy_BrSynapse(syn_src=S_transitive, syn_trg=S_untransitive)
S_origin = copy_BrSynapse(syn_src=S_transitive, syn_trg=S_untransitive)

print('Convergences')
# print 'MLI->PKJ convergence: ', len(S_MLI_PKJ)/float(N_PKJ)
print ('MLI->MLI convergence: ', len(S_transitive)/float(N_MLI))
# print 'PKJ->MLI-BS convergence: ', len(S_PKJ_MLI)/(float(N_MLI))
print ('\nDivergences')
# print 'MLI->PKJ divergence: ', len(S_MLI_PKJ)/float(N_MLI)
print ('MLI->MLI divergence: ', len(S_transitive)/float(N_MLI))
# print 'PKJ->MLI-BS divergence: ', len(S_PKJ_MLI)/float(N_PKJ)

##################### TOPALG : PRE CORRECTION ANALYSIS - CORRECTION - POST CORRECTION ANALYSIS ##############################



# Analyze the genuine connectivity to start from there
Analysis_src = Nsample_sampling__Ntriads_patchings__triadsAnalysis(Nsample=2000, Ntriads=1, SynapseObj=S_transitive, 
    MLItable=MLItable, strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100)
src_dis = Analysis_src[5]
transitive_dis=src_dis.copy()
untransitive_dis=src_dis.copy()

if HARD_TOPALG==True: # use all the transitive/untransitive motifs
    ## Transitive distribution
    # 1) winner takes all between untransitive including loops (030C) and transitive including  feedforward (030T)motifs !
    transitive_dis['030T']+=(transitive_dis['030C']-0.005) # percentage of FF = FF+LP
    transitive_dis['030C']=0.005 # no LP left
    transitive_dis['120D']+=(transitive_dis['120C']-0.005)
    transitive_dis['120C']=0.005
    transitive_dis['300']+=transitive_dis['201']*0.1
    transitive_dis['300']+=transitive_dis['021C']*0.1
    transitive_dis['201']*=0.9
    transitive_dis['021C']*=0.9
    # 2) Carefully use some of the 012 to generate transitive motifs
    transitive_dis['030T']+=(transitive_dis['012']*0.3)*1./3 # three 1 edge motifs turned in one 3 edges motif and two no edges motifs
    transitive_dis['003']+=(transitive_dis['012']*0.3)*2./3
    transitive_dis['012']*=0.7

    ## Untransitive distribution
    # 1) winner takes all between untransitive including loops (030C) and transitive including  feedforward (030T)motifs !
    untransitive_dis['030C']+=untransitive_dis['030T'] # percentage of LP = FF+LP
    untransitive_dis['030T']=0. # no FF left
    untransitive_dis['120C']+=untransitive_dis['120D']
    untransitive_dis['120D']=0.
    untransitive_dis['201']+=untransitive_dis['300']/2
    untransitive_dis['021C']+=untransitive_dis['300']/2
    untransitive_dis['300']=0.
    # 2) Carefully use some of the 012 to generate transitive motifs
    untransitive_dis['030C']+=(untransitive_dis['012']*0.3)*1./3 # three 1 edge motifs turned in one 3 edges motif and two no edges motifs
    untransitive_dis['003']+=(untransitive_dis['012']*0.3)*2./3
    untransitive_dis['012']*=0.7

elif HARD_TOPALG==False: # only feedforwards and loops
    ## Transitive distribution
    # 1) winner takes all between untransitive including loops (030C) and transitive including  feedforward (030T)motifs !
    transitive_dis['030T']+=transitive_dis['030C'] # percentage of FF = FF+LP
    transitive_dis['030C']=0. # no LP left
    # 2) Carefully use some of the 012 to generate transitive motifs
    transitive_dis['030T']+=(transitive_dis['012']*0.3)*1./3 # three 1 edge motifs turned in one 3 edges motif and two no edges motifs
    transitive_dis['003']+=(transitive_dis['012']*0.3)*2./3
    transitive_dis['012']*=0.7
    
    ## Untransitive distribution
    # 1) winner takes all between untransitive including loops (030C) and transitive including  feedforward (030T)motifs !
    untransitive_dis['030C']+=untransitive_dis['030T'] # percentage of LP = FF+LP
    untransitive_dis['030T']=0. # no FF left
    # 2) Carefully use some of the 012 to generate transitive motifs
    untransitive_dis['030C']+=(untransitive_dis['012']*0.3)*1./3 # three 1 edge motifs turned in one 3 edges motif and two no edges motifs
    untransitive_dis['003']+=(untransitive_dis['012']*0.3)*2./3
    untransitive_dis['012']*=0.7


## Check there is no mistake
#transitive
sum_transitive_dis = 0
for val in transitive_dis.values():
    sum_transitive_dis+=val
if sum_transitive_dis<=0.99 or sum_transitive_dis>=1.01:
    print ('/!\ WARNING TRANS DISTRIBUTIONS SUM FAR FROM 1 : ', sum_transitive_dis)
elif 0.99<=sum_transitive_dis<=1.01:
    print('OK, distributions trans sum between 0.99 and 1.01 : ', sum_transitive_dis)
#untransitive
sum_untransitive_dis = 0
for val in untransitive_dis.values():
    sum_untransitive_dis+=val
if sum_untransitive_dis<=0.99 or sum_untransitive_dis>=1.01:
    print ('/!\ WARNING UNTRANS DISTRIBUTIONS SUM FAR FROM 1 : ', sum_untransitive_dis)
elif 0.99<=sum_untransitive_dis<=1.01:
    print('OK, distributions untrans sum between 0.99 and 1.01 : ', sum_untransitive_dis)

## Make the genuine connectivity converge toward this functionnaly different connectivity

# transitive
print("/n/n/n Let's go for the transitive TOPALG ! /n/n/n")
iterative_corrections_output = iterative_corrections(Niterations=10, Ncorrection=1000, NtriadsCorrection=1, Nanalysis=2000, NtriadsAnalysis=1, SynapseObj_real=S_transitive,
                                                     MLItable=MLItable,
                                                     TRY=10, prev_aim_diff=0.008, prev_aim_diff_out=0.003,
                                                     strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100,
                                                     Dis003=transitive_dis['003'], Dis012=transitive_dis['012'], Dis021C=transitive_dis['021C'],
                                                     Dis021D=transitive_dis['021D'],
                                                     Dis021U=transitive_dis['021U'], Dis030C=transitive_dis['030C'], Dis030T=transitive_dis['030T'],
                                                     Dis102=transitive_dis['102'],
                                                     Dis111D=transitive_dis['111D'], Dis111U=transitive_dis['111U'], Dis120C=transitive_dis['120C'],
                                                     Dis120D=transitive_dis['120D'],
                                                     Dis120U=transitive_dis['120U'], Dis201=transitive_dis['201'], Dis210=transitive_dis['210'],
                                                     Dis300=transitive_dis['300'], 
                                                     figsDirectory='./experiments_results/MLInet_ExcInput/Connectivity/RUN9/Transitive')

S_transitive = iterative_corrections_output[0]
print("(iteration no (0 before the first correction), meanDiff) : ")
print(iterative_corrections_output[1])
# save the transitive connectivity
syn_dir = './experiments_results/MLInet_ExcInput/Connectivity/RUN9/Transitive/Synapses/'
save_synapses(S_transitive, 'transitive', syn_dir)

# untransitive
print("/n/n/n Let's go for the untransitive TOPALG ! /n/n/n")
iterative_corrections_output = iterative_corrections(Niterations=10, Ncorrection=1000, NtriadsCorrection=1, Nanalysis=2000, NtriadsAnalysis=1, SynapseObj_real=S_untransitive,
                                                     MLItable=MLItable,
                                                     TRY=10, prev_aim_diff=0.008, prev_aim_diff_out=0.003,
                                                     strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100,
                                                     Dis003=untransitive_dis['003'], Dis012=untransitive_dis['012'], Dis021C=untransitive_dis['021C'],
                                                     Dis021D=untransitive_dis['021D'],
                                                     Dis021U=untransitive_dis['021U'], Dis030C=untransitive_dis['030C'], Dis030T=untransitive_dis['030T'],
                                                     Dis102=untransitive_dis['102'],
                                                     Dis111D=untransitive_dis['111D'], Dis111U=untransitive_dis['111U'], Dis120C=untransitive_dis['120C'],
                                                     Dis120D=untransitive_dis['120D'],
                                                     Dis120U=untransitive_dis['120U'], Dis201=untransitive_dis['201'], Dis210=untransitive_dis['210'],
                                                     Dis300=untransitive_dis['300'], 
                                                     figsDirectory='./experiments_results/MLInet_ExcInput/Connectivity/RUN9/Untransitive')

S_untransitive = iterative_corrections_output[0]
print("(iteration no (0 before the first correction), meanDiff) : ")
print(iterative_corrections_output[1])
# save the untransitive connectivity
syn_dir = './experiments_results/MLInet_ExcInput/Connectivity/RUN9/Untransitive/Synapses/'
save_synapses(S_untransitive, 'untransitive', syn_dir)


##########################  RUNNING caracteristics   ##################################

# Monitors
MS_MLI = SpikeMonitor(MLI)
MR_MLI = PopulationRateMonitor(MLI,bin=1*ms)
MISI_MLI = ISIHistogramMonitor(MLI,bins=arange(0,162,2)*ms)
MV_MLI = StateMonitor(MLI, 'V', record=range(N_MLI))

# Runtime
runtime = 400*second

# Deterministic input
EXC_INPUT=False
START = 0*second
EXC_PARAMS = [0.06, 0.5, 0.01*second] # ampin nA, period ("delay") in seconds, duration in seconds
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
network_origin = Network(MLI, S_origin, MS_MLI, MR_MLI, MISI_MLI, MV_MLI, exc_input, random_current)
network_transitive = Network(MLI, S_transitive, MS_MLI, MR_MLI, MISI_MLI, MV_MLI, exc_input, random_current)
network_untransitive = Network(MLI, S_untransitive, MS_MLI, MR_MLI, MISI_MLI, MV_MLI, exc_input, random_current)

##########################  RUN - transitive connectivity, nostim   ##################################

# Which stimuli
EXC_INPUT=False
RANDOM_CURRENT=True

# Run
TIME_BEFORE_RUN = time.time()
network_transitive.run(runtime)
print(time.time()-TIME_BEFORE_RUN, ' seconds of run - realtime.')
##########################  PLOT ACTIVITY - transitive connectivity, nostim ##################################

# Plot raster plot
rp = raster_plot(MS_MLI, newfigure=True)
rp.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/transitivenoStim/rasterplot_transitive-connectivity_nostim.png',dpi=600*6.93/25)

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
fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/transitivenoStim/autocorrelation_transitive-connectivity_nostim.png',
    dpi=600*6.93/25)

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
    fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/transitivenoStim/population_rate_histograms_transitive-connectivity_nostim.png',
        dpi=600*3.35/6)
except:
    pass

# plot mean firing rate = f(t)
times = array([])
for i in range(len(MLI)):
    x = np.hstack([times, MS_MLI.spiketimes[i]])
    times = x
times = np.fmod(times, EXC_PARAMS[1]) # stack together all the periods
bin_mfr=0.002 # in seconds. Divide the weights by this to have a result per second = in Hz
fig = figure(figsize=(6,7))
ax = fig.add_subplot(211)
array_mfr=np.ones(len(times))
array_mfr[0]=0
ax.hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
simpleaxis(ax)
tick_params(labelsize=18)
ylabel('Mean firing rate (Hz)', fontsize=18)
xlabel('time (s)', fontsize=18, labelpad=10)
title('MLI mean firing rates over time', fontsize=18)
yticks(arange(0,10,2))

tight_layout()
fig.subplots_adjust(hspace=.3)
fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/transitivenoStim/mean_firing_rate_transitive-connectivity_nostim.png',
    dpi=600*3.35/6)
data = hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
trg=open('./experiments_results/MLInet_ExcInput/Activity/RUN9/transitivenoStim/mean_firing_rate_transitive-connectivity_nostim.txt', 'w')
trg.write(str(data))
trg.close()



##########################  RUN - transitive connectivity, stim   ##################################

# restore the network's monitors and clock + the neurongroup without the synapse
network_transitive.reinit(states=False)
START = 0*second # not needed, just in case

# Which stimuli
EXC_INPUT, EXC_PARAMS[0] = True, 0.6 #Higher current amp, in nA
RANDOM_CURRENT = True

# Run
TIME_BEFORE_RUN = time.time()
network_transitive.run(runtime)
print(time.time()-TIME_BEFORE_RUN, ' seconds of run - realtime.')

##########################  PLOT ACTIVITY - transitive connectivity, stim ##################################

# Plot raster plot
rp = raster_plot(MS_MLI, newfigure=True)
rp.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/transitiveStim/rasterplot_transitive-connectivity_stim.png',
    dpi=600*6.93/25)

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
fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/transitiveStim/autocorrelation_transitive-connectivity_stim.png',
    dpi=600*6.93/25)

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
    fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/transitiveStim/population_rate_histograms_transitive-connectivity_stim.png',
        dpi=600*3.35/6)
except:
    pass

# plot mean firing rate = f(t)
times = array([])
for i in range(len(MLI)):
    x = np.hstack([times, MS_MLI.spiketimes[i]])
    times = x
times = np.fmod(times, EXC_PARAMS[1]) # stack together all the periods
bin_mfr=0.002 # in seconds. Divide the weights by this to have a result per second = in Hz
fig = figure(figsize=(6,7))
ax = fig.add_subplot(211)
array_mfr=np.ones(len(times))
array_mfr[0]=0
ax.hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
simpleaxis(ax)
tick_params(labelsize=18)
ylabel('Mean firing rate (Hz)', fontsize=18)
xlabel('time (s)', fontsize=18, labelpad=10)
title('MLI mean firing rates over time', fontsize=18)
yticks(arange(0,10,2))

tight_layout()
fig.subplots_adjust(hspace=.3)
fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/transitiveStim/mean_firing_rate_transitive-connectivity_stim.png',dpi=600*3.35/6)
data = hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
trg=open('./experiments_results/MLInet_ExcInput/Activity/RUN9/transitiveStim/mean_firing_rate_transitive-connectivity_stim.txt', 'w')
trg.write(str(data))
trg.close()

##########################  RUN - untransitive connectivity, nostim   ##################################

# restore the network's monitors and clock
network_transitive.reinit(states=False) # reinit transitive's monitors and clock since they are the same as untransitive's
START = 0*second # not needed, just in case

# Which stimuli
EXC_INPUT=False
RANDOM_CURRENT=True

# Run
TIME_BEFORE_RUN = time.time()
network_untransitive.run(runtime)
print(time.time()-TIME_BEFORE_RUN, ' seconds of run - realtime.')

##########################  PLOT ACTIVITY - untransitive connectivity, nostim ##################################

# Plot raster plot
rp = raster_plot(MS_MLI, newfigure=True)
rp.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/untransitivenoStim/rasterplot_untransitive-connectivity_nostim.png',
    dpi=600*6.93/25)

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
fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/untransitivenoStim/autocorrelation_untransitive-connectivity_nostim.png',
    dpi=600*6.93/25)

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
    fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/untransitivenoStim/population_rate_histograms_untransitive-connectivity_nostim.png',
        dpi=600*3.35/6)
except:
    pass

# plot mean firing rate = f(t)
times = array([])
for i in range(len(MLI)):
    x = np.hstack([times, MS_MLI.spiketimes[i]])
    times = x
times = np.fmod(times, EXC_PARAMS[1]) # stack together all the periods
bin_mfr=0.002 # in seconds. Divide the weights by this to have a result per second = in Hz
fig = figure(figsize=(6,7))
ax = fig.add_subplot(211)
array_mfr=np.ones(len(times))
array_mfr[0]=0
ax.hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
simpleaxis(ax)
tick_params(labelsize=18)
ylabel('Mean firing rate (Hz)', fontsize=18)
xlabel('time (s)', fontsize=18, labelpad=10)
title('MLI mean firing rates over time', fontsize=18)
yticks(arange(0,10,2))

tight_layout()
fig.subplots_adjust(hspace=.3)
fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/untransitivenoStim/mean_firing_rate_untransitive-connectivity_nostim.png',
    dpi=600*3.35/6)
data = hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
trg=open('./experiments_results/MLInet_ExcInput/Activity/RUN9/untransitivenoStim/mean_firing_rate_untransitive-connectivity_nostim.txt', 'w')
trg.write(str(data))
trg.close()


##########################  RUN - untransitive connectivity, stim   ##################################

# restore the network's monitors and clock
network_untransitive.reinit(states=False)
START = 0*second # global variable of the stim which has to be reset (was unset in transitive connectivity - stim)

# Which stimuli
EXC_INPUT, EXC_PARAMS[0] = True, 0.6 # Higher current amp, in nA
RANDOM_CURRENT = True

# Run
TIME_BEFORE_RUN = time.time()
network_untransitive.run(runtime)
print(time.time()-TIME_BEFORE_RUN, ' seconds of run - realtime.')

##########################  PLOT ACTIVITY - untransitive connectivity, stim ##################################

# Plot raster plot
rp = raster_plot(MS_MLI, newfigure=True)
rp.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/untransitiveStim/rasterplot_untransitive-connectivity_stim.png',
    dpi=600*6.93/25)

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
fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/untransitiveStim/autocorrelation_untransitive-connectivity_stim.png',
    dpi=600*6.93/25)

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
    fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/untransitiveStim/population_rate_histograms_untransitive-connectivity_stim.png',
        dpi=600*3.35/6)
except:
    pass

# plot mean firing rate = f(t)
times = array([])
for i in range(len(MLI)):
    x = np.hstack([times, MS_MLI.spiketimes[i]])
    times = x
times = np.fmod(times, EXC_PARAMS[1]) # stack together all the periods
bin_mfr=0.002 # in seconds. Divide the weights by this to have a result per second = in Hz
fig = figure(figsize=(6,7))
ax = fig.add_subplot(211)
array_mfr=np.ones(len(times))
array_mfr[0]=0
ax.hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
simpleaxis(ax)
tick_params(labelsize=18)
ylabel('Mean firing rate (Hz)', fontsize=18)
xlabel('time (s)', fontsize=18, labelpad=10)
title('MLI mean firing rates over time', fontsize=18)
yticks(arange(0,10,2))

tight_layout()
fig.subplots_adjust(hspace=.3)
fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/untransitiveStim/mean_firing_rate_untransitive-connectivity_stim.png',
    dpi=600*3.35/6)
data = hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
trg=open('./experiments_results/MLInet_ExcInput/Activity/RUN9/untransitiveStim/mean_firing_rate_untransitive-connectivity_stim.txt', 'w')
trg.write(str(data))
trg.close()

##########################  RUN - origin connectivity, nostim   ##################################

# restore the network's monitors and clock
network_untransitive.reinit(states=False) # reinit untransitive's monitors and clock since they are the same as origin's
START = 0*second # not needed, just in case

# Which stimuli
EXC_INPUT=False
RANDOM_CURRENT=True

# Run
TIME_BEFORE_RUN = time.time()
network_origin.run(runtime)
print(time.time()-TIME_BEFORE_RUN, ' seconds of run - realtime.')

##########################  PLOT ACTIVITY - origin connectivity, nostim ##################################

# Plot raster plot
rp = raster_plot(MS_MLI, newfigure=True)
rp.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/originnoStim/rasterplot_origin-connectivity_nostim.png',dpi=600*6.93/25)

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
fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/originnoStim/autocorrelation_origin-connectivity_nostim.png',
    dpi=600*6.93/25)

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
    fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/originnoStim/population_rate_histograms_origin-connectivity_nostim.png',
        dpi=600*3.35/6)
except:
    pass

# plot mean firing rate = f(t)
times = array([])
for i in range(len(MLI)):
    x = np.hstack([times, MS_MLI.spiketimes[i]])
    times = x
times = np.fmod(times, EXC_PARAMS[1]) # stack together all the periods
bin_mfr=0.002 # in seconds. Divide the weights by this to have a result per second = in Hz
fig = figure(figsize=(6,7))
ax = fig.add_subplot(211)
array_mfr=np.ones(len(times))
array_mfr[0]=0
ax.hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
simpleaxis(ax)
tick_params(labelsize=18)
ylabel('Mean firing rate (Hz)', fontsize=18)
xlabel('time (s)', fontsize=18, labelpad=10)
title('MLI mean firing rates over time', fontsize=18)
yticks(arange(0,10,2))

tight_layout()
fig.subplots_adjust(hspace=.3)
fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/originnoStim/mean_firing_rate_origin-connectivity_nostim.png',dpi=600*3.35/6)
data = hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
trg=open('./experiments_results/MLInet_ExcInput/Activity/RUN9/originnoStim/mean_firing_rate_origin-connectivity_nostim.txt', 'w')
trg.write(str(data))
trg.close()


##########################  RUN - origin connectivity, stim   ##################################

# restore the network's monitors and clock
network_origin.reinit(states=False)
START = 0*second # global variable of the stim which has to be reset (was unset in transitive connectivity - stim)

# Which stimuli
EXC_INPUT, EXC_PARAMS[0] = True, 0.6 # Higher current amp, in nA
RANDOM_CURRENT = True

# Run
TIME_BEFORE_RUN = time.time()
network_origin.run(runtime)
print(time.time()-TIME_BEFORE_RUN, ' seconds of run - realtime.')

##########################  PLOT ACTIVITY - origin connectivity, stim ##################################

# Plot raster plot
rp = raster_plot(MS_MLI, newfigure=True)
rp.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/originStim/rasterplot_origin-connectivity_stim.png',dpi=600*6.93/25)

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
fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/originStim/autocorrelation_origin-connectivity_stim.png',dpi=600*6.93/25)

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
    fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/originStim/population_rate_histograms_origin-connectivity_stim.png',
        dpi=600*3.35/6)
except:
    pass

# plot mean firing rate = f(t)
times = array([])
for i in range(len(MLI)):
    x = np.hstack([times, MS_MLI.spiketimes[i]])
    times = x
times = np.fmod(times, EXC_PARAMS[1]) # stack together all the periods
bin_mfr=0.002 # in seconds. Divide the weights by this to have a result per second = in Hz
fig = figure(figsize=(6,7))
ax = fig.add_subplot(211)
array_mfr=np.ones(len(times))
array_mfr[0]=0
ax.hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
simpleaxis(ax)
tick_params(labelsize=18)
ylabel('Mean firing rate (Hz)', fontsize=18)
xlabel('time (s)', fontsize=18, labelpad=10)
title('MLI mean firing rates over time', fontsize=18)
yticks(arange(0,10,2))

tight_layout()
fig.subplots_adjust(hspace=.3)
fig.savefig('./experiments_results/MLInet_ExcInput/Activity/RUN9/originStim/mean_firing_rate_origin-connectivity_stim.png',dpi=600*3.35/6)
data = hist(times, bins=np.arange(0,float(EXC_PARAMS[1]), bin_mfr), weights=np.ones(len(times))*(1./len(MLI))*(1./bin_mfr)*(1./(runtime/EXC_PARAMS[1])), color='#4b188c')
trg=open('./experiments_results/MLInet_ExcInput/Activity/RUN9/originStim/mean_firing_rate_origin-connectivity_stim.txt', 'w')
trg.write(str(data))
trg.close()