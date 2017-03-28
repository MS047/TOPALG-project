from __future__ import print_function
import datetime
import os
from os import chdir
from brian import *

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
defaultclock.dt = .25*ms
from statsmodels.tsa.stattools import acf
from pandas import *

######################### MAKE THE NETWORK NEURONS ###################################

load_saved_synapses = False
save_results = False

T = 300*second
# N_MLI = 160
# N_PKJ = 16
# MLI = MLIGroup(N_MLI)
# PKJ = PurkinjeCellGroup(N_PKJ)

N_MLI_geometric_OX=10 # MLItable "columns" : len(MLItable[0])
N_MLI_geometric_OY=16 # MLItable "raws" : len(MLItable)
N_MLI = N_MLI_geometric_OX*N_MLI_geometric_OY
MLI = Geometric_MLI(N_MLI)
MLItable_output = create_MLItable(MLI, N_MLI_geometric_OX, N_MLI_geometric_OY)
MLItable = MLItable_output[0]
MLI = MLItable_output[1]

########################### CONNECT THE NETWORK #################################

# synaptic weights
# w_mli_pkj = 1.25
w_mli_mli = 1.
# w_pkj_mli = 1.

# Synapses
# S_MLI_PKJ = Synapses(MLI,PKJ,model='w:1',pre='g_inh+=PKJ.g_inh_*w')
S_MLI_MLI = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')
# S_PKJ_MLI = Synapses(PKJ,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')

# Connections
if load_saved_synapses:
    syn_dir = './'
    # S_MLI_PKJ = load_synapses(S_MLI_PKJ, 'S_MLI_PKJ', syn_dir)
    S_MLI_MLI = load_synapses(S_MLI_MLI, 'S_MLI_MLI', syn_dir)
# S_PKJ_MLI = load_synapses(S_PKJ_MLI, 'S_PKJ_MLI', syn_dir)
else:
    # connect_mli_pkj(S_MLI_PKJ,pkj_dist=8,syn_prob=.25)
    # connect_mli_mli(S_MLI_MLI,dist=80,syn_prob=.05)
    #connect_mli_mli_personal(MLItable, S_MLI_MLI, syn_prob=0.19) # according to experimental prob : 0.2
    Nx = len(MLItable)
    Ny = len(MLItable[0])
    
    for y1 in range(Ny): # for each MLItable raw
        for x1 in range(Nx): # we screen MLIs inside the MLItable columns : we have a firest MLI chosen
            i=MLItable[x1][y1][1] # from which we extract the index, an attribute of the neuron
            
            for y2 in range(Ny):
                for x2 in range(Nx): # and we do the same to chose a second MLI
                    j=MLItable[x2][y2][1] # and hop the second neuron index !
                    if i!=j: # There are cases of MLIs connected to themselves : but then they hyperpolarize themselves when they are already in a refractory period... Function not well understood > Let's consider they cannot connect to themselves. Ask a precise paper to Arnd
                        S_MLI_MLI[i,j] = True


    # S_PKJ_MLI[:,:] = '((j/(N_MLI/N_PKJ)-i)%N_PKJ <= 2) & ((j/(N_MLI/N_PKJ)-i)%N_PKJ > 0) & (j%(N_MLI/N_PKJ)<3) & (rand()<.5)'
    # S_MLI_PKJ.w[:,:] = 'rand()*w_mli_pkj'
    S_MLI_MLI.w[:,:] = 'rand()*w_mli_mli'
# S_PKJ_MLI.w[:,:] = 'rand()*w_pkj_mli'


print('Convergences')
# print 'MLI->PKJ convergence: ', len(S_MLI_PKJ)/float(N_PKJ)
print ('MLI->MLI convergence: ', len(S_MLI_MLI)/float(N_MLI))
# print 'PKJ->MLI-BS convergence: ', len(S_PKJ_MLI)/(float(N_MLI))
print ('\nDivergences')
# print 'MLI->PKJ divergence: ', len(S_MLI_PKJ)/float(N_MLI)
print ('MLI->MLI divergence: ', len(S_MLI_MLI)/float(N_MLI))
# print 'PKJ->MLI-BS divergence: ', len(S_PKJ_MLI)/float(N_PKJ)

print('Number of connections when fully connected network : '+str(len(S_MLI_MLI)))


########################## WHOLE NETWORK PRE-CORRECTION ANALYSIS - CORRECTION - WHOLE NETWORK POST-CORRECTION ANALYSIS ##################################

iterative_corrections(Niterations=1, Ncorrection=100, NtriadsCorrection=1, Nanalysis=500, NtriadsAnalysis=1, SynapseObj_real=S_MLI_MLI,
                      MLItable=MLItable,
                      TRY=10, prev_aim_diff=0.003,
                      strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100,
                      Dis003=65.0 / 173, Dis012=46.0 / 173, Dis021C=8.0 / 173,
                      Dis021D=18.0 / 173,
                      Dis021U=10.0 / 173, Dis030C=0.0 / 173, Dis030T=13.0 / 173,
                      Dis102=4.0 / 173,
                      Dis111D=0.0 / 173, Dis111U=5.0 / 173, Dis120C=1.0 / 173,
                      Dis120D=0.0 / 173,
                      Dis120U=3.0 / 173, Dis201=0.0 / 173, Dis210=0.0 / 173,
                      Dis300=0.0 / 173, figsDirectory='/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/personnalCon_noPKJ/figures/motifs distributions/Debug')



'''
########################## WHOLE NETWORK PRE-CORRECTION ANALYSIS ##################################

chdir('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/personnalCon_noPKJ/connectivity')
from MLI_connectivity_TableReadout import *
write_synapses_table(S_MLI_MLI, 'personalCon_noPKJ.txt')
print(len(S_MLI_MLI), " synapses in S_MLI_MLI.")



import pandas as pd
from numpy import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

Analysis_confirmation=Nsample_sampling__Ntriads_patchings__triadsAnalysis(Nsample=1000, Ntriads=1, SynapseObj=S_MLI_MLI, MLItable=MLItable, strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100)
print ("\n\n\n\nTriads analysis : ", Analysis_confirmation[5])
print ("Aimed analysis : ", Analysis_confirmation[4])
print ("\n >> Difference between two : ", Analysis_confirmation[3])
print ("Connection Probability :", Analysis_confirmation[2])
print ("Number of synapses inS_MLI_MLI :", len(S_MLI_MLI))

patList = []
patValList = []
for pat in Analysis_confirmation[0]:
    patList.append(pat)
    patValList.append((Analysis_confirmation[4][pat]*100, Analysis_confirmation[5][pat]*100))

patDataFrame = pandas.DataFrame(data = patValList, index = patList, columns = ['Aimed dist. -- '+str(Analysis_confirmation[3]), 'Network dist.'])

ax = patDataFrame.plot.bar()  # s is an instance of Series
fig = ax.get_figure()
fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/personnalCon_noPKJ/figures/motifs distributions/Debug/wholeDebugNet_preCorrection.pdf')



########################## CORRECTION ##################################

correctorOutput = triadsPatterns_distribution_corrector(Nsample=100, Ntriads=1, SynapseObj=S_MLI_MLI, MLItable=MLItable,
                                                        TRY=20, prev_aim_diff=0.001,
                                                        strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100,
                                                        Dis003=65.0 / 173, Dis012=46.0 / 173, Dis021C=8.0 / 173,
                                                        Dis021D=18.0 / 173,
                                                        Dis021U=10.0 / 173, Dis030C=0.0 / 173, Dis030T=13.0 / 173,
                                                        Dis102=4.0 / 173,
                                                        Dis111D=0.0 / 173, Dis111U=5.0 / 173, Dis120C=1.0 / 173, Dis120D=0.0 / 173,
                                                        Dis120U=3.0 / 173, Dis201=0.0 / 173, Dis210=0.0 / 173, Dis300=0.0 / 173)
#low prev_aim_diff : plus il est faible, plus les listes incPat/decPat resteront pleines facilement, ce qui assure que le programme continuera de tourner plus longtemps.
S_MLI_MLI = correctorOutput[0]
endDistribution = correctorOutput[1]
aimedDistribution = correctorOutput[2]

patList = []
patValList = []
for pat in endDistribution:
    patList.append(pat)
    patValList.append((aimedDistribution[pat]*100, endDistribution[pat]*100))

patDataFrame = pandas.DataFrame(data = patValList, index = patList, columns = ['Aimed dist. -- '+str(correctorOutput[3]), 'Sample dist.'])

ax = patDataFrame.plot.bar()  # s is an instance of Series
fig = ax.get_figure()
fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/personnalCon_noPKJ/figures/motifs distributions/Debug/sampledDebugNet_forCorrection.pdf')



########################## WHOLE NETWORK POST-CORRECTION ANALYSIS ##################################

# Connectivity readout
from os import chdir
chdir('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/personnalCon_noPKJ/connectivity')
from MLI_connectivity_TableReadout import *
write_synapses_table(S_MLI_MLI, 'personalCon_noPKJ_patternDisCorrected.txt')



Analysis_confirmation=Nsample_sampling__Ntriads_patchings__triadsAnalysis(Nsample=1000, Ntriads=1, SynapseObj=S_MLI_MLI, MLItable=MLItable, strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100)
print ("\n\n\n\nTriads analysis : ", Analysis_confirmation[5])
print ("Aimed analysis : ", Analysis_confirmation[4])
print ("\n >> Difference between two : ", Analysis_confirmation[3])
print ("Connection Probability :", Analysis_confirmation[2])
print ("Number of synapses in S_MLI_MLI :", len(S_MLI_MLI))

patList = []
patValList = []
for pat in Analysis_confirmation[0]:
    patList.append(pat)
    patValList.append((Analysis_confirmation[4][pat]*100, Analysis_confirmation[5][pat]*100))

patDataFrame = pandas.DataFrame(data = patValList, index = patList, columns = ['Aimed dist. -- '+str(Analysis_confirmation[3]), 'Network dist.'])

ax = patDataFrame.plot.bar()  # s is an instance of Series
fig = ax.get_figure()
fig.savefig('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/personnalCon_noPKJ/figures/motifs distributions/Debug/wholeDebugNet_postCorrection.pdf')'''

'''
##########################  RUN  ##################################

@network_operation(Clock(dt=defaultclock.dt))
def random_current():
    MLI.I = gamma(3.966333,0.006653,size=len(MLI)) * nA
# PKJ.I = gamma(0.430303,0.195962,size=len(PKJ)) * nA

# Monitor
MS_MLI = SpikeMonitor(MLI)
MR_MLI = PopulationRateMonitor(MLI,bin=1*ms)
MISI_MLI = ISIHistogramMonitor(MLI,bins=arange(0,162,2)*ms)
MV_MLI = StateMonitor(MLI, 'V', record=range(N_MLI))

# MS_PKJ = SpikeMonitor(PKJ)
# MR_PKJ = PopulationRateMonitor(PKJ,bin=1*ms)
# MISI_PKJ = ISIHistogramMonitor(PKJ,bins=arange(0,162,2)*ms)
# MV_PKJ = StateMonitor(PKJ, 'V', record=range(N_PKJ))

start = time.time()
run(T)
print(time.time() - start)

if save_results:
    # save monitors for reproducibility
    monitors = {'MS_MLI':MS_MLI, 'MR_MLI':MR_MLI, 'MISI_MLI':MISI_MLI, 'MV_MLI':MV_MLI, 'MS_PKJ':MS_PKJ, 'MR_PKJ':MR_PKJ, 'MISI_PKJ':MISI_PKJ, 'MV_PKJ':MV_PKJ}
    out_dir = './'
    for name, mon in monitors.iteritems():
        cPickle.dump(mon, open(out_dir+'%s.pkl'%name,'w'))

    # save connections for reproducibility
    syn_dir = './'
    # save_synapses(S_MLI_PKJ, 'S_MLI_PKJ', syn_dir)
    save_synapses(S_MLI_MLI, 'S_MLI_MLI', syn_dir)
# save_synapses(S_PKJ_MLI, 'S_PKJ_MLI', syn_dir)

##########################  PLOT ACTIVITY  ##################################

chdir('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/personnalCon_noPKJ/figures/ISI_histogram_')

load_monitors = False
if load_monitors:
    in_dir = './'
    MS_MLI = cPickle.load(open(in_dir+'MS_MLI.pkl'))
    MV_MLI = cPickle.load(open(in_dir+'MV_MLI.pkl'))
# MS_PKJ = cPickle.load(open(in_dir+'MS_PKJ.pkl'))
# MV_PKJ = cPickle.load(open(in_dir+'MV_PKJ.pkl'))

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
fig.savefig('MLI_DebugNet_color.tiff',dpi=600*6.93/25)
fig.savefig('MLI_DebugNet_color.png',dpi=600*6.93/25)

from scipy.stats import spearmanr

def population_fr_stats(spike_monitor):
    mean_frs = []
    for ind in range(len(spike_monitor.spiketimes)):
        mean_frs.append(mean(diff(spike_monitor.spiketimes[ind]))**-1)
    return mean(mean_frs), std(mean_frs), max(mean_frs), min(mean_frs), mean_frs

def population_isi_cv_stats(spike_monitor):
    cvs = []
    for ind in range(len(spike_monitor.spiketimes)):
        isi_mean, isi_std = isi_mean_and_std(spike_monitor, ind)
        cvs.append(isi_std/isi_mean)
    return mean(cvs), std(cvs), max(cvs), min(cvs), cvs

m,s,ma,mi,frs = population_fr_stats(MS_MLI)
print('Mean MLI FR: %s, Std: %s, Max: %s, Min: %s'%(m,s,ma,mi))
m,s,ma,mi,cvs = population_isi_cv_stats(MS_MLI)
print('Mean MLI CV: %s, Std: %s, Max: %s, Min: %s'%(m,s,ma,mi))
print("MLI FR-CV correlation.  Spearman's R: %s, p = %s" % spearmanr(frs,cvs))
# m,s,ma,mi,frs = population_fr_stats(MS_PKJ)
# print 'Mean PKJ FR: %s, Std: %s, Max: %s, Min: %s'%(m,s,ma,mi)
# m,s,ma,mi,cvs = population_isi_cv_stats(MS_PKJ)
# print 'Mean PKJ CV: %s, Std: %s, Max: %s, Min: %s'%(m,s,ma,mi)
# print "PKJ FR-CV correlation.  Spearman's R: %s, p = %s" % spearmanr(frs,cvs)

# plot histogram
fig = figure(figsize=(6,7))

ax = fig.add_subplot(211)
ax.hist([mean(diff(MS_MLI.spiketimes[i]))**-1 for i in range(N_MLI)],15,color='#8C2318', edgecolor='w')
simpleaxis(ax)
tick_params(labelsize=20)
xlabel('Mean firing rate (Hz)', fontsize=20)
ylabel('Number of cells', fontsize=20, labelpad=10)
title('MLI mean firing rates', fontsize=20)
yticks(arange(0,19,3))

# ax = fig.add_subplot(212)
# counts, bins, _ = ax.hist([mean(diff(MS_PKJ.spiketimes[i]))**-1 for i in range(N_PKJ)],9,color='#0B486B',edgecolor='w')
# simpleaxis(ax)
# tick_params(labelsize=20)
# xlabel('Mean firing rate (Hz)', fontsize=20)
# ylabel('Number of cells', fontsize=20, labelpad=10)
# title('PKJ mean firing rates', fontsize=20)
# yticks(range(int(counts.max()+1)))

tight_layout()
fig.subplots_adjust(hspace=.3)
fig.savefig('debugPopulation_rate_histograms_color.tiff',dpi=600*3.35/6)
fig.savefig('debugPopulation_rate_histograms_color.png',dpi=600*3.35/6)'''