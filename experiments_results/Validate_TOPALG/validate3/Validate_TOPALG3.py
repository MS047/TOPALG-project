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

######################### MAKE THE NETWORK ###################################

load_saved_synapses = False
save_results = False

T = 300*second
# N_MLI = 160
# N_PKJ = 16
# MLI = MLIGroup(N_MLI)
# PKJ = PurkinjeCellGroup(N_PKJ)

N_MLI_geometric_OX=16 # MLItable "columns" : len(MLItable[0])
N_MLI_geometric_OY=10 # MLItable "raws" : len(MLItable)
N_MLI = N_MLI_geometric_OX*N_MLI_geometric_OY
MLI = Geometric_MLI(N_MLI)
MLItable_output = create_MLItable(MLI, N_MLI_geometric_OX, N_MLI_geometric_OY)
MLItable = MLItable_output[0]
MLI = MLItable_output[1]

########### CONNECT THE SOURCE SYNAPSE OBJECT - GET ITS MOTIFS DISRIBUTION ##################
# any network works, it simply has to indicate any possible distribution within a single network
# To make it a bit different from the test network : add "dist=18" to the connection function

# synaptic weights
w_mli_mli = 1.


# Synapses
S_MLI_MLI_src = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')
print('/n source synapse created')

# Connections
connect_mli_mli_personal(MLItable, S_MLI_MLI_src, syn_prob=0.19, dist=18)
S_MLI_MLI_src.w[:,:] = 'rand()*w_mli_mli'
print('/n source synapse connected')
    
Analysis_src = Nsample_sampling__Ntriads_patchings__triadsAnalysis(Nsample=1000, Ntriads=1, SynapseObj=S_MLI_MLI_src, MLItable=MLItable, strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100)
src_dis = Analysis_src[5]
print("/n Source network distribution : ", src_dis)

########### CONNECT THE TEST SYNAPSE OBJECT ##################
# its connection must be quite different from the source network
# This network will be the one which will be modified by TOPALG. If its distribution reaches the source network distribution, TOPALG works.
# synaptic weights
w_mli_mli = 1.


# Synapses
S_MLI_MLI = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')


# Connections
connect_mli_mli_personal(MLItable, S_MLI_MLI, syn_prob=0.19, dist=13.15)
S_MLI_MLI.w[:,:] = 'rand()*w_mli_mli'


########################## PRE CORRECTION ANALYSIS - CORRECTION - POST CORRECTION ANALYSIS ##################################
# at Ncorrection=1000, less performant than 500 because at each iteration, one single edge has a big probability to be screened several times > destruction of the work previously done
# at Nanaysis = 1000, ~872 triads only are sampled > such a big number is useless ?
S_MLI_MLI = iterative_corrections(Niterations=10, Ncorrection=300, NtriadsCorrection=1, Nanalysis=1000, NtriadsAnalysis=1, SynapseObj_real=S_MLI_MLI,
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
                      Dis300=src_dis['300'], figsDirectory='/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/Validate_TOPALG')
