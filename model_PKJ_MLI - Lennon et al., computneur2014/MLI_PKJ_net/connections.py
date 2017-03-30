from pylab import *
from util import cartesian

def connect_mli_mli(syn, dist, syn_prob, dir_prob=.5):
    '''
    connects MLIs to MLIs up to a maximum distance of dist with
    probability syn_prob.  The axons extends unilaterally randomly
    in one direction or the other with probability dir_prob.
    
    directly modifies the Synapses object

    syn: synapses object
    '''
    N = len(syn.source)
    for src in xrange(N):
        axon_direction = 1
        if rand() < dir_prob: 
            axon_direction = -1
        for d in xrange(1,dist+1):
            if rand() <= syn_prob:
                trg = (src + d * axon_direction) % N
                syn[src,trg] = True
    return syn

def connect_mli_pkj(syn, pkj_dist, syn_prob, dir_prob=.5):
    '''
    connects MLIs to PKJs.  MLIs are grouped by PKJs.  Connects MLIs
    to PKJs up to a maximum distance of pkj_dist with
    probability syn_prob.  The axons extends unilaterally randomly
    in one direction or the other with probability dir_prob.

    directly modifies the Synapses object
    
    syn: synapses object
    '''
    N_MLI = len(syn.source)
    N_PKJ = len(syn.target)
    ratio = int(N_MLI/N_PKJ)
    for src in xrange(N_MLI):
        axon_direction = 1
        if rand() < dir_prob:
            axon_direction = -1
        closest_pkj_ind = src/ratio
        for d in xrange(pkj_dist):
            if rand() < syn_prob:
                trg = (closest_pkj_ind + d*axon_direction) % N_PKJ
                syn[src,trg] = True
    return syn
