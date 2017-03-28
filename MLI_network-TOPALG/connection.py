from pylab import *
from util import cartesian
from math import *
from brian import *
import networkx as nx

### Two options to connect a BRIAN network :
### 1 - Directly implement the connectivity rules which add connections to the synapse object
### 2 - Transcript a NetworkX graph into the synapse object connections
### 3 - Copy the connectivity from a preexisting brian synapse (load_connectivity doesn't allow to modify a synapse whose connectivity has been loaded --')

'''
How to find the index from the coordinates (x,y):

Xvalue = what we decide
Yvalue = what we decide

for i in range(N):
    if (GeometricGroup[i,i+1].x==Xvalue) & (GeometricGroup[i,i+1].y==Yvalue):
    print(GeometricGroup[i,i+1], 'has the index', i)
'''

### 1 - Directly implement the connectivity rules which add connections to the synapse object

def connect_mli_mli_personal(NEURONtable, syn, dist=13.15, syn_prob=0.5, dir_prob=0.5):
    '''
        "SYNAPSE MODIFIER"
    
    This program connects MLIs (i index, sources) to MLIs (j index, targets) up to a maximum distance of dist with probability syn_prob.
    
    The axons extends randomly in any direction towards the 2D grid with the probability dir_prob.
    
    directly modifies the Synapses object syn made between the SourceGroup and the TargetGroup
    example : syn = Synapses(SourceGroup, TargetGroup, model='w:1', pre='g_inh+=MLI.g_inh_*w')

    '''
    Nx = len(NEURONtable)
    Ny = len(NEURONtable[0])
    
    for y1 in range(Ny): # for each MLItable raw
        for x1 in range(Nx): # we screen MLIs inside the MLItable columns : we have a firest MLI chosen
            i=NEURONtable[x1][y1][1] # from which we extract the index, an attribute of the neuron
            
            for y2 in range(Ny):
                for x2 in range(Nx): # and we do the same to chose a second MLI
                    j=NEURONtable[x2][y2][1] # and hop the second neuron index !
            
                    if sqrt(
                            (y1-y2)**2
                            +
                            (x1-x2)**2
                            ) <= dist: #  if the two MLIs are close enough
                        if rand() <= syn_prob: # we try to make a synapse
                            if i!=j and syn.w[i,j].tolist()==[]: # There are cases of MLIs connected to themselves : but then they hyperpolarize themselves when they are already in a refractory period... Function not well understood > Let's consider they cannot connect to themselves. Ask a precise paper to Arnd
                                syn[i,j] = True
    return syn

### 2 - Transcript a NetworkX graph into the synapse object connections

def connect_networkXbased(networkXgraph, syn):
    '''Connect a BRIAN synapse "syn" according to a NetworkX Digraph "networkXgraph" connectivity, whose nodes refer to neurons indexes.'''
    
    for node1 in networkXgraph.keys(): # screen all the nodes node1 of the graph.
        for node2 in networkXgraph[node1].keys(): # for the nodes node2 to which each is bound,
            if syn.w[node1, node2].tolist() == []:
                syn[node1, node2]=True
            syn.w[node1, node2] = 'rand()'
    return syn



### It may be useful to be able to create a NetworkX graph from a BRIAN synapse object

def BrSynapseToNxGraph(syn, MLItable):
    '''syn is an empty BRIAN synapse object between two BRIAN neuron groups, and all the weights are at [].
        syn = S_MLI_MLI
        MLIable = the table storing MLIs (in order to calculate the number of neurons connected by the SYnapseObj)
        
        /!\ works only if syn source NeuronGroup = syn target NeuronGroup'''
    Nindex = len(MLItable)*len(MLItable[0])
    NxGraph = nx.DiGraph() # create the graph
    for n in range(Nindex):
        NxGraph.add_node(n) # screen all neurons indexes from the synapse object and make them nodes in the NxGraph
    
    for i in range(Nindex): # screen all neurons indexes from the synapse object
        for j in range(Nindex): # idem
            print('i, j = ',i, ', ', j)
            if syn.w[i,j].tolist()!=[] and syn.w[i,j].tolist()!=[0.0]:
                NxGraph.add_edge(i, j)

    return NxGraph


### 3 - Copy the connectivity from a preexisting brian synapse

def copy_BrSynapse(syn_src, syn_trg):
    '''Copy the connectivity from a preexisting syn_src brian synapse in a target syn_trg brian synapse(load_connectivity doesn't allow to modify a synapse whose connectivity has been loaded --')'''
    Nindex = len(syn_src.source)
    for i in range(Nindex): # screen all neurons indexes from the source synapse
        for j in range(Nindex): # idem
            if syn_src.w[i,j].tolist()!=[]:
                if syn_trg.w[i,j].tolist()==[]:
                    syn_trg[i,j]=True
                syn_trg.w[i,j] = syn_src.w[i,j].tolist()[0] # array > list > float = list element
            elif syn_src.w[i,j].tolist()==[]: # useless for still not connected synapses
                if syn_trg.w[i,j].tolist()!=[]:
                    syn_trg.w[i,j] = 0

    return syn_src





