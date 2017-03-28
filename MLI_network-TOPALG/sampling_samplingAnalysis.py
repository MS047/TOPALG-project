from __future__ import print_function
from molecular_layer_geometric import create_MLItable
from connections import BrSynapseToNxGraph
from random import *
from random import random
import networkx as nx
from networkx import triadic_census

from math import sqrt
import pylab as pl
import numpy as np
import pandas as pd
from numpy import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from os import chdir
from MLI_connectivity_TableReadout import *

def sample_MLI_interactive(MLItable):

    ## warn the user about the size of the MLItable he wants to sample and ask him the sample size
    OY, OX = len(MLItable), len(MLItable[0])
    OYum, OXum = OY*38, OX*38 # one intersoma distance between 2 MLI is 38 um
    # print("Your table ranges along", OY, "neurons on the y axis and along", OX, "neurons on the x axis.")
    # print("It simulates a sagital slice of", OYum, "micrometer of lenght (oy) and", OXum, "micrometer of width(ox).")

    ## ask the user the sampling strategy, and the coordinates unit (neuron index or micrometer)
    while 1: # how to sample
        samplingStrategyChoice = raw_input("\nyou can choose either to sample a random square of given width and length, either to enter precise sample coordinates. Choose <rdm> or <prc> : ")
        if samplingStrategyChoice =='rdm' or samplingStrategyChoice=='prc':
            break
        else:
            print("ERROR : other choice than <rdm> or <prc> detected. Try again.")

    while 1: # which units use
        unitChoice = raw_input("\nyou can choose to sample a square either from neuron indexes (x,y) coordinates or from micrometer (x,y) coordinates. Choose <idx> or <um> : ")
        if unitChoice=='idx' or unitChoice=='um':
            break
        else:
            print("\nERROR : other choice than <idx> or <um> detected. Try again.")

    ## define the square index unit coordinates in function of the chosen strategy and unit
    if samplingStrategyChoice=='rdm': # RANDOM sample strategy

        if unitChoice == 'idx':
            while 1:
                Ysize = input("\nPlease enter the length (along oy) of your sample square, in index units (how many neurons do you wish to sample) : ")
                Xsize = input("Please enter the width (along ox) of your sample square, in index units (how many neurons do you wish to sample) : ")
                if Ysize<=OY or Xsize<=OX: # the sample size is maximum the number of neurons in the table
                    break
                else:
                    print("       /!\ WARNING /!\ : you seem to try to sample a bigger square than your neurons table is. Try again.")

        elif unitChoice == 'um':
            while 1:
                Ysize = int(float(input("\nPlease enter the length (oy) of your sample square in um : ")) / 38)+1 # direct conversion in index units, +1 to "cut" the slice longer
                Xsize = int(float(input("Please enter the width (ox) of your sample square in um : ")) / 38)+1 # direct conversion in index units, +1 to "cut" the slice wider
                if Ysize <= OY or Xsize <= OX:  # the sample size is maximum the number of beurons in the table
                    break
                else:
                    print("       /!\ WARNING /!\ : you seem to try to sample a bigger square than your neurons table is. Try again.")

        upperYlim = OY - Ysize
        if upperYlim == 0:
            y1 = 0
        elif upperYlim > 0:
            y1 = randrange(0,upperYlim)  # selects a sarting y value to draw the square, between index (0) and index (max) - (size of the square side)
        y2 = y1 + Ysize - 1  # for 5 neurons to sample, you want to go from 0 to 4 for instance, not 0 to 5
        upperXlim = OX - Xsize
        if upperXlim == 0:
            x1 = 0
        elif upperYlim > 0:
            x1 = randrange(0, upperXlim)  # idem with x
        x2 = x1 + Xsize - 1  # idem with x





    elif samplingStrategyChoice=='prc': # PRECISE COORDINATES sample strategy

        if unitChoice == 'idx':
            x1 = input("enter x index of the first x neuron of the sample")
            x2 = input("enter x index of the last x neuron of the sample")
            y1 = input("enter y index of the first y neuron of the sample")
            y2 = input("enter y index of the last y neuron of the sample")
        elif unitChoice == 'um':
            x1 = float(input("enter x1 coordinate of the square in um"))/38
            x2 = float(input("enter x2 coordinate of the square in um"))/38
            y1 = float(input("enter y1 coordinate of the square in um"))/38
            y2 = float(input("enter y2 coordinate of the square in um"))/38

    ## turn x and y coordinates of the square as integers if floats (happens during the conversion in um), below the low values and above the high values
    ## and print them

    if unitChoice == 'um':
        x1 = int(x1)  # lower value
        y1 = int(y1)  # lower value
        x2 = int(x2)+1 # higher value
        y2 = int(y2)+1 # higher value

    print("\n\n     >>> sampled square coordinates : y1=", y1, ", y2=", y2, ", x1=", x1, ", x2=", x2)

    ## write the new table with the so-obtained x1, x2, y1 and y2 coordinates

    MLIsample=[]

    raw=0
    Y=y1
    while Y<=y2:
        MLIsample.append([]) # creates the "raws"
        X=x1
        while X<=x2:
            MLIsample[raw].append(MLItable[Y][X])
            X+=1
        Y+=1
        raw+=1

    print(MLIsample)
    return MLIsample

def sample_MLI(MLItable, strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100):

    ## warn the user about the size of the MLItable he wants to sample and ask him the sample size
    OY, OX = len(MLItable), len(MLItable[0])
    OYum, OXum = OY*38, OX*38 # one intersoma distance between 2 MLI is 38 um
    # print("Your table ranges along", OY, "neurons on the y axis and along", OX, "neurons on the x axis.")
    # print("It simulates a sagital slice of", OYum, "micrometer of lenght (oy) and", OXum, "micrometer of width(ox).")

    ## ask the user the sampling strategy, and the coordinates unit (neuron index or micrometer)
    while 1: # how to sample
        samplingStrategyChoice = strategy #raw_input("\nyou can choose either to sample a random square of given width and length, either to enter precise sample coordinates. Choose <rdm> or <prc> : ")
        if samplingStrategyChoice=='rdm' or samplingStrategyChoice=='prc':
            break
        else:
            print("ERROR : other choice than <rdm> or <prc> detected. Try again.")

    while 1: # which units use
        unitChoice = coordUnit #raw_input("\nyou can choose to sample a square either from neuron indexes (x,y) coordinates or from micrometer (x,y) coordinates. Choose <idx> or <um> : ")
        if unitChoice=='idx' or unitChoice=='um':
            break
        else:
            print("\nERROR : other choice than <idx> or <um> detected. Try again.")

    ## define the square index unit coordinates in function of the chosen strategy and unit
    if samplingStrategyChoice=='rdm': # RANDOM sample strategy

        if unitChoice == 'idx':
            while 1:
                Ysize = input("\nPlease enter the length (along oy) of your sample square, in index units (how many neurons do you wish to sample) : ")
                Xsize = input("Please enter the width (along ox) of your sample square, in index units (how many neurons do you wish to sample) : ")
                if Ysize<=OY or Xsize<=OX: # the sample size is maximum the number of neurons in the table
                    break
                else:
                    print("       /!\ WARNING /!\ : you seem to try to sample a bigger square than your neurons table is. Try again.")

        elif unitChoice == 'um':
            while 1:
                Ysize = int(float(Ysizeum) / 38) + 1
                #Ysize = int(float(input("\nPlease enter the length (oy) of your sample square in um : ")) / 38)+1 # direct conversion in index units, +1 to "cut" the slice longer
                Xsize = int(float(Xsizeum) / 38) + 1
                #Xsize = int(float(input("Please enter the width (ox) of your sample square in um : ")) / 38)+1 # direct conversion in index units, +1 to "cut" the slice wider
                if Ysize <= OY or Xsize <= OX:  # the sample size is maximum the number of beurons in the table
                    break
                else:
                    print("       /!\ WARNING /!\ : you seem to try to sample a bigger square than your neurons table is. Try again.")

        upperYlim = OY - Ysize
        if upperYlim == 0:
            y1 = 0
        elif upperYlim > 0:
            y1 = randrange(0,
                           upperYlim)  # selects a sarting y value to draw the square, between index (0) and index (max) - (size of the square side)
        y2 = y1 + Ysize - 1  # for 5 neurons to sample, you want to go from 0 to 4 for instance, not 0 to 5
        upperXlim = OX - Xsize
        if upperXlim == 0:
            x1 = 0
        elif upperYlim > 0:
            x1 = randrange(0, upperXlim)  # idem with x
        x2 = x1 + Xsize - 1  # idem with x





    elif samplingStrategyChoice=='prc': # PRECISE COORDINATES sample strategy

        if unitChoice == 'idx':
            x1 = input("enter x index of the first x neuron of the sample")
            x2 = input("enter x index of the last x neuron of the sample")
            y1 = input("enter y index of the first y neuron of the sample")
            y2 = input("enter y index of the last y neuron of the sample")
        elif unitChoice == 'um':
            x1 = float(input("enter x1 coordinate of the square in um"))/38
            x2 = float(input("enter x2 coordinate of the square in um"))/38
            y1 = float(input("enter y1 coordinate of the square in um"))/38
            y2 = float(input("enter y2 coordinate of the square in um"))/38

    ## turn x and y coordinates of the square as integers if floats (happens during the conversion in um), below the low values and above the high values
    ## and print them

    if unitChoice == 'um':
        x1 = int(x1)  # lower value
        y1 = int(y1)  # lower value
        x2 = int(x2)+1 # higher value
        y2 = int(y2)+1 # higher value

    # >>> sampled square coordinates : y1=", y1, ", y2=", y2, ", x1=", x1, ", x2=", x2

    ## write the new table with the so-obtained x1, x2, y1 and y2 coordinates

    MLIsample=[]

    raw=0
    Y=y1
    while Y<=y2:
        MLIsample.append([]) # creates the "raws"
        X=x1
        while X<=x2:
            MLIsample[raw].append(MLItable[Y][X])
            X+=1
        Y+=1
        raw+=1

    return MLIsample

def patch_MLI(MLIsample, SynapseObj):

    MLIgraph=nx.DiGraph()

    # generate 3 random pairs of coordinates to extract 3 random neurons from the MLIsample
    maxY = len(MLIsample)-1
    maxX = len(MLIsample[0])-1
    y1, x1 = randrange(0, maxY), randrange(0, maxX)
    while 1: # to be sure that neuron 2 != neuron 1
        y2, x2 = randrange(0, maxY), randrange(0, maxX)
        if y2!=y1 or x2!=x1:
            break
    while 1: # to be sure that neuron 3 != neuron 1 nor neuron 2
        y3, x3 = randrange(0, maxY), randrange(0, maxX)
        if y3 != y1 or x3 != x1:
            if y3!=y2 or x3!=x2:
                break
    ls = [(y1, x1),(y2, x2),(y3, x3)]

    # consider them as nodes and create edges between them if they have non zero weights
    for i in ls:
        MLIgraph.add_node(MLIsample[i[0]][i[1]][1])
        for j in ls:
            w = SynapseObj.w[MLIsample[i[0]][i[1]][1], MLIsample[j[0]][j[1]][1]].tolist() # the weight is non zero only if there is a chemical synapse which is directed
            if w!=[] and w!=[0.0]:
                MLIgraph.add_edge(MLIsample[i[0]][i[1]][1], MLIsample[j[0]][j[1]][1], weight=1) # directed graph DiGraph() : the edge is directed
    # print('y1 :',y1,'x1 :',x1,'y2 :',y2,'x2 :',x2,'y3 :',y3,'x3 :',x3)
    return MLIgraph

def Nsample_sampling__Ntriads_patchings__triadsAnalysis(Nsample, Ntriads, SynapseObj, MLItable, strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100,
                                                        Dis003=65.0 / 173, Dis012=46.0 / 173, Dis021C=8.0 / 173, Dis021D=18.0 / 173,
                                                        Dis021U=10.0 / 173, Dis030C=0.0 / 173, Dis030T=13.0 / 173, Dis102=4.0 / 173,
                                                        Dis111D=0.0 / 173, Dis111U=5.0 / 173, Dis120C=1.0 / 173, Dis120D=0.0 / 173,
                                                        Dis120U=3.0 / 173, Dis201=0.0 / 173, Dis210=0.0 / 173, Dis300=0.0 / 173, PRINT=False):

    AimDis = {'003': Dis003, '012': Dis012, '021C': Dis021C, '021D': Dis021D,
              '021U': Dis021U, '030C': Dis030C, '030T': Dis030T, '102': Dis102,
              '111D': Dis111D, '111U': Dis111U, '120C': Dis120C, '120D': Dis120D,
              '120U': Dis120U, '201': Dis201, '210': Dis210, '300': Dis300}

    ### generate N "patching" experiments, each time analyse their triads patterns of connectivity, gather those analyses in a dictionnary in order to ;ake a histogram
    triadsAnalysis = {}
    memoryDic={} # dictionary whose values are the triad graphs dictionaries without doubles
    memory = {}  # dictionary whose values are the triad graphs without doubles

    ## sampling and patching
    for samp in range(Nsample):
        MLIsample = sample_MLI(MLItable, strategy, coordUnit, Ysizeum, Xsizeum)
        for i in range(Ntriads):
            MLIgraph = patch_MLI(MLIsample, SynapseObj)
            # Si une meme triade est generee deux fois, meme si les 3 neurones le sont dans des ordres differents chaque fois, les donnees s'organisent par cle dans le dico.
            # Et comme une meme cle appelee deux fois aura toujours la meme valeur (les connections netre les neurones ne changent pas),
            # deux graphs avec les memes neurones seront egaux.
            if MLIgraph.adj in memoryDic.values():
                pass
                #print(MLIgraph.adj)
                #print('Triad already encountered. Triadic census not added to final analysis.')
                #print('\n')
            else:
                memoryDic[samp]=MLIgraph.adj # on ajoute la triade au sein des valeurs du dic memory, avec comme cle l'index de l'experience de patching
                memory[samp]=MLIgraph
                if PRINT==True :
                    print('triad sampled :', MLIgraph.adj, end = " -- ")
                    print(triadic_census(MLIgraph))
                    print('\n')
    ## Analysis of the patched graphs
    for MLI_graph in memory.values():
        triadAnalysis = triadic_census(MLI_graph)
        for k,v in triadAnalysis.items(): # On ajoute l'analyse de cette triade a l'analyse finale
            triadsAnalysis[k] = triadsAnalysis.get(k, 0) + v # v est soit egal a 1 soit egal a 0
    ActDis={} # Actual distribution normalized with len(memory)
    for key in triadsAnalysis:
        ActDis[key]=float(triadsAnalysis[key])/len(memory)


    ## Comparison to the aimed distribution

    n = 0
    meanDiffSum = 0
    for pat in ActDis:
        PrevVal, AimVal = ActDis[pat], AimDis[pat]
        Prev_Aim_diff = PrevVal - AimVal
        meanDiffSum += sqrt((Prev_Aim_diff) ** 2)
        n += 1
    meanDiff = float(meanDiffSum) / n


    ### Create a graph from the synapse object and plot it
    #NxGraph = BrSynapseToNxGraph(SynapseObj, MLItable)

    ### Calculate the probability of connection between MLI which have been "patched"
    # thanks to "memoryDic", the dictionnary which doesn't gather triads in double (they can randomly be generated twice). So len(memory) is slightly < than N
    ###### print('Probability of connection :', end='')
    Nc = 0
    for i in memoryDic.values():  # memory values : MLI.adj
        for j in i.values():  # MLI.adj values : {node2 : {'weight':1}} if there is a connection, {} if there is no connection
            Nc += len(j)  # 0 if {}, 1 if {neuronIndex: 'weight'=1}, 2 if 2 connections...
    Npairs=len(memory)*3
    Pc=float(Nc)/(2*Npairs)
    ##### print(Pc)

    return (triadsAnalysis, memory, Pc, meanDiff, AimDis, ActDis)


def triadsPatterns_distribution_corrector(Nsample, Ntriads, SynapseObj, MLItable,
                                             TRY=5, prev_aim_diff=0.01, prev_aim_diff_out=0.003,
                                             strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100,
                                             Dis003=65.0 / 173, Dis012=46.0 / 173, Dis021C=8.0 / 173,
                                             Dis021D=18.0 / 173,
                                             Dis021U=10.0 / 173, Dis030C=0.0 / 173, Dis030T=13.0 / 173,
                                             Dis102=4.0 / 173,
                                             Dis111D=0.0 / 173, Dis111U=5.0 / 173, Dis120C=1.0 / 173, Dis120D=0.0 / 173,
                                             Dis120U=3.0 / 173, Dis201=0.0 / 173, Dis210=0.0 / 173, Dis300=0.0 / 173, iteration=0, previous_newConnectionsStore=0, figsDirectory='/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/personnalCon_noPKJ/figures/motifs distributions/duringRun'):


    '''This function allows you to correct a previous triads patterns distribution memoryDis (a dictionnary returned by the function Nsample_sampling__Ntriads_patchings__triadsAnalysis)
     by "flipping" some connections, in order to make our distribution stick to experimental data.
    It has the structure of the function Nsample_sampling__Ntriads_patchings__triadsAnalysis(),
    but the difference is that at each triad sampling ('patching') from a giver square sample,
    it analyses the triad pattern and changes it into a other one if needed. Changes it by mofifying the DiGraph... But also
    modifying the neuronal network replacing a synapse by a new one ! So use this function CAREFULLY.

    It runs until the mean difference between the frequences of "aimed" (experimental for instance) and "previous" pattern distribution (meanDiff) reaches less than "prev_aim_diff".
     For each pattern, there is a flip probability P... which has the default value of 1.

     Nsample is the number of squares of MLIs cut into MLItable, Ntriads is the number of triads patched in one sample (Sarahs patched 4 neurons per sample > it makes 4 triads. We can also stay at one.)
    '''

    #### Generation des triades stockees dans memory et de leurs dictionnaires respectifs dans memoryDis :
    triadsTuple = Nsample_sampling__Ntriads_patchings__triadsAnalysis(Nsample, Ntriads, SynapseObj, MLItable, strategy,
                                                                      coordUnit, Ysizeum,
                                                                      Xsizeum)  # returns the tuple (triadsAnalysis, memory)
    memory = triadsTuple[1]  # dictionary whose values are the triad graphs without doubles
    InitialMemory = len(memory)
    print("Number of triads in the sample used for triadic motifs distribution correction : ", InitialMemory)

    #### Creation des dictionnaires de distributions et utilisation de ces derniers pour caracteriser les patterns a augmenter ou baisser
    ### dico des distributions de la derniere generation de Nsample*Ntriads triades a partir du modele
    memoryDis = triadsTuple[0]
    for PrevKey in memoryDis:
        memoryDis[PrevKey] = float(memoryDis[PrevKey]) / (len(memory))  # turns the int values in proportion values in order to compare to the experiment
    ### dico des distributions visees (par default : distributions experimentales pour 173 triades)
    AimDis = {'003': Dis003, '012': Dis012, '021C': Dis021C, '021D': Dis021D,
              '021U': Dis021U, '030C': Dis030C, '030T': Dis030T, '102': Dis102,
              '111D': Dis111D, '111U': Dis111U, '120C': Dis120C, '120D': Dis120D,
              '120U': Dis120U, '201': Dis201, '210': Dis210, '300': Dis300}
    ### Caracterisation initiale des patterns (trop, trop peu) en comparant les deux dicos + declaration de la difference moyenne
    ## Liste des patterns a baisser :
    decPat = []
    ## Liste des index de patterns a augmenter :
    incPat = []
    ## Remplissage :
    n = 0
    meanDiffSum = 0
    for pat in memoryDis:
        PrevVal, AimVal = memoryDis[pat], AimDis[pat]
        Prev_Aim_diff = PrevVal - AimVal
        if Prev_Aim_diff > prev_aim_diff:
            decPat.append(pat)
        elif Prev_Aim_diff < -prev_aim_diff:
            incPat.append(pat)
        meanDiffSum += abs(Prev_Aim_diff)
        n += 1
    meanDiff = float(meanDiffSum) / n

    #### Initialisation du storage des connections ajoutees ou enlevees :
    # We add a variable to store the connections added or removed to change "unswitchable" patterns (not changed after 30 switches) despite all (see particular treatment line 468).
    # It can be >0 (more connections have been added than removed and there is a difference compared to the original network) or <0 (reverse).
    # After having screened the graphs of memory, we need to re-screen them in order to empty newConnectionsStore until 0 by dealing the connections to any patterns.
    newConnectionsStore = previous_newConnectionsStore
    # Indication du nombre de connections par pattern, pour compter le nombre de con ajoutees/retirees a chaque fois :
    patternNumCon = {'003': 0, '012': 1, '021C': 2, '021D': 2,
              '021U': 2, '030C': 3, '030T': 3, '102': 2,
              '111D': 3, '111U': 3, '120C': 4, '120D': 4,
              '120U': 4, '201': 4, '210': 5, '300': 6}

    #### Boucle : Try essais de redistribution des patterns
    Try = 1
    while 1:
        meanDiffCheck = float(meanDiff)
        memoryCheck = memory.copy()
        print("\n\n\n\n\nNew memory graphs screen, Try number : ", Try)
        print("The actual pattern distribution is :")
        print(memoryDis)  # memoryDis is updated at each try line 305
        print("The aimed experimental pattern distribution is :")
        print(AimDis)
        print("Patterns to decrease : ", decPat)
        print("Patterns to increase : ", incPat)
        print("The mean patterns distribution difference is :")
        print(meanDiff)
        #EOOnum = 0
        #for blabla in memory.values():
        #    blablaAnalysis = triadic_census(blabla)
        #    EOOnum += blablaAnalysis['300']  # either 0 or 1
        #print("number of 300 at the beginning : ", EOOnum, end = " ")
        #print("in a number of samples of : ", len(memory))


        ### Screen des graphs de memory : si son pattern est "trop present" = dans decPat, on change la connectivite du graph jusqu'a obtenir un pattern caracterise comme "trop peu present" = dans incPat
        ### > on change alors la connectivite du reseau BRIAN pour appliquer le nouveau pattern au reseau en amont du graph
        STEP = 0
        for step in range(len(memory)):
            STEP+=1
            MLI_graphKey = memory.items()[step][0] # So the list in which MLI_graph is taken is updated as often as memory is updated
            MLI_graph_real = memory.items()[step][1]
            MLI_graph = MLI_graph_real.copy()

            ## analyse individuelle des graphs de memory
            triadAnalysis = triadic_census(MLI_graph)
            print("\nIteration no "+str(iteration)+", Try no "+str(Try)+", Graph tested : ", MLI_graph.adj, " = graph no ", STEP, "among ", len(memory), ".")
            for Pat in triadAnalysis:
                if triadAnalysis[Pat] == 1:
                    triadPatPre = Pat
            print(" Graph pattern : ", triadPatPre)

            ## Tentative de switch pour modifier le graph si necessaire
            if triadPatPre in decPat:
                # if a pattern to lower is this triad pattern  ####### NOTE : no need to create the elif situation, because if some patern is underrepresented it means that an other one is too much represented...
                print(">pattern to lower !")
                SUCCESS=0

                # On tente le switch
                switchesTrials = 0
                while 1:  # We play with the pattern as soon as it has been found to be in decPat

                    # Generation des index de l'ancienne et de la nouvelle synapse potentielles
                    KEY1 = randrange(0,
                                     3)  # Choose random keys between 0 and 2 > keys of the triad graph (the 3 neurons indexes) to try every connection within the graph
                    while 1:
                        KEY2 = randrange(0, 3)
                        if KEY2 != KEY1:  # neurons are never connected to themselves
                            break
                    while 1:
                        KEY3 = randrange(0, 3)
                        while 1:
                            KEY4 = randrange(0, 3)
                            if KEY4 != KEY3:  # neurons are never connected to themselves
                                break
                        if (KEY3, KEY4) != (KEY1, KEY2):
                            break
                    i = MLI_graph.adj.keys()[KEY1]  # i=presynaptic previous synapse neuron index
                    j = MLI_graph.adj.keys()[KEY2]  # j=postsynaptic previous synapse neuron index
                    i1 = MLI_graph.adj.keys()[KEY3]  # i1=presynaptic new synapse neuron index
                    j1 = MLI_graph.adj.keys()[KEY4]  # j1=postynaptic new synapse neuron index
                    print(">>Is there a synapse from ", i, " to ", j, " AND no one from ", i1, " to ", j1, " ?")

                    # Try to change connections >> try to "switch" : if there is a connection between i and j and no one between j and i, replace it by one between i1 and j1. The goal is to keep the same total number of connections.
                    if j in MLI_graph[i].keys() and j1 not in MLI_graph[i1].keys():
                        print(">>Yes ! Edge switched : edge ", i, ">", j, " to edge ", i1, ">", j1, ".", end=" -->> ")
                        # erase this connection
                        MLI_graph.remove_edge(i, j)
                        # create a new one in the opposite way
                        MLI_graph.add_edge(i1, j1, weight=1)
                        # switchesMemory.append((i, j, i1, j1))  # we need to memorize all the flips made in a triad until it reached an upperPat pattern : as many flips have to be done in BRIAN network, not only the last one !
                        triadAnalysis = triadic_census(MLI_graph)
                        for Pat in triadAnalysis:
                            if triadAnalysis[Pat] == 1:
                                triadPatPost = Pat
                        print("New graph : ", MLI_graph.adj, end=" / ")
                        print("New pattern : ", triadPatPost)
                        if triadPatPost in incPat and triadAnalysis[triadPatPost] == 1:
                            # Si le nouveau graph a un pattern dont on veut augmenter la frequence,
                            print(">>>pattern obtained and increased :", triadPatPost)
                            # changer la connection du reseau neuronal BRIAN
                            BRIANswitchesCon = []
                            BRIANswitchesDisCon = []
                            for I in MLI_graph.adj:  # For the new MLI_graph
                                for J in MLI_graph[I]:  # <=> if J is one of I keys <=> if there is an edge from I to J
                                    if SynapseObj.w[I, J].tolist() == [] or SynapseObj.w[I, J].tolist() == [0.0]:  # Only if the synapse is not already created
                                        BRIANswitchesCon.append((I, J))
                                        if SynapseObj.w[I, J].tolist() == []:
                                            SynapseObj[I, J] = True
                                        SynapseObj.w[I, J] = 'rand()*1'  # w_mli_mli=1.0 in the notebook
                                    else:
                                        if SynapseObj.w[I, J].tolist() != [] and SynapseObj.w[I, J].tolist() != [0.0]:  # Only if there is a synapse to erase
                                            BRIANswitchesDisCon.append((I, J))
                                            SynapseObj.w[I, J] = '0.'  # resets to 0 weights
                            print("BRIAN network changes achieved : ", end="")
                            for BRIANswitch in BRIANswitchesCon:
                                print(" #connected -> w[", BRIANswitch[0], ",", BRIANswitch[1], "] = ",
                                      SynapseObj.w[BRIANswitch[0], BRIANswitch[1]].tolist(), end=" ")
                            for BRIANswitch in BRIANswitchesDisCon:
                                print(" #disconnected -> w[", BRIANswitch[0], ",", BRIANswitch[1], "] = ",
                                      SynapseObj.w[BRIANswitch[0], BRIANswitch[1]].tolist())

                            # Time to go to next graph, successful switch !
                            SUCCESS+=1
                            break

                    # If no successful switch after 30 trials, we consider that the pattern cannot be changed with the same number of connections (ex : pattern 003, 012, 300...).
                    # It is an "unswitchable" pattern.

                    switchesTrials += 1
                    if switchesTrials > 30:
                        print("Unswitchable pattern ; particular treatment --->>>")
                        break

                # si unswitchable, on tente le add_remove
                add_rem_Trials = 0
                #trans_newConnectionsStore = newConnectionsStore
                while 1:  # equivalent to the while 1 of line 38
                    # For the "unswitchables" patterns :
                    # We then authorize it to add or remove ONE connection from the original pattern. This difference of connections from the original pattern is then stored in newConnectionsPool.
                    # To do so we re-write the algorithm "try to switch", with a difference : there is only one condition not two :
                    # if j in MLI_graph[i].keys() and j1 not in MLI_graph[i1].keys() becomes either [if j in MLI_graph[i].keys()] or [if j not in MLI_graph[i].keys()].
                    # We add a random value "rand" in order to balance a little bit more the number of connections added and removed,
                    # compared to if it was only due
                    if switchesTrials <= 30:
                        break  # This loop only runs when we have reached an "unswitchable" pattern, which by definition a pattern not changed after 30 switch trials

                    rand = random.random()
                    KEY1 = randrange(0,
                                     3)  # Choose random keys between 0 and 2 > keys of the triad graph (the 3 neurons indexes) to try every connection within the graph
                    while 1:
                        KEY2 = randrange(0, 3)
                        if KEY2 != KEY1:  # neurons are never connected to themselves
                            break
                    i = MLI_graph.adj.keys()[KEY1]  # i=presynaptic previous synapse neuron index
                    j = MLI_graph.adj.keys()[KEY2]  # j=postsynaptic previous synapse neuron index
                    print(">>Is there a OR no synapse from ", i, " to ", j, " ?")

                    flag = 0
                    if rand <= 0.5 and j not in MLI_graph[i].keys():
                        print(">>Yes ! Edge added : edge ", i, ">", j, ".", end=" -->> ")
                        ## authorize to add a connection without removing an other one
                        #trans_newConnectionsStore += 1
                        # create a new connection
                        MLI_graph.add_edge(i, j, weight=1)
                        # unswitchable_add_switchesMemory.append((i,j))
                        flag += 1

                    elif rand > 0.5 and j in MLI_graph[i].keys():
                        print(">>Yes ! Edge removed : edge ", i, ">", j, ".", end=" -->> ")
                        ## authorize to remove a connection without adding an other one
                        #trans_newConnectionsStore -= 1
                        # erase this connection
                        MLI_graph.remove_edge(i, j)
                        # unswitchable_rem_switchesMemory.append((i,j))
                        flag += 1

                    if flag != 0:  # <=> Only if at least one connection has been modified in the graph
                        triadAnalysis = triadic_census(MLI_graph)
                        for Pat in triadAnalysis:
                            if triadAnalysis[Pat] == 1:
                                triadPatPost = Pat
                        print("New graph : ", MLI_graph.adj, end=" / ")
                        print("New pattern : ", triadPatPost)
                        if triadPatPost in incPat and triadAnalysis[triadPatPost] == 1:
                            #newConnectionsStore = trans_newConnectionsStore # ssi SUCCESS, evidemment !
                            #ConnectionsAdded_Removed = patternNumCon[triadPatPost]-patternNumCon[triadPatPre]  # >0 if connections added, <0 if removed
                            #newConnectionsStore+=ConnectionsAdded_Removed
                            # Si le nouveau graph a un pattern dont on veut augmenter la frequence,
                            print(">>>pattern obtained and increased :", triadPatPost)
                            # changer la connection du reseau neuronal BRIAN
                            BRIANswitchesCon = []
                            BRIANswitchesDisCon = []
                            for I in MLI_graph.adj:  # For the new MLI_graph
                                for J in MLI_graph.adj:
                                    if J in MLI_graph[I]:  # <=> if J is one of I keys <=> if there is an edge from I to J
                                        if SynapseObj.w[I, J].tolist() == [] or SynapseObj.w[I, J].tolist() == [0.0]:  # Only if the synapse is not already created
                                            newConnectionsStore+=1
                                            BRIANswitchesCon.append((I, J))
                                            if SynapseObj.w[I, J].tolist() == []:
                                                SynapseObj[I, J] = True
                                            SynapseObj.w[I, J] = 'rand()*1'  # w_mli_mli=1.0 in the notebook
                                    else:
                                        if SynapseObj.w[I, J].tolist() != [] and SynapseObj.w[I, J].tolist() != [0.0]:  # Only if there is a synapse to erase
                                            newConnectionsStore-=1
                                            BRIANswitchesDisCon.append((I, J))
                                            SynapseObj.w[I, J] = '0.'  # resets to 0 weights
                            print("BRIAN network changes achieved : ", end="")
                            for BRIANswitch in BRIANswitchesCon:
                                print(" #connected -> w[", BRIANswitch[0], ",", BRIANswitch[1], "] = ",
                                      SynapseObj.w[BRIANswitch[0], BRIANswitch[1]].tolist(), end=" ")
                            for BRIANswitch in BRIANswitchesDisCon:
                                print(" #disconnected -> w[", BRIANswitch[0], ",", BRIANswitch[1], "] = ",
                                      SynapseObj.w[BRIANswitch[0], BRIANswitch[1]].tolist())
                            print("BRIAN network changes achieved for this unswitchable pattern ! << Price : newConnectionsStore", newConnectionsStore)

                            # Time to go to next graph, successful switch !
                            SUCCESS += 1
                            break

                    add_rem_Trials += 1
                    if add_rem_Trials > 60:
                        print("60 trials for the unswitchable pattern ; no pattern must be left in the incPat list ? Didn't have enough trials, just bad luck ?")
                        break

                # si le switch ou le aadd_remove est un succes, on update memory, memoryDis, incPat-decPat
                if SUCCESS != 0:

                    # On update l'ENSEMBLE de memory a partir du nouveau SynapseObj (pour que l'affection des graphs adjacents soient prise en compte)

                    #EOOnum=0
                    #for blabla in memory.values():
                    #    blablaAnalysis = triadic_census(blabla)
                    #    EOOnum += blablaAnalysis['300']  # either 0 or 1
                    #print("number of 300 before update : ", EOOnum, end = " ")
                    #print("in a number of samples of : ", len(memory))

                    for key_oldGraph, oldGraph in memory.items():
                        newGraph = nx.DiGraph()

                        ls=[]
                        for potentialNeuronPre in oldGraph.adj:
                            newGraph.add_node(potentialNeuronPre)
                            for potentialNeuronPost in oldGraph.adj:
                                ls.append((potentialNeuronPre, potentialNeuronPost)) # list of 6 potential edges to use later to screen SynapseObj

                        for synapseTuple in ls:
                            w = SynapseObj.w[synapseTuple[0], synapseTuple[1]].tolist()  # the weight is non zero only if there is a chemical synapse (which is directed)
                            if w != [] and w != [0.0]:
                                newGraph.add_edge(synapseTuple[0], synapseTuple[1], weight=1)  # directed graph DiGraph() : the edge is directed

                        memory[key_oldGraph] = newGraph

                    #EOOnum=0
                    #for blabla in memory.values():
                    #    blablaAnalysis = triadic_census(blabla)
                    #    EOOnum += blablaAnalysis['300'] # either 0 or 1
                    #print("number of 300 after update : ", EOOnum, end = " ")
                    #print("in a number of samples of : ", len(memory))

                    #memory[MLI_graphKey] = MLI_graph
                    if len(memory) != InitialMemory:
                        raise ValueError("ERROR : GRAPHS MYSTERIOUSLY ADDED TO MEMORY : len(memory)=", len(memory),
                                         " but Nsample*Ntriads=", str(Nsample * Ntriads))
                    print("memory storing graphs updated.")

                    # On update memoryDis (triadsAnalysis) a partir de memory
                    memoryDis = {}
                    for graph in memory.values():
                        triadAnalysis = triadic_census(graph)
                        for k, v in triadAnalysis.items():  # On ajoute l'analyse de cette triade a l'analyse finale
                            memoryDis[k] = memoryDis.get(k, 0) + v  # v est soit egal a 1 soit egal a 0
                    for PrevKey in memoryDis:
                        memoryDis[PrevKey] = float(memoryDis[PrevKey]) / (len(memory))
                    print("memoryDis storing patterns distribution updated : ", memoryDis)

                    # On update decPat et incPat en utilisant la nouvelle memoryDis
                    decPat = []
                    incPat = []
                    n = 0
                    meanDiffSum = 0
                    for pat in memoryDis:
                        PrevVal, AimVal = memoryDis[pat], AimDis[pat]
                        Prev_Aim_diff = PrevVal - AimVal
                        if Prev_Aim_diff > prev_aim_diff:
                            decPat.append(pat)
                        elif Prev_Aim_diff < -prev_aim_diff:
                            incPat.append(pat)
                        meanDiffSum += sqrt((Prev_Aim_diff) ** 2)
                        n += 1
                    meanDiff = float(meanDiffSum) / n
                    print("List of patterns to decrease updated :")
                    print(decPat)
                    print("List of patterns to increase updated :")
                    print(incPat)
                    print("Updated meanDiff : ", meanDiff)



        ### time to empty newConnectionsStore !

        newConnectionStoreFLAG = 0
        maxTrials = 0
        while 1:

            ## Conditions de sortie de boucle
            maxTrials += 1
            if maxTrials > 5:
                break
            if newConnectionStoreFLAG == 1:  # <=> newConnectionsStore == 0 (we use a flag in order to check after each graph if newConnectionsStore==0, not only after each screen
                break
            if decPat == []:
                break
            print("\n\n\n\n >>> Let's empty newConnectionsStore !", end=" --- ")
            print("newConnectionsStore : ", newConnectionsStore)

            ## Screen all the memory items one more time if newConnectionsStore hasn't reached 0...
            STEPBIS=0
            for stepBIS in range(len(memory)):
                STEPBIS+=1
                MLI_graphKey = memory.items()[stepBIS][0]
                MLI_graph_real = memory.items()[stepBIS][1]
                MLI_graph = MLI_graph_real.copy()
                if abs(newConnectionsStore) == 0:  # we check it after each graph (+/- 1 connection usually, so we won't miss the moment when newConnectionsStore reaches 0)
                    newConnectionStoreFLAG = 1
                    break  # out from the for loop (stop screening memory motifs)

                ## analyse individuelle des graphs de memory
                triadAnalysis = triadic_census(MLI_graph)
                print("\nIteration no "+str(iteration)+", Try no "+str(Try)+", 2nd screening : Graph tested : ", MLI_graph.adj, " = graph no ", STEPBIS, "among ", len(memory), ".")
                for Pat in triadAnalysis:
                    if triadAnalysis[Pat] == 1:
                        triadPatPreBIS = Pat
                print(" Graph pattern : ", triadPatPreBIS)
                print("decPat : ", decPat)
                print("incPat : ", incPat)
                print("memoryDis : ", memoryDis)
                print("newConnectionsStore :", newConnectionsStore)

                ## Tentative de switch pour modifier le graph si necessaire
                if triadPatPreBIS in decPat:
                    # if a pattern to lower is this triad pattern  ####### NOTE : no need to create the elif situation, because if some patern is underrepresented it means that an other one is too much represented...
                    print(">Empty NCS screening : pattern to lower !")

                    emptying_newConnectionsStore_tries = 0
                    #transitory_newConnectionsStore = newConnectionsStore  # a transitory variable which will be mofified during the adds/removes and will be compared to the unchanged newConnectionsStore
                    while 1:  # The same while loop as the second while loop of the first for loop... But which considers all kinds of patterns, not only the unswotchables ones !
                        emptying_newConnectionsStore_tries += 1
                        if emptying_newConnectionsStore_tries > 30:
                            print("This pattern cannot help to empty newConnectionsStore (ex : newConnectionsStore>0 can't be emptied by modifying 003")
                            break
                        KEY1 = randrange(0,3)  # Choose random keys between 0 and 2 > keys of the triad graph (the 3 neurons indexes) to try every connection within the graph
                        while 1:
                            KEY2 = randrange(0, 3)
                            if KEY2 != KEY1:  # neurons are never connected to themselves
                                break
                        i = MLI_graph.adj.keys()[KEY1]  # i=presynaptic previous synapse neuron index
                        j = MLI_graph.adj.keys()[KEY2]  # j=postsynaptic previous synapse neuron index
                        print(">>Empty NCS screening : Is there a OR no synapse from ", i, " to ", j, " ?")

                        transitory_newConnectionsStore=newConnectionsStore
                        flag = 0
                        if newConnectionsStore < 0 and j not in MLI_graph[i].keys():
                            print(">>Yes ! Edge added : edge ", i, ">", j, ".", end=" -->> ")
                            ## authorize to add a connection without removing an other one
                            transitory_newConnectionsStore += 1
                            # create a new connection
                            MLI_graph.add_edge(i, j, weight=1)
                            # unswitchable_add_switchesMemory.append((i, j))
                            flag += 1
                        elif newConnectionsStore > 0 and j in MLI_graph[i].keys():
                            print(">>Yes ! Edge removed : edge ", i, ">", j, ".", end=" -->> ")
                            ## authorize to remove a connection without adding an other one
                            transitory_newConnectionsStore -= 1
                            # erase this connection
                            MLI_graph.remove_edge(i, j)
                            # unswitchable_rem_switchesMemory.append((i, j))
                            flag += 1

                        if flag != 0 and abs(transitory_newConnectionsStore) < abs(newConnectionsStore):  # <=> Only if newConnectionsStore gets closer to 0 (It is optionnal, just a check+makes the code more readable, since you can only add connections of it is <0 and vice versa...)
                            triadAnalysis = triadic_census(MLI_graph)
                            for Pat in triadAnalysis:
                                if triadAnalysis[Pat] == 1:
                                    triadPatPostBIS = Pat
                            print("New graph : ", MLI_graph.adj, end=" / ")
                            print("New pattern : ", triadPatPostBIS)
                            if triadPatPostBIS in incPat:
                                #newConnectionsStore = transitory_newConnectionsStore  # newConnectionsStore is effectively reduced !
                                #ConnectionsAdded_Removed = patternNumCon[triadPatPostBIS] - patternNumCon[triadPatPreBIS]  # >0 if connections added, <0 if removed
                                #newConnectionsStore += ConnectionsAdded_Removed
                                # Si le nouveau graph a un pattern dont on veut augmenter la frequence,
                                print(">>>pattern obtained and increased :", triadPatPostBIS)
                                # changer la connection du reseau neuronal BRIAN
                                BRIANswitchesCon = []
                                BRIANswitchesDisCon = []
                                for I in MLI_graph.adj:  # For the new MLI_graph
                                    for J in MLI_graph.adj:
                                        if J in MLI_graph[I]:  # <=> if J is one of I keys <=> if there is an edge from I to J
                                            if SynapseObj.w[I, J].tolist() == [] or SynapseObj.w[I, J].tolist() == [0.0]:  # Only if the synapse is not already created
                                                newConnectionsStore += 1
                                                BRIANswitchesCon.append((I, J))
                                                if SynapseObj.w[I, J].tolist() == []:
                                                    SynapseObj[I, J] = True
                                                SynapseObj.w[I, J] = 'rand()*1'  # w_mli_mli=1.0 in the notebook
                                        else:
                                            if SynapseObj.w[I, J].tolist() != [] and SynapseObj.w[I, J].tolist() != [0.0]:  # Only if there is a synapse to erase
                                                newConnectionsStore -= 1
                                                BRIANswitchesDisCon.append((I, J))
                                                SynapseObj.w[I, J] = '0.'  # resets to 0 weights
                                print("BRIAN network changes achieved : ", end="")
                                for BRIANswitch in BRIANswitchesCon:
                                    print(" #connected -> w[", BRIANswitch[0], ",", BRIANswitch[1], "] = ",
                                          SynapseObj.w[BRIANswitch[0], BRIANswitch[1]].tolist(), end=" ")
                                for BRIANswitch in BRIANswitchesDisCon:
                                    print(" #disconnected -> w[", BRIANswitch[0], ",", BRIANswitch[1], "] = ",
                                          SynapseObj.w[BRIANswitch[0], BRIANswitch[1]].tolist())
                                print("BRIAN network changes achieved during the second screening : newConnectionsStore=",
                                      newConnectionsStore, " and is closer to 0 !")

                                # On update l'ENSEMBLE de memory a partir du nouveau SynapseObj (pour que l'affection des graphs adjacents lors de la modification de ce graph soient prise en compte)

                                #EOOnum = 0
                                #for blabla in memory.values():
                                #    blablaAnalysis = triadic_census(blabla)
                                #    EOOnum += blablaAnalysis['300']  # either 0 or 1
                                #print("number of 300 before update : ", EOOnum, end = " ")
                                #print("in a number of samples of : ", len(memory))

                                for key_oldGraph, oldGraph in memory.items():
                                    newGraph = nx.DiGraph()

                                    ls = []
                                    for potentialNeuronPre in oldGraph.adj:
                                        newGraph.add_node(potentialNeuronPre)
                                        for potentialNeuronPost in oldGraph.adj:
                                            ls.append((potentialNeuronPre,
                                                       potentialNeuronPost))  # list of 6 potential edges to use later to screen SynapseObj

                                    for synapseTuple in ls:
                                        w = SynapseObj.w[synapseTuple[0], synapseTuple[
                                            1]].tolist()  # the weight is non zero only if there is a chemical synapse (which is directed)
                                        if w != [] and w != [0.0]:
                                            newGraph.add_edge(synapseTuple[0], synapseTuple[1],
                                                              weight=1)  # directed graph DiGraph() : the edge is directed

                                    memory[key_oldGraph] = newGraph

                                #EOOnum = 0
                                #for blabla in memory.values():
                                #    blablaAnalysis = triadic_census(blabla)
                                #    EOOnum += blablaAnalysis['300']  # either 0 or 1
                                #print("number of 300 after update : ", EOOnum, end = " ")
                                #print("in a number of samples of : ", len(memory))


                                #memory[MLI_graphKey] = MLI_graph
                                if len(memory) != InitialMemory:
                                    raise ValueError("ERROR : GRAPHS MYSTERIOUSLY ADDED TO MEMORY : len(memory)=",
                                                     len(memory), " but Nsample*Ntriads=", str(Nsample * Ntriads))
                                print("memory storing graphs updated.")

                                # On update memoryDis (triadsAnalysis) a partir de memory
                                memoryDis = {}
                                for graph in memory.values():
                                    triadAnalysis = triadic_census(graph)
                                    for k, v in triadAnalysis.items():  # On ajoute l'analyse de cette triade a l'analyse finale
                                        memoryDis[k] = memoryDis.get(k, 0) + v  # v est soit egal a 1 soit egal a 0
                                for PrevKey in memoryDis:
                                    memoryDis[PrevKey] = float(memoryDis[PrevKey]) / (len(memory))
                                print("memoryDis storing patterns distribution updated : ", memoryDis)

                                # On update decPat et incPat en utilisant la nouvelle memoryDis
                                decPat = []
                                incPat = []
                                n = 0
                                meanDiffSum = 0
                                for pat in memoryDis:
                                    PrevVal, AimVal = memoryDis[pat], AimDis[pat]
                                    Prev_Aim_diff = PrevVal - AimVal
                                    if Prev_Aim_diff > prev_aim_diff:
                                        decPat.append(pat)
                                    elif Prev_Aim_diff < -prev_aim_diff:
                                        incPat.append(pat)
                                    meanDiffSum += sqrt((Prev_Aim_diff) ** 2)
                                    n += 1
                                meanDiff = float(meanDiffSum) / n
                                print("List of patterns to decrease updated :")
                                print(decPat)
                                print("List of patterns to increase updated :")
                                print(incPat)
                                print("meanDiff : ", meanDiff)

                                # Time to go to next graph, successful switch !
                                break
            maxTrials += 1
            if maxTrials > 50:
                break


        ### to force meanDiff to converge : if meanDiff is bigger at this trial (Try) than after the last one, we reset it with the one obtained after the last one
        print("\n\n\n\n\n\nmeanDiff before Try "+str(Try)+" : ", meanDiffCheck, " -- meanDiff after Try "+str(Try)+" : ", meanDiff, ".")
        if meanDiff<=meanDiffCheck:
            print("meanDiff reduced :)")
            ### Plot a histogram at each successful Try
            patList = []
            patValList = []
            for pat in memoryDis:
                patList.append(pat)
                patValList.append((AimDis[pat] * 100, memoryDis[pat] * 100))
            patDataFrame = pd.DataFrame(data=patValList, index=patList,
                                        columns=['Aimed dist. -- ' + str(meanDiff), 'Network dist.'])
            ax = patDataFrame.plot.bar()  # s is an instance of Series
            fig = ax.get_figure()
            fig.savefig(str(figsDirectory) + '/duringRun/wholeDebugNet_duringCorrection_Iteration' + str(
                iteration) + '_Try' + str(Try) + '.jpeg')
            pass
        elif meanDiff>meanDiffCheck:
            print("meanDiff increased : reset memory to the one before this Try.")
            memory = memoryCheck.copy()
            meanDiff = float(meanDiffCheck)

        ### Definition des sorties de boucle
        Try += 1
        if Try > TRY:
            break
        if meanDiff < prev_aim_diff_out:  # meandiff updated after each graph switch. We have to be more demanding with prev_aim_diff_out (threshold under which we stop iterating TOPALG, difference ~ 0.005) than with prev_aim_diff (threshold under which a pattern is attributed to dec or inc lists : much bigger difference ~ 0.02, 0.05)
            break


    #### Conclusion
    print("We can find in this corrected sample ", len(memory), " graphs.")
    print("\n\n\n\nThe final (+/- modified) pattern distribution is : ", memoryDis,
          ". It has a mean difference from the aimed distribution of : ", meanDiff)
    if newConnectionsStore > 0:
        print("\nThere is a final number of connections added of : ", newConnectionsStore)
    elif newConnectionsStore < 0:
        print("\nThere is a final number of connections removed of : ", newConnectionsStore)
    elif newConnectionsStore == 0:
        print("\nThere is finally no connections neither added nor removed ! newConnectionsStore = ", newConnectionsStore)
    print("The modified synapse object contains : ", len(SynapseObj))

    #### Calculate the probability of connection between MLI which have been "patched"
    #### thanks to "memoryDis", the dictionnary which doesn't gather triads in double (they can randomly be generated twice). So len(memory) is slightly < than N
    Nc = 0
    for i in memory.values():  # memory values : MLI.adj
        for j in i.adj.values():  # MLI.adj values : {node2 : {'weight':1}} if there is a connection, {} if there is no connection
            Nc += len(j)  # 0 if {}, 1 if {neuronIndex: 'weight'=1}, 2 if 2 connections...
    Npairs = len(memory) * 3
    Pc = float(Nc) / (2 * Npairs)
    print('Total number of connections in memory = Nc : ', Nc)
    print('Probability of connection according to memory:', Pc)

    Nc_SynapseObj=0 # calculate the number of connections thanks to the BRIAN synapse object
    MLItable_neuronsN=len(MLItable)*len(MLItable[0])
    for i in range(MLItable_neuronsN):
        for j in range(MLItable_neuronsN):
            if SynapseObj.w[i,j].tolist()!=[] and SynapseObj.w[i,j].tolist()!=[0.0]:
                Nc_SynapseObj+=1
    print('Total number of connections in S_MLI_MLI (neither [] nor [0.0] = Nc_SynapseObj : ', Nc_SynapseObj)
    print('NB : len(SynapseObj), with [0.0] weights = ', len(SynapseObj))


    return (SynapseObj, memoryDis, AimDis, meanDiff, newConnectionsStore)  # like triadsAnalysis


def iterative_corrections(Niterations, Ncorrection, NtriadsCorrection, Nanalysis, NtriadsAnalysis, SynapseObj_real,
                                                            MLItable,
                                                            TRY=10, prev_aim_diff=0.003, prev_aim_diff_out=0.003,
                                                            strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100,
                                                            Dis003=65.0 / 173, Dis012=46.0 / 173, Dis021C=8.0 / 173,
                                                            Dis021D=18.0 / 173,
                                                            Dis021U=10.0 / 173, Dis030C=0.0 / 173, Dis030T=13.0 / 173,
                                                            Dis102=4.0 / 173,
                                                            Dis111D=0.0 / 173, Dis111U=5.0 / 173, Dis120C=1.0 / 173,
                                                            Dis120D=0.0 / 173,
                                                            Dis120U=3.0 / 173, Dis201=0.0 / 173, Dis210=0.0 / 173,
                                                            Dis300=0.0 / 173, figsDirectory='/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/personnalCon_noPKJ/figures/motifs distributions/wholeNet_before_afterCorrection-sampledNet_afterRun'):
    '''
    Scheme of the function output :
    There are Niteration iterations. For 1 iteration,
    - prints the "pre" analysis of the whole network, made on Nanalysis*NtriadsAnalysis triads.
    - makes the correction, made on Ncorrection*NtriadsCorrection triads.
    - makes the "post" analysis of the whole network, in the same way. prints the result only if the meanDiff is lower.
    For the second iteration and all the next ones, the "pre" analyis of the whole network is the "post" analysis of the last iteration.

    So the output is a bunch of graphs :
    - One preAnalysis graph of the whole network
    - Several postAnalysis graphs of the whole network, only if the meanDiff has been reduced.
    - Several forAnalysis graphs of the sample used for the connection, after each iteration.
    + All the analysis of the sample used for the correction after each try for each iteration.
    '''

    meanDiff_list = []
    meanDiff_irrelevant = []
    iteration = 0
    prev_nCS = 0
    while iteration < Niterations: # < and not <= because iteration+=1 at the beginning of the loop for a better presentation

        iteration+=1

        ########################## WHOLE NETWORK PRE-CORRECTION ANALYSIS (SynapseObj_real is used) ##################################

        if iteration == 1:  # There is a preCorrection analysis only for the first iteration
            # Connectivity readout as a table

            chdir('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/personnalCon_noPKJ/connectivity')
            write_synapses_table(SynapseObj_real, 'personalCon_noPKJ_patternDisPreCorrection' + str(iteration) + '.txt')

            # Connectivity readout as a histogram
            Analysis_preCorrection = Nsample_sampling__Ntriads_patchings__triadsAnalysis(Nsample=Nanalysis, Ntriads=NtriadsAnalysis,
                                                                                         SynapseObj=SynapseObj_real, MLItable=MLItable,
                                                                                         strategy=strategy,
                                                                coordUnit=coordUnit, Ysizeum=Ysizeum, Xsizeum=Xsizeum,
                                                                Dis003=Dis003, Dis012=Dis012, Dis021C=Dis021C,
                                                                Dis021D=Dis021D,
                                                                Dis021U=Dis021U, Dis030C=Dis030C,
                                                                Dis030T=Dis030T, Dis102=Dis102,
                                                                Dis111D=Dis111D, Dis111U=Dis111U, Dis120C=Dis120C,
                                                                Dis120D=Dis120D,
                                                                Dis120U=Dis120U, Dis201=Dis201, Dis210=Dis210,
                                                                Dis300=Dis300, PRINT=False)

            print("\n\n\n\n Analysis pre correction number : ", iteration)
            print("\n\n\n\nActual analysis : ", Analysis_preCorrection[5])
            print("Aimed analysis : ", Analysis_preCorrection[4])
            print("\n >> Mean of differences between them : ", Analysis_preCorrection[3])
            print("Connection Probability :", Analysis_preCorrection[2])
            print("Number of synapses in S_MLI_MLI :", len(SynapseObj_real))
            preCor_meanDiff = Analysis_preCorrection[3]
            patList = []
            patValList = []
            for pat in Analysis_preCorrection[0]:
                patList.append(pat)
                patValList.append((Analysis_preCorrection[4][pat] * 100, Analysis_preCorrection[5][pat] * 100))

            patDataFrame = pd.DataFrame(data=patValList, index=patList,
                                        columns=['Aimed dist. -- ' + str(preCor_meanDiff), 'Network dist.'])

            ax = patDataFrame.plot.bar()  # s is an instance of Series
            fig = ax.get_figure()
            fig.savefig(
                str(figsDirectory)+'/wholeNet_iteration' + str(
                    iteration) + '_preCorrection.jpeg')

            meanDiff_list.append((0, Analysis_preCorrection[3]))

        elif iteration > 1:  # for the next ones, we use the post-analysis result
            pass #preCor_meanDiff remains the same



        ########################## CORRECTION : from SynapseObj_real, to SynapseObj ##################################

        # SynapseObj_real is given as argument : it corresponds either to the SynapseObj modified by the previous iteration (SynapseObj_real has taken the value of SynapseObj), either by one before (it has not for the last iteration)
        correctorOutput = triadsPatterns_distribution_corrector(Nsample=Ncorrection, Ntriads=NtriadsCorrection, SynapseObj=SynapseObj_real,
                                                                MLItable=MLItable,
                                                                TRY=TRY, prev_aim_diff=prev_aim_diff, prev_aim_diff_out=prev_aim_diff_out,
                                                                strategy=strategy, coordUnit=coordUnit, Ysizeum=Ysizeum, Xsizeum=Xsizeum,
                                                                Dis003=Dis003, Dis012=Dis012, Dis021C=Dis021C,
                                                                Dis021D=Dis021D,
                                                                Dis021U=Dis021U, Dis030C=Dis030C, Dis030T=Dis030T,
                                                                Dis102=Dis102,
                                                                Dis111D=Dis111D, Dis111U=Dis111U, Dis120C=Dis120C,
                                                                Dis120D=Dis120D,
                                                                Dis120U=Dis120U, Dis201=Dis201, Dis210=Dis210,
                                                                Dis300=Dis300, iteration=iteration, previous_newConnectionsStore=prev_nCS, figsDirectory=figsDirectory)
        # low prev_aim_diff : plus il est faible, plus les listes incPat/decPat resteront pleines facilement, ce qui assure que le programme continuera de tourner plus longtemps.

        # this resets SynapseObj at each iteration
        SynapseObj = correctorOutput[0] # we still didn't modify SynapseObj_real : cf. all the mistakes I had done, functions do NOT modify global values, unless you say 'global variable = function output' afterwards

        #plot
        endDistribution = correctorOutput[1]
        aimedDistribution = correctorOutput[2]

        patList = []
        patValList = []
        for pat in endDistribution:
            patList.append(pat)
            patValList.append((aimedDistribution[pat] * 100, endDistribution[pat] * 100))

        patDataFrame = pd.DataFrame(data=patValList, index=patList,
                                        columns=['Aimed dist. -- ' + str(correctorOutput[3]), 'Sample dist.'])

        ax = patDataFrame.plot.bar()  # s is an instance of Series
        fig = ax.get_figure()
        fig.savefig(str(figsDirectory)+'/sampledNet_iteration'+str(iteration)+'_forCorrection.jpeg')

        ########################## WHOLE NETWORK POST-CORRECTION ANALYSIS (SynapseObj is used) ##################################

        # Connectivity readout


        chdir('/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/personnalCon_noPKJ/connectivity')
        write_synapses_table(SynapseObj, 'personalCon_noPKJ_patternDisPostCorrection'+str(iteration)+'.txt')

        Analysis_postCorrection = Nsample_sampling__Ntriads_patchings__triadsAnalysis(Nsample=Nanalysis,
                                                                                     Ntriads=NtriadsAnalysis,
                                                                                     SynapseObj=SynapseObj,
                                                                                     MLItable=MLItable,
                                                                                     strategy=strategy,
                                                                                     coordUnit=coordUnit,
                                                                                     Ysizeum=Ysizeum, Xsizeum=Xsizeum,
                                                                                     Dis003=Dis003, Dis012=Dis012,
                                                                                     Dis021C=Dis021C,
                                                                                     Dis021D=Dis021D,
                                                                                     Dis021U=Dis021U, Dis030C=Dis030C,
                                                                                     Dis030T=Dis030T, Dis102=Dis102,
                                                                                     Dis111D=Dis111D, Dis111U=Dis111U,
                                                                                     Dis120C=Dis120C,
                                                                                     Dis120D=Dis120D,
                                                                                     Dis120U=Dis120U, Dis201=Dis201,
                                                                                     Dis210=Dis210,
                                                                                     Dis300=Dis300, PRINT=False)

        print("\n\n\n\n Analysis post correction number : ", iteration)
        print("\n\n\n\nActual analysis : ", Analysis_postCorrection[5])
        print("Aimed analysis : ", Analysis_postCorrection[4])
        print("\n >> Mean of differences between them : ", Analysis_postCorrection[3])
        print("Connection Probability :", Analysis_postCorrection[2])
        print("Number of synapses in S_MLI_MLI :", len(SynapseObj))

        meanDiff_list.append((iteration, Analysis_postCorrection[3]))

        ########################## Choose to effectively modify or not the SynapseObj_real as SynapseObj has been modified ##################################
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nIteration : "+str(iteration)+" >> Analysis_postCorrection : ", Analysis_postCorrection[3], ", Analysis_preCorrection : ", preCor_meanDiff, ".")
        if Analysis_postCorrection[3] < preCor_meanDiff:  # compare the meanDiffs
            print("\n\nApparently the meanDiff after the iteration " + str(iteration) + " is lower than after the last one.")
            preCor_meanDiff = Analysis_postCorrection[3]  # the new lower wholeNetwork meanDiff is Analysis_postCorrection[3]
            prev_nCS += correctorOutput[4]  # update the next iteration initial newConnectionsStore
            # need to make a COPY of SynapseObj in SynapseObj_real... The only difference between them is their connectivity ! Only need to copy the connectivity.
            neuronsNumber = len(MLItable) * len(MLItable[0])
            for neuronIndex1 in range(neuronsNumber):
                for neuronIndex2 in range(neuronsNumber):
                    if SynapseObj.w[neuronIndex1, neuronIndex2].tolist() != [] and SynapseObj.w[neuronIndex1, neuronIndex2].tolist() != [0.0]:
                        if SynapseObj_real.w[neuronIndex1, neuronIndex2].tolist() == []:
                            SynapseObj_real[neuronIndex1, neuronIndex2] = True
                        SynapseObj_real.w[neuronIndex1, neuronIndex2] = str(SynapseObj.w[neuronIndex1, neuronIndex2].tolist()[0])
                    elif SynapseObj.w[neuronIndex1, neuronIndex2].tolist() == [] or SynapseObj.w[neuronIndex1, neuronIndex2].tolist() == [0.0]:
                        SynapseObj_real.w[neuronIndex1, neuronIndex2] = '0.0'

            # Plotting only if SynapseObj_real updated
            patList = []
            patValList = []
            for pat in Analysis_postCorrection[0]:
                patList.append(pat)
                patValList.append((Analysis_postCorrection[4][pat] * 100, Analysis_postCorrection[5][pat] * 100))

            patDataFrame = pd.DataFrame(data=patValList, index=patList,
                                        columns=['Aimed dist. -- ' + str(Analysis_postCorrection[3]), 'Network dist.'])

            ax = patDataFrame.plot.bar()  # s is an instance of Series
            fig = ax.get_figure()
            fig.savefig(
                str(figsDirectory)+'/wholeNet_iteration' + str(
                    iteration) + '_postCorrection.jpeg')

        else:
            print("\n\nApparently the meanDiff after the iteration "+str(iteration)+" is higher than after the last one. No updating of the synapse object, no plotting of this postCorrection analysis.")
            meanDiff_irrelevant.append((iteration, Analysis_postCorrection[3]))

    # Plot the mean Diff evolution through iterations

    meanDiffEvolutionDataFrame = pd.DataFrame(
                data=[meanDiff_list[i][1] for i in range(len(meanDiff_list))],
                index=[meanDiff_list[i][0] for i in range(len(meanDiff_list))],
                columns=['meanDiff'])

    ax = meanDiffEvolutionDataFrame.plot()  # s is an instance of Series
    for anot in meanDiff_irrelevant:
        ax.annotate('No updating', xy=(anot[0], anot[1]), xytext=(anot[0]+0.5, anot[1]+0.003),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=4))
    fig = ax.get_figure()
    fig.savefig(str(figsDirectory)+'/meanDiff_iterationsEvolution.jpeg')

    return (SynapseObj_real, meanDiff_list)

def triadsPatterns_distribution_creator(Nsample, Ntriads, SynapseObj, MLItable,
                                              TRY=5, prev_aim_diff=0.01,
                                              strategy='rdm', coordUnit='um', Ysizeum=100, Xsizeum=100,
                                              Dis003=65.0 / 173, Dis012=46.0 / 173, Dis021C=8.0 / 173,
                                              Dis021D=18.0 / 173,
                                              Dis021U=10.0 / 173, Dis030C=0.0 / 173, Dis030T=13.0 / 173,
                                              Dis102=4.0 / 173,
                                              Dis111D=0.0 / 173, Dis111U=5.0 / 173, Dis120C=1.0 / 173,
                                              Dis120D=0.0 / 173,
                                              Dis120U=3.0 / 173, Dis201=0.0 / 173, Dis210=0.0 / 173, Dis300=0.0 / 173,
                                              iteration=0, previous_newConnectionsStore=0):


    '''The goal of this function is to connect a BRIAN network (here MLItable) with a given pattern distribution.

    Derivated from the function "triadsPatterns_distribution_corrector", which was only correcting an already-connected
    BRIANnetwork without changing its total number of connections (thanks to newConnectionsStore which had to be close
    to 0 after the correction).

    This time, we start with 0 connections and the final newConnectionStore is free to reach the number it needs : the
    only constraint is the final pattern distribution.

    Nsample must be big enough to cover the maximum number of neurons (the goal is to connect the whole network). Ntriads=1.

    TEST : the maximal final newConnectionStore must be at
    (len(MLItable)*len(MLItable[0]))*(len(MLItable)*len(MLItable[0])-1)/2*2(connections bidirectionnelles)
     <=> all the neurons are connected (for 160 neurons : 25440)
    '''

    #### Generation des triades stockees dans memory et de leurs dictionnaires respectifs dans memoryDis :
    triadsTuple = Nsample_sampling__Ntriads_patchings__triadsAnalysis(Nsample, Ntriads, SynapseObj, MLItable, strategy,
                                                                      coordUnit, Ysizeum,
                                                                      Xsizeum)  # returns the tuple (triadsAnalysis, memory)
    memory = triadsTuple[1]  # dictionary whose values are the triad graphs without doubles
    InitialMemory = len(memory)
    print("Number of triads in the sample used for triadic motifs distribution correction : ", InitialMemory)

    #### Creation des dictionnaires de distributions et utilisation de ces derniers pour caracteriser les patterns a augmenter ou baisser
    ### dico des distributions de la derniere generation de Nsample*Ntriads triades a partir du modele
    memoryDis = triadsTuple[0]
    for PrevKey in memoryDis:
        memoryDis[PrevKey] = float(memoryDis[PrevKey]) / (
        len(memory))  # turns the int values in proportion values in order to compare to the experiment
    ### dico des distributions visees (par default : distributions experimentales pour 173 triades)
    AimDis = {'003': Dis003, '012': Dis012, '021C': Dis021C, '021D': Dis021D,
              '021U': Dis021U, '030C': Dis030C, '030T': Dis030T, '102': Dis102,
              '111D': Dis111D, '111U': Dis111U, '120C': Dis120C, '120D': Dis120D,
              '120U': Dis120U, '201': Dis201, '210': Dis210, '300': Dis300}
    ### Caracterisation initiale des patterns (trop, trop peu) en comparant les deux dicos + declaration de la difference moyenne
    ## Liste des patterns a baisser :
    decPat = []
    ## Liste des index de patterns a augmenter :
    incPat = []
    ## Remplissage :
    n = 0
    meanDiffSum = 0
    for pat in memoryDis:
        PrevVal, AimVal = memoryDis[pat], AimDis[pat]
        Prev_Aim_diff = PrevVal - AimVal
        if Prev_Aim_diff > prev_aim_diff:
            decPat.append(pat)
        elif Prev_Aim_diff < -prev_aim_diff:
            incPat.append(pat)
        meanDiffSum += sqrt((Prev_Aim_diff) ** 2)
        n += 1
    meanDiff = float(meanDiffSum) / n

    #### Initialisation du storage des connections ajoutees ou enlevees :
    # We add a variable to store the connections added or removed to change "unswitchable" patterns (not changed after 30 switches) despite all (see particular treatment line 468).
    # It can be >0 (more connections have been added than removed and there is a difference compared to the original network) or <0 (reverse).
    # After having screened the graphs of memory, we need to re-screen them in order to empty newConnectionsStore until 0 by dealing the connections to any patterns.
    newConnectionsStore = previous_newConnectionsStore
    # Indication du nombre de connections par pattern, pour compter le nombre de con ajoutees/retirees a chaque fois :
    patternNumCon = {'003': 0, '012': 1, '021C': 2, '021D': 2,
                     '021U': 2, '030C': 3, '030T': 3, '102': 2,
                     '111D': 3, '111U': 3, '120C': 4, '120D': 4,
                     '120U': 4, '201': 4, '210': 5, '300': 6}

    #### Boucle : Try essais de redistribution des patterns
    Try = 1
    while 1:
        meanDiffCheck = float(meanDiff)
        memoryCheck = memory.copy()
        print("\n\n\n\n\nNew memory graphs screen, Try number : ", Try)
        print("The actual pattern distribution is :")
        print(memoryDis)  # memoryDis is updated at each try line 305
        print("The aimed experimental pattern distribution is :")
        print(AimDis)
        print("Patterns to decrease : ", decPat)
        print("Patterns to increase : ", incPat)
        print("The mean patterns distribution difference is :")
        print(meanDiff)
        # EOOnum = 0
        # for blabla in memory.values():
        #    blablaAnalysis = triadic_census(blabla)
        #    EOOnum += blablaAnalysis['300']  # either 0 or 1
        # print("number of 300 at the beginning : ", EOOnum, end = " ")
        # print("in a number of samples of : ", len(memory))


        ### Screen des graphs de memory : si son pattern est "trop present" = dans decPat, on change la connectivite du graph jusqu'a obtenir un pattern caracterise comme "trop peu present" = dans incPat
        ### > on change alors la connectivite du reseau BRIAN pour appliquer le nouveau pattern au reseau en amont du graph
        STEP = 0
        for step in range(len(memory)):
            STEP += 1
            MLI_graphKey = memory.items()[step][
                0]  # So the list in which MLI_graph is taken is updated as often as memory is updated
            MLI_graph_real = memory.items()[step][1]
            MLI_graph = MLI_graph_real.copy()

            ## analyse individuelle des graphs de memory
            triadAnalysis = triadic_census(MLI_graph)
            print("\nIteration no " + str(iteration) + ", Try no " + str(Try) + ", Graph tested : ", MLI_graph.adj,
                  " = graph no ", STEP, "among ", len(memory), ".")
            for Pat in triadAnalysis:
                if triadAnalysis[Pat] == 1:
                    triadPatPre = Pat
            print(" Graph pattern : ", triadPatPre)

            ## Tentative de switch pour modifier le graph si necessaire
            if triadPatPre in decPat:
                # if a pattern to lower is this triad pattern  ####### NOTE : no need to create the elif situation, because if some patern is underrepresented it means that an other one is too much represented...
                print(">pattern to lower !")
                SUCCESS = 0

                # On tente le switch
                switchesTrials = 0
                while 1:  # We play with the pattern as soon as it has been found to be in decPat

                    # Generation des index de l'ancienne et de la nouvelle synapse potentielles
                    KEY1 = randrange(0,
                                     3)  # Choose random keys between 0 and 2 > keys of the triad graph (the 3 neurons indexes) to try every connection within the graph
                    while 1:
                        KEY2 = randrange(0, 3)
                        if KEY2 != KEY1:  # neurons are never connected to themselves
                            break
                    while 1:
                        KEY3 = randrange(0, 3)
                        while 1:
                            KEY4 = randrange(0, 3)
                            if KEY4 != KEY3:  # neurons are never connected to themselves
                                break
                        if (KEY3, KEY4) != (KEY1, KEY2):
                            break
                    i = MLI_graph.adj.keys()[KEY1]  # i=presynaptic previous synapse neuron index
                    j = MLI_graph.adj.keys()[KEY2]  # j=postsynaptic previous synapse neuron index
                    i1 = MLI_graph.adj.keys()[KEY3]  # i1=presynaptic new synapse neuron index
                    j1 = MLI_graph.adj.keys()[KEY4]  # j1=postynaptic new synapse neuron index
                    print(">>Is there a synapse from ", i, " to ", j, " AND no one from ", i1, " to ", j1, " ?")

                    # Try to change connections >> try to "switch" : if there is a connection between i and j and no one between j and i, replace it by one between i1 and j1. The goal is to keep the same total number of connections.
                    if j in MLI_graph[i].keys() and j1 not in MLI_graph[i1].keys():
                        print(">>Yes ! Edge switched : edge ", i, ">", j, " to edge ", i1, ">", j1, ".", end=" -->> ")
                        # erase this connection
                        MLI_graph.remove_edge(i, j)
                        # create a new one in the opposite way
                        MLI_graph.add_edge(i1, j1, weight=1)
                        # switchesMemory.append((i, j, i1, j1))  # we need to memorize all the flips made in a triad until it reached an upperPat pattern : as many flips have to be done in BRIAN network, not only the last one !
                        triadAnalysis = triadic_census(MLI_graph)
                        for Pat in triadAnalysis:
                            if triadAnalysis[Pat] == 1:
                                triadPatPost = Pat
                        print("New graph : ", MLI_graph.adj, end=" / ")
                        print("New pattern : ", triadPatPost)
                        if triadPatPost in incPat and triadAnalysis[triadPatPost] == 1:
                            # Si le nouveau graph a un pattern dont on veut augmenter la frequence,
                            print(">>>pattern obtained and increased :", triadPatPost)
                            # changer la connection du reseau neuronal BRIAN
                            BRIANswitchesCon = []
                            BRIANswitchesDisCon = []
                            for I in MLI_graph.adj:  # For the new MLI_graph
                                for J in MLI_graph[I]:  # <=> if J is one of I keys <=> if there is an edge from I to J
                                    if SynapseObj.w[I, J].tolist() == [] or SynapseObj.w[I, J].tolist() == [
                                        0.0]:  # Only if the synapse is not already created
                                        BRIANswitchesCon.append((I, J))
                                        if SynapseObj.w[I, J].tolist() == []:
                                            SynapseObj[I, J] = True
                                        SynapseObj.w[I, J] = 'rand()*1'  # w_mli_mli=1.0 in the notebook
                                    else:
                                        if SynapseObj.w[I, J].tolist() != [] and SynapseObj.w[I, J].tolist() != [
                                            0.0]:  # Only if there is a synapse to erase
                                            BRIANswitchesDisCon.append((I, J))
                                            SynapseObj.w[I, J] = '0.'  # resets to 0 weights
                            print("BRIAN network changes achieved : ", end="")
                            for BRIANswitch in BRIANswitchesCon:
                                print(" #connected -> w[", BRIANswitch[0], ",", BRIANswitch[1], "] = ",
                                      SynapseObj.w[BRIANswitch[0], BRIANswitch[1]].tolist(), end=" ")
                            for BRIANswitch in BRIANswitchesDisCon:
                                print(" #disconnected -> w[", BRIANswitch[0], ",", BRIANswitch[1], "] = ",
                                      SynapseObj.w[BRIANswitch[0], BRIANswitch[1]].tolist())

                            # Time to go to next graph, successful switch !
                            SUCCESS += 1
                            break

                    # If no successful switch after 30 trials, we consider that the pattern cannot be changed with the same number of connections (ex : pattern 003, 012, 300...).
                    # It is an "unswitchable" pattern.

                    switchesTrials += 1
                    if switchesTrials > 30:
                        print("Unswitchable pattern ; particular treatment --->>>")
                        break

                # si unswitchable, on tente le add_remove
                add_rem_Trials = 0
                # trans_newConnectionsStore = newConnectionsStore
                while 1:  # equivalent to the while 1 of line 38
                    # For the "unswitchables" patterns :
                    # We then authorize it to add or remove ONE connection from the original pattern. This difference of connections from the original pattern is then stored in newConnectionsPool.
                    # To do so we re-write the algorithm "try to switch", with a difference : there is only one condition not two :
                    # if j in MLI_graph[i].keys() and j1 not in MLI_graph[i1].keys() becomes either [if j in MLI_graph[i].keys()] or [if j not in MLI_graph[i].keys()].
                    # We add a random value "rand" in order to balance a little bit more the number of connections added and removed,
                    # compared to if it was only due
                    if switchesTrials <= 30:
                        break  # This loop only runs when we have reached an "unswitchable" pattern, which by definition a pattern not changed after 30 switch trials

                    rand = random.random()
                    KEY1 = randrange(0,
                                     3)  # Choose random keys between 0 and 2 > keys of the triad graph (the 3 neurons indexes) to try every connection within the graph
                    while 1:
                        KEY2 = randrange(0, 3)
                        if KEY2 != KEY1:  # neurons are never connected to themselves
                            break
                    i = MLI_graph.adj.keys()[KEY1]  # i=presynaptic previous synapse neuron index
                    j = MLI_graph.adj.keys()[KEY2]  # j=postsynaptic previous synapse neuron index
                    print(">>Is there a OR no synapse from ", i, " to ", j, " ?")

                    flag = 0
                    if rand <= 0.5 and j not in MLI_graph[i].keys():
                        print(">>Yes ! Edge added : edge ", i, ">", j, ".", end=" -->> ")
                        ## authorize to add a connection without removing an other one
                        # trans_newConnectionsStore += 1
                        # create a new connection
                        MLI_graph.add_edge(i, j, weight=1)
                        # unswitchable_add_switchesMemory.append((i,j))
                        flag += 1

                    elif rand > 0.5 and j in MLI_graph[i].keys():
                        print(">>Yes ! Edge removed : edge ", i, ">", j, ".", end=" -->> ")
                        ## authorize to remove a connection without adding an other one
                        # trans_newConnectionsStore -= 1
                        # erase this connection
                        MLI_graph.remove_edge(i, j)
                        # unswitchable_rem_switchesMemory.append((i,j))
                        flag += 1

                    if flag != 0:  # <=> Only if at least one connection has been modified in the graph
                        triadAnalysis = triadic_census(MLI_graph)
                        for Pat in triadAnalysis:
                            if triadAnalysis[Pat] == 1:
                                triadPatPost = Pat
                        print("New graph : ", MLI_graph.adj, end=" / ")
                        print("New pattern : ", triadPatPost)
                        if triadPatPost in incPat and triadAnalysis[triadPatPost] == 1:
                            # newConnectionsStore = trans_newConnectionsStore # ssi SUCCESS, evidemment !
                            # ConnectionsAdded_Removed = patternNumCon[triadPatPost]-patternNumCon[triadPatPre]  # >0 if connections added, <0 if removed
                            # newConnectionsStore+=ConnectionsAdded_Removed
                            # Si le nouveau graph a un pattern dont on veut augmenter la frequence,
                            print(">>>pattern obtained and increased :", triadPatPost)
                            # changer la connection du reseau neuronal BRIAN
                            BRIANswitchesCon = []
                            BRIANswitchesDisCon = []
                            for I in MLI_graph.adj:  # For the new MLI_graph
                                for J in MLI_graph.adj:
                                    if J in MLI_graph[I]:  # <=> if J is one of I keys <=> if there is an edge from I to J
                                        if SynapseObj.w[I, J].tolist() == [] or SynapseObj.w[I, J].tolist() == [
                                            0.0]:  # Only if the synapse is not already created
                                            newConnectionsStore += 1
                                            BRIANswitchesCon.append((I, J))
                                            if SynapseObj.w[I, J].tolist() == []:
                                                SynapseObj[I, J] = True
                                            SynapseObj.w[I, J] = 'rand()*1'  # w_mli_mli=1.0 in the notebook
                                    else:
                                        if SynapseObj.w[I, J].tolist() != [] and SynapseObj.w[I, J].tolist() != [
                                            0.0]:  # Only if there is a synapse to erase
                                            newConnectionsStore -= 1
                                            BRIANswitchesDisCon.append((I, J))
                                            SynapseObj.w[I, J] = '0.'  # resets to 0 weights
                            print("BRIAN network changes achieved : ", end="")
                            for BRIANswitch in BRIANswitchesCon:
                                print(" #connected -> w[", BRIANswitch[0], ",", BRIANswitch[1], "] = ",
                                      SynapseObj.w[BRIANswitch[0], BRIANswitch[1]].tolist(), end=" ")
                            for BRIANswitch in BRIANswitchesDisCon:
                                print(" #disconnected -> w[", BRIANswitch[0], ",", BRIANswitch[1], "] = ",
                                      SynapseObj.w[BRIANswitch[0], BRIANswitch[1]].tolist())
                            print(
                                "BRIAN network changes achieved for this unswitchable pattern ! << Price : newConnectionsStore",
                                newConnectionsStore)

                            # Time to go to next graph, successful switch !
                            SUCCESS += 1
                            break

                    add_rem_Trials += 1
                    if add_rem_Trials > 60:
                        print(
                            "60 trials for the unswitchable pattern ; no pattern must be left in the incPat list ? Didn't have enough trials, just bad luck ?")
                        break

                # si le switch ou le add_remove est un succes, on update memory, memoryDis, incPat-decPat
                if SUCCESS != 0:

                    # On update l'ENSEMBLE de memory a partir du nouveau SynapseObj (pour que l'affection des graphs adjacents soient prise en compte)

                    # EOOnum=0
                    # for blabla in memory.values():
                    #    blablaAnalysis = triadic_census(blabla)
                    #    EOOnum += blablaAnalysis['300']  # either 0 or 1
                    # print("number of 300 before update : ", EOOnum, end = " ")
                    # print("in a number of samples of : ", len(memory))

                    for key_oldGraph, oldGraph in memory.items():
                        newGraph = nx.DiGraph()

                        ls = []
                        for potentialNeuronPre in oldGraph.adj:
                            newGraph.add_node(potentialNeuronPre)
                            for potentialNeuronPost in oldGraph.adj:
                                ls.append((potentialNeuronPre,
                                           potentialNeuronPost))  # list of 6 potential edges to use later to screen SynapseObj

                        for synapseTuple in ls:
                            w = SynapseObj.w[synapseTuple[0], synapseTuple[
                                1]].tolist()  # the weight is non zero only if there is a chemical synapse (which is directed)
                            if w != [] and w != [0.0]:
                                newGraph.add_edge(synapseTuple[0], synapseTuple[1],
                                                  weight=1)  # directed graph DiGraph() : the edge is directed

                        memory[key_oldGraph] = newGraph

                    # EOOnum=0
                    # for blabla in memory.values():
                    #    blablaAnalysis = triadic_census(blabla)
                    #    EOOnum += blablaAnalysis['300'] # either 0 or 1
                    # print("number of 300 after update : ", EOOnum, end = " ")
                    # print("in a number of samples of : ", len(memory))

                    # memory[MLI_graphKey] = MLI_graph
                    if len(memory) != InitialMemory:
                        raise ValueError("ERROR : GRAPHS MYSTERIOUSLY ADDED TO MEMORY : len(memory)=", len(memory),
                                         " but Nsample*Ntriads=", str(Nsample * Ntriads))
                    print("memory storing graphs updated.")

                    # On update memoryDis (triadsAnalysis) a partir de memory
                    memoryDis = {}
                    for graph in memory.values():
                        triadAnalysis = triadic_census(graph)
                        for k, v in triadAnalysis.items():  # On ajoute l'analyse de cette triade a l'analyse finale
                            memoryDis[k] = memoryDis.get(k, 0) + v  # v est soit egal a 1 soit egal a 0
                    for PrevKey in memoryDis:
                        memoryDis[PrevKey] = float(memoryDis[PrevKey]) / (len(memory))
                    print("memoryDis storing patterns distribution updated : ", memoryDis)

                    # On update decPat et incPat en utilisant la nouvelle memoryDis
                    decPat = []
                    incPat = []
                    n = 0
                    meanDiffSum = 0
                    for pat in memoryDis:
                        PrevVal, AimVal = memoryDis[pat], AimDis[pat]
                        Prev_Aim_diff = PrevVal - AimVal
                        if Prev_Aim_diff > prev_aim_diff:
                            decPat.append(pat)
                        elif Prev_Aim_diff < -prev_aim_diff:
                            incPat.append(pat)
                        meanDiffSum += sqrt((Prev_Aim_diff) ** 2)
                        n += 1
                    meanDiff = float(meanDiffSum) / n
                    print("List of patterns to decrease updated :")
                    print(decPat)
                    print("List of patterns to increase updated :")
                    print(incPat)
                    print("Updated meanDiff : ", meanDiff)


        ### to force meanDiff to converge : if meanDiff is bigger at this trial (Try) than after the last one, we reset it with the one obtained after the last one
        print("\n\n\n\n\n\nmeanDiff before Try " + str(Try) + " : ", meanDiffCheck,
              " -- meanDiff after Try " + str(Try) + " : ", meanDiff, ".")
        if meanDiff <= meanDiffCheck:
            print("meanDiff reduced :)")
            ### Plot a histogram at each successful Try
            patList = []
            patValList = []
            for pat in memoryDis:
                patList.append(pat)
                patValList.append((AimDis[pat] * 100, memoryDis[pat] * 100))
            patDataFrame = pd.DataFrame(data=patValList, index=patList,
                                        columns=['Aimed dist. -- ' + str(meanDiff), 'Network dist.'])
            ax = patDataFrame.plot.bar()  # s is an instance of Series
            fig = ax.get_figure()
            fig.savefig(
                '/Users/maximebeau/Desktop/Science/4_Stage_UCL_WIBR_HAUSSER/My_project/experiments_results/personnalCon_noPKJ/figures/motifs distributions/duringRun/wholeDebugNet_duringCorrection_Iteration' + str(
                    iteration) + '_Try' + str(Try) + '.jpeg')
            pass
        elif meanDiff > meanDiffCheck:
            print("meanDiff increased : reset memory to the one before this Try.")
            memory = memoryCheck.copy()
            meanDiff = float(meanDiffCheck)
        ### Definition des sorties de boucle
        Try += 1
        if Try > TRY:
            break
        if meanDiff < prev_aim_diff:  # meandiff updated after each graph switch
            break

    #### Conclusion
    print("We can find in this corrected sample ", len(memory), " graphs.")
    print("\n\n\n\nThe final (+/- modified) pattern distribution is : ", memoryDis,
          ". It has a mean difference from the aimed distribution of : ", meanDiff)
    if newConnectionsStore > 0:
        print("\nThere is a final number of connections added of : ", newConnectionsStore)
    elif newConnectionsStore < 0:
        print("\nThere is a final number of connections removed of : ", newConnectionsStore)
    elif newConnectionsStore == 0:
        print("\nThere is finally no connections neither added nor removed ! newConnectionsStore = ", newConnectionsStore)
    print("The modified synapse object contains : ", len(SynapseObj))

    #### Calculate the probability of connection between MLI which have been "patched"
    #### thanks to "memoryDis", the dictionnary which doesn't gather triads in double (they can randomly be generated twice). So len(memory) is slightly < than N
    Nc = 0
    for i in memory.values():  # memory values : MLI.adj
        for j in i.adj.values():  # MLI.adj values : {node2 : {'weight':1}} if there is a connection, {} if there is no connection
            Nc += len(j)  # 0 if {}, 1 if {neuronIndex: 'weight'=1}, 2 if 2 connections...
    Npairs = len(memory) * 3
    Pc = float(Nc) / (2 * Npairs)
    print('Total number of connections : ', Nc)
    print('Probability of connection :', Pc)

    return (SynapseObj, memoryDis, AimDis, meanDiff, newConnectionsStore)  # like triadsAnalysis
