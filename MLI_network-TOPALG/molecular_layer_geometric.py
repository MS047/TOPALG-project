from pylab import *
from brian import *
from molecular_layer import *
from util import *

# a new class of MLI neurons which includes their x, y coordinates and their indexes as an attribute
class Geometric_MLI(AbstractNeuronGroup):
    '''
    This group purpose is to allow instanciation of single MLIs with particular (x, y) coordinates in order to create a geometry-dependant network.
    '''
    
    def __init__(self,
                 N,
                 Vth = -53. * mvolt,           # Firing threshold, Volts -- Midtgaard (1992)
                 Cm = 14.6 * pfarad,           # Membrane capacitance -- Hausser and Clark (1997)
                 El = -68. * mvolt,            # leak reversal potential -- derived
                 Eex = 0. * mvolt,             # Excitatory reversal potential -- Carter and Regehr (2002)
                 Einh = -82. * mvolt,          # Inhibitory reversal potential -- Carter and Regehr (2002)
                 Eahp = -82. * mvolt,          # After hyperpolarization reversal potential -- derived to match Lachamp et al. (2009)
                 gl = 1.6 * nsiemens,          # maximum leak conductance -- derived from Hausser and Clark (1997)
                 g_inh_ = 4. * nsiemens,       # maximum inhibitory conductance -- Carter and Regehr (2002)
                 g_ahp_ = 50. * nsiemens,      # maximum after hyperpolarization conductance 100 used in paper -- derived to match Lachamp et al. (2009)
                 g_ampa_ = 1.3 * nsiemens,     # maximum AMPAR mediated synaptic conductance -- Carter and Regehr (2002)
                 tau_ahp = 2.5 * msecond,      # AHP time constant -- derived to resemble Lachamp et al. (2009)
                 tau_inh = 4.6 * msecond,      # Inhbitory time constant -- Carter and Regehr (2002)
                 tau_ampa = .8 * msecond,      # AMPAR unitary EPSC time constant -- Carter and Regehr (2002)
                 rand_V_init = True,
                 tau_adjust = True,
                 X=0,
                 Y=0,
                 INDEX=0,
                 **kwargs):
        # argument taken out : "g_ahp_ = 50. * nsiemens,      # maximum after hyperpolarization conductance 100 used in paper -- derived to match Lachamp et al. (2009)"
        self.Vth, self.Cm, self.El, self.Eex, self.Einh, self.Eahp = Vth, Cm, El, Eex, Einh, Eahp
        self.gl, self.g_ampa_, self.g_inh_, self.g_ahp_ = gl, g_ampa_, g_inh_, g_ahp_
        if tau_adjust:
            dt = defaultclock.dt
            tau_ampa = adjust_tau(dt, tau_ampa)
            tau_inh = adjust_tau(dt, tau_inh)
            tau_ahp = adjust_tau(dt, tau_ahp)
        self.tau_ampa,  self.tau_inh, self.tau_ahp = tau_ampa, tau_inh, tau_ahp
        
        self.x=X ### Geometric_MLI is all about those three ###
        self.y=Y
        self.index=INDEX
    
        self.eqns = Equations('''
        # Membrane equation
        dV/dt = 1/Cm*(-gl*(V-El)-g_ampa*(V-Eex)-g_ahp*(V-Eahp)-g_inh*(V-Einh) + I + Istim) : mV
        
        # After hyperpolarization
        dg_ahp/dt = -g_ahp/tau_ahp : nS
        
        # Glutamate
        dg_ampa/dt = -g_ampa/tau_ampa : nS
        
        # GABA
        dg_inh/dt = -g_inh/tau_inh : nS
        
        # Input current
        I : nA
        Istim : nA
        ''')
            
        super(Geometric_MLI, self).__init__(N, self.eqns,Vth,reset='g_ahp=g_ahp_',**kwargs) #appelle la methode constructeur de la classe dont Geometric_MLI herite, i.e. absract_NeuronGroup... Which has no constructor > calls NEuronGroup constructor !
        
        
        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)
        
        if self.clock.dt > .25*ms:
            warnings.warn('Clock for MLI group should be .25*ms for numerical stability')

#    def __repr__(self):
#        return 'oneMLI'
    
    def get_coordinates(self):
        return {'x':self.x, 'y':self.y}

    def get_index(self):
        return {'index':self.index}

    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
            'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_, 'g_ahp_':self.g_ahp_,
            'tau_ampa':self.tau_ampa, 'tau_ahp':self.tau_ahp,'eqns':self.eqns,
            '(x,y)':(self.x, self.y), 'x':self.x, 'y':self.y, 'index':self.index
            }
        return params


# a way to ordonate 1MLIsubgroups of an instance of this class, in a table > it is a 2D grid of neurons
def create_MLItable(GeometricGroup, N_MLI_geometric_OX, N_MLI_geometric_OY):
    '''
    GeometricGroup has to be a group of N_MLI_geometric_OX*N_MLI_geometric_OY neurons.
    
    -Creation of a table with N_MLI_geometric_OX "columns" and N_MLI_geometric_OY "raws"
    -Filling of this table with Geometric_MLI SUBGROUPS : the column and raw indexes are equivalent to their x and y coordinates
    -The table stores at [y][x] the tupke (MLIsubgroup of 1, idx) where idx is the neuron index (according to 
    BRIAN nomenclature i.e. the NeuronGroup[idx] neuron + the neuron recognized by the synapse object as idx.
    '''

    N_MLI=N_MLI_geometric_OX*N_MLI_geometric_OY
    
    MLItable=[]
    idx=0
    for i in range(N_MLI_geometric_OY):
        MLItable.append([]) # creates the "raws"
        for j in range(N_MLI_geometric_OX):
            MLItable[i].append((GeometricGroup.subgroup(1), idx)) # add as many subgroups as we want "columns"
            idx+=1
    print('MLItable created ! >> ', len(MLItable)*len(MLItable[0]), 'neurons stored.')

    return (MLItable, GeometricGroup)
