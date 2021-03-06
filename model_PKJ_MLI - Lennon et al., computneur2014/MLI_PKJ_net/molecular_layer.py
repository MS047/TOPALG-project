from pylab import *
from brian import *
from abstract_neuron_group import *
from util import *

class MLIGroup(AbstractNeuronGroup):
    '''
    Group of Molecular Layer Interneurons (MLIs).
    
    This model assumes the physiological properties of basket and stellate 
    cells are similar enough to share the same model.
    
    Modeled as conductance-based leaky-integrate-and-fire neurons.
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
                 g_ahp_ = 50. * nsiemens,      # maximum after hyperpolarization conductance 100 used in paper -- derived to match Lachamp et al. (2009)
                 g_inh_ = 4. * nsiemens,       # maximum inhibitory conductance -- Carter and Regehr (2002)
                 g_ampa_ = 1.3 * nsiemens,     # maximum AMPAR mediated synaptic conductance -- Carter and Regehr (2002)
                 tau_ahp = 2.5 * msecond,      # AHP time constant -- derived to resemble Lachamp et al. (2009)
                 tau_inh = 4.6 * msecond,      # Inhbitory time constant -- Carter and Regehr (2002)
                 tau_ampa = .8 * msecond,      # AMPAR unitary EPSC time constant -- Carter and Regehr (2002) 
                 rand_V_init = True,
                 tau_adjust = True,
                 **kwargs):

        self.Vth, self.Cm, self.El, self.Eex, self.Einh, self.Eahp = Vth, Cm, El, Eex, Einh, Eahp
        self.gl, self.g_ampa_, self.g_inh_, self.g_ahp_ = gl, g_ampa_, g_inh_, g_ahp_
        if tau_adjust:
            dt = defaultclock.dt
            tau_ampa = adjust_tau(dt, tau_ampa)
            tau_inh = adjust_tau(dt, tau_inh)
            tau_ahp = adjust_tau(dt, tau_ahp)
        self.tau_ampa,  self.tau_inh, self.tau_ahp = tau_ampa, tau_inh, tau_ahp   
               
        self.eqns = Equations('''
        # Membrane equation
        dV/dt = 1/Cm*(-gl*(V-El)-g_ampa*(V-Eex)-g_ahp*(V-Eahp)-g_inh*(V-Einh) + I) : mV
        
        # After hyperpolarization
        dg_ahp/dt = -g_ahp/tau_ahp : nS
        
        # Glutamate
        dg_ampa/dt = -g_ampa/tau_ampa : nS
        
        # GABA
        dg_inh/dt = -g_inh/tau_inh : nS
        
        # Input current
        I : nA
        ''')
                
        super(MLIGroup, self).__init__(N, self.eqns,Vth,reset='g_ahp=g_ahp_',**kwargs)
        
        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)
            
        if self.clock.dt > .25*ms:
            warnings.warn('Clock for MLI group should be .25*ms for numerical stability')

    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
                  'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_, 'g_ahp_':self.g_ahp_, 
                  'tau_ampa':self.tau_ampa, 'tau_ahp':self.tau_ahp,'eqns':self.eqns
                  }
        return params
    
