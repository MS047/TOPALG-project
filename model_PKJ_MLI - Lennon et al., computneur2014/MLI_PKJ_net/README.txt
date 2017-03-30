Here we describe the code behind the experiments.

abstract_neuron_group.py, molecular_layer.py and purkinje_layer.py contain "NeuronGroups" (in the BRIAN parlance) for
MLIs and PKJs.  They contain the physiological parameters which define the model.

connections.py contains functions that connect MLIs to PKJs and MLIs.

util.py and plotting_util.py contain utility functions that are referenced in the experiments and ipython notebooks.

./experiments - contains experiments which create a single MLI providing feedforward inhibition to a PKJ.  A number of
                trials are run and the membrane potential of the model neurons is measure.  MLI_PKJ_ISI_delay_rand_nS.py
                performs this test by randomly varying the MLI->PKJ maximum synpatic conductance.  The results are saved
                to disk to be later analyzed in MLI_PKJ_ISI_delay_analysis.ipynb

./parameter_search - contains scripts that run a search for the parameters of the random current or connection strengths
                     that result in ISI distributions closest to the data reported by Hausser and Clark (1997). These
                     also output data to disk to be later analyzed in the ipynbs in ./notebooks.

./notebooks - contains the iPython notebooks which analyze and plot the data seen in the final figures in the paper.
                - MLI_gamma_current_parameters.ipynb - analyzes the output of MLI_gamma_current_param_sweep.py and runs
                  a simulation of an MLI with these parameters.  Plots the data seen in the top row of figure 2 of the
                  paper.
                - PKJ_gamma_current_parameters.ipynb - same but for PKJs, top row Figure 3
                - MLI_PKJ_ISI_delay_analysis.ipynb - analyzes the data output by MLI_PKJ_ISI_delay.py and
                  MLI_PKJ_ISI_delay_rand_nS.py.  Plots Figure 5.
                - MLI_PKJ_ISI.ipynb - creates a network of MLIs and PKJs and simulates the activity.  Plots the bottom
                  panels of Figures 2,3, &4.
                - MLI_PKJ_mean_fr_hist.ipynb - randomly creates several networks as in MLI_PKJ_ISI.ipynb and collects
                  the mean firing rates of MLIs and PKJs to plot histograms.  This data is not in the paper.