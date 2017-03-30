from pylab import *
from brian import *
from util import *

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
def plot_ISI_histogram(ISI_monitor, spike_monitor, rate_monitor, xy, xytext, **plotargs):
    hist_plot(ISI_monitor, newfigure=False, **plotargs)
    mew, std = isi_mean_and_std(spike_monitor)
    s = 'rate = %0.1f Hz\nCV = %0.2f' % (mean(rate_monitor.rate),std/mew)
    annotate(s,xy=xy,xytext=xytext,xycoords='data',fontsize=24)
    tick_params(labelsize=20)
    xlabel('ISI (ms)', fontsize=20)
    ylabel('Count', fontsize=20)


def plot_neuron_ISI_histogram(spike_monitor, ind, ax=None, xy=(20,0), xytext=None, nbins=100, **plotargs):
    if ax is None:
        subplot(111)
    counts, bins, _ = ax.hist(diff(spike_monitor.spiketimes[ind])*1000, nbins, **plotargs)
    mew, std = isi_mean_and_std(spike_monitor,ind)
    s = 'rate = %0.1f Hz\nCV = %0.2f' % (1000/mew,std/mew)
    if xytext is None:
        ax.annotate(s,xy=xy,xytext=(bins[int(bins.shape[0]*.5)],counts.max()*.75),xycoords='data',fontsize=24)
    else:
        ax.annotate(s,xy=xy,xytext=xytext,xycoords='data',fontsize=24)
    ax.tick_params(labelsize=20)
    xlabel('ISI (ms)', fontsize=20)
    ylabel('Count', fontsize=20)
    return ax

def plot_spike_correlogram(T1, T2, width=20 * ms, bin=1 * ms, T=None, auto_ylim=True, ax=None, **plotargs):
    '''
    MODIFIED from brian.tools.statistics.correlogram

    T1,T2 are ordered arrays of spike times.

    Returns a cross-correlogram with lag in [-width,width] and given bin size.
    T is the total duration (optional) and should be greater than the duration of T1 and T2.
    The result is number of coincidences in each bin.

    auto_ylim automatically sets the ylim to be 1.2 times the second greatest value in the correlogram.

    N.B.: units are discarded.
    '''
    if (T1==[]) or (T2==[]): # empty spike train
        return NaN
    # Remove units
    width = float(width)
    T1 = array(T1)
    T2 = array(T2)
    i = 0
    j = 0
    n = int(ceil(width / bin)) # Histogram length
    l = []
    for t in T1:
        while i < len(T2) and T2[i] < t - width: # other possibility use searchsorted
            i += 1
        while j < len(T2) and T2[j] < t + width:
            j += 1
        l.extend(T2[i:j] - t)
    H, _ = histogram(l, bins=arange(2 * n + 1) * bin - n * bin)
    if ax is None:
        ax = subplot(111)
    ax.plot(linspace(-width*1000,width*1000,H.shape[0]),H,**plotargs)
    ax.fill_between(linspace(-width*1000,width*1000,H.shape[0]),H,**plotargs)
    xlim([-width*1000,width*1000])
    tick_params(labelsize=16)
    xlabel('Time (ms)',fontsize=20)
    ylabel('Count',fontsize=20)
    if auto_ylim:
        ylim([0,H[:H.shape[0]/2].max()*1.2])
     
# Source: https://gist.github.com/dmeliza/3251476
# LICENSE: Python Software Foundation (http://docs.python.org/license.html)        
from matplotlib.offsetbox import AnchoredOffsetbox
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=3,
                 pad=-.5, borderpad=0.1, sep=2, prop=None, **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
 
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0,0), sizex, 0, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0,0), 0, sizey, fc="none"))
 
        if sizex and labelx:
            bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False)],
                           align="center", pad=0, sep=sep)
        if sizey and labely:
            bars = HPacker(children=[TextArea(labely), bars],
                            align="center", pad=0, sep=sep)
 
        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)
 
def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes
 
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
 
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
 
    Returns created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l)>1 and (l[1] - l[0])
    
    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])
        
    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)
 
    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)
 
    return sb