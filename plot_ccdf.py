import pandas as pd
import pylab as p
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
import os
import matplotlib.gridspec as gridspec
def plot_cdf(ax,x,_color,copy=True, fractional=True):#, **kwargs):
    """
    Add a log-log CCDF plot to the current axes.
 
    Arguments
    ---------
    x : array_like
        The data to plot
 
    copy : boolean
        copy input array in a new object before sorting it. If data is a *very*
        large, the copy can avoided by passing False to this parameter.
 
    fractional : boolean
        compress the data by means of fractional ranking. This collapses the
        ranks from multiple, identical observations into their midpoint, thus
        producing smaller figures. Note that the resulting plot will NOT be the
        exact CCDF function, but an approximation.
 
    Additional keyword arguments are passed to `matplotlib.pyplot.loglog`.
    Returns a matplotlib axes object.
 
    """
    N = float(len(x))
    if copy:
        x = x.copy()
    x.sort()
    if fractional:
        t = []
        for x, chunk in groupby(enumerate(x, 1), itemgetter(1)):
            xranks, _ = zip(*list(chunk))
            t.append((float(x), xranks[0] + np.ptp(xranks) / 2.0))
        t = np.asarray(t)
    else:
        t = np.c_[np.asfarray(x), np.arange(N) + 1]
    #if 'ax' not in kwargs:
    #ax = plt.gca()
    #else:
    #ax = kwargs.pop('ax')
    ax.loglog(t[:, 0], (N - t[:, 1]) / N, c=_color)#'ow',c=_color,fillstyle='none',markersize = 2)#, **kwargs)

def main_plot():
    vdir = '/scratch/azadnema/twitch_anonymize/results/viewers/'
    cdir = '/scratch/azadnema/twitch_anonymize/results/channels/'

    viewer_day_df = pd.read_csv(vdir+'viewers_day_count.csv',names=['user','day'])
    viewer_msg_df = pd.read_csv(vdir+'viewers_msg_count.csv',names=['user','msg'])
    viewer_df = pd.read_csv(vdir+'viewers_user_count.csv',names=['user','room'])
    viewer_df['msg']  = viewer_msg_df['msg']
    viewer_df['day'] = viewer_day_df['day']
    viewer_df['msg_per_day'] = viewer_df['msg'].astype(float)/ viewer_df['day']

    channel_day_df = pd.read_csv(cdir+'rooms_day_count.csv',names=['channel','day'])
    channel_msg_df = pd.read_csv(cdir+'rooms_msg_count.csv',names=['channel','msg'])
    channel_df = pd.read_csv(cdir+'rooms_user_count.csv',names=['channel','user'])
    channel_df['msg']  = channel_msg_df['msg']
    channel_df['day'] = channel_day_df['day']
    channel_df['msg_per_day'] = channel_df['msg'].astype(float)/ viewer_df['day']

    #print channel_df

    #plt.figure(figsize = (2,3))
    #gs1 = gridspec.GridSpec(2, 3)
    #gs1.update(wspace=0.025, hspace=0.05)

    f,((ax1,ax2,ax3), (ax4,ax5,ax6))= plt.subplots(2,3,sharey=True)#, sharex=True)
    f.subplots_adjust(hspace=0.001, wspace=0.001)
    #plt.figure(figsize=(12, 3))
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    for ax,feature, x_label,y_label,_color,plot_type in zip([ax1,ax2,ax3,ax4,ax5,ax6],['msg','room','day','msg','user','day'],['Messages','Channels','Days','Messages','Users','Days'],['$P(X<x)$','','','$P(X<x)$','',''],['k','k','k','k','k','k'],['Users','','','Channels','','']):
        if ax in [ax1,ax2,ax3]:
            x = viewer_df[feature].tolist()
        elif ax in [ax4,ax5,ax6]: 
            x = channel_df[feature].tolist()
        x = np.array(x)
        plot_cdf(ax,x,_color[0])
        ax.set_xlabel(x_label,fontsize='x-large')
        ax.set_ylabel(y_label,fontsize='xx-large')
        #ax.text(right, top, 'Users',horizontalalignment='left',verticalalignment='bottom', transform=ax.transAxes)
        ax.set_ylabel(y_label,fontsize='xx-large')
        if ax in [ax1,ax2,ax4,ax5]:
            ax.text(3,0.0001,plot_type ,fontsize='x-large')
            #ax.set_xlim(1,10000000)
            xticks = ax.xaxis.get_major_ticks()
            xticks[1].label1.set_visible(False)
            xticks[3].label1.set_visible(False)
            xticks[5].label1.set_visible(False)
            xticks[7].label1.set_visible(False)
        else:
            #ax.text(5,0.0001,plot_type ,fontsize='x-large')
            xticks = ax.xaxis.get_major_ticks()
            xticks[1].label1.set_visible(False)
            #xticks[3].label1.set_visible(False)
            #xticks[5].label1.set_visible(False)
            #xticks[7].label1.set_visible(False)
        if ax in [ax1,ax4]:
            ax.set_xlim(1,10000000)
        if ax in [ax2,ax5]:
            ax.set_xlim(1,1000000)

    ax.set_ylim(1e-7,1)
    f.subplots_adjust(wspace=0.07,hspace=0.3)

    #f.tight_layout()
    f.savefig('viewer_channel_ccdf_withRank.pdf')
    f.savefig('viewer_channel_ccdf_withRank.png')
    plt.show()


if __name__ == '__main__':
    main_plot()

