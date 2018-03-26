import matplotlib as mpl
import os
import csv
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from pylab import *
from scipy.stats import poisson, expon, binom
from scipy.optimize import minimize, leastsq,show_options
from scipy.special import erf
from numpy import random
import pylab
from sklearn import mixture
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
from scipy import stats
from collections import Counter
import collections

def drop_zeros(a_list):
    return [i for i in a_list if i>0]

def log_binning(x,y,bin_count=35):
    max_x = log10(max(x))
    max_y = log10(max(y))
    max_base = max([max_x,max_y])
    min_x = log10(min(drop_zeros(x)))
    bins = np.logspace(min_x,max_x,num=bin_count)
    inds_x = np.digitize(x[1:], bins)

    bin_x = {}
    bin_y = {}
    for i in inds_x:
        bin_x[i] = []
        bin_y[i] = []
    for i in range(len(inds_x)):
        bin_x[inds_x[i]].append(x[i])
        bin_y[inds_x[i]].append(y[i])

    return_x = []
    return_y = []
    return_err = []
    for i in bin_y:
        #return_y.append(np.mean(bin_y[i]))
        return_y.append(np.median(bin_y[i]))
        return_err.append(stats.sem(bin_y[i]))
    for i in bin_x:
        #return_x.append(np.mean(bin_x[i]))
        return_x.append(np.median(bin_x[i]))
    return return_x , return_y, return_err


def plot_text():
    discourse_file = '/scratch/azadnema/twitch_anonymize/results/channels/msgcount_discourse_marker_greater_one.csv'
    qmark_mentioned_len_file = '/scratch/azadnema/twitch_anonymize/results/channels/msgcount_qmark_mentioned_len_greater_one.csv'
    emote_file = '/scratch/azadnema/twitch_anonymize/results/channels/msgcount_emote_greater_one_merge.csv'
    len_file = '/scratch/azadnema/twitch_anonymize/results/channels/msgcount_len_greater_one.csv'
    ratio_file = '/scratch/azadnema/twitch_anonymize/results/channels/msgcount_ratio_greater_one_merge.csv'

    df_discourse = pd.read_table(discourse_file,sep=',',names=['speed','dprop','derr'])
    df_qmark_mentioned_len  = pd.read_table(qmark_mentioned_len_file,sep = ',',names=['speed','qmark','mentioned','len_w'])
    df_emote = pd.read_table(emote_file,sep=',',names=['speed','emote','emot_err'])
    df_len = pd.read_table(len_file,sep=',',names=['speed','len'])
    df_ratio = pd.read_table(ratio_file,sep=',',names=['speed','ratio'])

    df_merge_two =  pd.merge(df_discourse,df_qmark_mentioned_len,on='speed')
    df_merge_three = pd.merge(df_merge_two,df_emote,on='speed')
    df_merge_four =  pd.merge(df_merge_three,df_ratio,on='speed')
    
    df_merge = pd.merge(df_merge_four,df_len,on='speed') 

    df_merge.dropna(inplace=True)
    df_merge.sort(['speed'],inplace=True)
    df_merge['speed'] = df_merge['speed'].astype(float)
    df_merge['dprop'] = df_merge['dprop'].astype(float)*1000
    df_merge['derr'] = df_merge['derr'].astype(float)
    df_merge['emote'] = df_merge['emote'].astype(float)*100000
    df_merge['emot_err'] = df_merge['emot_err'].astype(float)
    df_merge['qmark'] = df_merge['qmark'].astype(float)
    df_merge['mentioned'] = df_merge['mentioned'].astype(float)
    df_merge['len'] = df_merge['len'].astype(float)
    df_merge['ratio'] = df_merge['ratio'].astype(float)


    df_merge[['speed','dprop','mentioned','emote','qmark','len','ratio']].to_csv('features.csv',index=False,sep=',')

    f,((ax1,ax2, ax3),(ax4,ax5,ax6))= plt.subplots(2,3,sharex=True,figsize=(7.5, 4))#, sharex=True)

    import matplotlib as mpl 
    mpl.rcParams['text.latex.preamble']=['\usepackage{wasysym}']
    plt.rc('text',usetex=True)

    for ax,x_label,y_label,plot_type , y ,ti in zip([ax1,ax2,ax3,ax4,ax5,ax6],['','','','$V$','$V$','$V$'],['$p_@$','$p_?$',r'$p_d$ ',r'$\rho$','$l_m$',r'$p_{\text{\sf :-)}}$'],['X','Y','Z','W','T','U'],[ 'mentioned','qmark','dprop','ratio','len','emote'],['a','b','c','d','e','f']):
        x_list = df_merge['speed'].tolist()
        y_list = df_merge[y].tolist()
        ba_x, ba_y, y_err  = log_binning(x_list , y_list, bin_count=20)
        ax.errorbar(ba_x,ba_y,yerr=y_err,c='k',ls='',fmt='o',mfc='k',markersize=3,barsabove=0,capsize=0,elinewidth=0)
        ax.set_xlabel(x_label,fontsize='x-large',fontname= 'sans-serif')
        ax.set_ylabel(y_label,fontsize='x-large',fontname= 'sans-serif')
        ax.locator_params(axis ='y',nbins=6)
        if ax == ax1:
            ax.text(1000, 0.032, ti, color='black',fontname= 'sans-serif', weight='bold',size='large')
        if ax == ax2:
            ax.text(1000, 0.06, ti, color='black',fontname= 'sans-serif', weight='bold',size='large')
        if ax == ax3:
            ax.set_title(r'$\times10^{3}$', fontsize="large",fontname= 'sans-serif',loc='left')
            ax.text(1000, 0.4, ti, color='black',fontname= 'sans-serif', weight='bold',size='large')
        if ax == ax4:
            ax.text(1000, 0.5, ti, color='black',fontname= 'sans-serif', weight='bold',size='large')
        if ax == ax5:
            ax.text(1000, 25, ti, color='black',fontname= 'sans-serif', weight='bold',size='large')
        if ax ==ax6:
            ax.set_title(r'$\times10^{5}$', fontsize="large",fontname= 'sans-serif',loc='left')
            ax.text(1000, 6, ti, color='black',fontname= 'sans-serif', weight='bold',size='large')
            ax.set_ylim(0,8)
        ax.set_xscale('log')
        ax.axvspan(40, 1e4, color='grey', alpha=0.2)
        ax.axvline(x=40 , linewidth=1, color='k')
        ax.axvline(x=200 , linewidth=1, color='r',ls='-.')

        ax.set_ylim(0)
    #f.subplots_adjust(wspace=0.07,hspace=0.3)
    f.tight_layout()
    f.savefig('features_new.pdf')
    f.savefig('features_new.svg')
    plt.show()

if __name__ == '__main__':
    plot_text()

