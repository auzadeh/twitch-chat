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
from matplotlib.colors import LogNorm

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
        #index_list = inds_x.index(i)
        bin_x[inds_x[i]].append(x[i])
        bin_y[inds_x[i]].append(y[i])
    return_x = []
    return_y = []
    return_err = []
    for i in bin_y:
        return_y.append(np.median(bin_y[i]))
        #return_y.append(np.median(bin_y[i]))
        return_err.append(stats.sem(bin_y[i]))
    for i in bin_x:
        return_x.append(np.median(bin_x[i]))
    return return_x , return_y, return_err

def get_hdfs():
    in_dir = '/scratch/azadnema/twitch_anonymize/results/channels/five_mins_merge_game/'
    col = ['Time','ucount','mcount','ucount_botrmd','mcount_botrmd','pattern','msg_per_user','msg_per_user_botrmd']
    all_data = []
    #count = 0
    for file in os.listdir(in_dir):
    #if count <10:
    #    count += 1
        df = pd.read_table(in_dir+file, sep=',',header=None,names=col,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None,skiprows=[0])
        df = df[['mcount','ucount_botrmd','msg_per_user_botrmd']]
        df.replace('','NaN',inplace =True)
        df.replace(np.inf,'NaN',inplace =True)
        df.dropna(inplace=True)
        df['mcount'] = df['mcount'].astype(float)
        df['ucount_botrmd'] = df['ucount_botrmd'].astype(float)
        df['msg_per_user_botrmd'] = df['msg_per_user_botrmd'].astype(float)
        all_data.append(df)
    df_all =pd.concat(all_data)
    df_all.dropna(inplace=True)
    df_all.to_hdf('/scratch/azadnema/twitch_anonymize/results/channels/five_mins_all.h5','five_mins_all')

def get_hdfs_per_quartile():
    room_msgcount_df = pd.read_table('/scratch/azadnema/twitch_anonymize/results/channels/rooms_msg_count.csv',names=['room','msgcount'],sep=',')
    room_usercount_df = pd.read_table('/scratch/azadnema/twitch_anonymize/results/channels/rooms_user_count.csv',names=['room','usercount'],sep=',')
    room_daycount_df = pd.read_table('/scratch/azadnema/twitch_anonymize/results/channels/rooms_day_count.csv',names=['room','daycount'],sep=',')
    df_merge_ = pd.merge(room_msgcount_df,room_usercount_df,on='room')
    df_merge = pd.merge(df_merge_,room_daycount_df,on='room')
    df_filter = df_merge[room_msgcount_df['msgcount']>1000]
    df_filter = df_filter[df_filter['usercount']>100]
    df_filter = df_filter[df_filter['daycount']>1]
    df_filter.sort(['msgcount'],inplace =True)
    temp = np.array_split(df_filter, 4)
    q_room_dic = {}
    for i in range(len(temp)) :
        q_room_dic[i] = temp[i]['room'].tolist()
    in_dir = '/scratch/azadnema/twitch_anonymize/results/channels/five_mins_merge_game/'
    col = ['Time','ucount','mcount','ucount_botrmd','mcount_botrmd','pattern','msg_per_user','msg_per_user_botrmd']
    all_data = []
    for q in q_room_dic:
        outfile = '/scratch/azadnema/twitch_anonymize/results/channels/five_mins_'+str(q)+'.h5'
        for room in q_room_dic[q]:
            df = pd.read_table(in_dir+str(room)+'.log', sep=',',header=None,names=col,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None,skiprows=[0])
            df = df[['mcount','ucount_botrmd','msg_per_user_botrmd']]
            df.replace('','NaN',inplace =True)
            df.replace(np.inf,'NaN',inplace =True)
            df.dropna(inplace=True)
            df['mcount'] = df['mcount'].astype(float)
            df['ucount_botrmd'] = df['ucount_botrmd'].astype(float)
            df['msg_per_user_botrmd'] = df['msg_per_user_botrmd'].astype(float)
            all_data.append(df)
        df_all =pd.concat(all_data)
        df_all.dropna(inplace=True)
        df_all.to_hdf(outfile,'five_mins_all'+str(q))

def read_channel():
    in_dir = '/scratch/azadnema/twitch_anonymize/results/channels/five_mins_merge_game/'
    col = ['Time','ucount','mcount','ucount_botrmd','mcount_botrmd','pattern','msg_per_user','msg_per_user_botrmd']
    all_data = []
    count = 0
    for file in os.listdir(in_dir):
        df = pd.read_table(in_dir+file, sep=',',header=None,names=col,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None,skiprows=[0])
        df = df[['mcount','ucount_botrmd','msg_per_user_botrmd']]
        df.replace('','NaN',inplace =True)
        df.replace(np.inf,'NaN',inplace =True)
        df.dropna(inplace=True)
        df['mcount'] = df['mcount'].astype(float)
        df['ucount_botrmd'] = df['ucount_botrmd'].astype(float)
        df['msg_per_user_botrmd'] = df['msg_per_user_botrmd'].astype(float)
        all_data.append(df)
    df_all =pd.concat(all_data)
    df_all.dropna(inplace=True)
    ba_x,ba_y ,y_std= log_binning(df_all['mcount'].tolist(),df_all['msg_per_user_botrmd'].tolist(),50)
    df_plot = pd.DataFrame({'x':ba_x,'y':ba_y,'sterr':y_std})
    df_plot.to_csv('/scratch/azadnema/twitch_anonymize/results/channels/binned_msg_mpuserbrmd.csv',sep=',',index=False)
    ba_x1,ba_y1 ,y_std1 = log_binning(df_all['mcount'].tolist(),df_all['ucount_botrmd'].tolist(),50)
    df_plot1 = pd.DataFrame({'x':ba_x1,'y':ba_y1,'sterr':y_std1})
    df_plot1.to_csv('/scratch/azadnema/twitch_anonymize/results/channels/binned_msg_userbrmd.csv',sep=',',index=False)
    
def plot_msg_per_user():
    #df = pd.read_table('/u/azadnema/twitch/code/data/msg_userbrmd.csv',sep=',')
    df = pd.read_table('/scratch/azadnema/twitch_anonymize/results/channels/binned_msg_mpuserbrmd.csv',sep=',')
    ba_x = df['x'].tolist()
    ba_y = df['y'].tolist()
    y_std = df['sterr'].tolist()
    fig,ax=plt.subplots(1, 1)
    ax.errorbar(ba_x,ba_y,yerr=y_std,c='k',ls='',fmt='o',mfc='k',markersize=4)
    plt.axvspan(40, 1e4, color='grey', alpha=0.2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Messages',fontsize='xx-large')
    ax.set_ylabel('Messages per Users', fontsize='xx-large')
    #ax.set_xlim(1, 1e4)
    #ax.set_ylim(0,6)
    plt.tight_layout()
    plt.savefig('logbin_message_msgpusr_botrm_ylim_notset.pdf')
    plt.savefig('logbin_message_msgpusr_botrm_ylim_notset.png')
    plt.show()

def plot_user():
    #df = pd.read_table('/u/azadnema/twitch/code/data/msg_userbrmd.csv',sep=',')
    #df = pd.read_table('/scratch/azadnema/twitch_anonymize/results/channels/binned_msg_mpuserbrmd.csv',sep=',')
    df = pd.read_hdf('/scratch/azadnema/twitch_anonymize/results/channels/five_mins_all.h5','five_mins_all')
    df = df[df['ucount_botrmd']>1]
    ba_x,ba_y ,y_std = log_binning(df['mcount'].tolist(),df['msg_per_user_botrmd'].tolist(),50)
    fig,ax=plt.subplots(1, 1,figsize=(4, 2.5))
    ax.errorbar(ba_x,ba_y,yerr=y_std,c='k',ls='',fmt='o',mfc='k',markersize=3,barsabove=0,capsize=0,elinewidth=0)#,linestyle='None')
    plt.axvspan(40, 1e4, color='grey', alpha=0.2)
    plt.axvline(x=40 , linewidth=1, color='k')
    plt.axvline(x=200 , linewidth=1, color='r',ls='-.')
    ax.set_xscale('log')
    ax.set_xlabel('$V$',fontsize='x-large',fontname= 'sans-serif')
    ax.set_ylabel('$M_u$', fontsize='x-large',fontname= 'sans-serif')
    ax.set_xlim(1, 1e4)
    ax.set_ylim(1.2,3.8)
    plt.tight_layout()
    plt.savefig('logbin_message_usr_botrm_ucount_less1_removed_med.pdf')
    plt.savefig('logbin_message_usr_botrm_ucount_less1_removed_med.png')
    plt.show()

def plot_quartile():
    #df = pd.read_table('/u/azadnema/twitch/code/data/msg_userbrmd.csv',sep=',')
    #df = pd.read_table('/scratch/azadnema/twitch_anonymize/results/channels/binned_msg_mpuserbrmd.csv',sep=',')
    df_q0 = pd.read_hdf('/scratch/azadnema/twitch_anonymize/results/channels/five_mins_0.h5','five_mins_all0')
    df_q1 = pd.read_hdf('/scratch/azadnema/twitch_anonymize/results/channels/five_mins_1.h5','five_mins_all1')
    df_q2 = pd.read_hdf('/scratch/azadnema/twitch_anonymize/results/channels/five_mins_2.h5','five_mins_all2')
    df_q3 = pd.read_hdf('/scratch/azadnema/twitch_anonymize/results/channels/five_mins_3.h5','five_mins_all3')
    df_q0 = df_q0[df_q0['ucount_botrmd']>1]
    df_q1 = df_q1[df_q1['ucount_botrmd']>1]
    df_q2 = df_q2[df_q2['ucount_botrmd']>1]
    df_q3 = df_q3[df_q3['ucount_botrmd']>1]
    f,((ax1,ax2),(ax3,ax4))= plt.subplots(2,2,sharex=True,figsize=(5, 4))#, sharex=True)
    import matplotlib as mpl
    mpl.rcParams['text.latex.preamble']=['\usepackage{wasysym}']
    for ax,x_label,y_label ,ti in zip([ax1,ax2,ax3,ax4],['','','$V$','$V$'],['$M_u$','','$M_u$',''],['$Q_1$','$Q_2$','$Q_3$','$Q_4$']):
        if ax == ax1:
            x_list = df_q0['mcount'].tolist()
            y_list = df_q0['msg_per_user_botrmd'].tolist()
            ax.text(1000, 3, ti, color='black',fontname= 'sans-serif', weight='bold',size='large')
        if ax == ax2:
            x_list = df_q1['mcount'].tolist()
            y_list = df_q1['msg_per_user_botrmd'].tolist()
            ax.text(1000, 3, ti, color='black',fontname= 'sans-serif', weight='bold',size='large')
        if ax == ax3:
            x_list = df_q2['mcount'].tolist()
            y_list = df_q2['msg_per_user_botrmd'].tolist()
            ax.text(1000, 3, ti, color='black',fontname= 'sans-serif', weight='bold',size='large')
        if ax == ax4:
            x_list = df_q3['mcount'].tolist()
            y_list = df_q3['msg_per_user_botrmd'].tolist()
            ax.text(1000, 3, ti, color='black',fontname= 'sans-serif', weight='bold',size='large')
        ba_x, ba_y, y_err  = log_binning(x_list , y_list, bin_count=20)
        ax.errorbar(ba_x,ba_y,yerr=y_err,c='k',ls='',fmt='o',mfc='k',markersize=3,barsabove=0,capsize=0,elinewidth=0)
        ax.set_xlabel(x_label,fontsize='x-large',fontname= 'sans-serif')
        ax.set_ylabel(y_label,fontsize='x-large',fontname= 'sans-serif')
        ax.locator_params(axis ='y',nbins=6)
        ax.axvspan(40, 1e4, color='grey', alpha=0.2)
        ax.set_xscale('log')
        ax.set_xlim(1, 1e4)
        ax.set_ylim(0.5,7.2)
    plt.tight_layout()
    plt.savefig('quartile_less2_removed_med.pdf')
    plt.savefig('quartile_less2_removed_med.png')
    plt.show()

def plot_usercount():
    #df = pd.read_table('/u/azadnema/twitch/code/data/msg_userbrmd.csv',sep=',')
    #df = pd.read_table('/scratch/azadnema/twitch_anonymize/results/channels/binned_msg_mpuserbrmd.csv',sep=',')
    df = pd.read_hdf('/scratch/azadnema/twitch_anonymize/results/channels/five_mins_all.h5','five_mins_all')
    df = df[df['ucount_botrmd']>1]
    ba_x,ba_y ,y_std = log_binning(df['mcount'].tolist(),df['ucount_botrmd'].tolist(),50)
    fig,ax=plt.subplots(1, 1)
    ax.errorbar(ba_x,ba_y,yerr=y_std,c='k',ls='',fmt='o',mfc='k',markersize=5,barsabove=0,capsize=0,elinewidth=0)#,linestyle='None')
    plt.axvspan(40, 1e4, color='grey', alpha=0.2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.axvspan(40, 1e4, color='grey', alpha=0.2)
    plt.axvline(x=40 , linewidth=1, color='k')
    plt.axvline(x=200 , linewidth=1, color='r',ls='-.')
    ax.set_xlabel('$V$',fontsize='xx-large')
    ax.set_ylabel('$U$', fontsize='xx-large')
    plt.tight_layout()
    plt.savefig('logbin_message_ucountbotrm_less1_removed.pdf')
    plt.savefig('logbin_message_ucountbotrm_less1_removed.png')
    plt.show()

def plot_hexbin():
    #df_all.to_hdf('/scratch/azadnema/twitch_anonymize/results/channels/five_mins_all.h5','five_mins_all')
    temp = pd.read_hdf('/scratch/azadnema/twitch_anonymize/results/channels/five_mins_all.h5','five_mins_all')
    temp.sort(['msg_per_user_botrmd'],inplace =True,ascending=False)
    temp_list = temp['msg_per_user_botrmd'].tolist()
    l = [956.0, 889.0, 793.0, 669.0, 666.0, 651.0, 647.0, 642.0, 641.0]
    plt.show()

if __name__ == '__main__':
    #get_hdfs()
    #read_channel()
    #plot_user()
    #plot_usercount()
    #plot_hexbin()
    #get_hdfs_per_quartile()
    plot_quartile()
