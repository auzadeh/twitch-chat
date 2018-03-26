import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
import os
from matplotlib.colors import LogNorm
import csv

def plot_cdf(ax,x,_color,copy=True, fractional=True):#, **kwargs):
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
    ax.loglog(t[:, 0], (N - t[:, 1]) / N,c=_color)#, 'ow',c=_color,fillstyle='none',markersize = 2)#, **kwargs)

def get_viewers_intertime():
    df_reg =  pd.read_csv("/u/azadnema/twitch/code/data/labeled_viewers_combined.csv")
    outfile = '/scratch/azadnema/twitch_anonymize/results/viewers/labeled_viewers_lag_min/user_lag_min_'
    df_reg.columns = ["name", "type"]
    df_reg['name'] = df_reg['name'].str.strip('.csv')
    df_reg = df_reg[df_reg['type'].isin([0,1,3])==True]
    viewers_list = df_reg['name'].astype(int).tolist()
    df_viewers_adrs = pd.read_csv('/scratch/azadnema/twitch_anonymize/results/viewers/viewer_data_path.log', names=['viewer','adrs'])
    df_viewers_adrs = df_viewers_adrs[df_viewers_adrs['viewer'].isin(viewers_list)]
    viewers_adrs = df_viewers_adrs['adrs'].tolist()
    for v in viewers_adrs:
        df_viewer = pd.read_table(v, sep='\x1e',header=None,quoting = csv.QUOTE_NONE,names=['channel','text','time'],lineterminator='\n', na_filter=None)
        df_viewer['time'] = df_viewer['time'].astype('datetime64[ns]')
        df_viewer['Delta'] = (df_viewer['time']-df_viewer['time'].shift()).fillna(0)
        df_viewer['Delta_min'] = df_viewer['Delta'].apply(lambda x: x  / np.timedelta64(1,'m')).astype('int64') #% (60)
        df_viewer[df_viewer['Delta_min']>=0]['Delta_min'].to_csv(outfile+str(v.split('/')[7]).strip('.log')+'.csv',index=False,header=False)

def plot_viewer_lag():
    label_viewers = '/u/azadnema/twitch/code/data/labeled_viewers_combined.csv'
    dir = '/scratch/azadnema/twitch_anonymize/results/viewers/labeled_viewers_lag_min/user_lag_min_'
    label_viewers_df = pd.read_csv(label_viewers,names=['adr','status'])
    label_viewers_df['user'] = label_viewers_df['adr'].str.rstrip('.csv')
    label_viewers_df['user'] =  label_viewers_df['user'].astype(int)
    bots = label_viewers_df[label_viewers_df['status']==1]['user'].tolist()
    human = label_viewers_df[label_viewers_df['status']==0]['user'].tolist()
    cp = label_viewers_df[label_viewers_df['status']==3]['user'].tolist()
    bot_list = []
    for b in bots:
        if os.path.exists(dir+str(b)+'.csv'):
            bot_df = pd.read_csv(dir+str(b)+'.csv')
            bot_list.append(bot_df)
    bots_df = pd.concat(bot_list)
    human_list = []
    for b in human:
        if os.path.exists(dir+str(b)+'.csv'):
            human_df = pd.read_csv(dir+str(b)+'.csv')
            human_list.append(human_df)
    humans_df = pd.concat(human_list)
    cp_list = []
    for b in cp:
        if os.path.exists(dir+str(b)+'.csv'):
            cp_df = pd.read_csv(dir+str(b)+'.csv')
            cp_list.append(cp_df)
    cps_df = pd.concat(cp_list)
    f,(ax1,ax2,ax3)= plt.subplots(1,3,sharey=True,figsize=(9, 4))#, sharex=True)
    f.subplots_adjust(hspace=0.001, wspace=0.001)
    for ax, x_label,y_label,_color,plot_type in zip([ax1,ax2,ax3],[' Speed','Speed','Speed'],['CCDF','',''],['k','k','k'],['Bot','Human',' Copy-Paster']):
        if ax in [ax1]:
            x = bots_df['0'].tolist()
        elif ax in [ax2]:
            x = humans_df['0'].tolist()
        elif ax in [ax3]:
            x = cps_df['0'].tolist()
        ax.text(0.17,0.001,plot_type ,fontsize='x-large')
        x = np.array(x)
        x = 1.0/x
        plot_cdf(ax,x,_color[0])
        ax.set_xlabel(x_label,fontsize='x-large')
        ax.set_ylabel(y_label,fontsize='xx-large')
        ax.set_xlim(1e-6,10)
        ax.set_ylim(0.75,1.01)
        xticks = ax.xaxis.get_major_ticks()
        xticks[1].label1.set_visible(False)
    f.tight_layout()
    f.subplots_adjust(wspace=0.07,hspace=0.3)
    plt.savefig('labeled_viewers_speed_mins.pdf')
    plt.savefig('labeled_viewers_speed_mins.png')
    plt.show()

def get_file():
    vfile = '/u/azadnema/twitch/code/data/viewer_vector_msg10_day1.csv'
    rfile = '/scratch/azadnema/twitch_anonymize/results/viewers/viewers_ratio.csv'
    viewer_df = pd.read_csv(vfile,names=['user','room','msg','day','wrong_ratio','speed'])
    viewer_df.dropna(inplace = True)
    viewer_df.drop(viewer_df.columns[[4]], axis=1,inplace=True)
    ratio_df = pd.read_csv(rfile,names=['1','2','3','4','5','6','7','user','ratio','err'],sep='/|,', engine='python')
    ratio_df.drop(ratio_df.columns[[0,1, 2, 3,4,5,6,9]], axis=1,inplace=True)
    merge_df = pd.merge(viewer_df,ratio_df,on='user')
    merge_df.dropna(inplace = True)
    merge_df = merge_df[merge_df['msg']>100]
    merge_df['log'] = np.log10(merge_df['ratio'])
    botts = merge_df[merge_df['log']<-0.44]
    botfile =  '/u/azadnema/twitch/code/data/botts_msg100_day1.csv'
    botts.to_csv(botfile,sep=',',index=False)

def main_plot():
    #status can be (bot :1) (human:0) (copy-paster : 3) (uncertain: -1) (not-english:2)
    label_viewers = '/u/azadnema/twitch/code/data/labeled_viewers_combined.csv'
    label_viewers_df = pd.read_csv(label_viewers,names=['adr','status'])
    label_viewers_df['user'] = label_viewers_df['adr'].str.rstrip('.csv')
    label_viewers_df['user'] =  label_viewers_df['user'].astype(int)
    bots = label_viewers_df[label_viewers_df['status']==1]['user'].tolist()
    human = label_viewers_df[label_viewers_df['status']==0]['user'].tolist()
    cp = label_viewers_df[label_viewers_df['status']==3]['user'].tolist()
    vfile = '/u/azadnema/twitch/code/data/viewer_vector_msg100_day1.csv'
    viewer_df = pd.read_csv(vfile,names=['user','room','msg','day','speed','ratio'],skiprows=1)
    viewer_df['speed_min'] = viewer_df['speed']/60
    viewer_df.dropna(inplace = True)
    human_df = viewer_df.loc[viewer_df['user'].isin(human)]
    x_human = (1.0/human_df['speed_min']).tolist()
    y_human = human_df['ratio'].tolist()
    bot_df = viewer_df.loc[viewer_df['user'].isin(bots)]
    x_bot = (1.0/bot_df['speed_min']).tolist()
    y_bot = bot_df['ratio'].tolist()
    cp_df = viewer_df.loc[viewer_df['user'].isin(cp)]
    x_cp = (1.0/cp_df['speed_min']).tolist()
    y_cp = cp_df['ratio'].tolist()
    x = (1.0/viewer_df['speed_min']).tolist()
    y = viewer_df['ratio'].tolist()
    plt.hexbin(x, y, cmap=plt.cm.gray_r,norm=LogNorm(),xscale = 'log', yscale = 'log',alpha=0.5)
    cb = plt.colorbar()
    cb.set_label('Frequency',fontsize='xx-large')
    ax = plt.gca()
    xticks = ax.xaxis.get_major_ticks()
    yticks = ax.yaxis.get_major_ticks()
    plt.plot(x_human,y_human,ls='',marker='s',c='b',markersize=7,markerfacecolor='white',markeredgewidth=2,label='Human')
    plt.plot(x_cp,y_cp,ls='',marker='+',markeredgecolor='b',markersize=9,markerfacecolor='white',markeredgewidth=3,label='Copy-paster' )
    plt.plot(x_bot,y_bot,ls='',marker='o',markeredgecolor='r',markersize=7,markerfacecolor='white',markeredgewidth=2,label='Bot') 
    plt.xlabel(r'inter-message time: $\tau$',fontsize='xx-large')
    plt.ylabel(r'Compression Ratio: $\rho$',fontsize='xx-large')
    plt.legend(prop={'size':'medium'}, frameon=False, loc='lower right', numpoints=1)
    plt.tight_layout()
    plt.savefig('bots_min2.pdf')
    plt.savefig('bots_min2.png')
    plt.show()

if __name__ == '__main__':
    #get_file()
    main_plot()
    #for figuer intertime 
    #get_viewers_intertime()
    #plot_viewer_lag()
    #status can be (bot :1) (human:0) (copy-paster : 3) (uncertain: -1) (not-english:2)
    #name , status, commnet
    #labeled_data = pd.read_csv('/u/azadnema/twitch/code/data/labeled_viewers_combined.csv',names=['user','label','comment'])

