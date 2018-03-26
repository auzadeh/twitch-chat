from matplotlib.ticker import NullFormatter
from matplotlib.colors import LogNorm
import os
import csv
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from pylab import *
from scipy.optimize import minimize, leastsq,show_options
from scipy.special import erf
from numpy import random
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
        #index_list = inds_x.index(i)
        bin_x[inds_x[i]].append(x[i])
        bin_y[inds_x[i]].append(y[i])
    return_x = []
    return_y = []
    return_err = []
    for i in bin_y:
        return_y.append(np.mean(bin_y[i]))
        #return_y.append(np.median(bin_y[i]))
        return_err.append(stats.sem(bin_y[i]))
    for i in bin_x:
        #return_x.append(i)
        return_x.append(np.mean(bin_x[i]))
    return return_x , return_y, return_err

def split_msg(dir):
    channel_files = pd.read_csv(dir+'channels_1day_1000msg_100users.log',names=['room','file'])
    gameroom_dir = dir +'five_mins_merge_game_userlist/'
    split_dir = dir+'msg_userlist/'
    all_data = []
    for file in channel_files['file'].tolist():
        df_game = pd.read_table(gameroom_dir+file.split('/')[7], sep='\x1e',quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        all_data.append(df_game)
    df_all_data = pd.concat(all_data)
    df_all_data['mcount'] = df_all_data['mcount'].astype(float)
    df_all_data.dropna(inplace=True)
    df_all_data_grouped = df_all_data.groupby('mcount')['ulist_botrm'].apply(lambda x: '\x1f'.join(x))
    for k, v in df_all_data_grouped.iteritems():
        outfile  = open(split_dir+str(k)+'.log','w')
        outfile.write(v)

def msg_usercount(dir):
    split_dir = dir+'msg_userlist/'
    outfile = open(dir+'/msgcount_username_usercount_userprob.csv','w')
    for file in os.listdir(split_dir):
     if file =='0.0.log':
         pass
     else:
        print file 
        users_list = []
        allusers_list = []
        ofile = open(split_dir+file,'r')
        for i in ofile:
            users_list += i.split('\x1f')
            allusers_list += i.strip().replace('[','').replace(']','').replace(',','\x1f').split('\x1f')
        msg_speed = file.split('.')[0]
        user_count = len(allusers_list)
        msg_user_count = collections.Counter(allusers_list)
        for u in msg_user_count:
            if u=='':
                pass
            else:
                ucount =  msg_user_count[u]
                uprob = float(ucount)/user_count
                outfile.write(str(msg_speed)+','+str(u)+','+str(ucount)+','+str(uprob)+'\n')

def user_activity(dir,theta):
    infile =dir+'msgcount_username_usercount_userprob.csv'
    df = pd.read_table(infile,names =['msgcount','username','usercount','userprob'],sep=',')
    df['msgcount'] = df['msgcount'].astype(float)
    df['usercount'] = df['usercount'].astype(float)
    df['userprob'] = df['userprob'].astype(float)
    df_less_list = set(df[df['msgcount'] < theta]['username'].tolist())
    df_greater_list = set(df[df['msgcount'] > theta]['username'].tolist())
    user_intersect = df_less_list & df_greater_list
    df_intersect = df[df['username'].isin(user_intersect)]
    df_intersect.to_hdf(dir+str(theta)+'_msgcount_username_usercount_userprob.h5','msgcount_username_usercount_userprob')

#create the file of users with delta
def user_activity_analysis(dir,theta):
    infile =dir+'msgcount_username_usercount_userprob.csv'
    df = pd.read_table(infile,names =['msgcount','username','usercount','userprob'],sep=',')
    df['msgcount'] = df['msgcount'].astype(float)
    df['usercount'] = df['usercount'].astype(float)
    df['userprob'] = df['userprob'].astype(float)
    df_less = df[df['msgcount'] < theta]
    df_greater = df[df['msgcount'] > theta]
    df_less = pd.DataFrame({'sum_less' : df_less.groupby(['username'])['usercount'].sum()}).reset_index()
    df_greater = pd.DataFrame({'sum_greater' : df_greater.groupby(['username'])['usercount'].sum()}).reset_index()
    df_merge = pd.merge(df_less,df_greater,on='username')
    df_merge.fillna(0,inplace=True)
    df_merge['delta'] = df_merge['sum_less']-df_merge['sum_greater']
    df_merge['sum'] = df_merge['sum_less']+df_merge['sum_greater']
    df_merge['delta_norm'] = df_merge['delta'].astype(float)/df_merge['sum']
    df_merge.to_hdf(dir+str(theta)+'_alluser_count_delta.h5','user_count_meanless_mean_greater_delta')

def devide_users(dir,theta):
    infile =dir+'msgcount_username_usercount_userprob.csv'
    df = pd.read_hdf(dir+'msgcount_username_usercount_userprob.h5','msgcount_username_usercount_userprob')
    df['msgcount'] = df['msgcount'].astype(float)
    df['usercount'] = df['usercount'].astype(float)
    df['userprob'] = df['userprob'].astype(float)
    users_list = pd.read_hdf('username_greater_100_msg','username')
    count = 0
    sample_size = 100000
    user_sample = []
    l = len(users_list)/4
    for i in range(4):
        user_sample += np.random.choice(users_list[i*l: (i+1)*l],sample_size/4).tolist()
    df = df[df['username'].isin(users_list)]
    u_slope_g = []
    u_slope_l = []
    user_slope_dic = {}
    user_slope_norm_dic = {}
    of = open(dir+'sample_2_user_slopes_all_4.csv','w')
    ofn = open(dir+'sample_2_user_slopes_norm_all_4.csv','w')
    for u in user_sample:
        df_u = df[df['username']==u]
        df_u_less = df_u[df_u['msgcount'] < theta]
        df_u_greater = df_u[df_u['msgcount'] > theta]
        df_u_greater = df_u_greater[df_u_greater['msgcount'] < 200]
        if len(df_u_less) < 4:
            pass
        elif len(df_u_greater) < 4:
            pass
        else:
            count += 1
            x_l,y_l,std = log_binning(df_u_less['msgcount'].tolist(),df_u_less['usercount'].tolist(),bin_count=10)
            mean_v_l = np.mean(x_l)
            std_v_l = np.std(x_l)
            mean_mu_l = np.mean(y_l)
            std_mu_l = np.std(y_l)
            v_l_star = [float(x - mean_v_l)/std_v_l for x in x_l]
            mu_l_star = [float(x - mean_mu_l)/std_mu_l for x in y_l]
            x_g,y_g,std = log_binning(df_u_greater['msgcount'].tolist(),df_u_greater['usercount'].tolist(),bin_count=10)
            mean_v_g = np.mean(x_g)
            std_v_g = np.std(x_g)
            mean_mu_g = np.mean(y_g)
            std_mu_g = np.std(y_g)
            v_g_star = [float(x - mean_v_g)/std_v_g for x in x_g]
            mu_g_star = [float(x - mean_mu_g)/std_mu_g for x in y_g]
            slope_l, intercept_l, r_value_l, p_value_l, std_err_l = stats.linregress(x_l,y_l)
            slope_g, intercept_g, r_value_g, p_value_g, std_err_g = stats.linregress(x_g,y_g)
            u_slope_g.append(slope_g)
            u_slope_l.append(slope_l)
            user_slope_dic[u]=(slope_l,slope_g)
            slope_l_n, intercept_l_n, r_value_l_n, p_value_l_n, std_err_l_n = stats.linregress(v_l_star,mu_l_star)
            slope_g_n, intercept_g_n, r_value_g_n, p_value_g_n, std_err_g_n = stats.linregress(v_g_star,mu_g_star)
            user_slope_norm_dic[u]=(slope_l_n,slope_g_n)
            of.write(str(u)+','+str(slope_l)+','+str(slope_g)+'\n')
            ofn.write(str(u)+','+str(slope_l_n)+','+str(slope_g_n)+'\n')

