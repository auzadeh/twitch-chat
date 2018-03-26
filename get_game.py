import re
import os
import operator
import csv
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from pylab import *
from scipy.stats import poisson, expon, binom
from scipy.optimize import minimize, leastsq
from scipy.special import erf
from numpy import random
from scipy.optimize import minimize, show_options
import pylab

def select_game(file,outfile):
    msgcol = ['Time' ,'mcount']
    df = pd.read_table(file, sep=',',header=None,names=msgcol,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
    df['Time'] = pd.to_datetime(df['Time'])
    df['Time'] =  pd.DatetimeIndex(df['Time'])
    df = df.set_index('Time')
    df_rolling = df
    #df_rolling =pd.rolling_mean(df, window=12, min_periods=1,center=True).fillna(0)
    count_mean = df_rolling['mcount'].mean() 
    df_rolling['pattern'] = (df_rolling['mcount']>count_mean).astype('int')
    time_list = df_rolling.index.tolist()
    pattern_list = df_rolling['pattern'].tolist()
    pattern_str = ''
    for i in pattern_list:
        pattern_str += str(i)
    while '1101' in pattern_str:#len([m.start() for m in re.finditer('FTF',mystr)])>0:
            pattern_str = pattern_str.replace("1101","1111")
    while '0010' in pattern_str:#len([m.start() for m in re.finditer('TFT',mystr)])>0:
            pattern_str = pattern_str.replace("0010","0000")
    new_pattern_list = []
    for i in pattern_str:
        new_pattern_list.append(int(i))
    df_game = pd.DataFrame({'time':time_list,'pattern':new_pattern_list})
    df_game = df_game[df_game['pattern']==1]
    df_game['time'].to_csv(outfile,index=False,header=False)

def read_channels():
    room_msgcount = '/l/nx/data/twitch_anonymize/icwsm/five_mins_msgcount/'
    out_dir = '/l/nx/data/twitch_anonymize/icwsm/games_per_room/'
    for file in os.listdir(room_msgcount): 
        outfile = out_dir+'game'+file.split('_')[1]
        select_game(room_msgcount+file,outfile)

def merge_game():
    dir = '/l/nx/data/twitch_anonymize/icwsm/'
    channel_files = pd.read_csv(dir+'channels_1day_1000msg_100users.log',names=['room','file'])
    out_dir = dir+'five_mins_merge_game/'
    game_dir = dir+'games_per_room/'
    msg_dir = dir+'five_mins_msgcount/'
    user_dir = dir+'five_mins_usercount/'
    msg_rmbot_dir = dir+'five_mins_msgcount_rmbot/'
    user_rmbot_dir = dir+'five_mins_usercount_rmbot/'
    for file in channel_files['file'].tolist():
        outfile = out_dir+file.split('/')[7]
        df_game = pd.read_table(game_dir+'game'+file.split('/')[7], sep=',',header=None,names=['Time'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_msg = pd.read_table(msg_dir+'msgcount_'+file.split('/')[7], sep=',',header=None,names=['Time' ,'mcount'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_msg['Time'] = pd.to_datetime(df_msg['Time'])
        df_msg['Time'] =  pd.DatetimeIndex(df_msg['Time'])
        df_msg = df_msg.set_index('Time')
        df_user = pd.read_table(user_dir+'usercount_'+file.split('/')[7], sep=',',header=None,names=['Time' ,'ucount'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_user['Time'] = pd.to_datetime(df_user['Time'])
        df_user['Time'] =  pd.DatetimeIndex(df_user['Time'])
        df_user = df_user.set_index('Time')
        df_msg_rmbot = pd.read_table(msg_rmbot_dir+'msgcount_rmbot_'+file.split('/')[7], sep=',',header=None,names=['Time' ,'mcount'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_msg_rmbot['Time'] = pd.to_datetime(df_msg_rmbot['Time'])
        df_msg_rmbot['Time'] =  pd.DatetimeIndex(df_msg_rmbot['Time'])
        df_msg_rmbot = df_msg_rmbot.set_index('Time')
        df_user_rmbot = pd.read_table(user_rmbot_dir+'usercount_rmbot_'+file.split('/')[7], sep=',',header=None,names=['Time' ,'ucount'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_user_rmbot['Time'] = pd.to_datetime(df_user_rmbot['Time'])
        df_user_rmbot['Time'] =  pd.DatetimeIndex(df_user_rmbot['Time'])
        df_user_rmbot = df_user_rmbot.set_index('Time')
        df_user['mcount'] = df_msg['mcount']
        df_user['ucount_botrm'] = df_user_rmbot['ucount']
        df_user['mcount_botrm'] = df_msg_rmbot['mcount']
        df_user['pattern'] = df_user.index.isin(df_game['Time'])
        temp  = df_user[df_user['pattern']==True]
        temp['msg_per_user'] = temp['mcount']/temp['ucount']
        temp['msg_per_user_botrm']= temp['mcount_botrm']/temp['ucount_botrm']
        temp.to_csv(outfile,sep=',')


def merge_game_text():
    dir = '/l/nx/data/twitch_anonymize/icwsm/'
    channel_files = pd.read_csv(dir+'channels_1day_1000msg_100users.log',names=['room','file'])
    out_dir = dir+'five_mins_merge_game_attr_text/'#five_mins_merge_game_text/'
    out_dir2 = dir+'five_mins_merge_game_ucount_greater_one_text/'
    game_dir = dir+'games_per_room/'
    msg_dir = dir+'five_mins_msgcount/'
    user_rmbot_dir = dir+'five_mins_usercount_rmbot/'
    text_rmbot_dir = dir+'five_mins_text_rmbot/'
    all_data = []
    for file in channel_files['file'].tolist():
        outfile = out_dir+file.split('/')[7]
        outfile2 = out_dir2+file.split('/')[7]
        df_game = pd.read_table(game_dir+'game'+file.split('/')[7], sep=',',header=None,names=['Time'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_msg = pd.read_table(msg_dir+'msgcount_'+file.split('/')[7], sep=',',header=None,names=['Time' ,'mcount'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_msg['Time'] = pd.to_datetime(df_msg['Time'])
        df_msg['Time'] =  pd.DatetimeIndex(df_msg['Time'])
        df_msg = df_msg.set_index('Time')
        df_user_rmbot = pd.read_table(user_rmbot_dir+'usercount_rmbot_'+file.split('/')[7], sep=',',header=None,names=['Time' ,'ucount'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_user_rmbot['Time'] = pd.to_datetime(df_user_rmbot['Time'])
        df_user_rmbot['Time'] =  pd.DatetimeIndex(df_user_rmbot['Time'])
        df_user_rmbot = df_user_rmbot.set_index('Time')
        df_text_rmbot = pd.read_table(text_rmbot_dir+'text_rmbot_'+file.split('/')[7], sep='\x1e',header=None,names=['Time' ,'text_botrm'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_text_rmbot['Time'] = pd.to_datetime(df_text_rmbot['Time'])
        df_text_rmbot['Time'] =  pd.DatetimeIndex(df_text_rmbot['Time'])
        df_text_rmbot = df_text_rmbot.set_index('Time')
        df_msg['ucount_botrm'] = df_user_rmbot['ucount']
        df_msg['text_botrm'] = df_text_rmbot['text_botrm']
        df_msg['pattern'] = df_msg.index.isin(df_game['Time'])
        temp  = df_msg[df_msg['pattern']==True]
        temp.reset_index()
        temp[['mcount','ucount_botrm','text_botrm']].to_csv(outfile,sep='\x1e',index=False)
        temp2 = temp[temp['ucount_botrm']>1]
        temp2[['mcount','text_botrm']].to_csv(outfile2,sep='\x1e',index=False)

def merge_game_userlist():
    dir = '/l/nx/data/twitch_anonymize/icwsm/'
    channel_files = pd.read_csv(dir+'channels_1day_1000msg_100users.log',names=['room','file'])
    out_dir = dir+'five_mins_merge_game_userlist/'
    game_dir = dir+'games_per_room/'
    msg_dir = dir+'five_mins_msgcount/'
    user_dir = dir+'five_mins_usercount/'
    msg_rmbot_dir = dir+'five_mins_msgcount_rmbot/'
    user_rmbot_dir = dir+'five_mins_usercount_rmbot/'
    ulist_rmbot_dir = dir+'five_mins_userslist_rmbot/'
    all_data = []
    for file in channel_files['file'].tolist():
        outfile = out_dir+file.split('/')[7]
        df_game = pd.read_table(game_dir+'game'+file.split('/')[7], sep=',',header=None,names=['Time'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_msg = pd.read_table(msg_dir+'msgcount_'+file.split('/')[7], sep=',',header=None,names=['Time' ,'mcount'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_msg['Time'] = pd.to_datetime(df_msg['Time'])
        df_msg['Time'] =  pd.DatetimeIndex(df_msg['Time'])
        df_msg = df_msg.set_index('Time')
        df_user_rmbot = pd.read_table(user_rmbot_dir+'usercount_rmbot_'+file.split('/')[7], sep=',',header=None,names=['Time' ,'ucount'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_user_rmbot['Time'] = pd.to_datetime(df_user_rmbot['Time'])
        df_user_rmbot['Time'] =  pd.DatetimeIndex(df_user_rmbot['Time'])
        df_user_rmbot = df_user_rmbot.set_index('Time')
        df_ulist_rmbot = pd.read_table(ulist_rmbot_dir+'users_rmbot_'+file.split('/')[7], sep='\x1e',header=None,names=['Time' ,'ulist_botrm'],quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_ulist_rmbot['Time'] = pd.to_datetime(df_ulist_rmbot['Time'])
        df_ulist_rmbot['Time'] =  pd.DatetimeIndex(df_ulist_rmbot['Time'])
        df_ulist_rmbot = df_ulist_rmbot.set_index('Time')
        df_msg['ucount_botrm'] = df_user_rmbot['ucount']
        df_msg['ulist_botrm'] = df_ulist_rmbot['ulist_botrm']
        df_msg['pattern'] = df_msg.index.isin(df_game['Time'])
        temp  = df_msg[df_msg['pattern']==True]
        temp.reset_index()
        temp[['mcount','ulist_botrm']].to_csv(outfile,sep='\x1e',index=False)

if __name__ == "__main__":
    #read_channels()
    #merge_game()
    #merge_game_text()
    merge_game_userlist()
   
