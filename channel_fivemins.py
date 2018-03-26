from scipy import stats
import os
import operator
import csv
import numpy as np
import pandas as pd
from pylab import *
from random import sample

def perRoomTS_msgCount(file,outfile):
    col = ['User','Text','Time']
    df = pd.read_table(file, sep='\x1e',header=None,names=col,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
    df['SortedTime'] = pd.to_datetime(df['Time'])
    df.sort('SortedTime',inplace=True)
    df['SortedTime'] =  pd.DatetimeIndex(df['SortedTime'])
    df = df.set_index('SortedTime')
    tsw = df.groupby(pd.TimeGrouper('5Min')).size().dropna(0)
    tsw.to_csv(outfile,sep=',',header=False)

def perRoomTS_usercount(file,outfile,bot_flag):
    usrcol = ['User','Text','Time']
    df = pd.read_table(file, sep='\x1e',header=None,names=usrcol,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
    df['SortedTime'] = pd.to_datetime(df['Time'])
    df.sort('SortedTime',inplace=True)
    df['SortedTime'] =  pd.DatetimeIndex(df['SortedTime'])
    df = df.set_index('SortedTime')
    if bot_flag == True:
        tsw = dfw.groupby(pd.TimeGrouper('5Min')).agg(lambda x: len(x[x['bot']==False]['User'].unique()))
    else:
        tsw = df.groupby(pd.TimeGrouper('5Min'))['User'].apply(lambda x: len(x.unique()))#sum().fillna(0)
    tsw.to_csv(outfile,sep=',',header=False)

def perRoomTS_userslist(file,outfile,bot_flag):
    usrcol = ['User','Text','Time']
    df = pd.read_table(file, sep='\x1e',header=None,names=usrcol,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
    df['SortedTime'] = pd.to_datetime(df['Time'])
    df.sort('SortedTime',inplace=True)
    df['SortedTime'] =  pd.DatetimeIndex(df['SortedTime'])
    df = df.set_index('SortedTime')
    if bot_flag == True:
        tsw = dfw.groupby(pd.TimeGrouper('5Min')).agg(lambda x: len(x[x['bot']==False]['User'].unique()))
    else:
        tsw = df.groupby(pd.TimeGrouper('5Min'))['User'].apply(lambda x: list(x))#sum().fillna(0)
    tsw.to_csv(outfile,sep=',',header=False)

def perRoomTS_text(file,outfile,bot_flag):
    usrcol = ['User','Text','Time']
    df = pd.read_table(file, sep='\x1e',header=None,names=usrcol,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
    df['SortedTime'] = pd.to_datetime(df['Time'])
    df.sort('SortedTime',inplace=True)
    df['SortedTime'] =  pd.DatetimeIndex(df['SortedTime'])
    df = df.set_index('SortedTime')
    if bot_flag == True:
        tsw = dfw.groupby(pd.TimeGrouper('5Min')).agg(lambda x: len(x[x['bot']==False]['User'].unique()))
    else:
        tsw = df.groupby(pd.TimeGrouper('5Min'))['Text'].apply(lambda x:'\x1f'.join(x)).reset_index() 
    tsw.to_csv(outfile,sep='\x1e',header=False)

def read_channels(dir ,compute_what):
    channel_files = pd.read_csv(dir+'channels_1day_1000msg_100users.log',names=['room','file'])
    for file in channel_files['file'].tolist():
        print file.split('/')[7]
        if compute_what == 'msgcount':
            outfile = dir+'five_mins_msgcount/msgcount_'+file.split('/')[7]
            perRoomTS_msgCount(file,outfile)
        if compute_what == 'usercount' :
            outfile = dir+'five_mins_stat_usercount/usercount_'+file.split('/')[7]
            perRoomTS_usercount(file,outfile)
        if compute_what == 'text':
            outfile = dir + 'five_mins_stat_text/text_'+file.split('/')[7]
            perRoomTS_text(file,outfile)
        if compute_what == 'userlist':
            outfile = dir+'five_mins_stat_userslist/users_'+file.split('/')[7]
            perRoomTS_userslist(file,outfile)

def perRoomTS_msgCount_rmbot(file,outfile):
    botfile =  '/u/azadnema/twitch/code/data/botts_msg100_day1.csv'
    botcol = ['user','room','msg','day','speed','ratio','log']
    bot_df = pd.read_csv(botfile,names = botcol,skiprows=[0])
    bot_list = bot_df['user'].tolist()

    col = ['User','Text','Time']
    df = pd.read_table(file, sep='\x1e',header=None,names=col,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
    df['SortedTime'] = pd.to_datetime(df['Time'])
    df.sort('SortedTime',inplace=True)
    df['SortedTime'] =  pd.DatetimeIndex(df['SortedTime'])
    df = df.set_index('SortedTime')
    df['bot'] = df['User'].isin(bot_list)#.astype(int)
    df['human'] = ~df['bot']
    df['human'] = df['human'].astype(int)
    tsw = df.groupby(pd.TimeGrouper('5Min'))#['bot'].sum().dropna(0)
    tsw_sum =tsw['human'].sum().fillna(0)
    tsw_sum.to_csv(outfile,sep=',',header=False)
    return len(df),len(df[df['bot']==True]) ,len(set(df[df['bot']==True]['User'].tolist()))

def perRoomTS_usercount_rmbot(file,outfile):
    botfile =  '/u/azadnema/twitch/code/data/botts_msg100_day1.csv'
    botcol = ['user','room','msg','day','speed','ratio','log']
    bot_df = pd.read_csv(botfile,names = botcol,skiprows=[0])
    bot_list = bot_df['user'].tolist()
    usrcol = ['User','Text','Time']
    df = pd.read_table(file, sep='\x1e',header=None,names=usrcol,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
    df['SortedTime'] = pd.to_datetime(df['Time'])
    df.sort('SortedTime',inplace=True)
    df['SortedTime'] =  pd.DatetimeIndex(df['SortedTime'])
    df = df.set_index('SortedTime')
    df['bot'] = df['User'].isin(bot_list)
    tsw = df.groupby(pd.TimeGrouper('5Min')).agg(lambda x: len(x[x['bot']==False]['User'].unique()))
    tsw['User'].to_csv(outfile,sep=',',header=False)


def perRoomTS_userslist_rmbot(file,outfile):
    botfile =  '/u/azadnema/twitch/code/data/botts_msg100_day1.csv'
    botcol = ['user','room','msg','day','speed','ratio','log']
    bot_df = pd.read_csv(botfile,names = botcol,skiprows=[0])
    bot_list = bot_df['user'].tolist()
    usrcol = ['User','Text','Time']
    df = pd.read_table(file, sep='\x1e',header=None,names=usrcol,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
    df['SortedTime'] = pd.to_datetime(df['Time'])
    df.sort('SortedTime',inplace=True)
    df['SortedTime'] =  pd.DatetimeIndex(df['SortedTime'])
    df = df.set_index('SortedTime')
    df['bot'] = df['User'].isin(bot_list)
    df_temp = df[df['bot']==False]
    tsw = df_temp.groupby(pd.TimeGrouper('5Min'))['User'].apply(lambda x: list(x))
    tsw.to_csv(outfile,sep='\x1e',header=False)

def perRoomTS_text_rmbot(file,outfile):
    botfile =  '/u/azadnema/twitch/code/data/botts_msg100_day1.csv'
    botcol = ['user','room','msg','day','speed','ratio','log']
    bot_df = pd.read_csv(botfile,names = botcol,skiprows=[0])
    bot_list = bot_df['user'].tolist()
    usrcol = ['User','Text','Time']
    df = pd.read_table(file, sep='\x1e',header=None,names=usrcol,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
    df['SortedTime'] = pd.to_datetime(df['Time'])
    df.sort('SortedTime',inplace=True)
    df['SortedTime'] =  pd.DatetimeIndex(df['SortedTime'])
    df = df.set_index('SortedTime')
    df['bot'] = df['User'].isin(bot_list)
    df_temp = df[df['bot']==False]
    tsw = df_temp.groupby(pd.TimeGrouper('5Min'))['Text'].apply(lambda x: '\x1f'.join(x))
    tsw.to_csv(outfile,sep='\x1e',header=False)

def read_channels_rmbot_msg():
    outfile_stat = open('/scratch/azadnema/twitch_anonymize/results/channels/room_botmsg_count.csv','w')
    channel_files = pd.read_csv('/scratch/azadnema/twitch_anonymize/results/channels/channels_1day_1000msg_100users.log',names=['room','file'])
    for file in channel_files['file'].tolist():
        outfile = '/scratch/azadnema/twitch_anonymize/results/channels/five_mins_stat_msgcount_rmbot/msgcount_rmbot_'+file.split('/')[7]
        msgcount,bot_msgcount ,bot_count= perRoomTS_msgCount_rmbot(file,outfile)
        outfile_stat.write(str(file.split('/')[7])+','+str(msgcount)+','+str(bot_msgcount)+','+str(bot_count)+'\n')

def read_channels_rmbot_user():
    channel_files = pd.read_csv('/scratch/azadnema/twitch_anonymize/results/channels/channels_1day_1000msg_100users.log',names=['room','file'])
    for file in channel_files['file'].tolist():
        outfile = '/scratch/azadnema/twitch_anonymize/results/channels/five_mins_stat_usercount_rmbot/usercount_rmbot_'+file.split('/')[7]
        perRoomTS_usercount_rmbot(file,outfile)
def read_channels_rmbot_text(dir):
    channel_files = pd.read_csv(dir+'channels_1day_1000msg_100users.log',names=['room','file'])
    for file in channel_files['file'].tolist():
            outfile = dir+'five_mins_text_rmbot/text_rmbot_'+file.split('/')[7]
            perRoomTS_text_rmbot(file,outfile)

def read_channels_rmbot_user():
    channel_files = pd.read_csv('/scratch/azadnema/twitch_anonymize/results/channels/channels_1day_1000msg_100users.log',names=['room','file'])
    for file in channel_files['file'].tolist():
        outfile = '/scratch/azadnema/twitch_anonymize/results/channels/five_mins_stat_usercount_rmbot/usercount_rmbot_'+file.split('/')[7]
        perRoomTS_usercount_rmbot(file,outfile)

def read_channels_rmbot_userlist(dir):
    channel_files = pd.read_csv(dir+'channels_1day_1000msg_100users.log',names=['room','file'])
    for file in channel_files['file'].tolist():
        outfile = dir+'/five_mins_userslist_rmbot/users_rmbot_'+file.split('/')[7]
        infile = '/l/nx/data/twitch_anonymize/icwsm/channels/'+file.split('/')[6]+'/'+file.split('/')[7]
        perRoomTS_userslist_rmbot(infile,outfile)

if  __name__ == "__main__":
    dir = '/l/nx/data/twitch_anonymize/icwsm/'
    #read_channels(dir)
    #read_channels_rmbot_user()
    #read_channels_rmbot_text(dir)
    read_channels_rmbot_userlist(dir)


