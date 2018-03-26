import re
import pickle
import os
import operator
import csv
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from pylab import *

def room_stats(data,odir,tw):
    cols = ['Time' ,'Room','User','Text']
    room_msg_list = [] 
    room_user_list = []
    room_day_list = []
    for file in os.listdir(data_dir):
        if file.startswith(tw[0]) or file.startswith(tw[1]):
            df = pd.read_table(data_dir+file, sep='\x1e',header=None,names=cols,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
            from datetime import datetime
            datetime.fromtimestamp
            df['Timestamp'] = df['Time'].map(datetime.fromtimestamp)
            df['MD'] = df.Timestamp.map(lambda x: x.strftime('%m-%d')) 
            df_groupby_room = df.groupby('Room')
            room_msg_count = pd.DataFrame({'msg_count' : df_groupby_room.size()}).reset_index()
            room_user_count = pd.DataFrame({'user_count' : df_groupby_room['User'].apply(lambda x: len(x.unique()))}).reset_index()
            room_day_count =  pd.DataFrame({'day_count' : df_groupby_room['MD'].apply(lambda x: len(x.unique()))}).reset_index()
            room_msg_list.append(room_msg_count)
            room_user_list.append(room_user_count)
            room_day_list.append(room_day_count)
    print 'done reading'
    msg_frame = pd.concat(room_msg_list)
    user_frame = pd.concat(room_user_list)
    day_frame = pd.concat(room_day_list)
    rooms_msg_count = msg_frame.groupby('Room').sum()
    rooms_msg_count.to_csv(odir+'rooms_msg_count.csv',header=False)
    rooms_user_count = user_frame.groupby('Room').sum()
    rooms_user_count.to_csv(odir+'rooms_user_count.csv',header=False)
    rooms_day_count = day_frame.groupby('Room').sum()
    rooms_day_count.to_csv(odir+'rooms_day_count.csv',header=False)

def room_botrm_stats(data_dir,out_dir,tw):
    bot_df = pd.read_table('/u/azadnema/twitch/code/data/botts_msg100_day1.csv',sep=',')
    bot_list = bot_df['user'].tolist()
    cols = ['Time' ,'Room','User','Text']
    room_msg_list = []
    room_user_list = []
    room_day_list = []
    for file in os.listdir(data_dir):
        if file.startswith(tw[0]) or file.startswith(tw[1]):
            df = pd.read_table(data_dir+file, sep='\x1e',header=None,names=cols,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
            from datetime import datetime
            datetime.fromtimestamp
            df['Timestamp'] = df['Time'].map(datetime.fromtimestamp)
            df['MD'] = df.Timestamp.map(lambda x: x.strftime('%m-%d'))
            df['bot'] = df['User'].isin(bot_list)
            df = df[df['bot']==False]
            df_groupby_room = df.groupby('Room')
            room_msg_count = pd.DataFrame({'msg_count' : df_groupby_room.size()}).reset_index()
            room_user_count = pd.DataFrame({'user_count' : df_groupby_room['User'].apply(lambda x: len(x.unique()))}).reset_index()
            room_day_count =  pd.DataFrame({'day_count' : df_groupby_room['MD'].apply(lambda x: len(x.unique()))}).reset_index()
            room_msg_list.append(room_msg_count)
            room_user_list.append(room_user_count)
            room_day_list.append(room_day_count)
    print 'done reading'
    msg_frame = pd.concat(room_msg_list)
    user_frame = pd.concat(room_user_list)
    day_frame = pd.concat(room_day_list)
    rooms_msg_count = msg_frame.groupby('Room').sum()
    rooms_msg_count.to_csv(odir+'rooms_msg_count_botrm.csv',header=False)
    rooms_user_count = user_frame.groupby('Room').sum()
    rooms_user_count.to_csv(odir+'rooms_user_count_botrm.csv',header=False)
    rooms_day_count = day_frame.groupby('Room').sum()
    rooms_day_count.to_csv(odir+'rooms_day_count_botrm.csv',header=False)

def channel_split(data_dir,out_dir,tw):
    cols = ['Time' ,'Room','User','Text']
    all_data_list = []
    for file in os.listdir(data_dir):
        if file.startswith(tw[0]) or file.startswith(tw[1]):
            df = pd.read_table(data_dir+file, sep='\x1e',header=None,names=cols,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
            from datetime import datetime
            datetime.fromtimestamp
            df['Timestamp'] = df['Time'].map(datetime.fromtimestamp)
            all_data_list.append(df)
    all_data_df = pd.concat(all_data_list)
    data_groupby_room = all_data_df.groupby('Room')
    del all_data_df
    count = 0
    for room in data_groupby_room:
        count +=1
        df_room = pd.DataFrame(room[1])
        df_room.drop(['Time','Room'], axis=1, inplace=True)
        df_room.to_csv(out_dir+'channels_data'+str(count/100000)+'/'+str(room[0])+'.log',sep='\x1e',header=False,index=False)

if  __name__ == "__main__":
    data_dir = '/scratch/azadnema/twitch_anonymize/data/'
    out_dir = '/scratch/azadnema/twitch_anonymize/results/channels/'
    tw = ['tmi_firehose_2014-10','tmi_firehose_2014-11']
    #room_stats(data_dir,out_dir,tw)
    #room_botrm_stats(data_dir,out_dir,tw)
    channel_split(data_dir,out_dir,tw)
