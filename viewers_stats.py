import re
import pickle
import os
import operator
import csv
import numpy as np
import pandas as pd
import sys
import zlib
from datetime import datetime
from pylab import *
from datetime import datetime


#/**** this function produces three files: for each users, total number of messages, total number of room, total number of day **************/ 
def viewer_stats(data,odir,tw):
    cols = ['Time' ,'Room','User','Text']
    viewer_msg_list = []
    viewer_room_list = []
    viewer_day_list = []
    for file in os.listdir(data_dir):
        if file.startswith(tw[0]) or file.startswith(tw[1]):
            df = pd.read_table(data_dir+file, sep='\x1e',header=None,names=cols,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
            #from datetime import datetime
            datetime.fromtimestamp
            df['Timestamp'] = df['Time'].map(datetime.fromtimestamp)
            df['MD'] = df.Timestamp.map(lambda x: x.strftime('%m-%d'))
            df_groupby_viewer = df.groupby('User')
            viewer_msg_count = pd.DataFrame({'msg_count' : df_groupby_viewer.size()}).reset_index()
            viewer_room_count = pd.DataFrame({'room_count' : df_groupby_viewer['Room'].apply(lambda x: len(x.unique()))}).reset_index()
            viewer_day_count =  pd.DataFrame({'day_count' : df_groupby_viewer['MD'].apply(lambda x: len(x.unique()))}).reset_index()
            viewer_msg_list.append(viewer_msg_count)
            viewer_room_list.append(viewer_room_count)
            viewer_day_list.append(viewer_day_count)
    print 'done reading'
    msg_frame = pd.concat(viewer_msg_list)
    room_frame = pd.concat(viewer_room_list)
    day_frame = pd.concat(viewer_day_list)
    viewers_msg_count = msg_frame.groupby('User').sum()
    viewers_msg_count.to_csv(odir+'viewers_msg_count.csv',header=False)
    viewers_room_count = room_frame.groupby('User').sum()
    viewers_room_count.to_csv(odir+'viewers_user_count.csv',header=False)
    viewers_day_count = day_frame.groupby('User').sum()
    viewers_day_count.to_csv(odir+'viewers_day_count.csv',header=False)

#/**** this function produces two files: for each users, compression ratio, number of messages per second **************/ 
def viewer_split(data_dir,out_dir,tw):
    cols = ['Time' ,'Room','User','Text']
    all_data_list = []
    for file in os.listdir(data_dir):
        if file.startswith(tw[0]) or file.startswith(tw[1]):
            df = pd.read_table(data_dir+file, sep='\x1e',header=None,names=cols,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
            #from datetime import datetime
            datetime.fromtimestamp
            df['Timestamp'] = df['Time'].map(datetime.fromtimestamp)
            all_data_list.append(df) 
    print 'reading is done'
    all_data_df = pd.concat(all_data_list)
    data_groupby_user = all_data_df.groupby('User')
    del all_data_df
    count = 0
    for user in data_groupby_user:
        count +=1
        df_user = pd.DataFrame(user[1])
        df_user.drop(['Time','User'], axis=1, inplace=True)
        df_user.to_csv(out_dir+'viewers_data'+str(count/1000000)+'/'+str(user[0])+'.log',sep='\x1e',header=False,index=False)

def get_ratio(data_dir,outfile):
    cols = ['Text']
    for file in os.listdir(data_dir):
        df_reader = pd.read_table(data_dir+file, sep='\x1e',header=None,names=cols,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None,usecols=[1])
        df_reader['Text'].to_csv('temp.log',index=False,header=False)
        list_ratio = []
        with open('temp.log', "rb") as f:
            byte = f.read(10000)
            msg_text_zip = zlib.compress(byte)
            ratio = float(len(bytes(msg_text_zip))-6)/ len(bytes(byte))
            list_ratio.append(ratio)
            while byte != "":
                byte = f.read(10000)
                if len(bytes(byte)) ==0:#!= 10000:
                    pass
                else:
                    msg_text_zip = zlib.compress(byte)
                    ratio = float(len(bytes(msg_text_zip))-6)/ len(bytes(byte))
                    list_ratio.append(ratio)
        user = file.strip('.log')
        outfile.write(str(user)+','+str(np.mean(list_ratio))+','+str(stats.sem(list_ratio))+'\n')

def get_speed(data_dir,fout):
    cols = ['Time']
    total_byte = 0
    string_temp = ''
    ratio = []
    #fout = open(out_dir+'viewers_speed.csv','w')
    for file in os.listdir(data_dir):
        user  = file.strip('.log')
        df = pd.read_table(data_dir+file, sep='\x1e',header=None,names=cols,quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None,usecols=[2])
        df['SortedTime'] = pd.to_datetime(df['Time'])
        df.sort('SortedTime',inplace=True)
        df['Delta'] = (df['SortedTime']-df['SortedTime'].shift()).fillna(0)
        df['ans'] = df['Delta'].apply(lambda x: x  / np.timedelta64(1,'s')).astype('int64')
        df_lag = df[df['ans']<3600]
        sumDelta_user = df_lag['ans'].sum()#.astype(float)/math.pow(10,9)).sum()
        sizeDelta_user = len(df_lag['Delta'])
        if sumDelta_user !=0:
            lambda_user= float(sumDelta_user)/sizeDelta_user
            print user , lambda_user
            fout.write(str(user)+','+str(lambda_user)+'\n')       
    #fout.close() 

def main_viewer_stats():
    data_dir = '/scratch/azadnema/twitch_anonymize/data/'
    out_dir = '/scratch/azadnema/twitch_anonymize/results/viewers/'
    tw = ['tmi_firehose_2014-10','tmi_firehose_2014-11']
    viewer_stats(data_dir,out_dir,tw)

def main_viewer_split(data_dir,out_dir,tw):
    data_dir = '/scratch/azadnema/twitch_anonymize/data/'
    out_dir = '/scratch/azadnema/twitch_anonymize/results/viewers/'
    tw = ['tmi_firehose_2014-10','tmi_firehose_2014-11']
    viewer_split(data_dir,out_dir,tw)

def main_get_ratio():
    data_dir = '/scratch/azadnema/twitch_anonymize/results/viewers/viewers_data'
    out_file = open('/scratch/azadnema/twitch_anonymize/results/viewers/viewers_ratio.csv','w')
    for i in range(0,7):
        get_ratio(data_dir+str(i)+'/',out_file)

def main_get_speed():
    data_dir = '/scratch/azadnema/twitch_anonymize/results/viewers/viewers_data'
    out_file = open('/scratch/azadnema/twitch_anonymize/results/viewers/viewers_speed.csv','w')
    tw = ['tmi_firehose_2014-10','tmi_firehose_2014-11']
    for i in range(0,7):
        get_speed(data_dir+str(i)+'/',out_file)

def viewer_vector():
    vdir = '/scratch/azadnema/twitch_anonymize/results/viewers/'
    viewer_day_df = pd.read_csv(vdir+'viewers_day_count.csv',names=['user','day'])
    viewer_msg_df = pd.read_csv(vdir+'viewers_msg_count.csv',names=['user','msg'])
    viewer_ratio_df = pd.read_csv(vdir+'viewers_ratio.csv',names=['user','ratio','err'])
    viewer_speed_df = pd.read_csv(vdir+'viewers_speed.csv',names=['user','speed'])
    viewer_df = pd.read_csv(vdir+'viewers_user_count.csv',names=['user','room'])
    viewer_df['msg']  = viewer_msg_df['msg']
    viewer_df['day'] = viewer_day_df['day']
    viewer_df['ratio'] = viewer_ratio_df['ratio']
    viewer_df['speed'] = viewer_speed_df['speed']
    temp = viewer_df[viewer_df['day']>1]
    temp[temp['msg']>10].to_csv('/u/azadnema/twitch/code/data/viewer_vector_msg10_day1.csv',index=False)


if __name__ == "__main__":
    viewer_vector()
