import os
import zlib
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

def split_text():
    channel_files = pd.read_csv('/scratch/azadnema/twitch_anonymize/results/channels/channels_1day_1000msg_100users.log',names=['room','file'])
    gameroom_dir = '/scratch/azadnema/twitch_anonymize/results/channels/five_mins_merge_game_ucount_greater_one_text/'#five_mins_merge_game_text/'
    split_dir = '/scratch/azadnema/twitch_anonymize/results/channels/msg_text_greater_one/'#msg_text/'
    all_data = []
    for file in channel_files['file'].tolist():
        df_game = pd.read_table(gameroom_dir+file.split('/')[7], sep='\x1e',quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        all_data.append(df_game)
    df_all_data = pd.concat(all_data)
    df_all_data['mcount'] = df_all_data['mcount'].astype(float)
    df_all_data.dropna(inplace=True)
    df_all_data_grouped = df_all_data.groupby('mcount')['text_botrm'].apply(lambda x: '\x1f'.join(x))
    for k, v in df_all_data_grouped.iteritems():
        print k
        outfile  = open(split_dir+str(k)+'.log','w')
        outfile.write(v)

def read_split():
    split_dir = '/scratch/azadnema/twitch_anonymize/results/channels/msg_text/'
    count = 0
    for file in os.listdir(split_dir):
     text_list = []
     if count < 10:
        count += 1
        print file
        ofile = open(split_dir+file,'r')
        for i in ofile:
            text_list += i.split('\x1f')[0:10]

def ratio():
    channel_files = pd.read_csv('/scratch/azadnema/twitch_anonymize/results/channels/channels_1day_1000msg_100users.log',names=['room','file'])
    gameroom_dir = '/scratch/azadnema/twitch_anonymize/results/channels/five_mins_merge_game_ucount_greater_one_text/'#five_mins_merge_game_text/'
    outfile = open('/scratch/azadnema/twitch_anonymize/results/channels/msgcount_ratio_err_room_greater_one.csv','w')
    outfile2 = open('/scratch/azadnema/twitch_anonymize/results/channels/msgcount_ratio_greater_one.csv','w')
    for file in channel_files['file'].tolist():
        df_game = pd.read_table(gameroom_dir+file.split('/')[7], sep='\x1e',quoting = csv.QUOTE_NONE,lineterminator='\n', na_filter=None)
        df_game['mcount'] = df_game['mcount'].astype(float)
        df_game.dropna(inplace=True)
        df_game_grouped = df_game.groupby('mcount')['text_botrm'].apply(lambda x: '\x1f'.join(x))
        for k, v in df_game_grouped.iteritems():
            tempfile  = open('test.log','w')
            tempfile.write(v)
            tempfile.close()
            list_ratio = []
            with open('test.log', "rb") as f:
                byte = f.read(10000)
                msg_text_zip = zlib.compress(byte)
                ratio = float(len(bytes(msg_text_zip))-6)/ len(bytes(byte))
                if ratio > 1:
                    pass
                else:
                    list_ratio.append(ratio)
                while byte != "":
                    byte = f.read(10000)
                    if len(bytes(byte)) < 1000:
                        pass
                    else:
                        msg_text_zip = zlib.compress(byte)
                        ratio = float(len(bytes(msg_text_zip))-6)/ len(bytes(byte))
                        list_ratio.append(ratio)
            outfile.write(str(k)+','+str(np.mean(list_ratio))+','+str(stats.sem(list_ratio))+','+str(file.split('/')[7].strip('.log'))+'\n')
            outfile2.write(str(k)+','+str(np.mean(list_ratio))+'\n')

def  qmark_mentioned_len(split_dir,outfile):
    #split_dir = '/scratch/azadnema/twitch_anonymize/results/channels/msg_text_greater_one/'
    #outfile = open('/scratch/azadnema/twitch_anonymize/results/channels/msgcount_qmark_mentioned_len_greater_one.csv','w')
    for file in os.listdir(split_dir):
        text_list = []
        world_list = []
        ofile = open(split_dir+file,'r')
        for i in ofile:
            text_list += i.split('\x1f')
            world_list += i.strip().replace(' ','\x1f').split('\x1f')
        msg_speed = file.split('.')[0]
        msg_count = len(text_list)
        qcount = 0
        for s in filter (lambda x: x.endswith('?'), text_list):qcount+=1 
        qmark_prob = float(qcount)/msg_count
        mcount = 0
        for s in filter (lambda x: x.startswith('@') and len(x)>1, world_list):mcount+=1
        mentioned_prob = float(mcount)/msg_count
        charcount = 0
        for i in world_list:
            charcount += len(i)
        char_prob = float(charcount)/len(world_list)
        outfile.write(str(msg_speed)+','+str(qmark_prob)+','+str(mentioned_prob)+','+str(char_prob)+'\n')
    outfile.close()

def discours_marker(dfile,split_dir,ofile):
    #dfile = '/u/azadnema/twitch/code/data/discoursMarker.txt'
    #split_dir = '/scratch/azadnema/twitch_anonymize/results/channels/msg_text_greater_one/'
    #ofile = '/scratch/azadnema/twitch_anonymize/results/channels/msgcount_discourse_marker_greater_one.csv'
    dmarkers = open(dfile,'r')
    dmarkers_list = []
    for dmarker in dmarkers:
        dmarkers_list.append(dmarker.strip())
    outfile = open(ofile,'w')
    for file in os.listdir(split_dir):
        text_list = []
        world_list = []
        ofile = open(split_dir+file,'r')
        for i in ofile:
            text_list += i.split('\x1f')
            world_list += i.strip().replace(' ','\x1f').lower().split('\x1f')
        msg_speed = file.split('.')[0]
        msg_count = len(text_list)
        word_count = len(world_list)
        dmarker_count_list = []
        msg_word_count = collections.Counter(world_list)
        for dmarker in set(dmarkers_list):
            dmarker_count_list.append(float(msg_word_count[dmarker])/word_count)
        mean_marker_prob = np.mean(dmarker_count_list)
        er_marker_prob = stats.sem(dmarker_count_list)
        outfile.write(str(msg_speed)+','+str(mean_marker_prob)+','+str(er_marker_prob)+'\n')

def emote():
    comon_emote = open('/u/azadnema/twitch/code/data/emote_common.txt','r')
    subscriber_emote = open('/u/azadnema/twitch/code/data/emotes_subscription.txt','r')
    outfile = open('/scratch/azadnema/twitch_anonymize/results/channels/msgcount_emote_greater_one_chunck_xaa.csv','w')
    emote_dic = {}
    len_emote_dic = {}
    for i in comon_emote:
        emote_dic[i.strip()]=0
        len_emote = len(i.strip())
        if not len_emote_dic.has_key(len_emote):
            len_emote_dic[len_emote] = [i.strip(),]
        else:
            len_emote_dic[len_emote].append(i.strip())
    for i in subscriber_emote:
        emote_dic[i.strip()]=0
        len_emote = len(i.strip())
        if not len_emote_dic.has_key(len_emote):
            len_emote_dic[len_emote] = [i.strip(),]
        else:
            len_emote_dic[len_emote].append(i.strip())
    count_test = 0
    files_df = pd.read_table('xaa',names=['file'])
    files_list = files_df['file'].tolist()
    for file in files_list :#os.listdir(split_dir):
        text_list = []
        word_list = []
        ofile = open(file,'r')
        for i in ofile:
            text_list += i.split('\x1f')
            word_list += i.strip().replace(' ','\x1f').lower().split('\x1f')
        msg_speed = file.split('.')[0]
        msg_count = len(text_list)
        word_count = len(word_list)
        all_shingles = []
        for shingleLength in len_emote_dic.keys():
            for token in word_list:
                all_shingles += [token[i:i+shingleLength] for i in range(len(token) - shingleLength + 1) ]
        all_shingle_count = collections.Counter(all_shingles)
        all_shingles_len = len(all_shingles)
        emote_list = []
        for shingle in all_shingle_count:
            if  emote_dic.has_key(shingle):
                emote_list.append(float(all_shingle_count[shingle])/all_shingles_len)
        mean_emote_prob = np.mean(emote_list)
        er_emote_prob = stats.sem(emote_list)
        outfile.write(str(msg_speed)+','+str(mean_emote_prob)+','+str(er_emote_prob)+'\n')


if __name__ == '__main__':
    #split_text()
    #read_split()
    #qmark_mentioned_len()
    #discours_marker()
    #emote()
    ratio()
    
