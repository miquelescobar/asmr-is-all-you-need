#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:39:41 2020

@author: jordi
"""

from data_utils import extractor


#%%

token_path = "/home/jordi/Documents/UPC/6e quatri/POE/Projecte/Data/token.pickle"

my_extractor = extractor(token_path)

#%%

asmr_bakery = "UCDH70xAv1bBeaCt5Cc7a96A" #https://www.youtube.com/channel/UCDH70xAv1bBeaCt5Cc7a96A

videos_asmr_bakery = my_extractor.getVideosUser(asmr_bakery)

#%%
#Download one video

num_video = 15

my_extractor.downloadVideo(videos_asmr_bakery[num_video]['id'])

#%%

#Split the video in intervals

videoId = videos_asmr_bakery[num_video]['id']
timestamps = videos_asmr_bakery[num_video]['timestamps']


my_extractor.createPartitions(videoId, timestamps, PATH = './audio', remove = False)