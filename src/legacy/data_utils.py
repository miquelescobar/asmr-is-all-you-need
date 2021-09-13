import os
import pickle
import youtube_dl
from scipy.io import wavfile
from os import system

import google.oauth2.credentials

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


class extractor(object):
    def __init__(self, token_path = None, CLIENT_SECRETS_FILE = None, PATH_AUDIO = './audio'):    
        SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
        API_SERVICE_NAME = 'youtube'
        API_VERSION = 'v3'
        credentials = None
        if token_path is not None:
            with open(token_path, 'rb') as token:
                credentials = pickle.load(token)
        
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_console()
            #ave the credentials for the next run
            with open('token.pickle', 'wb') as token:
            	pickle.dump(credentials, token)
                
                
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
        self.service = build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

        self.ydl_opts = {    
            'format': 'bestaudio/best',
            'outtmpl': f'{PATH_AUDIO}/%(id)s_full.webm',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'prefer_ffmpeg': True
        }


    
    def getVideosUser(self, channel_id):

    	response = self.service.channels().list(
                part = 'contentDetails',
                id = channel_id,
                ).execute()

    	upload_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    	videos_uploaded = self._videosPlaylist(upload_playlist_id)

    	videos_features = [self._videoFeatures(video) for video in videos_uploaded]

    	return videos_features
    
    
    def downloadVideo(self, video_id):
        with youtube_dl.YoutubeDL(self.ydl_opts) as  ydl:
            ydl.download([video_id])
    
    def createPartitions(self, videoId, timestamps, PATH = './audio', remove = False):
        
        filename = f'{PATH}/{videoId}_full.wav'
        sr, sample = wavfile.read(filename)
        
        prev_time = 0
        prev_tag = None
        
        final_timestamps = timestamps.copy()
        final_timestamps.append(('End', 'End'))
            
        for time, tag in final_timestamps:
            
            if prev_tag is not None:
                prev_tag = prev_tag.replace(' ', '-')
                output_filename = f'{PATH}/{videoId}_{prev_tag}.wav'
                
                init = self._timeToIndex(prev_time, sr)
                final = self._timeToIndex(time, sr)
                
                interval = sample[init:final]
                
                wavfile.write(output_filename, sr, interval)
                
            prev_tag = tag
            prev_time = time
            
        if remove:
            system(f'rm {filename}')
            
    def _timeToIndex(self, time, sr):
    
        if time == 'End': #Last timestamp lasts till the end of the video 
            return None
        
        times = list(map(int, time.split(':')))
        
        seconds = 0
        for i, time in enumerate(times[::-1]):
            seconds += time * (60**i)
        
        index = seconds * sr
        
        return index
    
    def _videosPlaylist(self, playlist_key):
    
	    videos = []
	    next_page_token = None
	    
	    while True:
	        res = self.service.playlistItems().list(
	                playlistId = playlist_key,
	                part = 'contentDetails',
	                maxResults = 50,
	                pageToken = next_page_token
	                ).execute()
	        
	        videos += [video['contentDetails']['videoId'] for video in res['items']]
	        next_page_token = res.get('nextPageToken') #If there isnt more pages returns None
	        
	        if next_page_token is None:
	            break
	        
	    return videos
    
    def _videoFeatures(self, video_key):
        res = self.service.videos().list(
                part = 'snippet, contentDetails',
                id = video_key,
                ).execute()
        
        video = {}
        
        video['id'] = video_key
        video['title'] = res['items'][0]['snippet'].get('title')
        video['description'] = res['items'][0]['snippet'].get('description')
        video['channelId'] = res['items'][0]['snippet'].get('channelId')
        video['duration'] = res['items'][0]['contentDetails'].get('duration')
        
        video['timestamps'] = self._timestamps(video['description'])
        
        return video
    
    
    def _timestamps(self, description):
    
	    intervals = []
	    
	    for line in description.splitlines():
	        if line[0:2].isdigit() and line[2] == ':' and line[3:5].isdigit():
	            time = line[:5]
	            
	            i = len(line)
	            char = 'a'
	            while not char.isdigit() and i >= 0:
	                i -= 1
	                char = line[i]
	            
	            i += 1 if i < len(line)-1 else 0  
	            
	            while line[i] == ' ' and i < len(line):
	                i += 1
	                
	            tag = line[i:]
	            
	            interval = (time, tag)
	            
	            intervals.append(interval)
	            
	    return intervals