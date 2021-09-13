from typing import (
    Union,
    Dict,
    List
)
import os
import os.path
import logging
import subprocess
import time
import youtube_dl


def download_video(video_id: str,
                   path: str = './audio/',
                   start_time: Union[str,float] = None,
                   end_time: Union[str,float] = None,
                   duration: Union[str,float] = None,
                   filename: str = '%(id)s',
                   format: str = 'wav',
                   verbose: bool = False,
                   ydl_opts_kwargs: Dict[str,object] = None):
    """
    """
    ydl_opts_kwargs = ydl_opts_kwargs or dict()
    download_path = './tmp/' if (start_time or end_time or duration) else path
    ydl_opts = generate_ydl_options(download_path, filename, format, **ydl_opts_kwargs)
    #
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        logging.info(f'Downloading viedo with id {video_id}')
        ydl.download([video_id])
    #
    if start_time or end_time or duration: 
        command = f'ffmpeg -i {download_path}{video_id if filename=="%(id)s" else filename}.wav'
        if start_time:
            command += f' -ss {start_time}'
        if end_time:
            command += f' -to {end_time}'
        elif duration:
            command += f' -t {start_time}'
        command += f' -y {path}{video_id if filename=="%(id)s" else filename}.wav'
        #
        logging.info(f'Executing {command}')
        if verbose:
            print(f'Executing {command}')
        subprocess.call(command, shell=True)
        time.sleep(0.5)
        if verbose:
            print(f'Removing ./tmp/ directory and files')
        logging.info(f'Removing ./tmp/ directory and files')
        subprocess.call(f'rm -rf ./tmp')
        
        
        
def download_video_audios(video_id: str,
                          path: str,
                          video_timestamps: Dict[str,List[Dict[str,str]]],
                          format: str = 'wav',
                          verbose: bool = False,
                          ydl_opts_kwargs: Dict[str,object] = None):
    """
    """
    ydl_opts_kwargs = ydl_opts_kwargs or dict()
    download_path = './tmp/'
    ydl_opts = generate_ydl_options(download_path, video_id, format, **ydl_opts_kwargs)
    #
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        if verbose:
            print(f'Downloading viedo with id {video_id}')
        logging.info(f'Downloading viedo with id {video_id}')
        ydl.download([video_id])
    #
    for audio_type, timestamps in video_timestamps.items():
        if verbose:
            print(f'    Extracting audios of type {audio_type}...')
        if not os.path.isdir(f'{path}/{audio_type}'):
            logging.info(f'Creating directory "{path}/{audio_type}"')
            if verbose:
                print(f'        Creating directory "{path}/{audio_type}"')
            os.mkdir(f'{path}/{audio_type}')
        #
        for timestamp in timestamps:
            audio_path = f'{path}/{audio_type}/{video_id}-{timestamp["detail"]}.wav'
            command = f'ffmpeg -i {download_path}{video_id}.wav'
            command += f' -ss {timestamp["start_time"]}'
            command += f' -to {timestamp["end_time"]}'
            command += f' -y {audio_path}'
            #
            logging.info(f'Executing "{command}"')
            if verbose:
                print(f'        ...Executing "{command}"')
            subprocess.call(command, shell=True)
            logging.info(f'Extracted audio of type {audio_type} and detail {timestamp["detail"]}.')
            if verbose:
                print(f'        Extracted audio of type {audio_type} and detail {timestamp["detail"]}.')
            time.sleep(0.25)
        #
    time.sleep(0.25)
    if verbose:
        print(f'Removing ./tmp/ directory and files')
    logging.info(f'Removing ./tmp/ directory and files')
    subprocess.call(f'rm -rf ./tmp')        


def generate_ydl_options(path: str = './audio/',
                         filename: str = '%(id)s',
                         format: str = 'wav',
                         **kwargs) -> Dict[str, object]:
    """
    """
    ydl_opts = {    
        'format': 'bestaudio/best',
        'outtmpl': f'{path}{filename}.{format}',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': format,
            'preferredquality': '192',
        }],
        'prefer_ffmpeg': True
    }
    ydl_opts.update(kwargs)
    return ydl_opts



if __name__ == '__main__':

    download_video(video_id='2Ix8aYfmMQg',
                   path='./audio/',
                   filename='full',
                   start_time = '00:00:15.00',
                   end_time = '00:00:20.00')
    
    