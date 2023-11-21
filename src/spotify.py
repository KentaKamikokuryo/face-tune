import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from pprint import pprint
import pandas as pd
import numpy as np

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
                                                    client_id=os.environ['SPOTIFY_CLIENT_ID'], 
                                                    client_secret=os.environ['SPOTIFY_CLIENT_SECRET']), 
                     language='ja')


weekly_top_playlist_ids_dict = dict(
    Japan='37i9dQZEVXbKqiTGXuCOsB',  # 日本
    Korea='37i9dQZEVXbJZGli0rRP3r',  # 韓国
    America='37i9dQZEVXbLp5XoPON0wI',  # アメリカ
)


FILENAME = "./data/spotify_top_songs_data.csv"


use_features_list = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                     'speechiness', 'acousticness', 'instrumentalness',
                     'valence', 'tempo', 'liveness']


def _get_playlist_tracks(playlist_ids:dict, playlist_len:int):

    tracks_df = pd.DataFrame(np.zeros((playlist_len*len(playlist_ids.keys()), 5)), \
                             columns=['country', 'weekly_rank', 'artist', 'title', 'uri'])
    for i, country in enumerate(playlist_ids.keys()):
        playlist = sp.playlist(playlist_id=playlist_ids[country], market='JP')
        print(playlist['description'])
        tracks = playlist['tracks']['items']
        for j, track in enumerate(tracks):
            idx = (i*50) + j
            track_info = track['track']
            tracks_df.loc[idx, 'country'] = country
            tracks_df.loc[idx, 'weekly_rank'] = int(j+1)
            tracks_df.loc[idx, 'artist'] = track_info['artists'][0]['name']
            tracks_df.loc[idx, 'title'] = track_info['name']
            tracks_df.loc[idx, 'uri'] = track_info['uri']
            tracks_df.loc[idx, 'popularity'] = track_info['popularity']
    return tracks_df


def _get_playlist_features(tracks_df, use_features_list, chunk_n=100):
    
    tracks_df_idx_list = tracks_df.index.tolist()
    tracks_df_idx_chunk_list = [tracks_df_idx_list[i:i+chunk_n] for i in range(0, len(tracks_df_idx_list), chunk_n)]
    print(f"All:{len(tracks_df_idx_list)} -> ChunkSize:{chunk_n} | ChunkLength:{len(tracks_df_idx_chunk_list)}")

    for idx_chunk in tracks_df_idx_chunk_list:
        
        tmp_df = tracks_df.loc[idx_chunk]
        uri_list = tmp_df.uri.tolist()
        sp_features_list = sp.audio_features(tracks=uri_list)

        for idx, sp_features in zip(idx_chunk, sp_features_list):
            if sp_features:  # Check if sp_features is not None
                for feature_name in use_features_list:
                    tracks_df.loc[idx, feature_name] = sp_features[feature_name]

    return tracks_df


def extract_the_most_played_song_in_each_country():
    
    playlist_len = 50  # 例として各プレイリストの長さを50とします
    tracks_df = _get_playlist_tracks(weekly_top_playlist_ids_dict, playlist_len)
    tracks_with_features_df = _get_playlist_features(tracks_df, use_features_list)
    
    tracks_with_features_df.to_csv(FILENAME, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    
    extract_the_most_played_song_in_each_country()
    
    