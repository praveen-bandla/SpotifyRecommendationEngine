import json
import pandas as pd
import os
from utils import DataDefinitions, UtilFuncs
import random
import math
import torch

class Data: 
    def __init__ (self, cur_dir):
        self.cur_dir = cur_dir
        self.data_dir = f'{self.cur_dir}processed_data/'
        self.info_dir = f'{self.data_dir}info.json'
        self.model_dir = f'{self.cur_dir}model/'
        self.get = self.Get(self.cur_dir, self.data_dir, self.info_dir)
        self.update = self.Update(self.cur_dir, self.data_dir, self.info_dir)
        self.manage = self.Manage(self.cur_dir, self.data_dir, self.info_dir)
        self.save = self.Save(self.cur_dir, self.model_dir)

    class Get:
        def __init__(self, cur_dir, data_dir, info_dir):
            self.cur_dir = cur_dir
            self.data_dir = data_dir
            self.info_dir = info_dir

        def info(self):
            try:
                with open(self.info_dir, 'r') as json_file:
                    info_dict = json.load(json_file)
                return info_dict.values()
            except FileNotFoundError:
                print(f"File not found: {self.info_dir}")
                return None

        def csv(self, fname):
            file_path = f'{self.data_dir}{fname}.csv'
            try:
                df_imported = pd.read_csv(file_path)
                return df_imported
            except FileNotFoundError:
                print(f"{fname}.csv does not exist")
                return None
            
        def unique_songs(self):
            df_songs = self.csv('songs')
            if UtilFuncs.is_csv_empty(df_songs):
                return {}

            unique_songs_dict = dict(zip(df_songs['native_song_id'], df_songs['song_id']))
            return unique_songs_dict
        
        def unique_song_ids(self):
            df_songs = self.csv('songs')
            if UtilFuncs.is_csv_empty(df_songs):
                return {}
            else:
                return set(df_songs['song_id'])

        def numeric_song_id(self, df_songs, native_song_id):
            song_id = df_songs[df_songs['native_song_id'] == native_song_id]\
                ['song_id'].values[0]
            return song_id
        
        def edge_splits(self, edges, splits, shuffle = True):
            if shuffle:
                random.shuffle(edges)
            a = math.floor(len(edges)*splits[1])
            b = math.floor(len(edges)*splits[2])
            test_edges = edges[:a]
            val_edges = edges[a:a+b]
            train_edges = edges[a+b:] 
            return test_edges, val_edges, train_edges

        def edge_data_parsed(self, fname):
            df = self.csv(fname)
            playlist_ids = torch.LongTensor(df['playlist_id'].values)
            song_ids = torch.LongTensor(df['song_id'].values)
            label_ids = torch.LongTensor(df['label'].values)

            return playlist_ids, song_ids, label_ids
        
        def num_playlists(self):
            df = self.csv('playlists')
            all_playlist_ids = set(df['playlist_id'])
            return len(all_playlist_ids)
        
        def num_songs(self):
            df = self.csv('songs')
            all_songs_ids = set(df['song_id'])
            return len(all_songs_ids)
        
        def songs_in_playlist(self, pid):
            df = self.csv('edges')
            songs = set(df[df['playlist_id'] == pid]['song_id'])
            return songs
        
        def song_info(self, song_id):
            df = self.csv('songs')
            song_row = df[df['song_id'] == song_id]
            name = song_row['name'].values[0]
            artist = song_row['artist'].values[0]

            return name, artist
        
        def train_model_data(self):
            data = Data(self.cur_dir)
            train_playlist_ids, train_song_ids, train_label_ids = data.get.edge_data_parsed('train_edges')
            test_playlist_ids, test_song_ids, test_label_ids = data.get.edge_data_parsed('test_edges')
            val_playlist_ids, val_song_ids, val_label_ids = data.get.edge_data_parsed('val_edges')

            parsed_data = {
                'train': {
                    'playlist_ids': train_playlist_ids,
                    'song_ids': train_song_ids,
                    'label_ids': train_label_ids,
                },
                'test': {
                    'playlist_ids': test_playlist_ids,
                    'song_ids': test_song_ids,
                    'label_ids': test_label_ids,
                },
                'val': {
                    'playlist_ids': val_playlist_ids,
                    'song_ids': val_song_ids,
                    'label_ids': val_label_ids,
                }
            }

            return parsed_data



    class Update:
        def __init__(self, cur_dir, data_dir, info_dir):
            self.cur_dir = cur_dir
            self.data_dir = data_dir
            self.info_dir = info_dir

        def csv(self, fname, data):
            file_path = f'{self.data_dir}{fname}.csv'
            try:
                df_file = pd.read_csv(file_path)
            except FileNotFoundError:
                return 'File not found'

            cols = df_file.columns

            if df_file.empty:
                new_data_df_file = pd.DataFrame(data, columns=cols)
                df_file = new_data_df_file
            else:
                new_data_df_file = pd.DataFrame(data, columns=cols)
                df_file = df_file.append(new_data_df_file, ignore_index=True)

            df_file.to_csv(file_path, index=False)

            return

        def info(self, files_parsed, playlists, songs, train_edges, \
                 val_edges, test_edges, pos_edges, neg_edges):
            
            new_info = vars()
            del new_info['self']

            with open(self.info_dir, 'r') as f:
                info = json.load(f)

            with open(self.info_dir, 'w') as g:
                json.dump(new_info, g, indent=4)

            return
        

    class Manage:
        def __init__(self, cur_dir, data_dir, info_dir):
            self.cur_dir = cur_dir
            self.data_dir = data_dir
            self.info_dir = info_dir

        def create_files(self, create_info = True):
            
            blank_songs_df = pd.DataFrame(columns = DataDefinitions.get_song_cols())
            blank_playlist_df = pd.DataFrame(columns = DataDefinitions.get_playlist_cols())
            blank_edges_df = pd.DataFrame(columns = DataDefinitions.get_edges_cols())

            file_paths = [f'{self.data_dir}{fname}' for fname in \
                          DataDefinitions.get_csv_filenames()]
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    # File does not exist, so save the DataFrame
                    if file_path.endswith('songs.csv'):
                        blank_songs_df.to_csv(file_path, index=False)
                    elif file_path.endswith('playlists.csv'):
                        blank_playlist_df.to_csv(file_path, index=False)
                    else:
                        blank_edges_df.to_csv(file_path, index=False)
                else:
                    # File already exists
                    print(f'File already exists: {file_path}')

            if create_info:
                    if not os.path.exists(self.info_dir):
                        keys = DataDefinitions.get_info_keys()
                        info_dict = {key: 0 for key in keys}
                        with open(self.info_dir, 'w') as json_file:
                            json.dump(info_dict, json_file, indent = 4)
                    else:
                        print("'Info.json' already exists")

            return
        
        def remove_files(self):
            '''
            If files need to be deleted, remove
            '''
            try:
                if UtilFuncs.is_directory_empty(self.data_dir):
                    print ('Folder is already empty')
                    return
                for filename in os.listdir(self.data_dir):
                    file_path = os.path.join(self.data_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            except Exception as e:
                print(f"An error occurred while deleting files: {str(e)}")
            return

    class Save:
        def __init__(self, cur_dir, model_dir):
            self.cur_dir = cur_dir
            self.model_dir = model_dir

        def plot(self, fig, config):
            fig.write_html(file = f'{self.model_dir}training_visual.html', 
                           config = config)
            return
        
        def model_weights(self, model):
            torch.save(model.state_dict(), f'{self.model_dir}model_weights.pth')
            return
        
        def train_data(self, output_data):
            df = pd.DataFrame({'Epoch number': range(len(output_data[0])), 
                               'Loss': output_data[0],
                               'Train AUC': output_data[1],
                               'Val AUC': output_data[2],
                               'Test AUC': output_data[3],
                               'Precision': output_data[4],
                               'Recall': output_data[5],
                               'F1': output_data[6]})
            
            df_rounded = df.round(2)
            df_rounded.to_csv(f'{self.model_dir}training_data.csv', index = False)
            return
            



class PlaylistSlice:
    def __init__(self, data_loader, file_num):
        self.data_loader = data_loader
        start_playlist_id = (file_num*1000)
        filename = f'{self.data_loader.cur_dir}spotify_million_playlist_dataset/data/mpd.slice.{start_playlist_id}-{start_playlist_id+999}.json'
        with open(filename, 'r') as json_file:
            self.data = json.load(json_file)

    def get_data(self):
        return self.data

class Playlist:
    def __init__(self, playlist_slice, pid):
        self.data = playlist_slice.get_data()['playlists'][pid]

    def get_data(self):
        return self.data

    def get_playlist_info(self):
        return self.data['name'], self.data['collaborative'], self.data['pid'], \
        self.data['modified_at'], self.data['num_tracks'], self.data['num_albums'],\
        self.data['num_followers']

    def get_num_tracks(self):
        return self.data['num_tracks']
    

class Song:
    def __init__(self, playlist, song_pos):
        self.data = playlist.get_data()['tracks'][song_pos]

    def get_song_info(self):
        return self.data.values()
    
    def get_song_id(self):
        return self.data['track_uri'].split(':')[2]