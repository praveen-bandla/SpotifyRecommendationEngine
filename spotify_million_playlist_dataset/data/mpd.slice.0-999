import os

class DataDefinitions:
    
    @staticmethod
    def get_csv_filenames():
        return ['songs.csv', 'playlists.csv', 'edges.csv', 'train_edges.csv',
             'test_edges.csv', 'val_edges.csv']

    @staticmethod
    def get_info_keys():
        keys = ['files_parsed', 'playlists', 'songs','train_edges',
                'val_edges', 'test_edges','pos_edges', 'neg_edges']
        return keys
    
    @staticmethod
    def get_song_cols():
        return ['song_id', 'native_song_id', 'name', 'artist']

    @staticmethod
    def get_playlist_cols():
        return ['playlist_id', 'playlist_name', 'num_tracks', 'num_followers']

    @staticmethod
    def get_edges_cols():
        return ['playlist_id', 'song_id', 'label']


class UtilFuncs:

    @staticmethod
    def is_directory_empty(cur_dir):
        return not bool(os.listdir(cur_dir))

    @staticmethod
    def is_csv_empty(df_to_check):
        return len(df_to_check) == 0
