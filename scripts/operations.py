from utils import DataDefinitions, UtilFuncs
from data_manager import *


def add_songs(cur_dir, num_files_to_parse):

    data = Data(cur_dir)

    if UtilFuncs.is_directory_empty(cur_dir):
       data.manage.create_files()

    unique_songs = data.get.unique_songs()

    num_files_parsed, num_playlists, num_songs, num_train_edges, \
    num_val_edges, num_test_edges, num_pos_edges, num_neg_edges= data.get.info()

    pos_edges_to_add, songs_to_add, playlists_to_add = [], [], []

    for slice_idx in range(num_files_to_parse):
        slice = PlaylistSlice(data, num_files_parsed + slice_idx)
        

        for pid in range(1000):
            songs_in_playlist = set()
            playlist = Playlist(slice, pid)

            playlist_name, _, playlist_id, _, num_tracks, _, num_followers = \
              playlist.get_playlist_info()
 
            playlists_to_add.append((playlist_id, playlist_name, num_tracks, \
                                     num_followers))
            num_playlists +=1

            for song_idx in range(num_tracks):
                song = Song(playlist, song_idx)
                native_song_id = song.get_song_id()
                song_id = -1

                if not native_song_id in unique_songs.keys():
                    _, artist_name, _, _, track_name, _, _, _ = song.get_song_info()
                    new_song_id = num_songs
                    num_songs+=1
                    new_song_info = (new_song_id, native_song_id, track_name, \
                                     artist_name)

                    unique_songs[native_song_id] = new_song_id
                    songs_to_add.append(new_song_info)
                    song_id = new_song_id

                else:
                    song_id = unique_songs[native_song_id]

                songs_in_playlist.add(native_song_id)
                pos_edges_to_add.append((playlist_id, song_id, 1))
                num_pos_edges +=1
    
    num_files_parsed +=num_files_to_parse
    data.update.csv('songs', songs_to_add)
    data.update.csv('playlists', playlists_to_add)
    data.update.csv('edges', pos_edges_to_add)

    data.update.info(num_files_parsed, num_playlists, num_songs, num_train_edges, \
                     num_val_edges,num_test_edges, num_pos_edges, num_neg_edges)
    return


def create_splits(cur_dir, splits = [0.7, 0.15, 0.15], collect_neg_edges = True, ratio = 1):
    
    data = Data(cur_dir)

    num_files_parsed, num_playlists, num_songs, num_train_edges, \
    num_val_edges, num_test_edges, num_pos_edges, num_neg_edges= data.get.info()

    unique_song_ids = data.get.unique_song_ids()
    df_edges = data.get.csv('edges')

    train_edges_to_add, test_edges_to_add, val_edges_to_add, \
      neg_edges_to_add = [], [], [], []

    for pid in range(num_playlists):
      playlist_tracks = df_edges[df_edges['playlist_id'] == pid]['song_id'].values

      test_pos_edges, val_pos_edges, train_pos_edges = \
        data.get.edge_splits(playlist_tracks, splits)

      test_edges_to_add.extend([(pid,x, 1) for x in test_pos_edges])
      val_edges_to_add.extend([(pid,x, 1) for x in val_pos_edges])
      train_edges_to_add.extend([(pid,x, 1) for x in train_pos_edges])

      if collect_neg_edges:
        playlist_neg_edges= []
        modulo = list(unique_song_ids - set(playlist_tracks))
        sampled_modulo = random.sample(modulo, len(playlist_tracks)*ratio)
        for sample_track in sampled_modulo:
          playlist_neg_edges.append(sample_track)


        test_neg_edges, val_neg_edges, train_neg_edges = \
          data.get.edge_splits(playlist_neg_edges, splits, False)
        
        neg_edges_to_add.extend([(pid,x,0) for x in playlist_neg_edges])
        test_edges_to_add.extend([(pid,x, 0) for x in test_neg_edges])
        val_edges_to_add.extend([(pid,x, 0) for x in val_neg_edges])
        train_edges_to_add.extend([(pid,x, 0) for x in train_neg_edges])

    data.update.csv('edges', neg_edges_to_add)
    data.update.csv('train_edges', train_edges_to_add)
    data.update.csv('test_edges', test_edges_to_add)
    data.update.csv('val_edges', val_edges_to_add)

    data.update.info(num_files_parsed, num_playlists, num_songs, len(train_edges_to_add),\
                     len(val_edges_to_add), len(test_edges_to_add), num_pos_edges,\
                      len(neg_edges_to_add))

    return

def reset_data(cur_dir):
    data = Data(cur_dir)
    data.manage.remove_files()
    data.manage.create_files()
    return
