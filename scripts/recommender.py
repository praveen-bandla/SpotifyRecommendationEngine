import torch
from data_manager import Data

def recommend_tracks(model, df_songs, all_songs, songs_in_playlist, \
                     pid, num_recs, threshold):

  modulo = torch.LongTensor([x for x in all_songs if x not in songs_in_playlist])
  pid_tensor = pid * torch.ones_like(modulo)
  model.eval()
  probs_raw = model(pid_tensor, modulo)
  probs = probs_raw.view(-1)
  mask = probs>=threshold
  top_indices = torch.nonzero(mask, as_tuple = True)
  top_probs, top_values = probs[top_indices], modulo[top_indices]
  rec_values = top_values[torch.randperm(top_values.size(0))[:num_recs]]
  song_info, rec_info = [], []

  for song_id in songs_in_playlist:
        song_row = df_songs[df_songs['song_id'] == song_id]
        name, artist = song_row['name'].values[0], song_row['artist'].values[0]
        song_info.append(f"{name} by {artist}")

  rec_values_np = rec_values.cpu().numpy()
  for value in rec_values_np:
        song_row = df_songs[df_songs['song_id'] == value]
        name, artist = song_row['name'].values[0], song_row['artist'].values[0]
        rec_info.append(f"{name} by {artist}")

  return song_info, rec_info


def recommender(model, cur_dir, pid, num_recs, threshold):
     data = Data(cur_dir)
     df_songs = data.get.csv('songs')
     all_songs = list(data.get.unique_song_ids())
     songs_in_playlist = data.get.songs_in_playlist(pid)

     song_info, rec_info = recommend_tracks(model, df_songs, \
                                            all_songs, songs_in_playlist,\
                                                pid, num_recs, threshold)
     
     return song_info, rec_info