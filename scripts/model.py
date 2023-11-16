import torch
import torch.nn as nn

class GraphSAGELinkPrediction(nn.Module):
    def __init__(self, num_playlists, num_songs, playlist_dim, song_dim, 
                 dropout_prob):
        super(GraphSAGELinkPrediction, self).__init__()
        self.playlist_embedding = nn.Embedding(num_playlists, playlist_dim)
        self.song_embedding = nn.Embedding(num_songs, song_dim)
        self.fc = nn.Linear(playlist_dim + song_dim, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, playlist_ids, song_ids):
        playlist_embedded = self.playlist_embedding(playlist_ids)
        song_embedded = self.song_embedding(song_ids)
        concatenated_embeddings = torch.cat((playlist_embedded, song_embedded), dim=1)
        concatenated_embeddings = self.dropout(concatenated_embeddings)  # Add dropout
        prediction = torch.sigmoid(self.fc(concatenated_embeddings))

        return prediction