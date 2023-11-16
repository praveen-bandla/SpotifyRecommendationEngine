import torch
import torch.nn as nn
from data_manager import Data
import plotly.graph_objects as go
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from model import *
from hyperparameters import *


def train_model(parsed_data, model, device):
  
  model = model.to(device)

  sample_weights = torch.where(parsed_data['train']['label_ids'].to(device) == 1, \
                               torch.tensor(pos_edge_weight), \
                                torch.tensor(neg_edge_weight))
  
  criterion = nn.BCEWithLogitsLoss(weight = sample_weights)

  if optimizer == 'adam':
      model_optimizer = optim.Adam(model.parameters(), lr = learning_rate)

  train_losses, train_auc_scores, val_auc_scores, test_auc_scores = [], [], [], []

  test_precision_scores, test_recall_scores, test_f1_scores = [], [], []

  for epoch in range(num_epochs):
      model.train()

      model_optimizer.zero_grad()

      outputs = model(parsed_data['train']['playlist_ids'].to(device), \
                      parsed_data['train']['song_ids'].to(device))

      loss = criterion(outputs.view(-1), parsed_data['train']['label_ids'].to(device, dtype=torch.float))
      train_losses.append(loss.item())

      loss.backward()
      model_optimizer.step()

      with torch.no_grad():
          train_probs = outputs.cpu().view(-1).numpy()
          train_auc = roc_auc_score(parsed_data['train']['label_ids'].numpy(), train_probs)
          train_auc_scores.append(train_auc)

      model.eval()
      val_outputs = model(parsed_data['val']['playlist_ids'].to(device), \
                           parsed_data['val']['song_ids'].to(device))

      with torch.no_grad():
          val_probs = val_outputs.cpu().view(-1).numpy()
          val_auc = roc_auc_score(parsed_data['val']['label_ids'].numpy(), val_probs)
          val_auc_scores.append(val_auc)


      test_outputs = model(parsed_data['test']['playlist_ids'].to(device), \
                           parsed_data['test']['song_ids'].to(device))

      with torch.no_grad():
          test_probs = test_outputs.cpu().view(-1).numpy()
          test_auc = roc_auc_score(parsed_data['test']['label_ids'].numpy(), test_probs)
          test_auc_scores.append(test_auc)

      with torch.no_grad():
          test_preds = (test_probs > 0.5).astype(int) 
          test_precision = precision_score(parsed_data['test']['label_ids'].numpy(), test_preds)
          test_recall = recall_score(parsed_data['test']['label_ids'].numpy(), test_preds)
          test_f1 = f1_score(parsed_data['test']['label_ids'].numpy(), test_preds)
          
          test_precision_scores.append(test_precision)
          test_recall_scores.append(test_recall)
          test_f1_scores.append(test_f1)

      if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train AUC: {train_auc:.4f}, Validation AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

  output_data = [train_losses, train_auc_scores, val_auc_scores, test_auc_scores, test_precision_scores, test_recall_scores, test_f1_scores]

  return model, output_data



def create_loss_auc_plot(train_losses, train_auc_scores, val_auc_scores, test_auc_scores, num_playlists):
    epochs = list(range(1, num_epochs + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode='lines', name='Train Loss', \
                             line=dict(dash='dot', color='red'), 
                             hovertemplate = '%{y:.3f}'))
    fig.add_trace(go.Scatter(x=epochs, y=train_auc_scores, mode='lines', name='Train AUC', \
                             line=dict(color='blue'), hovertemplate = '%{y:.3f}' ))
    fig.add_trace(go.Scatter(x=epochs, y=val_auc_scores, mode='lines', name='Validation AUC', \
                             line=dict(color='green'), hovertemplate = '%{y:.3f}'))
    fig.add_trace(go.Scatter(x=epochs, y=test_auc_scores, mode='lines', name='Test AUC', \
                             line=dict(color='orange'), hovertemplate = '%{y:.3f}'))

    def format_title(title, num_playlists):
      title = f'<b>{title}</b>'
      subtitle = f'<span style = "font-size: 15px;">Trained using {num_playlists} playlists'
      return f'{title}<br>{subtitle}'

    fig.update_layout(
        xaxis_title='Epoch',
        yaxis_title='Value',
        hovermode='x',
        width=900,
        height=700,
        margin=dict(l=50, r=20, t=100, b=50),
        legend=dict(x=0.05, y=0.9, font=dict(family='Arial', size=10)), 
        xaxis=dict(title=dict(text='Epoch', font=dict(family = 'Arial', size=16)), 
                   tickfont=dict(size=14)), 
        yaxis=dict(title=dict(text='AUC/Loss', font=dict(family = 'Arial', size=16)), 
                   tickfont=dict(size=14), range=[0, 1]),
        title = dict(text = format_title('Training and Evaluation Metrics over Epoch', 
                                         num_playlists),
                     font = dict(family = 'verdana', size = 30), x = 0.5, y = 0.97)
      )

    fig.update_traces(hoverinfo='text')
    config = dict({'scrollZoom': False, 'staticPlot': False})

    return fig, config


def train(cur_dir, model_arch, device, return_model = False):
   
   data = Data(cur_dir)
   num_playlists = data.get.num_playlists()
   num_songs = data.get.num_songs()
   parsed_data = data.get.train_model_data()

   if model_arch == 'GraphSAGE':
      model = GraphSAGELinkPrediction(num_playlists, num_songs, playlist_embedding_dim, 
                                      song_embedding_dim, dropout)

   new_model, output_data = train_model(parsed_data, model, device)
   
   fig, config = create_loss_auc_plot(output_data[0], output_data[1], output_data[2], 
                                      output_data[3], num_playlists)

   data.save.plot(fig, config)
   data.save.model_weights(new_model)
   data.save.train_data(output_data)   

   return new_model if return_model else None