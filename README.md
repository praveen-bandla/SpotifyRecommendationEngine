# Spotify Recommendation Engine

<div align="center">
    <img width="900" alt="" src="https://github.com/praveen-bandla/SpotifyRecommendationEngine/assets/114946455/16829fc3-bc48-4d93-9ee7-4ad05d054741">
</div>

# Table of Contents
1. [Introduction](#introduction)
2. [Overview of Project](#overview-of-project)
3. [Key Decisions](#key-decisions)
4. [Data](#data)
   - [Dataset](#dataset)
   - [Data Extraction](#data-extraction)
   - [Data Splits](#data-splits)
5. [Model](#model)
   - [Construction](#construction)
   - [Training](#training)
   - [Storage and Evaluation](#storage-and-evaluation)
6. [Recommendation](#recommendation)
7. [Conclusion](#conclusion)
   - [Discussion](#discussion)
   - [Weaknesses](#weaknesses)
8. [Future Works](#future-works)


## Introduction

Recommendation engines have become a crucial element in the digital landscape, revolutionizing user interactions with content across various domains. A prime example is the music streaming industry, where Spotify owes much of its success to its advanced recommendation engines. Central to this achievement are **Graph Neural Networks (GNNs)**, a rapidly advancing area within the machine learning space over the past 5-10 years. While many established algorithms rely on static graph data, a notably growing subfield within GNNs is the study of inductive learning methods, designed to handle dynamically changing graph data. These algorithms are incredibly relevant in their real-world applications and are set to continue developing over the years to come.
<br>
<br>
For my project, I aimed to develop a GNN-based song recommendation engine, centered around three core principles:  
- Implementing an inductive learning algorithm
- Ensuring efficient database management
- Yielding high-quality training results

<br>

## Overview of Project

The project's goal was to develop a recommendation engine that evaluates a playlist and its tracks to suggest similar songs.
<br>
<br>
I used the 2020 iteration of the `Spotify Million Playlist Dataset`. The data – comprising of playlists and their associated tracks - was then organized into a bipartite graph, where playlists and songs formed the two distinct sets of nodes. These nodes were interconnected by unweighted edges, linking each song to the playlists that feature it. To train the data, I implemented a GNN using the `GraphSAGE` learning algorithm, focusing on a link prediction task. In this setup, the positive edges were denoted by the actual connections between playlists and their tracks, while the negative edges for a playlist were randomly sampled from the remaining set of songs. To generate recommendations, I fed the model with a playlist and its tracks, calculated the probability of a link to every song outside this set, and then recommended the top `n` tracks based on these predictions.
<br>
<br>
The model was trained on a dataset comprising *150,000 playlists*, which included *850,000 unique songs* and a total of *20 million edges* (10 million positive and 10 million negative). Training was completed in *48 minutes using an A100 GPU*. The trained model achieved a ***92% test AUC and an 87% F1 score***. Here is a snapshot of the training progress. See the the linked `plotly` visual for the full interactive version.

<br>

<img width="530" alt="Snapshot of the Training Evaluation" src="https://github.com/praveen-bandla/SpotifyRecommendationEngine/assets/114946455/a7c2f430-f8c4-459b-82a2-d50cded16ca5">



## Key Decisions

Here are three key decisions I took during my project along with the underlying rationale:

1. **Bipartite Graph construction** <br> In many of the conventional graphs I came across in my research leading up to the project, songs were represented by nodes, with weighted edges denoting a relation derived from playlists – such as the number of co-occurrences of two songs across playlists. However, this proves inefficient with scaled data, due to a high edge count and a time complexity of $O(n^2)$ for playlist additions. Instead, I elected for a bipartite graph representing playlists and songs as my two sets of nodes, connected by unweighted edges mapping songs to the respective playlists that contain them. This reduced the edge count by ***~90% (estimated 90 million)***. This significantly increased training speed and enhanced storage efficiency. Moreover, my bipartite graph – unlike the conventional structure – did not require updating previous weights when adding new playlists, yielding a significantly improved time complexity of $O(n)$.

3. **GraphSAGE** <br> Many existing `GNN` learning methods – including `GCN` and `DeepWalk` – are transductive and rely on a static graph to train the data. Thus, any change to the database (adding new playlists for example) would necessitate re-training the entire model. In line with the guiding principles of my project, I wanted to construct a `GNN` that would facilitate a dynamically changing database. To this end, I specifically sought an inductive learning method that I could use to train on my bipartite graph structure. After research, I decided to use `GraphSAGE` with one layer of message passaging. I constructed a corresponding `GNN` with the number of song and playlist embeddings set at 64 (adjustable as a hyperparameter).


4. **Weighted Binary Cross Entropy Loss function** <br> A weighted binary cross entropy loss function was chosen, where the weights assigned for the positive and negative edges were 2.0 and 1.0 respectively. Given that the use case was to identify edges with the highest probability of a link existing, rendering many of the edges irrelevant, the weighted edges ensured that the positive learning was prioritized over the negative. Moreover, given that the negative edges were randomly sampled across all songs, they were deemed a less reliable source of information as compared to the positive edges.

<br>

## Data

### Dataset

The `Spotify Million Playlist Dataset` is a large-scale resource comprising one million user-generated playlists, released by Spotify for the RecSys Challenge 2018, and updated in 2020. It includes detailed information such as playlist titles, track lists, and associated metadata. For this project, I did not use any of the associated metadata.

### Data Extraction

The data from the JSON files was categorized using the `PlaylistSlice`, `Playlist`, `Song` classes (datamanager.py) and stored into csv files containing playlist information, song information, and edges. `info.json` was used to keep track of the aggregate information.

### Data Splits

This contained two steps – sampling negative edges and creating training splits. These tasks were completed simultaneously with the negative edges being stored in `edges.csv` with the label 0. The respective splits were stored in `test_edges.csv`, `train_edges.csv`, and `val_edges.csv`. The number of negative edges was equal to the number of positive ones in aggregation and within each split. See a more detailed description here 

<br>

## Model

### Construction

As referenced earlier, the model used was a `GraphSAGE-based GNN` for a link prediction task. In accordance with my bipartite graph, I used one layer of message passing with 64 playlist and song embedding dimensions (adjustable as a hyperparameter). The full model can be found in `model.py`.

### Training

To train, the `Adam` optimizer was used alongside a weighted `Binary Cross Entropy Loss` function (discussed earlier). At each epoch, the `train, test, val AUC scores` were recorded, along with the `train loss, test precision/recall/f1 scores`. These values were all returned along with the model in the `train_model()` function in `train.py`.


### Storage and Evaluation

At the end of training, the outputs were all stored as `training_data.csv`, while the model was stored as `model_weights.pth`. The AUC and loss values were plotted over epochs as a `plotly` visual saved as `training_visual.html`. The training procedure was conducted twice, on the sampled dataset represented in this repository along with the full version of the *150,000 playlists*. The model folder contains two distinct subfolders for each training, and the respective files can be found in both.

<br>

## Recommendation

For a given playlist, recommendations were generated by predicting the probability of links between every other song in the set and the playlist, reporting the highest. Here is specifically how it worked:

<br>

$$ 
\begin{align*}
&\text {Let }  p \text{ be a given playlist and } p_{s} := \{ \text {songs in }p\} \\
&\text {Let } modulo := \{\ S/\ p_{s}\} \text{ where } S \text{ is the set of all unique songs} \\
&\text {Calculate }\  GNN(s,\ p) \ \ \forall s \in modulo \ \   \text{where } GNN \text{ represents the trained model} \\
&\text{Set a threshold } t \\
&similar := \{\ s \in modulo : GNN(s, \ p) \geq t \ \} \\
&\text{recs} = \{ x_1, x_2, \ldots, x_n \mid x_i \in \text{similar}, \text{ for } i = 1, 2, \ldots, n \}
\end{align*}
$$

<br>

The rationale for doing so was to create a certain ‘random’ element to the predictions. Once a song is deemed ‘similar’, a probability of 0.95 is not very off 0.94. Thus, randomizing after the given threshold adds variety, as opposed to static predictions that would have been fixed for every given playlist.

<br>

## Conclusion

### Discussion

In this project, I set out to use a bipartite graph opposed to the conventional approaches, in the name of efficiency. I also set a condition of using an inductive learning algorithm to ensure effective compatibility with a dynamic database. Despite what could otherwise be considered constraints, the results – ***92% test AUC*** – are extremely strong. The usage of a bipartite graph had another unintended benefit – I did not have to actually store a graph. Instead, I stored my edges as rows in a `csv` file, and processed them as an edge list, avoiding the need for an otherwise cumbersome `networkX` graph, which would have slowed down training. This, combined with a significantly lowered edge count, meant that I was able to train on *150K playlists* and *20 million edges*, though the marginal improvements to accuracy after the first *75K playlists* were fairly negligible. That said, there are some areas of weakness in this project.


### Weaknesses 

1. **Negative edge sampling** <br> As mentioned, in this project, the negative edges were randomly sampled from the modulo set for each playlist with a **1:1 ratio** between the positive and negative edges. While this is a relatively common practice in `GNN` link prediction tasks, there are superior methods given this specific dataset. For example, a co-occurrence study would have yielded songs that are much more likely to not fall into playlists and more reliable negative edge. This could have improved `AUC` further along with the quality of predictions. In future works, this could be explored

2. **Justification of improved efficiency** <br> Comparing the bipartite and conventional graph structures, the reduced worse-case time-complexity and lowered edge count are well documented. However, a study of the real run-times for addition of playlists between the two structures could have given a more robust assessment of the time saved. 

<br>

## Future works

1. **Usage of song features** <br> This project did not take into account any features listed in the metadata, or use the song/artist/playlist name, or playlist followers. In future works, these could be incorporated for an in-depth learning algorithm.

2. **Interactive recommender** <br> using this model, a web interface could be developed to show the predictions in real time. This would give another insight into the model’s performance.

3. **Negative edge sampling methods** <br> As mentioned earlier, there could be alternative negative edge sampling methods explored with their implications on training and quality of recommendations studied as well.

4. **Dynamic music database** <br> While the inductive learning algorithm and bipartite graph ensure that playlists can be added efficiently and effectively without having to restart training, a continuation of the process could be implemented to extract new data, resample training splits, and resume training. 

