import numpy as np
import runpy as rp
import os
import pandas as pd
import librosa as lb
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from musicnn.tagger import top_tags
from musicnn.extractor import extractor
from sklearn.metrics.pairwise import cosine_similarity

def analyzeSong(song, model, window_duration, tags):
    """
    Analyzes a song every (*window_duration*) second(s) using musicnn
    

    Returns a list of song data in format: Title, t in s, BPM, 50 tags
    """
    song_duration = lb.get_duration(filename='songs'+'\\'+song)
    window_start = 0
    song_data = [['Title', 't in s', 'BPM'] + tags]
    
    while window_start + window_duration <= song_duration:
        taggram, label = extractor('songs'+'\\'+song, model=model, input_length=window_duration, input_overlap=window_start, extract_features=False)
        tags_mean = np.mean(taggram, axis=0)
        x, sr = lb.load('songs'+'\\'+song, offset=window_start, duration=window_duration)
        onset_env = lb.onset.onset_strength(y=x, sr=sr)
        tempo = int((lb.feature.tempo(onset_envelope=onset_env, sr=sr))[0])
        song_data.append([song, window_start, tempo] + list(tags_mean))
        window_start += window_duration
        
    return song_data

def calculateStdDeviation(song_data):
    """
    Appends to a given list of data the standard deviation
    """
    song_data.append(["Standard deviation", None, None])
    for j in range(3, len(song_data[0])):
        std_deviation = np.std([float(song_data[i][j]) for i in range(1, (len(song_data))-1)])
        song_data[len(song_data)-1].append(std_deviation)
    return song_data

def plotTags(song, values):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    for j in range(3, len(values[song][0])):
        tag_values = [float(values[song][i][j]) for i in range(1, (len(values[song]))-1)]
        ax.plot(tag_values, label=values[song][0][j])
    ax.set_xlabel('Time (in 3 second intervals)')
    ax.set_ylabel('Tag Probability')
    ax.set_title(song)
    ax.legend(loc='upper right')
    return fig

def plotTempo(tempo_data):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    for song, tempos in tempo_data.items():
        ax.plot(tempos, label=song)    
    ax.set_xlabel('Time (in 3 second intervals)')
    ax.set_ylabel('Tempo (BPM)')
    ax.legend(loc='upper right')
    return fig
    
def calculateSimilarity(song_data1, song_data2):
    """
    Calculates the similarity between two songs

    Returns:
    The cosine similarity between the two songs 
    """
    # Get the tag vectors of the two songs
    tag_vector1 = [row[2:] for row in song_data1[1:-1]]
    tag_vector2 = [row[2:] for row in song_data2[1:-1]]

    # Calculate the mean of the tag vectors
    mean_vector1 = np.mean(tag_vector1, axis=0)
    mean_vector2 = np.mean(tag_vector2, axis=0)

    # Calculate the cosine similarity between the mean vectors
    similarity = cosine_similarity([mean_vector1], [mean_vector2])[0][0]

    return similarity

def main():
    
     # Create a window
    window = tk.Tk()
    window.geometry("600x600")
    
    similarity_label = tk.Label(window)
    similarity_label.pack(side=tk.TOP)
    
    # Create a button to execute the script
    def script():
        model = 'MTT_vgg' # all: 'MTT_vgg', 'MSD_musicnn', 'MSD_musicnn_big' or 'MSD_vgg'
        tags = 'guitar, classical, slow, techno, strings, drums, electronic, rock, fast, piano, ambient, beat, violin, vocal, synth, female, indian, opera, male, singing, vocals, no vocals, harpsichord, loud, quiet, flute, woman, male vocal, no vocal, pop, soft, sitar, solo, man, classic, choir, voice, new age, dance, male voice, female vocal, beats, harp, cello, no voice, weird, country, metal, female voice, choral'
        tags = tags.split(", ")
        songs = os.listdir('songs')

        values = {}
        for song in songs:
            song_data = analyzeSong(song, model, 3, tags)
            song_data = calculateStdDeviation(song_data)
            values[song] = song_data

        tag_fig = []
        tempo_data = {}
        for song in values: 
            pd.DataFrame(values[song]).to_csv(song[:-4] + '.csv', index=False, header=False)
            tag_fig.append(plotTags(song, values))
            tempos = [row[2] for row in values[song][1:]]
            tempo_data[song] = tempos
        

        tempo_fig = plotTempo(tempo_data)

        for fig in tag_fig:
             canvas = FigureCanvasTkAgg(fig, master=window)
             canvas.draw()
             canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        tempo_canvas = FigureCanvasTkAgg(tempo_fig, master=window)
        tempo_canvas.draw()
        tempo_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # calculate similarity between two datasets, todo: should do it automatically if 2 or more songs are availible
        # maybe useful for the designers, they could skip the simulation of very similar songs?
        if len(values) == 2:
            similarity = calculateSimilarity(values[list(values.keys())[0]], values[list(values.keys())[1]])
            print(similarity)
            similarity_label.config(text=f"Similarity: {similarity}")
            tk.Label(window, text=f"Similarity: {similarity}").pack(side=tk.BOTTOM)   
    
        plot_height = 400
        total_height = len(tag_fig) * plot_height + plot_height

        # fit window size for plots
        window.geometry(f"600x{total_height}")

    tk.Button(window, text="Execute script", command=script).pack()

    window.attributes('-topmost',1)
    window.mainloop()
    
main()
