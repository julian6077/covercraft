import numpy as np
import os
import pandas as pd
import librosa as lb
import librosa.feature.rhythm
import matplotlib.pyplot as plt
from musicnn.tagger import top_tags
from musicnn.extractor import extractor
import warnings
warnings.filterwarnings("ignore")

def load_songs(directory):
    return os.listdir(directory)

def analyzeSong(song, model, window_duration, tags):
    """
    Analyzes a song every (*window_duration*) second(s) using musicnn.
    (to work properly you should use 3 seconds)
    Also creates the outline of the data
    

    Returns a list of song data.
    """
    song_duration = lb.get_duration(filename='songs'+'\\'+song)
    window_start = 0
    song_data = [['Title', 't in s', 'BPM'] + tags]
    
    while window_start + window_duration <= song_duration:
        taggram, label = extractor('songs'+'\\'+song, model=model, input_length=window_duration, input_overlap=window_start, extract_features=False)
        tags_mean = np.mean(taggram, axis=0)
        x, sr = lb.load('songs'+'\\'+song, offset=window_start, duration=window_duration)
        onset_env = lb.onset.onset_strength(y=x, sr=sr)
        tempo = int((lb.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr))[0])
        song_data.append([song, window_start, tempo] + list(tags_mean))
        window_start += window_duration
        
    return song_data

def calculateStdDeviation(song_data):
  
    song_data.append(["Standard deviation", None, None])
    for j in range(3, len(song_data[0])):
        std_deviation = np.std([float(song_data[i][j]) for i in range(1, (len(song_data))-1)])
        song_data[len(song_data)-1].append(std_deviation)
    return song_data

def plotTags(song, values):
  
    for j in range(3, len(values[song][0])):
        tag_values = [float(values[song][i][j]) for i in range(1, (len(values[song]))-1)]
        plt.plot(tag_values, label=values[song][0][j])
    plt.xlabel('Time (in 3 second intervals)')
    plt.ylabel('Tag Probability')
    plt.legend(loc='upper right')
    plt.show()

def plotTempo(tempo_data):
  
    for song, tempos in tempo_data.items():
        plt.plot(tempos, label=song)
    plt.xlabel('Time (in 3 second intervals)')
    plt.ylabel('Tempo (BPM)')
    plt.legend(loc='upper right')
    plt.show()

def main():
    model = 'MTT_vgg' # others: 'MTT_vgg', 'MSD_musicnn', 'MSD_musicnn_big' or 'MSD_vgg'
    tags = 'guitar, classical, slow, techno, strings, drums, electronic, rock, fast, piano, ambient, beat, violin, vocal, synth, female, indian, opera, male, singing, vocals, no vocals, harpsichord, loud, quiet, flute, woman, male vocal, no vocal, pop, soft, sitar, solo, man, classic, choir, voice, new age, dance, male voice, female vocal, beats, harp, cello, no voice, weird, country, metal, female voice, choral'
    tags = tags.split(", ")
    songs = load_songs('songs')
    
    # create a dictionary where key = song name, value = list of song data
    values = {}
    for song in songs:
        song_data = analyzeSong(song, model, 3, tags)
        song_data = calculateStdDeviation(song_data)
        values[song] = song_data
   
    # apply analytical functions to each list in the dictionary over time
    tempo_data = {}
    for song in values: 
        # create .csv file  
        pd.DataFrame(values[song]).to_csv(song[:-4] + '.csv', index=False, header=False)
        plotTags(song, values)
        tempos = [row[2] for row in values[song][1:]]
        tempo_data[song] = tempos
    # plot tempo for all songs
    plotTempo(tempo_data)

main()