import numpy as np
import os
import pandas as pd
import librosa as lb
import librosa.feature.rhythm
from musicnn.tagger import top_tags
from musicnn.extractor import extractor
import warnings
warnings.filterwarnings("ignore")

model = 'MTT_musicnn' # others: 'MTT_vgg', 'MSD_musicnn', 'MSD_musicnn_big' or 'MSD_vgg'
tags = 'guitar, classical, slow, techno, strings, drums, electronic, rock, fast, piano, ambient, beat, violin, vocal, synth, female, indian, opera, male, singing, vocals, no vocals, harpsichord, loud, quiet, flute, woman, male vocal, no vocal, pop, soft, sitar, solo, man, classic, choir, voice, new age, dance, male voice, female vocal, beats, harp, cello, no voice, weird, country, metal, female voice, choral'
tags = tags.split(", ")

# create list with given tags
values = [['Title', 'BPM']]
for i in range(0,50):
  values[0].append(tags[i])

# read files in dir 'songs' and append the [string] result from func extractor to values
songs = os.listdir('songs')
for song in songs:  
 # list.append([song] + top_tags('songs'+'\\'+song, model=model, topN=n))
 taggram, label = extractor('songs'+'\\'+song, model=model, extract_features=False)
 tags_mean = np.mean(taggram, axis=0)
 
 x, sr = lb.load('songs'+'\\'+song, duration=60)
 onset_env = lb.onset.onset_strength(y=x, sr=sr)
 tempo = int((lb.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr))[0])
 
 values.append([song] + [tempo] + list(tags_mean))
 
 
 
 
# create .csv file  
pd.DataFrame(values).to_csv('tags.csv', index=False, header=False)




"""
import requests
import pandas as pd
import json

url = "https://sonoteller-ai1.p.rapidapi.com/music"

payload = { "file": "https://storage.googleapis.com/musikame-files/thefatrat-mayday-feat-laura-brehm-lyriclyrics-videocopyright-free-music.mp3" }
headers = {
	"content-type": "application/json",
	"X-RapidAPI-Key": "KEY",
	"X-RapidAPI-Host": "sonoteller-ai1.p.rapidapi.com"
}

response = requests.post(url, json=payload, headers=headers)

response = {
{
  'summary': 'The lyrics describe a person lost in outer space, ...',
  'keywords': [
    'space',
    'lost',
    'darkness',
    'hope',
    'help'
  ],
  'moods': [
    {
      'despair': 100
    },
    {
      'desperation': 100
    },
    {
      'loneliness': 80
    },
    {
      'hopefulness': 60
    },
    {
      'fear': 50
    }
  ],
  'themes': [
    {
      'isolation': 100
    },
    {
      'darkness': 80
    },
    {
      'hope': 60
    },
    {
      'desperation': 50
    },
    {
      'crying out for help': 40
    }
  ],
  'language': 'English',
  'explicit': 'No'
}
}

jstring = json.dumps(response)
print (jstring)

print (response)
df = pd.DataFrame(response)
df.to_csv('response.csv')
print (df)
 """