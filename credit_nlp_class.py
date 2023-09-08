#ввод библиотек
import pandas as pd
import numpy as np
import torch
import regex
import re
from pymorphy2 import MorphAnalyzer

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

import nltk
from nltk.corpus import stopwords
import stop_words

from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider, Range1d
from bokeh.layouts import column
from bokeh.palettes import all_palettes

from wordcloud import WordCloud

import umap

#ввод данных
df = pd.read_excel('CRA_train_1200.xlsx')
df['pr_txt'] = df['pr_txt'].astype(str).str.zfill(6)

#перекодировка рейтинга
rating_map = {'C': 0,
                  'B-': 1,
                  'B': 2,
                  'B+': 3,
                  'BB-': 4,
                  'BB': 5,
                  'BB+': 6,
                  'BBB-':7,
                  'BBB':8,
                  'BBB+':9,
                  'A-': 10,
                  'A': 11,
                  'A+': 12,
                  'AA-': 13,
                  'AA': 14,
                  'AA+': 15,
                  'AAA': 16}

df['Уровень рейтинга'] = df['Уровень рейтинга'].map(rating_map)
df

regexp = RegexpTokenizer('\w+')

#токенизация
df['txt_token']=df['pr_txt'].apply(regexp.tokenize)

#удаление стоп-слов и лемматизация
stopwords_ru = stopwords.words('russian')
patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
morph = MorphAnalyzer()

def lemmatize(doc):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            
            tokens.append(token)
    if len(tokens) > 2:
        return tokens
    return None

df['txt_nostop'] = df['txt_token'].apply(lemmatize)
df.head(10)
