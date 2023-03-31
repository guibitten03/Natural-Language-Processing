import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd


class DataPreProcessing():
    def __init__(self, data:pd.DataFrame, steps:dict={}, language:str='english'):
        '''
            This is a class to preprocess text from dataframe.
            
            data: Dataframe with a columns ['text']
            list: List of process you want to do in text:
                {
                    'special_char': True,
                    'stop_words': True,
                    'stemmer': True,
                    'lemmatizer: True,
                    'norm': True
                }
                - By default, all the process are true. If you want take off any process, pass
                dict by parameter
            
        '''
        self.data = data
        self.language = language
        self.process_seted = {
                    'special_char': True,
                    'stop_words': True,
                    'stemmer': True,
                    'lemmatizer': True,
                    'norm': True
                }
        
        self.preprocess()
        
            
    def preprocess(self):
        # Tirar caracteres especiais do tipo (!,?)
        
        text_col = self.data['text']
        
        text_col = [nltk.word_tokenize(text) for text in text_col]
        
        if self.process_seted['special_char']:
            for text in text_col:
                text = [word for word in text if word.isalpha()]
                
        if self.process_seted['stop_words']:
            stop_words = set(stopwords.words(self.language))
            for text in text_col:
                text = [word for word in text if not word in stop_words]
                
        if self.process_seted['stemmer']:
            stemmer = SnowballStemmer(self.language)
            for text in text_col:
                text = [stemmer.stem(word) for word in text]
                
        if self.process_seted['lemmatizer']:
            lemma = WordNetLemmatizer()
            for text in text_col:
                text = [lemma.lemmatize(word) for word in text]
                
        if self.process_seted['norm']:
            for text in text_col:
                text = [word.replace('etc.', 'et cetera') for word in text]
                text = [word.replace('mr.', 'mister') for word in text]
                text = [word.replace('$', 'dollar') for word in text]
                
        self.data['text'] = text_col
        
        
    def get_preprocessed_data(self):
        return self.data
        
        
    