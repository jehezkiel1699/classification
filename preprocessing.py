import re
import time
import numpy as np
import pandas as pd 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


df1 = pd.read_csv("beritaKompas.csv", encoding= 'unicode_escape')
df2 = pd.read_csv("beritaSindo.csv", encoding= 'unicode_escape')

df = pd.concat([df1,df2])

def preprocessing_text(text):
    
    
    encoded_string = text.encode("ascii", "ignore") #remove asci
    text = encoded_string.decode() #remove asci
    
    text = text.lower() #lowercase
    
    text = ''.join([i for i in text if not i.isdigit()]) #remove number
    
    #text = ''.join([i for i in text if i not in text.punctuation])
    text = re.sub(r'[/(){}\[\]\|@,;#_]', '', text) #remove punctuation
    text = re.sub(r'http\S+', '', text) #remove url
    
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    text = stopword.remove(text)
    
    #factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)
    
    return text


start = time.time()
df['preprocessing_text'] = df.isi.apply(preprocessing_text)
end = time.time()
df.to_csv('datasetBerita.csv', index=False, encoding='utf-8')
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
