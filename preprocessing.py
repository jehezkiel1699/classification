import re
import time
import numpy as np
import pandas as pd 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42
#df = pd.read_csv("outputBerita1.csv", encoding= 'unicode_escape')
df = pd.read_csv("datasetRaw1.csv", encoding= 'unicode_escape')

#df = pd.concat([df1,df2])

def preprocessing_text(text):
    
    encoded_string = text.encode("ascii", "ignore") #remove asci
    text = encoded_string.decode() #remove asci
    
    text = re.sub(r'http\S+', '', text) #remove url
    
    text = text.lower() #lowercase
    
    text = ''.join([i for i in text if not i.isdigit()]) #remove number
    
    #text = ''.join([i for i in text if i not in text.punctuation])
    #text = re.sub(r'[/(){}\[\]\|@,;#_]', '', text) #remove punctuation
    
    
    
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*“”‘’_~+=|\t\n'''

    for char in text:
        if char in punctuations:
            text = text.replace(char, "")

    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    text = stopword.remove(text)
    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)
    
    return text


start = time.time()
df['preprocessing_text'] = df.isi.apply(preprocessing_text)
#df.to_csv('datasetRaw.csv', index=False, encoding='utf-8')
dummies = pd.get_dummies(df.kategori)
df = pd.concat([df, dummies], axis='columns')

#df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

end = time.time()
#df.to_csv('dataset1RawWithDummies.csv',  index=False, encoding='utf-8')
df.to_csv('datasetBaruDenganDummies.csv',  index=False, encoding='utf-8')
#df_train.to_csv('train_berita.csv', index=False, encoding='utf-8')
#df_test.to_csv('test_berita.csv', index=False, encoding='utf-8')
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))