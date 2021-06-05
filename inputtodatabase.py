from datetime import date
import requests
import time
from bs4 import BeautifulSoup
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import mysql.connector
# tesUrl = []
today = date.today()
start = time.time()
def scrapingKompas(kategori):
    txtJudul = []
    txtKategori = []
    txtIsi = []
    #txtUrl = []
    txtGbr = []
    txtTgl = []
    txtSumber = []
    url = 'https://indeks.kompas.com/?site={}&page='.format(kategori)
    for page in range(1,2):
        req = requests.get(url+"{}".format(page))
        soup = BeautifulSoup(req.text, 'html.parser')
        items = soup.findAll('div', 'article__list__title')
        images = soup.findAll('div', 'article__asset')
        tanggal = soup.findAll('div', 'article__date')
        count = 0
        for tgl in tanggal:
            count += 1
            if count > 10:
                break
            txtTgl.append(tgl.text[:10])
        count = 0
        for image in images:
            count += 1
            if count > 10:
                break
            txtGbr.append(image.find('img').get('src'))
        count = 0
        for item in items:
            count += 1
            if count > 10:
                break
            
            judul = item.find('a', 'article__link').text
            judul = judul.replace('"', "")
            judul = judul.replace("'", "")
            
            txtJudul.append(judul)
            txtKategori.append(kategori)
            
            
            txtSumber.append("Kompas")
            
            tUrl = "{}?page=all".format(item.find('a').get('href'))
            reqUrl = requests.get(tUrl)
            soupUrl = BeautifulSoup(reqUrl.text, 'html.parser')
            
            contents = soupUrl.findAll('div', 'read__content')
            
            for content in contents:
                textHapus = []
                
                fullText = content.text

                listText=content.find('p').find('strong')
                
                """if(listText.text):
                    if(content.find('strong').text[-1] == "m"):
                        textHapus.append(content.find('strong').text + " -")
                    else:
                        textHapus.append(content.find('strong').text)
                else:
                    if(content.findAll('strong')[1].text[-1] == "m"):
                        textHapus.append(content.findAll('strong')[1].text + " -")
                    else:
                        textHapus.append(content.findAll('strong')[1].text)"""

                fullText = re.sub('\([Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa]([\s\S]*?)\)','',fullText)
                fullText = re.sub('[Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa].*','',fullText)
                
                fullText = re.sub('[Kk][Oo][Mm][Pp][Aa][Ss].[Cc][Oo][Mm] ?\- ?', '', fullText)

                fullText = re.sub("\n", "<br>", fullText)          
                fullText = re.sub("(<br>)+", "<br>", fullText)

                for i in textHapus:
                    fullText = fullText.replace(re.sub('<[^<]+?>', '', str(i)),'')
                fullText = fullText.replace("'", "") #karena db kalau ada tanda '(single quote) error
                
                #fullText = fullText.replace("/$/mg", "<br />")
                fullText = fullText.encode("ascii", "ignore").decode()
                
                #print(txtGbr)
                #print("\n")
                
                txtIsi.append(fullText)
    return txtGbr, txtJudul, txtTgl, txtIsi, txtKategori, txtSumber

def scrapingSindo(kategori, tanggal):
    txtJudul = []
    txtKategori = []
    txtIsi = []
    #txtUrl = []
    txtGbr = []
    txtTgl = []
    txtSumber = []
    
    if kategori == "edukasi":
        tmp = 'https://index.sindonews.com/index/144'
    elif kategori == "tekno":
        tmp = 'https://index.sindonews.com/index/612'
    elif kategori == "sports":
        tmp = 'https://index.sindonews.com/index/10'
    elif kategori == "lifestyle":
        tmp = 'https://index.sindonews.com/index/154'
    
    page = 0
    
    for i in range(1,2):
        url1 = tmp + "/{}/?t={}".format(page,tanggal)

        
        req1 = requests.get(url1, timeout=None)
        soup1 = BeautifulSoup(req1.text, 'html.parser')
        items = soup1.findAll('div', 'indeks-rows')
        count = 0
        for item in items:
            count += 1
            if count > 10:
                break
            judul = item.find('div', 'indeks-title').text
            judul = judul.replace('"', "")
            judul = judul.replace("'", "")
            txtJudul.append(judul)
            txtGbr.append(item.find('div', 'indeks-pict').find('img').get('data-src'))
            
            strTanggal = tanggal.strftime("%d/%m/%Y")
            txtTgl.append(strTanggal)
            
            txtSumber.append("Sindonews")
            txtKategori.append(kategori)
            
            tUrl = "{}?showpage=all".format(item.find('div', 'indeks-title').find("a").get('href'))
            reqUrl = requests.get(tUrl, timeout=None)
            soupUrl = BeautifulSoup(reqUrl.text, 'html.parser')
            
            if kategori == 'edukasi':
                contents = soupUrl.findAll('div', 'article')
            elif kategori == 'tekno':
                contents = soupUrl.findAll('div', 'desc-artikel-detail')
            else:
                contents = soupUrl.findAll('div', {"id": "content"})
            
            for content in contents:
                for br in content.findAll('br'):
                        br.replace_with("\n")
                fullText = ""
                textHapus = []
                
                if(kategori == 'edukasi'):
                    fullText = content.text
                    listText = content.findAll('strong')
                    fullText = re.sub('\([Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa]([\s\S]*?)\)','',fullText)
                    fullText = re.sub('[Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa].*','',fullText)
                    
                    fullText = re.sub('\([Ll][Ii][Hh][Aa][Tt] [Jj][Uu][Gg][Aa]([\s\S]*?)\)','',fullText)
                    fullText = re.sub('[Ll][Ii][Hh][Aa][Tt] [Jj][Uu][Gg][Aa].*','',fullText)
                    i = 0
                    
                    if(content.find('div', 'baca-inline')):
                        textHapus.append(content.find('div', 'baca-inline').text)
                    if(content.find('div', 'reporter')):
                        textHapus.append(content.find('div', 'reporter').text)
                    for a in content.findAll('div', 'social-embed'):
                        textHapus.append(a)
                    for a in listText:
                        i+=1
                        if i==1:
                            textHapus.append(a.text + ' - ')
                        else:
                            #print(a.text[1:10].lower(), a.text[-1])
                            if a.text[1:10].lower()=='baca juga':
                                #print(a.findNext('strong').text, tesIndeks[-1])
                                textHapus.append(a.text)
                                textHapus.append(a.text + a.findNext("a").text)
                elif(kategori == 'sports'):
                    fullText = content.text
                    listText = content.findAll('strong')
                    fullText = re.sub('\([Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa]([\s\S]*?)\)','',fullText)
                    fullText = re.sub('[Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa].*','',fullText)
                    fullText = re.sub('\([Ii][Kk][Uu][Tt][Ii] [Ss][Uu][Rr][Vv][Ee][Ii]([\s\S]*?)\)','',fullText)
                    fullText = re.sub('[Ii][Kk][Uu][Tt][Ii] [Ss][Uu][Rr][Vv][Ee][Ii].*','',fullText)
                    fullText = re.sub('\([Ll][Ii][Hh][Aa][Tt] [Gg][Rr][Aa][Ff][Ii][Ss]([\s\S]*?)\)','',fullText)
                    
                    fullText = re.sub('\([Ll][Ii][Hh][Aa][Tt] [Jj][Uu][Gg][Aa]([\s\S]*?)\)','',fullText)
                    fullText = re.sub('[Ll][Ii][Hh][Aa][Tt] [Jj][Uu][Gg][Aa].*','',fullText)
                    i = 0
                    
                    if(content.find('div', 'baca-inline')):
                        textHapus.append(content.find('div', 'baca-inline').text)
                    if(content.find('div', 'editor')):
                        textHapus.append(content.find('div', 'editor').text)
                    #listSurvey = content.findAll('br')
                    #textHapus+=content.findAll('div', 'box-outlink')
                    if(content.find('div', 'box-outlink')):
                        textHapus.append(content.find('div','box-outlink').text)
                    for a in content.findAll('div', 'social-embed'):
                        textHapus.append(a)
                    for a in listText:
                        i+=1
                        if i==1:
                            textHapus.append(a.text + ' - ')
                        else:
                            if a.text[1:11].lower()=='(baca juga':
                                textHapus.append(a.text)
                                #print(a.text, tesIndeks[-1])
                                #print(a.text, tesIndeks[-1])
                            elif a.text[1:10].lower()=='baca juga':
                                textHapus.append(a.text)
                                #print(a.text, tesIndeks[-1])
                                #print(a.text, tesIndeks[-1])
                            elif a.text[0:9].lower()=='baca juga':
                                textHapus.append(a.text)
                                #print(a.text, tesIndeks[-1])
                            elif a.text[0:14].lower()=='(lihat grafis:':
                                textHapus.append(a.text)
                elif(kategori == 'tekno'):

                    fullText = content.text
                    fullText = re.sub('\([Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa]([\s\S]*?)\)','',fullText)
                    fullText = re.sub('[Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa].*','',fullText)
                    
                    fullText = re.sub('\([Ll][Ii][Hh][Aa][Tt] [Jj][Uu][Gg][Aa]([\s\S]*?)\)','',fullText)
                    fullText = re.sub('[Ll][Ii][Hh][Aa][Tt] [Jj][Uu][Gg][Aa].*','',fullText)
                    
                    listText = content.findAll('strong')
                    if(content.find('div', 'warp-baca-juga')):
                        textHapus.append(content.find('div', 'warp-baca-juga').text)
                        
                    if(content.find('div', 'editor')):
                        textHapus.append(content.find('div', 'editor').text)
                        
                    #if(content.find('div', 'box-outlink')):
                    #    textHapus.append(content.find('div','box-outlink').text)
                    for a in content.findAll('div', 'social-embed'):
                        textHapus.append(a)
                    i = 0
                    for a in listText:
                        i+=1
                        if i==1:
                            textHapus.append(a.text + ' - ')
                
                elif(kategori == 'lifestyle'):
                    fullText = content.text
                    #fullText = re.sub('\([Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa][^)]*\)','',fullText)
                    #fullText = re.sub('[Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa]([\s\S]*?)(<br>)','',fullText)
                    
                    
                    fullText = re.sub('\([Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa]([\s\S]*?)\)','',fullText)
                    fullText = re.sub('[Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa].*','',fullText)
                    
                    fullText = re.sub('\([Ll][Ii][Hh][Aa][Tt] [Jj][Uu][Gg][Aa]([\s\S]*?)\)','',fullText)
                    fullText = re.sub('[Ll][Ii][Hh][Aa][Tt] [Jj][Uu][Gg][Aa].*','',fullText)
                    #print(fullText)
                    if(content.find('div', 'baca-inline')):
                        textHapus.append(content.find('div', 'baca-inline').text)
                    if(content.find('div', 'editor')):
                        textHapus.append(content.find('div', 'editor').text)
                    if(content.find('div', 'box-outlink')):
                        textHapus.append(content.find('div','box-outlink').text)
                    listText = content.findAll('strong')
                    for a in content.findAll('div', 'social-embed'):
                        textHapus.append(a)
                    i = 0
                    for a in listText:
                        i+=1
                        if i==1:
                            textHapus.append(a.text + ' - ')
                            
                    listText = content.findAll('strong')
                    listText += content.findAll('em')
                    for a in listText:
                        i+=1
                        if str(a.text[1:10].lower())=='baca juga' or str(a.text[0:9].lower())=='baca juga':
                            textHapus.append(a)
                        elif i == 1:
                            textHapus.append(a.text+' - ')
                for i in textHapus:
                    fullText = fullText.replace(re.sub('<[^<]+?>', '', str(i)),'')
                
                fullText = fullText.replace("“", "")
                fullText = fullText.replace("'", "") #karena db kalau ada tanda '(single quote) error
                fullText = re.sub("\n", "<br>", fullText)          
                fullText = re.sub("(<br>)+", "<br>", fullText)
                fullText = fullText.encode("ascii", "ignore").decode()
                txtIsi.append(fullText)
        page+=15
    return txtGbr, txtJudul, txtTgl, txtIsi, txtKategori, txtSumber

def preprocessing_text(text):

    encoded_string = text.encode("ascii", "ignore") #remove asci
    text = encoded_string.decode() #remove asci
    
    text = text.lower() #lowercase
    
    text = ''.join([i for i in text if not i.isdigit()]) #remove number
    
    #text = ''.join([i for i in text if i not in text.punctuation])
    
    text = re.sub(r'http\S+', '', text) #remove url
    
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    text = stopword.remove(text)
    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)
    
    text = re.sub("[^\w\s]",'',text) #remove punctuation
    text = re.sub(r'[/(){}\[\]\|@,;#_]', '', text) #remove punctuation
    
    return text
if torch.cuda.is_available():
  device = torch.device('cuda')

  print('there are %d GPU(s) available.' % torch.cuda.device_count())

  print('we will use the GPU: ', torch.cuda.get_device_name(0))

else:
  print("No GPU available, using the CPU instead")
  device = torch.device("cpu")
  print(device)


class_names = ['edukasi', 'tekno', 'sports', 'health', 'lifestyle']
MAX_LEN = 320
PRE_TRAINED_MODEL_BAHASA =  'indobenchmark/indobert-base-p1'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_BAHASA)

checkpoint = torch.load("D:/Petra/Informatika/Skripsi/Skripsi/code/KODINGAN_FIX/indobert-base-p1_1e-05_0.2_16_2_1.bin", map_location=torch.device('cpu'))
EPOCHS = checkpoint['epoch']
DROPOUT = checkpoint['dropout']
BATCH_SIZE = checkpoint['batch_size']
MAX_LEN = checkpoint['max_len']
LEARNING_RATE = checkpoint['learning_rate']
class Klasifikasi(nn.Module):
  def __init__(self, n_classes, dropout_value=0.2):
    super(Klasifikasi, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_BAHASA)
    self.drop = nn.Dropout(p=dropout_value)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False
    )
    output = self.drop(pooled_output)
    return self.out(pooled_output)



    
def PredictText(model, text):
    model = model.eval()
    temp_labels = []
    encoded_review = tokenizer.encode_plus(
      text,
      max_length=MAX_LEN,
      add_special_tokens=True,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      truncation=True,
      return_tensors='pt',
    )
    
    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    
    output = model(input_ids, attention_mask)
    prob = torch.softmax(output, dim=1)
    y_pred_prob = prob.detach().cpu().numpy()
    text = []
    for a in range(len(y_pred_prob)):
        indeks = 0
        for b in range(len(y_pred_prob[a])):
          if(y_pred_prob[a][b]>0.5):
            text.append(class_names[b])
    if not text:
        temp = torch.softmax(output, dim=1)
        temp_prob = bubbleSort(temp[0].detach().cpu().numpy())      
        hasil = temp_prob[0]
        
        for a in range(len(y_pred_prob)):
            for b in range(len(y_pred_prob[a])):
                if(y_pred_prob[a][b]==hasil):
                    text.append(class_names[b])
    return text
    """#temp = torch.softmax(output, dim=1)
    #top2 = []

    #y_pred_prob = prob[0].detach().cpu().numpy()
    temp = bubbleSort(temp[0].detach().cpu().numpy())

    
    for a in range(len(temp)):
      if(a==2):
        break
      top2.append(temp[a])
    for a in range(len(y_pred_prob)):
      for b in range(len(top2)):
        if(y_pred_prob[a] == top2[b]):
          text.append(class_names[a])"""
 
    #return temp_labels
def toString(s):
    str1 = "<br>"
    return (str1.join(s))
	
def bubbleSort(arr):
    n = len(arr)
    # Traverse through all array elements
    for i in range(n-1):
    # range(n) also work but outer loop will repeat one time more than needed.
  
        # Last i elements are already in place
        for j in range(0, n-i-1):
  
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] < arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
#INISIALISASI SEMUA DATA SETELAH DI SCRAPE - PREPROCESSING TEXT
#YYYY-MM-DD	
#class_names = ['edukasi', 'tekno', 'sports', 'health', 'lifestyle']
gambar = []
judul = []
tanggal = []
isiberita = []
kategori = []
sumber = []
preprocessing_isi = []

gbr, jdl, tgl, isi, ktgri, sbr = scrapingKompas("edukasi")

gambar += gbr
judul += jdl
tanggal += tgl
isiberita += isi
kategori += ktgri
sumber += sbr

gbr, jdl, tgl, isi, ktgri, sbr = scrapingKompas("tekno")

gambar += gbr
judul += jdl
tanggal += tgl
isiberita += isi
kategori += ktgri
sumber += sbr

gbr, jdl, tgl, isi, ktgri, sbr = scrapingKompas("sports")

gambar += gbr
judul += jdl
tanggal += tgl
isiberita += isi
kategori += ktgri
sumber += sbr

gbr, jdl, tgl, isi, ktgri, sbr = scrapingKompas("health")

gambar += gbr
judul += jdl
tanggal += tgl
isiberita += isi
kategori += ktgri
sumber += sbr

gbr, jdl, tgl, isi, ktgri, sbr = scrapingKompas("lifestyle")

gambar += gbr
judul += jdl
tanggal += tgl
isiberita += isi
kategori += ktgri
sumber += sbr

gbr, jdl, tgl, isi, ktgri, sbr = scrapingSindo("edukasi", today)

gambar += gbr
judul += jdl
tanggal += tgl
isiberita += isi
kategori += ktgri
sumber += sbr

gbr, jdl, tgl, isi, ktgri, sbr = scrapingSindo("tekno", today)

gambar += gbr
judul += jdl
tanggal += tgl
isiberita += isi
kategori += ktgri
sumber += sbr

gbr, jdl, tgl, isi, ktgri, sbr = scrapingSindo("sports", today)

gambar += gbr
judul += jdl
tanggal += tgl
isiberita += isi
kategori += ktgri
sumber += sbr

gbr, jdl, tgl, isi, ktgri, sbr = scrapingSindo("lifestyle", today)

gambar += gbr
judul += jdl
tanggal += tgl
isiberita += isi
kategori += ktgri
sumber += sbr


for a in range(len(isiberita)):
    preprocessing_isi.append(preprocessing_text(isiberita[a]))

pred_label = []
pred_prob = []
#LOAD MODEL


model = Klasifikasi(len(class_names), DROPOUT)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#model = Klasifikasi(len(class_names))
#model.load_state_dict(torch.load("D:/Petra/Informatika/Skripsi/Skripsi/code/KODINGAN_FIX/softmax_layer_3e-05_0.2_3.bin", map_location=torch.device('cpu')))
#model.to(device)

for a in range(len(preprocessing_isi)):
    pred_label.append(PredictText(model, preprocessing_isi[a]))
    
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

try:
 	mydb = mysql.connector.connect(
 	host="localhost",
 	user="root",
 	password="",
 	database="skripsi"
 	)
 	mycursor = mydb.cursor()

 	for a in range(len(judul)):
 		mycursor.execute("INSERT into `news` (`tanggal`,`gambar`,`judul`,`isi`,`kategori`,`sumber`, `prediksi`) values('%s', '%s', '%s', '%s', '%s', '%s', '%s')" % (tanggal[a],gambar[a],judul[a], isiberita[a], kategori[a], sumber[a], toString(pred_label[a])))
 		mydb.commit()
except mysql.connector.Error as err:
 	print(err)
else:
 	mydb.close()
