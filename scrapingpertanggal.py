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
        for tgl in tanggal:
            txtTgl.append(tgl.text[:10])
        for image in images:
            txtGbr.append(image.find('img').get('src'))
        for item in items:
            txtJudul.append(item.find('a', 'article__link').text)
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
                if(listText.text):
                    if(content.find('strong').text[-1] == "m"):
                        textHapus.append(content.find('strong').text + " -")
                    else:
                        textHapus.append(content.find('strong').text)
                else:
                    if(content.findAll('strong')[1].text[-1] == "m"):
                        textHapus.append(content.findAll('strong')[1].text)
                    else:
                        textHapus.append(content.findAll('strong')[1].text)
                fullText = re.sub('\([Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa]([\s\S]*?)\)','',fullText)
                fullText = re.sub('[Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa].*','',fullText)


                for i in textHapus:
                    fullText = fullText.replace(re.sub('<[^<]+?>', '', str(i)),'')
                fullText = fullText.replace("'", "") #karena db kalau ada tanda '(single quote) error
                #print(txtGbr)
                #print("\n")
                
                txtIsi.append(fullText)
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
MAX_LEN = 400
PRE_TRAINED_MODEL_BAHASA =  'indobenchmark/indobert-base-p1'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_BAHASA)
class Klasifikasi(nn.Module):

  def __init__(self, n_classes):
    super(Klasifikasi, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_BAHASA)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False
    )
    output = self.drop(pooled_output)
    return self.out(output)
"""
def scrapingSindo(kategori, tanggal):
	if kategori == "edukasi":
		tmp = 'https://index.sindonews.com/index/144'
	elif kategori == "tekno":
		tmp = 'https://index.sindonews.com/index/612'
	elif kategori == "sports":
		tmp = 'https://index.sindonews.com/index/10'
	elif kategori == "lifestyle":
		tmp = 'https://index.sindonews.com/index/154'
		
	page = 0
	for i in range(1,3):
		url1 = tmp + "/{}/?t={}".format(page,tanggal)
		req1 = requests.get(url1, timeout=None)
		soup1 = BeautifulSoup(req1.text, 'html.parser')
		items = soup1.findAll('div', 'indeks-rows')
		for item in items:
			txtUrl.append(item.find('div', 'indeks-title').find("a").get('href'))
			txtJudul.append(item.find('div', 'indeks-title').text)
			txtGbr.append(item.find('div', 'indeks-pict').find('img').get('data-src'))
			txtKategori.append(kategori)
			txtTgl.append(tanggal)
			txtSumber.append("Sindonews")

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
				listText = content.findAll('strong')
				textHapus = []
				fullText = content.text
				i = 0

				for a in listText:
					i+=1
					if str(a.text[1:11].lower())=='baca juga:':
						textHapus.append(a)
					elif i == 1:
						textHapus.append(a.text+' - ')
				
				textHapus+=content.findAll('div', 'box-outlink')
				textHapus+=content.findAll('div', 'baca-inline')
				textHapus+=content.findAll('div', 'editor')
				textHapus+=content.findAll('div', 'reporter')

				for i in textHapus:
					fullText = fullText.replace(re.sub('<[^<]+?>', '', str(i)),'')

				fullText = fullText.replace("'", "")
				txtIsi.append(fullText)
		
		page += 15
"""
    
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
    
    

    #input_ids = encoded_review['input_ids'].to(device)
    #attention_mask = encoded_review['attention_mask'].to(device)
    
    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    
    output = model(input_ids, attention_mask)
    
    prob = torch.sigmoid(output)
    y_pred_prob = prob.detach().cpu().numpy()
    text = []
    for a in range(len(y_pred_prob)):
        
    
        indeks = 0
        for b in range(len(y_pred_prob[a])):
          if(y_pred_prob[a][b]>0.5):
            """indeks+=1
            if indeks==1:
              text += "%s"%(class_names[b])
            else:
              text += ", %s"%(class_names[b])
            #print(text)"""
            text.append(class_names[b])
    
    
    return text
    #return temp_labels
def toString(s):
    str1 = " "
    return (str1.join(s))
"""def scrapingSindo(kategori, tanggal):
	if kategori == "edukasi":
		tmp = 'https://index.sindonews.com/index/144'
	elif kategori == "tekno":
		tmp = 'https://index.sindonews.com/index/612'
	elif kategori == "sports":
		tmp = 'https://index.sindonews.com/index/10'
	elif kategori == "lifestyle":
		tmp = 'https://index.sindonews.com/index/154'
		
	page = 0
	for i in range(1,3):
		url1 = tmp + "/{}/?t={}".format(page,tanggal)
		req1 = requests.get(url1, timeout=None)
		soup1 = BeautifulSoup(req1.text, 'html.parser')
		items = soup1.findAll('div', 'indeks-rows')
		for item in items:
			txtUrl.append(item.find('div', 'indeks-title').find("a").get('href'))
			txtJudul.append(item.find('div', 'indeks-title').text)
			txtGbr.append(item.find('div', 'indeks-pict').find('img').get('data-src'))
			txtKategori.append(kategori)
			txtTgl.append(tanggal)
			txtSumber.append("Sindonews")

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
				listText = content.findAll('strong')
				textHapus = []
				fullText = content.text
				i = 0

				for a in listText:
					i+=1
					if str(a.text[1:11].lower())=='baca juga:':
						textHapus.append(a)
					elif i == 1:
						textHapus.append(a.text+' - ')
				
				textHapus+=content.findAll('div', 'box-outlink')
				textHapus+=content.findAll('div', 'baca-inline')
				textHapus+=content.findAll('div', 'editor')
				textHapus+=content.findAll('div', 'reporter')

				for i in textHapus:
					fullText = fullText.replace(re.sub('<[^<]+?>', '', str(i)),'')

				fullText = fullText.replace("'", "")
				txtIsi.append(fullText)
		
		page += 15"""

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


for a in range(len(isiberita)):
    preprocessing_isi.append(preprocessing_text(isiberita[a]))
#preprocessing_isi = preprocessing_text(isiberita)

pred_label = []
pred_prob = []
#LOAD MODEL
model = Klasifikasi(len(class_names))
#model.load_state_dict(torch.load("/content/drive/MyDrive/Skripsi/model/isiberita_epoch4.bin", map_location=torch.device('cpu')))

model.load_state_dict(torch.load("isiberita_epoch4.bin", map_location=torch.device('cpu')))

#model.to(device)


for a in range(len(preprocessing_isi)):
    pred_label.append(PredictText(model, preprocessing_isi[a]))
    


# scrapingKompas("lifestyle", "2021-03-09")

#scrapingSindo("edukasi", "2021-03-09")
#scrapingSindo("tekno", "2021-03-09")
#scrapingSindo("sports", "2021-03-09")
#scrapingSindo("lifestyle", "2021-03-09")

# for a in range(len(txtJudul)):
# 	print(txtGbr[a])




end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

# for a in range(len(txtJudul)):
# 	print(txtJudul[a])
#print(len(txtTgl), len(txtGbr), len(txtJudul), len(txtUrl), len(txtIsi), len(txtKategori), len(txtSumber) )

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
#scrapingKompas("lifestyle", "2021-03-09")

#scrapingSindo("edukasi", "2021-03-09")
#scrapingSindo("tekno", "2021-03-09")
#scrapingSindo("sports", "2021-03-09")
#scrapingSindo("lifestyle", "2021-03-09")

# for a in range(len(txtJudul)):
# 	print(txtGbr[a])

