import requests
import time
from bs4 import BeautifulSoup
import re
import pandas as pd

txtJudul = []
txtKategori = []
txtIsi = []
txtUrl = []
txtGbr = []
tesUrl = []
tesIndeks = []
tesListText1 = []

start = time.time()
def scrapingKompas(kategori):
    url = 'https://indeks.kompas.com/?site={}&date='.format(kategori)
    #32
    for tgl in range(1,32):
      #Looping per page
      url1 = url+'2021-01-{}&page='.format(tgl)  
      #manual sdh di cek maks hny 4
      for page in range(1,4):
          req1 = requests.get(url1+"{}".format(page))
          soup = BeautifulSoup(req1.text, 'html.parser')
          items = soup.findAll('div', 'article__list__title')
          for item in items:
           txtUrl.append(item.find('a').get('href'))
           txtJudul.append(item.find('a', 'article__link').text)
           txtKategori.append(kategori)
           
           tUrl = "{}?page=all".format(item.find('a').get('href'))
           reqUrl = requests.get(tUrl)
           soupUrl = BeautifulSoup(reqUrl.text, 'html.parser')
           
           contents = soupUrl.findAll('div', 'read__content')
           #tesUrl.append(tUrl)
           
           for content in contents:
               textHapus = []
               listText = content.findAll('strong')
               fullText = content.text
               #listText=content.find('p').find('strong')
               #print(listText.text)
               i = 0
               for a in listText:
                   i+=1
                   if(a.text[:10])=="Baca juga:":
                       textHapus.append(a)
                   elif (a.text[:10].lower())=="kompas.com":
                       textHapus.append(a)
                    
               fullText = re.sub('\([Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa]([\s\S]*?)\)','',fullText)
               fullText = re.sub('[Bb][Aa][Cc][Aa] [Jj][Uu][Gg][Aa].*','',fullText)
               
               for i in textHapus:
                   #fullText = fullText.replace(re.sub('<[^<]+?>', '', str(i)),'')
                   fullText = fullText.replace(re.sub('<[^<]+?>', '', str(i)),'')

               fullText = fullText.replace("“", "")
               fullText = fullText.replace("'", "") #karena db kalau ada tanda '(single quote) error
               txtIsi.append(fullText)
def scrapingSindo(kategori):
    #url = ""

    if kategori == "edukasi":
        tmp = 'https://index.sindonews.com/index/144'

    elif kategori == "tekno":
        tmp = 'https://index.sindonews.com/index/612'

    elif kategori == "sports":
        tmp = 'https://index.sindonews.com/index/10'

    elif kategori == "lifestyle":
        tmp = 'https://index.sindonews.com/index/154'

    
    for tgl in range(1,32):
        
        #url = tmp + '?t=2021-01-{}'.format(tgl)
        page = 0
        
        
        for i in range(1,4):
            url1 = tmp + "/{}/?t=2021-01-{}".format(page,tgl)
            req1 = requests.get(url1, timeout=None)
            soup1 = BeautifulSoup(req1.text, 'html.parser')
            items = soup1.findAll('div', 'indeks-rows')
            
            for item in items:
                txtUrl.append(item.find('div', 'indeks-title').find("a").get('href'))
                txtJudul.append(item.find('div', 'indeks-title').text)
                txtKategori.append(kategori)
                tesIndeks.append(len(txtJudul))
                tUrl = "{}?showpage=all".format(item.find('div', 'indeks-title').find("a").get('href'))
                tesUrl.append(tUrl)
                reqUrl = requests.get(tUrl)
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
                    #txtIsi.append(content.text)
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
                        """for a in listSurvey:
                            if a.findNext('strong'):
                                if(a.findNext('strong').text[1:13]=='Ikuti Survei'):
                                
                                    textHapus.append(a.findNext('strong').text)"""
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
                                    #print(a.text, tesIndeks[-1])
                                
                                    #print(a.text, tesIndeks[-1])
                                #print(a.text)
                    elif(kategori == 'tekno'):
                        """
                        tekno:
                        BACA JUGA:          3,1
                         BACA JUGA-         11
                        (Baca juga :        6,5         v
                        (Baca juga          8           v
                         BACA JUGA -        2           v
                        (Baca juga: )       5
                        Baca juga           7
                        """
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
                    
                    txtIsi.append(fullText)
                    
            page+=15

scrapingKompas("edukasi")
scrapingKompas("tekno")
scrapingKompas("sports")
scrapingKompas("health")
scrapingKompas("lifestyle")

scrapingSindo('edukasi')
scrapingSindo('sports')
scrapingSindo('tekno')
scrapingSindo('lifestyle')

for i in range(len(txtIsi)):
    if re.search('baca juga', txtIsi[i], re.IGNORECASE):
        print("baca juga:",i)
    if re.search('ikuti survei', txtIsi[i], re.IGNORECASE):
        print("ikuti survey", i)
    if re.search('lihat juga', txtIsi[i], re.IGNORECASE):
        print("lihat juga:", i)
    if re.search('lihat grafis', txtIsi[i], re.IGNORECASE):
        print("lihat grafis:", i)
    
df = pd.DataFrame({
        "judul": txtJudul,
        "url": txtUrl,
        "isi": txtIsi,
        "kategori": txtKategori
    })
df.to_csv('datasetBerita.csv', index=False, encoding='utf-8')
#df.to_csv('beritaSindo.csv', index=False, encoding='utf-8')
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
