from django.shortcuts import render,HttpResponse
from home.models import Destination
import pandas as pd
from django.http import JsonResponse
import json


# from rest_framework.views import APIView
# from rest_framework.response import Response



# Create your views here.





def contact(request):
    return render(request,'contact.html')

def cleanrepo(request):
    return render(request,'cleanrepo.html')

def link(request):
    return render(request,'link.html')

def analyhome(request):
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = df.index #will contain index of all CSE papers
    
    uri = word_cloud(df, index, 'CSE')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    depcount = df1.groupby(['Department'], as_index=False).agg({'count': 'sum'})
    x=list(depcount['Department'])
    y=list(depcount['count'])





    dictionary={'labels':x,'dataset':y,'data':uri,'dept_name':'Overall'}

    return render(request,'analytics_home.html',dictionary)
    # return render(request,'analytics_home.html')

def analysis(request):
    return render(request,'analysis.html')

def analytics(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = df.index #will contain index of all CSE papers
    
    uri = word_cloud(df, index, 'CSE')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    depcount = df1.groupby(['Department'], as_index=False).agg({'count': 'sum'})
    x=list(depcount['Department'])
    y=list(depcount['count'])





    dictionary={'labels':x,'dataset':y,'data':uri, 'dept_name':"Overall"}

    return render(request,'analytics_home.html',dictionary)

def home(request):
    return render(request,'home.html')
def recom(request):
    return render(request,'recom.html')
def repo(request):
    data=Destination.objects.all()

    return render(request,'repo.html',{"datacoll":data})
def splash(request):
    return render(request,'splash.html')

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from scipy.sparse import coo_matrix
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn):
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    for idx, score in sorted_items:
        
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    
    return results

import io
import urllib, base64

def word_cloud(dataset, index, department): #dataset is the entire dataframe, index is indices of dept
    cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=1000)
    corpus_dept = [] #contains the abstract of all papers for 'department'
    for i in index:
      corpus_dept.append(dataset['Abstract'][i])

    X = cv.fit_transform(corpus_dept)

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(X)
    feature_names = cv.get_feature_names()

    keywords_dict = {}
    for i in range(len(corpus_dept)):
      doc = corpus_dept[i]
      tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
      sorted_items = sort_coo(tf_idf_vector.tocoo())
      keywords = extract_topn_from_vector(feature_names, sorted_items, 30)
      keywords_dict.update(keywords)

    wordcloud = WordCloud(background_color='white',
                          stopwords=stop_words,
                          max_words=1000,
                          max_font_size=50, 
                          random_state=42,
                         ).generate_from_frequencies(keywords_dict)
    print(wordcloud)
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return uri 
    

def cseanalytics(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'CSE':
            index.append(i)

    uri = word_cloud(df, index, 'CSE')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("CSE")
    x=list(df2['Date1'])
    y=list(df2['count'])

    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'CSE'}
    return render(request,'analytics_home.html',dictionary)

def ae(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'AE':
            index.append(i)

    uri = word_cloud(df, index, 'AE')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("AE")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'AE'}

    return render(request,'analytics_home.html',dictionary)


def bsbe(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'BSBE':
            index.append(i)

    uri = word_cloud(df, index, 'BSBE')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("BSBE")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'BSBE'}

    return render(request,'analytics_home.html',dictionary)



def ce(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'CE':
            index.append(i)

    uri = word_cloud(df, index, 'CE')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("CE")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'CE'}

    return render(request,'analytics_home.html',dictionary)

def celp(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'CELP':
            index.append(i)

    uri = word_cloud(df, index, 'CELP')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("CELP")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'CELP'}

    return render(request,'analytics_home.html',dictionary)

def che(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'CHE':
            index.append(i)

    uri = word_cloud(df, index, 'CHE')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("CHE")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'CHE'}

    return render(request,'analytics_home.html',dictionary)

def chm(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'CHM':
            index.append(i)

    uri = word_cloud(df, index, 'CHM')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("CHM")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'CHM'}

    return render(request,'analytics_home.html',dictionary)

def civil(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'Civil':
            index.append(i)

    uri = word_cloud(df, index, 'Civil')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("CIVIL")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'CIVIL'}

    return render(request,'analytics_home.html',dictionary)

def des(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'DES':
            index.append(i)

    uri = word_cloud(df, index, 'DES')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("DES")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'DES'}

    return render(request,'analytics_home.html',dictionary)

def dp(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'DP':
            index.append(i)

    uri = word_cloud(df, index, 'DP')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("DP")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'DP'}

    return render(request,'analytics_home.html',dictionary)

def eco(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'ECO':
            index.append(i)

    uri = word_cloud(df, index, 'ECO')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("ECO")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'ECO'}

    return render(request,'analytics_home.html',dictionary)

def ee(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'EE':
            index.append(i)

    uri = word_cloud(df, index, 'EE')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("EE")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'EE'}

    return render(request,'analytics_home.html',dictionary)

def eem(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'EEM':
            index.append(i)

    uri = word_cloud(df, index, 'EEM')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("EEM")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'EEM'}

    return render(request,'analytics_home.html',dictionary)

def eemp(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'EEMP':
            index.append(i)

    uri = word_cloud(df, index, 'EEMP')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("EEMP")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'EEMP'}

    return render(request,'analytics_home.html',dictionary)

def es(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'ES':
            index.append(i)

    uri = word_cloud(df, index, 'ES')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("ES")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'ES'}

    return render(request,'analytics_home.html',dictionary)

def hss(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'HSS':
            index.append(i)

    uri = word_cloud(df, index, 'HSS')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("HSS")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'HSS'}

    return render(request,'analytics_home.html',dictionary)

def ime(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'IME':
            index.append(i)

    uri = word_cloud(df, index, 'IME')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("IME")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'IME'}

    return render(request,'analytics_home.html',dictionary)

def lt(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'LT':
            index.append(i)

    uri = word_cloud(df, index, 'LT')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("LT")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'LT'}

    return render(request,'analytics_home.html',dictionary)
def ltp(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'LTP':
            index.append(i)

    uri = word_cloud(df, index, 'LTP')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("LTP")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri, 'dept_name':"LTP"}

    return render(request,'analytics_home.html',dictionary)

def math(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'MATH & STATS':
            index.append(i)

    uri = word_cloud(df, index, 'MATH & STATS')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("MATH & STATS")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'MATH'}

    return render(request,'analytics_home.html',dictionary)
def me(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'ME':
            index.append(i)

    uri = word_cloud(df, index, 'ME')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("ME")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'ME'}

    return render(request,'analytics_home.html',dictionary)
def mme(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'MME':
            index.append(i)

    uri = word_cloud(df, index, 'MME')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("MME")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'MME'}

    return render(request,'analytics_home.html',dictionary)
def mse(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'MSE':
            index.append(i)

    uri = word_cloud(df, index, 'MSE')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("MSE")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'MSE'}

    return render(request,'analytics_home.html',dictionary)
def msp(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'MSP':
            index.append(i)

    uri = word_cloud(df, index, 'MSP')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("MSP")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'MSP'}

    return render(request,'analytics_home.html',dictionary)
def net(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'NET':
            index.append(i)

    uri = word_cloud(df, index, 'NET')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("NET")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'NET'}

    return render(request,'analytics_home.html',dictionary)
def netp(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'NETP':
            index.append(i)

    uri = word_cloud(df, index, 'NETP')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("NETP")
    x=list(df2['Date1'])
    y=list(df2['count'])





    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'NETP'}

    return render(request,'analytics_home.html',dictionary)
def phy(request):
    #data = Destination.objects.all()
    df=pd.DataFrame(list(Destination.objects.all().values()))
    index = [] #will contain index of all CSE papers
    for i in range(df.shape[0]):
        if df['Department'][i] == 'PHY':
            index.append(i)

    uri = word_cloud(df, index, 'PHY')
    df['count'] = 1
    df1 = df.groupby(['Date1', 'Department'], as_index=False).agg({'count': 'sum'})
    df1.sort_values(by=['Date1', 'Department'], inplace=True)
    df2 = df1.groupby(df1.Department)
    df2=df2.get_group("PHY")
    x=list(df2['Date1'])
    y=list(df2['count'])
    dictionary={'labels':x,'dataset':y, 'data':uri,'dept_name':'PHY'}

    return render(request,'analytics_home.html',dictionary)

def cserepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("CSE")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'cserepo.html',{"data":data1})








def autosuggest(request):
    #print(request.GET)
    query_original = request.GET.get('term')
    query_set =  Destination.objects.filter(Title__icontains=query_original)
     #print(query_set)
    my_list = []
    my_list += [x.Title for x in query_set]
    if len(my_list) > 10:
        my_list = my_list[0:10]
        #print(temp)
    return JsonResponse(my_list, safe=False)


def getidrec(request):
    if request.method=='GET':
        print(request.GET)
        search=request.GET.get('search')
        print(search)



    return render(request,"recom.html")


import scipy.sparse as sp
from numpy import dot
from numpy.linalg import norm
import numpy as np
import sys

vector = sp.load_npz('./vector.npz')


def recommend(article):
    a = (vector[article]).toarray()
    scores = []
    for i in range(17293):
        if i != article:
            b = (vector[i]).toarray()
            cos_sim = dot(a, np.transpose(b)) / (norm(a) * norm(b))
            scores.append([i, cos_sim[0]])
    scores = sorted(scores, reverse=True, key=lambda x: x[1])

    # print(Destination.objects.get(id=article+1))
    # print("====================================================================================================")
    # print("Recommended articles = >")
    templist = []  
    for i in range(5):
        temp_dict = {}
        temp_dict['Title'] = Destination.objects.get(id=scores[i][0] + 1).Title
        temp_dict['Author'] = Destination.objects.get(id=scores[i][0] + 1).Author
        temp_dict['Supervisor'] = Destination.objects.get(id=scores[i][0] + 1).Supervisor
        temp_dict['Degree'] = Destination.objects.get(id=scores[i][0] + 1).Degree
        temp_dict['Department'] = Destination.objects.get(id=scores[i][0] + 1).Department
        temp_dict['URL'] = Destination.objects.get(id=scores[i][0] + 1).URL

        # print("=================================================================================================")
        # print(Destination.objects.get(id=scores[i][0] + 1).Title)
        # templist.append(Destination.objects.get(id=scores[i][0] + 1).Title)
        templist.append(temp_dict)

    return templist



def recommend_based_on_sup(article_id):
    title = Destination.objects.get(id=article_id).Title
    supervisor = Destination.objects.get(id=article_id).Supervisor
    # print(supervisor)
    my_list = Destination.objects.filter(Supervisor=supervisor)
    
    recommend_list = []

    # print(my_list)
    for i in my_list:
        # print(i)
        recommend_dict = {}
        recommend_dict['Title'] =  Destination.objects.get(Title=i).Title
        recommend_dict['Author'] =  Destination.objects.get(Title=i).Author
        recommend_dict['Supervisor'] =  Destination.objects.get(Title=i).Supervisor
        recommend_dict['Degree'] =  Destination.objects.get(Title=i).Degree
        recommend_dict['Department'] =  Destination.objects.get(Title=i).Department
        recommend_dict['URL'] =  Destination.objects.get(Title=i).URL 

        recommend_list.append(recommend_dict)

    if len(recommend_list)>5:
        recommend_list = recommend_list[0:5]

    return recommend_list


def recommend_results(request):
    if request.method == 'GET':
        # print(request.GET)
        search = request.GET.get('search')
        # print(search)

        # machine recommendation model 
            # id = Destination.objects.filter(Title__icontains=search)
        id_data = Destination.objects.get(Title=search).id
        recommendations_list = recommend(id_data-1)
        recommendation_based_sup = recommend_based_on_sup(id_data)
        # print(recommendation_based_sup)

        input_dict = {}
        input_dict['Title'] =  Destination.objects.get(id=id_data).Title
        input_dict['Author'] =  Destination.objects.get(id=id_data).Author
        input_dict['Supervisor'] =  Destination.objects.get(id=id_data).Supervisor
        input_dict['Degree'] =  Destination.objects.get(id=id_data).Degree
        input_dict['Department'] =  Destination.objects.get(id=id_data).Department
        input_dict['URL'] =  Destination.objects.get(id=id_data).URL
        #

        return render(request, 'home.html', {'recommendations':recommendations_list,
                                                'recommendations_supervisor':recommendation_based_sup,
                                                'input':input_dict,
                                                'data':'yes'})

#########
def aerepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("AE")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'aerepo.html',{"data":data1})
#########
def bsberepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("BSBE")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'bsberepo.html',{"data":data1})
##########
def cerepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("CE")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'cerepo.html',{"data":data1})
#######
def celprepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("CELP")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'celprepo.html',{"data":data1})
########
def cherepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("CHE")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'cherepo.html',{"data":data1})
#####
def chmrepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("CHM")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'chmrepo.html',{"data":data1})
##########
def civilrepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("CIVIL")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'civilrepo.html',{"data":data1})
##########
def desrepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("DES")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'desrepo.html',{"data":data1})
########
def dprepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("DP")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'dprepo.html',{"data":data1})
########
def ecorepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("ECO")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'ecorepo.html',{"data":data1})
##########
def eerepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("EE")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'eerepo.html',{"data":data1})
######
def eemrepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("EEM")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'eemrepo.html',{"data":data1})
##########
def eemprepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("EEMP")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'eemprepo.html',{"data":data1})

#########
def esrepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("ES")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'esrepo.html',{"data":data1})
#######3
def hssrepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("HSS")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'hssrepo.html',{"data":data1})
####
def imerepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("IME")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'imerepo.html',{"data":data1})
########
def ltrepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("LT")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'ltrepo.html',{"data":data1})
########3
def ltprepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("LTP")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'ltprepo.html',{"data":data1})
#####
def mathrepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("MATH")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'mathrepo.html',{"data":data1})
######
def merepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("ME")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'merepo.html',{"data":data1})

#########
def mmerepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("MME")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'mmerepo.html',{"data":data1})
#######
def mserepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("MSE")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'mserepo.html',{"data":data1})
#######
def msprepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("MSP")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'msprepo.html',{"data":data1})
########
def netrepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("NET")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'netrepo.html',{"data":data1})
##########
def netprepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("NETP")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'netprepo.html',{"data":data1})
########
def phyrepo(request):
    df = pd.DataFrame(list(Destination.objects.all().values()))
    print(df.columns)
    #df1=df.groupby(['Department'],as_index=False)
    df1 = df.groupby(df.Department)
    df2 = df1.get_group("PHY")
    print(len(df2))
    json_records = df2.to_json(orient='records')
    data1 = []
    data1 = json.loads(json_records)

    return render(request, 'phyrepo.html',{"data":data1})