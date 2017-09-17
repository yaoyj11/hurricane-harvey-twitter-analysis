import nltk.sentiment.util as su
import pandas as pd
import numpy as np
import nltk

reader=pd.read_csv("Harvey_tweets.csv",encoding="ISO-8859-1").dropna(how='any').as_matrix()
of=open("Harvey_tweets_senti.csv","w")
for tweet in reader:
  a=list(tweet)
  a.append(su.demo_liu_hu_lexicon(tweet[1],False,False))
  of.write(str(a[0])+","+str(a[1])+","+str(a[2])+","+str(a[3])+","+str(a[4])+",\""+a[5]+"\",\""+a[6]+"\"\n")
of.close()





