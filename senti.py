import nltk.sentiment.util as su
import pandas as pd
import numpy as np
import nltk
reader=pd.read_csv("Harvey_tweets.csv",encoding="ISO-8859-1").dropna(how='any').as_matrix()
tmp_list=list(reader)
a=np.array(tmp_list)
text0=np.delete(a,(0,1,2,3,4),1)
text1=[]
for tweet in text0:
    a=[]
    a.append(tweet[0])
    a.append(su.demo_liu_hu_lexicon(tweet[1],False,False))
    text1+=a
print(text1[0])
