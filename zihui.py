#read in file
import pandas as pd
import numpy as np
import nltk

reader=pd.read_csv("~/Desktop/ncdata/Harvey_tweets.csv",encoding="ISO-8859-1").dropna(how='any').as_matrix()
tmp_list=list(reader)
text0=np.array(tmp_list)

#tokenize tweets
from nltk import TweetTokenizer
tt=TweetTokenizer()
text1=[]
for line in text0:
    res=tt.tokenize(line[-1])
    text1.append(res)  

import string
import operator

#count numbers of each word and store into a dictionary
word_count={}
for line in text1:
    for word in line:
        if all(c in string.punctuation or c.isdigit() for c in word):
            continue
        if len(word)==1 and all(c.isalpha() for c in word):
            continue;
        if word.lower() in word_count.keys():
            word_count[word.lower()]+=1
        else:
            word_count[word.lower()]=1

#sort the dictionary
sorted_count=sorted(word_count.items(), key=lambda x: (-x[1], x[0]))


nltk.download('stopwords')


#remove stopwords and "hurricane" related words
from nltk.corpus import stopwords
to_remove=stopwords.words('english')
to_remove=to_remove+["i'm",'u','us','via',"it's",'hurricane','hurricanes','harvey','#harvey2017','#hurricaneharvey','#hurricaneharvey2017','#hurricane',"harvey's"]
filtered=[tuple for tuple in sorted_count if not tuple[0] in to_remove]
print(filtered[0:1000])


words=[t[0] for t in filtered]
tags_no_count=nltk.pos_tag(words)
tags=[]
for i in range(len(filtered)):
    l=[]
    l.append(filtered[i][0])
    l.append(filtered[i][1])
    l.append(tags_no_count[i][1])
    tags=tags+l

