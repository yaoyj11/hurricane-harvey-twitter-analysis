from __future__ import print_function
from operator import add
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import re
import random
import numpy
from array import array
import math
import nltk
import pandas as pd
from nltk import TweetTokenizer

def parseTweets(tweet):
  csv_reader=csv.reader([tweet])


def readAllTweets(sc,filepath,num_partitions=10):
  print("read tweets\n")
  #tweets=sc.textFile(filepath,num_partitions).map(parseTweets)
  customSchema = StructType([\
      StructField("NO",IntegerType(),True),\
      StructField("ID",FloatType(),True),\
      StructField("Likes",IntegerType(),True),\
      StructField("Replies",IntegerType(),True),\
      StructField("Retweets",IntegerType(),True),\
      StructField("Time",StringType(),True),\
      StructField("Tweet",StringType(),True)])
        #NO,ID,Likes,Replies,Retweets,Time,Tweet
        # 26,9.01E+17,1,0,1,8/25/2017 14:44,Hurricane Harvey dumbass bet not curve
  df = sqlContext.read.format('com.databricks.spark.csv').options(header='true',mode="DROPMALFORMED").load(filepath,schema=customSchema)
  return df.rdd.map(lambda x: (x["NO"],x["ID"],x["Likes"],x["Replies"],x["Retweets"],x["Time"],x["Tweet"]))

def parseWord(tweet,tt,types,to_remove=[]):
  words=tt.tokenize(tweet[-1])
  tags=nltk.pos_tag(words)
  res=[]
  for i in range(len(tags)):
    if tags[i][0] not in to_remove and (types is None or tags[i][1] in types):
      res.append((tags[i][0].lower(),tags[i][1],tweet[4],tweet[5]))
  return res

def getWords(tweets,types=None,to_remove=[]):
  tt=TweetTokenizer()
  words=tweets.flatMap(lambda x:parseWord(x,tt,types,to_remove))
  return words


def countWord(words,types=None,retweets=False):

  word_count=words.filter(lambda x:types is None or x[1] in types)
  if retweets:
    word_count=word_count.map(lambda x:(x[0],x[2]+1))
  else:
    word_count=word_count.map(lambda x:(x[0],1))
  word_count=word_count.reduceByKey(add)
  return word_count

#words: ("word",label,retweets,time)
#return (word,time),count
def groupWordsByTime(words,retweets=True):
  if retweets:
    words=words.map(lambda x:((x[0],x[3].split(":")[0]),x[2]+1))
  else:
    words=words.map(lambda x:((x[0],x[3].split(":")[0]),1))
  words=words.reduceByKey(add)
  return words

#(words,time),count
def firstK(l,k):
  l=list(l)
  l=sorted(l,key=lambda x:-x[1])
  return l[0:k]

def mergeWords(words_count,K=100,num_partitions=10):
  agg=words_count.map(lambda x:(x[0][1],(x[0][0],x[1])))\
      .groupByKey()\
      .map(lambda x:(x[0],firstK(x[1],K)))
  return agg

#draw word_cloud


if __name__== "__main__":
  ST="8/25/2017 14"
  ET="8/29/2017 14"
  conf = SparkConf()
  conf.setMaster("local[8]").setAppName("YELP")
  sc = SparkContext(conf=conf)
  log4j = sc._jvm.org.apache.log4j
  log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)
  print("Set log level to Error")
  num_partitions=10
  sqlContext=SQLContext(sc)
  #tweets=readAllTweets(sqlContext,"small.csv",num_partitions)
  tweets=readAllTweets(sqlContext,"Harvey_tweets.csv",num_partitions)
  print("finished reading ",tweets.count())
#364255
  print(tweets.top(10))
  #remove words
  from nltk.corpus import stopwords
  to_remove=stopwords.words('english')
  to_remove=to_remove+["i'm",'u','us','via',"it's"]
  types=["DT","JJ","JJR","JJS","MD","NN","NNP","NNPS","NNS","PDT","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ"]
  #get words
  words=getWords(tweets,types,to_remove)
  print("number of words:",words.count())
  words.saveAsTextFile("words")
  words.saveAsPickleFile("wordspickle")
  #count words
  word_count=countWord(words)
  word_count.saveAsTextFile("word_count")
  word_count.saveAsPickleFile("word_countpickle")
  print(word_count.sortBy(keyfunc=lambda x:x[1],ascending=False,numPartitions=num_partitions).top(50))

  words=groupWordsByTime(words)
#(word,time),count
  print(words.top(10))
  words=mergeWords(words)
  print(words.top(10))
  words.saveAsTextFile("period")


  prin  (words.top(10))
