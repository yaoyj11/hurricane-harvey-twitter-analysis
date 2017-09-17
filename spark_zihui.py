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
from operator import add

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

def parseWord(tweet,tt,types=None,to_remove=[]):
  words=tt.tokenize(tweet[-1])
  tags=nltk.pos_tag(words)
  res=[]
  for i in range(len(words)):
    if words[i] not in to_remove and (types is None or tags[i] in types):
      res.append((words[i].lower(),tags[i]))
  return res

def getWords(tweets,types=None,to_remove=[]):
  tt=TweetTokenizer()
  words=tweets.flatMap(lambda x:parseWord(x,tt,types,to_remove))
  return words


def countWord(words,types=None):
  word_count=words.filter(lambda x:types is None or x[1] in types)\
    .map(lambda x:(x[0],1))\
    .reduceByKey(add)
  return word_count


if __name__== "__main__":
  conf = SparkConf()
  conf.setMaster("local[8]").setAppName("YELP")
  sc = SparkContext(conf=conf)
  log4j = sc._jvm.org.apache.log4j
  log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)
  print("Set log level to Error")
  num_partitions=10
  sqlContext=SQLContext(sc)
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
  words.saveAsTextFile("words")
  #count words
  word_count=countWord(words)
  word_count.saveAsTextFile("word_count")
  print(word_count.sortBy(keyfunc=lambda x:x[1],ascending=False,numPartitions=num_partitions).top(50))



