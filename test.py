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

if __name__=="__main__":
  conf = SparkConf()
  conf.setMaster("local[8]").setAppName("YELP")
  sc = SparkContext(conf=conf)
  log4j = sc._jvm.org.apache.log4j
  log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)
  data=sc.sequenceFile("wordspickle")
  for rec in data.top(10):
    print(rec)
