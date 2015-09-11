#!/usr/bin/env python

import sys
from operator import add
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
from pyspark.sql import SQLContext
import os

# put in the path to the kaggle data
PATH_TO_JSON = "/user/alexeys/KaggleDato/Preprocessed/"
PATH_TO_TRAIN_LABELS = "/scratch/network/alexeys/KaggleDato/train.json"
PATH_TO_SUB_LABELS = "/scratch/network/alexeys/KaggleDato/sampleSubmission.json"

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Module-level global variables for the `tokenize` function below
#PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Function to break text into "tokens"
def tokenize(text):
    tokens = word_tokenize(text)
    no_stopwords = filter(lambda x: x not in STOPWORDS,tokens)
    stemmed = map(lambda w: STEMMER.stem(w),no_stopwords)
    #s = set(stemmed)
    #stemmed = list(s)
    return filter(lambda x: len(x) > 1,stemmed)


def main(argv):
    #STEP1: data ingestion
    sc = SparkContext(appName="PythonWordCount")
    sqlContext = SQLContext(sc)

    #read data into RDD
    input_schema_rdd = sqlContext.read.json("file:///scratch/network/alexeys/KaggleDato/Preprocessed/0_1/part-*")
    #input_schema_rdd.show() 
    #input_schema_rdd.printSchema()
    #input_schema_rdd.select("id").show()

    train_label_rdd = sqlContext.read.json("file://"+PATH_TO_TRAIN_LABELS)
    sub_label_rdd = sqlContext.read.json("file://"+PATH_TO_SUB_LABELS)

    input_schema_rdd.registerTempTable("input")
    train_label_rdd.registerTempTable("train_label")
    sub_label_rdd.registerTempTable("sub_label")

    # SQL can be run over DataFrames that have been registered as a table.
    train_wlabels_0 = sqlContext.sql("SELECT title,text,images,links,label FROM input JOIN train_label WHERE input.id = train_label.id AND label = 0")
    train_wlabels_1 = sqlContext.sql("SELECT title,text,images,links,label FROM input JOIN train_label WHERE input.id = train_label.id AND label = 1")

    text_only_0 = train_wlabels_0.map(lambda p: p.text)
    text_only_1 = train_wlabels_1.map(lambda p: p.text)

    counts0 = text_only_0.flatMap(lambda line: tokenize(line))\
          .map(lambda x: (x, 1)) \
          .reduceByKey(add)

    counts1 = text_only_1.flatMap(lambda line: tokenize(line))\
          .map(lambda x: (x, 1)) \
          .reduceByKey(add)

    relevance = counts0.subtractByKey(counts1).map(lambda (x,y): (y,x)).sortByKey(False, 1)
    relevance.saveAsTextFile("/user/alexeys/KaggleDato/WordCount")

if __name__ == "__main__":
   main(sys.argv)
