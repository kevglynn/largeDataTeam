#!/usr/bin/env python

from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
from pyspark.sql import SQLContext

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import re
import sys
import os
import json
import numpy as np

# put in the path to the kaggle data
PATH_TO_JSON = "/user/alexeys/KaggleDato/Preprocessed/"
PATH_TO_TRAIN_LABELS = "/scratch/network/alexeys/KaggleDato/train.json"
PATH_TO_SUB_LABELS = "/scratch/network/alexeys/KaggleDato/sampleSubmission.json"

# Module-level global variables for the `tokenize` function below
#PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Function to break text into "tokens"
def tokenize(text):
    tokens = word_tokenize(text)
    no_stopwords = filter(lambda x: x not in STOPWORDS,tokens)
    stemmed = map(lambda w: STEMMER.stem(w),no_stopwords)
    s = set(stemmed)
    stemmed = list(s)
    return filter(None,stemmed)

# Load and parse the data
def parsePoint(label,beast):
    #This is the beast:
    #((SparseVector(10, {}), 0), 0),0)
    arraypart = beast[0][0][0].toArray()

    #adhoc non-text features
    image = beast[0][0][1]
    links = beast[0][1]
    title = len(beast[1])

    adhocpart = np.array([image,links,title])

    return LabeledPoint(label, np.hstack((arraypart,adhocpart))) #arraypart)
    

def main(argv):
    #STEP1: data ingestion
    sc = SparkContext(appName="KaggleDato_Step2")
    sqlContext = SQLContext(sc)

    #read data into RDD
    input_schema_rdd = sqlContext.read.json("file:///scratch/network/alexeys/KaggleDato/Preprocessed/0_1/part-00000")
    #input_schema_rdd.show() 
    #input_schema_rdd.printSchema()
    #input_schema_rdd.select("id").show()

    train_label_rdd = sqlContext.read.json(PATH_TO_TRAIN_LABELS)
    sub_label_rdd = sqlContext.read.json(PATH_TO_SUB_LABELS)

    input_schema_rdd.registerTempTable("input")
    train_label_rdd.registerTempTable("train_label")
    sub_label_rdd.registerTempTable("sub_label")

    # SQL can be run over DataFrames that have been registered as a table.
    train_wlabels_0 = sqlContext.sql("SELECT title,text,images,links,label FROM input JOIN train_label WHERE input.id = train_label.id AND label = 0")
    train_wlabels_1 = sqlContext.sql("SELECT title,text,images,links,label FROM input JOIN train_label WHERE input.id = train_label.id AND label = 1")

    sub_wlabels = sqlContext.sql("SELECT title,text,images,links,label FROM input JOIN sub_label WHERE input.id = sub_label.id")

    text_only_0 = train_wlabels_0.map(lambda p: p.text)
    text_only_1 = train_wlabels_1.map(lambda p: p.text)
    image_only_0 = train_wlabels_0.map(lambda p: p.images)
    image_only_1 = train_wlabels_1.map(lambda p: p.images)
    links_only_0 = train_wlabels_0.map(lambda p: p.links)
    links_only_1 = train_wlabels_1.map(lambda p: p.links)
    title_only_0 = train_wlabels_0.map(lambda p: p.title)
    title_only_1 = train_wlabels_1.map(lambda p: p.title)

    tf = HashingTF(numFeatures=10)
    #preprocess text features
    text_documents_0 = text_only_0.map(lambda line: tokenize(line)).map(lambda word: tf.transform(word))
    text_documents_1 = text_only_1.map(lambda line: tokenize(line)).map(lambda word: tf.transform(word))

    #add them adhoc non-text features
    documents_0 = text_documents_0.zip(image_only_0).zip(links_only_0).zip(title_only_0)
    documents_1 = text_documents_1.zip(image_only_1).zip(links_only_1).zip(title_only_1)

    #turn into a format expected by MLlib classifiers
    labeled_tfidf_0 = documents_0.map(lambda row: parsePoint(0,row))
    labeled_tfidf_1 = documents_1.map(lambda row: parsePoint(1,row))
    #print labeled_tfidf_0.take(2)

    labeled_tfidf = labeled_tfidf_0.union(labeled_tfidf_1)
    #print labeled_tfidf.count()
    #print labeled_tfidf.collect()
    labeled_tfidf.cache()

    #CV split
    (trainData, cvData) = labeled_tfidf.randomSplit([0.7, 0.3])
    trainData.cache()
    cvData.cache()

    #Try various classifiers
    #With logistic regression only use training data
    #model = LogisticRegressionWithLBFGS.train(trainData)
    #Logistic regression works a lot better
    #model = NaiveBayes.train(trainData)
    #random forest
    model = RandomForest.trainClassifier(trainData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

    ## Evaluating the model on training data
    #labelsAndPreds = cvData.map(lambda p: (p.label, model.predict(p.features)))
    ##print labelsAndPreds.collect()
    #trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(cvData.count())
    #print("CV Error = " + str(trainErr))

    # Evaluate model on test instances and compute test error
    predictions = model.predict(cvData.map(lambda x: x.features))
    labelsAndPredictions = cvData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(cvData.count())
    print('Test Error = ' + str(testErr))
    print('Learned classification forest model:')
    print(model.toDebugString())

if __name__ == "__main__":
   main(sys.argv)
