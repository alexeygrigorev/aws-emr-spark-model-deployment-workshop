import pickle
from itertools import islice

import numpy as np
import pandas as pd
import xgboost as xgb

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql import types

import boto3


def split_into_batches(iterable, size):
    while True:
        batch = islice(iterable, size)
        batch = list(batch)
        if len(batch) == 0:
            break
        yield batch


s3 = boto3.client('s3')
s3.download_file('spark-workshop-data', 'model.pkl', 'model.pkl')


with open('model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)



def apply_model(columns, model, partition):
    for batch in split_into_batches(partition, 10000):
        df_batch = pd.DataFrame(batch, columns=columns)

        X = df_batch[['f_views_fraction', 'f_replies_fraction', 'f_scrolls_fraction']].values
        dm = xgb.DMatrix(X)

        y_pred = model.predict(dm)
        df_batch['prediction'] = y_pred

        for _, row in df_batch[['session_long', 'prediction']].iterrows():
            yield (row.session_long, float(row.prediction))


spark = SparkSession\
        .builder\
        .appName("spark test") \
        .getOrCreate()


df = spark.read.parquet('s3://spark-workshop-data/data-sessions/')


df = df \
    .withColumn('f_views_fraction', df.f_view_sessions / df.f_sessions) \
    .withColumn('f_replies_fraction', df.f_reply_sessions / df.f_sessions) \
    .withColumn('f_scrolls_fraction', df.f_scroll_sessions / df.f_sessions) \
    .select('session_long', 'f_views_fraction', 'f_replies_fraction', 'f_scrolls_fraction')

columns = df.columns


output_schema =  types.StructType([
    types.StructField("session_long", types.StringType()),
    types.StructField("predictions", types.FloatType()),
])


df_output = df.rdd \
    .mapPartitions(lambda p: apply_model(columns, model, p)) \
    .toDF(output_schema)


df_output.write.mode('overwrite').parquet('s3://spark-workshop-data/output/2020-10-09/')