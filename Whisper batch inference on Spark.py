# Databricks notebook source
# MAGIC %md
# MAGIC # Whisper batch inference on Spark
# MAGIC This example notebook shows wrapping whisper as a MLflow model for use with batch inference on Spark, where you pass in a path to the model for inference. This approach allows processing of large audio files in bulk.
# MAGIC
# MAGIC ## Cluster requirements
# MAGIC MLR >= 13.1, GPU

# COMMAND ----------

# MAGIC %pip install --upgrade transformers accelerate protobuf==3.19.5

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

# Hugging Face model name
source_model_name = 'openai/whisper-medium'
# MLflow model registry model name to register the model to
registered_model_name = 'whisper_file'

# dbfs disk storage used when constructing the MLflow model. 
model_path = '/mlflow_models/whisper_file'
pipeline_output_dir = '/local_disk0/model_artifacts/whisper-file-pipeline'

# dbfs path to download a sample audio file
audio_file_path = '/dbfs/samples/van-morrison-interview.mp3'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download a large audio file as a sample
# MAGIC Credit: Joe Smith Collection at the Library of Congress, Motion Picture, Broadcasting and Recorded Sound Division <br>
# MAGIC [Off the record interview with Van Morrison, 1988-01-12](https://www.loc.gov/item/jsmith000239/)

# COMMAND ----------

import urllib.request
import shutil
import os
shutil.rmtree(audio_file_path, ignore_errors=True)
os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
os.system(f'wget https://tile.loc.gov/storage-services/service/mbrs/mbrsjoesmith/1836263/1836263.mp3 -O {audio_file_path}')

# COMMAND ----------

shutil.rmtree(f'/dbfs{model_path}', ignore_errors=True)
shutil.rmtree(pipeline_output_dir, ignore_errors=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Wrap whisper as a MLflow model and register it to Model Registry

# COMMAND ----------

pipeline_task = "automatic-speech-recognition"
pipeline_artifact_name = 'pipeline'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Construct Hugging Face transformers pipeline and save it to local disk

# COMMAND ----------

from transformers import pipeline, WhisperTokenizerFast
tokenizer = WhisperTokenizerFast.from_pretrained(source_model_name)
pipe = pipeline(pipeline_task, tokenizer=tokenizer, model=source_model_name, chunk_length_s=30)
pipe.save_pretrained(pipeline_output_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a custom model class for MLflow
# MAGIC We will soon be adding direct support for Whisper models in the `transformers` flavor within MLflow. When it is available, you will not need to create a custom model class to save your Whisper models in MLflow.

# COMMAND ----------

import mlflow
from mlflow.models import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import torch
import pandas as pd

class SpeechRecognitionPipelineModel(mlflow.pyfunc.PythonModel):
  signature = ModelSignature(inputs=Schema([ColSpec(name="audio_path", type=DataType.string)]),
                             outputs=Schema([ColSpec(name="transcription", type=DataType.string)]))

  def __init__(self):
    self.pipeline = None

  def load_context(self, context):
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = WhisperTokenizerFast.from_pretrained(context.artifacts[pipeline_artifact_name])
    self.pipeline = pipeline(pipeline_task, context.artifacts[pipeline_artifact_name],
                             chunk_length_s=30, tokenizer=tokenizer, device=device)

  def predict(self, context, model_input):
    audio = model_input[model_input.columns[0]].to_list()
    transcriptions = [prediction['text'] for prediction in self.pipeline(audio)]
    return pd.Series(transcriptions, name='transcription')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Save as a MLflow model to DBFS

# COMMAND ----------

pymodel = SpeechRecognitionPipelineModel()
mlflow.pyfunc.save_model(artifacts={pipeline_artifact_name: pipeline_output_dir},
                         path=f'/dbfs{model_path}', python_model=pymodel,
                         signature=pymodel.signature)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Model Registry

# COMMAND ----------

model_version = mlflow.register_model(f'dbfs:{model_path}', registered_model_name, await_registration_for=3600)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Use model for batch inference

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model as a Spark UDF

# COMMAND ----------

model_uri = f'models:/{model_version.name}/{model_version.version}'
whisper_udf = mlflow.pyfunc.spark_udf(spark, model_uri, "string")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Construct Spark DataFrame with path to audio file

# COMMAND ----------

df = spark.createDataFrame(pd.DataFrame(pd.Series([audio_file_path], name="path")))
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transcribe audio using the UDF
# MAGIC This step can take several minutes.

# COMMAND ----------

transcribed_df = df.select(df.path, whisper_udf(df.path).alias('transcription')).cache()
display(transcribed_df)
