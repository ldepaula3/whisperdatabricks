# Databricks notebook source
# MAGIC %pip install --upgrade transformers accelerate protobuf==3.19.5

# COMMAND ----------

# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

import shutil
from transformers import pipeline, WhisperTokenizerFast
import mlflow
from mlflow.models import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import torch
import pandas as pd


# COMMAND ----------

pipeline_task = "automatic-speech-recognition"
pipeline_artifact_name = 'pipeline'

# dbfs disk storage used when constructing the MLflow model. 
pipeline_output_dir = 'dbfs:/FileStore/whisper/model'

# COMMAND ----------

shutil.rmtree(pipeline_output_dir, ignore_errors=True)

# COMMAND ----------

# download and save model
tokenizer = WhisperTokenizerFast.from_pretrained('openai/whisper-tiny')
pipe = pipeline("automatic-speech-recognition", tokenizer=tokenizer, model='openai/whisper-tiny', chunk_length_s=30)
pipe.save_pretrained(pipeline_output_dir)

# COMMAND ----------


class SpeechRecognitionPipelineModel(mlflow.pyfunc.PythonModel):
  signature = ModelSignature(inputs=Schema([ColSpec(name="audio", type=DataType.binary)]),
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

pymodel = SpeechRecognitionPipelineModel()

with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=pymodel,
        artifacts={pipeline_artifact_name: pipeline_output_dir},
        signature=pymodel.signature,
        pip_requirements=["torch", "transformers", "accelerate"]
    )
