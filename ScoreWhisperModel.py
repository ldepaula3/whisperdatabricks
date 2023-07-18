# Databricks notebook source
import base64
import json
import requests

# COMMAND ----------

# Load Data

df = spark.read.format('binaryFile').load("dbfs:/FileStore/whisper/audio")

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We need to decode it with UTF-8 so it becomes a STRING. We can't serialize a binary format and pass it over JSON.

# COMMAND ----------

df_pd = df.toPandas()

df_pd['content_utf'] = df_pd['content'].apply(lambda x: base64.b64encode(x).decode('utf-8'))

# COMMAND ----------

df_pd

# COMMAND ----------

# MAGIC %md
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/score-model-serving-endpoints
# MAGIC #### We can pass parameters in two ways:
# MAGIC <br>
# MAGIC
# MAGIC 1) dataframe_records - this will pass a array of column:row - ex: {
# MAGIC      "dataframe_records": [
# MAGIC      {
# MAGIC         "audio": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4LjEyLjEw"
# MAGIC       }
# MAGIC      ]
# MAGIC    }
# MAGIC
# MAGIC
# MAGIC 2) dataframe_split - this will pass two rows - column (names) and rows (records) - ex: 
# MAGIC {
# MAGIC     "dataframe_split": [{
# MAGIC       "index": [0, 1],
# MAGIC       "columns": ["audio"],
# MAGIC       "data": [["SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4LjEyLjEw"]]
# MAGIC     }]
# MAGIC   }

# COMMAND ----------

def score_model(audio):
  url = 'https://[YOUR_ENVIRONMENT].databricks.com/serving-endpoints/[YOUR_MODEL_NAME]/invocations'
  headers = {'Authorization': f'Bearer [YOUR_TOKEN]', 'Content-Type': 'application/json'}

  payload = json.dumps({"dataframe_records": [{"audio": audio}]})

  response = requests.request("POST", url, headers=headers, data=payload)

  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  return(response.text)

# COMMAND ----------

firstaudio = df_pd["content_utf"][0]

transcription = score_model(firstaudio)

print(transcription)
