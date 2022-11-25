# Databricks notebook source
import mlflow
model_name = "🤗_pii_detector"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# COMMAND ----------

# DBTITLE 1,Let's look at an example again of the PII detector in action
import pprint 
pprint.pprint(model.predict("Ajmal"))

# COMMAND ----------

# _ = spark.sql("USE ajmal_demo_catalog.seek_pii;")
_ = spark.sql("USE seek_pii;")
bronze_cv_data = table("bronze_cv_data")

# COMMAND ----------

display(bronze_cv_data)

# COMMAND ----------

def generate_predictions(df):
  df = df.toPandas()
  inference_results = df["cv_content"].apply(lambda x: model.predict(x))
  df.loc[:, "anonymized"] = [results["anonymized"] for results in inference_results]
  emails = []
  for id, result in enumerate(inference_results):
    start_index = [i["start"] for i in result["found"] if i["entity_type"] == "EMAIL_ADDRESS"][0]
    end_index = [i["end"] for i in result["found"] if i["entity_type"] == "EMAIL_ADDRESS"][0]
    emails = emails + [df.loc[id, "cv_content"][start_index:end_index]]
  df.loc[:, "emails"] = emails
  return df
bronze_cv_data = spark.createDataFrame(generate_predictions(df=bronze_cv_data))

# COMMAND ----------

display(bronze_cv_data)

# COMMAND ----------

bronze_cv_data.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("bronze_cv_data")

# COMMAND ----------


