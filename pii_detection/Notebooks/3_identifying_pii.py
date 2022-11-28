# Databricks notebook source
import os
os.system("spacy download en_core_web_lg") # Add this into init script

# COMMAND ----------

import mlflow
model_name = "🤗_pii_detector"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# COMMAND ----------

# DBTITLE 1,Let's look at an example again of the PII detector in action
import pprint 
import pandas as pd
 
pprint.pprint(model.predict(pd.DataFrame(["Ajmal"])))

# COMMAND ----------

# _ = spark.sql("USE ajmal_demo_catalog.seek_pii;")
_ = spark.sql("USE seek_pii;")
bronze_cv_data = table("bronze_cv_data")

# COMMAND ----------

display(bronze_cv_data)

# COMMAND ----------

# Create the PySpark UDF
import mlflow.pyfunc
from pyspark.sql.functions import struct

pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/production", result_type="string")
predicted_df = bronze_cv_data.withColumn("result", pyfunc_udf(struct('cv_content')))
display(predicted_df)

# COMMAND ----------

from pyspark.sql.functions import col, from_json, expr
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, DoubleType

schema = StructType([
  StructField("found", ArrayType(StructType([
    StructField("entity_type", StringType(), False),
    StructField("start", IntegerType(), False),
    StructField("end", IntegerType(), False),
    StructField("score", DoubleType(), False),
    StructField("analysis_explanation", StringType(), False),
    StructField("recognition_metadata", StructType([
      StructField("recognizer_name", StringType(), False),
      StructField("recognizer_identifier", StringType(), False)
    ]), True)
  ]), True), True),
  StructField("anonymized", StringType(), False)
])

parsed_df = predicted_df.withColumn("entity_json", from_json(col("result"), schema))
parsed_df = parsed_df.select("file_paths", "cv_content", "result", expr("entity_json.found as entities"), "entity_json.anonymized")
display(parsed_df)

# COMMAND ----------

display(parsed_df)

# COMMAND ----------

parsed_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("bronze_cv_data")
