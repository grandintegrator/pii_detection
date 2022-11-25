# Databricks notebook source
# MAGIC %sql
# MAGIC USE ajmal_demo_catalog.seek_pii;

# COMMAND ----------

import pandas as pd

filepaths = [f.path for f in dbutils.fs.ls("dbfs:/FileStore/ajmal_aziz/cv_examples/")]
cv_df = spark.createDataFrame(pd.DataFrame({'file_paths': filepaths}))

# COMMAND ----------

# MAGIC %md
# MAGIC This function just extracts text from uploaded PDF files. These PDF files sit in S3 and contain the CVs that we want to analyse.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col
import pandas as pd
from PyPDF2 import PdfFileReader
from io import BytesIO
from pyspark.sql import functions as F
from pyspark.sql.window import Window

@pandas_udf("string")
def extract_text_from_pdf(s: pd.Series) -> pd.Series:
    #Scan the folder having the pdf files
    def path_to_text(path):
      path = "/" + path.replace(":", "")
      pdf = PdfFileReader(path, strict=False)  
      text = [pdf.getPage(i).extractText() for i in range(0, pdf.getNumPages())]
      return "\n".join(text)
    return s.apply(path_to_text)

# COMMAND ----------

cv_df = cv_df.withColumn("cv_content", extract_text_from_pdf(col("file_paths")))
cv_df.createOrReplaceTempView("cv_df")
display(cv_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE bronze_cv_data AS
# MAGIC SELECT * FROM cv_df;

# COMMAND ----------

# MAGIC %sql
# MAGIC COMMENT ON TABLE bronze_cv_data IS 'This table contains paths to CVs and a raw parse of the CV text which includes PII data.';

# COMMAND ----------


