-- Databricks notebook source
USE ajmal_demo_catalog.seek_pii;

-- COMMAND ----------

SELECT * FROM bronze_cv_data

-- COMMAND ----------

-- DBTITLE 1,Create all of the tables (silver and gold)
CREATE OR REPLACE TABLE silver_data_anonymised
AS SELECT file_paths, anonymized FROM bronze_cv_data;

CREATE OR REPLACE TABLE gold_data_anonymised
AS SELECT anonymized FROM silver_data_anonymised;

CREATE OR REPLACE TABLE silver_all_pii
AS SELECT file_paths, cv_content, emails FROM bronze_cv_data;

CREATE OR REPLACE TABLE gold_data_pii
AS SELECT cv_content, emails FROM silver_all_pii;

CREATE OR REPLACE TABLE gold_emails
AS SELECT emails FROM silver_all_pii;

-- COMMAND ----------

COMMENT ON TABLE gold_emails IS 'This table contains just the emails that we extracted from the CVs. This contains PII.';
COMMENT ON TABLE gold_data_pii IS 'This table contains all of the information (except raw file paths) from cvs. This contains PII.';
COMMENT ON TABLE silver_all_pii IS 'This silver table has paths but has inference results from the model and contains PII.';
COMMENT ON TABLE silver_data_anonymised IS 'This table is fully anonymised but contains file paths.';
COMMENT ON TABLE gold_data_anonymised IS 'This table is fully anonymised and contains no paths.';

-- COMMAND ----------

ALTER TABLE gold_data_anonymised ALTER COLUMN anonymized COMMENT "This column is an anonymised version of the CV that was uploaded.";
ALTER TABLE gold_emails ALTER COLUMN emails COMMENT "This column contains all of the emails that we extracted from the CVs.";
ALTER TABLE gold_data_pii ALTER COLUMN cv_content COMMENT "This column has cv content including all PII information.";

-- COMMAND ----------


