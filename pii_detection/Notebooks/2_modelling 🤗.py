# Databricks notebook source
# MAGIC %sql
# MAGIC -- USE ajmal_demo_catalog.seek_pii;
# MAGIC USE seek_pii;

# COMMAND ----------

from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine
from typing import List

from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts
from transformers import pipeline

import os
os.system("spacy download en_core_web_lg")

# COMMAND ----------

# list of entities: https://microsoft.github.io/presidio/supported_entities/#list-of-supported-entities
PII_ENTITIES = [
    "CREDIT_CARD",
    "DATE_TIME",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "IP_ADDRESS",
    "NRP",
    "LOCATION",
    "PERSON",
    "PHONE_NUMBER",
    "MEDICAL_LICENSE",
    "URL",
    "ORGANIZATION"
]

# COMMAND ----------

# DBTITLE 1,Build and define our customer 🤗 model for NER and PII detection
import mlflow

class HFTransformersPIIDetector(mlflow.pyfunc.PythonModel):
  """
  Class to train and use FastText Models
  """
  def __init__(self):
    self.engine = AnonymizerEngine()

  def load_context(self, context):
    self.engine = AnonymizerEngine()

  class HFTransformersRecognizer(EntityRecognizer):
    def __init__(
        self,
        model_id_or_path=None,
        aggregation_strategy="simple",
        supported_language="en",
        ignore_labels=["O", "MISC"],
    ):
        # inits transformers pipeline for given mode or path
        self.pipeline = pipeline(
            "token-classification", model=model_id_or_path, aggregation_strategy=aggregation_strategy, ignore_labels=ignore_labels
        )
        # map labels to presidio labels
        self.label2presidio = {
            "PER": "PERSON",
            "LOC": "LOCATION",
            "ORG": "ORGANIZATION",
        }

        # passes entities from model into parent class
        super().__init__(supported_entities=list(self.label2presidio.values()), supported_language=supported_language)

    def load(self) -> None:
        """No loading is required."""
        pass

    def analyze(
        self, text: str, entities: List[str] = None, nlp_artifacts: NlpArtifacts = None
    ) -> List[RecognizerResult]:
        """
        Extracts entities using Transformers pipeline
        """
        results = []

        # keep max sequence length in mind
        predicted_entities = self.pipeline(text)
        if len(predicted_entities) > 0:
            for e in predicted_entities:
                converted_entity = self.label2presidio[e["entity_group"]]
                if converted_entity in entities or entities is None:
                    results.append(
                        RecognizerResult(
                            entity_type=converted_entity, start=e["start"], end=e["end"], score=e["score"]
                        )
                    )
        return results

  def model_fn(self, model_dir):
    transformers_recognizer = self.HFTransformersRecognizer(model_dir)
    # Set up the engine, loads the NLP module (spaCy model by default) and other PII recognizers
    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(transformers_recognizer)
    return analyzer

  def predict(self, context, model_input):
    analyzer = self.model_fn("Jean-Baptiste/roberta-large-ner-english")

    # identify entities
    results = analyzer.analyze(text=model_input, entities=PII_ENTITIES, language="en")

    # anonymize text
    anonymised = self.engine.anonymize(text=model_input, analyzer_results=results)
    
    return {"found": [entity.to_dict() for entity in results], "anonymized": anonymised.text}

# COMMAND ----------

# MAGIC %md
# MAGIC Let's go through an example 

# COMMAND ----------

mlflow_model = HFTransformersPIIDetector()

# COMMAND ----------

example_payload = """
Hello, my name is David Johnson and I live in Maine.
I work as a software engineer at Amazon.
You can call me at (123) 456-7890.
My credit card number is 4095-2609-9393-4932.

On September 18 I visited microsoft.com and sent an email to test@presidio.site, from the IP 192.168.0.1.
My passport: 191280342 and my phone number: (212) 555-1234.
This is a valid International Bank Account Number: IL150120690000003111111. Can you please check the status on bank account 954567876544?
Kate's social security number is 078-05-1126.  Her driver license? it is 1234567A.

"""

mlflow_model.predict(None, example_payload)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try our model against single name entities to test robustness. This is key because there can be certain columns that just contain names for example and we'd like to recognise those too.

# COMMAND ----------

single_name_payload = "Ajmal Aziz"
mlflow_model.predict(None, single_name_payload)

# COMMAND ----------

single_name_payload = "Graham"
mlflow_model.predict(None, single_name_payload)

# COMMAND ----------

single_name_payload = "Thet Ko is awesome."
mlflow_model.predict(None, single_name_payload)

# COMMAND ----------

# MAGIC %md
# MAGIC We can now log this model into MLFlow

# COMMAND ----------

from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle

with mlflow.start_run(run_name='HuggingFacePIIDetector'):
  mlflow_model = HFTransformersPIIDetector()
  mlflow.pyfunc.log_model("HuggingFacePIIDetectorModel", python_model=mlflow_model)

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='status="FINISHED"').iloc[0].run_id
model_name = "🤗_pii_detector"
model_version = mlflow.register_model(f"runs:/{run_id}/HuggingFacePIIDetectorModel", model_name)

# COMMAND ----------

import time
time.sleep(15)

# COMMAND ----------

from mlflow.tracking import MlflowClient
 
client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------


