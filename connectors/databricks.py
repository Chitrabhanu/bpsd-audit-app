from pyspark.sql import functions as F
from .base import BaseConnector

class DatabricksConnector(BaseConnector):

    def __init__(self, config, spark):
        self.spark = spark
        self.note_table = config["data"]["note_embedding_table"]
        self.symptom_table = config["data"]["symptom_embedding_table"]

    def get_symptom_categories(self):
        return (
            self.spark.table(self.symptom_table)
            .select("Category")
            .distinct()
            .orderBy("Category")
            .toPandas()["Category"]
            .tolist()
        )

    def fetch_note_units(self, note_id):
        return (
            self.spark.table(self.note_table)
            .filter(F.col("ClinicalNoteKey").cast("string") == str(note_id))
            .select(
                "UnitRowId",
                "ClinicalNoteKey",
                "EncounterKey",
                "PatientDurableKey",
                "Section",
                "UnitId",
                "UnitText",
                "UnitTextEmbedding",
                "UnitTextEmbeddingNorm"
            )
            .orderBy("UnitId")
            .toPandas()
        )

    def fetch_symptom_terms(self, symptom_key):
        return (
            self.spark.table(self.symptom_table)
            .filter(F.col("Category") == symptom_key)
            .select(
                "Category",
                "TermIdx",
                "QueryText",
                "QueryTextEmbedding",
                "QueryTextEmbeddingNorm"
            )
            .orderBy("TermIdx")
            .toPandas()
        )