import pandas as pd
import ast
from .base import BaseConnector


class LocalFileConnector(BaseConnector):

    def __init__(self, config):

        self.note_path = config["data"]["note_embeddings_path"]
        self.symptom_path = config["data"]["symptom_embeddings_path"]

        self.note_df = pd.read_csv(self.note_path)
        self.symptom_df = pd.read_csv(self.symptom_path)

        # --------------------------------------------------
        # Convert embedding strings back into Python lists
        # --------------------------------------------------

        self.note_df["UnitTextEmbedding"] = (
            self.note_df["UnitTextEmbedding"]
            .apply(self._parse_embedding)
        )

        self.symptom_df["QueryTextEmbedding"] = (
            self.symptom_df["QueryTextEmbedding"]
            .apply(self._parse_embedding)
        )

    def _parse_embedding(self, value):

        # Already parsed
        if isinstance(value, list):
            return value

        # Nulls
        if pd.isna(value):
            return []

        # CSV stores arrays as strings like:
        # "[0.123, 0.456, ...]"
        try:
            return ast.literal_eval(value)

        except Exception:
            return []

    def get_symptom_categories(self):

        return sorted(
            self.symptom_df["Category"]
            .dropna()
            .unique()
            .tolist()
        )

    def fetch_note_units(self, note_id):

        df = self.note_df[
            self.note_df["ClinicalNoteKey"].astype(str) == str(note_id)
        ].copy()

        return df.sort_values("UnitId")

    def fetch_symptom_terms(self, symptom_key):

        df = self.symptom_df[
            self.symptom_df["Category"] == symptom_key
        ].copy()

        return df.sort_values("TermIdx")