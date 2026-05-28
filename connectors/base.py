from abc import ABC, abstractmethod

class BaseConnector(ABC):

    @abstractmethod
    def get_symptom_categories(self):
        pass

    @abstractmethod
    def fetch_note_units(self, note_id):
        pass

    @abstractmethod
    def fetch_symptom_terms(self, symptom_key):
        pass