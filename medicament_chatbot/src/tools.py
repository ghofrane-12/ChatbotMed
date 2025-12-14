# Cette section définit des outils personnalisés pour le chatbot de médicaments.
# Ils travaillent avec les données CSV contenant les informations sur les médicaments.

from langchain_core.tools import Tool
from typing import List, Dict
import csv

class MedicamentTools:
    def __init__(self, medicament_path: str):
        # Charger les médicaments à partir d'un fichier CSV.
        # Le CSV contient les colonnes :
        # id, generic_name, form, dosage, available, sale_type, price_tnd
        self.medicaments: List[Dict] = []
        with open(medicament_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.medicaments = list(reader)

    def search_medicaments(self, query: str, lang: str = "en") -> List[Dict]:
        """
        Recherche des médicaments basés sur une requête utilisateur.
        Recherche souple sur generic_name, form, dosage et sale_type.
        """
        matches: List[Dict] = []
        q = query.lower()

        for med in self.medicaments:
            if med.get("lang", "en") != lang:
                continue

            text_parts = [
                str(med.get("generic_name", "")),
                str(med.get("form", "")),
                str(med.get("dosage", "")),
                str(med.get("sale_type", "")),
            ]
            text = " ".join(text_parts).lower()

            # ✅ recherche souple
            if q in text or any(term in text for term in q.split()):
                matches.append(med)

        return matches

    def check_availability(self, generic_name: str, lang: str = "en") -> bool:
        """
        Vérifie si un médicament est disponible en fonction de son nom générique et de la langue.
        """
        gname = generic_name.lower()
        for med in self.medicaments:
            if med.get("lang", "en") != lang:
                continue
            if med.get("generic_name", "").lower() == gname:
                val = str(med.get("available", "0")).strip().lower()
                return val in ["1", "true", "yes", "oui"]
        return False


    def get_tools(self) -> List[Tool]:
        # Retourner les outils sous forme de liste pour les Agents LangChain
        return [
            Tool(
                name="SearchMedicaments",
                func=self.search_medicaments,
                description=(
                    "Recherche des médicaments dans la base CSV "
                    "(nom générique, forme, dosage, type de vente). "
                    "Entrée : texte libre décrivant le médicament."
                ),
            ),
            Tool(
                name="CheckMedicamentAvailability",
                func=self.check_availability,
                description=(
                    "Vérifie si un médicament est disponible en pharmacie "
                    "en fonction de son nom générique. Entrée : nom du médicament."
                ),
            ),
        ]