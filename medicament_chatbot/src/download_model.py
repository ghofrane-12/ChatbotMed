from transformers import AutoModelForSequenceClassification,AutoTokenizer
from medicament_chatbot.src.configg import CONFIG

# Chemin pour sauvegarder le modèle pré-entraîné
model_path = "medicament_chatbot/models/fine_tuned_model"

# Télécharger et sauvegarder le modèle XLM-RoBERTa pré-entraîné
tokenizer = AutoTokenizer.from_pretrained(CONFIG["fine_tuned_model"])
model = AutoModelForSequenceClassification.from_pretrained(CONFIG["fine_tuned_model"])

# Sauvegarder le modèle et le tokenizer dans le dossier
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)

print(f"Modèle XLM-RoBERTa sauvegardé dans {model_path}")
