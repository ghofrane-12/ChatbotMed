from medicament_chatbot.src.anomaly_detector import AnomalyDetector


def load_training_questions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions

# Charger les questions depuis un fichier
training_texts = load_training_questions("medicament_chatbot/data/anomalie.txt")

# Entraîner et sauvegarder le modèle
detector = AnomalyDetector(
    training_texts,nu=0.01,
    model_path="medicament_chatbot/models/anomaly_detector.pkl"
)

print("✅ Modèle d'anomalie entraîné et sauvegardé dans medicament_chatbot/models/anomaly_detector.pkl")
X_train = detector.vectorizer.transform(training_texts)
preds = detector.model.predict(X_train)
print("Inliers sur train:", (preds == 1).mean())
scores = detector.model.decision_function(X_train).ravel()
print("threshold =", detector.threshold)
print("Inliers sur train (threshold):", (scores >= detector.threshold).mean())