import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
import joblib
import os

class AnomalyDetector:
    """
    Détecteur d'anomalies pour filtrer les questions hors domaine.
    Utilise OCSVM avec TF-IDF.
    """
    def __init__(self, training_texts=None, kernel='linear', nu=0.05, gamma='scale', model_path=None):
        self.vectorizer = None
        self.model = None
        self.model_path = model_path
        self.threshold = 0.0

        if training_texts:
            print(f"Entraînement du modèle avec {len(training_texts)} exemples...")
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1,2),
                sublinear_tf=True,
                min_df=2,
                max_df=0.95
            )
            X = self.vectorizer.fit_transform(training_texts)
            self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
            self.model.fit(X)

            train_scores = self.model.decision_function(X).ravel()
            self.threshold = np.percentile(train_scores, 10)  # garde 95% comme "in-domain"

            if model_path:
                self.save(model_path)
                print(f"Modèle sauvegardé dans {model_path}")

    def save(self, model_path=None):
        path = model_path or self.model_path
        if not path:
            raise ValueError("Aucun chemin de sauvegarde fourni.")
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model,
            'threshold': self.threshold
        }, path)

    @classmethod
    def load(cls, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fichier introuvable: {model_path}")
        data = joblib.load(model_path)
        instance = cls.__new__(cls)
        instance.vectorizer = data['vectorizer']
        instance.model = data['model']
        instance.model_path = model_path
        instance.threshold = data.get("threshold", 0.0)
        print(f"Modèle chargé depuis {model_path}")
        return instance

    def is_in_domain(self, text: str) -> bool:
        """Retourne True si le texte est considéré comme in-domain"""
        X_test = self.vectorizer.transform([text])
        score = float(self.model.decision_function(X_test)[0])
        return score >= self.threshold

    def predict_batch(self, texts):
        """Retourne une liste de booléens pour plusieurs textes"""
        X_test = self.vectorizer.transform(texts)
        preds = self.model.predict(X_test)
        return [p == 1 for p in preds]
