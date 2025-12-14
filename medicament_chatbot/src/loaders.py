from langchain_text_splitters import RecursiveCharacterTextSplitter
from medicament_chatbot.src.configg import CONFIG
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import pandas as pd
from langchain_core.documents import Document
class DocumentManager:
    def __init__(self):
        # Initialisation des embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])
        # Text splitter (optionnel, mais utile si certains champs sont longs)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def load_documents(self):
        # Charger le CSV avec pandas
        df = pd.read_csv(CONFIG["knowledge_base_path"])
        
        # Construire une colonne "text" riche à partir des champs utiles
        df["text"] = (
            "Nom générique: " + df["generic_name"].fillna("") +
            " | Forme: " + df["form"].fillna("") +
            " | Dosage: " + df["dosage"].fillna("") +
            " | Disponible: " + df["available"].fillna("") +
            " | Type de vente: " + df["sale_type"].fillna("") +
            " | Prix (TND): " + df["price_tnd"].astype(str)
        ).str.strip()

        # Convertir en documents LangChain
        raw_docs = []
        for _, row in df.iterrows():
            raw_docs.append({
                "page_content": row["text"],
                "metadata": {"lang": row.get("lang", "en")}
            })

        # Découper si nécessaire (mais souvent une ligne suffit)
        documents = []
        for _, row in df.iterrows():
            lang = row["lang"] if "lang" in df.columns else "en"
            # Découpage si nécessaire
            chunks = self.text_splitter.split_text(row["text"])
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={"lang": lang}
                ))
        return documents
    def setup_vectorstore(self, rebuild=False):
        if os.path.exists(CONFIG["vectorstore_path"]) and not rebuild:
            print("Rechargement du vectorstore existant...")
            vectorstore = Chroma(
                persist_directory=CONFIG["vectorstore_path"],
                embedding_function=self.embeddings
            )
        else:
            print("Construction du vectorstore...")
            documents = self.load_documents()
            vectorstore = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=CONFIG["vectorstore_path"]
            )
            vectorstore.persist()
        return vectorstore
