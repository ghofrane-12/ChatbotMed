from transformers import AutoTokenizer, AutoModelForSequenceClassification, M2M100ForConditionalGeneration, M2M100Tokenizer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import os
import time
import re
from medicament_chatbot.src.anomaly_detector import AnomalyDetector
from medicament_chatbot.src.configg import CONFIG
from medicament_chatbot.src.memory import ChatMemoryManager


class MedicamentAgent:
    def __init__(self):
        # Mémoire conversationnelle
        self.memory_manager = ChatMemoryManager()
        self.last_call_time = 0
        self.min_interval = 2

        # Anomaly detector
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, "models", "anomaly_detector.pkl")
        self.anomaly_detector = AnomalyDetector.load(model_path)

        # Détection de langue (corrigé: fix_mistral_regex=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["language_detection_model"], fix_mistral_regex=True
        )
        self.lang_model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG["language_detection_model"]
        )

        # Traduction M2M100
        self.m2m_model_name = "facebook/m2m100_418M"
        self.m2m_tokenizer = M2M100Tokenizer.from_pretrained(self.m2m_model_name)
        self.m2m_model = M2M100ForConditionalGeneration.from_pretrained(self.m2m_model_name)

        # Embeddings + Chroma
        self.embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])
        self.vectorstore = Chroma(
            persist_directory=CONFIG["vectorstore_path"],
            embedding_function=self.embeddings
        )

        # Mots-clés médicaux multilingues (FR/EN/AR)
        self.medical_keywords = set([
            # FR/EN
            "médicament","medicament","drug","medicine","ordonnance","prescription",
            "sans ordonnance","otc","dosage","dose","mg","ml","effets secondaires",
            "side effects","antibiotique","antibiotics","complément","vitamine","vitamin",
            "paracetamol","acetaminophen","ibuprofen","omeprazole","amoxicilline","amoxicillin",
            "insuline","insulin","bromhexine","losartan","zinc",
            # AR
            "دواء","أدوية","جرعة","وصفة","بدون وصفة","الجرعة","آثار جانبية",
            "مضاد حيوي","مضادات حيوية","باراسيتامول","اسيتامينوفين","ايبوبروفين",
            "اوميبرازول","اموكسيسيلين","انسولين","زنك"
        ])

    # ---------------------------
    # Langue
    # ---------------------------
    def detect_language(self, text: str) -> str:
        if any("\u0600" <= ch <= "\u06FF" for ch in text):
            return "ar"
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.lang_model(**inputs)
        lang = CONFIG["supported_languages"][outputs.logits.argmax().item()]
        if any(ch in "àâäçéèêëîïôöùûüÿ" for ch in text.lower()) and lang == "en":
            return "fr"
        return lang

    # ---------------------------
    # Normalisation arabe
    # ---------------------------
    @staticmethod
    def normalize_arabic(text: str) -> str:
        if not text:
            return ""
        t = str(text)
        t = re.sub(r"[\u064B-\u065F]", "", t)   # harakat
        t = re.sub(r"[إأآا]", "ا", t)           # alif
        t = re.sub(r"[ى]", "ي", t)             # ya
        t = re.sub(r"ـ", "", t)                # tatweel
        return t.strip()

    # ---------------------------
    # Traduction M2M100
    # ---------------------------
    def translate_response(self, text, target_lang, src_lang=None):
        lang_map = {"fr": "fr", "en": "en", "ar": "ar"}
        target = lang_map.get(target_lang, "en")

        if src_lang is None:
            src_lang = self.detect_language(text)
        src = lang_map.get(src_lang, "en")

        self.m2m_tokenizer.src_lang = src
        encoded = self.m2m_tokenizer(text, return_tensors="pt", truncation=True)
        generated_tokens = self.m2m_model.generate(
            **encoded,
            forced_bos_token_id=self.m2m_tokenizer.get_lang_id(target),
            max_new_tokens=512,
        )
        return self.m2m_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    # ---------------------------
    # Domaine
    # ---------------------------
    def is_out_of_scope(self, text: str) -> bool:
        """True si hors domaine (non médical)."""
        # Heuristique mots-clés: si mot médical détecté → in-domain
        if any(kw in text.lower() for kw in self.medical_keywords):
            return False
        # Sinon, on fait confiance à l'anomaly detector
        return not self.anomaly_detector.is_in_domain(text)

    # ---------------------------
    # RAG (E5 query prefix + normalisation AR)
    # ---------------------------
    def retrieve_documents(self, query: str, target_lang: str, src_lang: str | None = None):
        if src_lang is None:
            src_lang = self.detect_language(query)

        q = query
        if target_lang == "ar":
            q = self.normalize_arabic(q)

        # ✅ E5 requires query: prefix
        q_pref = "query: " + q

        # 1) Recherche dans la langue demandée
        results = self.vectorstore.similarity_search(q_pref, k=8, filter={"lang": target_lang})

        # 2) Fallback AR → FR → EN si rien trouvé
        if not results and target_lang == "ar":
            q_fr = "query: " + self.translate_response(query, "fr", src_lang=src_lang)
            results = self.vectorstore.similarity_search(q_fr, k=8, filter={"lang": "fr"})
            if not results:
                q_en = "query: " + self.translate_response(query, "en", src_lang=src_lang)
                results = self.vectorstore.similarity_search(q_en, k=8, filter={"lang": "en"})

        # 3) Fallback global sans filtre
        if not results:
            results = self.vectorstore.similarity_search(q_pref, k=CONFIG.get("top_k", 10))

        return results

    # ---------------------------
    # Prompt
    # ---------------------------
    def build_prompt(self, chat_history, context, question, language, mode="grounded"):
        template = CONFIG["PROMPT_TEMPLATES"][language]
        lang_instruction = {
            "ar": "أجب باللغة العربية فقط.",
            "fr": "Réponds uniquement en français.",
            "en": "Answer only in English.",
        }.get(language, "Answer only in English.")

        if context.strip():
            instructions = "- Utilise exclusivement les informations des documents pertinents ci-dessus."
        else:
            instructions = "- Aucun document pertinent trouvé, tu peux donner une réponse générale."

        return (
            template.format(
                chat_history=chat_history or "Aucune conversation précédente.",
                context=context or "Aucun document pertinent trouvé.",
                question=question,
                language=language,
            )
            + "\n"
            + instructions
            + "\n"
            + lang_instruction
        )

    # ---------------------------
    # Appel Gemini
    # ---------------------------
    def call_gemini(self, prompt):
        api_url = (
            "https://generativelanguage.googleapis.com/v1/models/"
            f"{CONFIG['gemini_model']}:generateContent?key={CONFIG['gemini_api_key']}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": CONFIG.get("temperature", 0.3),
                "maxOutputTokens": CONFIG.get("max_tokens", 1200),
            }
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return data["candidates"][0].get("output_text", "Aucune réponse générée par Gemini.")

    # ---------------------------
    # Pipeline principal
    # ---------------------------
    def ask(self, user_message):
        lang = self.detect_language(user_message)
        print("Langue détectée:", lang)

        # Quota simple
        now = time.time()
        if now - self.last_call_time < self.min_interval:
            time.sleep(self.min_interval - (now - self.last_call_time))
        self.last_call_time = time.time()

    # 1️⃣ Vérifier domaine AVANT RAG
        in_domain = self.anomaly_detector.is_in_domain(user_message)

        # heuristique mots-clés médicaux
        if any(kw in user_message.lower() for kw in self.medical_keywords):
            in_domain = True

        if not in_domain:
            # ✅ Blocage strict : on ne construit pas de prompt Gemini
            return {
                "ar": "سؤالك لا يبدو متعلقًا بمجال الأدوية. لا أستطيع الإجابة عليه.",
                "fr": "Votre question ne semble pas liée aux médicaments. Je ne peux pas y répondre.",
                "en": "Your question does not seem related to medicines. I cannot answer it.",
            }.get(lang, "Your question does not seem related to medicines. I cannot answer it.")

        # 2️⃣ RAG
        docs = self.retrieve_documents(user_message, target_lang=lang, src_lang=lang)
        context = "\n".join([doc.page_content for doc in docs]) if docs else ""

        # Historique
        history = self.memory_manager.get_chat_history()
        formatted_history = "\n".join(
            [f"User: {h['user']} | Assistant: {h.get('assistant','')}" for h in history]
        ) if history else "Aucune conversation précédente."

        # Prompt
        prompt = self.build_prompt(
            chat_history=formatted_history,
            context=context,
            question=user_message,
            language=lang,
            mode="grounded"
        )

        try:
            answer = self.call_gemini(prompt)
            # Harmoniser la langue de sortie
            ans_lang = self.detect_language(answer)
            if ans_lang != lang:
                answer = self.translate_response(answer, lang, src_lang=ans_lang)
        except Exception as e:
            return f"Erreur Gemini: {e}"

        self.memory_manager.add_to_memory(user_message, answer)
        return answer
