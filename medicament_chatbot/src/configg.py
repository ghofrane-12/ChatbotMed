import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    # Détection de langue
    "language_detection_model": "medicament_chatbot/models/fine_tuned_model",
    "fine_tuned_model": "medicament_chatbot/models/fine_tuned_model",

    # Modèle génératif Gemini
    "gemini_model": "gemini-2.5-flash",   # ou gemini-2.0-pro
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),

    # Embeddings pour RAG
    "embedding_model": "intfloat/multilingual-e5-large",

    # Base de connaissances
    "knowledge_base_path": "medicament_chatbot/data/medications_6000_multilang.csv",
    "vectorstore_path": "medicament_chatbot/data/vectorstore",

    # Langues supportées
    "supported_languages": ["fr", "en", "ar"],

    # Paramètres génération
    "max_tokens": 512,
    "temperature": 0.3,
    "similarity_threshold": 0.32,
    "top_k": 10,

    "PROMPT_TEMPLATES": {
        "fr": """
Historique de la conversation :
{chat_history}

Documents pertinents (traduits si nécessaire) :
{context}

Question de l'utilisateur :
{question}

Instructions :
- Si des documents pertinents sont disponibles ci-dessus, utilise exclusivement leur contenu pour répondre en {language}.
- Si aucun document pertinent n'est trouvé mais que la question est médicale, donne une réponse générale en {language}.
- ⚠️ Si la question n'est pas liée aux médicaments, réponds clairement : "Votre question ne semble pas liée aux médicaments. Je ne peux pas y répondre."
- Même si les documents sont en anglais, ta réponse doit être en français.
- Sois professionnel et courtois.
- Ne donne pas de conseils médicaux personnalisés, reste général.

Réponse :
""",
        "en": """
Conversation history:
{chat_history}

Relevant documents:
{context}

User question:
{question}

Instructions:
- If relevant documents are available above, use ONLY their content to answer in {language}.
- If no relevant documents are found but the question is medical, provide a general answer in {language}.
- ⚠️ If the question is not related to medicines, clearly state: "Your question does not seem related to medicines. I cannot answer it."
- Be professional and polite.
- Do not provide personalized medical advice, only general information.

Response:
""",
        "ar": """
سجل المحادثة:
{chat_history}

المستندات ذات الصلة (مترجمة إذا لزم الأمر):
{context}

سؤال المستخدم:
{question}

التعليمات:
- إذا كانت هناك مستندات ذات صلة أعلاه، استخدم محتواها فقط للإجابة باللغة {language}.
- إذا لم يتم العثور على مستندات ذات صلة ولكن السؤال طبي، قدم إجابة عامة باللغة {language}.
- ⚠️ إذا لم يكن السؤال متعلقًا بالأدوية، فقل بوضوح: "سؤالك لا يبدو متعلقًا بمجال الأدوية. لا أستطيع الإجابة عليه."
- حتى لو كانت المستندات باللغة الإنجليزية، يجب أن تكون إجابتك بالعربية.
- كن محترفًا ومهذبًا.
- لا تقدم نصائح طبية شخصية، فقط معلومات عامة.

الإجابة:
"""
    }
}
