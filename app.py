import streamlit as st
from medicament_chatbot.src.agent import MedicamentAgent
from medicament_chatbot.src.configg import CONFIG
from transformers import AutoTokenizer

# Charger le tokenizer pour la troncature
tokenizer = AutoTokenizer.from_pretrained(CONFIG["language_detection_model"])
MAX_LENGTH = 512  # Limite fixe pour les textes


def truncate_text(text: str, max_length: int = MAX_LENGTH) -> str:
    """Tronque le texte en respectant la limite de tokens."""
    encoded = tokenizer.encode(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return tokenizer.decode(encoded[0], skip_special_tokens=True)


def main():
    st.set_page_config(page_title="Assistant Virtuel - Médicaments", layout="wide")

    st.title("Assistant Virtuel - Médicaments")
    st.markdown(
        """
        Bienvenue ! Je suis votre assistant virtuel pour les questions sur les médicaments.
        Je peux vous aider (information générale) en français, anglais ou arabe.
        ⚠️ Je ne remplace pas un avis médical professionnel.
        """
    )

    # Initialisation de l'agent
    if "agent" not in st.session_state:
        with st.spinner("Initialisation de l'assistant..."):
            try:
                st.session_state.agent = MedicamentAgent()
                st.session_state.error_count = 0
            except Exception as e:
                st.error(f"Erreur d'initialisation : {str(e)}")
                return

    # Historique des messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage des messages précédents
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Entrée utilisateur
    if user_input := st.chat_input("Quelle est votre question sur les médicaments ?"):
        processed_input = truncate_text(user_input)

        st.session_state.messages.append({"role": "user", "content": processed_input})
        with st.chat_message("user"):
            st.write(processed_input)

        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("Recherche de la meilleure réponse..."):
                try:
                    response = st.session_state.agent.ask(processed_input)

                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.error_count = 0

                except Exception as e:
                    st.session_state.error_count += 1
                    error_msg = str(e)
                    if st.session_state.error_count >= 3:
                        response = (
                            "Je rencontre des difficultés techniques. "
                            "Veuillez réessayer plus tard ou contacter le support."
                        )
                    else:
                        response = (
                            "Je suis désolé, je n'ai pas bien compris. "
                            "Pouvez-vous reformuler votre question ?"
                        )

                    st.error(f"Erreur: {error_msg}")
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
