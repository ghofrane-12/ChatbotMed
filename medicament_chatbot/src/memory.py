# Cette section gère la mémoire conversationnelle du chatbot.
# Elle enregistre les interactions passées pour conserver le contexte.

class ChatMemoryManager:
    def __init__(self):
        # On utilise une simple liste pour stocker les échanges
        self.chat_history = []

    def add_to_memory(self, user_input: str, response: str):
        """
        Sauvegarde une interaction utilisateur-assistant dans la mémoire.
        """
        self.chat_history.append({
            "user": user_input,
            "assistant": response
        })

    def get_chat_history(self):
        """
        Retourne l'historique des conversations enregistrées.
        """
        return self.chat_history

    def clear_memory(self):
        """
        Efface tout le contexte enregistré.
        """
        self.chat_history = []


# Exemple d'utilisation
if __name__ == "__main__":
    memory = ChatMemoryManager()


    # Charger et afficher l'historique des conversations
    print("Historique des conversations :")
    print(memory.get_chat_history())

    # Effacer la mémoire
    memory.clear_memory()
    print("Mémoire après effacement :", memory.get_chat_history())
