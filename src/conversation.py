class Conversation:
    def __init__(self):
        self.history = []

    def add_turn(self, question, answer):
        self.history.append({"q": question, "a": answer})

    def get_context(self, last_n=3):
        return " ".join([f"Q: {h['q']} A: {h['a']}" for h in self.history[-last_n:]])
