from vector_store import VectorStore

class Retriever:
    def __init__(self):
        self.store = VectorStore()

    def index_corpus(self, contexts):
        self.store.build(contexts)

    def retrieve(self, question, top_k=3):
        return self.store.search(question, top_k)
