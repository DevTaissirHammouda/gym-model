from vector_store import VectorStore

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.store = VectorStore(model_name=model_name, device=device)

    def index_corpus(self, contexts):
        """Build the semantic search index"""
        self.store.build(contexts)

    def retrieve(self, question, top_k=3):
        """Retrieve top_k most relevant contexts for the question"""
        return self.store.search(question, top_k)
