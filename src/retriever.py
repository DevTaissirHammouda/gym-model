from vector_store import VectorStore

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None, emb_path="data/embeddings.pt", doc_path="data/documents.pt"):
        self.store = VectorStore(model_name=model_name, device=device)
        self.emb_path = emb_path
        self.doc_path = doc_path

    def index_corpus(self, contexts):
        """Build the semantic search index"""
        self.store.build(contexts)

    def retrieve(self, question, top_k=3):
        """Retrieve top_k most relevant contexts for the question"""
        return self.store.search(question, top_k)

    def save(self):
        """Save embeddings + documents using predefined paths"""
        self.store.save(self.emb_path, self.doc_path)

    def load(self):
        """Load embeddings + documents using predefined paths"""
        self.store.load(self.emb_path, self.doc_path)
