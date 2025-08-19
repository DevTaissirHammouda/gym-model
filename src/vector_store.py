from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.model = SentenceTransformer(model_name, device=device)
        self.documents = []       # list of contexts
        self.embeddings = None    # torch.Tensor

    def build(self, docs):
        """Encode all documents and store embeddings"""
        self.documents = docs
        self.embeddings = self.model.encode(docs, convert_to_tensor=True, show_progress_bar=True)

    def add_documents(self, new_docs):
        """Add new documents and update embeddings"""
        new_embeddings = self.model.encode(new_docs, convert_to_tensor=True, show_progress_bar=True)
        self.documents.extend(new_docs)
        self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)

    def search(self, query, top_k=3):
        """Return top_k most similar documents for a query"""
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        results = [(self.documents[idx], float(score)) for score, idx in zip(top_results.values, top_results.indices)]
        return results
