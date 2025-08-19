from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.vectors = None

    def build(self, docs):
        self.documents = docs
        self.vectors = self.vectorizer.fit_transform(docs)

    def add_documents(self, new_docs):
        self.documents.extend(new_docs)
        self.vectors = self.vectorizer.fit_transform(self.documents)

    def search(self, query, top_k=3):
        q_vec = self.vectorizer.transform([query])
        scores = np.dot(self.vectors, q_vec.T).toarray().squeeze()
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in top_indices]
