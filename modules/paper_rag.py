import logging
import sqlite3
import faiss
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

os.environ['ALL_PROXY'] = 'http://127.0.0.1:7890'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetrieval:
    def __init__(self, db_path, faiss_index_path):
        self.db_path = db_path
        self.faiss_index_path = faiss_index_path
        self.model = None
        self.tokenizer = None
        self.index = None
        self.abstracts = []
        self.embeddings = []

        # Set device to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        try:
            # Load the tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct")
            self.model = AutoModel.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct")
            self.model.to(self.device)  # Move model to GPU or CPU based on availability
            logger.info("Alibaba-NLP/gte-Qwen2-7B-instruct model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_abstracts_from_db(self, table_name):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"SELECT abstracts FROM {table_name}")  # Use the passed table name
            rows = cursor.fetchall()
            conn.close()
            self.abstracts = [row[0] for row in rows]
            logger.info(f"Loaded {len(self.abstracts)} abstracts from the table '{table_name}'.")
        except Exception as e:
            logger.error(f"Error loading abstracts from database: {e}")
            raise

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def compute_embeddings(self):
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Model or tokenizer is not loaded.")

            # Tokenize the abstracts
            inputs = self.tokenizer(self.abstracts, padding=True, truncation=True, return_tensors="pt").to(self.device)

            # Get the embeddings from the model
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

            self.embeddings = embeddings.cpu().numpy()  # Convert embeddings to numpy array
            logger.info(f"Embeddings for {len(self.abstracts)} abstracts computed successfully.")
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            raise

    def create_faiss_index(self):
        try:
            if len(self.embeddings) == 0:
                raise ValueError("No embeddings to index.")

            if os.path.exists(self.faiss_index_path):
                # If FAISS index already exists, load it
                self.index = faiss.read_index(self.faiss_index_path)
                logger.info("Loaded existing FAISS index.")
            else:
                faiss_embeddings = self.embeddings.astype('float32')
                self.index = faiss.IndexFlatL2(faiss_embeddings.shape[1])  # L2 distance
                self.index.add(faiss_embeddings)
                logger.info("Created new FAISS index and added embeddings.")
        except Exception as e:
            logger.error(f"Error creating or loading FAISS index: {e}")
            raise

    def persist_faiss_index(self):
        try:
            faiss.write_index(self.index, self.faiss_index_path)
            logger.info(f"FAISS index persisted to {self.faiss_index_path}.")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise

    def query_faiss(self, query, top_k=3):
        try:
            if not self.index:
                raise ValueError("FAISS index is not created.")

            query_embedding = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                query_embedding = self.model(**query_embedding).last_hidden_state.mean(dim=1).cpu().numpy().astype(
                    'float32')

            distances, indices = self.index.search(query_embedding, top_k)
            return distances, indices
        except Exception as e:
            logger.error(f"Error in querying FAISS index: {e}")
            raise

    def get_most_similar_papers(self, query, top_k=3):
        try:
            distances, indices = self.query_faiss(query, top_k)
            similar_papers = [self.abstracts[idx] for idx in indices[0]]
            return similar_papers
        except Exception as e:
            logger.error(f"Error retrieving most similar papers: {e}")
            raise


# 8. Main program
def main():
    db_path = "../data/papers.db"
    faiss_index_path = "faiss_index.index"  # FAISS index file path
    query = "brain disorder identification"  # Example query

    try:
        # Create RAGRetrieval instance
        rag_retrieval = RAGRetrieval(db_path, faiss_index_path)

        # Load model and data
        rag_retrieval.load_model()
        rag_retrieval.load_abstracts_from_db("IJCAI_2024")

        # Compute embeddings and create FAISS index
        rag_retrieval.compute_embeddings()
        rag_retrieval.create_faiss_index()

        # Persist FAISS index to a file
        rag_retrieval.persist_faiss_index()

        # Get most similar papers
        similar_papers = rag_retrieval.get_most_similar_papers(query, top_k=3)

        # Print most similar papers
        logger.info(f"Most similar papers to the query '{query}':")
        for paper in similar_papers:
            logger.info(f"- {paper}")

    except Exception as e:
        logger.error(f"Error in the main processing: {e}")


if __name__ == "__main__":
    main()
