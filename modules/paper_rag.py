import logging
import sqlite3
import faiss
import os
import numpy as np
import torch
import torch.nn.functional as F
import config.settings as settings
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetrieval:
    def __init__(self, db_path, faiss_index_path, search_weight=None):
        self.db_path = db_path
        self.faiss_index_path = faiss_index_path
        self.model = None
        self.tokenizer = None
        self.index = None
        self.abstracts = []
        self.keywords = []
        self.titles = []
        self.embeddings = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self._init_retrieval_weights(search_weight)

    def _init_retrieval_weights(self, search_weight):
        """ Initialize the weights for title, abstract, and keywords for retrieval. """
        try:
            temp_weight = {}
            if isinstance(search_weight, dict):
                temp_weight = search_weight
            else:
                if hasattr(settings, 'SEARCH_WEIGHT'):
                    if isinstance(settings.SEARCH_WEIGHT, dict):
                        temp_weight = settings.SEARCH_WEIGHT
                    else:
                        raise ValueError(
                            "SEARCH_WEIGHT should be a dictionary with a format such as: {'title': 0.15, 'abstract': 0.35, 'keywords': 0.5}")
            sum_weights = sum(temp_weight.values())
            if sum_weights != 1:
                for key in temp_weight:
                    temp_weight[key] = temp_weight[key] / sum_weights
            self.search_weight = temp_weight
            logger.info(f"Retrieval weights initialized: {self.search_weight}")
        except Exception as e:
            logger.error(f"Error initializing retrieval weights: {e}")
            raise

    def load_model(self):
        try:
            # Load the tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained("/ai/teacher/mwt/code/by/models/GTE-1.5B",
                                                           trust_remote_code=True)
            self.model = AutoModel.from_pretrained("/ai/teacher/mwt/code/by/models/GTE-1.5B", trust_remote_code=True)
            self.model.to(self.device)  # Move model to GPU or CPU based on availability
            logger.info("Alibaba-NLP/gte-Qwen2-1.5B-instruct model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_abstracts_from_db(self, table_name):
        try:
            self._load_data_from_db(table_name, "abstracts")
        except Exception as e:
            logger.error(f"Error loading abstracts from database: {e}")
            raise

    def load_titles_from_db(self, table_name):
        try:
            self._load_data_from_db(table_name, "titles")
        except Exception as e:
            logger.error(f"Error loading titles from database: {e}")

    def load_keywords_from_db(self, table_name):
        try:
            self._load_data_from_db(table_name, "keywords")
        except Exception as e:
            logger.error(f"Error loading keywords from database: {e}")
            raise

    def _load_data_from_db(self, table_name, column_name):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"SELECT {column_name} FROM {table_name}")
            rows = cursor.fetchall()
            conn.close()
            self.titles = [row[0] for row in rows]
            logger.info(f"Loaded {len(self.titles)} titles from the table '{table_name}'.")
        except Exception as e:
            logger.error(f"Error loading titles from database: {e}")

    def last_token_pool(self, last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def compute_embeddings(self):
        """
        Compute embeddings for the given column data.
        :param column_name: Column name for which embeddings need to be computed. Choose from 'abstracts', 'titles', or 'keywords'.
        :return:            None
        """
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Model or tokenizer is not loaded.")

            # Tokenize the data
            data_title = self.titles
            data_abstract = self.abstracts
            data_keywords = self.keywords

            input_title = self.tokenizer(data_title, return_tensors="pt", padding=True, truncation=True).to(self.device)
            input_abstract = self.tokenizer(data_abstract, return_tensors="pt", padding=True, truncation=True).to(
                self.device)
            input_keywords = self.tokenizer(data_keywords, return_tensors="pt", padding=True, truncation=True).to(
                self.device)

            # Get the embeddings from the model
            with torch.no_grad():
                output1 = self.model(**inputs)
                embeddings = self._last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

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

    def get_most_similar_papers(self, query, top_k=3, weight_keywords=0.7, weight_abstracts=0.3):
        try:
            distances, indices = self.query_faiss(query, top_k, )

            similar_papers = [self.abstracts[idx] for idx in indices[0]]
            return similar_papers
        except Exception as e:
            logger.error(f"Error retrieving most similar papers: {e}")
            raise


# 8. Main program
def main():
    db_path = "../data/papers.db"
    faiss_index_path = "faiss_index.index"
    query = "The paper related to image-text retrieval."

    try:
        # Create RAGRetrieval instance
        rag_retrieval = RAGRetrieval(db_path, faiss_index_path)

        # Load model and data
        rag_retrieval.load_model()
        rag_retrieval.load_abstracts_from_db("IJCAI_2024")
        rag_retrieval.load_keywords_from_db("IJCAI_2024")
        rag_retrieval.load_titles_from_db("IJCAI_2024")

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
