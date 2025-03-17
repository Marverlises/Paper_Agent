import logging
import faiss
import os
import numpy as np
import torch
import config.settings as settings
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from paper_sql import PaperSQL

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetrieval:
    def __init__(self, db_path, faiss_index_path, search_weight=None, paper_sql: PaperSQL = None):
        self.db_path = db_path
        self.faiss_index_path = faiss_index_path
        self.model = None
        self.tokenizer = None

        self.abstracts = []
        self.keywords = []
        self.titles = []
        self.abstract_index = None
        self.keywords_index = None
        self.title_index = None
        self.weighted_index = None

        self.embeddings = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self._init_retrieval_weights(search_weight)
        self.paper_sql = paper_sql if paper_sql else PaperSQL(db_path)

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
            if abs(sum_weights - 1) <= 1e-5:
                for key in temp_weight:
                    temp_weight[key] = temp_weight[key] / sum_weights
            self.search_weight = temp_weight
            logger.info(f"Retrieval weights initialized: {self.search_weight}")
        except Exception as e:
            logger.error(f"Error initializing retrieval weights: {e}")
            raise

    def load_model(self, model_path=None, tokenizer_path=None):
        try:
            # Load the tokenizer and model
            if not tokenizer_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self.model.to(self.device)
            logger.info(f"{model_path} model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_info_from_db(self, table_name):
        try:
            self.titles = self.paper_sql.load_column_from_db(table_name, "title")
            self.abstracts = self.paper_sql.load_column_from_db(table_name, "abstracts")
            self.keywords = self.paper_sql.load_column_from_db(table_name, "keywords")
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise e

    def compute_embeddings(self):
        """
        Compute embeddings for the given column data.
        :return:            None
        """

        def last_token_pool(last_hidden_states: Tensor,
                            attention_mask: Tensor) -> Tensor:
            """ Pooling the embeddings from the last token of the sequence. """
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

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
                output_title = self.model(**input_title)
                output_abstract = self.model(**input_abstract)
                output_keywords = self.model(**input_keywords)
                embeddings_title = last_token_pool(output_title.last_hidden_state, input_title['attention_mask'])
                embeddings_abstract = last_token_pool(output_abstract.last_hidden_state,
                                                      input_abstract['attention_mask'])
                embeddings_keywords = last_token_pool(output_keywords.last_hidden_state,
                                                      input_keywords['attention_mask'])
            # normalize embeddings
            embeddings_title = torch.nn.functional.normalize(embeddings_title, p=2, dim=1)
            embeddings_abstract = torch.nn.functional.normalize(embeddings_abstract, p=2, dim=1)
            embeddings_keywords = torch.nn.functional.normalize(embeddings_keywords, p=2, dim=1)
            self.embeddings['title'] = embeddings_title.cpu().numpy()
            self.embeddings['abstract'] = embeddings_abstract.cpu().numpy()
            self.embeddings['keywords'] = embeddings_keywords.cpu().numpy()
            self.embeddings['weighted_embeddings'] = self.embeddings['title'] * self.search_weight['title'] + \
                                                     self.embeddings['abstract'] * self.search_weight['abstracts'] + \
                                                     self.embeddings['keywords'] * self.search_weight['keywords']
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            raise

    def create_faiss_index(self):
        try:
            if len(self.embeddings) == 0:
                raise ValueError("No embeddings to index.")
            # compute embeddings
            title_embeddings = self.embeddings['title'].astype('float32')
            abstract_embeddings = self.embeddings['abstract'].astype('float32')
            keywords_embeddings = self.embeddings['keywords'].astype('float32')

            self.title_index = faiss.IndexFlatL2(title_embeddings.shape[1])  # Title index
            self.abstract_index = faiss.IndexFlatL2(abstract_embeddings.shape[1])  # Abstract index
            self.keywords_index = faiss.IndexFlatL2(keywords_embeddings.shape[1])  # Keywords index
            self.weighted_index = faiss.IndexFlatL2(self.embeddings['weighted_embeddings'].shape[1])  # Weighted index
            self.title_index.add(title_embeddings)
            self.abstract_index.add(abstract_embeddings)
            self.keywords_index.add(keywords_embeddings)
            self.weighted_index.add(self.embeddings['weighted_embeddings'])

            logger.info("Created new FAISS index and added embeddings.")
        except Exception as e:
            logger.error(f"Error creating or loading FAISS index: {e}")
            raise

    def persist_faiss_index(self):
        try:
            faiss.write_index(self.title_index, self.faiss_index_path + "_title")
            faiss.write_index(self.abstract_index, self.faiss_index_path + "_abstract")
            faiss.write_index(self.keywords_index, self.faiss_index_path + "_keywords")
            faiss.write_index(self.weighted_index, self.faiss_index_path + "_weighted")

            logger.info(f"FAISS index persisted to {self.faiss_index_path}.")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise

    def query_faiss(self, query, top_k=3, use_column='weighted'):
        try:
            if not self.title_index or not self.abstract_index or not self.keywords_index:
                raise ValueError("FAISS index is not created.")

            query_vector = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                query_vector = self.model(**query_vector)
                query_vector = query_vector.last_hidden_state.mean(dim=1).cpu().numpy()

            if use_column == 'title':
                title_distances, title_indices = self.title_index.search(query_vector, top_k)
                return title_distances, title_indices
            elif use_column == 'abstract':
                abstract_distances, abstract_indices = self.abstract_index.search(query_vector, top_k)
                return abstract_distances, abstract_indices
            elif use_column == 'keywords':
                keywords_distances, keywords_indices = self.keywords_index.search(query_vector, top_k)
                return keywords_distances, keywords_indices
            elif use_column == 'weighted':
                weighted_distances, weighted_indices = self.weighted_index.search(query_vector, top_k)
                return weighted_distances, weighted_indices
            else:
                raise ValueError("Invalid column name for querying FAISS index.")
        except Exception as e:
            logger.error(f"Error in querying FAISS index: {e}")
            raise

    def get_most_similar_papers(self, query, top_k=3, method='weighted'):
        """
        Get the most similar papers to the given query.
        :param query:   Query string
        :param top_k:   Number of similar papers to retrieve
        :param method:  Column to use for retrieval result
        :return:
        """
        try:
            distances, indices = self.query_faiss(query, top_k, use_column=method)

            similar_papers_info = {}
            for i in range(top_k):
                similar_papers_info[i] = {
                    'title': self.titles[indices[0][i]],
                    'abstract': self.abstracts[indices[0][i]],
                    'keywords': self.keywords[indices[0][i]],
                    'distance': distances[0][i]
                }

            return similar_papers_info
        except Exception as e:
            logger.error(f"Error retrieving most similar papers: {e}")
            raise


# 8. Main program
def main():
    db_path = "../data/papers.db"
    faiss_index_path = "../data/faiss_index.index"
    key_words = 'Image-text matching or image-text retrieval'
    query = f'Given some text, retrieve relevant passages that related to the "{key_words}" task.'

    try:
        # Create RAGRetrieval instance
        rag_retrieval = RAGRetrieval(db_path, faiss_index_path)

        # Load model and data
        rag_retrieval.load_model(settings.RAG_MODEL_PATH)
        rag_retrieval.load_info_from_db("IJCAI_2024")

        # Compute embeddings and create FAISS index
        rag_retrieval.compute_embeddings()
        rag_retrieval.create_faiss_index()

        # Persist FAISS index to a file
        rag_retrieval.persist_faiss_index()

        # Get most similar papers
        similar_papers = rag_retrieval.get_most_similar_papers(query, top_k=10, method='weighted')

        # Print most similar papers
        logger.info(f"Most similar papers to the query '{query}':")
        for i, paper in similar_papers.items():
            logger.info(f"Paper {i + 1}:")
            logger.info(f"Title: {paper['title']}")
            logger.info(f"Abstract: {paper['abstract']}")
            logger.info(f"Keywords: {paper['keywords']}")
            logger.info(f"Distance: {paper['distance']}")

    except Exception as e:
        logger.error(f"Error in the main processing: {e}")


if __name__ == "__main__":
    main()
