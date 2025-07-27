import os

from google import genai
import vertexai

from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
from vertexai import rag

class VertexAIRetrieval:

    def __init__(self):
        self.model_id = os.getenv("model_id")
        self.client = None
        
    def create_client(self):
        PROJECT_ID = os.getenv("PROJECT_ID")
        LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

        vertexai.init(project=PROJECT_ID, location=LOCATION)
        self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    def create_corporus(self):
        EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"

        rag_corpus = rag.create_corpus(
            display_name="my-rag-corpus",
            backend_config=rag.RagVectorDbConfig(
                rag_embedding_model_config=rag.RagEmbeddingModelConfig(
                    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                        publisher_model=EMBEDDING_MODEL
                    )
                )
            ),
        )

        self.rag_corpus = rag_corpus
        print(rag.list_corpora())

        INPUT_GCS_BUCKET = "swarm-invoices"
        response = rag.import_files(
            corpus_name=rag_corpus.name,
            paths=[f"gs://{INPUT_GCS_BUCKET}/"],
            transformation_config=rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(chunk_size=1024, chunk_overlap=100)
            ),
            max_embedding_requests_per_min=900,
        )
