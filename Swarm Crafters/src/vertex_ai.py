# import os

# from google import genai
# import vertexai

# from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
# from vertexai import rag

# class VertexAIRetrieval:

#     def __init__(self):
#         self.model_id = os.getenv("model_id")
#         self.client = None


#     def create_client(self):
#         PROJECT_ID = os.getenv("PROJECT_ID") # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}
#         # if not PROJECT_ID or PROJECT_ID == PROJECT_ID:
#         #     PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

#         LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

#         vertexai.init(project=PROJECT_ID, location=LOCATION)
#         self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
#         return self.client


#     def create_corporus(self):
#         EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"  # @param {type:"string", isTemplate: true}

#         rag_corpus = rag.create_corpus(
#             display_name="my-rag-corpus",
#             backend_config=rag.RagVectorDbConfig(
#                 rag_embedding_model_config=rag.RagEmbeddingModelConfig(
#                     vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
#                         publisher_model=EMBEDDING_MODEL
#                     )
#                 )
#             ),
#         )

#         self.rag_corpus = rag_corpus
#         print(rag.list_corpora())



#         INPUT_GCS_BUCKET = "swarm-invoices"

#         response = rag.import_files(
#             corpus_name=rag_corpus.name,
#             paths=[INPUT_GCS_BUCKET],
#             # Optional
#             transformation_config=rag.TransformationConfig(
#                 chunking_config=rag.ChunkingConfig(chunk_size=1024, chunk_overlap=100)
#             ),
#             max_embedding_requests_per_min=900,  # Optional
#         )
     
#     def retriever(self):
#         response = rag.retrieval_query(
#         rag_resources=[
#             rag.RagResource(
#                 rag_corpus=self.rag_corpus.name,
#                 # Optional: supply IDs from `rag.list_files()`.
#                 # rag_file_ids=["rag-file-1", "rag-file-2", ...],
#             )
#         ],
#         rag_retrieval_config=rag.RagRetrievalConfig(
#             top_k=10,  # Optional
#             filter=rag.Filter(
#                 vector_distance_threshold=0.5,  # Optional
#             ),
#         ),
#         text=self.query,
#         )
#         return response


#     def execute_task(self,query):
#         self.client = self.create_client()
#         self.create_corporus()
#         response = self.retriever()
#         # response = self.client.models.generate_content(
#         #     model=self.model_id,
#         #     contents=query,
#         #     config=GenerateContentConfig(tools=[rag_retrieval_tool]),
#         # )

#         return response


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
        LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-west4")

        vertexai.init(project=PROJECT_ID, location=LOCATION)
        self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

        return 

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

    def retriever(self,query):
        response = rag.retrieval_query(
        rag_resources=[
            rag.RagResource(
                rag_corpus=self.rag_corpus.name,
                # Optional: supply IDs from `rag.list_files()`.
                # rag_file_ids=["rag-file-1", "rag-file-2", ...],
            )
        ],
        rag_retrieval_config=rag.RagRetrievalConfig(
            top_k=10,  # Optional
            filter=rag.Filter(
                vector_distance_threshold=0.5,  # Optional
            ),
        ),
        text=query,
        )
        return response

    def execute_task(self,query):
        self.create_client()
        self.create_corporus()
        response = self.retriever(query)
        print("response :     ",response)
        return response
        # response = self.client.models.generate_content(
        #     model=self.model_id,
        #     contents=query,
        #     config=GenerateContentConfig(tools=[rag_retrieval_tool]),
        # )

        # return response
# import os

# from google import genai
# import vertexai

# # No longer need Tool, Retrieval, VertexRagStore from genai.types as we're not using it as a tool
# from vertexai import rag

# class VertexAIRetrieval:

#     def __init__(self):
#         self.model_id = os.getenv("model_id")
#         self.client = None
#         self.rag_corpus = None

#     def create_client(self):
#         PROJECT_ID = os.getenv("PROJECT_ID")
#         LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-west4")

#         vertexai.init(project=PROJECT_ID, location=LOCATION)
#         self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
#         return

#     def create_corporus(self):
#         EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"

#         print("Creating RAG corpus...")
#         rag_corpus = rag.create_corpus(
#             display_name="my-rag-corpus",
#             backend_config=rag.RagVectorDbConfig(
#                 rag_embedding_model_config=rag.RagEmbeddingModelConfig(
#                     vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
#                         publisher_model=EMBEDDING_MODEL
#                     )
#                 )
#             ),
#         )

#         self.rag_corpus = rag_corpus
#         print(f"RAG Corpus created: {rag_corpus.name}")
#         print("Listing corpora:")
#         for corpus in rag.list_corpora():
#             print(f"- {corpus.name} (Display Name: {corpus.display_name})")

#         INPUT_GCS_BUCKET = "swarm-invoices"
#         print(f"Importing files from gs://{INPUT_GCS_BUCKET}/ into corpus {rag_corpus.name}...")
#         response = rag.import_files(
#             corpus_name=rag_corpus.name,
#             paths=[f"gs://{INPUT_GCS_BUCKET}/"],
#             transformation_config=rag.TransformationConfig(
#                 chunking_config=rag.ChunkingConfig(chunk_size=1024, chunk_overlap=100)
#             ),
#             max_embedding_requests_per_min=900,
#         )
#         # print(f"File import initiated. Operation name: {response.name}")
#         # In a production environment, you would typically poll the operation here
#         # to ensure the import is complete before attempting retrieval.
#         # For simplicity, we'll proceed, but be aware of timing.


#     def retrieve_documents(self, query: str):
#         """
#         Performs the RAG retrieval directly and returns the retrieved content.
#         """
#         if not self.rag_corpus:
#             raise ValueError("RAG Corpus has not been created. Call create_corporus() first.")

#         print(f"Performing direct retrieval for query: {query}")
#         retrieval_results = rag.retrieval(
#             retrieval_queries=[
#                 rag.RetrievalQuery(
#                     query=query,
#                     vertex_rag_store=rag.VertexRagStore(
#                         rag_corpora=[self.rag_corpus.name],
#                         similarity_top_k=10,
#                         vector_distance_threshold=0.5,
#                     )
#                 )
#             ]
#         )

#         retrieved_texts = []
#         if retrieval_results and retrieval_results.retrieval_results:
#             for result in retrieval_results.retrieval_results:
#                 for chunk in result.chunks:
#                     retrieved_texts.append(chunk.content)
#                     if chunk.source:
#                         print(f"  Retrieved from source: {chunk.source.url if hasattr(chunk.source, 'url') else chunk.source.document_id}")
#         return retrieved_texts

#     def execute_task(self, query: str):
#         self.create_client()
#         if not self.rag_corpus:
#             self.create_corporus()

#         # Step 1: Perform retrieval directly
#         retrieved_content_list = self.retrieve_documents(query)

#         if not retrieved_content_list:
#             print("No relevant content found through retrieval. Generating response without additional context.")
#             # If no content is retrieved, you might choose to still send the query
#             # or return an empty response based on your application's logic.
#             response = self.client.models.generate_content(
#                 model=self.model_id,
#                 contents=query,
#             )
#             return response.text if response.text else "No relevant information found and model did not generate a direct answer."


#         # Step 2: Construct the prompt with retrieved content
#         context = "\n".join(retrieved_content_list)
#         return context
#         # full_prompt = f"Given the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

#         # print("\n--- Sending request to model with retrieved context ---")
#         # # Step 3: Send the augmented prompt to the generative model
#         # response = self.client.models.generate_content(
#         #     model=self.model_id,
#         #     contents=full_prompt,
#         #     # No tools are passed here, as retrieval is handled manually
#         # )

#         # return response.text # Return the model's generated answer
# import os

# from google import genai
# import vertexai

# from vertexai import rag
# # from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore # No longer needed for manual RAG

# class VertexAIRetrieval:

#     def __init__(self):
#         self.model_id = os.getenv("model_id")
#         self.client = None
#         self.rag_corpus = None

#     def create_client(self):
#         PROJECT_ID = os.getenv("PROJECT_ID")
#         LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-west4")

#         vertexai.init(project=PROJECT_ID, location=LOCATION)
#         self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
#         return

#     def create_corporus(self):
#         EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"

#         print("Creating RAG corpus...")
#         # Check if corpus already exists to avoid recreating
#         # In a real application, you'd list corpora and try to find it by display_name or name
#         # For simplicity, if self.rag_corpus is None, we create a new one.
#         # This part might need more robust handling for existing corpora in a persistent app.
#         if self.rag_corpus:
#             print(f"Corpus '{self.rag_corpus.name}' already exists. Skipping creation.")
#             return

#         rag_corpus = rag.create_corpus(
#             display_name="my-rag-corpus", # Consider making this dynamic or checking for existing one
#             backend_config=rag.RagVectorDbConfig(
#                 rag_embedding_model_config=rag.RagEmbeddingModelConfig(
#                     vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
#                         publisher_model=EMBEDDING_MODEL
#                     )
#                 )
#             ),
#         )

#         self.rag_corpus = rag_corpus
#         print(f"RAG Corpus created: {rag_corpus.name}")
#         print("Listing corpora:")
#         for corpus in rag.list_corpora():
#             print(f"- {corpus.name} (Display Name: {corpus.display_name})")

#         INPUT_GCS_BUCKET = "swarm-invoices"
#         print(f"Importing files from gs://{INPUT_GCS_BUCKET}/ into corpus {rag_corpus.name}...")
#         response = rag.import_files(
#             corpus_name=rag_corpus.name,
#             paths=[f"gs://{INPUT_GCS_BUCKET}/"],
#             transformation_config=rag.TransformationConfig(
#                 chunking_config=rag.ChunkingConfig(chunk_size=1024, chunk_overlap=100)
#             ),
#             max_embedding_requests_per_min=900,
#         )
#         # print(f"File import initiated. Operation name: {response.name}")
#         # In a production environment, you would typically poll the operation here
#         # to ensure the import is complete before attempting retrieval.
#         # For simplicity, we'll proceed, but be aware of timing.


#     def retrieve_documents(self, query: str):
#         """
#         Performs the RAG retrieval directly using rag.query_corpus and returns the retrieved content.
#         """
#         if not self.rag_corpus:
#             # Attempt to list existing corpora and set self.rag_corpus if 'my-rag-corpus' exists
#             print("RAG Corpus not set. Attempting to find existing corpus...")
#             found_corpus = None
#             for corpus in rag.list_corpora():
#                 if corpus.display_name == "my-rag-corpus": # Or match by a known ID
#                     found_corpus = corpus
#                     break
#             if found_corpus:
#                 self.rag_corpus = found_corpus
#                 print(f"Found existing corpus: {self.rag_corpus.name}")
#             else:
#                 raise ValueError("RAG Corpus has not been created and no existing 'my-rag-corpus' found. Call create_corporus() first.")


#         print(f"Performing direct retrieval for query: {query} from corpus {self.rag_corpus.name}")

#         # The direct retrieval method is typically rag.query_corpus or similar
#         # and it returns a QueryCorpusResponse which contains the chunks.
#         query_results = rag.query_corpus(
#             name=self.rag_corpus.name,
#             query=query,
#             top_k=10, # Corresponds to similarity_top_k
#             # If there's a vector_distance_threshold equivalent, it would be here
#         )

#         retrieved_texts = []
#         if query_results and query_results.retrieved_contents:
#             for content in query_results.retrieved_contents:
#                 retrieved_texts.append(content.content)
#                 if content.source:
#                     print(f"  Retrieved from source: {content.source.url if hasattr(content.source, 'url') else content.source.document_id}")
#         return retrieved_texts

#     def execute_task(self, query: str):
#         self.create_client()
#         # Ensure the corpus exists or create it
#         # This will now attempt to find an existing corpus first
#         if not self.rag_corpus:
#             self.create_corporus()

#         # Step 1: Perform retrieval directly
#         retrieved_content_list = self.retrieve_documents(query)

#         if not retrieved_content_list:
#             print("No relevant content found through retrieval. Generating response without additional context.")
#             # If no content is retrieved, you might choose to still send the query
#             # or return an empty response based on your application's logic.
#             response = self.client.models.generate_content(
#                 model=self.model_id,
#                 contents=query,
#             )
#             return response.text if response.text else "No relevant information found and model did not generate a direct answer."


#         # Step 2: Construct the prompt with retrieved content
#         context = "\n".join(retrieved_content_list)
#         full_prompt = f"Given the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

#         print("\n--- Sending request to model with retrieved context ---")
#         # Step 3: Send the augmented prompt to the generative model
#         response = self.client.models.generate_content(
#             model=self.model_id,
#             contents=full_prompt,
#         )

#         return response.text # Return the model's generated answer