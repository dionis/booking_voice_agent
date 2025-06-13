from supabase_client import search_similar_embedding_experiences
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

if __name__ == "__main__":
   text = "Enjoy a full day full of adventure, Starting the"

   model = SentenceTransformer(EMBEDDING_MODEL)

   # Create embeddings for each chunk
   embedding = model.encode(text).tolist()
   print(embedding)

   ## Fix error https://github.com/langchain-ai/langchain/issues/10065
   result_rag = search_similar_embedding_experiences(embedding)

   if len(result_rag.data) == 0:
      print(f"Not exist result :{result_rag.data}")
   else:
      for iResult_in_rag in result_rag.data:
         print(f"The most related experiences is : {iResult_in_rag['experience_id']}")

   #print(search_similar_embedding_experiences(embedding))


