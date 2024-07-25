from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from .indexer import get_index 
import chromadb 

def access_engine(database_path: str, collection_name: str, 
                  content_path: str, nlp, similarity_top_k=10, streaming=True): 

    # using chromadb to get a vectorstoreindex 
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # get vectorstore index 
    index = get_index(
            nlp=nlp, 
            vector_store=vector_store, 
            storage_context=storage_context, 
            content_path=content_path, 
        ) 

    llm = index.as_query_engine(streaming=streaming, similarity_top_k=similarity_top_k) 
    return llm 


