from llama_index.core import VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from processor.reader import ProcessedReader

import os 

def get_index(nlp, storage_context, vector_store, content_path: str) -> VectorStoreIndex: 
    if "dronedb" not in os.listdir(): 
        # loading your data and generating nodes  
        loader = SimpleDirectoryReader(input_dir=content_path,  
                                       file_extractor={".pdf": ProcessedReader(nlp=nlp)})
        documents = loader.load_data()

        # text splitter to make nodes out of your documents 
        text_splitter = TokenTextSplitter(
                chunk_size=512,
                chunk_overlap=80
                )

        # ingestion pipeline for ingesting out data 
        pipeline = IngestionPipeline(transformations=[text_splitter])
        nodes = pipeline.run(documents=documents)
        index = VectorStoreIndex(nodes, storage_context=storage_context)

    else: 
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context) 

    return index 


