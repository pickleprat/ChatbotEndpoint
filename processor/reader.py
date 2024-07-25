from llama_index.core.readers.base import BaseReader
from llama_index.core import Document 
from typing import List 

import re 
import pymupdf 

# defining a document loader to clean our documents. 
class ProcessedReader(BaseReader): 
    def __init__(self, nlp): 
        self.nlp = nlp 

    def remove_names(self, content: str) -> str: 
        text = self.nlp(content)
        for ent in text.ents: 
            if ent.label_ == "PERSON": 
                content = content.replace(ent.text, "")
        return content  

    def remove_credentials(self, content: str) -> str: 
        pattern = re.compile(
                r'([^\s@]+@[^\s@]+\.[^\s@]+)|(\+91 \d{5} \d{5})')

        content= re.sub(pattern, "", content)
        return content  

    def load_data(self, file: str, extra_info=None) -> List[Document]: 
        doc = pymupdf.open(file)  
        documents = []

        for page in doc: 
            text = page.get_text()
            if not str(file).endswith("AutoMicroUAS Overview.pdf"): 
                text = self.remove_credentials(text)
                text = self.remove_names(text) 
            documents.append(Document(
                text=text, extra_info = extra_info or {},  
                ))

        doc.close()
        return documents


