#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pinecone
from langchain.vectorstores import Pinecone
from langchain.utils import get_from_dict_or_env
from pydantic.class_validators import root_validator
from typing import Any, Dict, Optional
from langchain.embeddings import OpenAIEmbeddings


# In[2]:


class VectorIndexer():
    dimensions: int = 1536
    metric: str ='dotproduct'
    
    def __init__(self, values={}):
        env = self.validate_environment(values)

        self.embeddings = OpenAIEmbeddings(openai_api_key=env['openai_api_key'])
        self.pinecone = pinecone.init(
            api_key=env['pinecone_api_key'],  # find at app.pinecone.io
            environment=env['pinecone_api_env']
        )
        
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        openai_api_key = get_from_dict_or_env(values, "openai_api_key", "OPENAI_API_KEY")
        pinecone_api_key = get_from_dict_or_env(values, "pinecone_api_key", "PINECONE_API_KEY")
        env = get_from_dict_or_env(values, "pinecone_api_env", "PINECONE_API_ENV")
        values["pinecone_api_key"] = pinecone_api_key
        values["pinecone_api_env"] = env
        values["openai_api_key"] = openai_api_key
        
        return values
    
    def create_index(self, index_name):
        if index_name not in pinecone.list_indexes():
            # we create a new index
            print(f"Creating new index: {index_name}")
            pinecone.create_index(
                name=index_name,
                metric=self.metric,
                dimension=self.dimensions  # 1536 dim of text-embedding-ada-002
            )

    def insert_text(self, index_name, text):
        self.create_index(index_name)
        print(f"Inserting text to index: {index_name}")
        return Pinecone.from_texts(text, self.embeddings, index_name=index_name)
    
    def insert_pdf_pages(self, index_name, pages):
        print(f"Inserting PDF docs to index: {index_name}")
        return self.insert_text(index_name, [t.page_content for t in pages])
    
    def docsearch(self, index_name):
        if index_name not in pinecone.list_indexes():
            raise ValueError(
                f"index not found with name: {index_name}. Execute 'create_index' or 'insert' to continue."
            )
        return Pinecone.from_existing_index(index_name, self.embeddings)

