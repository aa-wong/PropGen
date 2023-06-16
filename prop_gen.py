#!/usr/bin/env python
# coding: utf-8

# In[3]:


import magic
import nltk
from langchain.utils import get_from_dict_or_env
from llm_chain import LLMChain
from vector_indexer import VectorIndexer
from pydantic.class_validators import root_validator
from typing import Any, Dict, Optional
from preprocessor import preprocess_pdf_documents_in_directory, convert_directory


# In[4]:


nltk.download('punkt')


# In[5]:


class PropGen():
    index_name = "Index_1"
    preprocessed_directory = "./output_directory"

    def __init__(self, values={}):
        self.llm = LLMChain(values)
        self.indexer = VectorIndexer(values)
        
        env = self.validate_environment(values)
        self.input_directory = env["input_directory"]
        self.index_name = env['index_name']
        
    def get_index_name(self):
        return self.index_name
      
    def set_index_name(self, index_name):
        self.index_name = index_name
        
    def get_input_directory(self):
        return self.input_directory
      
    def set_input_directory(self, index_name):
        self.input_directory = input_directory
        
    def get_preprocessed_directory(self):
        return self.preprocessed_directory
      
    def set_preprocessed_directory(self, index_name):
        self.preprocessed_directory = preprocessed_directory
        
    def get_llm(self):
        return self.llm
      
    def set_llm(self, llm):
        self.llm = llm
        
    def get_indexer(self):
        return self.indexer
      
    def set_indexer(self, indexer):
        self.llm = indexer

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        index_name = get_from_dict_or_env(values, "index_name", "INDEX_NAME")
        input_directory = get_from_dict_or_env(values, "input_directory", "INPUT_DIRECTORY")
        values["index_name"] = index_name
        values["input_directory"] = input_directory

        return values
        
    def query(self, query):
        docsearch = self.indexer.docsearch(self.index_name)
        docs = docsearch.similarity_search(query)

        return self.llm.query(query, docs)
        
    def update_index(self):
        convert_directory(self.input_directory, self.preprocessed_directory)
        pages = preprocess_pdf_documents_in_directory(self.preprocessed_directory)
        print(f"Updating index: {self.index_name}")
        insert_pdf_pages(self, self.index_name, pages)

