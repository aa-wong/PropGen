#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.utils import get_from_dict_or_env
from pydantic.class_validators import root_validator
from typing import Any, Dict, Optional


# In[ ]:


class LLMChain():
    temperature: int = 0
    
    def __init__(self, values={}):
        env = self.validate_environment(values)
#         "HUGGINGFACEHUB_API_TOKEN"
#         self.llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b", model_kwargs={"temperature":0, "max_length":64})
        self.llm = OpenAI(temperature=self.temperature, openai_api_key=env['openai_api_key'])
        
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        api_key = get_from_dict_or_env(values, "openai_api_key", "OPENAI_API_KEY")
        values["openai_api_key"] = api_key

        return values

    def query(self, query, docs):
        chain = load_qa_chain(self.llm, chain_type="stuff")
        return chain.run(input_documents=docs, question=query)

