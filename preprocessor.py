#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import pdfkit
import os
from pypdf import PdfMerger


# In[ ]:


def preprocess_pdf_documents_in_directory(directory):
    print(f"Preprocessing pdfs in directory {directory}")
    loader = DirectoryLoader(directory, glob='**/*.pdf')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)


# In[11]:


def excel_to_pdf(input_file, output_file):
    print(f"Converting excel: {input_file}")
    path_split = input_file.split('/')
    filename = path_split[len(path_split) - 1].split('.')[0]
        
#     if os.path.isdir(filename) is False:
#         os.mkdir(filename) 

    temp_pdfs = []
    temp_htmls = []
    xls = pd.ExcelFile(input_file)
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        df = df.fillna('')
        df.to_html(
            temp_file := "".join([filename, "_", str(sheet_name), ".html"])
        )
        pdfkit.from_file(
            temp_file,
            temp_pdf := "".join([filename, "_", str(sheet_name), ".pdf"]),
            options={"enable-local-file-access": "", "quiet": ""},
        )
        temp_htmls.append(temp_file)
        temp_pdfs.append(temp_pdf)
    
    def merge_pdf(pdfs, output):
        merger = PdfMerger()

        for pdf in pdfs:
            merger.append(pdf)

        merger.write(output)
        merger.close()

    merge_pdf(temp_pdfs, output_file + ".pdf")
    
    remove = temp_pdfs + temp_htmls
    
    for file in remove:
        if os.path.exists(file):
            os.remove(file)


# In[24]:


from docx2pdf import convert


# In[25]:


def convert_docx_to_pdf(input_path, output_path):
    try:
        print(f"Converting docx: {input_path}")
        convert(input_path, output_path + ".pdf")
    except Exception as e:
        print(f'Error while converting {input_path} to {output_path}: {e}')


# In[14]:


from os import listdir
from os.path import isfile, join


# In[18]:


def convert_directory(dir_path, output_path):
    print(f"converting files in directoy: {dir_path} to {output_path}")
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path) 

    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    
    for file in files:
        path_split = file.split('/')
        filename = path_split[len(path_split) - 1].split('.')[0]
    
        if file.endswith((".xls", ".xlsx")):
            excel_to_pdf(dir_path + "/" + file, output_path + "/" + filename)
        elif file.endswith(('.docx', ".doc")):
            convert_docx_to_pdf(dir_path + "/" + file, output_path + "/" + filename)


# In[ ]:




