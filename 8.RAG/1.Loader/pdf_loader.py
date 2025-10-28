from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Prajakta_Pawar_Resume.pdf')


docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)