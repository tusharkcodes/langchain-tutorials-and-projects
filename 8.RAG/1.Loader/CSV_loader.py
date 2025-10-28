from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path=".....")

docs = loader.load()

print(docs[0].page_content)