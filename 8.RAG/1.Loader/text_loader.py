from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('1.txt')    

docs = loader.load()

llm1 = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2-Exp',
    task='text-generation'
)

model1 = ChatHuggingFace(llm=llm1)


prompt = PromptTemplate(
    template='Give a proper name to this poem \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

chain = prompt | model1 | parser
result = chain.invoke({'poem': docs[0].page_content})

print(result)
