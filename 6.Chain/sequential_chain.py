from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2-Exp',
    task='text-generation',
    temperature= 0.9
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Give me some point in the {topic} related',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='{text} ive me information related to the batsman',
    input_variables=['text']
)

parser = StrOutputParser()



chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : 'cricket'})

print(result)