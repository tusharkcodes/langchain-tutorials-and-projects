from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain.schema.runnable import RunnableSequence

from dotenv import load_dotenv

load_dotenv()
 
llm1 = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2-Exp',
    task='text-generation'
)

# model2 = GoogleGenerativeAI(model='gemini-2.5-flash')


prompt = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

model = ChatHuggingFace(llm=llm1)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

print(chain.invoke({'topic':'AI'}))