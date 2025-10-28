from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch

from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2-Exp',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Generate the detail report {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the report on {topic}',
    input_variables=['topic']
)

gen_report= RunnableSequence(prompt, model, parser)

parallel_chain = RunnableBranch(
    (lambda x : len(x.split())>5000, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(gen_report, parallel_chain)

result = chain.invoke({'topic': 'India vs Pakistan'})

print(result)