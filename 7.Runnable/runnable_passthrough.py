# Runnable passthrough is return the input same as the outpupt.
# Then Why runnable passthrough is required.
# suppose give the input and we want the same exact output


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough

from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2-Exp',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate the joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate the joke about {text}',
    input_variables=['text']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
   'joke' : RunnablePassthrough(),
   'explaination' : RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
print(final_chain.invoke({'topic' : "Cricket"}))

print(final_chain['joke'])

print('-----------------------')


print(final_chain['explaination'])
