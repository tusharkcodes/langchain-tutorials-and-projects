# Each runnable receives the same input and process it independantly, producing a dictionary of outputs

#                      topic = AI
#
#           LLM                            LLM

#         generate post              generate post
#         related tweet              related linkdin
#       

# both get the same input but excutes independantly
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2-Exp',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate the tweet post about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate the linkdin post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
   'tweet' : RunnableSequence(prompt1, model, parser),
   'linkdin' : RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic': 'AI'})

print(result)