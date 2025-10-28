from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel , RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2-Exp',
    task='text-generation'
)


model = ChatHuggingFace(llm=llm1)

class FeedBack(BaseModel):

    sentiment : Literal['positive', 'negative'] = Field('give the feedback positive or negative')

parser = StrOutputParser()
parser1 = PydanticOutputParser(pydantic_object=FeedBack)

prompt1 = PromptTemplate(
    template='give the feedback positive or negative given product review feedback \n {feedback} \n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser1.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser1 

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback} \n{format_instruction}',
    input_variables=['feedback'],
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback} \n{format_instruction}',
    input_variables=['feedback'],
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x : "could not find sentiment")

    # default chain
)
# --- Combine ---
chain = classifier_chain | branch_chain

# --- Run ---
result = chain.invoke({"feedback": "This is very good smartphone"})
print(result)