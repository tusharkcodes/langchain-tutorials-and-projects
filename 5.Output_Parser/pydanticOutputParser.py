# Pydantic models to enforce schema validationwhen processing LLM responses.


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2-Exp',
    task='text-generation',
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    
    name :str = Field(description="Name of the person")
    age : int = Field(description='age of the person')
    city : str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template='generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


# prompt = template.invoke({'place': 'indian'})

# result = model.invoke(prompt)

# print(result.content)

chain = template | model

result = chain.invoke({'place': 'canada'})

print(result.content)