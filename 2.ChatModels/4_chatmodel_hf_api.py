from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# ChatHuggingFace for use for chat
# HuggingFaceEndpoint is used for whn we communicate with API endpoint


from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of india")

print(result.content)