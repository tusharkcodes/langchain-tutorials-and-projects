from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2-Exp',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

paper_input = st.selectbox("Select Research Paper Name", ["Attension is all you need", "BERT: Pre-trainning of Deep Bidirectional Transformers", "GPT-3 : Language Models are Few-shots Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox("Select Explaination Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathmatical"])
length_input = st.selectbox("Select Explaination length", ["Shorts (1-2 paragraphs)", "Medium (3-5) paragraphs", "Long (Detailed explaination)"])

# Template
template = PromptTemplate(
    template="""
Please summarise the research paper titled "{paper_input}" with the followig specifications:
Explaination style : "{style_input}"
Explaination length : "{length_input}"

1 Mathmatical Details:
  Include mathmatical equation if present in the paper.
  Expalin mathmatical concept using simple, code snippet

2 Analogies:
  Use relatable analogies and simplifies the complex ideas.
  if cretain information is not available instead of guessing say insufficient
  """ ,
  input_variables=['paper_input','style_input','length_input']
)

template = load_prompt('template.json')

prompt = template.invoke({
    'paper_input':paper_input,
    'length_input':length_input,
    'style_input':style_input
})

# user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'length_input':length_input,
        'style_input':style_input
    })
   
    st.write(result.content)
