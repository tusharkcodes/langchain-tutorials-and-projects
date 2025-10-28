from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()
 
llm1 = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2-Exp',
    task='text-generation'
)

model1 = ChatHuggingFace(llm=llm1)
model2 = GoogleGenerativeAI(model='gemini-2.5-flash')

# prompt 1
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following topic \n {text}',
    input_variables=['text']
)

# prompt 2
prompt2 = PromptTemplate(
    template='Generate 5 questions and answere from the following topic \n {text}',
    input_variables=['text']
)

# prompt 3

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and {quiz}',
    input_variables=['notes', 'quiz']
)


parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser ,
    'quiz' : prompt2 | model2 | parser
})
merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

text = """
1.1. Linear Models
The following are a set of methods intended for regression in which the target value is expected to be a linear combination of the features. In mathematical notation, if 
 is the predicted value.

Across the module, we designate the vector 
 as coef_ and 
 as intercept_.

To perform classification with generalized linear models, see Logistic regression.

1.1.1. Ordinary Least Squares
LinearRegression fits a linear model with coefficients 
 to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation. Mathematically it solves a problem of the form:

 
../_images/sphx_glr_plot_ols_ridge_001.png
LinearRegression takes in its fit method arguments X, y, sample_weight and stores the coefficients 
 of the linear model in its coef_ and intercept_ attributes:

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
LinearRegression()
reg.coef_
array([0.5, 0.5])
reg.intercept_
0.0
The coefficient estimates for Ordinary Least Squares rely on the independence of the features. When features are correlated and some columns of the design matrix 
 have an approximately linear dependence, the design matrix becomes close to singular and as a result, the least-squares estimate becomes highly sensitive to random errors in the observed target, producing a large variance. This situation of multicollinearity can arise, for example, when data are collected without an experimental design.

"""

result = chain.invoke({'text' : text})
print(result)

# Tried over the manually

# chain1 = prompt1 | model1 | parser
# chain2 = chain1 | prompt2 | model2 | parser
# chain3 = chain2 | prompt3 | model2 | parser
# result = chain3.invoke({'text': 'Machine Learning'})
# print(result)

# Traditional way

# result1 = model1.invoke("What is the capital of India?")
# result2 = model2.invoke("What is the capital of Bangladesh?")

# print(result1.content)
# print(result2)
