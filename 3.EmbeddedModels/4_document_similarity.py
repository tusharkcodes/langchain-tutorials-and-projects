from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()


embedding = OpenAIEmbeddings(model='text-embedding-3-large')

documents = [
    "Shubman Gill has been appointed as India's new ODI captain, succeeding Rohit Sharma ahead of the three-match series against Australia starting October 19; he continues as Test captain and T20I vice-captain",
    "Rohit Sharma has been removed as ODI captain, with Shubman Gill taking over, marking a significant leadership change in Indian cricket.",
    "Virat Kohli returns to India's ODI squad for the upcoming series against Australia, marking his first international appearance since March 2025",
    "Jasprit Bumrah has been rested for the ODI series against Australia, with young pacer Yashasvi Jaiswal making a comeback to the squad",
    "hikhar Dhawan visited the Shri Mahakaleshwar Jyotirlinga Temple in Ujjain, Madhya Pradesh, on October 5, 2025, participating in the sacred Bhasma Aarti ritual, highlighting his spiritual engagement"
]  

query = 'tell me about virat kohli'


doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)

index, score = sorted(list(enumerate(scores)), key =lambda x:x[1])[-1]

print(query)
print(documents[index])