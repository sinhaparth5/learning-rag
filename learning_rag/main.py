from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transforms import SentenceTransformer, util
import numpy as np

# Sample dataset (you can expand this with more data)
documents = [
    "Python is a popular programming language.",
    "Java is used in many enterprise applications.",
    "Machine learning is a field of artificial intelligence.",
    "Natural language processing involves understanding and generating human language.",
    "PyTorch is a deep learning framework."
]

model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = model.encode(documents, convert_to_tensor=True)

def retrieve_relevant_documents(query, document_embeddings, top_k=2):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    top_results = np.argpartition(-scores, range(top_k))[0:top_k]
    retrieved_documents = [documents[idx] for idx in top_results]
    return retrieved_documents

generator_model_name = "t5-small"
generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
generator = pipeline('text2text-generation', model=generator_model, tokenizer=generator_tokenizer)

def rag_pipeline(query):
    # Retrieval step
    retrieved_docs = retrieve_relevant_documents(query, document_embeddings, top_k=2)
    
    # Combine the retrieved documents and query
    context = " ".join(retrieved_docs) + " " + query
    
    # Generate response
    generated_response = generator(context, max_length=50, num_return_sequences=1)[0]['generated_text']
    return generated_response

if __name__ == "__main__":
    query = "What is natural language processing?"
    response = rag_pipeline(query)
    print("Query:", query)
    print("Response:", response)