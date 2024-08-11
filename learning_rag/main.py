from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Sample dataset (you can expand this with more data)
documents = [
    "Python is a popular programming language.",
    "Java is used in many enterprise applications.",
    "Machine learning is a field of artificial intelligence.",
    "Natural language processing involves understanding and generating human language.",
    "PyTorch is a deep learning framework."
]

# Step 1: Encode the documents using a Sentence Transformer
model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = model.encode(documents, convert_to_tensor=True)

# Step 2: Define a function for retrieval
def retrieve_relevant_documents(query, document_embeddings, top_k=2):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    
    # Move scores tensor to CPU before converting to NumPy array
    scores = scores.cpu().numpy()
    
    # Get the indices of the top_k scores
    top_results = np.argpartition(-scores, range(top_k))[0:top_k]
    retrieved_documents = [documents[idx] for idx in top_results]
    return retrieved_documents

# Step 3: Set up the text generation model (e.g., T5)
generator_model_name = "t5-small"
generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)

# Use device 0 for GPU or -1 for CPU
device = 0
generator = pipeline('text2text-generation', model=generator_model, tokenizer=generator_tokenizer, device=device)

# Step 4: Implement the RAG pipeline
def rag_pipeline(query):
    # Retrieval step
    retrieved_docs = retrieve_relevant_documents(query, document_embeddings, top_k=2)
    
    # Combine the retrieved documents and query
    context = " ".join(retrieved_docs) + " " + query
    
    # Generate response
    generated_response = generator(context, max_length=50, num_return_sequences=1)[0]['generated_text']
    return generated_response

# Test the RAG pipeline
if __name__ == "__main__":
    query = "What is natural language processing?"
    response = rag_pipeline(query)
    print("Query:", query)
    print("Response:", response)