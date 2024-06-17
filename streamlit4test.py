import streamlit as st
from typing import List, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from tinydb import TinyDB, Query
import os
import io
from PyPDF2 import PdfReader
import torch

class DenseRetriever:
    def __init__(self, model):
        self.model = model

    def encode(self, texts: List[str]):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings

    def retrieve(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        query_embedding = self.encode([query])
        document_embeddings = self.encode(documents)
        similarities = cosine_similarity(query_embedding.cpu().numpy(), document_embeddings.cpu().numpy())
        top_k_indices = similarities.argsort(axis=1)[0][-top_k:][::-1]
        return [(index, similarities[0][index]) for index in top_k_indices]

class RAGSystem:
    def __init__(self, retriever, generator_model, tokenizer):
        self.retriever = retriever
        self.generator_model = generator_model
        self.tokenizer = tokenizer

    def generate_answer(self, query, chunks: List[Dict], top_k=3, temperature=1.0, pre_prompt="You can only answer questions about the provided context. If you know the answer but it is not based in the provided context, don't provide the answer, just state the answer is not in the context provided."):
        # Retrieve the top k most relevant chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        retrieved_indices = self.retriever.retrieve(query, chunk_texts, top_k)
        
        # Get the text content and indices of the retrieved chunks
        retrieved_chunks = [chunks[i] for i, _ in retrieved_indices]

        # Concatenate the pre-prompt and the retrieved chunks into a single context
        context = pre_prompt + " " + " ".join([chunk['text'] for chunk in retrieved_chunks])
        
        # Generate an answer using the RAG System with temperature control
        inputs = self.tokenizer.encode_plus(query, context, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.generator_model(**inputs)
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

        # Get the most likely beginning and end of the answer span
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = self.tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end])

        # Identify the source chunk and page
        reference_chunk = retrieved_chunks[0] if retrieved_chunks else None
        reference_page = reference_chunk['page'] if reference_chunk else None

        # Format the answer output with reference information
        answer_output = f"Answer: {answer} (Referenced from Document {reference_chunk['doc_id'] + 1}, Page {reference_page})"
        
        return answer_output

@st.cache(allow_output_mutation=True)
def load_models():
    retriever = SentenceTransformer('msmarco-distilbert-base-v2')
    generator_model = AutoModelForQuestionAnswering.from_pretrained('allenai/scibert_scivocab_uncased')  # Use SciBERT
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')  # Use SciBERT
    return retriever, generator_model, tokenizer

def extract_text_from_pdf(uploaded_file):
    """
    Extract text content from a PDF file and return it along with page number references.
    """
    file_contents = uploaded_file.read()
    reader = PdfReader(io.BytesIO(file_contents))
    chunks = []
    chunk_size = 300  # Define a chunk size (number of characters)
    for doc_id, page in enumerate(reader.pages):
        page_text = page.extract_text()
        for i in range(0, len(page_text), chunk_size):
            chunk = page_text[i:i+chunk_size]
            chunks.append({'text': chunk, 'page': doc_id + 1, 'doc_id': len(db)})
    return chunks

# Change storage to file-based storage
db = TinyDB('db.json')
Document = Query()

retriever, generator_model, tokenizer = load_models()
rag_system = RAGSystem(DenseRetriever(retriever), generator_model, tokenizer)

# Streamlit interface
st.title("RAG System for Cybersecurity Question Answering")

# Upload documents
uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type="pdf")
if uploaded_files:
    for uploaded_file in uploaded_files:
        document_chunks = extract_text_from_pdf(uploaded_file)
        db.insert({'chunks': document_chunks})

# Display documents in the database
st.subheader("Stored Documents")
all_chunks = []
for item in db.all():
    all_chunks.extend(item['chunks'])
if all_chunks:
    st.text(f"Total Chunks: {len(all_chunks)}")

# Query input
query = st.text_input("Enter your query:")
pre_prompt = "You are an expert in cybersecurity. Provide a detailed answer based on the provided documents."
if st.button("Generate Answer"):
    if query:
        temperature = 0.7  # Set the temperature value as desired
        answer = rag_system.generate_answer(query, all_chunks, top_k=3, temperature=temperature, pre_prompt=pre_prompt)
        st.write(answer)
    else:
        st.write("Please enter a query.")
