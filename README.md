# 📄 Extracting Information from Legal Documents Using RAG

This project implements a complete end-to-end Retrieval-Augmented Generation (RAG) pipeline to intelligently query and extract information from a large corpus of legal documents. It leverages state-of-the-art NLP models and frameworks to provide accurate, context-aware answers to complex legal questions.

## 📚 Table of Contents
- Objective  
- Project Pipeline  
- Tech Stack  
- Setup and Usage  
- Evaluation Results  
- Future Improvements  

## 🎯 Objective
The primary goal is to build a system that can understand and analyze legal agreements (like NDAs, privacy policies, and merger contracts) to answer specific user queries. This automates the time-consuming process of manual document review, enabling legal professionals to find relevant information quickly and efficiently.

## 🔁 Project Pipeline
The project is structured as a sequential pipeline, from raw data ingestion to evaluation:

### 📥 Data Loading & Preprocessing
- Loads 698 .txt legal documents from various categories (contractnli, cuad, maud, privacy_qa).
- Applies extensive text cleaning to remove irrelevant characters, extra whitespace, and contact information.
- Normalizes text by converting to lowercase and removing stopwords.

### 📊 Exploratory Data Analysis (EDA)
- Analyzes document statistics, including average, maximum, and minimum lengths.
- Performs word frequency analysis to identify common legal terminology.
- Calculates TF-IDF based cosine similarity to understand inter-document relationships.

### ✂️ Document Chunking
- Splits the cleaned documents into smaller, manageable chunks (chunk_size=1200, overlap=200) using RecursiveCharacterTextSplitter to preserve semantic context.

### 🧠 Vectorization & Storage
- Uses sentence-transformers/all-MiniLM-L6-v2 model to generate vector embeddings for each text chunk.
- Stores these embeddings in a FAISS vector database for efficient similarity search and retrieval.

### 🔗 RAG Chain Construction
- Builds a RAG chain using LangChain.
- The chain utilizes the FAISS vector store as its retriever, which fetches the top 4 most relevant document chunks for a given query.
- Google's gemini-1.5-flash model is used as the generator to synthesize a final answer based on the retrieved context.

### ✅ Evaluation
- Evaluates the pipeline on a benchmark of 100 questions against ground-truth answers.
- Employs a suite of metrics:  
  - Traditional Metrics: ROUGE-L and BLEU for lexical similarity.  
  - Ragas Metrics: Faithfulness, Answer Relevancy, Context Precision, and Context Recall for LLM-based evaluation.

## 🧰 Tech Stack

| Component          | Tool / Library                          |
|--------------------|------------------------------------------|
| Core Logic         | Python 3.10                              |
| Framework          | LangChain                                |
| LLM                | Google Gemini 1.5 Flash                  |
| Embeddings         | sentence-transformers/all-MiniLM-L6-v2   |
| Vector Database    | FAISS (Facebook AI Similarity Search)    |
| NLP & Data Tools   | transformers, pandas, nltk, scikit-learn |
| Evaluation         | ragas, evaluate (Hugging Face)           |
| Environment        | Jupyter Notebook, Google Colab           |

