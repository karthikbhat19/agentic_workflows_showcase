# Agentic Workflows Showcase

This repository demonstrates several Retrieval-Augmented Generation (RAG) approaches, each implemented in a separate notebook. Below is an overview of each approach and its workflow.

---

## 1. Vanilla RAG

**Approach:**  
Vanilla RAG implements the classic RAG pipeline with straightforward document retrieval and answer generation. It uses similarity search on embedded documents and generates answers using a simple prompt template.

**Flow:**  

- Load documents and create embeddings using HuggingFaceEmbeddings.
- Store embeddings in a FAISS vector database.
- For each query:
  - Perform similarity search to retrieve top-k relevant documents.
  - Format retrieved documents as context.
  - Use a simple template prompt to generate answers with ChatOpenAI.
  - Apply string output parsing for final response.

---

## 2. Agentic RAG

**Approach:**  
Agentic RAG uses a sophisticated agent-based workflow with tool usage, document relevance grading, and iterative query refinement. It features an intelligent agent that decides when to retrieve information and can rewrite queries when documents are not relevant.

**Flow:**  

- **Agent Decision**: An agent with retriever tools decides whether to retrieve documents or end the conversation.
- **Document Retrieval**: Uses LangChain's ToolNode with a retriever tool for information gathering.
- **Relevance Grading**: LLM evaluates retrieved documents for relevance using structured output.
- **Query Rewriting**: If documents are not relevant, the query is transformed and improved.
- **Iterative Refinement**: The process repeats with rewritten queries until relevant documents are found.
- **Answer Generation**: Uses RAG prompt from LangChain Hub for final synthesis.

---

## 3. Adaptive RAG

**Approach:**  
Adaptive RAG intelligently routes queries between different retrieval strategies (local retriever vs. web search) and includes document grading, query transformation, and web search fallback capabilities.

**Flow:**  

- **Query Routing**: LLM-based router decides between local retriever or web search based on query content.
- **Document Grading**: Retrieved documents are scored for relevance using structured LLM evaluation.
- **Multi-source Retrieval**: Can retrieve from both vector database and web search (Tavily).
- **Query Transformation**: Improves queries when retrieval quality is insufficient.
- **Hallucination Detection**: Grades generated answers to detect and prevent hallucinations.
- **Dynamic Strategy Selection**: Adapts retrieval strategy based on document relevance scores.

---

## 4. RARE RAG

**Approach:**  
RARE RAG (Retrieval-Augmented Reasoning and Evaluation) employs multiple independent reasoning agents that generate diverse chain-of-thought approaches, retrieve evidence for each perspective, and select the best answer through ensemble evaluation.

**Flow:**  

- **Multi-Agent CoT Generation**: Multiple agents (3 by default) generate different chain-of-thought approaches with varying perspectives.
- **Parallel Processing**: Each agent independently processes their CoT with different reasoning focuses.
- **Evidence Retrieval**: For each CoT step, relevant documents are retrieved and integrated.
- **Answer Refinement**: Each agent refines their reasoning using retrieved evidence.
- **Best Answer Selection**: LLM evaluates all agent outputs and selects the most comprehensive answer.
- **Ensemble Evaluation**: Combines multiple reasoning paths for robust answer generation.

---

## 5. RatRAG / CoT RAG

**Approach:**  
RatRAG implements a step-by-step Chain-of-Thought workflow where each thought step is individually processed through retrieval, evidence integration, and revision before aggregating into a final comprehensive answer.

**Flow:**  

- **Initial CoT Generation**: LLM generates a numbered list of thought steps for the query.
- **Step-by-Step Processing**: For each thought step:
  - Generate specific retrieval query for that step.
  - Retrieve relevant evidence using vector search.
  - Revise the thought step by incorporating retrieved evidence.
- **Sequential Refinement**: Process each step sequentially with evidence integration.
- **Final Aggregation**: Combine all revised thought steps into a comprehensive final answer.
- **Iterative Evidence Integration**: Each reasoning step is enhanced with specific supporting documents.

---

## 6. Hybrid RAG

**Approach:**  
Hybrid RAG is the most comprehensive approach that intelligently combines multiple RAG strategies in a single workflow. It adaptively chooses between direct retrieval, Chain-of-Thought decomposition, vector search, and web search based on confidence scoring. This approach acts as a meta-RAG system that incorporates elements from all other approaches.

**Flow:**  

- **Phase 1 - Direct Retrieval Attempt:**
  - Perform initial vector search with confidence scoring
  - If confidence is low (< 0.6), iteratively rewrite queries up to 3 times
  - Synthesize answer and evaluate confidence
  - If confidence ≥ 0.6, return direct answer

- **Phase 2 - Hybrid Search Fallback:**
  - If direct retrieval confidence remains low, fallback to web search (Tavily API)
  - Iteratively rewrite web queries up to 3 times for better results
  - Score and compare vector vs web search results
  - Choose the highest-scoring source

- **Phase 3 - Chain-of-Thought Decomposition:**
  - If initial attempts fail, break complex queries into 3-5 sub-questions
  - For each sub-question:
    - Perform hybrid retrieval (vector + web search as needed)
    - Rerank results using LLM-based relevance scoring
    - Aggregate supporting documents
  - Synthesize comprehensive answer from all retrieved contexts
  - Final confidence check - return "NO_ANSWER" if still below threshold

- **Advanced Features:**
  - LLM-based document relevance scoring and reranking
  - Query rewriting for both vector and web search optimization
  - Adaptive strategy selection based on confidence thresholds
  - RAGAS evaluation metrics for answer quality assessment

---

## Common Features

- **Document Loading:** Most approaches use DataFrameLoader or similar utilities to load and preprocess documents.
- **Vector Search:** FAISS and HuggingFaceEmbeddings are commonly used for semantic retrieval.
- **Web Search:** Some approaches integrate external search APIs for broader context.
- **LLM Synthesis:** All approaches use an LLM (e.g., OpenAI GPT, HuggingFaceHub) for answer synthesis and scoring.
- **Evaluation:** RAGAS metrics are used for quantitative evaluation of answer quality.

---

## Prerequisites

- Python 3.8+
- Required packages: `langchain`, `transformers`, `faiss-cpu`, `openai`, `tavily-python`, `ragas`, `pandas`
- API keys for OpenAI and Tavily (for web search)

---

## Notebook Overview

- **`Vanilla_RAG.ipynb`**: Classic RAG implementation with similarity search and template-based answer generation
- **`HybridRag.ipynb`**: Comprehensive meta-RAG system combining adaptive strategy selection, CoT decomposition, hybrid search (vector + web), and advanced confidence scoring
- **`AgenticRAG.ipynb`**: Agent-based workflow with tool usage, document relevance grading, and iterative query refinement
- **`AdaptiveRAG.ipynb`**: Intelligent query routing between retrieval strategies with document grading and web search fallback
- **`RARE_Rag.ipynb`**: Multi-agent ensemble reasoning with diverse CoT perspectives and best answer selection
- **`RatRAG_CoT_RAG.ipynb`**: Step-by-step Chain-of-Thought with evidence integration for each reasoning step

---

Each notebook is self-contained and demonstrates a unique RAG workflow, allowing you to compare strategies and adapt them to your needs.
