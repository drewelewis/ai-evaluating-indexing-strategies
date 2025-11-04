# Indexing Strategies Comparison

This document provides a comprehensive comparison of different indexing approaches, their strengths, limitations, and optimal use cases.

## 🔍 Overview of Indexing Strategies

### Strategy Comparison Matrix

| Strategy | Latency | Recall | Precision | Semantic Understanding | Setup Complexity | Cost |
|----------|---------|--------|-----------|----------------------|------------------|------|
| **Keyword (BM25)** | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⚡ | 💰 |
| **Vector (Dense)** | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⚡⚡ | 💰💰 |
| **Hybrid** | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⚡⚡⚡ | 💰💰💰 |
| **Sparse Vector** | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⚡⚡ | 💰💰 |

*Legend: ⚡ = Fast/Simple/Cheap, ⭐ = Good performance*

## 1. Keyword-Based Indexing (BM25/TF-IDF)

### How It Works
```python
# BM25 Scoring Formula
def bm25_score(query_terms, document, corpus_stats):
    score = 0
    for term in query_terms:
        tf = document.term_frequency(term)
        idf = math.log((len(corpus_stats.docs) - corpus_stats.docs_with_term(term) + 0.5) / 
                      (corpus_stats.docs_with_term(term) + 0.5))
        
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * document.length / corpus_stats.avg_doc_length))
    
    return score
```

### Strengths
- ✅ **Fast retrieval**: Sub-millisecond query processing
- ✅ **Exact match**: Perfect for specific terms, codes, IDs
- ✅ **Explainable**: Clear understanding of why documents match
- ✅ **Low cost**: Minimal computational requirements
- ✅ **Mature technology**: Well-understood, battle-tested

### Limitations
- ❌ **No semantic understanding**: "car" ≠ "automobile" 
- ❌ **Vocabulary mismatch**: Query and document must share terms
- ❌ **Poor synonym handling**: Requires manual synonym lists
- ❌ **Limited context**: Ignores word order and relationships

### Optimization Strategies
```python
bm25_config = {
    "k1": 1.5,        # Term frequency saturation (1.2-2.0)
    "b": 0.75,        # Document length normalization (0.5-0.8)
    "custom_analyzers": {
        "product_analyzer": {
            "tokenizer": "standard",
            "filters": ["lowercase", "stop", "synonym", "stemmer"]
        }
    },
    "field_boosts": {
        "title": 2.0,
        "description": 1.0,
        "tags": 1.5
    }
}
```

### Best Use Cases
- **Product search** with specific model numbers
- **Legal document** retrieval with exact citations
- **Technical documentation** with precise terminology
- **Code search** with function/variable names

### Performance Expectations
| Metric | Typical Range | Optimization Target |
|--------|---------------|-------------------|
| Latency | 1-10ms | < 5ms |
| Precision@5 | 0.6-0.8 | > 0.75 |
| Recall | 0.4-0.7 | > 0.6 |
| Storage | 10-20% of corpus | < 15% |

## 2. Vector-Based Indexing (Dense Embeddings)

### How It Works
```python
class VectorSearchEngine:
    def __init__(self, embedding_model, vector_db):
        self.encoder = embedding_model  # e.g., sentence-transformers
        self.vector_db = vector_db      # e.g., Pinecone, Weaviate, FAISS
    
    def index_documents(self, documents):
        embeddings = self.encoder.encode(documents)
        self.vector_db.upsert(embeddings, metadata=documents)
    
    def search(self, query, top_k=10):
        query_vector = self.encoder.encode([query])
        results = self.vector_db.query(query_vector, top_k=top_k)
        return results
```

### Vector Database Options

#### FAISS (Facebook AI Similarity Search)
```python
faiss_config = {
    "index_type": "IVF_HNSW",      # Inverted file + HNSW
    "nlist": 1024,                # Number of clusters
    "m": 16,                      # HNSW connections per node
    "ef_construction": 200,       # Build-time search width
    "ef_search": 100             # Query-time search width
}
# Pros: Fast, free, local deployment
# Cons: No built-in persistence, limited scalability
```

#### Pinecone
```python
pinecone_config = {
    "dimension": 768,
    "metric": "cosine",
    "pod_type": "p1.x1",
    "replicas": 2,
    "metadata_config": {
        "indexed": ["category", "price_range"]
    }
}
# Pros: Managed service, easy scaling, metadata filtering
# Cons: Cost, vendor lock-in
```

#### Azure Cognitive Search
```python
azure_vector_config = {
    "dimensions": 1536,
    "vectorSearchProfile": "my-vector-profile",
    "algorithm": "hnsw",
    "hnswParameters": {
        "metric": "cosine",
        "m": 4,
        "efConstruction": 400,
        "efSearch": 500
    }
}
# Pros: Integrated with Azure ecosystem, hybrid search
# Cons: Azure-specific, learning curve
```

### Embedding Model Selection

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| **all-MiniLM-L6-v2** | 384 | ⚡⚡⚡ | ⭐⭐ | General purpose, fast |
| **all-mpnet-base-v2** | 768 | ⚡⚡ | ⭐⭐⭐ | Better quality, slower |
| **OpenAI text-embedding-3-large** | 3072 | ⚡ | ⭐⭐⭐⭐ | Highest quality, API cost |
| **E5-large-v2** | 1024 | ⚡⚡ | ⭐⭐⭐ | Good balance |

### Optimization Strategies
1. **Chunking Strategy**
   ```python
   def smart_chunking(document, chunk_size=512, overlap=50):
       # Semantic chunking at sentence boundaries
       sentences = nltk.sent_tokenize(document)
       chunks = []
       current_chunk = ""
       
       for sentence in sentences:
           if len(current_chunk) + len(sentence) > chunk_size:
               chunks.append(current_chunk)
               current_chunk = sentence
           else:
               current_chunk += " " + sentence
       
       return chunks
   ```

2. **Index Tuning**
   ```python
   hnsw_params = {
       "M": 16,              # Higher M = better recall, more memory
       "efConstruction": 200, # Higher ef = better quality, slower build
       "efSearch": 100       # Higher ef = better recall, slower search
   }
   ```

### Best Use Cases
- **Semantic search**: "Find documents about machine learning concepts"
- **Multilingual search**: Cross-language information retrieval
- **Question answering**: Finding passages that answer questions
- **Recommendation systems**: Content-based recommendations

### Performance Expectations
| Metric | Typical Range | Optimization Target |
|--------|---------------|-------------------|
| Latency | 10-100ms | < 50ms |
| Recall | 0.7-0.9 | > 0.85 |
| nDCG@10 | 0.6-0.8 | > 0.75 |
| Storage | 4-8x text size | < 6x |

## 3. Hybrid Indexing (Keyword + Vector)

### Architecture Patterns

#### 1. Parallel Retrieval + Fusion
```python
class HybridSearchEngine:
    def __init__(self, keyword_engine, vector_engine):
        self.keyword_engine = keyword_engine
        self.vector_engine = vector_engine
    
    def search(self, query, alpha=0.5):
        # Parallel retrieval
        keyword_results = self.keyword_engine.search(query)
        vector_results = self.vector_engine.search(query)
        
        # Score fusion
        fused_results = self.reciprocal_rank_fusion(
            keyword_results, vector_results, alpha
        )
        return fused_results
    
    def reciprocal_rank_fusion(self, list1, list2, alpha=0.5):
        # RRF: 1/(rank + k) where k=60 is common
        k = 60
        scores = {}
        
        for rank, doc in enumerate(list1):
            scores[doc.id] = alpha / (rank + k)
        
        for rank, doc in enumerate(list2):
            scores[doc.id] = scores.get(doc.id, 0) + (1-alpha) / (rank + k)
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

#### 2. Query Classification + Routing
```python
class QueryRouter:
    def __init__(self, classifier_model):
        self.classifier = classifier_model
        
    def route_query(self, query):
        intent = self.classifier.predict(query)
        
        routing_rules = {
            "exact_match": {"engine": "keyword", "boost": 1.0},
            "semantic": {"engine": "vector", "boost": 1.0},
            "complex": {"engine": "hybrid", "alpha": 0.6}
        }
        
        return routing_rules.get(intent, routing_rules["hybrid"])
```

#### 3. Multi-Stage Retrieval
```python
def multi_stage_search(query, engines):
    # Stage 1: Fast keyword retrieval (large candidate set)
    candidates = engines["keyword"].search(query, top_k=1000)
    
    # Stage 2: Vector reranking (small precise set)
    reranked = engines["vector"].rerank(query, candidates, top_k=50)
    
    # Stage 3: Cross-encoder final ranking
    final_results = engines["cross_encoder"].rerank(query, reranked, top_k=10)
    
    return final_results
```

### Fusion Strategies

#### Reciprocal Rank Fusion (RRF)
```python
def reciprocal_rank_fusion(rankings, k=60):
    fused_scores = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1.0 / (position + k)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```

#### Weighted Score Fusion
```python
def weighted_score_fusion(keyword_results, vector_results, alpha=0.5):
    # Normalize scores to [0,1]
    keyword_scores = normalize_scores(keyword_results)
    vector_scores = normalize_scores(vector_results)
    
    fused_scores = {}
    all_docs = set(keyword_scores.keys()) | set(vector_scores.keys())
    
    for doc_id in all_docs:
        kw_score = keyword_scores.get(doc_id, 0)
        vec_score = vector_scores.get(doc_id, 0)
        fused_scores[doc_id] = alpha * kw_score + (1 - alpha) * vec_score
    
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```

### Configuration Optimization
```python
hybrid_config = {
    "fusion_method": "rrf",  # "rrf", "weighted", "learned"
    "alpha": 0.6,           # Keyword weight (0.5-0.7 typical)
    "retrieval_sizes": {
        "keyword": 500,
        "vector": 500,
        "final": 50
    },
    "query_routing": {
        "enable": True,
        "exact_threshold": 0.8,
        "semantic_threshold": 0.3
    }
}
```

### Best Use Cases
- **E-commerce search**: Combine exact product matches with similar items
- **Enterprise search**: Handle both specific lookups and exploratory queries
- **Customer support**: Find exact procedures and related context
- **Academic search**: Precise citations + conceptually related papers

### Performance Expectations
| Metric | Typical Range | Optimization Target |
|--------|---------------|-------------------|
| Latency | 50-200ms | < 100ms |
| Precision@5 | 0.7-0.9 | > 0.8 |
| Recall | 0.8-0.95 | > 0.9 |
| nDCG@10 | 0.75-0.9 | > 0.85 |

## 4. Sparse Vector Indexing (SPLADE, ColBERT)

### SPLADE (Sparse Lexical and Expansion)
```python
class SPLADEEncoder:
    def __init__(self, model_name="naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    def encode(self, texts):
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            
        # Get sparse representation
        sparse_vectors = torch.log(1 + torch.relu(outputs.logits)) * tokens.attention_mask.unsqueeze(-1)
        
        return sparse_vectors
```

### ColBERT (Efficient Multi-Vector)
```python
class ColBERTEngine:
    def __init__(self, checkpoint_path):
        self.searcher = Searcher(index=checkpoint_path)
    
    def search(self, query, k=10):
        results = self.searcher.search(query, k=k)
        return [(doc_id, score) for doc_id, score in results]
```

### Advantages of Sparse Vectors
- ✅ **Interpretable**: Can see which terms contribute to similarity
- ✅ **Expansion**: Automatically expands queries with related terms
- ✅ **Efficient**: Leverages existing inverted index infrastructure
- ✅ **Quality**: Often better than dense vectors for many tasks

### Best Use Cases
- **Academic search**: Term expansion for research queries
- **Legal search**: Interpretable relevance for compliance
- **Medical search**: Terminology expansion with safety requirements

## 🎯 Strategy Selection Framework

### Decision Tree
```
Query Type?
├── Exact Match Needed
│   ├── High Volume → Keyword (BM25)
│   └── Semantic Fallback → Hybrid (Keyword Primary)
├── Semantic Understanding Critical
│   ├── Multilingual → Vector (Dense)
│   ├── Interpretability Required → Sparse Vector
│   └── Best Quality → Hybrid
└── Mixed Requirements
    ├── Budget Constrained → Keyword + Query Expansion
    ├── Performance Critical → Hybrid (Optimized)
    └── Maximum Quality → Multi-Stage Hybrid
```

### Implementation Roadmap

#### Phase 1: Baseline (Week 1-2)
1. Implement BM25 keyword search
2. Establish evaluation metrics and golden dataset
3. Measure baseline performance

#### Phase 2: Vector Search (Week 3-4)
1. Choose embedding model and vector database
2. Implement dense vector search
3. Compare with keyword baseline

#### Phase 3: Hybrid Optimization (Week 5-8)
1. Implement fusion strategies
2. Optimize alpha parameters
3. Add query routing logic
4. Comprehensive evaluation

#### Phase 4: Production (Week 9-12)
1. A/B testing with live traffic
2. Monitoring and alerting setup
3. Performance optimization
4. Feedback loop implementation

---
*Next: [Advanced Techniques](./04-advanced-techniques.md)*