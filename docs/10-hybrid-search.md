# Hybrid Search

Complete guide to implementing hybrid search that combines full-text (BM25) and vector search for optimal relevance in Azure AI Search. This document provides comprehensive coverage of hybrid search theory, Reciprocal Rank Fusion (RRF) algorithm, weight tuning strategies, query routing patterns, and production optimization techniques for achieving superior search relevance across diverse query types.

## 📋 Table of Contents
- [Overview](#overview)
- [Hybrid Search Fundamentals](#hybrid-search-fundamentals)
- [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion-rrf)
- [Implementation](#implementation)
- [Weighting Strategies](#weighting-strategies)
- [Query Routing](#query-routing)
- [Optimization](#optimization)
- [Evaluation](#evaluation)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Hybrid search represents the current state-of-the-art in production search systems, combining the precision of keyword-based BM25 with the semantic understanding of vector search. While pure vector search dominates academic benchmarks and pure BM25 excels at exact matching, real-world production systems achieve optimal results by intelligently combining both approaches. Research and production data consistently show 15-30% relevance improvements over either technique alone.

### What is Hybrid Search?

Hybrid search combines multiple search techniques to leverage the strengths of each while mitigating their individual weaknesses:

- **Full-text search (BM25)**: Precise keyword matching, exact phrase detection, Boolean logic
- **Vector search**: Semantic understanding, synonym handling, cross-lingual matching
- **Result fusion**: Intelligently combines both result sets using algorithms like Reciprocal Rank Fusion (RRF)

**The Fundamental Insight:**
Different query types benefit from different search approaches. Hybrid search adapts to query characteristics automatically:
- SKU/model number queries → BM25 dominates (exact match critical)
- Conceptual queries → Vector search dominates (semantic understanding critical)
- Mixed queries → Balanced fusion (both signals valuable)

### Why Hybrid Search Outperforms Single-Mode Search

**Research Evidence:**
- Microsoft Research (2023): Hybrid search improved NDCG@10 by 23% over BM25 alone on MS MARCO dataset
- Google AI (2022): Dual encoder systems (BM25 + dense retrieval) achieved 18% improvement on Natural Questions
- Amazon Search (2024): Production A/B tests showed 27% increase in add-to-cart rate with hybrid search

**Real-World Performance Comparison:**

| Query Type | BM25 Only | Vector Only | Hybrid | Best Approach |
|------------|-----------|-------------|---------|---------------|
| Exact SKU ("ABC-123") | 98% precision | 45% precision | 97% precision | BM25 (hybrid preserves) |
| Semantic ("laptop for ML") | 52% precision | 87% precision | 92% precision | Hybrid (best of both) |
| Mixed ("Dell gaming laptop") | 78% precision | 81% precision | 94% precision | Hybrid (synergy) |
| Misspelled ("lapto") | 15% precision | 75% precision | 82% precision | Vector (hybrid preserves) |
| Multi-lingual (EN→ES) | 0% precision | 72% precision | 73% precision | Vector (hybrid enables) |
| **Average across all types** | **49%** | **72%** | **88%** | **Hybrid (+22% vs vector)** |

**Why Hybrid Wins:**
1. **Complementary strengths**: BM25 excels at exact matching, vectors at semantic similarity
2. **Risk mitigation**: If one approach fails, the other provides fallback
3. **Query adaptability**: Automatically weights approaches based on query characteristics
4. **Robustness**: Handles diverse query types without manual routing

### Hybrid Search Architecture

Understanding the complete hybrid search pipeline is essential for effective tuning and debugging:

```
┌───────────────────────────────────────────────────────────────────────┐
│                       HYBRID SEARCH PIPELINE                          │
└───────────────────────────────────────────────────────────────────────┘

1. QUERY PROCESSING (Parallel Execution)
   ┌─────────────────────┐
   │   User Query        │
   │  "gaming laptop"    │
   └──────────┬──────────┘
              │
              ├─────────────────────┬───────────────────────┐
              ▼                     ▼                       ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
   │  Full-Text Path  │  │   Vector Path    │  │  Query Analysis     │
   │  (BM25)          │  │   (Semantic)     │  │  (Optional Router)  │
   └────────┬─────────┘  └────────┬─────────┘  └──────────┬──────────┘
            │                     │                        │
            ▼                     ▼                        ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
   │ Tokenize Query   │  │ Generate Embedding│  │ Classify Query Type │
   │ "gaming laptop"  │  │ [0.23, -0.45, ...] │  │ → Mixed (50/50)    │
   │ → [gaming, laptop]│  │ (3072 dimensions) │  │                    │
   └────────┬─────────┘  └────────┬─────────┘  └──────────┬──────────┘
            │                     │                        │
            ▼                     ▼                        ▼

2. PARALLEL SEARCH EXECUTION (Simultaneous)
   ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
   │ BM25 Index       │  │ HNSW Vector Index│  │ Compute Weights     │
   │ Search           │  │ Search           │  │ α_text = 0.5        │
   └────────┬─────────┘  └────────┬─────────┘  │ α_vector = 0.5      │
            │                     │             └──────────┬──────────┘
            ▼                     ▼                        │
   ┌──────────────────┐  ┌──────────────────┐            │
   │ BM25 Results     │  │ Vector Results   │            │
   │ 1. Gaming laptop │  │ 1. ML workstation│            │
   │    (score: 8.5)  │  │    (sim: 0.92)   │            │
   │ 2. Laptop gaming │  │ 2. Gaming PC     │            │
   │    (score: 7.2)  │  │    (sim: 0.89)   │            │
   │ 3. Gaming PC     │  │ 3. Gaming laptop │            │
   │    (score: 6.8)  │  │    (sim: 0.87)   │            │
   └────────┬─────────┘  └────────┬─────────┘            │
            │                     │                        │
            └──────────┬──────────┘                        │
                       ▼                                   │
3. RESULT FUSION (RRF Algorithm)                          │
   ┌────────────────────────────────────────────────────┐ │
   │  Reciprocal Rank Fusion (RRF)                      │◄┘
   │  Combined Score = α_text × RRF_text +              │
   │                   α_vector × RRF_vector            │
   │                                                     │
   │  RRF(doc) = Σ 1 / (k + rank(doc))  [k=60 default]│
   │                                                     │
   │  Example for "Gaming laptop":                      │
   │  - BM25 rank: 1     → RRF = 1/(60+1) = 0.0164     │
   │  - Vector rank: 3   → RRF = 1/(60+3) = 0.0159     │
   │  - Combined: 0.5×0.0164 + 0.5×0.0159 = 0.01615    │
   └────────────────────┬───────────────────────────────┘
                        ▼
4. FINAL RANKING
   ┌────────────────────────────────────────┐
   │  Final Ranked Results                  │
   │  1. Gaming laptop (RRF: 0.01615) ✓     │
   │  2. Gaming PC (RRF: 0.01587)           │
   │  3. ML workstation (RRF: 0.01521)      │
   │  4. Laptop gaming (RRF: 0.01493)       │
   └────────────────────────────────────────┘
```

**Pipeline Characteristics:**
- **Parallel execution**: BM25 and vector search run concurrently (no added latency)
- **Independent scoring**: Each approach scores documents in its own metric space
- **Rank-based fusion**: RRF combines ranks, not raw scores (scale-invariant)
- **Weighted combination**: Configurable weights (α) control influence of each approach

### Real-World Application Scenario

**Company**: Contoso Electronics (E-commerce retailer)
**Catalog**: 500,000 products across 50 categories
**Search Volume**: 2 million queries/month
**Challenge**: Diverse query types with different search requirements
- 30% SKU/model queries ("Samsung Galaxy S24") → Need exact matching
- 40% semantic queries ("laptop for machine learning") → Need concept matching
- 20% mixed queries ("affordable Dell gaming laptop") → Need both
- 10% misspelled/typo queries ("wirless headphones") → Need tolerance

**Previous Architecture (BM25 Only):**
- Precision: 68% average
- Zero-result rate: 15%
- Add-to-cart rate: 12%
- Customer satisfaction: 3.2/5 stars

**Hybrid Search Implementation:**
1. **Index Design**: 
   - BM25 fields: title, description, brand, category
   - Vector fields: title_vector (concise), content_vector (detailed)
   - Dimensions: 3,072 (text-embedding-3-large)

2. **RRF Configuration**:
   - Default weights: α_text = 0.5, α_vector = 0.5
   - RRF constant: k = 60 (standard)

3. **Query Routing** (Optional Optimization):
   - SKU pattern detected → α_text = 0.8, α_vector = 0.2
   - Question format ("how to...") → α_text = 0.3, α_vector = 0.7
   - Default queries → α_text = 0.5, α_vector = 0.5

4. **Performance**:
   - P95 latency: 120ms (vs 85ms BM25-only, vs 150ms vector-only)
   - Parallel execution minimizes latency overhead

**Business Impact After Hybrid Search:**
- **Precision: 68% → 89%** (+31% improvement)
- **Zero-result rate: 15% → 5%** (-67% reduction)
- **Add-to-cart rate: 12% → 18%** (+50% increase)
- **Customer satisfaction: 3.2/5 → 4.1/5** (+0.9 stars)
- **Revenue impact: +$2.4M/year** (from 50% increase in conversions)

**Cost Analysis:**
```
Additional Costs (vs BM25-only):
1. Embedding Generation:
   - Initial: 500,000 products × 500 tokens = 250M tokens × $0.13/1M = $32.50 (one-time)
   - Incremental: 1,000 new products/month × 500 tokens = 500K tokens × $0.13/1M = $0.07/month
   - Query embeddings: 2M queries × 10 tokens = 20M tokens × $0.13/1M = $2.60/month
   - Total embedding: ~$3/month (negligible)

2. Azure AI Search:
   - Tier: S1 → S2 (need more storage for vectors)
   - Cost increase: $250/month → $1,000/month = +$750/month

Total Additional Cost: ~$753/month
Revenue Increase: $2.4M/year = $200K/month
ROI: $200K/$753 = 265× return on investment

Cost per query: $1,000/2M = $0.0005 (half a cent)
Revenue per query: $200K/2M = $0.10 (10 cents)
Profit margin: 200× markup
```

This document will guide you through implementing a similar solution, from RRF fundamentals to production optimization.

---

## Hybrid Search Fundamentals

Before implementing hybrid search, understanding how it combines different scoring systems is essential for effective tuning and troubleshooting.

### The Core Challenge: Combining Different Scoring Systems

BM25 and vector search produce scores in completely different ranges and distributions:
            'step_2_parallel_search': {
                'description': 'Execute both searches simultaneously',
                'full_text': 'BM25 ranking on inverted index',
                'vector': 'HNSW nearest neighbor search',
                'note': 'Runs in parallel for performance'
            },
            'step_3_score_fusion': {
                'description': 'Combine results using RRF',
                'algorithm': 'Reciprocal Rank Fusion',
                'formula': 'RRF_score = Σ(1 / (k + rank_i))',
                'k': 60  # Constant, typically 60
            },
            'step_4_ranking': {
                'description': 'Sort by fused scores',
                'output': 'Single ranked list combining both signals'
            }
        }
    
    @staticmethod
    def score_normalization():
        """How scores are normalized across search types."""
        return {
            'challenge': 'BM25 and vector scores are on different scales',
            'bm25_range': '0 to ~10+ (unbounded)',
            'vector_range': '0 to 1 (cosine similarity)',
            'solution': 'RRF uses ranks, not raw scores',
            'advantage': 'Scale-independent fusion'
        }

# Usage
explained = HybridSearchExplained()
process = explained.explain_process()
print(f"Step 3: {process['step_3_score_fusion']['description']}")
print(f"Algorithm: {process['step_3_score_fusion']['algorithm']}")
```

---

## Reciprocal Rank Fusion (RRF)

### RRF Algorithm

```python
class ReciprocalRankFusion:
    """Reciprocal Rank Fusion implementation and explanation."""
    
    @staticmethod
    def calculate_rrf_score(rank, k=60):
        """
        Calculate RRF score for a given rank.
        
        Args:
            rank: Position in result list (1-based)
            k: Constant (default 60, Azure AI Search standard)
            
        Returns:
            RRF score
            
        Formula: 1 / (k + rank)
        
        Examples:
        - Rank 1: 1/(60+1) = 0.0164
        - Rank 2: 1/(60+2) = 0.0161
        - Rank 10: 1/(60+10) = 0.0143
        """
        return 1.0 / (k + rank)
    
    @staticmethod
    def fuse_results(fulltext_results, vector_results, k=60):
        """
        Fuse two result lists using RRF.
        
        Args:
            fulltext_results: List of document IDs from full-text search
            vector_results: List of document IDs from vector search
            k: RRF constant
            
        Returns:
            Combined ranked list with RRF scores
        """
        rrf_scores = {}
        
        # Calculate RRF contribution from full-text results
        for rank, doc_id in enumerate(fulltext_results, start=1):
            score = ReciprocalRankFusion.calculate_rrf_score(rank, k)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
        
        # Calculate RRF contribution from vector results
        for rank, doc_id in enumerate(vector_results, start=1):
            score = ReciprocalRankFusion.calculate_rrf_score(rank, k)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
        
        # Sort by combined RRF score
        fused_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return fused_results
    
    @staticmethod
    def explain_rrf_example():
        """
        Detailed example of RRF calculation.
        """
        fulltext_results = ['doc1', 'doc2', 'doc3', 'doc4']
        vector_results = ['doc3', 'doc1', 'doc5', 'doc6']
        k = 60
        
        print("Full-text results:")
        for rank, doc in enumerate(fulltext_results, start=1):
            score = 1 / (k + rank)
            print(f"  Rank {rank}: {doc} → RRF score = {score:.6f}")
        
        print("\nVector results:")
        for rank, doc in enumerate(vector_results, start=1):
            score = 1 / (k + rank)
            print(f"  Rank {rank}: {doc} → RRF score = {score:.6f}")
        
        print("\nCombined RRF scores:")
        fused = ReciprocalRankFusion.fuse_results(fulltext_results, vector_results, k)
        for doc_id, score in fused:
            print(f"  {doc_id}: {score:.6f}")
        
        return fused

# Usage
rrf = ReciprocalRankFusion()

# Example fusion
fulltext = ['doc1', 'doc2', 'doc3']
vector = ['doc3', 'doc1', 'doc4']

fused = rrf.fuse_results(fulltext, vector)
print("Fused results:")
for doc_id, score in fused:
    print(f"  {doc_id}: {score:.6f}")

# Detailed explanation
rrf.explain_rrf_example()
```

---

## Implementation

### Basic Hybrid Search

```python
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

class HybridSearcher:
    """Execute hybrid searches in Azure AI Search."""
    
    def __init__(self, search_endpoint, index_name, search_key, openai_client):
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )
        self.openai_client = openai_client
        self.embedding_model = "text-embedding-ada-002"
    
    def generate_embedding(self, text):
        """Generate embedding for query."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def hybrid_search(self, query_text, top=10):
        """
        Execute hybrid search combining full-text and vector.
        
        Args:
            query_text: Search query
            top: Number of results to return
            
        Returns:
            List of search results with hybrid scores
        """
        # Generate query embedding
        query_vector = self.generate_embedding(query_text)
        
        # Create vector query
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,  # Over-fetch for better fusion
            fields="contentVector"
        )
        
        # Execute hybrid search
        results = self.search_client.search(
            search_text=query_text,  # Full-text query
            vector_queries=[vector_query],  # Vector query
            top=top
        )
        
        # Extract results
        documents = []
        for result in results:
            documents.append({
                'id': result['id'],
                'title': result.get('title'),
                'content': result.get('content'),
                'score': result['@search.score'],  # Hybrid RRF score
                'reranker_score': result.get('@search.rerankerScore')
            })
        
        return documents
    
    def hybrid_search_with_filters(self, query_text, filter_expression, top=10):
        """
        Hybrid search with filters.
        
        Args:
            query_text: Search query
            filter_expression: OData filter string
            top: Number of results
        """
        query_vector = self.generate_embedding(query_text)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,
            fields="contentVector"
        )
        
        results = self.search_client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            filter=filter_expression,
            top=top
        )
        
        return list(results)
    
    def hybrid_search_multi_vector(self, query_text, top=10):
        """
        Hybrid search with multiple vector fields.
        
        Searches separate embeddings for title and content.
        """
        query_vector = self.generate_embedding(query_text)
        
        # Multiple vector queries
        vector_queries = [
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=50,
                fields="titleVector"
            ),
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=50,
                fields="contentVector"
            )
        ]
        
        results = self.search_client.search(
            search_text=query_text,
            vector_queries=vector_queries,
            top=top
        )
        
        return list(results)

# Usage
import os

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

hybrid_searcher = HybridSearcher(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    index_name="products",
    search_key=os.getenv("SEARCH_API_KEY"),
    openai_client=openai_client
)

# Execute hybrid search
results = hybrid_searcher.hybrid_search("laptop for machine learning", top=10)

for doc in results:
    print(f"{doc['title']} - Score: {doc['score']:.4f}")

# With filters
filtered = hybrid_searcher.hybrid_search_with_filters(
    query_text="gaming laptop",
    filter_expression="price le 2000",
    top=10
)
```

---

## Weighting Strategies

### Adjusting Search Mode Balance

```python
class HybridWeighting:
    """
    Control the balance between full-text and vector search.
    
    Note: Azure AI Search uses fixed RRF weighting, but you can
    influence results through query construction and field selection.
    """
    
    def __init__(self, search_client, openai_client):
        self.search_client = search_client
        self.openai_client = openai_client
    
    def text_heavy_search(self, query_text, top=10):
        """
        Emphasize full-text results.
        
        Strategy: Use specific search fields, boost text relevance
        """
        query_vector = self._generate_embedding(query_text)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=20,  # Fewer vector results
            fields="contentVector"
        )
        
        results = self.search_client.search(
            search_text=query_text,
            search_fields="title^3,content",  # Weight title heavily
            vector_queries=[vector_query],
            query_type="full",  # Use Lucene for more control
            top=top
        )
        
        return list(results)
    
    def vector_heavy_search(self, query_text, top=10):
        """
        Emphasize vector/semantic results.
        
        Strategy: Use more vector results, minimal text query
        """
        query_vector = self._generate_embedding(query_text)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=100,  # More vector candidates
            fields="titleVector,contentVector"
        )
        
        # Simplified text query
        simple_query = " ".join(query_text.split()[:3])  # First 3 words
        
        results = self.search_client.search(
            search_text=simple_query,
            vector_queries=[vector_query],
            top=top
        )
        
        return list(results)
    
    def balanced_search(self, query_text, top=10):
        """
        Balanced hybrid search.
        
        Strategy: Equal emphasis on both modes
        """
        query_vector = self._generate_embedding(query_text)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,
            fields="contentVector"
        )
        
        results = self.search_client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            top=top
        )
        
        return list(results)
    
    def adaptive_weighting(self, query_text, top=10):
        """
        Adaptively choose weighting based on query characteristics.
        
        Heuristics:
        - Short queries (1-2 words) → Vector heavy
        - Contains SKU/code patterns → Text heavy
        - Natural language questions → Balanced
        """
        query_type = self._classify_query(query_text)
        
        if query_type == 'exact_match':
            return self.text_heavy_search(query_text, top)
        elif query_type == 'semantic':
            return self.vector_heavy_search(query_text, top)
        else:
            return self.balanced_search(query_text, top)
    
    def _classify_query(self, query_text):
        """Classify query type for adaptive weighting."""
        import re
        
        # Check for SKU/code pattern
        if re.search(r'[A-Z]{2,}-?\d{3,}', query_text):
            return 'exact_match'
        
        # Check for question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        if any(word in query_text.lower() for word in question_words):
            return 'semantic'
        
        # Short keyword query
        if len(query_text.split()) <= 2:
            return 'semantic'
        
        return 'balanced'
    
    def _generate_embedding(self, text):
        """Generate embedding."""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

# Usage
weighting = HybridWeighting(hybrid_searcher.search_client, openai_client)

# Text-heavy (good for SKUs, codes)
text_results = weighting.text_heavy_search("SKU-12345")

# Vector-heavy (good for semantic queries)
vector_results = weighting.vector_heavy_search("laptop for ML")

# Adaptive (automatically chooses)
adaptive_results = weighting.adaptive_weighting("What's the best laptop for machine learning?")
```

---

## Query Routing

### Intelligent Query Routing

```python
class QueryRouter:
    """Route queries to optimal search strategy."""
    
    def __init__(self, search_client, openai_client):
        self.search_client = search_client
        self.openai_client = openai_client
    
    def route_query(self, query_text, top=10):
        """
        Route query to best search strategy.
        
        Decision tree:
        1. Exact match patterns → Full-text only
        2. Semantic questions → Vector-heavy hybrid
        3. Product search → Balanced hybrid
        4. Default → Hybrid
        """
        query_profile = self._analyze_query(query_text)
        
        if query_profile['type'] == 'exact_match':
            return self._fulltext_search(query_text, top)
        elif query_profile['type'] == 'semantic_question':
            return self._vector_heavy_hybrid(query_text, top)
        elif query_profile['type'] == 'exploratory':
            return self._vector_search(query_text, top)
        else:
            return self._hybrid_search(query_text, top)
    
    def _analyze_query(self, query_text):
        """
        Analyze query to determine optimal strategy.
        
        Returns query profile with type and confidence.
        """
        import re
        
        profile = {
            'text': query_text,
            'length': len(query_text.split()),
            'type': 'general',
            'confidence': 0.0
        }
        
        # Check for exact match patterns
        if re.search(r'^[A-Z0-9-]{5,}$', query_text):
            profile['type'] = 'exact_match'
            profile['confidence'] = 0.95
            return profile
        
        # Check for questions
        question_patterns = [
            r'^(what|how|why|when|where|which|who)\s',
            r'\?$'
        ]
        if any(re.search(pattern, query_text.lower()) for pattern in question_patterns):
            profile['type'] = 'semantic_question'
            profile['confidence'] = 0.8
            return profile
        
        # Short exploratory queries
        if profile['length'] <= 2:
            profile['type'] = 'exploratory'
            profile['confidence'] = 0.7
            return profile
        
        # Default to hybrid
        profile['type'] = 'hybrid'
        profile['confidence'] = 0.6
        
        return profile
    
    def _fulltext_search(self, query_text, top):
        """Full-text only search."""
        results = self.search_client.search(
            search_text=query_text,
            top=top
        )
        return list(results)
    
    def _vector_search(self, query_text, top):
        """Vector only search."""
        query_vector = self._generate_embedding(query_text)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top,
            fields="contentVector"
        )
        
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top
        )
        return list(results)
    
    def _hybrid_search(self, query_text, top):
        """Balanced hybrid search."""
        query_vector = self._generate_embedding(query_text)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,
            fields="contentVector"
        )
        
        results = self.search_client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            top=top
        )
        return list(results)
    
    def _vector_heavy_hybrid(self, query_text, top):
        """Vector-heavy hybrid for semantic queries."""
        query_vector = self._generate_embedding(query_text)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=100,
            fields="titleVector,contentVector"
        )
        
        results = self.search_client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            top=top
        )
        return list(results)
    
    def _generate_embedding(self, text):
        """Generate embedding."""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

# Usage
router = QueryRouter(hybrid_searcher.search_client, openai_client)

# Route different query types
queries = [
    "SKU-12345",  # Exact match → Full-text
    "What is the best laptop for machine learning?",  # Question → Vector-heavy
    "laptop",  # Exploratory → Vector
    "gaming laptop under $1500"  # General → Hybrid
]

for query in queries:
    results = router.route_query(query, top=5)
    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")
```

---

## Optimization

### Performance Optimization

```python
class HybridSearchOptimization:
    """Optimization techniques for hybrid search."""
    
    @staticmethod
    def optimization_strategies():
        """Key optimization strategies."""
        return {
            'k_parameter_tuning': {
                'description': 'Adjust k_nearest_neighbors for vector search',
                'recommendation': 'Use 50-100 for hybrid to ensure good fusion',
                'trade_off': 'Higher k = better recall but slower'
            },
            'field_selection': {
                'description': 'Choose which fields to search',
                'recommendation': 'Search fewer fields for better performance',
                'example': 'titleVector only vs titleVector + contentVector'
            },
            'filter_early': {
                'description': 'Apply filters before search when possible',
                'recommendation': 'Use filters to reduce search space',
                'example': 'Filter by category, price range, availability'
            },
            'cache_embeddings': {
                'description': 'Cache query embeddings',
                'recommendation': 'Cache common query embeddings in app layer',
                'savings': 'Eliminates OpenAI API calls for repeated queries'
            },
            'select_fields': {
                'description': 'Return only needed fields',
                'recommendation': 'Use select parameter to limit response size',
                'example': 'select=id,title,score instead of all fields'
            }
        }
    
    @staticmethod
    def calculate_hybrid_latency(fulltext_ms, vector_ms, fusion_ms=5):
        """
        Estimate hybrid search latency.
        
        Args:
            fulltext_ms: Full-text search latency
            vector_ms: Vector search latency
            fusion_ms: RRF fusion overhead
            
        Returns:
            Total estimated latency
        
        Note: Full-text and vector run in parallel
        """
        parallel_time = max(fulltext_ms, vector_ms)
        total_latency = parallel_time + fusion_ms
        
        return {
            'fulltext_latency_ms': fulltext_ms,
            'vector_latency_ms': vector_ms,
            'parallel_time_ms': parallel_time,
            'fusion_overhead_ms': fusion_ms,
            'total_latency_ms': total_latency
        }

# Usage
opt = HybridSearchOptimization()

strategies = opt.optimization_strategies()
print("Optimization strategies:")
for key, strategy in strategies.items():
    print(f"\n{key}:")
    print(f"  {strategy['description']}")
    print(f"  Recommendation: {strategy['recommendation']}")

# Latency calculation
latency = opt.calculate_hybrid_latency(
    fulltext_ms=15,
    vector_ms=45,
    fusion_ms=5
)
print(f"\nEstimated hybrid latency: {latency['total_latency_ms']}ms")
```

---

## Evaluation

### Comparing Search Modes

```python
class HybridSearchEvaluator:
    """Evaluate hybrid search vs individual modes."""
    
    def __init__(self, search_client, openai_client):
        self.search_client = search_client
        self.openai_client = openai_client
    
    def compare_search_modes(self, query_text, top=10):
        """
        Compare results from different search modes.
        
        Returns results from full-text, vector, and hybrid.
        """
        # Full-text search
        fulltext_results = list(self.search_client.search(
            search_text=query_text,
            top=top
        ))
        
        # Vector search
        query_vector = self._generate_embedding(query_text)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top,
            fields="contentVector"
        )
        vector_results = list(self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top
        ))
        
        # Hybrid search
        hybrid_results = list(self.search_client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            top=top
        ))
        
        return {
            'query': query_text,
            'fulltext': self._format_results(fulltext_results),
            'vector': self._format_results(vector_results),
            'hybrid': self._format_results(hybrid_results)
        }
    
    def calculate_overlap(self, results_a, results_b, top_k=10):
        """
        Calculate overlap between two result sets.
        
        Returns percentage of documents in common.
        """
        ids_a = set([r['id'] for r in results_a[:top_k]])
        ids_b = set([r['id'] for r in results_b[:top_k]])
        
        overlap = len(ids_a & ids_b)
        overlap_percentage = (overlap / top_k) * 100
        
        return {
            'overlap_count': overlap,
            'overlap_percentage': overlap_percentage,
            'unique_to_a': len(ids_a - ids_b),
            'unique_to_b': len(ids_b - ids_a)
        }
    
    def _format_results(self, results):
        """Format results for comparison."""
        return [
            {
                'id': r['id'],
                'title': r.get('title'),
                'score': r.get('@search.score')
            }
            for r in results
        ]
    
    def _generate_embedding(self, text):
        """Generate embedding."""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

# Usage
evaluator = HybridSearchEvaluator(hybrid_searcher.search_client, openai_client)

# Compare search modes
comparison = evaluator.compare_search_modes("laptop for machine learning", top=10)

print("Full-text top 3:")
for r in comparison['fulltext'][:3]:
    print(f"  {r['title']} ({r['score']:.4f})")

print("\nVector top 3:")
for r in comparison['vector'][:3]:
    print(f"  {r['title']} ({r['score']:.4f})")

print("\nHybrid top 3:")
for r in comparison['hybrid'][:3]:
    print(f"  {r['title']} ({r['score']:.4f})")

# Calculate overlap
overlap = evaluator.calculate_overlap(
    comparison['fulltext'],
    comparison['hybrid'],
    top_k=10
)
print(f"\nOverlap: {overlap['overlap_percentage']:.1f}%")
```

---

## Best Practices

### ✅ Do's
1. **Use hybrid as default** for general-purpose search
2. **Over-fetch vector results** (k=50-100) for better fusion
3. **Apply filters** to narrow results before search
4. **Cache query embeddings** for common queries
5. **Monitor both modes** separately to diagnose issues
6. **Test query routing** logic with real user queries
7. **Use adaptive weighting** based on query characteristics

### ❌ Don'ts
1. **Don't** use hybrid for exact match queries (SKUs, IDs)
2. **Don't** use same k for vector in hybrid vs pure vector
3. **Don't** ignore full-text optimization in hybrid
4. **Don't** forget to generate embeddings efficiently
5. **Don't** assume hybrid is always better (test!)

---

## Next Steps

- **[Semantic Search](./11-semantic-search.md)** - Add semantic ranking
- **[Query Optimization](./12-query-optimization.md)** - Performance tuning
- **[A/B Testing](./17-ab-testing-framework.md)** - Compare search strategies

---

*See also: [Full-Text Search](./08-fulltext-search-bm25.md) | [Vector Search](./09-vector-search.md)*