# Hybrid Search - Part 3: Troubleshooting Guide

## Introduction

This guide addresses common issues encountered when implementing hybrid search (BM25 + vector embeddings) with Azure AI Search. Each issue includes symptoms, root causes, diagnostic tools, solutions, and prevention strategies.

---

## Issue 1: Vector Results Dominating RRF Fusion

### Symptoms

- BM25 results rarely appear in top 10 positions
- Exact SKU/model number queries show poor precision
- Score distribution heavily skewed toward vector matches
- Users complain about "irrelevant results for specific product searches"

**Example:**
```
Query: "Dell XPS 9520"
Expected: Exact product match at #1
Actual: Similar products at #1-8, exact match at #9
```

### Root Causes

1. **Vector k-value too high**: Over-fetching vectors (k=200) floods RRF with vector results
2. **Poor query classification**: Exact queries routed to vector-heavy weights
3. **Low BM25 result count**: BM25 returns <10 results, vector dominates fusion
4. **Embedding captures fuzzy similarity**: "XPS 9520" embeds similarly to "XPS 9530", "XPS 9510"

### Diagnostic Tools

```python
class HybridSearchDiagnostics:
    """Diagnose hybrid search result composition."""
    
    def analyze_result_sources(
        self,
        query: str,
        top: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze which search mode contributes which results.
        
        Returns:
            Analysis showing BM25 vs vector contribution to final results
        """
        # Execute BM25-only search
        bm25_results = self.search_client.search(
            search_text=query,
            top=top
        )
        bm25_ids = {r['id']: rank for rank, r in enumerate(bm25_results, 1)}
        
        # Execute vector-only search
        query_vector = self.embedding_function(query)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top,
            fields="titleVector"
        )
        vector_results = self.search_client.search(
            vector_queries=[vector_query],
            top=top
        )
        vector_ids = {r['id']: rank for rank, r in enumerate(vector_results, 1)}
        
        # Execute hybrid search
        hybrid_results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top
        )
        hybrid_ids = [r['id'] for r in hybrid_results]
        
        # Analyze composition
        analysis = {
            'query': query,
            'bm25_count': len(bm25_ids),
            'vector_count': len(vector_ids),
            'hybrid_top10_sources': [],
            'bm25_in_top10': 0,
            'vector_in_top10': 0,
            'both_in_top10': 0
        }
        
        for rank, doc_id in enumerate(hybrid_ids[:10], 1):
            in_bm25 = doc_id in bm25_ids
            in_vector = doc_id in vector_ids
            
            source = 'both' if (in_bm25 and in_vector) else ('bm25' if in_bm25 else 'vector')
            
            analysis['hybrid_top10_sources'].append({
                'rank': rank,
                'id': doc_id,
                'source': source,
                'bm25_rank': bm25_ids.get(doc_id),
                'vector_rank': vector_ids.get(doc_id)
            })
            
            if source == 'both':
                analysis['both_in_top10'] += 1
            elif source == 'bm25':
                analysis['bm25_in_top10'] += 1
            else:
                analysis['vector_in_top10'] += 1
        
        # Calculate dominance score
        analysis['vector_dominance'] = analysis['vector_in_top10'] / 10.0  # 0-1 score
        analysis['bm25_dominance'] = analysis['bm25_in_top10'] / 10.0
        
        return analysis

# Usage
diagnostics = HybridSearchDiagnostics(search_client, get_embedding)
analysis = diagnostics.analyze_result_sources("Dell XPS 9520", top=20)

print(f"Query: {analysis['query']}")
print(f"Vector dominance: {analysis['vector_dominance']:.1%}")  # 0.8 = 80% vector-only results
print(f"BM25 dominance: {analysis['bm25_dominance']:.1%}")      # 0.1 = 10% BM25-only results
print(f"Both sources: {analysis['both_in_top10']}")             # 1 = 10% from both

print("\nTop 10 composition:")
for item in analysis['hybrid_top10_sources']:
    print(f"  #{item['rank']}: {item['source']:6} (BM25 #{item['bm25_rank']}, Vector #{item['vector_rank']})")

# Example output showing vector dominance:
# Query: Dell XPS 9520
# Vector dominance: 80.0%
# BM25 dominance: 10.0%
# Both sources: 1
#
# Top 10 composition:
#   #1: vector (BM25 #None, Vector #1)
#   #2: vector (BM25 #None, Vector #2)
#   #3: vector (BM25 #None, Vector #3)
#   #4: both   (BM25 #1, Vector #5)     ← Exact match at #4, should be #1
#   #5: vector (BM25 #None, Vector #4)
#   ...
```

### Solutions

#### Solution 1: Reduce Vector k-value

**Reduce over-fetching to balance RRF fusion:**

```python
# Before: Over-fetching vectors
vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=200,  # Too many vector results
    fields="titleVector"
)

# After: Balanced fetching
vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=50,  # Recommended: 50-100
    fields="titleVector"
)
```

**Impact**: Reduces vector dominance from 80% → 40% (balanced fusion).

#### Solution 2: Improve Query Classification

**Better routing for exact queries:**

```python
class ImprovedQueryClassifier:
    """Enhanced classifier to catch exact product queries."""
    
    def classify(self, query: str) -> Tuple[str, Dict[str, float]]:
        """Classify with better exact-match detection."""
        
        # Priority 1: Exact SKU patterns
        if self._is_exact_sku(query):
            return 'exact_sku', {'alpha_text': 1.0, 'alpha_vector': 0.0}
        
        # Priority 2: Brand + Model patterns
        if self._is_brand_model(query):
            return 'brand_model', {'alpha_text': 0.8, 'alpha_vector': 0.2}  # Increased from 0.7
        
        # Priority 3: Contains specific numbers/codes
        if self._contains_product_codes(query):
            return 'product_code', {'alpha_text': 0.7, 'alpha_vector': 0.3}
        
        # Default: Balanced
        return 'general', {'alpha_text': 0.5, 'alpha_vector': 0.5}
    
    def _is_exact_sku(self, query: str) -> bool:
        """Enhanced SKU detection."""
        patterns = [
            r'^[A-Z]{2,4}-\d{4,6}$',       # ABC-1234
            r'^[A-Z]{2,4}\d{4,6}$',        # ABC1234
            r'^\d{2,4}-[A-Z]{2,4}-\d{3,6}$'  # 12-ABC-456
        ]
        return any(re.match(p, query.upper()) for p in patterns)
    
    def _is_brand_model(self, query: str) -> bool:
        """Detect brand + model patterns."""
        # Dell XPS 9520, MacBook Pro M3, Samsung Galaxy S24
        brand_model_pattern = r'^(Dell|HP|Lenovo|Apple|MacBook|Samsung|LG|Sony)\s+[A-Za-z]+\s*[A-Z0-9]{2,}$'
        return bool(re.match(brand_model_pattern, query, re.IGNORECASE))
    
    def _contains_product_codes(self, query: str) -> bool:
        """Check if query contains product-like codes."""
        # Contains patterns like M3, i7, RTX3080, etc.
        code_patterns = [
            r'[Mm]\d',           # M1, M2, M3
            r'i[357]',           # i3, i5, i7
            r'RTX\d{4}',         # RTX3080
            r'\d{4,6}[A-Z]'      # 9520M
        ]
        return any(re.search(p, query) for p in code_patterns)

# Impact: Exact queries now get 80-100% BM25 weighting
```

#### Solution 3: Boost BM25 Scores with Field Weighting

**Use Azure AI Search scoring profiles to boost BM25:**

```json
{
  "scoringProfiles": [
    {
      "name": "boost-exact-matches",
      "text": {
        "weights": {
          "sku": 5.0,
          "model_number": 4.0,
          "title": 2.0,
          "brand": 1.5,
          "description": 1.0
        }
      }
    }
  ],
  "defaultScoringProfile": "boost-exact-matches"
}
```

**Impact**: Exact matches in SKU/model fields get 4-5x boost, improving RRF ranking.

### Prevention

✅ **DO:**
- Use k_nearest_neighbors=50-100 (not 200+)
- Implement query classification with 85%+ accuracy
- Monitor vector dominance metric (target: 30-50% for balanced corpus)
- Use field boosting for exact-match fields (SKU, model number)

❌ **DON'T:**
- Over-fetch vectors (k=200) - causes dominance
- Use same weights for all queries (exact queries need high BM25 weight)
- Ignore query type distribution (monitor in production)

---

## Issue 2: Poor Fusion Results (Worse than Single-Mode)

### Symptoms

- Hybrid search precision < BM25-only OR vector-only
- Top results are "worst of both worlds" (neither BM25 nor vector top picks)
- User satisfaction drops after hybrid deployment
- A/B test shows degradation vs BM25 baseline

**Example:**
```
Query: "best budget laptop"
BM25-only precision: 78%
Vector-only precision: 82%
Hybrid precision: 71% ← Worse than both!
```

### Root Causes

1. **RRF fusing poor-quality results**: BM25 and vector both return low-quality results, RRF combines them
2. **Query mismatch**: Query type doesn't match index content (e.g., conversational query on spec-heavy index)
3. **Embedding quality issues**: Low-quality embeddings cause poor vector results
4. **Index field selection**: Wrong fields used for BM25 or vectors

### Diagnostic Tools

```python
class FusionQualityAnalyzer:
    """Diagnose fusion quality issues."""
    
    def compare_modes(
        self,
        query: str,
        relevant_docs: List[str],
        top: int = 10
    ) -> Dict[str, Any]:
        """
        Compare BM25, vector, and hybrid performance.
        
        Args:
            query: Search query
            relevant_docs: List of known relevant document IDs
            top: Number of results to analyze
        """
        results = {}
        
        # 1. BM25-only results
        bm25_results = list(self.search_client.search(
            search_text=query,
            top=top
        ))
        results['bm25'] = {
            'precision': self._calculate_precision(bm25_results, relevant_docs),
            'top_ids': [r['id'] for r in bm25_results]
        }
        
        # 2. Vector-only results
        query_vector = self.embedding_function(query)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top,
            fields="titleVector"
        )
        vector_results = list(self.search_client.search(
            vector_queries=[vector_query],
            top=top
        ))
        results['vector'] = {
            'precision': self._calculate_precision(vector_results, relevant_docs),
            'top_ids': [r['id'] for r in vector_results]
        }
        
        # 3. Hybrid results
        hybrid_results = list(self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top
        ))
        results['hybrid'] = {
            'precision': self._calculate_precision(hybrid_results, relevant_docs),
            'top_ids': [r['id'] for r in hybrid_results]
        }
        
        # 4. Analyze fusion quality
        results['analysis'] = {
            'hybrid_better_than_bm25': results['hybrid']['precision'] > results['bm25']['precision'],
            'hybrid_better_than_vector': results['hybrid']['precision'] > results['vector']['precision'],
            'hybrid_worse_than_both': (
                results['hybrid']['precision'] < results['bm25']['precision'] and
                results['hybrid']['precision'] < results['vector']['precision']
            ),
            'improvement_vs_bm25': results['hybrid']['precision'] - results['bm25']['precision'],
            'improvement_vs_vector': results['hybrid']['precision'] - results['vector']['precision']
        }
        
        return results
    
    def _calculate_precision(self, results: List[Any], relevant_docs: List[str]) -> float:
        """Calculate precision@k."""
        if not results:
            return 0.0
        relevant_count = sum(1 for r in results if r.get('id') in relevant_docs)
        return relevant_count / len(results)

# Usage
analyzer = FusionQualityAnalyzer(search_client, get_embedding)

# Test query with known relevant docs
relevant_docs = ["prod123", "prod456", "prod789"]
comparison = analyzer.compare_modes("best budget laptop", relevant_docs, top=10)

print(f"BM25 precision: {comparison['bm25']['precision']:.1%}")
print(f"Vector precision: {comparison['vector']['precision']:.1%}")
print(f"Hybrid precision: {comparison['hybrid']['precision']:.1%}")
print(f"Hybrid worse than both: {comparison['analysis']['hybrid_worse_than_both']}")

# Example output showing poor fusion:
# BM25 precision: 70.0%
# Vector precision: 80.0%
# Hybrid precision: 65.0% ← Problem!
# Hybrid worse than both: True
```

### Solutions

#### Solution 1: Analyze Embedding Quality

**Check if embeddings capture relevant semantics:**

```python
class EmbeddingQualityChecker:
    """Verify embedding quality for your domain."""
    
    def check_semantic_similarity(
        self,
        query: str,
        expected_similar: List[str],
        expected_dissimilar: List[str]
    ) -> Dict[str, Any]:
        """
        Check if embeddings properly capture semantic similarity.
        
        Args:
            query: Test query
            expected_similar: Phrases that should be similar
            expected_dissimilar: Phrases that should be dissimilar
        """
        query_emb = self.embedding_function(query)
        
        similar_scores = []
        for phrase in expected_similar:
            phrase_emb = self.embedding_function(phrase)
            similarity = self._cosine_similarity(query_emb, phrase_emb)
            similar_scores.append((phrase, similarity))
        
        dissimilar_scores = []
        for phrase in expected_dissimilar:
            phrase_emb = self.embedding_function(phrase)
            similarity = self._cosine_similarity(query_emb, phrase_emb)
            dissimilar_scores.append((phrase, similarity))
        
        # Expected: similar > 0.7, dissimilar < 0.5
        avg_similar = np.mean([s for _, s in similar_scores])
        avg_dissimilar = np.mean([s for _, s in dissimilar_scores])
        
        return {
            'query': query,
            'similar_scores': similar_scores,
            'dissimilar_scores': dissimilar_scores,
            'avg_similar': avg_similar,
            'avg_dissimilar': avg_dissimilar,
            'quality_ok': avg_similar > 0.7 and avg_dissimilar < 0.5
        }
    
    def _cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Usage
checker = EmbeddingQualityChecker(get_embedding)

result = checker.check_semantic_similarity(
    query="laptop for video editing",
    expected_similar=[
        "laptop for content creation",
        "high-performance laptop",
        "workstation laptop"
    ],
    expected_dissimilar=[
        "laptop charger",
        "laptop bag",
        "used laptop"
    ]
)

print(f"Average similarity (should be similar): {result['avg_similar']:.2f}")  # Should be > 0.7
print(f"Average similarity (should be dissimilar): {result['avg_dissimilar']:.2f}")  # Should be < 0.5
print(f"Embedding quality OK: {result['quality_ok']}")
```

**If embeddings are poor (avg_similar < 0.7):**
- Try different embedding model (text-embedding-3-large vs text-embedding-3-small)
- Check if your domain needs fine-tuned embeddings
- Verify embedding field content (garbage in = garbage out)

#### Solution 2: Adjust Vector Over-Fetching (k)

**If RRF is fusing low-quality vector results:**

```python
# Problem: Low k brings in only top vector results (may be poor quality)
vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=10,  # Too few, no room for RRF to improve
    fields="titleVector"
)

# Solution: Increase k for better fusion pool
vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=50,  # Better: RRF can pick best from larger pool
    fields="titleVector"
)
```

**Impact**: Gives RRF more candidates to fuse, often improves precision by 5-10pp.

#### Solution 3: Use Dual-Vector Strategy

**If single vector field doesn't capture all semantics:**

```python
# Single vector (title only) - may miss detailed semantic info
vector_query_title = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=50,
    fields="titleVector"
)

# Dual vector (title + content) - captures both concise and detailed semantics
vector_query_title = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=50,
    fields="titleVector"
)

vector_query_content = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=50,
    fields="contentVector"
)

# Use both in hybrid search
results = search_client.search(
    search_text=query,
    vector_queries=[vector_query_title, vector_query_content],
    top=10
)
```

**Impact**: Dual vectors typically improve precision by 5-15pp vs single vector.

### Prevention

✅ **DO:**
- Test embedding quality before production (similarity checks)
- Use k_nearest_neighbors=50-100 (gives RRF good fusion pool)
- Consider dual-vector strategy (title + content)
- A/B test hybrid vs single-mode before full deployment

❌ **DON'T:**
- Assume hybrid is always better (test on your data!)
- Use k=10 (too few vector candidates for good fusion)
- Skip embedding quality validation

---

## Issue 3: High Latency (>200ms p95)

### Symptoms

- Query latency p95 > 200ms (exceeds SLA)
- User complaints about "slow search"
- Timeout errors during peak traffic
- Latency p99 > 500ms (very poor user experience)

### Root Causes

1. **Uncached embedding generation**: Every query generates embedding (20-50ms penalty)
2. **Large vector dimensions**: 3,072-dim embeddings (text-embedding-3-large) slower than 1,536-dim
3. **HNSW parameters not optimized**: efSearch too high, causing slow vector search
4. **Over-fetching**: k_nearest_neighbors=200 retrieves too many vectors
5. **Synchronous operations**: Sequential BM25 → embedding → vector search

### Diagnostic Tools

```python
import time
from typing import Dict, List

class LatencyProfiler:
    """Profile hybrid search latency breakdown."""
    
    def profile_search(self, query: str) -> Dict[str, float]:
        """
        Profile search latency by component.
        
        Returns:
            Dict with timing for each step (in milliseconds)
        """
        timings = {}
        total_start = time.perf_counter()
        
        # 1. Query classification
        start = time.perf_counter()
        query_type, weights = self.classifier.classify(query)
        timings['classification_ms'] = (time.perf_counter() - start) * 1000
        
        # 2. Embedding generation
        start = time.perf_counter()
        query_vector = self.embedding_function(query)
        timings['embedding_ms'] = (time.perf_counter() - start) * 1000
        
        # 3. Hybrid search execution
        start = time.perf_counter()
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,
            fields="titleVector"
        )
        results = list(self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=10
        ))
        timings['search_ms'] = (time.perf_counter() - start) * 1000
        
        # Total
        timings['total_ms'] = (time.perf_counter() - total_start) * 1000
        
        # Breakdown estimate (Azure Search internal)
        # Azure parallelizes BM25 and vector, so search_ms ≈ max(bm25, vector) + fusion
        timings['estimated_bm25_ms'] = timings['search_ms'] * 0.4  # Typically 40% of search time
        timings['estimated_vector_ms'] = timings['search_ms'] * 0.5  # Typically 50% of search time
        timings['estimated_fusion_ms'] = timings['search_ms'] * 0.1  # Typically 10% of search time
        
        return timings
    
    def profile_batch(self, queries: List[str], iterations: int = 10) -> Dict[str, Any]:
        """Profile multiple queries and calculate statistics."""
        all_timings = []
        
        for query in queries:
            for _ in range(iterations):
                timings = self.profile_search(query)
                all_timings.append(timings)
        
        # Calculate percentiles
        total_times = [t['total_ms'] for t in all_timings]
        embedding_times = [t['embedding_ms'] for t in all_timings]
        search_times = [t['search_ms'] for t in all_timings]
        
        return {
            'total_p50': np.percentile(total_times, 50),
            'total_p95': np.percentile(total_times, 95),
            'total_p99': np.percentile(total_times, 99),
            'embedding_p50': np.percentile(embedding_times, 50),
            'embedding_p95': np.percentile(embedding_times, 95),
            'search_p50': np.percentile(search_times, 50),
            'search_p95': np.percentile(search_times, 95),
            'sample_breakdown': all_timings[0]  # Show one example
        }

# Usage
profiler = LatencyProfiler(search_client, get_embedding, QueryClassifier())

test_queries = [
    "laptop for video editing",
    "Dell XPS 9520",
    "best budget smartphone"
]

stats = profiler.profile_batch(test_queries, iterations=10)

print("Latency Statistics:")
print(f"  Total p50: {stats['total_p50']:.1f}ms")
print(f"  Total p95: {stats['total_p95']:.1f}ms")
print(f"  Total p99: {stats['total_p99']:.1f}ms")
print(f"\nEmbedding p95: {stats['embedding_p95']:.1f}ms")
print(f"Search p95: {stats['search_p95']:.1f}ms")

print(f"\nSample breakdown:")
for key, value in stats['sample_breakdown'].items():
    print(f"  {key}: {value:.1f}ms")

# Example output showing high latency:
# Latency Statistics:
#   Total p50: 145.2ms
#   Total p95: 287.5ms ← Exceeds 200ms SLA!
#   Total p99: 421.8ms
#
# Embedding p95: 52.3ms ← High (uncached)
# Search p95: 189.4ms
#
# Sample breakdown:
#   classification_ms: 0.8ms
#   embedding_ms: 48.2ms ← Slow
#   search_ms: 165.3ms
#   total_ms: 214.3ms
```

### Solutions

#### Solution 1: Implement Embedding Caching

**Cache frequent query embeddings (see Best Practices section for full implementation):**

```python
# Impact of caching:
# Before: embedding_p95 = 52ms
# After (70% hit rate): embedding_p95 = 0.3 * 52ms + 0.7 * 2ms = 17ms
# Savings: 35ms (67% reduction in embedding time)
```

#### Solution 2: Optimize HNSW Parameters

**Tune efSearch for latency/quality trade-off:**

```json
{
  "vectorSearch": {
    "algorithms": [
      {
        "name": "hnsw-config",
        "kind": "hnsw",
        "hnswParameters": {
          "m": 4,
          "efConstruction": 400,
          "efSearch": 500  // Default: High quality, slower
        }
      }
    ]
  }
}
```

**Latency vs Quality Trade-off:**

| efSearch | Vector Search Latency | Recall@10 | Recommendation |
|----------|----------------------|-----------|----------------|
| 200 | ~15ms | 0.92 | Too low quality |
| 300 | ~25ms | 0.96 | ✓ Good balance for speed |
| 500 | ~40ms | 0.98 | ✓ Default (quality) |
| 700 | ~58ms | 0.99 | Diminishing returns |

**Recommendation**: Use efSearch=300-400 for latency-sensitive applications (saves 15-20ms with minimal quality loss).

#### Solution 3: Reduce k_nearest_neighbors

```python
# Before: Over-fetching
vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=200,  # Slow: Retrieves 200 vectors
    fields="titleVector"
)
# Vector search time: ~65ms

# After: Balanced fetching
vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=50,  # Faster: Retrieves 50 vectors
    fields="titleVector"
)
# Vector search time: ~40ms
# Savings: 25ms (38% faster)
```

#### Solution 4: Use Smaller Embedding Model

**Trade-off: Latency vs quality:**

| Model | Dimensions | Embedding Latency | Vector Search Latency | Quality (MTEB) |
|-------|------------|-------------------|----------------------|----------------|
| text-embedding-3-large | 3,072 | 45ms | 40ms | 64.6 (best) |
| text-embedding-3-small | 1,536 | 35ms | 28ms | 62.3 (good) |

**Savings**: text-embedding-3-small saves ~25ms total (45→35 embedding + 40→28 search).

**Decision**: Use text-embedding-3-small if latency SLA is tight and quality loss (<5%) is acceptable.

### Prevention

✅ **DO:**
- Cache embeddings (70% hit rate = 35ms savings)
- Use efSearch=300-400 for latency-sensitive scenarios
- Set k_nearest_neighbors=50-100 (not 200)
- Monitor p95/p99 latency, not just average
- Consider text-embedding-3-small for very tight latency budgets

❌ **DON'T:**
- Generate embeddings for every query (expensive)
- Use efSearch=700+ (diminishing quality returns, high latency)
- Over-fetch vectors (k=200) unless necessary
- Ignore tail latency (p99 matters for user experience)

---

## Issue 4: Score Distribution Imbalance

### Symptoms

- One search mode (BM25 or vector) consistently dominates results
- Hybrid results identical to BM25-only or vector-only
- RRF fusion appears to have no effect
- Query classification not influencing results as expected

### Diagnostic Tools

```python
class ScoreDistributionAnalyzer:
    """Analyze score contribution from BM25 vs vector."""
    
    def analyze_score_distribution(self, query: str, top: int = 20) -> Dict[str, Any]:
        """
        Analyze how BM25 and vector contribute to final RRF scores.
        
        Note: Azure AI Search doesn't expose individual BM25/vector scores
        in RRF fusion, so we infer from separate searches.
        """
        # Get BM25 results
        bm25_results = list(self.search_client.search(
            search_text=query,
            top=top
        ))
        bm25_scores = {r['id']: r.get('@search.score', 0) for r in bm25_results}
        
        # Get vector results
        query_vector = self.embedding_function(query)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top,
            fields="titleVector"
        )
        vector_results = list(self.search_client.search(
            vector_queries=[vector_query],
            top=top
        ))
        vector_scores = {r['id']: r.get('@search.score', 0) for r in vector_results}
        
        # Get hybrid results
        hybrid_results = list(self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top
        ))
        
        # Analyze distribution
        analysis = {
            'query': query,
            'bm25_score_range': (min(bm25_scores.values() or [0]), max(bm25_scores.values() or [0])),
            'vector_score_range': (min(vector_scores.values() or [0]), max(vector_scores.values() or [0])),
            'top10_source_distribution': {'bm25_only': 0, 'vector_only': 0, 'both': 0}
        }
        
        for rank, result in enumerate(hybrid_results[:10], 1):
            doc_id = result['id']
            in_bm25 = doc_id in bm25_scores
            in_vector = doc_id in vector_scores
            
            if in_bm25 and in_vector:
                analysis['top10_source_distribution']['both'] += 1
            elif in_bm25:
                analysis['top10_source_distribution']['bm25_only'] += 1
            else:
                analysis['top10_source_distribution']['vector_only'] += 1
        
        # Calculate imbalance score (0 = perfect balance, 1 = complete imbalance)
        dist = analysis['top10_source_distribution']
        max_source = max(dist['bm25_only'], dist['vector_only'])
        analysis['imbalance_score'] = max_source / 10.0
        
        return analysis

# Usage
analyzer = ScoreDistributionAnalyzer(search_client, get_embedding)
distribution = analyzer.analyze_score_distribution("laptop for video editing")

print(f"BM25 score range: {distribution['bm25_score_range']}")
print(f"Vector score range: {distribution['vector_score_range']}")
print(f"Top 10 distribution: {distribution['top10_source_distribution']}")
print(f"Imbalance score: {distribution['imbalance_score']:.2f}")  # 0.8 = 80% from one source

# Example output showing imbalance:
# BM25 score range: (0.12, 4.8)
# Vector score range: (0.85, 0.98)
# Top 10 distribution: {'bm25_only': 1, 'vector_only': 8, 'both': 1}
# Imbalance score: 0.80 ← Problem: 80% vector-only
```

### Solutions

See **Issue 1: Vector Results Dominating** for solutions to score imbalance.

---

## Issue 5: Zero-Result Rate Not Improved

### Symptoms

- Hybrid search zero-result rate similar to BM25-only
- Expected: Vector search should "rescue" BM25 zero-result queries
- Actual: Still getting 10-15% zero-result rate

### Root Causes

1. **Query doesn't generate BM25 OR vector results**: Both searches fail
2. **Filters too restrictive**: Filters eliminate all results before fusion
3. **Embedding doesn't match indexed content**: Query embedding orthogonal to document embeddings

### Solutions

#### Solution 1: Implement Graceful Fallback

```python
class HybridSearchWithFallback:
    """Hybrid search with zero-result handling."""
    
    def search(self, query: str, filters: str = None, top: int = 10) -> List[Dict[str, Any]]:
        """Execute hybrid search with fallback strategies."""
        
        # 1. Try full hybrid search
        results = self._hybrid_search(query, filters, top)
        
        if len(results) > 0:
            return results
        
        # 2. Fallback: Remove filters, try hybrid again
        if filters:
            results = self._hybrid_search(query, filters=None, top=top)
            if len(results) > 0:
                return self._add_filter_warning(results, "Filters removed to show results")
        
        # 3. Fallback: Expand query with synonyms
        expanded_query = self._expand_query_with_synonyms(query)
        results = self._hybrid_search(expanded_query, filters=None, top=top)
        if len(results) > 0:
            return self._add_filter_warning(results, "Query expanded to show results")
        
        # 4. Final fallback: Show popular products
        return self._get_popular_products(top)
    
    def _hybrid_search(self, query: str, filters: str, top: int) -> List[Dict[str, Any]]:
        """Execute hybrid search."""
        query_vector = self.embedding_function(query)
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=50, fields="titleVector")
        
        results = list(self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            filter=filters,
            top=top
        ))
        return results
    
    def _expand_query_with_synonyms(self, query: str) -> str:
        """Add common synonyms to query."""
        # Simple synonym expansion (could use more sophisticated NLP)
        synonyms = {
            'laptop': ['laptop', 'notebook'],
            'phone': ['phone', 'smartphone', 'mobile'],
            'cheap': ['cheap', 'budget', 'affordable']
        }
        # Expand query with synonyms (implementation details omitted)
        return query

# Impact: Reduces zero-result rate from 12% → 5% (58% reduction)
```

### Prevention

✅ **DO:**
- Implement fallback strategies (remove filters → expand query → show popular)
- Monitor zero-result rate by query type
- Use broad embeddings that capture many concepts

❌ **DON'T:**
- Apply overly restrictive filters
- Assume hybrid automatically fixes zero-results

---

## Conclusion

Common hybrid search issues and key solutions:

1. **Vector dominance**: Reduce k, improve query classification, boost BM25 fields
2. **Poor fusion**: Validate embedding quality, use k=50-100, try dual-vector strategy
3. **High latency**: Cache embeddings (35ms savings), optimize HNSW (efSearch=300-400), reduce k
4. **Score imbalance**: See vector dominance solutions
5. **Zero-results**: Implement fallback strategies (remove filters, expand query, show popular)

**Monitoring Recommendations:**
- Track vector dominance (target: 30-50%)
- Monitor p95/p99 latency (target: <100ms / <200ms)
- Measure zero-result rate (target: <5%)
- Compare hybrid vs single-mode precision (target: +10-20pp)

**Total Word Count**: ~5,100 words
