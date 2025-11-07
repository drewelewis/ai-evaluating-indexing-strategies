# Query Optimization

Complete guide to optimizing Azure AI Search query performance, reducing latency, and improving efficiency. This document provides comprehensive coverage of query performance analysis, caching strategies, filter optimization, field selection, pagination techniques, and advanced optimization patterns for achieving sub-100ms query latency at scale.

## 📋 Table of Contents
- [Overview](#overview)
- [Query Performance Analysis](#query-performance-analysis)
- [Caching Strategies](#caching-strategies)
- [Filter Optimization](#filter-optimization)
- [Field Selection](#field-selection)
- [Pagination Optimization](#pagination-optimization)
- [Index Optimization](#index-optimization)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Query optimization is the practice of reducing search latency, improving throughput, and minimizing resource consumption while maintaining or improving relevance. In production systems, query performance directly impacts user experience, conversion rates, and infrastructure costs. Research consistently shows that every 100ms of added latency reduces conversion by 1%, making query optimization critical for business success.

### Why Query Optimization Matters

**User Experience Impact:**
- **0-100ms**: Instant, no perceived delay
- **100-300ms**: Noticeable delay, acceptable
- **300-1000ms**: Sluggish, user frustration begins
- **>1000ms**: Abandon rate increases sharply

**Business Impact:**
- Google: 400ms slower → 0.74% fewer searches (2009 study)
- Amazon: 100ms slower → 1% revenue loss
- Shopzilla: 5s → 1.2s latency = 25% traffic increase, 7-12% revenue increase

**Cost Impact:**
- Query optimization can reduce required search units by 50-70%
- Caching can eliminate 60-80% of redundant searches
- Proper filtering reduces compute by 2-10× per query

### Real-World Application Scenario

**Company**: Contoso Real Estate (Property search platform)
**Scale**: 2 million property listings, 500K users, 50 million queries/month
**Challenge**: Query latency degrading as index grows, high infrastructure costs

**Previous Architecture (Unoptimized):**
```
Index: 2M properties, all 40 fields marked searchable
Query Pattern: 
- Full-text search across all fields
- No caching
- Client-side filtering after retrieval
- Fetching all fields (avg 15KB per document)
- Offset-based pagination with high skip values

Performance Metrics:
- P50 latency: 280ms
- P95 latency: 850ms
- P99 latency: 1,400ms
- Cache hit rate: 0% (no caching)
- Queries per search unit: ~8 QPS
- Infrastructure: 4 S3 search units @ $3,200/month
- Bounce rate on slow queries: 35%
```

**Optimization Strategy Implemented:**

**Phase 1: Low-Hanging Fruit (Week 1)**
1. **Field Selection Optimization**
   - Changed: Fetch only 5 essential fields vs all 40
   - Code: Added `select="id,title,price,location,imageUrl"`
   - Result: Response size 15KB → 2KB (7.5× reduction)
   - Latency improvement: -45ms (P50: 280ms → 235ms)

2. **Basic Query Caching**
   - Implemented: In-memory cache with 5-minute TTL
   - Cache targets: Top 1000 common queries (from logs)
   - Hit rate achieved: 62%
   - Latency for cached queries: 235ms → 8ms (29× faster)
   - Overall P50 improvement: 235ms → 110ms (62% cached × 8ms + 38% uncached × 235ms)

3. **Pre-Filtering vs Post-Filtering**
   - Changed: Moved filters from application to query
   - Before: Fetch 100 results → filter in app → return 10
   - After: Filter in query → fetch 10 results
   - Query change: Added `filter="city eq 'Seattle' and price le 500000"`
   - Latency improvement: -35ms (P50: 110ms → 75ms)

**Phase 2: Advanced Optimizations (Week 2-3)**

4. **Index Schema Optimization**
   - Reduced searchable fields: 40 → 12 (only user-facing text)
   - Marked strategic fields filterable: city, state, price, bedrooms, bathrooms
   - Index rebuild impact: 30% smaller index, faster searches
   - Latency improvement: -15ms (P50: 75ms → 60ms)

5. **Pagination Optimization**
   - Changed: Offset pagination → continuation token
   - Before: Page 10 requires skip=90 (slow)
   - After: Continuation token (constant time)
   - Deep pagination: 450ms → 85ms (5.3× faster for page 10+)

6. **Embedding Cache for Vector Queries**
   - Cached: 5,000 common query embeddings
   - Embedding generation time: 25-40ms → 0ms (cached)
   - Hit rate: 58% (common location/property type queries)
   - Cost savings: 29M embedding calls/month eliminated = $3.77/month (small but adds up)
   - Latency improvement: -20ms avg (P50: 60ms → 40ms for vector queries)

**Final Performance After Full Optimization:**
```
Performance Metrics:
- P50 latency: 40ms (85% improvement from 280ms)
- P95 latency: 120ms (86% improvement from 850ms)
- P99 latency: 280ms (80% improvement from 1,400ms)
- Cache hit rate: 62% overall (query cache) + 58% (embedding cache)
- Queries per search unit: ~22 QPS (2.75× improvement from 8 QPS)
- Infrastructure: 2 S2 search units @ $1,000/month (69% cost reduction)
- Bounce rate: 35% → 12% (66% reduction)
```

**Business Impact:**
```
Revenue Impact:
- Conversion rate improvement: 1.8% → 2.4% (+33% from lower latency)
- Monthly transactions: 15,000 → 20,000 (+5,000)
- Avg commission per transaction: $3,200
- Additional monthly revenue: 5,000 × $3,200 = $16M/year

Cost Savings:
- Infrastructure: $3,200 → $1,000 = $2,200/month saved = $26,400/year
- Total annual benefit: $16M revenue + $26K savings ≈ $16M

ROI:
- Investment: 3 weeks developer time (~$15K)
- Annual return: $16M
- ROI: 1,067× return
```

**Key Takeaways from Contoso Case Study:**
1. **Field selection** (7.5× data reduction) was the easiest, highest-impact optimization
2. **Caching** eliminated 62% of query load with simple in-memory cache
3. **Pre-filtering** (2-5× faster) required only query restructuring, no code changes
4. **Cumulative effect**: Multiple 10-20% improvements compound to 85% total improvement
5. **Business value**: Every 100ms matters (1.8% → 2.4% conversion = $16M/year)

This document will guide you through implementing similar optimizations for your search application.

---

### Query Optimization Goals

```python
class QueryOptimizationGoals:
    """Understanding query optimization objectives."""
    
    @staticmethod
    def optimization_objectives():
        """Key optimization goals."""
        return {
            'latency_reduction': {
                'target': 'Sub-100ms query response time',
                'techniques': [
                    'Caching',
                    'Field selection',
                    'Filter optimization',
                    'Index tuning'
                ],
                'measurement': 'P95 latency'
            },
            'cost_reduction': {
                'target': 'Minimize search unit consumption',
                'techniques': [
                    'Query simplification',
                    'Result caching',
                    'Efficient pagination'
                ],
                'measurement': 'Queries per second per SU'
            },
            'relevance': {
                'target': 'Maintain or improve result quality',
                'constraint': 'Must not sacrifice relevance for speed',
                'measurement': 'nDCG, MAP'
            },
            'scalability': {
                'target': 'Handle increasing query volume',
                'techniques': [
                    'Replica scaling',
                    'Query distribution',
                    'Cache layers'
                ],
                'measurement': 'Max QPS'
            }
        }
    
    @staticmethod
    def performance_targets():
        """Recommended performance targets."""
        return {
            'latency': {
                'excellent': '<50ms',
                'good': '50-100ms',
                'acceptable': '100-200ms',
                'poor': '>200ms'
            },
            'throughput': {
                'basic_tier': '~50-100 QPS',
                'standard_s1': '~100-200 QPS',
                'standard_s2': '~200-400 QPS',
                'standard_s3': '~400-800 QPS'
            },
            'cache_hit_rate': {
                'excellent': '>80%',
                'good': '60-80%',
                'acceptable': '40-60%',
                'poor': '<40%'
            }
        }

# Usage
goals = QueryOptimizationGoals()
objectives = goals.optimization_objectives()

print("Optimization Objectives:")
for obj, details in objectives.items():
    print(f"\n{obj}:")
    print(f"  Target: {details['target']}")
    print(f"  Measurement: {details['measurement']}")

targets = goals.performance_targets()
print(f"\nLatency targets:")
for level, time in targets['latency'].items():
    print(f"  {level}: {time}")
```

---

## Query Performance Analysis

### Performance Measurement

```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import time
import statistics

class QueryPerformanceAnalyzer:
    """Analyze and measure query performance."""
    
    def __init__(self, search_endpoint, index_name, search_key):
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )
    
    def measure_query_latency(self, query_text, iterations=10):
        """
        Measure query latency over multiple iterations.
        
        Returns detailed statistics including:
        - Min, max, mean, median latency
        - Standard deviation
        - P95, P99 percentiles
        """
        latencies = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            results = list(self.search_client.search(
                search_text=query_text,
                top=10
            ))
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        latencies.sort()
        return {
            'iterations': iterations,
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'stdev_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'p95_ms': latencies[int(len(latencies) * 0.95)],
            'p99_ms': latencies[int(len(latencies) * 0.99)],
            'all_latencies': latencies
        }
    
    def compare_query_variants(self, queries):
        """
        Compare performance of different query variants.
        
        Args:
            queries: Dict of query_name -> query_config
        """
        results = {}
        
        for query_name, query_config in queries.items():
            start_time = time.perf_counter()
            
            search_results = list(self.search_client.search(**query_config))
            
            end_time = time.perf_counter()
            
            results[query_name] = {
                'latency_ms': (end_time - start_time) * 1000,
                'result_count': len(search_results),
                'config': query_config
            }
        
        return results
    
    def profile_search_components(self, query_text):
        """
        Profile individual search components.
        
        Compares:
        - Full-text only
        - Vector only (if applicable)
        - Hybrid
        - With/without filters
        """
        profiles = {}
        
        # Full-text only
        start = time.perf_counter()
        fulltext = list(self.search_client.search(
            search_text=query_text,
            top=10
        ))
        profiles['fulltext'] = (time.perf_counter() - start) * 1000
        
        # With select (limited fields)
        start = time.perf_counter()
        limited = list(self.search_client.search(
            search_text=query_text,
            select="id,title",
            top=10
        ))
        profiles['limited_fields'] = (time.perf_counter() - start) * 1000
        
        # With filter
        start = time.perf_counter()
        filtered = list(self.search_client.search(
            search_text=query_text,
            filter="price le 1000",
            top=10
        ))
        profiles['with_filter'] = (time.perf_counter() - start) * 1000
        
        return profiles

# Usage
import os

analyzer = QueryPerformanceAnalyzer(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    index_name="products",
    search_key=os.getenv("SEARCH_API_KEY")
)

# Measure latency
stats = analyzer.measure_query_latency("laptop", iterations=20)
print(f"Query latency statistics:")
print(f"  Mean: {stats['mean_ms']:.2f}ms")
print(f"  Median: {stats['median_ms']:.2f}ms")
print(f"  P95: {stats['p95_ms']:.2f}ms")
print(f"  P99: {stats['p99_ms']:.2f}ms")

# Compare variants
variants = {
    'simple': {'search_text': 'laptop', 'top': 10},
    'with_filter': {'search_text': 'laptop', 'filter': 'price le 1000', 'top': 10},
    'specific_fields': {'search_text': 'laptop', 'search_fields': 'title', 'top': 10}
}

comparison = analyzer.compare_query_variants(variants)
for name, result in comparison.items():
    print(f"{name}: {result['latency_ms']:.2f}ms")

# Profile components
profiles = analyzer.profile_search_components("laptop for gaming")
print("\nComponent Profiling:")
for component, latency in profiles.items():
    print(f"  {component}: {latency:.2f}ms")
```

---

## Caching Strategies

### Multi-Layer Caching

```python
import hashlib
import json
from datetime import datetime, timedelta

class SearchCache:
    """Multi-layer caching for search results."""
    
    def __init__(self, search_client):
        self.search_client = search_client
        self.l1_cache = {}  # In-memory cache
        self.l1_ttl_seconds = 60  # 1 minute
        self.l2_cache = {}  # Longer-term cache
        self.l2_ttl_seconds = 300  # 5 minutes
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'total_queries': 0
        }
    
    def _generate_cache_key(self, **search_params):
        """Generate cache key from search parameters."""
        # Sort parameters for consistent keys
        param_str = json.dumps(search_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def search(self, **search_params):
        """
        Execute search with caching.
        
        Cache hierarchy:
        1. L1 cache (1 minute TTL, in-memory)
        2. L2 cache (5 minute TTL, in-memory)
        3. Azure AI Search (cache miss)
        """
        self.stats['total_queries'] += 1
        cache_key = self._generate_cache_key(**search_params)
        now = datetime.now()
        
        # Check L1 cache
        if cache_key in self.l1_cache:
            entry = self.l1_cache[cache_key]
            if (now - entry['timestamp']).total_seconds() < self.l1_ttl_seconds:
                self.stats['l1_hits'] += 1
                return entry['results'], 'l1_cache'
            else:
                del self.l1_cache[cache_key]
        
        # Check L2 cache
        if cache_key in self.l2_cache:
            entry = self.l2_cache[cache_key]
            if (now - entry['timestamp']).total_seconds() < self.l2_ttl_seconds:
                self.stats['l2_hits'] += 1
                # Promote to L1
                self.l1_cache[cache_key] = {
                    'results': entry['results'],
                    'timestamp': now
                }
                return entry['results'], 'l2_cache'
            else:
                del self.l2_cache[cache_key]
        
        # Cache miss - execute search
        self.stats['misses'] += 1
        results = list(self.search_client.search(**search_params))
        
        # Store in both caches
        cache_entry = {
            'results': results,
            'timestamp': now
        }
        self.l1_cache[cache_key] = cache_entry.copy()
        self.l2_cache[cache_key] = cache_entry.copy()
        
        return results, 'cache_miss'
    
    def invalidate_cache(self, pattern=None):
        """
        Invalidate cache entries.
        
        Args:
            pattern: Optional pattern to match keys (None = clear all)
        """
        if pattern is None:
            self.l1_cache.clear()
            self.l2_cache.clear()
        else:
            # Remove matching keys
            self.l1_cache = {
                k: v for k, v in self.l1_cache.items()
                if pattern not in k
            }
            self.l2_cache = {
                k: v for k, v in self.l2_cache.items()
                if pattern not in k
            }
    
    def get_cache_stats(self):
        """Get cache performance statistics."""
        total = self.stats['total_queries']
        if total == 0:
            return {'hit_rate': 0.0}
        
        hits = self.stats['l1_hits'] + self.stats['l2_hits']
        hit_rate = (hits / total) * 100
        
        return {
            'total_queries': total,
            'l1_hits': self.stats['l1_hits'],
            'l2_hits': self.stats['l2_hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'l1_hit_rate': (self.stats['l1_hits'] / total) * 100,
            'l2_hit_rate': (self.stats['l2_hits'] / total) * 100
        }
    
    def get_cache_size(self):
        """Get current cache sizes."""
        return {
            'l1_entries': len(self.l1_cache),
            'l2_entries': len(self.l2_cache)
        }

# Usage
from azure.search.documents import SearchClient

search_client = SearchClient(
    endpoint=os.getenv("SEARCH_ENDPOINT"),
    index_name="products",
    credential=AzureKeyCredential(os.getenv("SEARCH_API_KEY"))
)

cache = SearchCache(search_client)

# Execute searches (some will be cached)
queries = ["laptop", "laptop", "gaming laptop", "laptop"]

for query in queries:
    results, source = cache.search(search_text=query, top=10)
    print(f"Query: '{query}' - Source: {source} - Results: {len(results)}")

# Check cache stats
stats = cache.get_cache_stats()
print(f"\nCache Statistics:")
print(f"  Total queries: {stats['total_queries']}")
print(f"  Hit rate: {stats['hit_rate']:.1f}%")
print(f"  L1 hits: {stats['l1_hits']}")
print(f"  L2 hits: {stats['l2_hits']}")
print(f"  Misses: {stats['misses']}")
```

### Query Embedding Cache

```python
import hashlib
import pickle

class EmbeddingCache:
    """Cache query embeddings to reduce OpenAI API calls."""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.cache = {}
        self.stats = {'hits': 0, 'misses': 0}
    
    def get_embedding(self, text, model="text-embedding-ada-002"):
        """
        Get embedding with caching.
        
        Cache key: MD5 hash of (text + model)
        """
        cache_key = hashlib.md5(f"{text}:{model}".encode()).hexdigest()
        
        if cache_key in self.cache:
            self.stats['hits'] += 1
            return self.cache[cache_key]
        
        # Cache miss - generate embedding
        self.stats['misses'] += 1
        response = self.openai_client.embeddings.create(
            input=text,
            model=model
        )
        embedding = response.data[0].embedding
        
        # Cache the embedding
        self.cache[cache_key] = embedding
        
        return embedding
    
    def save_cache(self, filepath):
        """Persist cache to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def load_cache(self, filepath):
        """Load cache from disk."""
        try:
            with open(filepath, 'rb') as f:
                self.cache = pickle.load(f)
            return len(self.cache)
        except FileNotFoundError:
            return 0
    
    def get_stats(self):
        """Get cache statistics."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'estimated_savings_usd': (self.stats['hits'] * 0.0001 / 1000)
        }

# Usage
from openai import AzureOpenAI

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

embedding_cache = EmbeddingCache(openai_client)

# Load existing cache
loaded = embedding_cache.load_cache("embedding_cache.pkl")
print(f"Loaded {loaded} cached embeddings")

# Get embeddings (some cached, some not)
queries = ["laptop", "laptop", "gaming PC", "laptop for ML"]

for query in queries:
    embedding = embedding_cache.get_embedding(query)
    print(f"Query: '{query}' - Embedding dim: {len(embedding)}")

# Save cache
embedding_cache.save_cache("embedding_cache.pkl")

# Stats
stats = embedding_cache.get_stats()
print(f"\nEmbedding Cache Stats:")
print(f"  Cache size: {stats['cache_size']}")
print(f"  Hit rate: {stats['hit_rate']:.1f}%")
print(f"  Estimated savings: ${stats['estimated_savings_usd']:.4f}")
```

---

## Filter Optimization

### Efficient Filtering

```python
class FilterOptimization:
    """Optimize filter usage in queries."""
    
    @staticmethod
    def filter_best_practices():
        """Best practices for filter optimization."""
        return {
            'filter_early': {
                'description': 'Apply filters before search to reduce scope',
                'example': "filter='price le 1000' reduces search space",
                'benefit': 'Faster search, lower resource usage'
            },
            'use_indexed_fields': {
                'description': 'Filter on filterable=True fields only',
                'requirement': 'Field must be marked filterable in schema',
                'performance': 'Indexed filters are much faster'
            },
            'combine_filters': {
                'description': 'Use AND/OR to combine multiple conditions',
                'example': "filter='price le 1000 and category eq \\'Laptop\\''",
                'note': 'Single filter expression is more efficient'
            },
            'avoid_functions': {
                'description': 'Minimize use of OData functions in filters',
                'slow': 'geo.distance(), search.in() with many values',
                'fast': 'Simple comparisons (eq, le, ge)'
            },
            'cache_filter_results': {
                'description': 'Cache filtered result sets',
                'use_case': 'Common filter combinations (e.g., category pages)',
                'benefit': 'Eliminates repeated filter evaluation'
            }
        }
    
    @staticmethod
    def example_filter_optimization():
        """Example of filter optimization."""
        
        # ❌ Inefficient: Multiple searches with post-filtering
        def inefficient_approach(search_client, query):
            # Get all results
            all_results = list(search_client.search(
                search_text=query,
                top=1000  # Large result set
            ))
            
            # Filter in application
            filtered = [r for r in all_results if r['price'] <= 1000]
            return filtered[:10]
        
        # ✅ Efficient: Filter in query
        def efficient_approach(search_client, query):
            # Filter at search time
            results = list(search_client.search(
                search_text=query,
                filter="price le 1000",
                top=10  # Only need 10 results
            ))
            return results
        
        return {
            'inefficient': 'Fetches 1000 results, filters in app',
            'efficient': 'Filters in search, fetches only 10 results',
            'speedup': '10-100x faster'
        }

# Usage
best_practices = FilterOptimization.filter_best_practices()
for practice, details in best_practices.items():
    print(f"\n{practice}:")
    print(f"  {details['description']}")
    if 'benefit' in details:
        print(f"  Benefit: {details['benefit']}")
```

### Filter Selectivity

```python
class FilterSelectivity:
    """Analyze filter selectivity for optimization."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def measure_filter_selectivity(self, filter_expression):
        """
        Measure how selective a filter is.
        
        Selectivity = filtered_count / total_count
        
        Low selectivity (< 0.1) = very selective filter
        High selectivity (> 0.9) = not very selective
        """
        # Total documents
        total_results = self.search_client.search(
            search_text="*",
            select="id",
            top=0,
            include_total_count=True
        )
        total_count = total_results.get_count()
        
        # Filtered documents
        filtered_results = self.search_client.search(
            search_text="*",
            filter=filter_expression,
            select="id",
            top=0,
            include_total_count=True
        )
        filtered_count = filtered_results.get_count()
        
        selectivity = filtered_count / total_count if total_count > 0 else 0
        
        return {
            'total_documents': total_count,
            'filtered_documents': filtered_count,
            'selectivity': selectivity,
            'interpretation': self._interpret_selectivity(selectivity)
        }
    
    def _interpret_selectivity(self, selectivity):
        """Interpret selectivity value."""
        if selectivity < 0.1:
            return 'Very selective (excellent for filtering)'
        elif selectivity < 0.3:
            return 'Moderately selective (good)'
        elif selectivity < 0.7:
            return 'Somewhat selective (acceptable)'
        else:
            return 'Not selective (consider removing filter)'
    
    def optimize_filter_order(self, filters):
        """
        Determine optimal filter order.
        
        Apply most selective filters first.
        """
        selectivities = []
        
        for filter_expr in filters:
            stats = self.measure_filter_selectivity(filter_expr)
            selectivities.append({
                'filter': filter_expr,
                'selectivity': stats['selectivity'],
                'filtered_count': stats['filtered_documents']
            })
        
        # Sort by selectivity (most selective first)
        selectivities.sort(key=lambda x: x['selectivity'])
        
        return selectivities

# Usage
selectivity_analyzer = FilterSelectivity(search_client)

# Measure filter selectivity
stats = selectivity_analyzer.measure_filter_selectivity("price le 500")
print(f"Filter selectivity: {stats['selectivity']:.2%}")
print(f"Filtered documents: {stats['filtered_documents']:,}")
print(f"Interpretation: {stats['interpretation']}")

# Optimize filter order
filters = [
    "category eq 'Laptop'",
    "price le 1000",
    "rating ge 4.0"
]

optimized = selectivity_analyzer.optimize_filter_order(filters)
print("\nOptimal filter order:")
for f in optimized:
    print(f"  {f['filter']} (selectivity: {f['selectivity']:.2%})")
```

---

## Field Selection

### Selective Field Retrieval

```python
class FieldSelectionOptimization:
    """Optimize queries by selecting only needed fields."""
    
    @staticmethod
    def field_selection_impact():
        """Impact of field selection on performance."""
        return {
            'benefits': [
                'Reduced response size (less network transfer)',
                'Faster serialization/deserialization',
                'Lower memory usage',
                'Better cache efficiency'
            ],
            'typical_savings': {
                'all_fields': '~10-50KB per document',
                'id_title_only': '~1-5KB per document',
                'speedup': '2-5x faster for large documents'
            },
            'recommendation': 'Always use select parameter in production'
        }
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def compare_field_selection(self, query_text, top=10):
        """Compare performance with different field selections."""
        import time
        
        # All fields (default)
        start = time.perf_counter()
        all_fields_results = list(self.search_client.search(
            search_text=query_text,
            top=top
        ))
        all_fields_time = (time.perf_counter() - start) * 1000
        
        # Essential fields only
        start = time.perf_counter()
        essential_results = list(self.search_client.search(
            search_text=query_text,
            select="id,title,price",
            top=top
        ))
        essential_time = (time.perf_counter() - start) * 1000
        
        # ID only
        start = time.perf_counter()
        id_only_results = list(self.search_client.search(
            search_text=query_text,
            select="id",
            top=top
        ))
        id_only_time = (time.perf_counter() - start) * 1000
        
        return {
            'all_fields': {
                'latency_ms': all_fields_time,
                'result_count': len(all_fields_results)
            },
            'essential_fields': {
                'latency_ms': essential_time,
                'result_count': len(essential_results),
                'speedup': all_fields_time / essential_time
            },
            'id_only': {
                'latency_ms': id_only_time,
                'result_count': len(id_only_results),
                'speedup': all_fields_time / id_only_time
            }
        }

# Usage
field_optimizer = FieldSelectionOptimization(search_client)

comparison = field_optimizer.compare_field_selection("laptop", top=50)

print("Field Selection Performance:")
print(f"  All fields: {comparison['all_fields']['latency_ms']:.2f}ms")
print(f"  Essential fields: {comparison['essential_fields']['latency_ms']:.2f}ms " +
      f"({comparison['essential_fields']['speedup']:.1f}x faster)")
print(f"  ID only: {comparison['id_only']['latency_ms']:.2f}ms " +
      f"({comparison['id_only']['speedup']:.1f}x faster)")
```

---

## Pagination Optimization

### Efficient Pagination

```python
class PaginationOptimization:
    """Optimize pagination for large result sets."""
    
    @staticmethod
    def pagination_strategies():
        """Different pagination approaches."""
        return {
            'offset_based': {
                'method': 'Use skip and top parameters',
                'example': 'skip=20, top=10 for page 3',
                'pros': 'Simple, stateless',
                'cons': 'Slow for deep pagination (high skip values)',
                'use_case': 'Shallow pagination (first few pages)'
            },
            'continuation_token': {
                'method': 'Use search.nextPageParameters',
                'example': 'Token-based pagination',
                'pros': 'Fast for all page depths',
                'cons': 'Stateful, tokens expire',
                'use_case': 'Deep pagination, sequential access'
            },
            'search_after': {
                'method': 'Use searchAfter parameter with sort',
                'example': 'searchAfter=lastDocSortValue',
                'pros': 'Efficient, consistent results',
                'cons': 'Requires stable sort field',
                'use_case': 'Large result sets, export scenarios'
            }
        }
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def offset_pagination(self, query_text, page_number, page_size=10):
        """
        Traditional offset-based pagination.
        
        Note: Performance degrades with high page numbers.
        """
        skip = (page_number - 1) * page_size
        
        results = self.search_client.search(
            search_text=query_text,
            skip=skip,
            top=page_size,
            include_total_count=True
        )
        
        documents = list(results)
        total_count = results.get_count()
        total_pages = (total_count + page_size - 1) // page_size
        
        return {
            'documents': documents,
            'page': page_number,
            'page_size': page_size,
            'total_results': total_count,
            'total_pages': total_pages,
            'has_more': page_number < total_pages
        }
    
    def efficient_deep_pagination(self, query_text, page_size=10):
        """
        Efficient pagination using continuation.
        
        Returns generator for memory efficiency.
        """
        search_results = self.search_client.search(
            search_text=query_text,
            top=page_size
        )
        
        page_number = 1
        
        for result in search_results.by_page():
            documents = list(result)
            
            yield {
                'documents': documents,
                'page': page_number,
                'count': len(documents)
            }
            
            page_number += 1
    
    def measure_pagination_performance(self, query_text, max_page=10, page_size=10):
        """
        Measure pagination performance at different depths.
        """
        import time
        
        latencies = {}
        
        for page in [1, 5, max_page]:
            start = time.perf_counter()
            
            skip = (page - 1) * page_size
            results = list(self.search_client.search(
                search_text=query_text,
                skip=skip,
                top=page_size
            ))
            
            latency = (time.perf_counter() - start) * 1000
            latencies[f'page_{page}'] = latency
        
        return latencies

# Usage
pagination = PaginationOptimization(search_client)

# Offset-based pagination
page_result = pagination.offset_pagination("laptop", page_number=2, page_size=10)
print(f"Page {page_result['page']} of {page_result['total_pages']}")
print(f"Results: {len(page_result['documents'])}")

# Efficient deep pagination
print("\nEfficient pagination:")
for page in pagination.efficient_deep_pagination("laptop", page_size=10):
    print(f"  Page {page['page']}: {page['count']} documents")
    if page['page'] >= 3:  # Stop after 3 pages for demo
        break

# Performance comparison
perf = pagination.measure_pagination_performance("laptop", max_page=20)
print("\nPagination Performance:")
for page, latency in perf.items():
    print(f"  {page}: {latency:.2f}ms")
```

---

## Index Optimization

### Index Configuration for Query Performance

```python
class IndexOptimizationForQueries:
    """Optimize index configuration for query performance."""
    
    @staticmethod
    def optimization_recommendations():
        """Index optimization recommendations."""
        return {
            'minimize_searchable_fields': {
                'description': 'Mark only necessary fields as searchable',
                'reason': 'Reduces index size and search scope',
                'example': 'Don\'t search on price, rating (use filters instead)'
            },
            'use_appropriate_analyzers': {
                'description': 'Choose analyzers matching query patterns',
                'recommendation': {
                    'keyword_queries': 'keyword analyzer',
                    'natural_language': 'language-specific analyzer (en.microsoft)',
                    'codes_skus': 'keyword or pattern analyzer'
                }
            },
            'strategic_filterable_sortable': {
                'description': 'Mark fields filterable/sortable strategically',
                'note': 'Each filterable field increases index size',
                'guideline': 'Only mark fields you\'ll actually filter/sort on'
            },
            'scoring_profile_optimization': {
                'description': 'Use scoring profiles vs complex queries',
                'benefit': 'Pre-computed boost values, faster at query time'
            },
            'replica_count': {
                'description': 'Add replicas for query throughput',
                'formula': 'QPS capacity ≈ replicas × base_QPS',
                'use_case': 'High query volume scenarios'
            }
        }
    
    @staticmethod
    def calculate_optimal_replicas(target_qps, base_qps_per_replica=100):
        """
        Calculate optimal replica count for target QPS.
        
        Args:
            target_qps: Target queries per second
            base_qps_per_replica: QPS capacity per replica
            
        Returns:
            Recommended replica count
        """
        replicas_needed = (target_qps + base_qps_per_replica - 1) // base_qps_per_replica
        
        # Azure Search supports 1-12 replicas
        replicas_needed = max(1, min(12, replicas_needed))
        
        return {
            'target_qps': target_qps,
            'base_qps_per_replica': base_qps_per_replica,
            'recommended_replicas': replicas_needed,
            'estimated_capacity': replicas_needed * base_qps_per_replica,
            'note': 'Add buffer for peak load (recommend +1 replica)'
        }

# Usage
index_opt = IndexOptimizationForQueries()

recommendations = index_opt.optimization_recommendations()
print("Index Optimization Recommendations:")
for key, rec in recommendations.items():
    print(f"\n{key}:")
    print(f"  {rec['description']}")

# Calculate replicas
replica_calc = index_opt.calculate_optimal_replicas(target_qps=500)
print(f"\nFor {replica_calc['target_qps']} QPS:")
print(f"  Recommended replicas: {replica_calc['recommended_replicas']}")
print(f"  Estimated capacity: {replica_calc['estimated_capacity']} QPS")
```

---

## Advanced Techniques

### Query Rewriting

```python
class QueryRewriter:
    """Rewrite queries for better performance."""
    
    @staticmethod
    def expand_synonyms(query_text, synonym_map):
        """
        Expand query with synonyms.
        
        Note: Azure Search has built-in synonym maps,
        but this shows the concept.
        """
        words = query_text.split()
        expanded = []
        
        for word in words:
            if word.lower() in synonym_map:
                # Add synonyms with OR
                synonyms = synonym_map[word.lower()]
                expanded.append(f"({word} OR {' OR '.join(synonyms)})")
            else:
                expanded.append(word)
        
        return " ".join(expanded)
    
    @staticmethod
    def simplify_complex_query(query_text, max_terms=5):
        """
        Simplify overly complex queries.
        
        Takes first N important terms.
        """
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}
        words = [w for w in query_text.split() if w.lower() not in stop_words]
        
        # Take first max_terms
        simplified = words[:max_terms]
        
        return " ".join(simplified)
    
    @staticmethod
    def optimize_phrase_queries(query_text):
        """
        Optimize phrase queries for performance.
        
        Converts "exact phrase" to proximity search where appropriate.
        """
        import re
        
        # Find quoted phrases
        phrases = re.findall(r'"([^"]*)"', query_text)
        
        optimized = query_text
        for phrase in phrases:
            words = phrase.split()
            if len(words) > 3:
                # Long phrase -> proximity search
                proximity = f'"{words[0]} {words[-1]}"~10'
                optimized = optimized.replace(f'"{phrase}"', proximity)
        
        return optimized

# Usage
rewriter = QueryRewriter()

# Synonym expansion
synonyms = {
    'laptop': ['notebook', 'portable computer'],
    'cheap': ['affordable', 'budget', 'inexpensive']
}
expanded = rewriter.expand_synonyms("cheap laptop", synonyms)
print(f"Expanded query: {expanded}")

# Simplify
complex_query = "the best and most affordable laptop for machine learning and AI"
simplified = rewriter.simplify_complex_query(complex_query, max_terms=4)
print(f"Simplified: {simplified}")

# Optimize phrases
phrase_query = '"high performance gaming laptop with RGB keyboard"'
optimized = rewriter.optimize_phrase_queries(phrase_query)
print(f"Optimized phrase: {optimized}")
```

---

## Best Practices

### 1. Field Selection and Data Transfer Optimization

**DO: Always use the `select` parameter to limit returned fields**

```python
# ✅ GOOD: Return only needed fields (2KB response)
results = search_client.search(
    search_text="laptop",
    select="id,title,price,rating,imageUrl",
    top=10
)

# ❌ BAD: Return all fields (15KB response, 7.5× larger)
results = search_client.search(
    search_text="laptop",
    top=10
)
```

**Impact**:
- Response size: 15KB → 2KB (7.5× reduction)
- Serialization time: -40ms per query
- Network transfer: -13KB × 10 results = -130KB per query
- At 1M queries/month: 130GB → 17GB (113GB saved)

**Guidelines**:
- **Search results page**: `id, title, price, thumbnail, rating` (minimal for display)
- **Detail page**: Fetch full document by ID (single doc, not search query)
- **Autocomplete**: `id, title` only
- **Facets/filters**: Just the facet fields needed

**DON'T: Retrieve all fields "just in case"**

Every extra field increases latency and costs:
- Small field (50 chars): +100 bytes
- Large field (5000 chars): +5KB
- Binary/encoded field: +10-50KB

**Decision Matrix: Which fields to include?**

| Field Purpose | Include in select? | Reasoning |
|---------------|-------------------|-----------|
| Display in results | ✅ Yes | Needed for UI |
| Sort/filter only | ❌ No | Use $orderby/$filter, don't return |
| Detail page only | ❌ No | Fetch separately by ID |
| Search-only field | ❌ No | Used for matching, not display |
| Facet field | ✅ Yes (if showing) | If displaying facet counts |

---

### 2. Caching Strategy and Implementation

**DO: Implement multi-layer caching with appropriate TTLs**

```python
# ✅ GOOD: Multi-layer cache with different TTLs
class OptimalSearchCache:
    def __init__(self, search_client):
        self.search_client = search_client
        # L1: Hot queries (1 minute TTL, 1000 entries max)
        self.l1_cache = {}  # In-memory, very fast
        self.l1_ttl = 60
        self.l1_max_size = 1000
        
        # L2: Warm queries (5 minute TTL, 10000 entries max)
        self.l2_cache = {}  # In-memory or Redis
        self.l2_ttl = 300
        self.l2_max_size = 10000
        
        # L3: Cold queries (1 hour TTL, unlimited)
        # Could be Redis, Memcached, or Azure Cache for Redis
        
    def search(self, **params):
        cache_key = self._hash_params(params)
        
        # Check L1 (hot)
        if cache_key in self.l1_cache and self._is_valid(self.l1_cache[cache_key], self.l1_ttl):
            return self.l1_cache[cache_key]['results'], 'l1_hit'
        
        # Check L2 (warm)
        if cache_key in self.l2_cache and self._is_valid(self.l2_cache[cache_key], self.l2_ttl):
            # Promote to L1
            self.l1_cache[cache_key] = self.l2_cache[cache_key]
            self._evict_if_needed(self.l1_cache, self.l1_max_size)
            return self.l2_cache[cache_key]['results'], 'l2_hit'
        
        # Cache miss - execute search
        results = list(self.search_client.search(**params))
        
        # Store in both caches
        entry = {'results': results, 'timestamp': time.time()}
        self.l1_cache[cache_key] = entry
        self.l2_cache[cache_key] = entry
        
        self._evict_if_needed(self.l1_cache, self.l1_max_size)
        self._evict_if_needed(self.l2_cache, self.l2_max_size)
        
        return results, 'cache_miss'
```

**Impact**:
- Hit rate target: >60% (62% in Contoso case study)
- Latency for cached queries: 250ms → 5-10ms (25-50× faster)
- Overall latency improvement: 40-60% (depends on hit rate)
- Cost savings: 60% fewer searches = 40% of original infrastructure

**DON'T: Cache without TTL or invalidation strategy**

```python
# ❌ BAD: Infinite cache, never invalidates
class BadCache:
    def __init__(self):
        self.cache = {}  # Grows forever, serves stale data
    
    def search(self, query):
        if query in self.cache:
            return self.cache[query]  # Could be months old!
        
        results = execute_search(query)
        self.cache[query] = results
        return results
```

**Problems**:
- Serves stale data after index updates
- Memory grows unbounded (cache poisoning)
- No cache eviction (oldest entries never removed)

**Cache Invalidation Rules**:
1. **Time-based**: Expire after TTL (common: 1-5 minutes)
2. **Event-based**: Invalidate on index update
3. **Size-based**: Evict LRU entries when cache full
4. **Manual**: Admin can force cache clear

---

### 3. Filter Optimization and Pre-Filtering

**DO: Apply filters in the query, not in application code**

```python
# ✅ GOOD: Filter at search time (2-5× faster)
results = search_client.search(
    search_text="laptop",
    filter="price le 1000 and category eq 'Electronics' and inStock eq true",
    top=10
)
# Azure Search filters BEFORE ranking, returns only 10 results

# ❌ BAD: Filter in application (slow, wasteful)
results = search_client.search(
    search_text="laptop",
    top=1000  # Fetch 1000 results
)
# Download 1000 results (15MB), filter in Python, keep 10
filtered = [r for r in results if r['price'] <= 1000 and r['category'] == 'Electronics'][:10]
```

**Impact**:
- Query time: 250ms → 85ms (2.9× faster)
- Data transfer: 15MB → 150KB (100× reduction)
- Compute: Server-side filtering (optimized) vs client-side (slow Python loops)

**DON'T: Filter on non-filterable fields**

```python
# ❌ BAD: Field not marked filterable in schema
results = search_client.search(
    search_text="laptop",
    filter="description eq 'High performance'"  # ERROR if not filterable!
)
```

**Solution**: Mark fields filterable in index schema:
```python
{
    "name": "price",
    "type": "Edm.Double",
    "filterable": True,  # ✅ Enable filtering
    "sortable": True,
    "facetable": True
}
```

**Filter Performance Guidelines**:

| Filter Type | Performance | When to Use |
|-------------|-------------|-------------|
| Equality (`eq`) | ⚡ Fastest | Exact matches: `category eq 'Laptop'` |
| Range (`le`, `ge`) | ⚡ Fast | Numeric ranges: `price le 1000` |
| Collection (`any`) | ⚡ Fast | Array fields: `tags/any(t: t eq 'gaming')` |
| String functions | ⚠️ Moderate | Only when necessary: `startswith(title, 'Mac')` |
| Geo distance | ⚠️ Moderate | Location-based: `geo.distance(...) le 10` |
| Complex expressions | ⚠️ Slower | Nested logic: multiple ANDs/ORs |

---

### 4. Search Scope and Field Targeting

**DO: Limit search to relevant fields using `search_fields`**

```python
# ✅ GOOD: Search only title and description (3× faster)
results = search_client.search(
    search_text="gaming laptop",
    search_fields="title,description",
    top=10
)

# ❌ BAD: Search all searchable fields (slow, noisy)
results = search_client.search(
    search_text="gaming laptop",  # Searches: title, description, reviews, tags, brand, category, ...
    top=10
)
```

**Impact**:
- Fewer fields → smaller inverted index to scan
- More precise matching (no false positives from irrelevant fields)
- Faster query execution: 150ms → 50ms (3× improvement)

**Field Selection Strategy**:
- **Product search**: `title^3, description, brand` (boost title 3×)
- **Document search**: `title^2, content`
- **People search**: `name^3, title, department`
- **Code search**: `filename^2, content`

**DON'T: Search numeric/date fields**

```python
# ❌ BAD: Searching numeric fields (nonsensical)
results = search_client.search(
    search_text="999.99",  # Searching price as text?
    top=10
)

# ✅ GOOD: Filter numeric fields instead
results = search_client.search(
    search_text="laptop",
    filter="price eq 999.99",  # Exact numeric match
    top=10
)
```

---

### 5. Pagination Efficiency

**DO: Use continuation tokens for deep pagination**

```python
# ✅ GOOD: Continuation-based pagination (constant time)
def paginate_with_continuation(search_client, query, page_size=10):
    results = search_client.search(
        search_text=query,
        top=page_size
    )
    
    for page in results.by_page():
        yield list(page)
        # Constant time per page, regardless of depth

# ❌ BAD: Offset-based pagination (slow for deep pages)
def paginate_with_offset(search_client, query, page_number, page_size=10):
    skip = (page_number - 1) * page_size
    results = search_client.search(
        search_text=query,
        skip=skip,  # Page 100: skip=990 (very slow!)
        top=page_size
    )
    return list(results)
```

**Performance Comparison**:

| Page Number | Offset Latency | Continuation Latency | Speedup |
|-------------|----------------|---------------------|---------|
| Page 1 | 50ms | 50ms | 1.0× |
| Page 5 | 85ms | 52ms | 1.6× |
| Page 10 | 150ms | 55ms | 2.7× |
| Page 50 | 450ms | 58ms | 7.8× |
| Page 100 | 850ms | 60ms | 14.2× |

**DON'T: Use high `skip` values**

```python
# ❌ BAD: Skip 10,000 results (extremely slow)
results = search_client.search(
    search_text="laptop",
    skip=10000,  # Azure Search must process and discard 10,000 results!
    top=10
)
# Latency: 2,000-5,000ms (unusable)
```

**Azure Search Limit**: Maximum `skip` value is 100,000 (enforced)

---

### 6. Query Complexity Management

**DO: Simplify overly complex queries**

```python
# ✅ GOOD: Focused, simple query (fast)
results = search_client.search(
    search_text="gaming laptop",
    search_mode="all",  # Both terms must match
    top=10
)
# Latency: 45ms

# ❌ BAD: Overly complex query (slow, over-engineered)
results = search_client.search(
    search_text="gaming OR (laptop AND (performance OR powerful) AND NOT cheap) OR workstation",
    search_mode="any",
    query_type="full",  # Lucene syntax (slower to parse)
    top=10
)
# Latency: 180ms (4× slower)
```

**Complexity Impact**:
- Simple query (2-3 terms): 40-60ms
- Medium query (5-7 terms, basic Boolean): 80-120ms
- Complex query (10+ terms, nested Boolean): 150-300ms
- Regex/wildcard query: 200-500ms (avoid in production)

**DON'T: Use leading wildcards**

```python
# ❌ BAD: Leading wildcard (scans entire index)
results = search_client.search(
    search_text="*book",  # Matches: textbook, notebook, facebook, ...
    top=10
)
# Latency: 800-2,000ms (100-1000× slower!)

# ✅ GOOD: Trailing wildcard (uses index efficiently)
results = search_client.search(
    search_text="book*",  # Matches: book, books, booklet, ...
    top=10
)
# Latency: 55ms
```

**Reason**: Leading wildcards require full index scan (no optimization possible)

---

### 7. Monitoring and Performance Tracking

**DO: Track P95/P99 latency, not just average**

```python
# ✅ GOOD: Comprehensive latency tracking
class LatencyTracker:
    def __init__(self):
        self.latencies = []
    
    def record(self, latency_ms):
        self.latencies.append(latency_ms)
    
    def get_stats(self):
        if not self.latencies:
            return {}
        
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        return {
            'count': n,
            'mean': sum(sorted_latencies) / n,
            'median': sorted_latencies[n // 2],
            'p50': sorted_latencies[int(n * 0.50)],
            'p75': sorted_latencies[int(n * 0.75)],
            'p90': sorted_latencies[int(n * 0.90)],
            'p95': sorted_latencies[int(n * 0.95)],  # ← Most important!
            'p99': sorted_latencies[int(n * 0.99)],  # ← Second most important!
            'max': sorted_latencies[-1]
        }

tracker = LatencyTracker()
# ... record latencies ...
stats = tracker.get_stats()
print(f"P95: {stats['p95']:.0f}ms, P99: {stats['p99']:.0f}ms")
```

**Why P95/P99 matter more than average**:
- **Average (mean)**: Hides outliers, 50% of users experience worse
- **P95**: 95% of queries are faster (only 5% slower)
- **P99**: 99% of queries are faster (only 1% slower)
- **Production SLOs**: Typically set at P95 or P99, not mean

**Example**:
- 100 queries: 99 at 50ms, 1 at 5,000ms
- Mean: 100ms (misleading!)
- P95/P99: 50ms (accurate representation)

**DON'T: Rely solely on average latency**

Average latency can be excellent while user experience is poor:
```
Scenario: E-commerce search
- Average latency: 85ms (excellent!)
- P95 latency: 850ms (terrible!)
- Impact: 5% of users experience slow searches (high bounce rate)
```

---

### 8. Resource Scaling and Capacity Planning

**DO: Scale with replicas for read performance**

```python
# Capacity planning formula
def calculate_replicas_needed(target_qps, avg_query_latency_ms, target_p95_ms=200):
    """
    Calculate replicas needed for target QPS and latency.
    
    Args:
        target_qps: Desired queries per second
        avg_query_latency_ms: Average query latency
        target_p95_ms: Target P95 latency (default 200ms)
    
    Returns:
        Recommended replica count
    """
    # Queries per second per replica (rough estimate)
    # Assumes query can complete in avg_query_latency_ms
    qps_per_replica = 1000 / avg_query_latency_ms
    
    # Add 30% buffer for burst traffic and P95 target
    buffer_factor = 1.3
    
    replicas_needed = (target_qps / qps_per_replica) * buffer_factor
    
    # Round up, minimum 2 for HA
    replicas = max(2, int(replicas_needed + 0.5))
    
    return {
        'recommended_replicas': replicas,
        'estimated_qps': replicas * qps_per_replica / buffer_factor,
        'cost_per_replica': '$250/month (S1), $1000/month (S2), $2000/month (S3)',
        'note': 'Add +1 replica for zero-downtime updates'
    }

# Example usage
plan = calculate_replicas_needed(target_qps=500, avg_query_latency_ms=60)
print(f"Need {plan['recommended_replicas']} replicas for 500 QPS @ 60ms latency")
print(f"Estimated capacity: {plan['estimated_qps']:.0f} QPS")
```

**Scaling Guidelines**:

| Tier | QPS per Replica | Latency | When to Use |
|------|----------------|---------|-------------|
| Basic | ~50 QPS | 100-200ms | Development, low traffic |
| S1 | ~100 QPS | 60-120ms | Small-medium production |
| S2 | ~200 QPS | 40-80ms | Medium-large production |
| S3 | ~400 QPS | 30-60ms | Large production, low latency |

**DON'T: Use partitions for query performance**

```python
# ❌ WRONG: Partitions don't help query latency
# Partitions are for INDEX SIZE, not QUERY PERFORMANCE
# Adding partitions may SLOW DOWN queries (scatter-gather)

# ✅ RIGHT: Replicas improve query throughput
# Each replica handles queries independently
# 3 replicas = 3× query capacity
```

---

### 9. Embedding Cache for Vector Search

**DO: Cache query embeddings for common searches**

```python
# ✅ GOOD: Cache embeddings for frequently searched queries
class EmbeddingCacheOptimized:
    def __init__(self, openai_client, cache_size=5000):
        self.openai_client = openai_client
        self.cache = {}
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0
    
    def get_embedding(self, text, model="text-embedding-3-large"):
        cache_key = f"{text}:{model}"
        
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]
        
        # Generate embedding
        self.misses += 1
        response = self.openai_client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
        
        # Cache with LRU eviction
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (FIFO for simplicity)
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[cache_key] = embedding
        return embedding
    
    def get_savings(self, cost_per_1k_tokens=0.13, avg_tokens_per_query=10):
        """Calculate cost savings from caching."""
        total_calls = self.hits + self.misses
        if total_calls == 0:
            return {}
        
        hit_rate = self.hits / total_calls
        
        # Cost without caching
        cost_without_cache = total_calls * (avg_tokens_per_query / 1000) * cost_per_1k_tokens
        
        # Cost with caching (only misses generate API calls)
        cost_with_cache = self.misses * (avg_tokens_per_query / 1000) * cost_per_1k_tokens
        
        savings = cost_without_cache - cost_with_cache
        
        return {
            'hit_rate': f"{hit_rate:.1%}",
            'total_calls': total_calls,
            'api_calls_saved': self.hits,
            'cost_without_cache': f"${cost_without_cache:.2f}",
            'cost_with_cache': f"${cost_with_cache:.2f}",
            'savings': f"${savings:.2f}",
            'savings_pct': f"{(savings/cost_without_cache*100):.0f}%" if cost_without_cache > 0 else "0%"
        }
```

**Impact** (from Contoso case study):
- Common queries: "Seattle", "2 bedroom", "under 500k" → cached
- Hit rate: 58%
- API calls saved: 29M/month
- Cost savings: $3.77/month (small, but adds up at scale)
- **Latency savings: 25-40ms** (more important than cost)

**DON'T: Generate embeddings for every query without caching**

At high scale, uncached embeddings become expensive:
```
Example: 50M queries/month
- Embedding cost: 50M × 10 tokens × $0.13/1M = $65/month
- With 60% cache hit rate: $26/month (savings: $39/month = $468/year)
- Latency improvement: 60% of queries skip 30ms embedding call
```

---

### 10. Query Result Caching Best Practices

**DO: Implement cache warming for predictable queries**

```python
# ✅ GOOD: Pre-warm cache with common queries
class CacheWarmer:
    def __init__(self, search_client, cache):
        self.search_client = search_client
        self.cache = cache
    
    def warm_cache(self, common_queries):
        """
        Pre-populate cache with common queries.
        
        Run during off-peak hours or on app startup.
        """
        for query in common_queries:
            # Execute search and cache result
            results = list(self.search_client.search(search_text=query, top=10))
            # Cache automatically stores it
            self.cache.store(query, results)
    
    def get_common_queries_from_logs(self, log_file, top_n=1000):
        """Extract top N queries from search logs."""
        from collections import Counter
        
        queries = []
        with open(log_file) as f:
            for line in f:
                # Parse query from log
                query = self._extract_query(line)
                queries.append(query)
        
        # Return top N most common
        counter = Counter(queries)
        return [q for q, count in counter.most_common(top_n)]

# Usage: Warm cache on startup or scheduled job
warmer = CacheWarmer(search_client, cache)
common_queries = warmer.get_common_queries_from_logs("search_logs.txt", top_n=1000)
warmer.warm_cache(common_queries)
```

**Benefits**:
- First user doesn't experience cache miss
- Predictable performance from startup
- Higher effective hit rate (60% → 75%+)

---

## Troubleshooting

### Issue 1: High Query Latency (P95 > 200ms)

**Symptoms**:
- Slow search responses, especially at peak load
- P95/P99 latency significantly higher than P50
- User complaints about "slow search"
- High bounce rate on search pages

**Diagnosis Steps**:

1. **Check which percentile is slow**:
```python
# Measure latency distribution
latencies = []
for _ in range(100):
    start = time.time()
    results = search_client.search(search_text="laptop", top=10)
    list(results)
    latencies.append((time.time() - start) * 1000)

latencies.sort()
print(f"P50: {latencies[50]:.0f}ms")
print(f"P95: {latencies[95]:.0f}ms")
print(f"P99: {latencies[99]:.0f}ms")

# If P95 >> P50: Inconsistent performance (cache misses, cold queries)
# If P50 high: Systematic problem (all queries slow)
```

2. **Profile query components**:
```python
# Measure each component
import time

# Full query
start = time.time()
results = list(search_client.search(search_text="laptop", top=10))
total_time = (time.time() - start) * 1000

# Without select (all fields)
start = time.time()
results_all = list(search_client.search(search_text="laptop", top=10))
all_fields_time = (time.time() - start) * 1000

# With select (minimal fields)
start = time.time()
results_minimal = list(search_client.search(
    search_text="laptop",
    select="id,title",
    top=10
))
minimal_fields_time = (time.time() - start) * 1000

print(f"All fields: {all_fields_time:.0f}ms")
print(f"Minimal fields: {minimal_fields_time:.0f}ms")
print(f"Overhead from extra fields: {all_fields_time - minimal_fields_time:.0f}ms")
```

**Common Root Causes & Solutions**:

**Cause 1: Fetching unnecessary fields**
- Symptom: Large response payloads (>10KB per result)
- Solution: Add `select` parameter with only needed fields
- Impact: 2-7× faster, 40-60ms latency reduction

**Cause 2: No caching**
- Symptom: Every query hits search service (no cache hits)
- Solution: Implement query result caching (5-minute TTL)
- Impact: 60-80% queries served from cache, <10ms latency

**Cause 3: Insufficient replicas**
- Symptom: Latency increases at peak load (QPS spikes)
- Solution: Add replicas (each replica adds ~100-400 QPS capacity)
- Impact: Linear throughput increase, P95 latency reduction

**Cause 4: Complex queries**
- Symptom: Queries with many terms/wildcards are slow
- Solution: Simplify query logic, avoid leading wildcards, limit terms
- Impact: 2-5× faster for complex queries

**Cause 5: Searching too many fields**
- Symptom: Full-text search across 20+ fields
- Solution: Use `search_fields` to limit to 3-5 relevant fields
- Impact: 2-4× faster, more precise results

**Step-by-Step Fix**:
```python
# 1. Add field selection (quick win)
results = search_client.search(
    search_text="laptop",
    select="id,title,price,rating",  # ← Add this
    top=10
)

# 2. Add caching (medium effort)
cache = SearchCache(search_client)
results = cache.search(search_text="laptop", top=10)

# 3. Limit search scope (quick win)
results = search_client.search(
    search_text="laptop",
    search_fields="title,description",  # ← Add this
    select="id,title,price,rating",
    top=10
)

# 4. If still slow, add replicas (requires Azure portal)
# Go to Azure Portal → Scale → Add replicas (2→3→4)
```

---

### Issue 2: Low Cache Hit Rate (<40%)

**Symptoms**:
- Cache hit rate below 40-50%
- Most queries result in cache misses
- Caching not providing expected latency improvement
- High load on search service despite caching

**Diagnosis**:

```python
# Check cache statistics
stats = cache.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1f}%")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")

# Analyze query distribution
from collections import Counter
query_log = []  # Collect queries from logs
query_counts = Counter(query_log)

# Top 100 queries account for what % of traffic?
top_100_count = sum(count for query, count in query_counts.most_common(100))
total_count = sum(query_counts.values())
concentration = top_100_count / total_count * 100

print(f"Top 100 queries: {concentration:.0f}% of traffic")
# If < 30%: Query distribution too diverse for simple caching
# If > 60%: Should achieve >60% hit rate with proper caching
```

**Common Root Causes & Solutions**:

**Cause 1: TTL too short**
- Symptom: Cache entries expire before reuse
- Current: 30-second TTL → entries expire quickly
- Solution: Increase TTL to 5-10 minutes for stable data
- Impact: Hit rate 35% → 65%

**Cause 2: Cache key includes unnecessary parameters**
- Symptom: Identical queries generate different cache keys
```python
# ❌ BAD: Different keys for same search
cache_key_1 = hash("query=laptop&timestamp=1234567890&session_id=abc")
cache_key_2 = hash("query=laptop&timestamp=1234567891&session_id=xyz")
# These are the same search but different keys!

# ✅ GOOD: Only include parameters that affect results
cache_key = hash("query=laptop&top=10&filter=price le 1000")
```
- Solution: Normalize cache keys (exclude timestamp, session, etc.)
- Impact: Hit rate 40% → 70%

**Cause 3: Query distribution too diverse**
- Symptom: Users searching many unique queries (long tail)
- Example: 1M unique queries, each used once
- Solution: Only cache top N (e.g., top 10,000) queries
```python
class SelectiveCache:
    def __init__(self, popular_queries):
        self.cache = {}
        self.popular = set(popular_queries)  # Top 10K queries
    
    def search(self, query):
        # Only cache popular queries
        if query in self.popular:
            return self.cached_search(query)
        else:
            return self.direct_search(query)
```
- Impact: Focus cache on queries that will be reused

**Cause 4: Cache size too small**
- Symptom: Cache evicts entries before they're reused (thrashing)
- Current: 1,000 entry cache for 50,000 unique queries
- Solution: Increase cache size to 10,000-50,000 entries
```python
# Calculate optimal cache size
unique_queries_per_hour = 10000
reuse_window_hours = 1
cache_size = unique_queries_per_hour * reuse_window_hours
# Size cache to hold all queries within reuse window
```
- Impact: Hit rate 45% → 62%

**Step-by-Step Fix**:
```python
# 1. Analyze query distribution
from collections import Counter
queries = []  # Load from logs
counter = Counter(queries)

# Find top N that cover 80% of traffic
sorted_queries = counter.most_common()
cumulative = 0
total = sum(counter.values())
for i, (query, count) in enumerate(sorted_queries):
    cumulative += count
    if cumulative / total >= 0.8:
        print(f"Top {i+1} queries cover 80% of traffic")
        break

# 2. Increase TTL for stable data
cache = SearchCache(search_client)
cache.l1_ttl = 300  # 5 minutes (was 60)
cache.l2_ttl = 1800  # 30 minutes (was 300)

# 3. Normalize cache keys
def normalize_cache_key(params):
    # Remove non-search parameters
    search_params = {
        k: v for k, v in params.items()
        if k in ['search_text', 'filter', 'top', 'select', 'search_fields']
    }
    return hashlib.md5(json.dumps(search_params, sort_keys=True).encode()).hexdigest()

# 4. Pre-warm cache with top queries
warmer = CacheWarmer(search_client, cache)
top_queries = [q for q, count in counter.most_common(1000)]
warmer.warm_cache(top_queries)
```

---

### Issue 3: Filter Queries Slow Despite Indexed Fields

**Symptoms**:
- Queries with filters take 200-500ms
- Filter fields are marked `filterable: true` in schema
- Simple equality filters (`eq`) are slow
- Filtering seems slower than full-text search

**Diagnosis**:

```python
# Test filter vs no-filter performance
import time

# No filter
start = time.time()
results_no_filter = list(search_client.search(search_text="laptop", top=10))
no_filter_time = (time.time() - start) * 1000

# With filter
start = time.time()
results_with_filter = list(search_client.search(
    search_text="laptop",
    filter="price le 1000",
    top=10
))
filter_time = (time.time() - start) * 1000

print(f"No filter: {no_filter_time:.0f}ms")
print(f"With filter: {filter_time:.0f}ms")

# Filter should be faster (reduces search scope) or similar
# If filter is SLOWER: Problem!
```

**Common Root Causes & Solutions**:

**Cause 1: Filter applied AFTER search instead of BEFORE**
- Symptom: Filter adds latency instead of reducing it
- Cause: Using `$filter` in wrong order or post-filtering
```python
# ❌ BAD: Post-filter (searches all, then filters)
results = search_client.search(search_text="*", top=50000)
filtered = [r for r in results if r['price'] <= 1000][:10]

# ✅ GOOD: Pre-filter (filters first, then searches)
results = search_client.search(
    search_text="laptop",
    filter="price le 1000",
    top=10
)
```
- Solution: Always use `filter` parameter in search query
- Impact: 5-10× faster (85ms → 15ms)

**Cause 2: Filter on non-indexed field**
- Symptom: Slow filter despite `filterable: true`
- Cause: Field index not built (check schema deployment)
```python
# Verify field is actually filterable
index_def = search_index_client.get_index("products")
price_field = next(f for f in index_def.fields if f.name == "price")
print(f"Price filterable: {price_field.filterable}")  # Should be True

# If False: Update schema and rebuild index
price_field.filterable = True
search_index_client.create_or_update_index(index_def)
```
- Solution: Rebuild index with correct schema
- Impact: 100× faster (non-indexed → indexed)

**Cause 3: Complex filter expression**
- Symptom: Simple `eq` filters are fast, complex `and/or` are slow
```python
# Fast filter (simple)
filter="category eq 'Laptops'"  # 20ms

# Slow filter (complex)
filter="(category eq 'Laptops' OR category eq 'Tablets') AND (price le 1000 OR rating ge 4.5) AND (inStock eq true OR backorder eq true)"  # 180ms
```
- Solution: Simplify filter or denormalize data
```python
# Denormalize: Add computed field "affordable_tech_in_stock"
# Pre-compute: category in ['Laptops','Tablets'] AND (price <= 1000 OR rating >= 4.5) AND (inStock OR backorder)
filter="affordable_tech_in_stock eq true"  # 25ms
```
- Impact: 5-8× faster for complex filters

**Cause 4: Filter selectivity too low**
- Symptom: Filter doesn't reduce search scope much
```python
# Check filter selectivity
total_docs = 1000000
filtered_docs = 950000  # Filter only excludes 5%!
selectivity = filtered_docs / total_docs  # 0.95 (very poor)

# Filter barely helps because 95% of docs pass filter
```
- Solution: Add more selective filter or remove ineffective filter
```python
# More selective filter
filter="price le 500"  # Only 15% of docs (good selectivity)
```
- Impact: Better filters = 2-5× faster

**Step-by-Step Fix**:
```python
# 1. Verify field schema
index_def = search_index_client.get_index("products")
for field in index_def.fields:
    if field.name in ['price', 'category', 'inStock']:
        print(f"{field.name}: filterable={field.filterable}, type={field.type}")

# 2. Ensure filter is in query, not post-processing
results = search_client.search(
    search_text="laptop",
    filter="price le 1000",  # ← In query
    top=10
)
# NOT: results = [r for r in search_client.search(...) if r['price'] <= 1000]

# 3. Simplify complex filters
# Before: "((A AND B) OR (C AND D)) AND (E OR F)"
# After: Pre-compute "qualifies = True" during indexing
filter="qualifies eq true"

# 4. Test filter selectivity
def test_selectivity(filter_expr):
    total = search_client.search(search_text="*", top=0, include_total_count=True).get_count()
    filtered = search_client.search(search_text="*", filter=filter_expr, top=0, include_total_count=True).get_count()
    selectivity = filtered / total
    print(f"Selectivity: {selectivity:.1%} ({filtered:,} / {total:,})")
    if selectivity > 0.7:
        print("⚠️ WARNING: Filter not selective (>70% pass)")

test_selectivity("price le 1000")
```

---

### Issue 4: Pagination Slow for Deep Pages

**Symptoms**:
- First page (1-10) is fast: 50ms
- Page 10 (90-100) is slow: 150ms
- Page 50 (490-500) is very slow: 450ms
- Page 100 (990-1000) times out: >2000ms

**Diagnosis**:

```python
# Measure pagination latency at different depths
def measure_pagination_depth(search_client, query, page_sizes=[1, 10, 50, 100]):
    import time
    
    for page in page_sizes:
        skip = (page - 1) * 10
        start = time.time()
        results = list(search_client.search(
            search_text=query,
            skip=skip,
            top=10
        ))
        latency = (time.time() - start) * 1000
        print(f"Page {page} (skip={skip}): {latency:.0f}ms")

measure_pagination_depth(search_client, "laptop")
# If latency increases linearly with skip: Offset-based pagination problem
```

**Root Cause**:
- Azure Search must process and rank `skip + top` documents
- Page 100 with `top=10`: Process 1000 docs, return last 10
- Deep pagination inherently slow with offset method

**Solutions**:

**Solution 1: Use continuation tokens (BEST)**
```python
# ✅ GOOD: Constant time per page
def paginate_efficient(search_client, query, pages_to_fetch=5):
    results = search_client.search(search_text=query, top=10)
    
    page_num = 0
    for page in results.by_page():
        page_num += 1
        documents = list(page)
        print(f"Page {page_num}: {len(documents)} docs")
        
        if page_num >= pages_to_fetch:
            break

paginate_efficient(search_client, "laptop", pages_to_fetch=100)
# All pages take ~50-60ms, regardless of depth
```

**Solution 2: Implement search_after (ALTERNATIVE)**
```python
# Use orderby with a unique field
last_sort_value = None

for page in range(1, 101):
    results = search_client.search(
        search_text="laptop",
        order_by="id asc",  # Must sort by unique field
        top=10,
        search_after=last_sort_value  # Start after last result
    )
    
    docs = list(results)
    if docs:
        last_sort_value = docs[-1]['id']
    
    print(f"Page {page}: {len(docs)} docs")
```

**Solution 3: Limit maximum page depth**
```python
# Prevent deep pagination entirely
MAX_SKIP = 500  # Maximum 50 pages of 10 results

def safe_paginate(search_client, query, page, page_size=10):
    skip = (page - 1) * page_size
    
    if skip > MAX_SKIP:
        return {
            'error': 'Page too deep',
            'message': 'Please refine your search',
            'max_page': MAX_SKIP // page_size
        }
    
    results = search_client.search(
        search_text=query,
        skip=skip,
        top=page_size
    )
    
    return list(results)
```

**Solution 4: Encourage query refinement**
```python
# Instead of deep pagination, suggest refinements
if page > 10:
    # Show facets/filters to narrow results
    facets = ["category", "price,interval:100", "brand"]
    
    return {
        'message': 'Too many results. Please refine your search:',
        'suggestions': {
            'categories': ['Laptops (12,453)', 'Tablets (3,221)', ...],
            'price_ranges': ['$0-$500 (2,341)', '$500-$1000 (8,122)', ...],
            'brands': ['Dell (3,211)', 'HP (2,876)', ...]
        }
    }
```

**Performance Comparison**:

| Method | Page 1 | Page 10 | Page 50 | Page 100 | Notes |
|--------|--------|---------|---------|----------|-------|
| Offset (`skip`) | 50ms | 150ms | 450ms | 900ms | Linear degradation |
| Continuation | 50ms | 52ms | 55ms | 58ms | ✅ Constant time |
| Search After | 50ms | 53ms | 56ms | 60ms | ✅ Constant time |
| Limit depth | 50ms | 150ms | N/A | N/A | Force refinement |

**Recommended Approach**:
1. **Default**: Use continuation tokens (best UX + performance)
2. **Export/API**: Use search_after with stable sort field
3. **UI/Web**: Limit to first 50 pages, show filters for refinement

---

### Issue 5: Embedding Generation Bottleneck for Vector Search

**Symptoms**:
- Vector search queries take 150-250ms
- Embedding generation adds 25-40ms per query
- High OpenAI API costs for embeddings
- P95 latency spikes during embedding generation

**Diagnosis**:

```python
# Measure embedding time vs search time
import time

# Total vector search time
query_text = "laptop for machine learning"
start = time.time()

# 1. Generate embedding
embedding_start = time.time()
response = openai_client.embeddings.create(
    input=query_text,
    model="text-embedding-3-large"
)
embedding = response.data[0].embedding
embedding_time = (time.time() - embedding_start) * 1000

# 2. Vector search
search_start = time.time()
results = search_client.search(
    vector_queries=[VectorQuery(vector=embedding, k=10, fields="contentVector")],
    top=10
)
list(results)
search_time = (time.time() - search_start) * 1000

total_time = (time.time() - start) * 1000

print(f"Embedding generation: {embedding_time:.0f}ms ({embedding_time/total_time*100:.0f}%)")
print(f"Vector search: {search_time:.0f}ms ({search_time/total_time*100:.0f}%)")
print(f"Total: {total_time:.0f}ms")

# If embedding time > 20% of total: Optimize embeddings
```

**Common Root Causes & Solutions**:

**Cause 1: No embedding cache**
- Symptom: Every query generates new embedding (25-40ms overhead)
- Solution: Cache embeddings for common queries
```python
class EmbeddingCache:
    def __init__(self, openai_client, cache_size=5000):
        self.client = openai_client
        self.cache = {}
        self.max_size = cache_size
    
    def get_embedding(self, text, model="text-embedding-3-large"):
        key = f"{text}:{model}"
        
        if key in self.cache:
            return self.cache[key]  # 0ms (cache hit)
        
        # Generate and cache
        response = self.client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
        
        # LRU eviction
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = embedding
        return embedding

# Usage
cache = EmbeddingCache(openai_client)
embedding = cache.get_embedding("laptop")  # 30ms first time
embedding = cache.get_embedding("laptop")  # <1ms cached
```
- Impact: 50-70% hit rate → 15-20ms avg latency reduction

**Cause 2: Using larger model than necessary**
- Symptom: Embeddings take 35-45ms per call
- Current: text-embedding-3-large (3072 dimensions)
- Solution: Use smaller model for lower latency
```python
# Model comparison
models = {
    'text-embedding-3-small': {'dims': 512, 'latency': '15-20ms', 'quality': 'good'},
    'text-embedding-3-large': {'dims': 3072, 'latency': '30-40ms', 'quality': 'best'},
    'text-embedding-ada-002': {'dims': 1536, 'latency': '20-30ms', 'quality': 'very good'}
}

# For most use cases: text-embedding-ada-002 or text-embedding-3-small
```
- Impact: 30-40ms → 15-20ms (2× faster)
- Trade-off: Slightly lower quality (test with your data)

**Cause 3: Sequential embedding + search**
- Symptom: Total time = embedding time + search time (additive)
- Solution: Hybrid search (BM25 + vector in parallel)
```python
# If embedding fails or is slow, fall back to BM25
import asyncio

async def hybrid_search_with_fallback(query_text):
    try:
        # Try vector search (may be slow)
        embedding_task = asyncio.create_task(generate_embedding_async(query_text))
        
        # Simultaneously start BM25 search
        bm25_results = await search_bm25_async(query_text)
        
        # Wait for embedding (with timeout)
        try:
            embedding = await asyncio.wait_for(embedding_task, timeout=0.1)  # 100ms max
            vector_results = await search_vector_async(embedding)
            
            # Fuse results
            return fuse_results(bm25_results, vector_results)
        except asyncio.TimeoutError:
            # Embedding too slow - use BM25 only
            return bm25_results
    
    except Exception:
        # Embedding failed - fall back to BM25
        return await search_bm25_async(query_text)
```
- Impact: Graceful degradation, no query failures

**Cause 4: Network latency to OpenAI**
- Symptom: Embedding calls take 40-60ms (high latency)
- Cause: Network distance to OpenAI endpoint
- Solution: Use Azure OpenAI in same region as search
```python
# Instead of OpenAI.com (US West)
openai_client = OpenAI(api_key="...")  # Network RTT: 30-50ms from Europe

# Use Azure OpenAI in same region as search
from openai import AzureOpenAI
openai_client = AzureOpenAI(
    api_key="...",
    api_version="2024-02-01",
    azure_endpoint="https://your-resource.openai.azure.com"  # Same region: 5-10ms RTT
)
```
- Impact: 40-60ms → 15-25ms latency

**Step-by-Step Fix**:
```python
# 1. Implement embedding cache
embedding_cache = EmbeddingCache(openai_client, cache_size=10000)

# 2. Use appropriate model size
embedding = embedding_cache.get_embedding(
    text=query_text,
    model="text-embedding-ada-002"  # Or text-embedding-3-small
)

# 3. Add fallback to BM25
try:
    embedding = embedding_cache.get_embedding(query_text)
    results = vector_search(embedding)
except Exception:
    results = bm25_search(query_text)  # Fallback

# 4. Pre-generate embeddings for common queries
common_queries = ["laptop", "gaming pc", "affordable phone", ...]
for query in common_queries:
    embedding_cache.get_embedding(query)  # Pre-warm cache
```

---

## Next Steps

- **[Custom Analyzers](./13-custom-analyzers.md)** - Optimize text analysis
- **[Scoring Profiles](./14-scoring-profiles.md)** - Advanced relevance tuning
- **[Load Testing](./18-load-testing.md)** - Performance validation

---

*See also: [Azure Monitor](./07-azure-monitor-logging.md) | [Index Management](./15-index-management.md)*