# Query Optimization

Complete guide to optimizing Azure AI Search query performance, reducing latency, and improving efficiency.

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

---

## Overview

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

### ✅ Do's
1. **Always use select parameter** to limit returned fields
2. **Cache frequently used queries** with appropriate TTL
3. **Apply filters before search** to reduce scope
4. **Use continuation tokens** for deep pagination
5. **Monitor P95/P99 latency** not just average
6. **Add replicas** for high query volume
7. **Profile queries** before and after optimization

### ❌ Don'ts
1. **Don't** fetch all fields if you only need a few
2. **Don't** use high skip values for pagination (use continuation)
3. **Don't** filter in application code (filter in query)
4. **Don't** ignore cache hit rates (should be >60%)
5. **Don't** over-fetch results (use appropriate top value)
6. **Don't** search all fields (use search_fields parameter)
7. **Don't** forget to invalidate cache when index updates

---

## Next Steps

- **[Custom Analyzers](./13-custom-analyzers.md)** - Optimize text analysis
- **[Scoring Profiles](./14-scoring-profiles.md)** - Advanced relevance tuning
- **[Load Testing](./18-load-testing.md)** - Performance validation

---

*See also: [Azure Monitor](./07-azure-monitor-logging.md) | [Index Management](./15-index-management.md)*