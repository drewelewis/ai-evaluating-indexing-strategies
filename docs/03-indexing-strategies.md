# Azure AI Search Indexing Strategies

**Purpose**: Understand which Azure AI Search indexing strategy to choose for your use case, how to implement it cost-effectively, and how to migrate between strategies as your needs evolve.

**Target Audience**: Search engineers, solution architects, Azure developers, product managers evaluating search options

**Reading Time**: 25-30 minutes

---

## Why Indexing Strategy Choice Matters

Choosing the wrong indexing strategy costs time, money, and user satisfaction. The difference between BM25, vector, hybrid, and semantic search isn't just technical—it's strategic.

**Real-world impact of strategy choice**:
```
Scenario: Mortgage lender with 50K documents, 10K searches/day

Wrong Choice (Vector-only search):
├── Decision: "Semantic understanding sounds great, let's go all-in on vectors"
├── Implementation cost: $15K (data prep, embeddings, tuning)
├── Monthly Azure costs: $2,800 (S2 tier + OpenAI embeddings)
├── User impact: 
│   ├── Navigational queries broken ("Home" returns articles about homes, not homepage)
│   ├── Exact match queries degraded ("Form 1003" returns similar forms, not Form 1003)
│   └── CTR dropped 18%, conversion dropped 12%
└── Total cost of mistake: $80K lost revenue (3 months before rollback)

Right Choice (Hybrid search):
├── Decision: "Use keyword for exact/navigational, vector for semantic, let hybrid balance"
├── Implementation cost: $18K (slightly more complex, but worth it)
├── Monthly Azure costs: $1,600 (S1 tier + selective embeddings)
├── User impact:
│   ├── Navigational queries perfect (keyword matching)
│   ├── Semantic queries improved (vector matching)
│   └── CTR improved 25%, conversion improved 15%
└── ROI: $250K additional revenue in first year

The difference: Understanding when to use each strategy.
```

**The three truths about indexing strategies**:

1. **No strategy is universally best**: BM25 wins for exact matching, vectors win for semantic understanding, hybrid wins for mixed workloads
2. **Cost scales dramatically**: BM25 = $300/month, Hybrid = $1,500/month, Semantic = $3,000/month (S1-S2 tier assumptions)
3. **Migration is expensive**: Switching strategies mid-flight costs weeks of engineering time—choose wisely upfront

This document provides production-ready guidance for selecting, implementing, and optimizing Azure AI Search indexing strategies.

---

This document provides a comprehensive comparison of Azure AI Search indexing strategies, their performance characteristics, and optimal use cases for Azure-native implementations.

## 🔍 Azure AI Search Overview

**What is Azure AI Search**: A fully managed cloud search service that provides multiple search paradigms out of the box. Unlike building search from scratch with Elasticsearch or Solr, Azure AI Search handles infrastructure, scaling, and integration with Azure AI services.

**Why Azure AI Search for evaluation**: This guide focuses on Azure AI Search because:
- Native integration with Azure OpenAI (embeddings, GPT models)
- Built-in hybrid search (no custom code to merge keyword + vector results)
- Semantic ranking powered by Microsoft's AI models
- Pay-as-you-go pricing (scale from $0/month free tier to enterprise $10K+/month)
- Simplified operations (Microsoft manages infrastructure, updates, security)

Azure AI Search (formerly Azure Cognitive Search) supports multiple search paradigms that can be evaluated and optimized:

1. **Full-text search** with BM25 scoring
   - *When to use*: Exact term matching, navigational queries, structured content (product codes, IDs, names)
   - *Example*: "FHA loan application form 1003" → Exact match on form number

2. **Vector search** with Azure OpenAI embeddings  
   - *When to use*: Semantic understanding, conceptual queries, when users don't know exact terminology
   - *Example*: "government assistance for first home buyers" → Matches "FHA loan" documents even without exact terms

3. **Hybrid search** combining text and vectors
   - *When to use*: Mixed query workload (some navigational, some semantic), want best of both worlds
   - *Example*: Handles both "Form 1003" (keyword) and "first-time buyer assistance" (semantic) well

4. **Semantic search** with AI-powered re-ranking
   - *When to use*: Willing to pay premium for best relevance, complex queries, multi-paragraph content
   - *Example*: "Should I refinance my 30-year fixed mortgage with 4.5% rate if rates dropped to 3.5%?" → Deep understanding

### Azure Search Strategy Comparison Matrix

**How to read this table**: Choose based on your constraints:
- **Latency-critical** (< 50ms) → BM25
- **Budget-constrained** (< $500/month) → BM25 or limited vector search
- **Semantic understanding required** → Vector or Hybrid
- **Mixed workload** → Hybrid
- **Premium quality** → Semantic

| Strategy | Latency | Recall | Precision | Semantic Understanding | Azure Setup | Monthly Cost (S1) |
|----------|---------|--------|-----------|----------------------|-------------|-------------------|
| **Full-text (BM25)** | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⚡ | $300 |
| **Vector Search** | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⚡⚡ | $800-1500 |
| **Hybrid Search** | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⚡⭐⚡ | $1000-2000 |
| **Semantic Search** | ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚡⚡ | $1500-3000 |

*Legend: ⚡ = Fast/Simple/Cheap, ⭐ = Good performance*
*Cost includes Azure AI Search + Azure OpenAI + semantic search fees*

**Cost breakdown explanation**:
```
BM25 ($300/month):
└── Azure AI Search S1 tier only (no embeddings, no semantic ranking)

Vector Search ($800-1500/month):
├── Azure AI Search S1: $300
├── Azure OpenAI embeddings: $200-400 (depends on document volume)
├── Embedding storage: $100-200 (larger indexes due to vector dimensions)
└── Compute for vector similarity: $200-600 (HNSW index overhead)

Hybrid Search ($1000-2000/month):
├── All vector search costs above
├── Plus: RRF (Reciprocal Rank Fusion) orchestration overhead
└── Plus: 30-50% more index size (store both keyword and vector representations)

Semantic Search ($1500-3000/month):
├── All hybrid search costs above
├── Plus: Semantic ranking fees ($500/month base + $2 per 1K queries)
└── Plus: Higher tier often needed (S2) for performance ($1000/month vs $300 for S1)

Note: Costs scale with document volume, query volume, and feature usage.
50K documents, 10K searches/day is baseline. 1M documents, 100K searches/day would be 5-10x higher.
```

## 1. Azure AI Search Full-text (BM25) Strategy

**When BM25 is the right choice**:
- Navigational queries ("homepage", "contact us", "login")
- Exact product codes/IDs ("FHA-203B", "Form 1003", "SKU-12345")
- Short, keyword-heavy queries (average < 5 words)
- Budget constraints (need cheapest option)
- Latency requirements (need < 50ms P95)
- Structured content (product catalogs, documentation with clear terminology)

**When BM25 fails**:
```
User query: "government help for buying first home"
BM25 thinks: Must contain "government" AND "help" AND "buying" AND "first" AND "home"
Misses relevant document titled "FHA Loan Programs for First-Time Buyers"
└── Why: No word overlap with "government help for buying"

User query: "How do I refinance?"
BM25 thinks: Must contain "refinance"
Misses document titled "Switch your existing mortgage to a lower rate"
└── Why: Document uses different terminology

Lesson: BM25 requires exact term matching. If users and content use different words, BM25 struggles.
```

**BM25 strengths in practice**:
```
E-commerce product search:
├── Query: "iPhone 15 Pro Max 256GB"
├── BM25 result: Exact product page (perfect match)
├── Why it works: Users know exact product names, exact matching is ideal
└── Latency: 15ms

Documentation search:
├── Query: "How to configure SSL certificate"
├── BM25 result: Article titled "Configuring SSL Certificates"
├── Why it works: Technical docs use consistent terminology
└── Latency: 20ms

Form/ID lookup:
├── Query: "Form 1003"
├── BM25 result: Loan Application Form 1003
├── Why it works: Unique identifier, exact match is required
└── Latency: 10ms
```

### How Azure AI Search BM25 Works

**Azure's BM25 implementation**: Microsoft's variant of the Okapi BM25 algorithm, tuned for cloud-scale search with built-in analyzer chain.

```python
# Azure AI Search BM25 Configuration
azure_bm25_index = {
    "name": "fulltext-search-index",
    "fields": [
        {
            "name": "id",
            "type": "Edm.String",
            "key": True,
            "searchable": False  # Primary key, not searched
        },
        {
            "name": "content",
            "type": "Edm.String", 
            "searchable": True,
            "analyzer": "en.microsoft"  # Microsoft's English analyzer (better than standard)
        },
        {
            "name": "title",
            "type": "Edm.String",
            "searchable": True,
            "analyzer": "en.microsoft"
        },
        {
            "name": "category",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,  # Can filter by category
            "facetable": True    # Can generate facet counts
        },
        {
            "name": "lastModified",
            "type": "Edm.DateTimeOffset",
            "searchable": False,
            "filterable": True,
            "sortable": True  # Can sort by date
        }
    ],
    "scoringProfiles": [
        {
            "name": "boost-title",
            "text": {
                "weights": {
                    "title": 3.0,  # Title matches worth 3x content matches
                    "content": 1.0
                }
            },
            "functions": [
                {
                    "type": "freshness",  # Boost recent documents
                    "fieldName": "lastModified",
                    "boost": 2.0,
                    "interpolation": "linear",
                    "freshness": {
                        "boostingDuration": "P30D"  # 30-day boost window
                    }
                }
            ]
        }
    ],
    
    # Important: Analyzer choice impacts recall and precision
    "analyzers": [
        {
            "name": "custom_en_analyzer",
            "@odata.type": "#Microsoft.Azure.Search.CustomAnalyzer",
            "tokenizer": "standard",
            "tokenFilters": [
                "lowercase",  # "iPhone" = "iphone"
                "asciifolding",  # "café" = "cafe"
                "elision",  # Remove "l'" from "l'hotel"
                "stopwords_en",  # Remove "the", "a", "is"
                "englishStemmer"  # "running" = "run"
            ]
        }
    ]
}
```

**Analyzer impact on relevance** (often overlooked):
```python
# Without stemming
query = "running shoes"
matches = ["running shoes", "running shoe"]  # Misses "runners" or "run"

# With English stemming (en.microsoft analyzer)
query = "running shoes"  # Stemmed to: "run shoe"
matches = [
    "running shoes",  # Stemmed to: "run shoe" ✓
    "runner shoe",    # Stemmed to: "run shoe" ✓
    "run shoe",       # Stemmed to: "run shoe" ✓
    "runs in these shoes"  # Stemmed to: "run shoe" ✓
]

Impact: +15-30% recall improvement for natural language queries
Cost: Slightly lower precision (may match unwanted variations)
Recommendation: Use en.microsoft for general content, standard for technical IDs
```

### Azure Search Client Implementation

**Production-ready BM25 search client**:
```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
import logging

class AzureBM25Search:
    """
    Production-ready BM25 search client for Azure AI Search.
    
    Features:
    - Error handling and retries
    - Scoring profile support
    - Filtering and faceting
    - Logging for debugging
    """
    
    def __init__(self, endpoint, index_name, api_key):
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )
        self.logger = logging.getLogger(__name__)
    
    def search(self, query, top_k=10):
        """
        Perform BM25 search with Azure AI Search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        
        try:
            results = self.client.search(
                search_text=query,
                top=top_k,
                scoring_profile="boost-title",  # Use custom scoring profile
                query_type="simple"  # Or "full" for advanced syntax
            )
            
            return [
                {
                    "id": result["id"],
                    "title": result.get("title", ""),
                    "content": result["content"],
                    "score": result["@search.score"],  # BM25 score
                    "category": result.get("category", "")
                }
                for result in results
            ]
            
        except HttpResponseError as e:
            self.logger.error(f"Search failed: {e.message}")
            raise
    
    def search_with_filters(self, query, filters, top_k=10):
        """
        BM25 search with additional filters.
        
        Example:
            filters = {
                'category': 'mortgages',
                'lastModified': '> 2024-01-01'
            }
        """
        
        filter_expression = self._build_filter_expression(filters)
        
        try:
            results = self.client.search(
                search_text=query,
                filter=filter_expression,
                top=top_k,
                scoring_profile="boost-title"
            )
            
            return [
                {
                    "id": result["id"],
                    "title": result.get("title", ""),
                    "content": result["content"],
                    "score": result["@search.score"],
                    "category": result.get("category", "")
                }
                for result in results
            ]
            
        except HttpResponseError as e:
            self.logger.error(f"Filtered search failed: {e.message}")
            raise
    
    def _build_filter_expression(self, filters):
        """
        Build OData filter expression from dict.
        
        Azure AI Search uses OData syntax:
        - Equality: category eq 'mortgages'
        - Comparison: lastModified gt 2024-01-01
        - Multiple: category eq 'mortgages' and lastModified gt 2024-01-01
        """
        
        filter_parts = []
        
        for field, value in filters.items():
            if isinstance(value, str) and value.startswith(">"):
                # Comparison operator
                comp_value = value.split()[1]
                filter_parts.append(f"{field} gt {comp_value}")
            elif isinstance(value, str) and value.startswith("<"):
                comp_value = value.split()[1]
                filter_parts.append(f"{field} lt {comp_value}")
            else:
                # Equality
                filter_parts.append(f"{field} eq '{value}'")
        
        return " and ".join(filter_parts)
    
    def search_with_facets(self, query, facet_fields, top_k=10):
        """
        BM25 search with facet counts.
        
        Facets are useful for:
        - "Refine by category" UI
        - Understanding result distribution
        - Guided navigation
        
        Example:
            facet_fields = ['category', 'author']
        """
        
        try:
            results = self.client.search(
                search_text=query,
                top=top_k,
                facets=facet_fields
            )
            
            documents = []
            facets = {}
            
            for result in results:
                documents.append({
                    "id": result["id"],
                    "title": result.get("title", ""),
                    "content": result["content"],
                    "score": result["@search.score"]
                })
            
            # Extract facet counts
            if hasattr(results, 'get_facets'):
                facet_results = results.get_facets()
                for facet_field in facet_fields:
                    if facet_field in facet_results:
                        facets[facet_field] = [
                            {"value": f["value"], "count": f["count"]}
                            for f in facet_results[facet_field]
                        ]
            
            return {
                "documents": documents,
                "facets": facets,
                "total_count": len(documents)
            }
            
        except HttpResponseError as e:
            self.logger.error(f"Faceted search failed: {e.message}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize client
    search_client = AzureBM25Search(
        endpoint="https://your-service.search.windows.net",
        index_name="mortgage-docs",
        api_key="your-admin-key"
    )
    
    # Simple search
    results = search_client.search("FHA loan requirements", top_k=10)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.2f})")
    
    # Filtered search
    filtered_results = search_client.search_with_filters(
        query="refinancing options",
        filters={
            "category": "mortgages",
            "lastModified": "> 2024-01-01"
        },
        top_k=10
    )
    
    # Faceted search (for "refine by" UI)
    faceted_results = search_client.search_with_facets(
        query="mortgage calculator",
        facet_fields=["category", "documentType"],
        top_k=10
    )
    
    print(f"\nFound {faceted_results['total_count']} results")
    print("Refine by category:")
    for facet in faceted_results['facets'].get('category', []):
        print(f"  {facet['value']}: {facet['count']} documents")
```

**Common BM25 pitfalls and solutions**:
```python
pitfalls_and_solutions = {
    "pitfall_1_wrong_analyzer": {
        "problem": "Using 'standard' analyzer for natural language content",
        "symptom": "Poor recall - 'running shoes' doesn't match 'runner'",
        "solution": "Use 'en.microsoft' analyzer for English content (includes stemming)",
        "impact": "+15-30% recall improvement"
    },
    
    "pitfall_2_no_scoring_profile": {
        "problem": "Treating all fields equally",
        "symptom": "Body text matches rank higher than title matches",
        "solution": "Create scoring profile to boost title/metadata fields",
        "impact": "+10-20% precision improvement (better top results)"
    },
    
    "pitfall_3_ignoring_freshness": {
        "problem": "Old documents rank highest",
        "symptom": "2020 mortgage rates shown for 'current rates' query",
        "solution": "Add freshness function to scoring profile",
        "impact": "Much better user experience for time-sensitive queries"
    },
    
    "pitfall_4_over_filtering": {
        "problem": "Too strict filters result in zero results",
        "symptom": "Many queries return 0 results",
        "solution": "Use filters as post-ranking (not pre-filtering), or fall back to no-filter",
        "impact": "Reduce zero-result rate from 15% to <3%"
    }
}
```
        
        results = self.client.search(
            search_text=query,
            filter=filter_expression,
            top=top_k,
            scoring_profile="boost-title"
        )
        
        return list(results)
    
    def _build_filter_expression(self, filters):
        """Build OData filter expression for Azure AI Search."""
        expressions = []
        for field, value in filters.items():
            if isinstance(value, list):
                # OR condition for multiple values
                or_expressions = [f"{field} eq '{v}'" for v in value]
                expressions.append(f"({' or '.join(or_expressions)})")
            else:
                expressions.append(f"{field} eq '{value}'")
        
        return " and ".join(expressions)
```

### Performance Characteristics

**BM25 performance profile**:
```
Latency benchmarks (Azure AI Search S1):
├── Simple query ("mortgage"): 10-15ms P50, 20-30ms P95
├── Multi-term query ("FHA loan requirements"): 15-25ms P50, 30-50ms P95
├── Complex query with filters: 20-40ms P50, 50-100ms P95
└── Faceted search: 30-60ms P50, 80-150ms P95

Throughput (queries per second):
├── S1 tier: 1000-1500 QPS sustained
├── S2 tier: 3000-5000 QPS sustained
├── S3 tier: 8000-12000 QPS sustained
└── Bottleneck: Usually network/app tier, not search service

Accuracy on typical workloads:
├── Navigational queries (exact terms): Precision@5 = 0.85-0.95 (excellent)
├── Informational queries (natural language): Precision@5 = 0.60-0.75 (decent)
├── Mixed workload: Precision@5 = 0.65-0.75 (baseline)
└── Recall@10 = 0.70-0.85 (good for keyword matching)

Cost efficiency:
├── S1 tier: $300/month = $0.01 per 1000 queries (at 1M queries/month)
├── Most economical Azure search option
├── No embedding costs, no reranking fees
└── Best cost/performance for exact matching use cases
```

- **Latency**: 10-30ms for most queries
- **Throughput**: 1000+ QPS on Standard S1
- **Precision@5**: 0.65-0.75 for exact matches
- **Cost**: Most economical Azure search option
- **Best for**: Product catalogs, document search, exact term matching

### Azure BM25 Optimization

**How to tune BM25 for your workload**:
```python
class AzureBM25Optimizer:
    """
    Optimize BM25 scoring based on your query logs and user behavior.
    """
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def optimize_scoring_profile(self, query_click_data):
        """
        Optimize scoring profile based on click-through data.
        
        Args:
            query_click_data: List of {query, clicked_doc_id, clicked_doc_fields}
        
        Strategy:
        - Analyze which fields drive clicks
        - Boost high-performing fields
        - Reduce weight of low-performing fields
        """
        
        # Analyze field importance from user clicks
        field_weights = self._analyze_field_performance(query_click_data)
        
        optimized_profile = {
            "name": "optimized-scoring",
            "text": {
                "weights": {
                    "title": field_weights.get("title", 2.0),
                    "content": field_weights.get("content", 1.0),
                    "tags": field_weights.get("tags", 1.5)
                }
            },
            "functions": [
                {
                    "type": "freshness",
                    "fieldName": "lastModified",
                    "boost": 1.5,
                    "interpolation": "linear",
                    "freshness": {
                        "boostingDuration": "P30D"
                    }
                }
            ]
        }
        
        return optimized_profile
    
    def _analyze_field_performance(self, query_click_data):
        """
        Determine optimal field weights from click data.
        
        Logic:
        - If users click docs where query matched title → boost title weight
        - If users click docs where query matched content → boost content weight
        - Calculate correlation between field matches and clicks
        """
        
        field_click_correlation = {
            "title": 0,
            "content": 0,
            "tags": 0
        }
        
        for interaction in query_click_data:
            query = interaction['query']
            clicked_doc = interaction['clicked_doc']
            
            # Check which fields matched the query in clicked doc
            if query.lower() in clicked_doc.get('title', '').lower():
                field_click_correlation['title'] += 1
            
            if query.lower() in clicked_doc.get('content', '').lower():
                field_click_correlation['content'] += 1
            
            if any(query.lower() in tag.lower() for tag in clicked_doc.get('tags', [])):
                field_click_correlation['tags'] += 1
        
        # Normalize to weights (baseline = 1.0 for content)
        total_clicks = len(query_click_data)
        
        weights = {
            "title": (field_click_correlation['title'] / total_clicks) * 3.0,  # Scale up
            "content": 1.0,  # Baseline
            "tags": (field_click_correlation['tags'] / total_clicks) * 2.0
        }
        
        return weights
    
    def test_scoring_profiles(self, test_queries, profiles):
        """
        A/B test different scoring profiles offline.
        
        Use golden dataset to compare profile performance before deploying.
        """
        
        results = {}
        
        for profile_name, profile_config in profiles.items():
            ndcg_scores = []
            
            for query_data in test_queries:
                query = query_data['query']
                ground_truth = query_data['relevant_docs']
                
                # Search with this profile
                search_results = self.search_client.search(
                    search_text=query,
                    scoring_profile=profile_name,
                    top=10
                )
                
                # Calculate nDCG
                ndcg = self._calculate_ndcg(search_results, ground_truth)
                ndcg_scores.append(ndcg)
            
            results[profile_name] = {
                "mean_ndcg": sum(ndcg_scores) / len(ndcg_scores),
                "median_ndcg": sorted(ndcg_scores)[len(ndcg_scores) // 2]
            }
        
        return results

# Example: Optimizing based on logs
optimizer = AzureBM25Optimizer(search_client)

# Your click data from logs
click_data = [
    {"query": "FHA loan", "clicked_doc": {"title": "FHA Loan Guide", "content": "..."}},
    {"query": "refinance calculator", "clicked_doc": {"title": "Calculators", "content": "refinance calculator..."}},
    # ... more examples
]

optimized_profile = optimizer.optimize_scoring_profile(click_data)
# Deploy optimized profile to index
```

## 2. Azure Vector Search Strategy

**When vector search is the right choice**:
- Semantic/conceptual queries ("how to save for a house")
- Users don't know exact terminology (first-time homebuyers)
- Content uses varied vocabulary (multiple ways to say same thing)
- Cross-lingual search needs (English query → Spanish content)
- Question answering (long query, need semantic match)
- Recommendation systems (find similar content)

**When vector search fails**:
```
Query: "FHA-203B loan"
Vector search thinks: "FHA" ≈ "government housing" (conceptually similar)
Returns: VA loans, USDA loans, HUD programs (all government housing, wrong program)
Misses: Exact FHA-203B documentation
└── Why: Vectors capture semantics but lose exact identifiers

Query: "Form 1003"
Vector search thinks: "Form" ≈ "document", "1003" ≈ random number
Returns: Forms 1004, 1005, generic loan documents
Misses: Exact Form 1003
└── Why: Embeddings don't preserve exact IDs/codes well

Query: "contact us"
Vector search thinks: "contact" ≈ "communication", "us" ≈ "we"
Returns: Articles about customer communication, company history
Misses: Actual contact page
└── Why: Navigational intent lost in semantic space

Lesson: Vector search trades exact matching for semantic understanding.
For exact/navigational queries, this is a bad tradeoff.
```

**Vector search strengths in practice**:
```
Query: "help with down payment for first home"
Vector result: "First-Time Homebuyer Programs" (no exact word overlap!)
Why it works: Embeddings understand concept even with different words
Traditional BM25: Would miss this (no keyword overlap)

Query: "What are my options if I can't afford 20% down?"
Vector result: "Low Down Payment Mortgage Options", "FHA 3.5% Down", "PMI Explained"
Why it works: Understands question intent, finds conceptually relevant answers
Traditional BM25: Would struggle (query too long, no keyword density)

Query: "Should I refinance now?" (user's current rate unknown)
Vector result: "When to Refinance Calculator", "Rate Comparison Tool", "Refinancing Decision Guide"
Why it works: Captures intent (decision-making) not just keywords
Traditional BM25: Would return articles with "refinance now" literally
```

### Azure OpenAI Integration

**How Azure vector search works**: Documents → Azure OpenAI embeddings → Store in Azure AI Search → Query → Azure OpenAI embedding → Vector similarity search → Results

```python
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import numpy as np

class AzureVectorSearch:
    """
    Production-ready vector search using Azure OpenAI + Azure AI Search.
    
    Architecture:
    1. Documents embedded via Azure OpenAI (text-embedding-ada-002 or text-embedding-3-small)
    2. Vectors stored in Azure AI Search vector fields
    3. Query embedded with same model
    4. HNSW (Hierarchical Navigable Small World) approximate nearest neighbor search
    5. Results ranked by cosine similarity
    """
    
    def __init__(self, search_endpoint, search_key, openai_endpoint, openai_key, index_name):
        # Initialize Azure AI Search client
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAI(
            azure_endpoint=openai_endpoint,
            api_key=openai_key,
            api_version="2024-02-01"
        )
        self.embedding_deployment = "text-embedding-ada-002"  # Or "text-embedding-3-small"
    
    def get_embedding(self, text):
        """
        Generate embedding for text using Azure OpenAI.
        
        Args:
            text: Text to embed (max 8191 tokens for ada-002)
        
        Returns:
            1536-dimensional vector (ada-002) or 1536-d (text-embedding-3-small)
        
        Cost: $0.0001 per 1K tokens (~$0.10 per 1M tokens)
        """
        
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_deployment
        )
        
        return response.data[0].embedding
    
    def create_vector_index(self):
        """
        Create vector search index configuration.
        
        Key parameters:
        - vectorSearchDimensions: Must match embedding model (1536 for ada-002)
        - metric: "cosine" (most common), "euclidean", or "dotProduct"
        - HNSW parameters:
          * m: Number of bi-directional links (4-16, higher = better recall, slower build)
          * efConstruction: Candidates during build (100-1000, higher = better quality, slower build)
          * efSearch: Candidates during search (100-1000, higher = better recall, slower query)
        """
        
        vector_index_config = {
            "name": "vector-search-index",
            "fields": [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "key": True,
                    "searchable": False
                },
                {
                    "name": "content",
                    "type": "Edm.String",
                    "searchable": True  # Still searchable for hybrid
                },
                {
                    "name": "content_vector",
                    "type": "Collection(Edm.Single)",  # Vector field
                    "searchable": True,
                    "vectorSearchDimensions": 1536,  # ada-002 dimension
                    "vectorSearchProfileName": "vector-profile"
                },
                {
                    "name": "title",
                    "type": "Edm.String",
                    "searchable": True
                },
                {
                    "name": "metadata",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True
                }
            ],
            "vectorSearch": {
                "algorithms": [
                    {
                        "name": "hnsw-algorithm",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "metric": "cosine",  # Cosine similarity
                            "m": 4,  # 4 is default, good balance
                            "efConstruction": 400,  # Build quality (400 is default)
                            "efSearch": 500  # Search quality (500 is default)
                        }
                    }
                ],
                "profiles": [
                    {
                        "name": "vector-profile",
                        "algorithm": "hnsw-algorithm"
                    }
                ]
            }
        }
        
        return vector_index_config
    
    def embed_text(self, text):
        """
        Generate embedding using Azure OpenAI.
        
        Batch optimization: For indexing operations, batch multiple texts
        to reduce API calls and improve throughput.
        
        Cost implications:
        - text-embedding-ada-002: $0.0001/1K tokens (~750 words)
        - text-embedding-3-small: $0.00002/1K tokens (5x cheaper)
        - text-embedding-3-large: $0.00013/1K tokens (30% more expensive, 3072 dimensions)
        
        Typical costs for 10,000 documents (avg 500 tokens each):
        - ada-002: $0.50 for initial indexing
        - 3-small: $0.10 for initial indexing
        - Query costs: $0.0001 per query (ada-002)
        
        Dimension tradeoffs:
        - 1536-d (ada-002, 3-small): Good balance, most common
        - 3072-d (3-large): Better accuracy (+5-10%), 2x storage cost, slower search
        - Smaller dimensions (512-d, 768-d): Possible with text-embedding-3-small/large
          but requires dimension parameter and loses some quality
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_deployment
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts, batch_size=16):
        """
        Batch embed multiple texts for efficiency.
        
        Azure OpenAI supports up to 16 texts per API call for embedding models.
        This reduces API overhead and improves throughput for indexing operations.
        
        Example indexing throughput:
        - Sequential (1 at a time): ~10 docs/sec
        - Batched (16 at a time): ~120 docs/sec (12x faster)
        - Parallel batches (4 concurrent): ~400 docs/sec
        
        Args:
            texts: List of strings to embed
            batch_size: Number of texts per API call (max 16)
        
        Returns:
            List of embeddings (same order as input texts)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.openai_client.embeddings.create(
                input=batch,
                model=self.embedding_deployment
            )
            
            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def vector_search(self, query, top_k=10, filter_expression=None):
        """
        Perform vector search with Azure AI Search.
        
        How HNSW search works:
        1. Query embedding generated via Azure OpenAI
        2. HNSW algorithm navigates graph structure to find approximate nearest neighbors
        3. efSearch parameter controls search quality vs speed:
           - 100: Fast but lower recall (~0.80)
           - 500: Default balance (recall ~0.95)
           - 1000: Slower but better recall (~0.98)
        4. Results ranked by cosine similarity (-1 to 1, higher = more similar)
        
        Performance tuning:
        - top_k=10: Typical for user-facing search (30-50ms)
        - top_k=100: For reranking pipelines (80-120ms)
        - Filters add 10-30ms depending on selectivity
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            filter_expression: OData filter (e.g., "category eq 'mortgage'")
        
        Returns:
            List of results with scores (cosine similarity 0-1)
        """
        
        # Generate query embedding
        query_vector = self.embed_text(query)
        
        # Create vector query
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )
        
        # Search with vector
        results = self.search_client.search(
            search_text=None,  # Pure vector search (no text component)
            vector_queries=[vector_query],
            filter=filter_expression,
            top=top_k
        )
        
        return [
            {
                "id": result["id"],
                "content": result["content"],
                "score": result["@search.score"],  # Cosine similarity
                "title": result.get("title", "")
            }
            for result in results
        ]

**Understanding Vector Search Performance**

Vector search latency breakdown (typical P95 values):
- **Embedding generation**: 15-30ms (Azure OpenAI API call)
- **Vector search (HNSW)**: 10-40ms depending on index size and efSearch
- **Result retrieval**: 5-10ms (fetch document content from storage)
- **Total**: 30-80ms for P95 latency

Factors affecting vector search speed:
1. **Index size**: 
   - 10K docs: ~10ms search time
   - 100K docs: ~20ms search time
   - 1M docs: ~40ms search time
   - 10M docs: ~80ms search time (consider sharding)

2. **HNSW parameters**:
   - m=4: Faster search, lower recall
   - m=8: Balanced (recommended for most use cases)
   - m=16: Slower search, higher recall
   - efSearch=100: Fast but recall ~0.80
   - efSearch=500: Default, recall ~0.95
   - efSearch=1000: Slower, recall ~0.98

3. **Dimension count**:
   - 1536-d (ada-002): Standard, good performance
   - 3072-d (3-large): 30-50% slower due to larger vectors
   - 512-d: Faster but lower quality

Cost efficiency for vector search:
```
Azure AI Search S1 tier ($250/month):
- Stores ~1M documents (1KB each with 1536-d vector)
- Handles ~500-1000 QPS
- Cost per 1000 queries: ~$0.25 (including OpenAI embedding: $0.10)
- Total cost per query: ~$0.00035

vs. BM25 (same S1 tier):
- Cost per query: ~$0.00010 (3.5x cheaper)

Tradeoff: Vector search costs more but handles semantic queries
BM25 can't solve (worth extra cost for user experience improvement)
```

### Performance Characteristics
- **Latency**: 30-80ms P95 including embedding generation
- **Semantic Understanding**: Excellent for conceptual queries (handles paraphrasing, multilingual, Q&A)
- **Recall**: 0.75-0.85 for semantic queries (vs. 0.40-0.60 for BM25 on same queries)
- **Precision@5**: 0.80-0.90 for intent-based searches
- **Throughput**: 500-1000 QPS on S1 tier (~50% of BM25 due to embedding overhead)
- **Cost**: $0.00035/query (3.5x more than BM25 due to Azure OpenAI embedding costs)
- **Best for**: Q&A systems, semantic search, conceptual queries, multilingual content, chatbot retrieval

**When to accept higher vector search costs**: If >30% of user queries are conceptual/semantic (e.g., "how do I..." questions), the improved user experience justifies 3.5x cost increase. Calculate ROI: better results → higher conversion → more revenue.

## 3. Azure Hybrid Search Strategy

**What is hybrid search?** Combines BM25 keyword matching with vector semantic search, then fuses results using Reciprocal Rank Fusion (RRF) algorithm. Best of both worlds: exact matching for names/IDs + semantic understanding for concepts.

**When hybrid search is the right choice**:

1. **Mixed query types** (30-50% navigational + 50-70% informational):
   ```
   User queries mix:
   - "Form 1003" (exact match needed → BM25 wins)
   - "What documents do I need for a mortgage?" (semantic → vector wins)
   - "FHA loan requirements" (hybrid: exact "FHA" + semantic understanding of "requirements")
   
   Hybrid handles all query types well (no single strategy dominates)
   ```

2. **Exact matching + semantic understanding both critical**:
   ```
   Query: "affordable housing programs in California"
   BM25 component: Finds "California" exact mentions
   Vector component: Understands "affordable housing" concept includes low-income, subsidized, first-time buyer programs
   Hybrid: Combines both for comprehensive results
   ```

3. **Mitigating BM25 weaknesses** while keeping exact match strength:
   ```
   Query: "Can I buy a house with student loan debt?"
   Pure BM25: Fails (no exact phrase match in docs)
   Pure Vector: Works but might miss docs with exact "student loan" mentions
   Hybrid: Finds "student loan" exact matches AND semantically related "debt-to-income ratio" content
   ```

4. **Handling synonym/variation queries**:
   ```
   Query: "closing costs"
   BM25: Finds "closing costs" exact matches
   Vector: Also finds "settlement fees", "transaction expenses", "escrow charges"
   Hybrid: Returns comprehensive coverage of all terminology
   ```

**When hybrid search fails**:

1. **Purely navigational queries** (BM25 alone is better and cheaper):
   ```
   Query: "Form W-2"
   Problem: Hybrid adds unnecessary vector search cost with no benefit
   Pure BM25 works perfectly: exact match, instant results
   Lesson: Use query classification to route navigational queries to BM25 only
   ```

2. **Very short queries** (1-2 words):
   ```
   Query: "rates"
   Problem: Insufficient context for semantic understanding
   Vector embeddings can't disambiguate: mortgage rates? interest rates? tax rates?
   BM25 works better: returns all "rates" mentions, user scans for context
   Lesson: Short queries need keyword matching, not semantic search
   ```

3. **Exact technical terms required**:
   ```
   Query: "26 U.S. Code § 121"
   Problem: Vector search might return conceptually related tax code sections
   User needs exact section 121 (home sale exclusion)
   Lesson: Legal, regulatory, technical queries need exact matching
   ```

**Hybrid search strengths in practice**:
```
Query: "What's the difference between pre-qualification and pre-approval?"
BM25 results: Articles with exact terms "pre-qualification" and "pre-approval"
Vector results: Related concepts like "mortgage readiness", "approval letter"
RRF fusion: Prioritizes docs with BOTH exact terms AND conceptual relevance
Outcome: Better answer quality than either strategy alone

Query: "help for veterans buying first home"
BM25: Finds "veterans" exact matches
Vector: Understands "help" → programs, benefits; "first home" → first-time buyer
Hybrid: VA loan programs + first-time buyer benefits (comprehensive coverage)

Cost-benefit: Hybrid adds ~$0.00020/query over BM25
If it improves conversion by >2%, ROI is positive
```

### Hybrid Search Implementation
```python
class AzureHybridSearch:
    """
    Production-ready hybrid search combining BM25 + vector search + RRF fusion.
    
    How RRF (Reciprocal Rank Fusion) works:
    1. BM25 search produces ranked list: [doc3, doc7, doc1, doc5, ...]
    2. Vector search produces ranked list: [doc1, doc3, doc9, doc2, ...]
    3. RRF assigns score to each doc: 1/(k + rank) where k=60 (default)
       - doc1: BM25 rank=3 → 1/63 = 0.016, Vector rank=1 → 1/61 = 0.016
         Combined: 0.016 + 0.016 = 0.032
       - doc3: BM25 rank=1 → 1/61 = 0.016, Vector rank=2 → 1/62 = 0.016
         Combined: 0.016 + 0.016 = 0.032
    4. Docs ranked by combined RRF score (higher = better)
    5. Top-k results returned to user
    
    Why RRF works: Balances both signals without manual weight tuning.
    Documents appearing high in BOTH rankings naturally score highest.
    """
    
    def __init__(self, search_client, openai_client):
        self.search_client = search_client
        self.openai_client = openai_client
        self.embedding_deployment = "text-embedding-ada-002"
    
    def create_hybrid_index(self):
        """
        Create hybrid search index supporting both text and vector.
        
        Key configuration choices:
        - content field: Searchable with analyzer for BM25
        - content_vector field: 1536-d vector for semantic search
        - Both can be searched simultaneously
        - m=8 for HNSW: Higher than pure vector (m=4) because hybrid
          benefits from better recall since BM25 also contributes
        """
        
        hybrid_index_config = {
            "name": "hybrid-search-index",
            "fields": [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "key": True
                },
                {
                    "name": "content",
                    "type": "Edm.String",
                    "searchable": True,
                    "analyzer": "en.microsoft"  # BM25 text search
                },
                {
                    "name": "title",
                    "type": "Edm.String",
                    "searchable": True,
                    "analyzer": "en.microsoft"
                },
                {
                    "name": "content_vector",
                    "type": "Collection(Edm.Single)",  # Vector field
                    "searchable": True,
                    "vectorSearchDimensions": 1536,
                    "vectorSearchProfileName": "hybrid-vector-profile"
                },
                {
                    "name": "category",
                    "type": "Edm.String",
                    "filterable": True,  # For post-fusion filtering
                    "facetable": True
                }
            ],
            "vectorSearch": {
                "algorithms": [
                    {
                        "name": "hybrid-hnsw",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "metric": "cosine",
                            "m": 8,  # Higher for hybrid (vs. m=4 for pure vector)
                            "efConstruction": 400,
                            "efSearch": 500
                        }
                    }
                ],
                "profiles": [
                    {
                        "name": "hybrid-vector-profile",
                        "algorithm": "hybrid-hnsw"
                    }
                ]
            }
        }
        
        return hybrid_index_config
    
    def hybrid_search(self, query, top_k=10, filter_expression=None):
        """
        Perform hybrid search combining text and vector with RRF fusion.
        
        Azure AI Search handles RRF fusion automatically when both
        search_text and vector_queries are provided.
        
        Parameter tuning:
        - k_nearest_neighbors: Set to top_k * 2 to get more vector candidates
          before fusion (improves recall by 5-10%)
        - top: Final number of results after fusion
        - query_type: SIMPLE (default) or FULL (for complex boolean queries)
        
        Performance implications:
        - Hybrid costs ~2x single strategy (runs both BM25 + vector)
        - Latency: BM25 (15ms) + Vector (40ms) = ~55ms (run in parallel)
        - RRF fusion: ~5ms overhead
        - Total: 60-110ms P95
        
        When to use filters:
        - Apply filters AFTER fusion (not before) for best quality
        - Filter reduces result count but doesn't change ranking
        - Example: Get top 10 mortgage results, then filter to California
        
        Args:
            query: User's search query (used for both text and vector)
            top_k: Number of results to return after fusion
            filter_expression: OData filter applied post-fusion
        
        Returns:
            Fused results ranked by RRF score
        """
        
        # Generate query embedding for vector component
        query_vector = self.embed_text(query)
        
        # Create vector query
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k * 2,  # Get more candidates for better fusion
            fields="content_vector"
        )
        
        # Perform hybrid search (Azure AI Search handles RRF fusion automatically)
        results = self.search_client.search(
            search_text=query,  # BM25 component
            vector_queries=[vector_query],  # Vector component
            filter=filter_expression,  # Applied post-fusion
            top=top_k,
            query_type=QueryType.SIMPLE  # Or QueryType.FULL for boolean queries
        )
        
        return [
            {
                "id": result["id"],
                "title": result.get("title", ""),
                "content": result["content"],
                "score": result["@search.score"],  # RRF combined score
                "reranker_score": result.get("@search.reranker_score"),  # If semantic reranking enabled
                "category": result.get("category", "")
            }
            for result in results
        ]
    
    def hybrid_search_with_score_breakdown(self, query, top_k=10):
        """
        Hybrid search with component score visibility (for debugging).
        
        Useful for understanding which component (BM25 vs. vector) is
        contributing more to final rankings. Helps diagnose issues like:
        - All results driven by BM25 → vector not adding value
        - All results driven by vector → BM25 not contributing
        - Mixed contributions → hybrid working as intended
        """
        
        query_vector = self.embed_text(query)
        
        # Get BM25-only results
        bm25_results = self.search_client.search(
            search_text=query,
            top=top_k * 2
        )
        bm25_scores = {r["id"]: r["@search.score"] for r in bm25_results}
        
        # Get vector-only results
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k * 2,
            fields="content_vector"
        )
        vector_results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k * 2
        )
        vector_scores = {r["id"]: r["@search.score"] for r in vector_results}
        
        # Get hybrid results
        hybrid_results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top_k
        )
        
        # Annotate with component scores
        annotated_results = []
        for result in hybrid_results:
            doc_id = result["id"]
            annotated_results.append({
                "id": doc_id,
                "title": result.get("title", ""),
                "content": result["content"],
                "hybrid_score": result["@search.score"],
                "bm25_score": bm25_scores.get(doc_id, 0.0),
                "vector_score": vector_scores.get(doc_id, 0.0),
                "dominant_component": "BM25" if bm25_scores.get(doc_id, 0) > vector_scores.get(doc_id, 0) else "Vector"
            })
        
        return annotated_results
    
    def embed_text(self, text):
        """Generate embedding using Azure OpenAI."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_deployment
        )
        return response.data[0].embedding
```

**Understanding Hybrid Search Performance**

Hybrid search latency breakdown (P95):
- **BM25 search**: 10-30ms (keyword matching)
- **Vector search**: 30-60ms (embedding + HNSW search, runs in parallel with BM25)
- **RRF fusion**: 3-8ms (combining ranked lists)
- **Total**: 40-100ms P95 (dominated by slowest component, usually vector)

Why hybrid is not 2x slower than single strategy:
- BM25 and vector search run in **parallel**, not sequential
- Total time = max(BM25 time, vector time) + fusion overhead
- Typical: max(20ms, 50ms) + 5ms = 55ms (not 20+50=70ms)

Performance tuning for hybrid:
```python
# Scenario 1: Optimize for speed (accept lower recall)
vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=top_k,  # Fewer candidates (faster)
    fields="content_vector"
)
# Result: 30-70ms P95, recall ~0.82

# Scenario 2: Optimize for quality (accept higher latency)
vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=top_k * 3,  # More candidates (slower)
    fields="content_vector"
)
# Result: 50-110ms P95, recall ~0.90

# Scenario 3: Balance (recommended default)
vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=top_k * 2,  # 2x candidates
    fields="content_vector"
)
# Result: 40-100ms P95, recall ~0.87
```

Cost analysis for hybrid search:
```
Azure AI Search S1 tier ($250/month):
- Supports hybrid (both BM25 + vector in same index)
- Handles ~400-800 QPS for hybrid queries
- Storage: ~800K docs (1KB each + 1536-d vector)

Per-query cost breakdown:
- BM25 component: $0.00010
- Vector component (incl. Azure OpenAI embedding): $0.00025
- RRF fusion: negligible
- Total: ~$0.00035/query

vs. Pure BM25: $0.00010/query (3.5x cheaper)
vs. Pure Vector: $0.00035/query (same cost as hybrid)

Key insight: Hybrid costs same as pure vector but provides
better coverage (handles both navigational + semantic queries)
```

When hybrid justifies its cost over BM25:
1. **Query diversity**: >30% of queries are conceptual/semantic
2. **Recall improvement**: Hybrid improves recall by >15% on your evaluation set
3. **Revenue impact**: Better results drive measurable business outcomes (conversion, engagement)

Example ROI calculation:
```
E-commerce site, 10M queries/month:
- BM25 cost: 10M * $0.00010 = $1,000/month
- Hybrid cost: 10M * $0.00035 = $3,500/month
- Additional cost: $2,500/month

A/B test results:
- Hybrid improves conversion by 3% (vs. 2% needed to break even)
- Revenue impact: +$15K/month
- Net benefit: $15K - $2.5K = $12.5K/month

Decision: Hybrid is worth the cost (5x ROI)
```

### Performance Characteristics
- **Latency**: 40-100ms P95 (BM25 + vector in parallel + RRF fusion)
- **Best Overall Performance**: Combines benefits of both approaches
- **Precision@5**: 0.75-0.90 across different query types (navigational: 0.88, informational: 0.82)
- **Recall@10**: 0.82-0.90 (better than either strategy alone)
- **Throughput**: 400-800 QPS on S1 tier (lower than pure BM25 due to vector component)
- **Cost**: $0.00035/query (same as pure vector, 3.5x more than BM25)
- **Best for**: General-purpose search, mixed query types, e-commerce, enterprise knowledge bases

## 4. Azure Semantic Search Strategy

**What is semantic search?** AI-powered reranking that uses deep learning models to understand query intent and document relevance. Takes initial search results (from BM25, vector, or hybrid) and reorders them using Microsoft's proprietary semantic models trained on web-scale data.

**How semantic search differs from vector search**:
- **Vector search**: User provides query → embeddings generate vectors → similarity search
- **Semantic search**: Initial search (BM25/hybrid) → deep learning model reranks top 50 results → enhanced relevance

**When semantic search is the right choice**:

1. **Complex natural language queries** where understanding matters more than matching:
   ```
   Query: "What should I do if I'm underwater on my mortgage?"
   Without semantic: Returns literal "underwater mortgage" definitions
   With semantic: Understands question intent → Returns refinancing options, short sale guidance, loan modification programs
   
   Quality improvement: +15-25% precision for question-based queries
   ```

2. **Knowledge base / support systems** with Q&A patterns:
   ```
   Query: "Why was I denied for a mortgage?"
   Without semantic: Matches "denied" + "mortgage" literally
   With semantic: Understands causation query → Returns denial reasons (credit, DTI, employment)
   
   Answer extraction: Semantic search provides direct answer snippets
   ```

3. **Long-form content** where relevance is nuanced:
   ```
   Query: "best practices for getting approved with student loans"
   Without semantic: Keyword matching misses context
   With semantic: Understands "best practices" + "student loans" → Returns DTI optimization, documentation tips, lender selection
   ```

**When semantic search fails or isn't worth the cost**:

1. **Navigational queries** (semantic adds latency with no benefit):
   ```
   Query: "Form 1003"
   Problem: User wants specific form, not semantic understanding
   Semantic reranking: +50ms latency, no relevance improvement
   Lesson: Route navigational queries to BM25 or hybrid without semantic
   ```

2. **Short keyword queries** (insufficient context for semantic understanding):
   ```
   Query: "rates"
   Problem: Semantic model can't disambiguate without context
   Result: No improvement over keyword matching
   Cost: $0.00050 extra per query for no benefit
   ```

3. **Very large result sets** (semantic only reranks top 50):
   ```
   Scenario: User wants top 200 results for analysis
   Problem: Semantic only reranks top 50, rest is unchanged
   Latency: +100ms for partial benefit
   Lesson: Semantic is for precision, not comprehensive recall
   ```

4. **Budget-constrained applications** (semantic is most expensive option):
   ```
   Cost comparison per query:
   - BM25: $0.00010
   - Vector: $0.00035
   - Hybrid: $0.00035
   - Semantic (on top of hybrid): $0.00085 (8.5x BM25)
   
   At 1M queries/month:
   - BM25: $100/month
   - Semantic: $850/month
   
   ROI requirement: Need 8x improvement in business outcomes to justify
   ```

**Semantic search strengths in practice**:
```
Query: "How do I know if I should refinance?"
BM25/Hybrid result: Articles mentioning "refinance" (may be generic)
Semantic reranked: Decision frameworks, refinance calculators, when-to-refinance guides (intent-matched)
Improvement: +20% precision@5

Query: "What's the difference between pre-qualification and pre-approval?"
Hybrid result: Articles with both terms (good baseline)
Semantic reranked: Comparison articles ranked higher, definition-only articles lower
Answer extraction: Direct answer pulled from top result ("Pre-qualification is an estimate...")
Improvement: Answer in 0 clicks vs. 1-2 clicks

Query: "I make 75k and have 30k student debt, can I buy a house?"
BM25: Struggles with long query, no exact matches
Hybrid: Finds related content semantically
Semantic: Reranks to prioritize DTI calculators, affordability guides, student loan strategies
Answer extraction: "With $75k income and $30k debt, your DTI is approximately 40%..."
Improvement: Direct answer + relevant next steps
```

### Semantic Search Configuration
```python
class AzureSemanticSearch:
    """
    Production-ready semantic search with AI-powered reranking.
    
    Semantic search workflow:
    1. Initial search (BM25, vector, or hybrid) retrieves top 50 candidates
    2. Azure's semantic model reranks these 50 using deep learning
    3. Captions extracted (relevant snippets highlighted)
    4. Answers extracted (direct answer to query if found)
    5. Reranked results + captions + answers returned to user
    
    Key configuration:
    - prioritizedFields: Tell semantic model which fields matter most
      * titleField: Highest signal (product name, article title)
      * contentFields: Body text, descriptions (prioritized in order)
      * keywordFields: Tags, categories (lower signal)
    - query_caption: Extractive snippets showing why result is relevant
    - query_answer: Direct answer extraction (if query is question-like)
    
    Cost implications:
    - Semantic search: 1000 free queries/month, then $500/month for unlimited
    - Effectively $0.00050/query at high volume (>1M queries/month)
    - Answer extraction included (no extra cost)
    """
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def create_semantic_index(self):
        """
        Create index with semantic search configuration.
        
        Field prioritization strategy:
        - titleField: Short, high-signal field (doc title, product name)
        - contentFields: Ordered by importance (main content first, metadata last)
        - keywordFields: Categorical data (tags, categories) - lower semantic signal
        
        Best practices:
        - Keep title concise (<200 chars) for best semantic understanding
        - Content fields: 1-3 fields (more doesn't improve quality)
        - Don't include every field (semantic model focuses on most important)
        """
        
        semantic_index_config = {
            "name": "semantic-search-index",
            "fields": [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "key": True
                },
                {
                    "name": "title",
                    "type": "Edm.String",
                    "searchable": True  # High-signal field for semantic
                },
                {
                    "name": "content",
                    "type": "Edm.String",
                    "searchable": True  # Primary content field
                },
                {
                    "name": "summary",
                    "type": "Edm.String",
                    "searchable": True  # Secondary content field
                },
                {
                    "name": "category",
                    "type": "Edm.String",
                    "searchable": True,  # Keyword field for semantic
                    "filterable": True
                }
            ],
            "semanticSearch": {
                "configurations": [
                    {
                        "name": "semantic-config",
                        "prioritizedFields": {
                            "titleField": {
                                "fieldName": "title"  # Highest priority
                            },
                            "prioritizedContentFields": [
                                {"fieldName": "content"},  # Primary content
                                {"fieldName": "summary"}   # Secondary content
                            ],
                            "prioritizedKeywordsFields": [
                                {"fieldName": "category"}  # Lower priority
                            ]
                        }
                    }
                ]
            }
        }
        
        return semantic_index_config
    
    def semantic_search(self, query, top_k=10, use_captions=True, use_answers=True):
        """
        Perform semantic search with AI-powered ranking.
        
        Semantic reranking details:
        - Reranks top 50 results from initial search (BM25 or hybrid)
        - Uses Microsoft's proprietary model trained on Bing data
        - Optimized for English queries (limited support for other languages)
        - Returns @search.reranker_score (higher = more semantically relevant)
        
        Caption extraction:
        - Highlights most relevant passage from document (150-200 chars)
        - Shows WHY document is relevant to query
        - Reduces need for users to open full document
        - Example: Query "refinance student loans" → Caption: "...refinancing federal student loans into a private mortgage may lower rates but loses federal protections..."
        
        Answer extraction:
        - Attempts to extract direct answer from top results
        - Works best for factoid questions ("What is...", "When should...")
        - Returns confidence score (0-1, >0.8 = high confidence)
        - Example: Query "What is PMI?" → Answer: "Private Mortgage Insurance (PMI) is required when down payment is less than 20%..."
        
        Performance characteristics:
        - Initial search: 20-50ms (BM25 or hybrid)
        - Semantic reranking: 40-100ms (deep learning inference)
        - Caption extraction: +10-20ms
        - Answer extraction: +10-20ms
        - Total: 80-190ms P95 (depends on configuration)
        
        Args:
            query: Natural language search query
            top_k: Number of results to return (max 50 for semantic)
            use_captions: Extract relevant snippets (adds 10-20ms)
            use_answers: Extract direct answers (adds 10-20ms)
        
        Returns:
            Results with reranker scores, captions, and answers
        """
        
        # Configure semantic search parameters
        query_caption = QueryCaptionType.EXTRACTIVE if use_captions else None
        query_answer = QueryAnswerType.EXTRACTIVE if use_answers else None
        
        results = self.search_client.search(
            search_text=query,
            top=min(top_k, 50),  # Semantic limited to 50 results
            query_type=QueryType.SEMANTIC,  # Enable semantic reranking
            semantic_configuration_name="semantic-config",
            query_caption=query_caption,
            query_answer=query_answer
        )
        
        search_results = []
        semantic_answers = []
        
        for result in results:
            # Extract semantic captions (highlighted relevant passages)
            captions = []
            if "@search.captions" in result:
                captions = [
                    {
                        "text": caption.text,  # Full caption text
                        "highlights": caption.highlights  # <em> tags around matched parts
                    }
                    for caption in result["@search.captions"]
                ]
            
            search_results.append({
                "id": result["id"],
                "title": result.get("title", ""),
                "content": result["content"],
                "score": result["@search.score"],  # Original search score
                "reranker_score": result.get("@search.reranker_score"),  # Semantic score (0-4+)
                "captions": captions  # Relevant snippets
            })
        
        # Extract semantic answers if available
        # Answers are query-level (not per-document)
        if hasattr(results, 'get_answers') and use_answers:
            answers = results.get_answers()
            if answers:
                semantic_answers = [
                    {
                        "text": answer.text,  # Direct answer text
                        "highlights": answer.highlights,  # Highlighted portions
                        "score": answer.score  # Confidence score (0-1)
                    }
                    for answer in answers
                ]
        
        return {
            "results": search_results,
            "answers": semantic_answers,  # May be empty if no good answer found
            "query": query
        }
    
    def semantic_hybrid_search(self, query, query_vector, top_k=10):
        """
        Semantic search on top of hybrid (recommended for best quality).
        
        This is the "premium" configuration:
        1. Hybrid search (BM25 + vector + RRF) gets initial candidates
        2. Semantic reranking refines top 50 results
        3. Captions and answers extracted
        
        Quality characteristics:
        - Precision@5: 0.85-0.95 (best possible)
        - Handles all query types well (navigational, informational, Q&A)
        - Answer extraction success rate: 60-75% for question queries
        
        Cost: $0.00035 (hybrid) + $0.00050 (semantic) = $0.00085/query
        Latency: 60ms (hybrid) + 80ms (semantic) = 140-180ms P95
        
        When to use: Premium search experiences where quality justifies cost
        (customer support, enterprise knowledge bases, high-value e-commerce)
        """
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,  # Get more candidates for semantic
            fields="content_vector"
        )
        
        results = self.search_client.search(
            search_text=query,  # Hybrid BM25 component
            vector_queries=[vector_query],  # Hybrid vector component
            top=min(top_k, 50),
            query_type=QueryType.SEMANTIC,  # Semantic reranking on top
            semantic_configuration_name="semantic-config",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE
        )
        
        return self._process_semantic_results(results)
    
    def _process_semantic_results(self, results):
        """Helper to process semantic search results consistently."""
        search_results = []
        semantic_answers = []
        
        for result in results:
            captions = []
            if "@search.captions" in result:
                captions = [
                    {
                        "text": caption.text,
                        "highlights": caption.highlights
                    }
                    for caption in result["@search.captions"]
                ]
            
            search_results.append({
                "id": result["id"],
                "title": result.get("title", ""),
                "content": result["content"],
                "score": result["@search.score"],
                "reranker_score": result.get("@search.reranker_score"),
                "captions": captions
            })
        
        if hasattr(results, 'get_answers'):
            answers = results.get_answers()
            if answers:
                semantic_answers = [
                    {
                        "text": answer.text,
                        "highlights": answer.highlights,
                        "score": answer.score
                    }
                    for answer in answers
                ]
        
        return {
            "results": search_results,
            "answers": semantic_answers
        }
```

**Understanding Semantic Reranker Scores**

Semantic reranker score interpretation:
- **Score range**: 0 to 4+ (not normalized, higher = better)
- **Typical distribution**:
  - Highly relevant: 2.5-4.0
  - Moderately relevant: 1.5-2.5
  - Somewhat relevant: 0.5-1.5
  - Not relevant: 0.0-0.5

Comparing original search score vs. reranker score:
```python
# Example results showing reranking impact
Query: "What are closing costs for a home purchase?"

Before semantic reranking (hybrid scores):
1. "Closing Cost Calculator" (score: 12.5)
2. "How to Close a Deal" (score: 11.8) ← Irrelevant but high BM25 score
3. "Understanding Settlement Fees" (score: 10.2)

After semantic reranking (reranker scores):
1. "Understanding Settlement Fees" (reranker: 3.2) ← Moved up
2. "Closing Cost Calculator" (reranker: 3.0)
3. "Typical Closing Costs by State" (reranker: 2.8) ← Wasn't in original top 3
...
10. "How to Close a Deal" (reranker: 0.4) ← Dropped down

Impact: Semantic correctly identifies "close a deal" is wrong context
```

When semantic reranking makes biggest difference:
1. **Ambiguous queries**: "apple" (fruit vs. company), "python" (snake vs. language)
2. **Long natural language queries**: "What should I do if..."
3. **Questions**: Semantic trained to identify question-answer pairs
4. **Synonym/paraphrase**: Query uses different words than doc but same meaning

When semantic reranking changes little:
1. **Unique exact matches**: "Form 1003" (only one correct result)
2. **Very short queries**: "mortgage" (insufficient context)
3. **Already-good BM25/hybrid results**: Semantic confirms existing ranking

### Performance Characteristics
- **Latency**: 80-190ms P95 (initial search 20-50ms + semantic 60-140ms + captions/answers 10-40ms)
- **Highest Quality**: Best relevance for natural language queries, question answering
- **Precision@5**: 0.85-0.95 for complex queries (15-25% better than hybrid alone)
- **Answer Extraction Success**: 60-75% for question-based queries
- **Throughput**: 200-500 QPS on S1 tier (limited by semantic inference)
- **Cost**: $0.00050-$0.00085/query (most expensive option, 5-8.5x BM25)
- **Best for**: Premium search experiences, knowledge bases, customer support, Q&A systems, high-value content

## 🎯 Azure Strategy Selection Framework

**How to choose the right indexing strategy**: Evaluate your requirements across 5 dimensions: query types, latency tolerance, budget, quality needs, and content characteristics. This framework provides data-driven recommendations.

```python
def select_azure_search_strategy(requirements):
    """
    Select optimal Azure AI Search strategy based on requirements.
    
    Decision dimensions:
    1. Latency requirements (SLA constraints)
    2. Query type distribution (navigational vs. informational vs. Q&A)
    3. Budget constraints (cost per query, monthly spend)
    4. Quality requirements (precision targets, user experience)
    5. Content characteristics (structured vs. unstructured, multilingual)
    
    Scoring methodology:
    - Each dimension awards points to strategies that fit best
    - Highest total score = recommended strategy
    - Ties broken by cost (cheaper wins)
    
    Returns recommendation with reasoning for transparency
    """
    
    score_card = {
        "fulltext": 0,  # BM25 keyword search
        "vector": 0,    # Semantic vector search
        "hybrid": 0,    # BM25 + vector + RRF
        "semantic": 0   # Hybrid + AI reranking
    }
    
    # Dimension 1: Latency requirements
    max_latency = requirements.get("max_latency_ms", 1000)
    if max_latency < 50:  # Very strict latency (e.g., autocomplete)
        score_card["fulltext"] += 3  # BM25 is fastest (10-30ms)
        score_card["vector"] += 1    # Vector possible but tight (30-80ms)
    elif max_latency < 100:  # Tight latency (e.g., instant search)
        score_card["fulltext"] += 2  # BM25 still best
        score_card["vector"] += 2    # Vector fits comfortably
        score_card["hybrid"] += 1    # Hybrid might exceed at P95
    elif max_latency < 150:  # Moderate latency (e.g., standard search)
        score_card["fulltext"] += 1
        score_card["vector"] += 2
        score_card["hybrid"] += 3    # Hybrid fits well
        score_card["semantic"] += 1  # Semantic might exceed at P99
    else:  # Relaxed latency (>150ms acceptable)
        score_card["hybrid"] += 2
        score_card["semantic"] += 3  # Semantic fits (80-190ms)
    
    # Dimension 2: Query type distribution analysis
    query_types = requirements.get("query_types", {})
    
    # Navigational queries (exact match needed: product IDs, form numbers, specific names)
    navigational_pct = query_types.get("navigational", 0)
    if navigational_pct > 0.5:  # >50% navigational
        score_card["fulltext"] += 3  # BM25 excels at exact matching
        score_card["hybrid"] += 2    # Hybrid handles navigational well
    elif navigational_pct > 0.3:  # 30-50% navigational
        score_card["fulltext"] += 2
        score_card["hybrid"] += 3    # Hybrid best for mixed
    
    # Informational queries (conceptual: "how to", "what is", topic exploration)
    informational_pct = query_types.get("informational", 0)
    if informational_pct > 0.5:  # >50% informational
        score_card["vector"] += 3    # Vector handles concepts well
        score_card["hybrid"] += 3    # Hybrid also good
        score_card["semantic"] += 2  # Semantic overkill unless Q&A
    
    # Question-answering queries (explicit questions needing direct answers)
    qa_pct = query_types.get("question_answering", 0)
    if qa_pct > 0.3:  # >30% Q&A queries
        score_card["semantic"] += 3  # Semantic has answer extraction
        score_card["vector"] += 2    # Vector good at Q&A too
        score_card["hybrid"] += 2
    
    # Dimension 3: Budget constraints
    budget_tier = requirements.get("budget_tier", "medium")
    monthly_queries = requirements.get("monthly_queries", 100000)
    
    if budget_tier == "low":
        score_card["fulltext"] += 3  # Cheapest option
        score_card["vector"] -= 1    # 3.5x more expensive
        score_card["hybrid"] -= 1
        score_card["semantic"] -= 2  # 8.5x more expensive
    elif budget_tier == "high":
        score_card["semantic"] += 2  # Premium option justified
        score_card["hybrid"] += 1
    
    # Cost-based adjustment for query volume
    if monthly_queries > 10_000_000:  # >10M queries/month
        # At scale, cost differences are magnified
        monthly_cost_fulltext = monthly_queries * 0.00010 / 1000
        monthly_cost_semantic = monthly_queries * 0.00085 / 1000
        cost_difference = monthly_cost_semantic - monthly_cost_fulltext
        
        if cost_difference > 5000:  # $5K+ monthly difference
            score_card["fulltext"] += 2  # Favor cheaper option
            score_card["semantic"] -= 2
    
    # Dimension 4: Quality requirements
    min_precision = requirements.get("min_precision_at_5", 0.7)
    if min_precision > 0.85:  # Very high precision required
        score_card["semantic"] += 3  # Semantic best (0.85-0.95)
        score_card["hybrid"] += 2    # Hybrid good (0.75-0.90)
        score_card["vector"] += 1    # Vector decent (0.80-0.90)
    elif min_precision > 0.75:  # High precision required
        score_card["hybrid"] += 3    # Hybrid sweet spot
        score_card["semantic"] += 2
        score_card["vector"] += 2
    else:  # Moderate precision acceptable
        score_card["fulltext"] += 2  # BM25 sufficient
        score_card["hybrid"] += 1
    
    min_recall = requirements.get("min_recall_at_10", 0.7)
    if min_recall > 0.85:  # High recall required
        score_card["hybrid"] += 3    # Best recall (0.82-0.90)
        score_card["semantic"] += 2
        score_card["vector"] += 2
    
    # Dimension 5: Content characteristics
    content = requirements.get("content_characteristics", {})
    
    # Multilingual content
    if content.get("multilingual", False):
        score_card["vector"] += 3    # Vector handles multilingual well
        score_card["hybrid"] += 2
        # Semantic is English-optimized, penalize slightly
        score_card["semantic"] -= 1
    
    # Long-form content (articles, documentation, guides)
    if content.get("long_form", False):
        score_card["semantic"] += 2  # Semantic excels at long content
        score_card["vector"] += 2
        score_card["hybrid"] += 1
    
    # Structured data (product catalogs, databases)
    if content.get("structured", False):
        score_card["fulltext"] += 2  # BM25 good for structured
        score_card["hybrid"] += 1
    
    # Calculate final recommendation
    ranked = sorted(score_card.items(), key=lambda x: x[1], reverse=True)
    
    # If tie, break by cost (cheaper wins)
    top_score = ranked[0][1]
    tied_strategies = [s for s, score in ranked if score == top_score]
    
    if len(tied_strategies) > 1:
        # Cost per query (ascending order)
        cost_order = ["fulltext", "hybrid", "vector", "semantic"]
        primary = next(s for s in cost_order if s in tied_strategies)
    else:
        primary = ranked[0][0]
    
    return {
        "primary_recommendation": primary,
        "scores": dict(ranked),
        "reasoning": _generate_azure_reasoning(requirements, primary),
        "cost_estimate": _estimate_monthly_cost(primary, monthly_queries),
        "expected_performance": _estimate_performance(primary)
    }

def _generate_azure_reasoning(requirements, recommendation):
    """
    Generate human-readable explanation for strategy recommendation.
    
    Provides context-specific reasoning so users understand WHY
    this strategy was recommended for their requirements.
    """
    
    reasons = []
    
    if recommendation == "fulltext":
        reasons.append("Azure AI Search BM25 optimized for exact keyword matches")
        reasons.append("Most cost-effective option ($0.00010/query, ~$100/1M queries)")
        reasons.append("Fastest performance (10-30ms P95 latency)")
        reasons.append("Best for: Product catalogs, navigational queries, exact phrase search")
        reasons.append("Consider upgrade to hybrid if: >30% of queries are conceptual/semantic")
    
    elif recommendation == "vector":
        reasons.append("Azure OpenAI embeddings provide semantic understanding")
        reasons.append("Excellent for conceptual queries and Q&A (no exact match needed)")
        reasons.append("Handles multilingual content effectively (cross-language retrieval)")
        reasons.append("Cost: $0.00035/query (~$350/1M queries)")
        reasons.append("Best for: Knowledge bases, semantic search, chatbot retrieval")
        reasons.append("Consider hybrid if: Also need to handle navigational queries well")
    
    elif recommendation == "hybrid":
        reasons.append("Azure RRF fusion combines BM25 keyword + vector semantic signals")
        reasons.append("Best overall performance across diverse query types")
        reasons.append("Handles both exact matches AND conceptual queries well")
        reasons.append("Balanced cost/performance ($0.00035/query, same as pure vector)")
        reasons.append("Enterprise-grade solution with broad applicability")
        reasons.append("Best for: General-purpose search, e-commerce, mixed workloads")
        reasons.append("Consider semantic upgrade if: >30% Q&A queries need answer extraction")
    
    elif recommendation == "semantic":
        reasons.append("Azure semantic search provides highest quality relevance")
        reasons.append("AI-powered reranking with deep learning models")
        reasons.append("Built-in answer extraction and caption generation")
        reasons.append("Best precision for natural language and question queries (0.85-0.95)")
        reasons.append("Premium cost: $0.00050-$0.00085/query (~$850/1M queries)")
        reasons.append("Best for: Customer support, premium search, knowledge management, Q&A systems")
        reasons.append("Worth the cost if: Quality improvement drives measurable business outcomes")
    
    return reasons

def _estimate_monthly_cost(strategy, monthly_queries):
    """Estimate monthly cost for given strategy and query volume."""
    
    cost_per_query = {
        "fulltext": 0.00010,
        "vector": 0.00035,
        "hybrid": 0.00035,
        "semantic": 0.00085
    }
    
    base_infrastructure = {
        "fulltext": 250,    # S1 tier
        "vector": 250,      # S1 tier + OpenAI
        "hybrid": 250,      # S1 tier + OpenAI
        "semantic": 750     # S1 tier + OpenAI + Semantic ($500)
    }
    
    query_cost = monthly_queries * cost_per_query[strategy]
    infrastructure_cost = base_infrastructure[strategy]
    total_cost = query_cost + infrastructure_cost
    
    return {
        "monthly_queries": monthly_queries,
        "cost_per_query": cost_per_query[strategy],
        "query_cost": round(query_cost, 2),
        "infrastructure_cost": infrastructure_cost,
        "total_monthly_cost": round(total_cost, 2)
    }

def _estimate_performance(strategy):
    """Estimate performance characteristics for strategy."""
    
    performance = {
        "fulltext": {
            "latency_p50_ms": 15,
            "latency_p95_ms": 30,
            "latency_p99_ms": 50,
            "throughput_qps": 5000,
            "precision_at_5": 0.85,
            "recall_at_10": 0.75
        },
        "vector": {
            "latency_p50_ms": 40,
            "latency_p95_ms": 60,
            "latency_p99_ms": 100,
            "throughput_qps": 800,
            "precision_at_5": 0.85,
            "recall_at_10": 0.82
        },
        "hybrid": {
            "latency_p50_ms": 50,
            "latency_p95_ms": 80,
            "latency_p99_ms": 120,
            "throughput_qps": 600,
            "precision_at_5": 0.87,
            "recall_at_10": 0.87
        },
        "semantic": {
            "latency_p50_ms": 100,
            "latency_p95_ms": 140,
            "latency_p99_ms": 200,
            "throughput_qps": 400,
            "precision_at_5": 0.90,
            "recall_at_10": 0.88
        }
    }
    
    return performance[strategy]
```

**Example usage of selection framework**:

```python
# Scenario 1: E-commerce product search
ecommerce_requirements = {
    "max_latency_ms": 80,
    "query_types": {
        "navigational": 0.6,  # 60% search for specific products
        "informational": 0.4   # 40% browse/explore
    },
    "budget_tier": "medium",
    "monthly_queries": 5_000_000,
    "min_precision_at_5": 0.80,
    "content_characteristics": {
        "structured": True,
        "multilingual": False
    }
}

recommendation = select_azure_search_strategy(ecommerce_requirements)
# Result: "hybrid" 
# Reasoning: Handles both product name searches (BM25) + browsing (vector)
# Cost: ~$1,750/month total, latency ~80ms P95

# Scenario 2: Customer support knowledge base
support_requirements = {
    "max_latency_ms": 200,
    "query_types": {
        "navigational": 0.1,
        "informational": 0.3,
        "question_answering": 0.6  # 60% are questions
    },
    "budget_tier": "high",
    "monthly_queries": 500_000,
    "min_precision_at_5": 0.85,
    "content_characteristics": {
        "long_form": True,
        "multilingual": False
    }
}

recommendation = select_azure_search_strategy(support_requirements)
# Result: "semantic"
# Reasoning: Answer extraction critical for Q&A, latency budget allows
# Cost: ~$1,175/month total, latency ~140ms P95, answers extracted

# Scenario 3: Cost-sensitive document search
budget_requirements = {
    "max_latency_ms": 50,
    "query_types": {
        "navigational": 0.8,  # Mostly specific document searches
        "informational": 0.2
    },
    "budget_tier": "low",
    "monthly_queries": 10_000_000,
    "min_precision_at_5": 0.75,
    "content_characteristics": {
        "structured": True
    }
}

recommendation = select_azure_search_strategy(budget_requirements)
# Result: "fulltext"
# Reasoning: BM25 handles navigational well, cost ($1,000/mo) vs semantic ($8,500/mo)
# Cost: ~$1,250/month total, latency ~30ms P95
```

## 🔧 Implementation Roadmap

**How to implement Azure AI Search systematically**: Start with simplest strategy (BM25), establish baseline, then progressively enhance based on evaluation results. Each phase builds on previous, allowing for incremental validation and cost optimization.

### Phase 1: Azure Foundation (Week 1)

**Objective**: Establish baseline BM25 search with evaluation framework.

1. **Azure Services Setup**
   - Create Azure AI Search service (Standard S1 tier):
     * Provision via Azure Portal or `az search service create`
     * Region selection: Choose region closest to users (latency) or closest to data source (indexing speed)
     * Enable managed identity for secure access to other Azure services
   
   - Configure Azure Monitor logging:
     * Enable diagnostic settings → Send to Log Analytics workspace
     * Track: Search latency, throttling, index size, query volume
     * Set up basic alerts (P95 latency >100ms, throttling >1%, error rate >0.5%)
   
   - Set up cost monitoring:
     * Azure Cost Management: Set budget alerts ($300/month for S1)
     * Tag resources for cost attribution (project: search-eval, env: dev)

2. **Basic Full-text Search (BM25)**
   - Implement BM25 search with custom scoring profiles:
     * Create index schema (see Section 1 for reference)
     * Configure analyzers (en.microsoft for English content)
     * Add scoring profile (boost recent docs, boost title matches)
   
   - Create evaluation baseline:
     * Implement metrics from `01-core-metrics.md` (Precision@5, Recall@10, MRR, NDCG)
     * Run evaluation on golden dataset (see `02-evaluation-frameworks.md`)
     * Document baseline: "BM25 achieves Precision@5 = 0.78, Recall@10 = 0.72"
   
   - Establish cost baseline:
     * Monitor for 1 week: Query volume, storage, compute
     * Calculate cost per query: Total cost / query count
     * Example: "$250 infrastructure + $50 queries = $0.00010/query"

**Phase 1 Success Criteria**:
- ✅ BM25 search functional with <50ms P95 latency
- ✅ Baseline metrics documented (Precision, Recall, MRR, NDCG)
- ✅ Cost tracking in place (<$300/month for dev workload)
- ✅ Monitoring dashboards operational

### Phase 2: Vector Enhancement (Week 2)

**Objective**: Add semantic vector search capability and evaluate improvement over BM25.

1. **Azure OpenAI Service Setup**
   - Provision Azure OpenAI resource:
     * Create via Azure Portal (requires application/approval)
     * Deploy embedding model: text-embedding-ada-002 or text-embedding-3-small
     * Region: Same as Azure AI Search for lowest latency
     * Configure rate limits (10K TPM for dev, 100K+ TPM for production)
   
   - Test embedding generation:
     ```python
     from openai import AzureOpenAI
     client = AzureOpenAI(...)
     embedding = client.embeddings.create(
         input="test document",
         model="text-embedding-ada-002"
     )
     print(f"Dimension: {len(embedding.data[0].embedding)}")  # Should be 1536
     ```

2. **Vector Index Configuration**
   - Add vector field to existing index (see Section 2 for code):
     * content_vector: Collection(Edm.Single), 1536 dimensions
     * Configure HNSW algorithm (m=4, efConstruction=400, efSearch=500)
   
   - Batch embed existing documents:
     * Use batch embedding (16 docs per API call for efficiency)
     * Monitor Azure OpenAI quota usage
     * Example: 10K docs * 500 tokens avg = 5M tokens = $0.50 (ada-002)
   
   - Update index with vectors:
     * Use Azure AI Search merge/upload API
     * Process in batches of 1000 documents
     * Monitor indexing latency and throttling

3. **Vector Search Evaluation**
   - Run evaluation on same golden dataset as Phase 1:
     * Calculate metrics: Precision@5, Recall@10, MRR, NDCG
     * Compare to BM25 baseline
     * Analyze query-level performance (where vector wins/loses vs. BM25)
   
   - Cost analysis:
     * Embedding costs: Query volume * $0.0001 per 1K tokens
     * Storage costs: Vector storage adds ~6KB per doc (1536-d * 4 bytes)
     * Compute costs: S1 tier handles both BM25 and vector
   
   - Document findings:
     * "Vector search improves Precision@5 by +8% on semantic queries"
     * "Vector search underperforms BM25 by -15% on navigational queries"
     * "Cost increase: +$0.00025/query for embedding generation"

**Phase 2 Success Criteria**:
- ✅ Vector search operational with <80ms P95 latency
- ✅ Embeddings generated for all documents
- ✅ Evaluation shows improvement on semantic queries (>+5% precision)
- ✅ Cost increase understood and acceptable (<2x BM25)

### Phase 3: Hybrid Integration (Week 3)

**Objective**: Combine BM25 + vector with RRF fusion for best-of-both-worlds performance.

1. **Hybrid Index Configuration**
   - Ensure index supports both text and vector search (see Section 3):
     * Text fields with analyzers (BM25 component)
     * Vector fields with HNSW profile (vector component)
     * Both searchable on same documents
   
   - Tune HNSW parameters for hybrid:
     * Increase m from 4 to 8 (hybrid benefits from better recall)
     * Keep efSearch=500 (good balance)
     * Monitor index build time (higher m = slower build)

2. **Hybrid Search Implementation**
   - Implement hybrid search with RRF fusion:
     * Provide both search_text and vector_queries to Azure AI Search
     * Azure automatically applies RRF with k=60
     * Return top-k results after fusion
   
   - Add intelligent query routing:
     ```python
     def route_query(query):
         if is_navigational(query):  # "Form 1003", "contact us"
             return "bm25_only"  # Don't waste cost on vector
         elif is_short(query):  # 1-2 words
             return "bm25_only"  # Insufficient context for vector
         else:
             return "hybrid"  # Use both for best results
     
     def is_navigational(query):
         # Check for patterns: form numbers, exact phrases, URLs
         return bool(re.match(r'(form|doc|page)\s+\w+', query.lower()))
     
     def is_short(query):
         return len(query.split()) <= 2
     ```

3. **A/B Testing Framework**
   - Set up A/B test infrastructure:
     * Control group: BM25 only
     * Treatment group: Hybrid search
     * Randomize users (50/50 split)
     * Track metrics: Precision@5, click-through rate, time-to-click
   
   - Run A/B test for 1-2 weeks:
     * Monitor statistical significance (p-value <0.05)
     * Calculate lift: "Hybrid improves CTR by +12% (p=0.003)"
     * Analyze cost vs. benefit
   
   - Document decision:
     * If lift >10% and cost acceptable → Roll out hybrid to 100%
     * If lift <5% or cost too high → Stay with BM25, revisit later
     * If mixed results → Use query routing (hybrid for some, BM25 for others)

**Phase 3 Success Criteria**:
- ✅ Hybrid search operational with <100ms P95 latency
- ✅ RRF fusion working correctly (verify top results come from both BM25 and vector)
- ✅ A/B test shows statistically significant improvement (>+5% on key metric)
- ✅ Query routing reduces unnecessary vector costs by 20-30%

### Phase 4: Semantic Enhancement (Week 4)

**Objective**: Add AI-powered reranking for premium quality on question-answering queries.

1. **Semantic Search Configuration**
   - Enable semantic search on Azure AI Search index:
     * Add semantic configuration (see Section 4 for code)
     * Prioritize fields: title (highest), content (primary), summary (secondary)
     * Enable semantic search tier (adds $500/month after 1K free queries)
   
   - Test semantic reranking:
     * Run queries with query_type=QueryType.SEMANTIC
     * Compare @search.score (original) vs. @search.reranker_score (semantic)
     * Verify reranking improves relevance (manually review top 10 results)

2. **Answer Extraction Implementation**
   - Enable extractive Q&A:
     * Set query_answer=QueryAnswerType.EXTRACTIVE
     * Semantic model extracts direct answers from top results
     * Returns answer with confidence score (0-1)
   
   - Implement answer display in UI:
     ```python
     results = semantic_search(query, use_answers=True)
     
     if results["answers"] and results["answers"][0]["score"] > 0.8:
         # High confidence answer → Display prominently
         show_direct_answer(results["answers"][0]["text"])
     else:
         # No confident answer → Show search results only
         show_search_results(results["results"])
     ```
   
   - Measure answer extraction success rate:
     * For question queries, what % get high-confidence answers (>0.8)?
     * Target: 60-75% answer extraction rate on question queries

3. **Production Monitoring and Optimization**
   - Set up comprehensive monitoring:
     * Latency: P50, P95, P99 for each strategy (BM25, vector, hybrid, semantic)
     * Quality: Precision@5, Recall@10, NDCG@10 (daily evaluation)
     * Cost: Query cost, storage cost, total monthly spend
     * Usage: Query volume by strategy, query type distribution
   
   - Create alerting rules:
     * Latency P95 >150ms → Alert (investigate slow queries)
     * Precision@5 drops >5% → Alert (model drift or data quality issue)
     * Cost increase >20% month-over-month → Alert (usage spike)
   
   - Optimize based on data:
     * If semantic used on navigational queries → Improve query routing
     * If latency high → Tune HNSW parameters or scale up tier
     * If cost too high → Review query volume, consider caching

**Phase 4 Success Criteria**:
- ✅ Semantic search operational with <180ms P95 latency
- ✅ Answer extraction working for 60-75% of question queries
- ✅ Precision@5 improved by >10% on Q&A queries vs. hybrid alone
- ✅ Monitoring dashboards show all key metrics (latency, quality, cost)
- ✅ ROI calculated: Quality improvement justifies cost increase

## 📊 Migration Paths Between Strategies

**How to migrate from one strategy to another**: Zero-downtime migration patterns for changing indexing strategies as requirements evolve.

### Migrating from BM25 to Vector Search

**Scenario**: BM25 deployed in production, want to add vector search without disruption.

**Migration approach** (Blue-Green deployment):

1. **Prepare vector-enabled index (Green)**:
   ```python
   # Create new index with vector fields
   vector_index = create_vector_index()  # Has both text + vector fields
   
   # Batch embed all documents (offline)
   docs = fetch_all_documents()
   embeddings = batch_embed(docs, batch_size=16)
   
   # Upload docs with vectors to new index
   for doc, embedding in zip(docs, embeddings):
       doc["content_vector"] = embedding
       upload_to_index(vector_index, doc)
   ```

2. **Test Green index**:
   - Run evaluation suite against new vector index
   - Compare metrics to Blue (BM25) baseline
   - Load test to verify performance under production traffic
   - Verify cost projections (embedding costs, storage)

3. **Gradual traffic shift** (canary deployment):
   ```python
   # Route 5% of traffic to vector search
   if random.random() < 0.05:
       results = vector_search(query)
   else:
       results = bm25_search(query)  # 95% still on BM25
   
   # Monitor for 24-48 hours:
   # - Latency within SLA? (P95 <100ms)
   # - Quality improved? (Precision@5 up)
   # - No errors or throttling?
   
   # If successful, increase to 25% → 50% → 100%
   ```

4. **Full cutover**:
   - Route 100% traffic to vector index
   - Monitor for 1 week
   - If stable, decommission old BM25-only index
   - Update application to use vector index permanently

**Rollback plan**: If issues arise, immediately route 100% back to Blue (BM25). Vector index stays operational for debugging.

### Migrating from Vector to Hybrid Search

**Scenario**: Pure vector search deployed, but navigational queries underperforming. Add BM25 component for hybrid.

**Migration approach** (In-place upgrade):

1. **Index already has vector fields, add BM25 search**:
   - Existing index has content + content_vector fields
   - Content field already searchable → BM25 automatically available
   - No index rebuild needed (just change query code)

2. **Update query logic**:
   ```python
   # Before (vector only):
   results = search_client.search(
       search_text=None,
       vector_queries=[vector_query]
   )
   
   # After (hybrid):
   results = search_client.search(
       search_text=query,  # Add BM25 component
       vector_queries=[vector_query]  # Keep vector component
   )
   # Azure AI Search automatically applies RRF fusion
   ```

3. **A/B test hybrid vs. vector**:
   - Route 50% to hybrid, 50% to pure vector
   - Compare metrics: Precision@5, Recall@10, latency, CTR
   - If hybrid wins → Roll out to 100%
   - If no improvement → Revert to pure vector

**Rollback plan**: Change query code back to `search_text=None` (pure vector). No index changes needed.

### Migrating from Hybrid to Semantic Search

**Scenario**: Hybrid search working well, but want premium quality for Q&A queries.

**Migration approach** (Opt-in enhancement):

1. **Enable semantic configuration** (see Section 4):
   - Add semantic configuration to existing index (no rebuild needed)
   - Enable semantic search tier ($500/month after free tier)
   - Test semantic reranking on sample queries

2. **Selective semantic usage** (query routing):
   ```python
   def search(query):
       if is_question(query):  # "What is...", "How do I..."
           # Use semantic for Q&A (premium quality)
           return semantic_hybrid_search(query)
       else:
           # Use hybrid for other queries (cost-effective)
           return hybrid_search(query)
   
   def is_question(query):
       question_words = ["what", "how", "why", "when", "where", "who"]
       return any(query.lower().startswith(qw) for qw in question_words)
   ```

3. **Monitor cost vs. benefit**:
   - Track: % queries using semantic, cost increase, quality improvement
   - Example metrics:
     * 30% queries routed to semantic (Q&A patterns)
     * Cost increase: +$0.00030/query on those 30% (+$150/month at 500K queries)
     * Quality improvement: Precision@5 +15% on Q&A queries
   - Decision: If quality improvement justifies cost → Keep, else revert

**Rollback plan**: Remove `query_type=QueryType.SEMANTIC` from code. Index unchanged, no cost penalty.

---
*Next: [Azure Advanced Techniques](./04-azure-advanced-techniques.md)*