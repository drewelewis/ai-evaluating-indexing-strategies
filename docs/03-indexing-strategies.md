# Azure AI Search Indexing Strategies

This document provides a comprehensive comparison of Azure AI Search indexing strategies, their performance characteristics, and optimal use cases for Azure-native implementations.

## 🔍 Azure AI Search Overview

Azure AI Search (formerly Azure Cognitive Search) supports multiple search paradigms that can be evaluated and optimized:

1. **Full-text search** with BM25 scoring
2. **Vector search** with Azure OpenAI embeddings  
3. **Hybrid search** combining text and vectors
4. **Semantic search** with AI-powered re-ranking

### Azure Search Strategy Comparison Matrix

| Strategy | Latency | Recall | Precision | Semantic Understanding | Azure Setup | Monthly Cost (S1) |
|----------|---------|--------|-----------|----------------------|-------------|-------------------|
| **Full-text (BM25)** | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⚡ | $300 |
| **Vector Search** | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⚡⚡ | $800-1500 |
| **Hybrid Search** | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⚡⚡⚡ | $1000-2000 |
| **Semantic Search** | ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚡⚡ | $1500-3000 |

*Legend: ⚡ = Fast/Simple/Cheap, ⭐ = Good performance*
*Cost includes Azure AI Search + Azure OpenAI + semantic search fees*

## 1. Azure AI Search Full-text (BM25) Strategy

### How Azure AI Search BM25 Works
```python
# Azure AI Search BM25 Configuration
azure_bm25_index = {
    "name": "fulltext-search-index",
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
            "searchable": True,
            "analyzer": "en.microsoft"
        },
        {
            "name": "title",
            "type": "Edm.String",
            "searchable": True,
            "analyzer": "en.microsoft"
        }
    ],
    "scoringProfiles": [
        {
            "name": "boost-title",
            "text": {
                "weights": {
                    "title": 3.0,
                    "content": 1.0
                }
            }
        }
    ]
}
```

### Azure Search Client Implementation
```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

class AzureBM25Search:
    def __init__(self, endpoint, index_name, api_key):
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )
    
    def search(self, query, top_k=10):
        """Perform BM25 search with Azure AI Search."""
        results = self.client.search(
            search_text=query,
            top=top_k,
            scoring_profile="boost-title",
            query_type="simple"
        )
        
        return [
            {
                "id": result["id"],
                "title": result.get("title", ""),
                "content": result["content"],
                "score": result["@search.score"]
            }
            for result in results
        ]
    
    def search_with_filters(self, query, filters, top_k=10):
        """BM25 search with additional filters."""
        filter_expression = self._build_filter_expression(filters)
        
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
- **Latency**: 10-30ms for most queries
- **Throughput**: 1000+ QPS on Standard S1
- **Precision@5**: 0.65-0.75 for exact matches
- **Cost**: Most economical Azure search option
- **Best for**: Product catalogs, document search, exact term matching

### Azure BM25 Optimization
```python
class AzureBM25Optimizer:
    def __init__(self, search_client):
        self.search_client = search_client
    
    def optimize_scoring_profile(self, query_click_data):
        """Optimize scoring profile based on click-through data."""
        
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
                    "interpolation": "linear"
                }
            ]
        }
        
        return optimized_profile
```

## 2. Azure Vector Search Strategy

### Azure OpenAI Integration
```python
from openai import AzureOpenAI

class AzureVectorSearch:
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
        self.embedding_deployment = "text-embedding-ada-002"
    
    def create_vector_index(self):
        """Create vector search index configuration."""
        
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
                    "searchable": True
                },
                {
                    "name": "content_vector",
                    "type": "Collection(Edm.Single)",
                    "searchable": True,
                    "vectorSearchDimensions": 1536,
                    "vectorSearchProfileName": "vector-profile"
                }
            ],
            "vectorSearch": {
                "algorithms": [
                    {
                        "name": "hnsw-algorithm",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "metric": "cosine",
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500
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
        """Generate embedding using Azure OpenAI."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_deployment
        )
        return response.data[0].embedding
    
    def vector_search(self, query, top_k=10):
        """Perform vector search with Azure AI Search."""
        
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
            search_text=None,
            vector_queries=[vector_query],
            top=top_k
        )
        
        return [
            {
                "id": result["id"],
                "content": result["content"],
                "score": result["@search.score"]
            }
            for result in results
        ]
```

### Performance Characteristics
- **Latency**: 30-80ms including embedding generation
- **Semantic Understanding**: Excellent for conceptual queries
- **Recall**: 0.75-0.85 for semantic queries
- **Cost**: Higher due to Azure OpenAI embedding costs
- **Best for**: Q&A systems, semantic search, multilingual content

## 3. Azure Hybrid Search Strategy

### Hybrid Search Implementation
```python
class AzureHybridSearch:
    def __init__(self, search_client, openai_client):
        self.search_client = search_client
        self.openai_client = openai_client
        self.embedding_deployment = "text-embedding-ada-002"
    
    def create_hybrid_index(self):
        """Create hybrid search index supporting both text and vector."""
        
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
                    "analyzer": "en.microsoft"
                },
                {
                    "name": "title",
                    "type": "Edm.String",
                    "searchable": True,
                    "analyzer": "en.microsoft"
                },
                {
                    "name": "content_vector",
                    "type": "Collection(Edm.Single)",
                    "searchable": True,
                    "vectorSearchDimensions": 1536,
                    "vectorSearchProfileName": "hybrid-vector-profile"
                }
            ],
            "vectorSearch": {
                "algorithms": [
                    {
                        "name": "hybrid-hnsw",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "metric": "cosine",
                            "m": 8,
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
    
    def hybrid_search(self, query, top_k=10):
        """Perform hybrid search combining text and vector."""
        
        # Generate query embedding
        query_vector = self.embed_text(query)
        
        # Create vector query
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k * 2,  # Get more candidates for fusion
            fields="content_vector"
        )
        
        # Perform hybrid search (Azure AI Search handles RRF fusion automatically)
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top_k,
            query_type=QueryType.SIMPLE
        )
        
        return [
            {
                "id": result["id"],
                "title": result.get("title", ""),
                "content": result["content"],
                "score": result["@search.score"],
                "reranker_score": result.get("@search.reranker_score")
            }
            for result in results
        ]
    
    def embed_text(self, text):
        """Generate embedding using Azure OpenAI."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_deployment
        )
        return response.data[0].embedding
```

### Performance Characteristics
- **Latency**: 40-100ms depending on configuration
- **Best Overall Performance**: Combines benefits of both approaches
- **Precision@5**: 0.75-0.90 across different query types
- **Cost**: Moderate, requires both search and embedding services
- **Best for**: General-purpose search, e-commerce, enterprise applications

## 4. Azure Semantic Search Strategy

### Semantic Search Configuration
```python
class AzureSemanticSearch:
    def __init__(self, search_client):
        self.search_client = search_client
    
    def create_semantic_index(self):
        """Create index with semantic search configuration."""
        
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
                    "searchable": True
                },
                {
                    "name": "content",
                    "type": "Edm.String",
                    "searchable": True
                }
            ],
            "semanticSearch": {
                "configurations": [
                    {
                        "name": "semantic-config",
                        "prioritizedFields": {
                            "titleField": {
                                "fieldName": "title"
                            },
                            "prioritizedContentFields": [
                                {"fieldName": "content"}
                            ]
                        }
                    }
                ]
            }
        }
        
        return semantic_index_config
    
    def semantic_search(self, query, top_k=10):
        """Perform semantic search with AI-powered ranking."""
        
        results = self.search_client.search(
            search_text=query,
            top=top_k,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="semantic-config",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE
        )
        
        search_results = []
        semantic_answers = []
        
        for result in results:
            # Extract semantic captions and highlights
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
        
        # Extract semantic answers if available
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

### Performance Characteristics
- **Latency**: 60-150ms due to deep learning inference
- **Highest Quality**: Best relevance for natural language queries
- **Precision@5**: 0.80-0.95 for complex queries
- **Cost**: Most expensive due to AI processing
- **Best for**: Knowledge bases, customer support, Q&A systems

## 🎯 Azure Strategy Selection Framework

```python
def select_azure_search_strategy(requirements):
    """Select optimal Azure AI Search strategy based on requirements."""
    
    score_card = {
        "fulltext": 0,
        "vector": 0,
        "hybrid": 0,
        "semantic": 0
    }
    
    # Latency requirements
    max_latency = requirements.get("max_latency_ms", 1000)
    if max_latency < 50:
        score_card["fulltext"] += 3
        score_card["vector"] += 1
    elif max_latency < 100:
        score_card["fulltext"] += 2
        score_card["vector"] += 2
        score_card["hybrid"] += 1
    
    # Query type analysis
    query_types = requirements.get("query_types", {})
    
    if query_types.get("exact_match", 0) > 0.5:
        score_card["fulltext"] += 3
        score_card["hybrid"] += 2
    
    if query_types.get("semantic", 0) > 0.5:
        score_card["vector"] += 3
        score_card["semantic"] += 3
        score_card["hybrid"] += 2
    
    if query_types.get("natural_language", 0) > 0.5:
        score_card["semantic"] += 3
        score_card["vector"] += 2
    
    # Budget considerations
    budget_tier = requirements.get("budget_tier", "medium")
    if budget_tier == "low":
        score_card["fulltext"] += 3
        score_card["vector"] -= 1
        score_card["semantic"] -= 2
    elif budget_tier == "high":
        score_card["semantic"] += 2
        score_card["hybrid"] += 1
    
    # Quality requirements
    min_precision = requirements.get("min_precision_at_5", 0.7)
    if min_precision > 0.8:
        score_card["semantic"] += 3
        score_card["hybrid"] += 2
        score_card["vector"] += 1
    
    # Return ranked recommendations
    ranked = sorted(score_card.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "primary_recommendation": ranked[0][0],
        "scores": dict(ranked),
        "reasoning": _generate_azure_reasoning(requirements, ranked[0][0])
    }

def _generate_azure_reasoning(requirements, recommendation):
    """Generate explanation for Azure strategy recommendation."""
    
    reasons = []
    
    if recommendation == "fulltext":
        reasons.append("Azure AI Search full-text optimized for exact matches")
        reasons.append("Most cost-effective Azure search solution")
        reasons.append("Best for product catalogs and document search")
    elif recommendation == "vector":
        reasons.append("Azure OpenAI embeddings provide semantic understanding")
        reasons.append("Excellent for Q&A and conceptual search")
        reasons.append("Handles multilingual content effectively")
    elif recommendation == "hybrid":
        reasons.append("Azure RRF fusion combines text and vector benefits")
        reasons.append("Balanced performance across all query types")
        reasons.append("Enterprise-grade solution with good ROI")
    elif recommendation == "semantic":
        reasons.append("Azure semantic search provides highest quality results")
        reasons.append("Built-in answer extraction and captions")
        reasons.append("Best for knowledge management and support systems")
    
    return reasons
```

## 🔧 Implementation Roadmap

### Phase 1: Azure Foundation (Week 1)
1. **Azure Services Setup**
   - Create Azure AI Search service (Standard S1)
   - Configure Azure OpenAI service
   - Set up monitoring with Azure Monitor

2. **Basic Full-text Search**
   - Implement BM25 search with custom scoring
   - Create evaluation baseline
   - Establish cost baseline

### Phase 2: Vector Enhancement (Week 2)
1. **Azure Vector Search**
   - Configure vector indexing with HNSW
   - Integrate Azure OpenAI embeddings
   - Performance vs. cost optimization

### Phase 3: Hybrid Integration (Week 3)
1. **Azure Hybrid Search**
   - Implement hybrid indexing with RRF
   - Add intelligent query routing
   - A/B testing framework

### Phase 4: Semantic Enhancement (Week 4)
1. **Azure Semantic Search**
   - Enable semantic ranking and captions
   - Implement answer extraction
   - Production monitoring and optimization

---
*Next: [Azure Advanced Techniques](./04-azure-advanced-techniques.md)*