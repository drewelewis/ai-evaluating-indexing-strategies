# Vector Search

Complete guide to implementing semantic vector search using embeddings and approximate nearest neighbor (ANN) algorithms in Azure AI Search. This document provides comprehensive coverage of vector search theory, embedding strategies, HNSW algorithm configuration, performance optimization, and production deployment patterns for semantic search at scale.

## 📋 Table of Contents
- [Overview](#overview)
- [Vector Search Fundamentals](#vector-search-fundamentals)
- [HNSW Algorithm](#hnsw-algorithm)
- [Index Configuration](#index-configuration)
- [Embedding Strategies](#embedding-strategies)
- [Query Implementation](#query-implementation)
- [Performance Optimization](#performance-optimization)
- [Distance Metrics](#distance-metrics)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Vector search represents a paradigm shift in information retrieval, moving beyond keyword matching to semantic understanding. By representing text, images, and other data as high-dimensional numerical vectors (embeddings), vector search can understand the *meaning* and *context* of queries and documents, not just lexical similarity. This enables transformative search experiences that were impossible with traditional full-text search alone.

### What is Vector Search?

Vector search, also known as semantic search or neural search, enables similarity-based retrieval by representing text as high-dimensional vectors (embeddings) and finding similar vectors using mathematical distance calculations. Instead of matching keywords, vector search matches *concepts* and *meanings*.

**How It Works:**
1. **Text → Vectors**: Transform text into numerical representations using embedding models
2. **Vector Index**: Store embeddings in specialized data structures (HNSW) for fast retrieval
3. **Similarity Search**: Find vectors nearest to the query vector using distance metrics
4. **Ranking**: Return documents ordered by semantic similarity to the query

**Key Benefits:**
- **Semantic understanding**: Matches meaning, not just keywords
  - Query "laptop for ML" matches "machine learning workstation" (no shared keywords)
  - Query "affordable housing" matches "low-cost apartments" (semantic equivalence)
  
- **Cross-lingual search**: Works across languages
  - English query "hello" matches Spanish "hola" (same semantic space)
  - Multilingual embeddings understand 100+ languages in unified vector space
  
- **Handles synonyms naturally**: Finds conceptually similar content
  - "car" matches "automobile", "vehicle", "sedan" without synonym dictionaries
  - "cheap" matches "affordable", "budget-friendly", "inexpensive" automatically
  
- **Query reformulation immunity**: Better handles poorly worded queries
  - Typos and grammatical errors less impactful (semantic meaning preserved)
  - Natural language questions work better than keyword lists

**Real-World Performance Comparison:**

| Scenario | Full-Text Search | Vector Search | Improvement |
|----------|------------------|---------------|-------------|
| Exact keyword match | 95% precision | 85% precision | -10% (keyword still better) |
| Semantic similarity | 45% precision | 90% precision | +100% (2× better) |
| Cross-lingual queries | 0% (no match) | 75% precision | ∞ (enables new capability) |
| Question answering | 30% precision | 85% precision | +183% (nearly 3× better) |
| Zero-result rate | 15% queries | 5% queries | -67% (fewer dead ends) |

### Vector Search Architecture

Understanding the complete vector search pipeline is essential for effective implementation:

```
┌─────────────────────────────────────────────────────────────────┐
│                     VECTOR SEARCH PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

1. EMBEDDING GENERATION (Offline - During Indexing)
   ┌──────────────────┐
   │  Source Document │
   │  "Gaming laptop  │
   │   with RTX 4090" │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────────────┐
   │  Azure OpenAI API        │
   │  text-embedding-3-large  │
   │  (3,072 dimensions)      │
   └────────┬─────────────────┘
            │
            ▼
   ┌──────────────────────────┐
   │  Document Embedding      │
   │  [0.023, -0.456, 0.789,  │
   │   ..., 0.123] (3072 dims)│
   └────────┬─────────────────┘
            │
            ▼
   ┌──────────────────────────┐
   │  Azure AI Search Index   │
   │  HNSW Vector Index       │
   │  (Optimized for ANN)     │
   └──────────────────────────┘

2. QUERY PROCESSING (Online - Real-Time)
   ┌──────────────────┐
   │  User Query      │
   │  "laptop for ML" │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────────────┐
   │  Azure OpenAI API        │
   │  SAME model & version    │
   │  text-embedding-3-large  │
   └────────┬─────────────────┘
            │
            ▼
   ┌──────────────────────────┐
   │  Query Embedding         │
   │  [0.034, -0.423, 0.801,  │
   │   ..., 0.098] (3072 dims)│
   └────────┬─────────────────┘
            │
            ▼
   ┌──────────────────────────┐
   │  Azure AI Search         │
   │  HNSW Vector Index       │
   │  Find k Nearest Neighbors│
   │  (Cosine Similarity)     │
   └────────┬─────────────────┘
            │
            ▼
   ┌──────────────────────────┐
   │  Ranked Results          │
   │  1. Gaming workstation   │
   │     (similarity: 0.92)   │
   │  2. ML development PC    │
   │     (similarity: 0.89)   │
   │  3. AI training laptop   │
   │     (similarity: 0.85)   │
   └──────────────────────────┘
```

**Critical Pipeline Requirements:**
- **Consistency**: Use SAME embedding model for indexing and querying
- **Version pinning**: Model updates change embeddings, breaking compatibility
- **Dimension matching**: Query vector must have same dimensions as indexed vectors
- **Normalization**: If using dot product, vectors must be normalized

### Real-World Application Scenario

**Company**: Contoso Knowledge Base (Enterprise IT documentation)
**Content**: 250,000 technical articles, support tickets, and knowledge base entries
**Challenge**: Employees struggle to find relevant documentation using keyword search
- Query "How to reset password" returns 10,000 articles (too many)
- Query "password reset procedure" returns different results (synonym problem)
- Cross-language queries fail (English query doesn't match Spanish docs)

**Solution Architecture:**
1. **Embedding Generation**: Use Azure OpenAI `text-embedding-3-large` (3,072 dimensions)
2. **Index Design**: Separate vector fields for title (concise) and content (detailed)
3. **HNSW Configuration**: m=4, efConstruction=400 (balanced quality/performance)
4. **Hybrid Search**: Combine vector (70% weight) + BM25 (30% weight) for best results
5. **Performance**: P95 latency <150ms for vector search at 500 QPS

**Business Impact:**
- 65% improvement in search relevance (measured by user satisfaction scores)
- 80% reduction in zero-result searches (semantic matching finds content)
- 40% reduction in support tickets (employees find answers faster)
- $250/month embedding costs + $500/month search service = $750/month total
- ROI: $120K/year in reduced support costs vs $9K/year search costs (13× ROI)

**Cost Breakdown:**
```
Embedding Generation (One-Time + Incremental):
- Initial: 250,000 articles × 500 tokens avg = 125M tokens
- Cost: 125M tokens ÷ 1M × $0.13 = $16.25 (one-time)
- Incremental: 500 new articles/month × 500 tokens = 250K tokens/month
- Cost: $0.03/month (negligible)

Query Embedding (Ongoing):
- Volume: 100,000 queries/month × 10 tokens avg = 1M tokens/month
- Cost: 1M ÷ 1M × $0.13 = $0.13/month (negligible)

Total Embedding Costs: ~$0.16/month (after initial $16 investment)

Azure AI Search Costs:
- Tier: S1 (25GB storage, 100 QPS, $250/month)
- Vector index overhead: ~2× storage (250K docs × 3072 dims × 4 bytes = 3GB vectors)
- Total storage: 8GB source + 3GB vectors + 5GB full-text index = 16GB (within S1 limit)

Total Monthly Cost: $250/month (search) + $0.16 (embeddings) = $250.16/month
```

This document will guide you through implementing a similar solution, from embedding strategy to production optimization.

---

## Vector Search Fundamentals

Before diving into implementation, understanding the mathematical and conceptual foundations of vector search is essential for making informed architectural decisions and diagnosing relevance issues.

### Understanding Embeddings

Embeddings are dense numerical representations of text (or other data) in high-dimensional space, where semantically similar content is located near each other. This seemingly simple concept unlocks powerful semantic search capabilities.
                'range': '-1 to 1 (1 = identical)',
                'use': 'Default for most embeddings'
            },
            'euclidean_distance': {
                'formula': 'd = √(Σ(ai - bi)²)',
                'range': '0 to ∞ (0 = identical)',
                'use': 'Absolute distance in space'
            },
            'dot_product': {
                'formula': 'A·B = Σ(ai × bi)',
                'range': '-∞ to ∞ (higher = more similar)',
                'use': 'Fast, works with normalized vectors'
            }
        }

# Usage
concepts = EmbeddingConcepts()
embedding_info = concepts.what_are_embeddings()
print(f"Dimensions: {embedding_info['dimensions']}")

similarity_info = concepts.similarity_calculation()
print(f"Cosine similarity range: {similarity_info['cosine_similarity']['range']}")
```

### Vector vs Full-Text Search

```python
class SearchComparison:
    """Compare vector and full-text search."""
    
    @staticmethod
    def when_to_use_vector_search():
        """Use cases for vector search."""
        return {
            'ideal_for': [
                'Semantic similarity (concept matching)',
                'Cross-lingual search',
                'Question answering',
                'Recommendation systems',
                'Image/document similarity',
                'Handling typos and variations'
            ],
            'examples': [
                {
                    'query': 'laptop for ML',
                    'vector_finds': ['machine learning workstation', 'AI development computer'],
                    'fulltext_finds': ['laptop', 'ML mentioned explicitly']
                },
                {
                    'query': 'affordable housing',
                    'vector_finds': ['low-cost apartments', 'budget-friendly homes'],
                    'fulltext_finds': ['affordable AND housing']
                }
            ]
        }
    
    @staticmethod
    def when_to_use_fulltext_search():
        """Use cases for full-text search."""
        return {
            'ideal_for': [
                'Exact keyword matching',
                'Product codes/SKUs',
                'Legal/compliance (exact terms)',
                'Technical documentation (specific terms)',
                'Boolean queries',
                'Phrase matching'
            ],
            'examples': [
                {
                    'query': 'SKU-12345',
                    'fulltext_better': 'Exact match required'
                },
                {
                    'query': 'Article 5 Section 2',
                    'fulltext_better': 'Precise legal reference'
                }
            ]
        }
    
    @staticmethod
    def hybrid_recommended_for():
        """When to combine both."""
        return [
            'E-commerce product search',
            'Enterprise document search',
            'Customer support knowledge base',
            'General-purpose search applications'
        ]
```

---

## HNSW Algorithm

### Hierarchical Navigable Small World

```python
class HNSWExplained:
    """Understanding HNSW algorithm in Azure AI Search."""
    
    @staticmethod
    def what_is_hnsw():
        """
        HNSW: Hierarchical Navigable Small World
        
        An approximate nearest neighbor (ANN) algorithm that provides:
        - Fast search (logarithmic complexity)
        - High recall (finds most relevant results)
        - Efficient indexing
        
        Trade-off: Approximate (not exact) results
        """
        return {
            'algorithm': 'Graph-based ANN',
            'complexity': 'O(log n) search time',
            'recall': '~0.95-0.99 typical',
            'index_time': 'O(n log n)',
            'memory': 'Higher than inverted index'
        }
    
    @staticmethod
    def hnsw_parameters():
        """
        HNSW configuration parameters.
        
        m: Number of bi-directional links (default: 4)
        - Higher m = Better recall, more memory
        - Range: 4-10
        - Recommendation: 4 for most cases
        
        efConstruction: Size of dynamic candidate list during indexing (default: 400)
        - Higher efConstruction = Better recall, slower indexing
        - Range: 100-1000
        - Recommendation: 400-800 for production
        
        efSearch: Size of dynamic candidate list during search (default: 500)
        - Higher efSearch = Better recall, slower queries
        - Range: 100-1000
        - Recommendation: 500 for balanced performance
        """
        return {
            'm': {
                'default': 4,
                'range': '4-10',
                'impact': 'Recall vs memory',
                'recommendation': 4
            },
            'efConstruction': {
                'default': 400,
                'range': '100-1000',
                'impact': 'Index quality vs build time',
                'recommendation': 400
            },
            'efSearch': {
                'default': 500,
                'range': '100-1000',
                'impact': 'Recall vs query latency',
                'recommendation': 500
            }
        }
    
    @staticmethod
    def choosing_parameters(index_size, recall_requirement):
        """
        Choose HNSW parameters based on requirements.
        
        Args:
            index_size: 'small' (<100K), 'medium' (100K-1M), 'large' (>1M)
            recall_requirement: 'balanced', 'high_recall', 'low_latency'
        """
        configs = {
            'balanced': {'m': 4, 'efConstruction': 400, 'efSearch': 500},
            'high_recall': {'m': 8, 'efConstruction': 800, 'efSearch': 800},
            'low_latency': {'m': 4, 'efConstruction': 400, 'efSearch': 300}
        }
        
        config = configs.get(recall_requirement, configs['balanced'])
        
        # Adjust for index size
        if index_size == 'large':
            config['efConstruction'] = min(config['efConstruction'] * 1.5, 1000)
        
        return config

# Usage
hnsw = HNSWExplained()

# Get parameter recommendations
config = hnsw.choosing_parameters(
    index_size='medium',
    recall_requirement='high_recall'
)
print(f"Recommended HNSW config: {config}")
```

---

## Index Configuration

### Vector Field Definition

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch
)
from azure.core.credentials import AzureKeyCredential

class VectorIndexBuilder:
    """Build vector search enabled index."""
    
    def __init__(self, endpoint, api_key):
        self.client = SearchIndexClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
    
    def create_vector_index(self, index_name="products_vector"):
        """Create index with vector search capability."""
        
        # Define HNSW algorithm configuration
        algorithm_config = HnswAlgorithmConfiguration(
            name="hnsw-config",
            parameters={
                "m": 4,
                "efConstruction": 400,
                "efSearch": 500,
                "metric": "cosine"  # cosine, euclidean, or dotProduct
            }
        )
        
        # Define vector search configuration
        vector_search = VectorSearch(
            algorithms=[algorithm_config],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config"
                )
            ]
        )
        
        # Define index fields
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True
            ),
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True
            ),
            SearchableField(
                name="description",
                type=SearchFieldDataType.String,
                searchable=True
            ),
            SearchField(
                name="titleVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # text-embedding-ada-002
                vector_search_profile_name="vector-profile"
            ),
            SearchField(
                name="descriptionVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="vector-profile"
            ),
            SimpleField(
                name="category",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SimpleField(
                name="price",
                type=SearchFieldDataType.Double,
                filterable=True,
                sortable=True
            )
        ]
        
        # Create index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        result = self.client.create_or_update_index(index)
        print(f"✅ Vector index '{index_name}' created")
        return result
    
    def create_multimodal_vector_index(self, index_name="documents_multimodal"):
        """Create index with multiple vector fields for different content types."""
        
        # HNSW configuration
        algorithm_config = HnswAlgorithmConfiguration(
            name="hnsw-config",
            parameters={
                "m": 4,
                "efConstruction": 400,
                "efSearch": 500,
                "metric": "cosine"
            }
        )
        
        vector_search = VectorSearch(
            algorithms=[algorithm_config],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config"
                )
            ]
        )
        
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            
            # Separate vectors for different content
            SearchField(
                name="titleVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="vector-profile"
            ),
            SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="vector-profile"
            ),
            SearchField(
                name="combinedVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="vector-profile"
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        result = self.client.create_or_update_index(index)
        print(f"✅ Multimodal vector index '{index_name}' created")
        return result

# Usage
import os

builder = VectorIndexBuilder(
    endpoint=os.getenv("SEARCH_ENDPOINT"),
    api_key=os.getenv("SEARCH_API_KEY")
)

# Create basic vector index
vector_index = builder.create_vector_index()

# Create multimodal index
multimodal_index = builder.create_multimodal_vector_index()
```

---

## Embedding Strategies

### Document Indexing with Vectors

```python
from azure.search.documents import SearchClient
from openai import AzureOpenAI

class VectorDocumentIndexer:
    """Index documents with vector embeddings."""
    
    def __init__(self, search_client, openai_client, deployment_name="text-embedding-ada-002"):
        self.search_client = search_client
        self.openai_client = openai_client
        self.deployment_name = deployment_name
    
    def generate_embedding(self, text):
        """Generate embedding for text."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.deployment_name
        )
        return response.data[0].embedding
    
    def index_document(self, document):
        """
        Index a single document with embeddings.
        
        Args:
            document: Dict with 'id', 'title', 'description', etc.
        """
        # Generate embeddings
        title_vector = self.generate_embedding(document['title'])
        description_vector = self.generate_embedding(document['description'])
        
        # Prepare document for indexing
        search_document = {
            'id': document['id'],
            'title': document['title'],
            'description': document['description'],
            'titleVector': title_vector,
            'descriptionVector': description_vector,
            'category': document.get('category'),
            'price': document.get('price')
        }
        
        # Upload to search index
        result = self.search_client.upload_documents(documents=[search_document])
        return result
    
    def index_documents_batch(self, documents, batch_size=100):
        """
        Index multiple documents efficiently.
        
        Args:
            documents: List of document dicts
            batch_size: Number of documents per batch
        """
        total_indexed = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            search_docs = []
            
            for doc in batch:
                # Generate embeddings
                title_vector = self.generate_embedding(doc['title'])
                description_text = doc.get('description', '')
                description_vector = self.generate_embedding(description_text) if description_text else None
                
                search_doc = {
                    'id': doc['id'],
                    'title': doc['title'],
                    'description': doc.get('description'),
                    'titleVector': title_vector,
                    'descriptionVector': description_vector,
                    'category': doc.get('category'),
                    'price': doc.get('price')
                }
                search_docs.append(search_doc)
            
            # Upload batch
            result = self.search_client.upload_documents(documents=search_docs)
            total_indexed += len(batch)
            print(f"Indexed {total_indexed}/{len(documents)} documents")
        
        return total_indexed
    
    def index_with_combined_embedding(self, document, title_weight=0.3, content_weight=0.7):
        """
        Create weighted combined embedding from multiple fields.
        
        Args:
            document: Document dict
            title_weight: Weight for title (0-1)
            content_weight: Weight for content (0-1)
        """
        import numpy as np
        
        # Generate individual embeddings
        title_vector = np.array(self.generate_embedding(document['title']))
        content_vector = np.array(self.generate_embedding(document.get('content', '')))
        
        # Weighted combination
        combined_vector = (
            title_vector * title_weight +
            content_vector * content_weight
        )
        
        # Normalize
        combined_vector = combined_vector / np.linalg.norm(combined_vector)
        
        search_doc = {
            'id': document['id'],
            'title': document['title'],
            'content': document.get('content'),
            'titleVector': title_vector.tolist(),
            'contentVector': content_vector.tolist(),
            'combinedVector': combined_vector.tolist()
        }
        
        result = self.search_client.upload_documents(documents=[search_doc])
        return result

# Usage
from azure.search.documents import SearchClient
from openai import AzureOpenAI

search_client = SearchClient(
    endpoint=os.getenv("SEARCH_ENDPOINT"),
    index_name="products_vector",
    credential=AzureKeyCredential(os.getenv("SEARCH_API_KEY"))
)

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

indexer = VectorDocumentIndexer(search_client, openai_client)

# Index single document
document = {
    'id': '1',
    'title': 'Gaming Laptop',
    'description': 'High-performance laptop for gaming and machine learning',
    'category': 'Computers',
    'price': 1499.99
}

indexer.index_document(document)

# Batch indexing
documents = [
    {'id': '1', 'title': 'Product 1', 'description': 'Description 1'},
    {'id': '2', 'title': 'Product 2', 'description': 'Description 2'},
    # ... more documents
]

indexer.index_documents_batch(documents, batch_size=100)
```

---

## Query Implementation

### Vector Search Queries

```python
class VectorSearcher:
    """Execute vector search queries."""
    
    def __init__(self, search_client, openai_client, deployment_name="text-embedding-ada-002"):
        self.search_client = search_client
        self.openai_client = openai_client
        self.deployment_name = deployment_name
    
    def generate_query_embedding(self, query_text):
        """Generate embedding for search query."""
        response = self.openai_client.embeddings.create(
            input=query_text,
            model=self.deployment_name
        )
        return response.data[0].embedding
    
    def vector_search(self, query_text, vector_fields=None, top=10):
        """
        Pure vector search.
        
        Args:
            query_text: Natural language query
            vector_fields: List of vector field names to search
            top: Number of results
        """
        from azure.search.documents.models import VectorizedQuery
        
        # Generate query embedding
        query_vector = self.generate_query_embedding(query_text)
        
        # Default to searching title vectors
        if vector_fields is None:
            vector_fields = ["titleVector"]
        
        # Create vector query
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top,
            fields=",".join(vector_fields)
        )
        
        # Execute search
        results = self.search_client.search(
            search_text=None,  # Pure vector search
            vector_queries=[vector_query],
            top=top
        )
        
        documents = []
        for result in results:
            documents.append({
                'id': result['id'],
                'title': result.get('title'),
                'description': result.get('description'),
                'score': result['@search.score']
            })
        
        return documents
    
    def multi_vector_search(self, query_text, top=10):
        """
        Search across multiple vector fields.
        
        Useful when you have separate embeddings for title, content, etc.
        """
        from azure.search.documents.models import VectorizedQuery
        
        query_vector = self.generate_query_embedding(query_text)
        
        # Search multiple vector fields
        vector_queries = [
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top,
                fields="titleVector"
            ),
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top,
                fields="descriptionVector"
            )
        ]
        
        results = self.search_client.search(
            search_text=None,
            vector_queries=vector_queries,
            top=top
        )
        
        return list(results)
    
    def filtered_vector_search(self, query_text, filter_expression, top=10):
        """
        Vector search with filters.
        
        Args:
            query_text: Search query
            filter_expression: OData filter (e.g., "category eq 'Computers'")
            top: Number of results
        """
        from azure.search.documents.models import VectorizedQuery
        
        query_vector = self.generate_query_embedding(query_text)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,  # Over-fetch for filtering
            fields="titleVector"
        )
        
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expression,
            top=top
        )
        
        return list(results)
    
    def vector_search_with_facets(self, query_text, facets, top=10):
        """
        Vector search with faceted navigation.
        
        Args:
            query_text: Search query
            facets: List of facet fields (e.g., ['category', 'price'])
            top: Number of results
        """
        from azure.search.documents.models import VectorizedQuery
        
        query_vector = self.generate_query_embedding(query_text)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top,
            fields="titleVector"
        )
        
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            facets=facets,
            top=top
        )
        
        # Extract results and facets
        documents = []
        for result in results:
            documents.append({
                'id': result['id'],
                'title': result.get('title'),
                'score': result['@search.score']
            })
        
        facet_results = results.get_facets()
        
        return {
            'documents': documents,
            'facets': facet_results
        }

# Usage
searcher = VectorSearcher(search_client, openai_client)

# Simple vector search
results = searcher.vector_search(
    query_text="laptop for machine learning",
    top=10
)

for doc in results:
    print(f"{doc['title']} (score: {doc['score']:.4f})")

# Filtered vector search
filtered_results = searcher.filtered_vector_search(
    query_text="gaming computer",
    filter_expression="price le 2000 and category eq 'Computers'",
    top=10
)

# Search with facets
faceted_results = searcher.vector_search_with_facets(
    query_text="laptop",
    facets=["category", "price,values:500|1000|1500|2000"],
    top=10
)

print(f"Categories: {faceted_results['facets']['category']}")
```

---

## Performance Optimization

### Optimizing Vector Search

```python
class VectorSearchOptimization:
    """Optimization techniques for vector search."""
    
    @staticmethod
    def index_optimization_tips():
        """Tips for optimizing vector index."""
        return {
            'hnsw_tuning': {
                'for_recall': 'Increase m (6-8) and efConstruction (600-800)',
                'for_speed': 'Decrease efSearch (300-400)',
                'for_memory': 'Decrease m (4) and use fewer vector fields'
            },
            'field_strategy': {
                'title_only': 'Fastest, good for short content',
                'content_only': 'Best for long documents',
                'separate_fields': 'Most flexible, can weight differently',
                'combined_field': 'Balance of performance and relevance'
            },
            'dimension_reduction': {
                'note': 'Azure uses 1536 dims for ada-002',
                'consideration': 'Cannot reduce without custom model'
            }
        }
    
    @staticmethod
    def query_optimization_tips():
        """Tips for optimizing queries."""
        return {
            'k_parameter': {
                'small_k': 'Faster, less comprehensive (k=10-20)',
                'large_k': 'Slower, more comprehensive (k=50-100)',
                'recommendation': 'Use smallest k that meets needs'
            },
            'filtering': {
                'pre_filter': 'Filter before vector search when possible',
                'over_fetch': 'Request more results (k=50) when filtering to get top 10'
            },
            'caching': {
                'query_cache': 'Identical queries cached ~1 minute',
                'embedding_cache': 'Cache query embeddings in application'
            },
            'batching': {
                'batch_queries': 'Process multiple queries together',
                'avoid': 'Sequential query execution'
            }
        }
    
    @staticmethod
    def estimate_performance(num_documents, dimensions=1536, m=4):
        """
        Estimate query performance.
        
        Args:
            num_documents: Number of indexed documents
            dimensions: Vector dimensions
            m: HNSW m parameter
        """
        import math
        
        # Rough estimates based on HNSW complexity
        hops = math.log2(num_documents)
        comparisons_per_hop = m * 2
        total_comparisons = hops * comparisons_per_hop
        
        # Latency estimate (very rough)
        base_latency_ms = 5
        per_comparison_ms = 0.01
        estimated_latency = base_latency_ms + (total_comparisons * per_comparison_ms)
        
        # Memory estimate
        bytes_per_vector = dimensions * 4  # 4 bytes per float
        graph_overhead = m * 8  # Rough estimate for links
        memory_per_doc_kb = (bytes_per_vector + graph_overhead) / 1024
        total_memory_mb = (memory_per_doc_kb * num_documents) / 1024
        
        return {
            'estimated_hops': int(hops),
            'estimated_comparisons': int(total_comparisons),
            'estimated_latency_ms': round(estimated_latency, 2),
            'estimated_memory_mb': round(total_memory_mb, 2)
        }

# Usage
opt = VectorSearchOptimization()

# Get optimization tips
index_tips = opt.index_optimization_tips()
print("For high recall:", index_tips['hnsw_tuning']['for_recall'])

# Estimate performance
perf = opt.estimate_performance(num_documents=1000000, m=4)
print(f"Estimated latency for 1M docs: {perf['estimated_latency_ms']}ms")
print(f"Estimated memory: {perf['estimated_memory_mb']}MB")
```

---

## Distance Metrics

### Understanding Similarity Metrics

```python
import numpy as np

class SimilarityMetrics:
    """Calculate and compare different similarity metrics."""
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Cosine similarity: Measures angle between vectors.
        
        Range: -1 (opposite) to 1 (identical)
        Use: Most common for text embeddings
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def euclidean_distance(vec1, vec2):
        """
        Euclidean distance: Straight-line distance in space.
        
        Range: 0 (identical) to ∞
        Use: When absolute distance matters
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        return np.linalg.norm(vec1 - vec2)
    
    @staticmethod
    def dot_product(vec1, vec2):
        """
        Dot product: Sum of element-wise products.
        
        Range: -∞ to ∞ (higher = more similar)
        Use: Fast, when vectors are normalized
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        return np.dot(vec1, vec2)
    
    @staticmethod
    def compare_metrics(vec1, vec2):
        """Compare all similarity metrics."""
        cosine = SimilarityMetrics.cosine_similarity(vec1, vec2)
        euclidean = SimilarityMetrics.euclidean_distance(vec1, vec2)
        dot = SimilarityMetrics.dot_product(vec1, vec2)
        
        return {
            'cosine_similarity': cosine,
            'euclidean_distance': euclidean,
            'dot_product': dot
        }

# Usage
vec1 = [0.1, 0.2, 0.3, 0.4]
vec2 = [0.15, 0.25, 0.35, 0.45]

metrics = SimilarityMetrics.compare_metrics(vec1, vec2)
print(f"Cosine similarity: {metrics['cosine_similarity']:.4f}")
print(f"Euclidean distance: {metrics['euclidean_distance']:.4f}")
print(f"Dot product: {metrics['dot_product']:.4f}")
```

---

## Best Practices

Vector search is a powerful technology, but it requires careful implementation to achieve optimal relevance, performance, and cost-efficiency. These best practices are derived from production deployments handling billions of vector searches across diverse use cases.

### Embedding Model Selection and Management

**✅ DO: Use Cosine Similarity for Text Embeddings (Default)**
- **Why**: Most embedding models (OpenAI, Cohere, etc.) are optimized for cosine similarity
- **How**: Embeddings are pre-normalized during training
- **Benefit**: Cosine similarity is rotation-invariant and scale-invariant
- **Implementation**: Azure AI Search default for vector fields

```python
from azure.search.documents.indexes.models import VectorSearch, VectorSearchAlgorithmConfiguration

vector_search = VectorSearch(
    algorithms=[
        VectorSearchAlgorithmConfiguration(
            name="hnsw-cosine",
            kind="hnsw",
            hnsw_parameters={
                "m": 4,
                "efConstruction": 400,
                "efSearch": 500,
                "metric": "cosine"  # ✅ Default for text embeddings
            }
        )
    ]
)
```

**❌ DON'T: Mix Embeddings from Different Models**
- **Problem**: Each model has unique vector space geometry
- **Impact**: Similarity scores meaningless across different models
- **Example**: text-embedding-ada-002 (1536 dims) + text-embedding-3-large (3072 dims) = incompatible

**Incompatibility Example:**
```python
# BAD: Mixing embedding models
document_embedding = generate_embedding_ada_002("laptop for ML")  # 1536 dims
query_embedding = generate_embedding_3_large("laptop for ML")      # 3072 dims
# Result: Dimension mismatch error OR incorrect similarity if padded

# GOOD: Consistent embedding model
document_embedding = generate_embedding_3_large("laptop for ML")  # 3072 dims
query_embedding = generate_embedding_3_large("laptop for ML")      # 3072 dims
# Result: Accurate similarity calculation
```

**✅ DO: Generate Embeddings Consistently (Same Model, Same Version)**
- **Critical**: Model updates change vector space, breaking compatibility
- **Action**: Pin specific model version in production
- **Monitoring**: Track model version in metadata for audit trail

```python
import openai
from datetime import datetime

class ConsistentEmbeddingGenerator:
    """Generate embeddings with version tracking."""
    
    def __init__(self, model="text-embedding-3-large", api_key=None):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        self.model_version = self._get_model_version()
    
    def _get_model_version(self):
        """Track model version for consistency."""
        # In production, retrieve from metadata or config
        return {
            'model': self.model,
            'timestamp': datetime.utcnow().isoformat(),
            'dimensions': 3072 if '3-large' in self.model else 1536
        }
    
    def generate_embedding(self, text, metadata=None):
        """Generate embedding with version tracking."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        
        embedding = response.data[0].embedding
        
        # Return embedding with metadata
        return {
            'embedding': embedding,
            'model_version': self.model_version,
            'text_length': len(text),
            'metadata': metadata or {}
        }
    
    def verify_compatibility(self, document_model_version):
        """Verify query embedding compatible with indexed documents."""
        if document_model_version['model'] != self.model_version['model']:
            raise ValueError(
                f"Model mismatch: Documents use {document_model_version['model']}, "
                f"query uses {self.model_version['model']}"
            )
        return True

# Usage
generator = ConsistentEmbeddingGenerator(model="text-embedding-3-large")

# Index time
doc_embedding_data = generator.generate_embedding(
    "Gaming laptop with RTX 4090",
    metadata={'doc_id': '12345'}
)

# Query time
query_embedding_data = generator.generate_embedding("laptop for ML")

# Verify compatibility
generator.verify_compatibility(doc_embedding_data['model_version'])
```

### Index Design and Field Configuration

**✅ DO: Index Separate Vector Fields for Title and Content**
- **Why**: Different granularities capture different semantic signals
- **Title field**: Concise, high-level concepts (50-100 tokens)
- **Content field**: Detailed, comprehensive information (500-2000 tokens)
- **Benefit**: Query against title for precision, content for recall

```python
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    VectorSearchField
)

# Separate vector fields for different granularities
fields = [
    SearchField(
        name="id",
        type=SearchFieldDataType.String,
        key=True
    ),
    SearchField(
        name="title",
        type=SearchFieldDataType.String,
        searchable=True  # For hybrid search
    ),
    VectorSearchField(
        name="title_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=3072,
        vector_search_configuration="hnsw-cosine"
    ),
    SearchField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True
    ),
    VectorSearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=3072,
        vector_search_configuration="hnsw-cosine"
    )
]

# Query strategy: Search both fields, combine results
results_title = search_client.search(
    search_text=None,
    vector=query_embedding,
    vector_fields=["title_vector"],  # High precision
    top=10
)

results_content = search_client.search(
    search_text=None,
    vector=query_embedding,
    vector_fields=["content_vector"],  # High recall
    top=10
)

# Merge and re-rank based on use case
```

**❌ DON'T: Index Embeddings Without Metadata Fields**
- **Problem**: Cannot filter or facet results (vector search only)
- **Impact**: Poor user experience (no category filters, no sorting)
- **Solution**: Always include filterable/sortable metadata fields

```python
# BAD: Vector field only (no filtering possible)
fields = [
    SearchField(name="id", type=SearchFieldDataType.String, key=True),
    VectorSearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=3072,
        vector_search_configuration="hnsw-cosine"
    )
]
# Result: Can search, but cannot filter by category, date, price, etc.

# GOOD: Vector + metadata fields
fields = [
    SearchField(name="id", type=SearchFieldDataType.String, key=True),
    VectorSearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=3072,
        vector_search_configuration="hnsw-cosine"
    ),
    # ✅ Add filterable metadata
    SearchField(
        name="category",
        type=SearchFieldDataType.String,
        filterable=True,
        facetable=True
    ),
    SearchField(
        name="date",
        type=SearchFieldDataType.DateTimeOffset,
        filterable=True,
        sortable=True
    ),
    SearchField(
        name="price",
        type=SearchFieldDataType.Double,
        filterable=True,
        sortable=True
    )
]
# Result: Can filter by category, sort by date, facet by price
```

### Query Optimization and Caching

**✅ DO: Cache Query Embeddings in Your Application**
- **Why**: Embedding generation is the slowest part of vector search (50-200ms)
- **Impact**: 50-80% latency reduction for repeated queries
- **Implementation**: Redis, in-memory cache, or Azure Cache for Redis

```python
import hashlib
import redis
import json

class EmbeddingCache:
    """Cache query embeddings to reduce latency and costs."""
    
    def __init__(self, redis_client, ttl_seconds=3600):
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.hit_count = 0
        self.miss_count = 0
    
    def get_or_generate_embedding(self, query_text, embedding_generator):
        """Get cached embedding or generate new one."""
        # Create cache key
        cache_key = f"emb:{hashlib.md5(query_text.encode()).hexdigest()}"
        
        # Try cache first
        cached = self.redis.get(cache_key)
        if cached:
            self.hit_count += 1
            return json.loads(cached)
        
        # Cache miss: Generate embedding
        self.miss_count += 1
        embedding = embedding_generator(query_text)
        
        # Store in cache
        self.redis.setex(
            cache_key,
            self.ttl,
            json.dumps(embedding)
        )
        
        return embedding
    
    def get_cache_stats(self):
        """Return cache hit rate."""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            'hit_rate': hit_rate,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count
        }

# Usage
redis_client = redis.Redis(host='localhost', port=6379)
cache = EmbeddingCache(redis_client, ttl_seconds=3600)  # 1 hour TTL

# First query: Cache miss (200ms total)
embedding = cache.get_or_generate_embedding(
    "laptop for ML",
    lambda text: generate_embedding(text)
)

# Repeated query: Cache hit (5ms total, 40× faster!)
embedding = cache.get_or_generate_embedding(
    "laptop for ML",
    lambda text: generate_embedding(text)
)

# Check performance
stats = cache.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")  # 50% typical in production
```

**✅ DO: Over-Fetch When Filtering (Retrieve k=50 to Get Top 10 After Filters)**
- **Why**: Filters applied AFTER vector search reduces result count
- **Problem**: Requesting top=10 might yield only 2 results after filtering
- **Solution**: Over-fetch by 3-5× to ensure sufficient filtered results

```python
# BAD: Under-fetching with filters
results = search_client.search(
    search_text=None,
    vector=query_embedding,
    vector_fields=["content_vector"],
    filter="category eq 'Laptops' and price le 1500",
    top=10  # ❌ Might get only 2-3 results after filter
)
# Result: Only 3 laptops under $1500 in top 10 vectors

# GOOD: Over-fetch to compensate for filtering
results = search_client.search(
    search_text=None,
    vector=query_embedding,
    vector_fields=["content_vector"],
    filter="category eq 'Laptops' and price le 1500",
    top=50  # ✅ Fetch 50, filter down to 10+
)
# Then manually take top 10 from filtered results
filtered_results = list(results)[:10]
```

**Over-Fetch Guidelines:**
```
Filter Selectivity      | Over-Fetch Multiplier | Example
------------------------|----------------------|---------------------------
High (>50% match)       | 1.5-2×               | top=15 to get 10 results
Medium (20-50% match)   | 3-5×                 | top=30 to get 10 results
Low (<20% match)        | 5-10×                | top=50 to get 10 results
Very Low (<5% match)    | 10-20×               | top=100 to get 10 results
```

### HNSW Parameter Tuning

**✅ DO: Monitor Recall Metrics to Tune HNSW Parameters**
- **Why**: Default parameters may not be optimal for your data
- **Recall metric**: % of true nearest neighbors found by HNSW
- **Trade-off**: Higher recall = slower queries, larger index

```python
class HNSWRecallTester:
    """Test HNSW recall vs ground truth."""
    
    def __init__(self, search_client, embedding_generator):
        self.search_client = search_client
        self.embedding_gen = embedding_generator
    
    def compute_ground_truth(self, query_embedding, all_docs, k=10):
        """Brute-force nearest neighbors (100% recall, slow)."""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Compute similarity to all documents
        all_embeddings = np.array([doc['embedding'] for doc in all_docs])
        similarities = cosine_similarity([query_embedding], all_embeddings)[0]
        
        # Get top k
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return set(all_docs[i]['id'] for i in top_k_indices)
    
    def compute_hnsw_recall(self, query_embedding, ground_truth_ids, k=10):
        """Recall of HNSW vs ground truth."""
        # HNSW approximate search
        results = self.search_client.search(
            search_text=None,
            vector=query_embedding,
            vector_fields=["content_vector"],
            top=k
        )
        
        hnsw_ids = set(r['id'] for r in results)
        
        # Recall = |HNSW ∩ GroundTruth| / |GroundTruth|
        recall = len(hnsw_ids & ground_truth_ids) / len(ground_truth_ids)
        return recall
    
    def evaluate_recall(self, test_queries, all_docs, k=10):
        """Evaluate recall across multiple queries."""
        recalls = []
        
        for query_text in test_queries:
            query_emb = self.embedding_gen(query_text)
            
            # Ground truth (brute force)
            gt_ids = self.compute_ground_truth(query_emb, all_docs, k)
            
            # HNSW recall
            recall = self.compute_hnsw_recall(query_emb, gt_ids, k)
            recalls.append(recall)
        
        avg_recall = sum(recalls) / len(recalls)
        return {
            'average_recall': avg_recall,
            'min_recall': min(recalls),
            'max_recall': max(recalls),
            'recalls': recalls
        }

# Usage
tester = HNSWRecallTester(search_client, generate_embedding)

test_queries = [
    "laptop for machine learning",
    "wireless headphones",
    "smartphone with good camera"
]

# Test current HNSW parameters
recall_stats = tester.evaluate_recall(test_queries, all_documents, k=10)
print(f"Average recall: {recall_stats['average_recall']:.1%}")

# If recall < 95%, tune HNSW parameters:
# - Increase efConstruction (400 → 800)
# - Increase m (4 → 8)
# - Increase efSearch (500 → 1000)
```

**HNSW Parameter Tuning Guidelines:**
```
Recall Target | m  | efConstruction | efSearch | Index Time | Query Time
--------------|----|-----------------|-----------|-----------|-----------
90% (fast)    | 4  | 400            | 500       | 1×        | 1×
95% (balanced)| 4  | 400            | 500       | 1×        | 1×
98% (accurate)| 8  | 800            | 1000      | 3×        | 2×
99.5% (max)   | 16 | 1600           | 2000      | 10×       | 5×
```

**❌ DON'T: Ignore HNSW Parameter Tuning**
- **Problem**: Default parameters optimized for speed, not accuracy
- **Impact**: Missing 5-10% of truly relevant results
- **Solution**: Measure recall on representative test set, tune parameters

### Performance and Cost Optimization

**✅ DO: Normalize Vectors When Using Dot Product**
- **Why**: Dot product not rotation-invariant unless vectors normalized
- **When**: If using dot product distance metric (rare for text)
- **How**: Divide vector by its L2 norm

```python
import numpy as np

def normalize_vector(vector):
    """Normalize vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

# If using dot product metric
vector = np.array([0.5, 0.3, 0.8, ...])
normalized_vector = normalize_vector(vector)

# Verify normalization
assert np.allclose(np.linalg.norm(normalized_vector), 1.0)
```

**❌ DON'T: Use Very High k Values Unnecessarily**
- **Problem**: k=1000 computes 1000 similarity scores (expensive)
- **Impact**: 10× slower queries for large k
- **Solution**: Use k=10-50 for most use cases, over-fetch if filtering

```python
# BAD: Fetching too many results (500ms latency)
results = search_client.search(
    search_text=None,
    vector=query_embedding,
    vector_fields=["content_vector"],
    top=1000  # ❌ Rarely need 1000 results
)

# GOOD: Fetch reasonable top-k (80ms latency, 6× faster)
results = search_client.search(
    search_text=None,
    vector=query_embedding,
    vector_fields=["content_vector"],
    top=20  # ✅ Sufficient for most use cases
)
```

**k Value Guidelines:**
```
Use Case                  | Recommended k | Reasoning
--------------------------|---------------|-----------------------------------
User-facing search        | 10-20         | Users rarely scroll past 2 pages
Recommendation engine     | 50-100        | Need diverse recommendations
Re-ranking pipeline       | 100-500       | First-stage retrieval
Evaluation/analytics      | 100-1000      | Comprehensive analysis
```

**❌ DON'T: Use Vector Search for Exact Keyword Matching**
- **Problem**: Vector search approximates, BM25 is exact for keywords
- **Impact**: 3-5× more expensive than full-text for keyword queries
- **Solution**: Use full-text (BM25) for keyword queries, vector for semantic

```python
# BAD: Vector search for exact keyword match
query = "Dell XPS 15"  # Brand + model (exact match needed)
results = search_client.search(
    search_text=None,
    vector=generate_embedding(query),  # ❌ Overkill for exact match
    vector_fields=["content_vector"],
    top=10
)

# GOOD: Full-text search for exact keywords
results = search_client.search(
    search_text="Dell XPS 15",  # ✅ BM25 for exact match
    search_fields=["title", "description"],
    top=10
)

# BEST: Hybrid search (combines both)
results = search_client.search(
    search_text="Dell XPS 15",
    vector=generate_embedding("Dell XPS 15"),
    vector_fields=["content_vector"],
    search_fields=["title", "description"],
    top=10
)
```

**✅ DO: Handle Embedding Generation Errors Gracefully**
- **Why**: OpenAI API can fail (rate limits, timeouts, network issues)
- **Impact**: Without error handling, entire search fails
- **Solution**: Implement retry logic, fallback to full-text search

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import openai

class RobustEmbeddingGenerator:
    """Generate embeddings with error handling."""
    
    def __init__(self, api_key, model="text-embedding-3-large"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def generate_embedding(self, text):
        """Generate embedding with automatic retry."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except openai.RateLimitError as e:
            print(f"Rate limit hit: {e}, retrying...")
            raise  # Retry via @retry decorator
        except openai.APIError as e:
            print(f"API error: {e}, retrying...")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None  # Return None, caller handles fallback
    
    def search_with_fallback(self, search_client, query_text):
        """Try vector search, fallback to full-text on failure."""
        # Try to generate embedding
        embedding = self.generate_embedding(query_text)
        
        if embedding is not None:
            # Vector search succeeded
            try:
                return search_client.search(
                    search_text=None,
                    vector=embedding,
                    vector_fields=["content_vector"],
                    top=10
                )
            except Exception as e:
                print(f"Vector search failed: {e}, falling back to full-text")
        
        # Fallback to full-text search
        return search_client.search(
            search_text=query_text,
            search_fields=["title", "content"],
            top=10
        )

# Usage
generator = RobustEmbeddingGenerator(api_key=os.getenv("OPENAI_API_KEY"))
results = generator.search_with_fallback(search_client, "laptop for ML")
```

### Summary: Vector Search Best Practices Checklist

**Before Indexing:**
- [ ] Choose consistent embedding model (pin version)
- [ ] Design separate vector fields for title and content
- [ ] Include filterable/sortable metadata fields
- [ ] Configure HNSW parameters (start with defaults)
- [ ] Plan for embedding generation costs

**During Querying:**
- [ ] Cache query embeddings for repeated queries
- [ ] Use cosine similarity for text embeddings
- [ ] Over-fetch when applying filters (3-5×)
- [ ] Implement retry logic for embedding generation
- [ ] Monitor embedding API rate limits

**Ongoing Optimization:**
- [ ] Measure recall metrics on test set
- [ ] Tune HNSW parameters based on recall
- [ ] Monitor query latency (P50, P95, P99)
- [ ] Track embedding costs monthly
- [ ] Consider hybrid search for best results

---

## Troubleshooting

Vector search introduces new failure modes and performance considerations compared to traditional full-text search. This section covers common issues and their solutions.

### Issue 1: Low Recall (Missing Relevant Results)

**Symptoms:**
- Query "laptop for ML" doesn't return obvious ML workstation products
- Recall metrics show <90% vs ground truth
- Users report missing expected results

**Root Causes & Solutions:**

**Solution 1: Increase HNSW efSearch Parameter**
```python
# Current configuration (low recall)
"hnsw_parameters": {
    "m": 4,
    "efConstruction": 400,
    "efSearch": 500  # Too low for high recall
}

# Improved configuration (higher recall)
"hnsw_parameters": {
    "m": 4,
    "efConstruction": 400,
    "efSearch": 1000  # ✅ 2× higher, improves recall 95% → 98%
}
# Trade-off: ~30% slower queries
```

**Solution 2: Verify Embedding Model Consistency**
```python
# Check if query and document embeddings use same model
doc_metadata = index_client.get_document(key="doc123")
doc_model_version = doc_metadata.get("embedding_model_version")

if doc_model_version != current_model_version:
    print(f"WARNING: Model mismatch!")
    print(f"Documents: {doc_model_version}")
    print(f"Query: {current_model_version}")
    # Solution: Re-index documents with current model
```

**Solution 3: Check Vector Dimensions Match**
```kusto
# Verify all documents have correct dimension count
# In Azure Portal → Search Index → Documents
# Check content_vector field length

# If dimensions mismatch:
# - Expected: 3072 (text-embedding-3-large)
# - Actual: 1536 (text-embedding-ada-002)
# Solution: Re-generate embeddings with correct model
```

### Issue 2: Slow Query Performance

**Symptoms:**
- Vector queries take 500-1000ms (vs <100ms expected)
- Timeout errors on complex queries
- High P95 latency

**Root Causes & Solutions:**

**Solution 1: Reduce k (Top Results Count)**
```python
# BAD: Fetching too many results
results = search_client.search(
    vector=query_embedding,
    vector_fields=["content_vector"],
    top=500  # ❌ Computing 500 similarities is expensive
)

# GOOD: Fetch only what you need
results = search_client.search(
    vector=query_embedding,
    vector_fields=["content_vector"],
    top=20  # ✅ 10-20× faster
)
```

**Solution 2: Reduce Vector Dimensions (If Applicable)**
```python
# text-embedding-3-large supports dimension reduction
response = client.embeddings.create(
    input="laptop for ML",
    model="text-embedding-3-large",
    dimensions=1536  # ✅ Reduce from 3072 to 1536
)
# Impact: 50% smaller index, 30-40% faster queries
# Trade-off: 2-5% lower recall (test on your data)
```

**Solution 3: Cache Frequent Query Embeddings**
```python
# Implement caching (see Best Practices section)
# Impact: 50-80% latency reduction for repeated queries
# Cache hit rate: 30-60% typical in production
```

### Issue 3: High Embedding Costs

**Symptoms:**
- Embedding API costs exceed search service costs
- Monthly bill 10× higher than expected
- Cost per query unsustainable

**Root Causes & Solutions:**

**Solution 1: Implement Query Embedding Caching**
```python
# Without caching: 100,000 queries × $0.00013 per 1K tokens = $13/month
# With caching (50% hit rate): 50,000 queries × $0.00013 = $6.50/month
# Savings: 50% cost reduction
```

**Solution 2: Use Smaller Embedding Model**
```python
# text-embedding-3-large: $0.13 per 1M tokens
# text-embedding-3-small: $0.02 per 1M tokens (6.5× cheaper!)

# For 1M queries × 10 tokens each:
# 3-large cost: 10M tokens × $0.13/1M = $1.30
# 3-small cost: 10M tokens × $0.02/1M = $0.20
# Savings: $1.10/month (85% reduction)

# Trade-off: 3-5% lower relevance (test on your data)
```

**Solution 3: Batch Embedding Generation**
```python
# OpenAI allows up to 2048 inputs per request
# Reduces API overhead, improves throughput

def generate_embeddings_batch(texts, batch_size=100):
    """Generate embeddings in batches."""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-large"
        )
        embeddings.extend([d.embedding for d in response.data])
    
    return embeddings

# Batch 1000 documents
embeddings = generate_embeddings_batch(document_texts, batch_size=100)
# Impact: 5-10× faster than individual requests
```

### Issue 4: Irrelevant Results in Top 10

**Symptoms:**
- Query "laptop" returns smartphones in top 5
- Semantically unrelated documents rank high
- User satisfaction scores low

**Root Causes & Solutions:**

**Solution 1: Use Hybrid Search (Vector + BM25)**
```python
# Vector-only search can miss keyword signals
results_vector_only = search_client.search(
    search_text=None,
    vector=query_embedding,
    vector_fields=["content_vector"],
    top=10
)

# Hybrid search combines semantic + keyword matching
results_hybrid = search_client.search(
    search_text="laptop",  # ✅ BM25 keyword signal
    vector=query_embedding,  # ✅ Semantic signal
    vector_fields=["content_vector"],
    search_fields=["title", "content"],
    top=10
)
# Impact: 15-30% relevance improvement
```

**Solution 2: Apply Filters to Narrow Semantic Search**
```python
# Vector search within category
results = search_client.search(
    search_text=None,
    vector=query_embedding,
    vector_fields=["content_vector"],
    filter="category eq 'Laptops'",  # ✅ Constrain to relevant category
    top=10
)
```

**Solution 3: Tune Embedding Model for Domain**
```python
# If using general-purpose embeddings (OpenAI),
# consider fine-tuning for your domain

# Option 1: Use domain-specific model (if available)
# Option 2: Fine-tune text-embedding-3-large on your data
# Option 3: Add domain-specific metadata to vector index
```

---

## Next Steps

- **[Hybrid Search](./10-hybrid-search.md)** - Combine vector and full-text
- **[Azure OpenAI Integration](./05-azure-openai-integration.md)** - Embedding generation
- **[Query Optimization](./12-query-optimization.md)** - Performance tuning

---

*See also: [Full-Text Search](./08-fulltext-search-bm25.md) | [Semantic Search](./11-semantic-search.md)*