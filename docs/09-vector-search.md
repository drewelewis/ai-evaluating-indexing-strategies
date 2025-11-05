# Vector Search

Complete guide to implementing semantic vector search using embeddings and approximate nearest neighbor (ANN) algorithms in Azure AI Search.

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

---

## Overview

### What is Vector Search?

Vector search enables semantic search by representing text as high-dimensional vectors (embeddings) and finding similar vectors using mathematical distance calculations.

**Key Benefits:**
- **Semantic understanding**: Matches meaning, not just keywords
- **Cross-lingual search**: Works across languages
- **Handles synonyms**: Finds conceptually similar content
- **Query reformulation**: Better handles poorly worded queries

### Vector Search Architecture

```
┌─────────────────────┐
│   User Query        │
│   "laptop for ML"   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────┐
│  Azure OpenAI           │
│  Generate Embedding     │
│  [0.12, -0.45, ...]    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Azure AI Search        │
│  HNSW Vector Index      │
│  Find Nearest Neighbors │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────┐
│  Ranked Results     │
│  (by similarity)    │
└─────────────────────┘
```

---

## Vector Search Fundamentals

### Understanding Embeddings

```python
class EmbeddingConcepts:
    """Core concepts of vector embeddings."""
    
    @staticmethod
    def what_are_embeddings():
        """Explain embeddings."""
        return {
            'definition': 'Dense numerical representations of text in high-dimensional space',
            'dimensions': '1536 for text-embedding-ada-002',
            'properties': [
                'Semantically similar texts have similar vectors',
                'Distance between vectors indicates similarity',
                'Position in vector space captures meaning'
            ],
            'example': {
                'text': 'laptop for machine learning',
                'embedding': '[0.12, -0.45, 0.78, ..., 0.23]',  # 1536 dimensions
                'similar_texts': [
                    'computer for AI development',
                    'ML workstation',
                    'deep learning machine'
                ]
            }
        }
    
    @staticmethod
    def similarity_calculation():
        """How similarity is computed."""
        return {
            'cosine_similarity': {
                'formula': 'cos(θ) = (A·B) / (||A|| × ||B||)',
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

### ✅ Do's
1. **Use cosine similarity** for text embeddings (default)
2. **Generate embeddings consistently** (same model, version)
3. **Index separate vector fields** for title and content
4. **Cache query embeddings** in your application
5. **Over-fetch when filtering** (k=50 to get top 10 after filter)
6. **Monitor recall metrics** to tune HNSW parameters
7. **Normalize vectors** when using dot product

### ❌ Don'ts
1. **Don't** mix embeddings from different models
2. **Don't** use very high k values unnecessarily
3. **Don't** forget to handle embedding generation errors
4. **Don't** index embeddings without metadata fields
5. **Don't** ignore HNSW parameter tuning
6. **Don't** use vector search for exact keyword matching

---

## Next Steps

- **[Hybrid Search](./10-hybrid-search.md)** - Combine vector and full-text
- **[Azure OpenAI Integration](./05-azure-openai-integration.md)** - Embedding generation
- **[Query Optimization](./12-query-optimization.md)** - Performance tuning

---

*See also: [Full-Text Search](./08-fulltext-search-bm25.md) | [Semantic Search](./11-semantic-search.md)*