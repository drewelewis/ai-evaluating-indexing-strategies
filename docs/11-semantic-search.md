# Semantic Search

Complete guide to Azure AI Search semantic ranking with AI-powered re-ranking, semantic captions, and spell correction.

## 📋 Table of Contents
- [Overview](#overview)
- [Semantic Ranking Fundamentals](#semantic-ranking-fundamentals)
- [Configuration](#configuration)
- [Implementation](#implementation)
- [Semantic Features](#semantic-features)
- [Optimization](#optimization)
- [Comparison](#comparison)
- [Best Practices](#best-practices)

---

## Overview

### What is Semantic Search?

Azure AI Search semantic ranking is a **Microsoft-provided AI capability** that:
- Re-ranks search results using deep learning models
- Generates semantic captions highlighting relevance
- Extracts semantic answers from documents
- Applies query spell correction
- Works on top of existing BM25 or hybrid results

### Semantic vs Vector Search

```python
class SemanticVsVector:
    """Understanding the difference between semantic ranking and vector search."""
    
    @staticmethod
    def comparison():
        """Compare semantic ranking and vector search."""
        return {
            'vector_search': {
                'description': 'Embedding-based similarity search',
                'requires': 'Pre-generated embeddings in index',
                'model': 'User-provided (e.g., text-embedding-ada-002)',
                'search_stage': 'Primary retrieval',
                'cost': 'OpenAI embedding costs + storage',
                'latency': '20-100ms depending on HNSW params',
                'use_case': 'Find semantically similar documents'
            },
            'semantic_ranking': {
                'description': 'AI-powered re-ranking of existing results',
                'requires': 'Text fields only (no embeddings needed)',
                'model': 'Microsoft-provided (Bing models)',
                'search_stage': 'L2 re-ranking after initial retrieval',
                'cost': 'Azure Search semantic pricing ($500/month Standard)',
                'latency': '50-200ms additional overhead',
                'use_case': 'Improve relevance of keyword/hybrid results'
            },
            'key_differences': [
                'Vector = retrieval method, Semantic = re-ranking method',
                'Vector requires embeddings, Semantic uses raw text',
                'Vector is primary search, Semantic is secondary refinement',
                'Can use BOTH together: Hybrid + Semantic'
            ]
        }
    
    @staticmethod
    def when_to_use():
        """Decision guide for semantic ranking vs vector search."""
        return {
            'use_semantic_ranking': [
                'Already have full-text index (no embeddings)',
                'Want quick relevance improvement without re-indexing',
                'Need semantic captions/answers features',
                'Budget for semantic pricing tier',
                'Queries are natural language questions'
            ],
            'use_vector_search': [
                'Need semantic similarity at retrieval stage',
                'Want to find similar items by meaning',
                'Have resources to generate/store embeddings',
                'Need multilingual semantic search',
                'Want more control over embedding model'
            ],
            'use_both': [
                'Best possible relevance (hybrid + semantic)',
                'Complex enterprise search requirements',
                'Production e-commerce or knowledge base'
            ]
        }

# Usage
comparison = SemanticVsVector()
diff = comparison.comparison()

print("Semantic Ranking:")
print(f"  {diff['semantic_ranking']['description']}")
print(f"  Search stage: {diff['semantic_ranking']['search_stage']}")

print("\nVector Search:")
print(f"  {diff['vector_search']['description']}")
print(f"  Search stage: {diff['vector_search']['search_stage']}")

print("\nKey Differences:")
for diff_point in diff['key_differences']:
    print(f"  • {diff_point}")
```

---

## Semantic Ranking Fundamentals

### How Semantic Ranking Works

```python
class SemanticRankingExplained:
    """Understanding the semantic ranking pipeline."""
    
    @staticmethod
    def ranking_pipeline():
        """
        Semantic ranking pipeline stages.
        
        L0: Initial retrieval (BM25/Vector/Hybrid)
         ↓
        L1: Fast semantic filtering (top 50)
         ↓
        L2: Deep semantic ranking (top 50)
         ↓
        Captions & Answers extraction
         ↓
        Final ranked results
        """
        return {
            'stage_L0_retrieval': {
                'description': 'Initial search using BM25, vector, or hybrid',
                'result_count': 'All matching documents',
                'algorithm': 'Standard search algorithm',
                'purpose': 'Retrieve candidate documents'
            },
            'stage_L1_filtering': {
                'description': 'Fast semantic filtering',
                'result_count': 'Top 50 documents',
                'algorithm': 'Lightweight semantic model',
                'purpose': 'Quick relevance filtering',
                'latency': '~10ms'
            },
            'stage_L2_reranking': {
                'description': 'Deep semantic re-ranking',
                'result_count': 'Top 50 documents',
                'algorithm': 'Transformer-based deep learning model',
                'purpose': 'Precise relevance scoring',
                'latency': '50-150ms'
            },
            'caption_generation': {
                'description': 'Extract relevant passages',
                'algorithm': 'Extractive summarization',
                'output': 'Semantic captions with <em> highlights',
                'max_captions': 5  # Default
            },
            'answer_extraction': {
                'description': 'Extract direct answers',
                'algorithm': 'Question-answering model',
                'output': 'Semantic answers to queries',
                'condition': 'Query must be question-like'
            }
        }
    
    @staticmethod
    def semantic_score_explained():
        """
        Semantic @search.rerankerScore explained.
        
        Score range: 0 to 4+
        - Higher scores = more relevant
        - Uses Bing semantic models
        - Considers query-document semantic similarity
        """
        return {
            'score_range': '0 to 4+',
            'interpretation': {
                '3.5+': 'Highly relevant',
                '2.5-3.5': 'Moderately relevant',
                '1.5-2.5': 'Somewhat relevant',
                '<1.5': 'Low relevance'
            },
            'note': 'Not directly comparable to BM25 scores'
        }

# Usage
explained = SemanticRankingExplained()
pipeline = explained.ranking_pipeline()

print("Semantic Ranking Pipeline:")
print(f"\nL0 - {pipeline['stage_L0_retrieval']['description']}")
print(f"L1 - {pipeline['stage_L1_filtering']['description']} ({pipeline['stage_L1_filtering']['latency']})")
print(f"L2 - {pipeline['stage_L2_reranking']['description']} ({pipeline['stage_L2_reranking']['latency']})")

scores = explained.semantic_score_explained()
print(f"\nSemantic Score Range: {scores['score_range']}")
print("Interpretation:")
for range_val, meaning in scores['interpretation'].items():
    print(f"  {range_val}: {meaning}")
```

---

## Configuration

### Semantic Configuration Setup

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch
)
from azure.core.credentials import AzureKeyCredential

class SemanticConfigurationSetup:
    """Configure semantic search in Azure AI Search index."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_semantic_index(self, index_name):
        """
        Create index with semantic configuration.
        
        Semantic configuration defines:
        - Title field(s): Most important for relevance
        - Content field(s): Main body text
        - Keywords field(s): Additional context
        """
        # Define fields
        fields = [
            SimpleField(name="id", type="Edm.String", key=True),
            SearchableField(name="title", type="Edm.String"),
            SearchableField(name="content", type="Edm.String"),
            SearchableField(name="category", type="Edm.String", filterable=True),
            SearchableField(name="tags", type="Collection(Edm.String)"),
            SimpleField(name="price", type="Edm.Double", filterable=True, sortable=True),
            SimpleField(name="rating", type="Edm.Double", filterable=True, sortable=True)
        ]
        
        # Define semantic configuration
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[
                    SemanticField(field_name="content")
                ],
                keywords_fields=[
                    SemanticField(field_name="category"),
                    SemanticField(field_name="tags")
                ]
            )
        )
        
        # Create semantic search settings
        semantic_search = SemanticSearch(
            configurations=[semantic_config]
        )
        
        # Create index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            semantic_search=semantic_search
        )
        
        result = self.index_client.create_or_update_index(index)
        print(f"Created index '{result.name}' with semantic configuration")
        return result
    
    def create_multi_semantic_config_index(self, index_name):
        """
        Create index with multiple semantic configurations.
        
        Use different configs for different query types.
        """
        fields = [
            SimpleField(name="id", type="Edm.String", key=True),
            SearchableField(name="productName", type="Edm.String"),
            SearchableField(name="description", type="Edm.String"),
            SearchableField(name="detailedSpecs", type="Edm.String"),
            SearchableField(name="brand", type="Edm.String"),
            SearchableField(name="category", type="Edm.String")
        ]
        
        # Config 1: Product-focused (emphasize name and brand)
        product_config = SemanticConfiguration(
            name="product-semantic",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="productName"),
                content_fields=[
                    SemanticField(field_name="description")
                ],
                keywords_fields=[
                    SemanticField(field_name="brand"),
                    SemanticField(field_name="category")
                ]
            )
        )
        
        # Config 2: Spec-focused (emphasize technical details)
        spec_config = SemanticConfiguration(
            name="spec-semantic",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="productName"),
                content_fields=[
                    SemanticField(field_name="detailedSpecs"),
                    SemanticField(field_name="description")
                ],
                keywords_fields=[
                    SemanticField(field_name="category")
                ]
            )
        )
        
        semantic_search = SemanticSearch(
            configurations=[product_config, spec_config]
        )
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            semantic_search=semantic_search
        )
        
        result = self.index_client.create_or_update_index(index)
        return result

# Usage
import os

semantic_setup = SemanticConfigurationSetup(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Create index with semantic config
index = semantic_setup.create_semantic_index("products-semantic")
print(f"Semantic configuration: {index.semantic_search.configurations[0].name}")

# Multiple configs
multi_index = semantic_setup.create_multi_semantic_config_index("products-multi-semantic")
print(f"Configurations: {[c.name for c in multi_index.semantic_search.configurations]}")
```

---

## Implementation

### Basic Semantic Search

```python
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType

class SemanticSearcher:
    """Execute semantic searches."""
    
    def __init__(self, search_endpoint, index_name, search_key):
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )
    
    def semantic_search(self, query_text, top=10):
        """
        Execute semantic search with re-ranking.
        
        Args:
            query_text: Natural language query
            top: Number of results
            
        Returns:
            Results with semantic scores and captions
        """
        results = self.search_client.search(
            search_text=query_text,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="my-semantic-config",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            top=top
        )
        
        # Process results
        documents = []
        for result in results:
            doc = {
                'id': result['id'],
                'title': result.get('title'),
                'content': result.get('content'),
                'score': result.get('@search.score'),  # BM25 score
                'reranker_score': result.get('@search.rerankerScore'),  # Semantic score
                'captions': self._extract_captions(result),
                'highlights': result.get('@search.highlights')
            }
            documents.append(doc)
        
        return documents
    
    def semantic_search_with_answers(self, query_text, top=10):
        """
        Semantic search with answer extraction.
        
        Best for question-like queries.
        """
        results = self.search_client.search(
            search_text=query_text,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="my-semantic-config",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            query_answer_count=3,  # Up to 3 answers
            top=top
        )
        
        # Extract answers
        answers = []
        for result in results:
            if '@search.answers' in result:
                for answer in result['@search.answers']:
                    answers.append({
                        'text': answer.get('text'),
                        'highlights': answer.get('highlights'),
                        'score': answer.get('score')
                    })
        
        # Extract documents
        documents = []
        for result in results:
            documents.append({
                'id': result['id'],
                'title': result.get('title'),
                'reranker_score': result.get('@search.rerankerScore'),
                'captions': self._extract_captions(result)
            })
        
        return {
            'answers': answers,
            'documents': documents
        }
    
    def hybrid_semantic_search(self, query_text, openai_client, top=10):
        """
        Combine hybrid (BM25 + vector) with semantic ranking.
        
        This is the most powerful configuration:
        - Initial retrieval: Hybrid (BM25 + vector)
        - Re-ranking: Semantic (L2)
        - Features: Captions + answers
        """
        # Generate query embedding
        embedding_response = openai_client.embeddings.create(
            input=query_text,
            model="text-embedding-ada-002"
        )
        query_vector = embedding_response.data[0].embedding
        
        # Vector query
        from azure.search.documents.models import VectorizedQuery
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,
            fields="contentVector"
        )
        
        # Hybrid + semantic search
        results = self.search_client.search(
            search_text=query_text,  # BM25
            vector_queries=[vector_query],  # Vector
            query_type=QueryType.SEMANTIC,  # Semantic re-ranking
            semantic_configuration_name="my-semantic-config",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            top=top
        )
        
        return list(results)
    
    def _extract_captions(self, result):
        """Extract semantic captions from result."""
        captions = []
        if '@search.captions' in result:
            for caption in result['@search.captions']:
                captions.append({
                    'text': caption.get('text'),
                    'highlights': caption.get('highlights')
                })
        return captions

# Usage
semantic_searcher = SemanticSearcher(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    index_name="products-semantic",
    search_key=os.getenv("SEARCH_API_KEY")
)

# Basic semantic search
results = semantic_searcher.semantic_search(
    "What is the best laptop for machine learning?",
    top=10
)

print("Semantic Search Results:")
for doc in results[:3]:
    print(f"\n{doc['title']}")
    print(f"  Reranker Score: {doc['reranker_score']:.4f}")
    if doc['captions']:
        print(f"  Caption: {doc['captions'][0]['highlights']}")

# With answers
answer_results = semantic_searcher.semantic_search_with_answers(
    "How much does the Dell XPS cost?",
    top=5
)

if answer_results['answers']:
    print("\nAnswers:")
    for answer in answer_results['answers']:
        print(f"  {answer['highlights']} (score: {answer['score']:.4f})")

# Hybrid + semantic (best)
from openai import AzureOpenAI
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

hybrid_semantic_results = semantic_searcher.hybrid_semantic_search(
    "laptop for deep learning",
    openai_client,
    top=10
)
```

---

## Semantic Features

### Semantic Captions

```python
class SemanticCaptions:
    """Working with semantic captions."""
    
    @staticmethod
    def caption_explained():
        """
        Semantic captions are AI-generated excerpts highlighting
        the most relevant parts of documents.
        
        Features:
        - Extractive (from document text)
        - Highlighted with <em> tags
        - Multiple captions per document (up to 5)
        - Automatically identifies relevant passages
        """
        return {
            'caption_type': 'Extractive',
            'max_captions_per_doc': 5,
            'highlight_format': '<em>highlighted text</em>',
            'use_case': 'Show users why document is relevant',
            'advantage': 'Better than simple field highlights'
        }
    
    @staticmethod
    def display_captions(results):
        """Display captions from search results."""
        for result in results:
            print(f"\nDocument: {result.get('title')}")
            print(f"Reranker Score: {result.get('@search.rerankerScore', 0):.4f}")
            
            if '@search.captions' in result:
                print("Captions:")
                for caption in result['@search.captions']:
                    # highlights has <em> tags
                    highlighted = caption.get('highlights', caption.get('text'))
                    print(f"  • {highlighted}")

# Usage
captions = SemanticCaptions()
info = captions.caption_explained()
print(f"Caption type: {info['caption_type']}")
print(f"Max per doc: {info['max_captions_per_doc']}")
```

### Semantic Answers

```python
class SemanticAnswers:
    """Working with semantic answers."""
    
    @staticmethod
    def answer_explained():
        """
        Semantic answers extract direct answers to questions.
        
        Features:
        - Question-answering model
        - Returns answer text + source document
        - Score indicates confidence
        - Best for factual questions
        
        Requirements:
        - Query should be question-like
        - Documents must contain factual information
        """
        return {
            'model': 'Question-answering transformer',
            'answer_count': 'Up to 3 answers (configurable)',
            'score_range': '0 to 1 (confidence)',
            'best_for': [
                'What questions',
                'How questions',
                'When/Where questions',
                'Factual queries'
            ],
            'not_for': [
                'Exploratory searches',
                'Product browsing',
                'Broad topics'
            ]
        }
    
    @staticmethod
    def extract_answers(search_results):
        """Extract and rank answers from results."""
        all_answers = []
        
        for result in search_results:
            if '@search.answers' in result:
                for answer in result['@search.answers']:
                    all_answers.append({
                        'text': answer.get('text'),
                        'highlights': answer.get('highlights'),
                        'score': answer.get('score'),
                        'source_doc': result.get('id')
                    })
        
        # Sort by score
        all_answers.sort(key=lambda x: x['score'], reverse=True)
        return all_answers

# Usage
answers_info = SemanticAnswers()
explained = answers_info.answer_explained()
print(f"Answer model: {explained['model']}")
print(f"Best for: {explained['best_for']}")
```

### Spell Correction

```python
class SemanticSpellCorrection:
    """Semantic search includes automatic spell correction."""
    
    @staticmethod
    def spell_correction_explained():
        """
        Semantic search automatically corrects spelling mistakes.
        
        Features:
        - Automatic query correction
        - Transparent to application
        - Uses Bing spell check models
        - No configuration needed
        
        Examples:
        - "machne lerning" → "machine learning"
        - "lpatop" → "laptop"
        - "dlel xps" → "dell xps"
        """
        return {
            'automatic': True,
            'configuration': 'None needed',
            'model': 'Bing spell check',
            'transparency': 'Corrected query used for search',
            'original_query': 'Not modified in response'
        }
    
    @staticmethod
    def test_spell_correction(search_client, misspelled_query):
        """
        Test spell correction with misspelled query.
        
        Compare results with and without semantic.
        """
        # Without semantic (no spell correction)
        standard_results = list(search_client.search(
            search_text=misspelled_query,
            top=5
        ))
        
        # With semantic (spell correction applied)
        semantic_results = list(search_client.search(
            search_text=misspelled_query,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="my-semantic-config",
            top=5
        ))
        
        return {
            'query': misspelled_query,
            'standard_count': len(standard_results),
            'semantic_count': len(semantic_results),
            'improvement': len(semantic_results) - len(standard_results)
        }

# Usage
spell_info = SemanticSpellCorrection()
explained = spell_info.spell_correction_explained()
print(f"Spell correction: {explained['automatic']}")
print(f"Model: {explained['model']}")
```

---

## Optimization

### Performance Optimization

```python
class SemanticSearchOptimization:
    """Optimize semantic search performance and cost."""
    
    @staticmethod
    def optimization_strategies():
        """Key optimization strategies."""
        return {
            'top_parameter': {
                'description': 'Limit results to reduce L2 re-ranking',
                'recommendation': 'Use top=10-50 for best performance',
                'note': 'Semantic re-ranks top 50 regardless of top value'
            },
            'semantic_configuration': {
                'description': 'Optimize field selection',
                'recommendation': 'Include only essential fields',
                'impact': 'Faster processing, better relevance'
            },
            'query_type_selection': {
                'description': 'Use semantic only when beneficial',
                'recommendation': 'Route queries: semantic for NL questions, standard for SKUs',
                'savings': 'Reduced semantic query costs'
            },
            'caching': {
                'description': 'Cache semantic results',
                'recommendation': 'Cache common queries at application level',
                'savings': 'Reduced latency and costs'
            }
        }
    
    @staticmethod
    def estimate_semantic_latency(base_search_ms, l1_ms=10, l2_ms=100):
        """
        Estimate semantic search latency.
        
        Args:
            base_search_ms: Base search latency (BM25/hybrid)
            l1_ms: L1 filtering latency
            l2_ms: L2 re-ranking latency
            
        Returns:
            Total estimated latency
        """
        total_latency = base_search_ms + l1_ms + l2_ms
        
        return {
            'base_search_ms': base_search_ms,
            'l1_filtering_ms': l1_ms,
            'l2_reranking_ms': l2_ms,
            'total_latency_ms': total_latency,
            'overhead_ms': l1_ms + l2_ms
        }
    
    @staticmethod
    def cost_analysis():
        """
        Semantic search pricing analysis.
        
        Pricing (as of 2024):
        - Standard tier: ~$500/month (1000 queries/hour)
        - Includes semantic re-ranking
        - No per-query charges
        """
        return {
            'pricing_tier': 'Standard or higher',
            'monthly_cost': '$500 (approximate)',
            'included_queries': '1000 queries/hour',
            'overage': 'Throttled if exceeded',
            'cost_per_query': '$0 (fixed monthly cost)',
            'recommendation': 'Use for high-value queries only'
        }

# Usage
opt = SemanticSearchOptimization()

strategies = opt.optimization_strategies()
print("Optimization strategies:")
for key, strategy in strategies.items():
    print(f"\n{key}: {strategy['recommendation']}")

# Latency estimation
latency = opt.estimate_semantic_latency(
    base_search_ms=30,
    l1_ms=10,
    l2_ms=100
)
print(f"\nEstimated semantic latency: {latency['total_latency_ms']}ms")
print(f"Semantic overhead: {latency['overhead_ms']}ms")

# Cost analysis
cost = opt.cost_analysis()
print(f"\nSemantic pricing: {cost['monthly_cost']}/month")
print(f"Included: {cost['included_queries']} queries/hour")
```

---

## Comparison

### Semantic vs Other Search Methods

```python
class SemanticComparison:
    """Compare semantic search with other methods."""
    
    @staticmethod
    def comparison_matrix():
        """Detailed comparison of search methods."""
        return {
            'bm25_fulltext': {
                'relevance': 'Good for exact matches',
                'latency': 'Fast (~10-30ms)',
                'cost': 'Low',
                'setup': 'Simple',
                'use_case': 'Keyword search, SKUs, exact match'
            },
            'vector_search': {
                'relevance': 'Good for semantic similarity',
                'latency': 'Medium (~20-100ms)',
                'cost': 'Medium (embeddings + storage)',
                'setup': 'Complex (embeddings required)',
                'use_case': 'Similarity search, recommendations'
            },
            'hybrid_search': {
                'relevance': 'Better (combines BM25 + vector)',
                'latency': 'Medium (~30-100ms)',
                'cost': 'Medium',
                'setup': 'Complex (embeddings required)',
                'use_case': 'General purpose, balanced'
            },
            'semantic_ranking': {
                'relevance': 'Best for natural language',
                'latency': 'Higher (~50-200ms)',
                'cost': 'High ($500/month)',
                'setup': 'Simple (no embeddings)',
                'use_case': 'Question answering, natural language queries'
            },
            'hybrid_plus_semantic': {
                'relevance': 'Best overall',
                'latency': 'Highest (~80-250ms)',
                'cost': 'Highest',
                'setup': 'Complex',
                'use_case': 'Premium search experiences, enterprise'
            }
        }
    
    @staticmethod
    def recommendation_guide():
        """Guide for choosing search method."""
        return {
            'start_with': 'BM25 full-text',
            'add_vector': 'If need semantic similarity',
            'add_semantic': 'If queries are natural language questions',
            'ultimate': 'Hybrid + Semantic for best results',
            'budget_option': 'BM25 or Hybrid (no semantic)',
            'premium_option': 'Hybrid + Semantic'
        }

# Usage
comparison = SemanticComparison()
matrix = comparison.comparison_matrix()

print("Search Method Comparison:")
for method, details in matrix.items():
    print(f"\n{method}:")
    print(f"  Relevance: {details['relevance']}")
    print(f"  Latency: {details['latency']}")
    print(f"  Cost: {details['cost']}")
    print(f"  Use case: {details['use_case']}")
```

---

## Best Practices

### ✅ Do's
1. **Use semantic for natural language queries** (questions, conversations)
2. **Configure title/content/keywords fields** appropriately
3. **Enable captions and answers** for better UX
4. **Combine with hybrid search** for best results
5. **Monitor semantic query volume** for cost control
6. **Test with real user queries** to validate improvement
7. **Use multiple semantic configs** for different query types

### ❌ Don'ts
1. **Don't** use semantic for simple keyword queries (SKUs, IDs)
2. **Don't** include irrelevant fields in semantic config
3. **Don't** ignore the cost (~$500/month minimum)
4. **Don't** use semantic if latency is critical (<50ms requirement)
5. **Don't** assume semantic always improves results (test!)
6. **Don't** forget to handle the additional latency in UX
7. **Don't** use semantic without proper tier (Standard+)

---

## Next Steps

- **[Query Optimization](./12-query-optimization.md)** - Performance tuning
- **[A/B Testing](./17-ab-testing-framework.md)** - Compare semantic vs non-semantic
- **[Cost Analysis](./19-cost-analysis.md)** - Semantic pricing details

---

*See also: [Hybrid Search](./10-hybrid-search.md) | [Vector Search](./09-vector-search.md)*