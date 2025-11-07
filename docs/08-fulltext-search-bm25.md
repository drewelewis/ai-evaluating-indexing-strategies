# Full-Text Search (BM25)

Complete guide to implementing and optimizing keyword-based full-text search using BM25 (Best Match 25) algorithm in Azure AI Search. This document provides comprehensive coverage of BM25 theory, practical implementation patterns, performance optimization strategies, and real-world troubleshooting scenarios.

## 📋 Table of Contents
- [Overview](#overview)
- [BM25 Algorithm](#bm25-algorithm)
- [Index Configuration](#index-configuration)
- [Query Implementation](#query-implementation)
- [Search Modes](#search-modes)
- [Analyzers](#analyzers)
- [Scoring Profiles](#scoring-profiles)
- [Optimization Techniques](#optimization-techniques)
- [Performance Tuning](#performance-tuning)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Full-text search remains the foundational search paradigm for most applications, powering everything from e-commerce product catalogs to enterprise document repositories. While modern search techniques like vector search and semantic ranking capture headlines, the majority of search queries—often 80-90% in production systems—are handled efficiently and effectively by keyword-based full-text search using the BM25 algorithm.

### What is BM25?

BM25 (Best Match 25) is a probabilistic ranking function used by search engines to estimate the relevance of documents to a search query. Developed in the 1970s-1990s by Stephen Robertson, Karen Spärck Jones, and others at the City University of London, BM25 has become the de facto standard for full-text search across modern search engines including Elasticsearch, Solr, and Azure AI Search.

**Why BM25 Matters in 2024:**
- **Proven effectiveness**: 40+ years of academic research and production validation
- **Computational efficiency**: Sub-millisecond query latency even on millions of documents
- **Interpretability**: Transparent scoring based on term frequency and document statistics
- **Broad applicability**: Effective for 80-90% of keyword-based search queries without ML/AI overhead

**Key Characteristics:**
- **Term frequency (TF)**: How often a term appears in a document
  - Example: "laptop" appears 8 times in document A vs 2 times in document B
  - Intuition: Documents with more term occurrences are more likely to be relevant
  
- **Inverse document frequency (IDF)**: Rarity of the term across all documents
  - Example: "laptop" appears in 50% of products vs "ultrabook" in 5%
  - Intuition: Rare terms are more discriminative and valuable for ranking
  
- **Document length normalization**: Prevents bias toward longer documents
  - Example: 8 occurrences in 100-word doc vs 8 occurrences in 10,000-word doc
  - Intuition: Term density matters more than absolute count
  
- **Configurable parameters**: k1 and b for tuning (though fixed in Azure AI Search)
  - k1 controls term frequency saturation (default 1.2)
  - b controls document length normalization (default 0.75)

### BM25 vs TF-IDF: Understanding the Difference

While BM25 and TF-IDF are often mentioned together, BM25 represents a significant evolution in ranking effectiveness:

```
┌──────────────────────────┬──────────────────────┬──────────────────────┐
│    Feature               │    BM25              │   TF-IDF             │
├──────────────────────────┼──────────────────────┼──────────────────────┤
│ Length normalization     │ Yes (tunable via b)  │ Basic (fixed)        │
│ Term frequency saturation│ Yes (via k1)         │ No (unbounded)       │
│ Modern search usage      │ Standard (2024)      │ Legacy (pre-2010)    │
│ Azure AI Search default  │ Yes                  │ No                   │
│ Research backing         │ Extensive TREC wins  │ Historical baseline  │
│ Over-weighting prevention│ Yes                  │ No                   │
└──────────────────────────┴──────────────────────┴──────────────────────┘
```

**TF-IDF Problem Example:**
- Document with "laptop" mentioned 100 times gets 100× the score of 1 mention
- BM25 saturates: 100 mentions ≈ 5× the score of 1 mention (more realistic relevance)

**When to Use Each:**
- **BM25**: 99% of use cases (default in Azure AI Search)
- **TF-IDF**: Legacy systems, academic comparisons only

### Real-World Application Scenario

**Company**: Contoso Electronics (e-commerce retailer)
**Catalog**: 500,000 products across 50 categories
**Search Volume**: 2 million queries/month
**Challenge**: Users search for products using natural language queries like "gaming laptop under $1500" or "wireless noise cancelling headphones"

**Solution Architecture:**
1. **Index Design**: Product catalog with title, description, category, brand, specs
2. **BM25 Scoring**: Rank products by keyword relevance to query
3. **Filtering**: Apply price, category, brand filters post-ranking
4. **Scoring Profiles**: Boost newer products and higher ratings
5. **Performance**: P95 latency <100ms for 95% of queries

**Business Impact:**
- 40% improvement in search relevance (measured by click-through rate)
- 25% reduction in zero-result searches (better analyzer configuration)
- 15% increase in conversion rate (relevant results ranked higher)
- $0.12 per 1000 queries (cost-effective vs semantic search at $2.50/1000)

This document will guide you through implementing a similar solution, from BM25 theory to production deployment.

---

## BM25 Algorithm

Understanding the BM25 algorithm at a deeper level helps you make informed decisions about index design, query construction, and scoring profile configuration. While Azure AI Search abstracts away much of the complexity, knowing how BM25 works enables you to diagnose relevance issues and optimize for your specific use case.

### Mathematical Formula

The complete BM25 formula for scoring a document D given query Q:

```
BM25(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))

Where:
- D = Document
- Q = Query
- qi = Query term i
- f(qi, D) = Frequency of term qi in document D
- |D| = Length of document D
- avgdl = Average document length in collection
- k1 = Controls term frequency saturation (default 1.2)
- b = Controls document length normalization (default 0.75)
- IDF(qi) = log((N - df + 0.5) / (df + 0.5))
  - N = Total number of documents
  - df = Number of documents containing qi
```

### Parameters Explained

```python
class BM25Parameters:
    """BM25 algorithm parameters."""
    
    @staticmethod
    def explain_k1():
        """
        k1: Term frequency saturation parameter (default 1.2)
        
        - Higher k1 (1.5-2.0): More weight to term frequency
          Use for: Long documents, repeated terms are important
        
        - Lower k1 (0.5-1.0): Less emphasis on repetition
          Use for: Short documents, avoid over-weighting repeats
        
        - Azure AI Search: Fixed at 1.2 (not configurable)
        """
        return {
            'default': 1.2,
            'range': '0.0-3.0',
            'azure_configurable': False,
            'recommendation': 'Use scoring profiles for weighting instead'
        }
    
    @staticmethod
    def explain_b():
        """
        b: Document length normalization parameter (default 0.75)
        
        - Higher b (0.9-1.0): Strong length normalization
          Use for: Mixed document lengths, prevent long docs from dominating
        
        - Lower b (0.3-0.5): Weak normalization
          Use for: Similar length documents
        
        - b = 0: No length normalization
        - b = 1: Full normalization
        
        - Azure AI Search: Fixed at 0.75 (not configurable)
        """
        return {
            'default': 0.75,
            'range': '0.0-1.0',
            'azure_configurable': False,
            'recommendation': 'Accept default, use field weights in scoring profiles'
        }

# Usage
k1_info = BM25Parameters.explain_k1()
print(f"k1 default: {k1_info['default']}")
print(f"Configurable in Azure: {k1_info['azure_configurable']}")
```

---

## Index Configuration

### Basic Full-Text Index

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    ComplexField
)
from azure.core.credentials import AzureKeyCredential

class FullTextIndexBuilder:
    """Build optimized full-text search index."""
    
    def __init__(self, endpoint, api_key):
        self.client = SearchIndexClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
    
    def create_product_index(self, index_name="products"):
        """Create product catalog index for full-text search."""
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
                analyzer_name="en.microsoft",  # English analyzer
                searchable=True,
                filterable=False,
                sortable=False
            ),
            SearchableField(
                name="description",
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft",
                searchable=True
            ),
            SearchableField(
                name="category",
                type=SearchFieldDataType.String,
                analyzer_name="keyword",  # Exact match for categories
                searchable=True,
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="brand",
                type=SearchFieldDataType.String,
                analyzer_name="keyword",
                searchable=True,
                filterable=True,
                facetable=True
            ),
            SimpleField(
                name="price",
                type=SearchFieldDataType.Double,
                filterable=True,
                sortable=True,
                facetable=True
            ),
            SimpleField(
                name="rating",
                type=SearchFieldDataType.Double,
                filterable=True,
                sortable=True
            ),
            SearchableField(
                name="tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                analyzer_name="en.microsoft",
                searchable=True,
                filterable=True,
                facetable=True
            ),
            SimpleField(
                name="inStock",
                type=SearchFieldDataType.Boolean,
                filterable=True
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields
        )
        
        result = self.client.create_or_update_index(index)
        print(f"✅ Index '{index_name}' created")
        return result

# Usage
import os

builder = FullTextIndexBuilder(
    endpoint=os.getenv("SEARCH_ENDPOINT"),
    api_key=os.getenv("SEARCH_API_KEY")
)

index = builder.create_product_index()
```

### Advanced Index with Multiple Languages

```python
class MultilingualIndexBuilder:
    """Build index supporting multiple languages."""
    
    def __init__(self, endpoint, api_key):
        self.client = SearchIndexClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
    
    def create_multilingual_index(self, index_name="docs_multilingual"):
        """Create index with language-specific fields."""
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True
            ),
            # English fields
            SearchableField(
                name="title_en",
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft",
                searchable=True
            ),
            SearchableField(
                name="content_en",
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft",
                searchable=True
            ),
            # French fields
            SearchableField(
                name="title_fr",
                type=SearchFieldDataType.String,
                analyzer_name="fr.microsoft",
                searchable=True
            ),
            SearchableField(
                name="content_fr",
                type=SearchFieldDataType.String,
                analyzer_name="fr.microsoft",
                searchable=True
            ),
            # Spanish fields
            SearchableField(
                name="title_es",
                type=SearchFieldDataType.String,
                analyzer_name="es.microsoft",
                searchable=True
            ),
            SearchableField(
                name="content_es",
                type=SearchFieldDataType.String,
                analyzer_name="es.microsoft",
                searchable=True
            ),
            # Language-agnostic metadata
            SimpleField(
                name="language",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SimpleField(
                name="publishDate",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True
            )
        ]
        
        index = SearchIndex(name=index_name, fields=fields)
        result = self.client.create_or_update_index(index)
        print(f"✅ Multilingual index '{index_name}' created")
        return result
```

---

## Query Implementation

### Simple Search

```python
from azure.search.documents import SearchClient

class FullTextSearcher:
    """Execute full-text searches."""
    
    def __init__(self, endpoint, index_name, api_key):
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )
    
    def simple_search(self, query_text, top=10):
        """
        Simple full-text search.
        
        Args:
            query_text: Search query
            top: Number of results to return
            
        Returns:
            List of search results
        """
        results = self.client.search(
            search_text=query_text,
            top=top,
            include_total_count=True
        )
        
        documents = []
        for result in results:
            documents.append({
                'id': result['id'],
                'score': result['@search.score'],
                'title': result.get('title'),
                'description': result.get('description')
            })
        
        return {
            'total_count': results.get_count(),
            'documents': documents
        }
    
    def search_with_highlights(self, query_text, top=10):
        """
        Search with hit highlighting.
        
        Returns matched text snippets.
        """
        results = self.client.search(
            search_text=query_text,
            top=top,
            highlight_fields="title,description",
            highlight_pre_tag="<mark>",
            highlight_post_tag="</mark>"
        )
        
        documents = []
        for result in results:
            highlights = result.get('@search.highlights', {})
            documents.append({
                'id': result['id'],
                'score': result['@search.score'],
                'title': result.get('title'),
                'description': result.get('description'),
                'highlights': {
                    'title': highlights.get('title', []),
                    'description': highlights.get('description', [])
                }
            })
        
        return documents
    
    def field_weighted_search(self, query_text, field_weights=None, top=10):
        """
        Search with field-specific weights.
        
        Args:
            query_text: Search query
            field_weights: Dict of field weights e.g., {'title': 3, 'description': 1}
            top: Number of results
        """
        # Build search fields parameter
        search_fields = None
        if field_weights:
            weighted_fields = [
                f"{field}^{weight}" for field, weight in field_weights.items()
            ]
            search_fields = ",".join(weighted_fields)
        
        results = self.client.search(
            search_text=query_text,
            search_fields=search_fields,
            top=top
        )
        
        documents = []
        for result in results:
            documents.append({
                'id': result['id'],
                'score': result['@search.score'],
                'title': result.get('title'),
                'description': result.get('description')
            })
        
        return documents

# Usage
searcher = FullTextSearcher(
    endpoint=os.getenv("SEARCH_ENDPOINT"),
    index_name="products",
    api_key=os.getenv("SEARCH_API_KEY")
)

# Simple search
results = searcher.simple_search("laptop computer", top=10)
print(f"Found {results['total_count']} results")

# Search with highlights
highlighted = searcher.search_with_highlights("gaming laptop")
for doc in highlighted:
    print(f"Title: {doc['title']}")
    if doc['highlights']['title']:
        print(f"  Matched: {doc['highlights']['title'][0]}")

# Weighted search (title 3x more important than description)
weighted = searcher.field_weighted_search(
    "laptop",
    field_weights={'title': 3, 'description': 1},
    top=10
)
```

### Advanced Query Syntax

```python
class AdvancedQueryBuilder:
    """Build complex full-text queries."""
    
    @staticmethod
    def phrase_search(phrase):
        """
        Exact phrase matching.
        
        Example: "gaming laptop" finds exact phrase
        """
        return f'"{phrase}"'
    
    @staticmethod
    def boolean_query(must_terms=None, should_terms=None, must_not_terms=None):
        """
        Boolean query with AND, OR, NOT operators.
        
        Args:
            must_terms: List of required terms
            should_terms: List of optional terms
            must_not_terms: List of excluded terms
        """
        query_parts = []
        
        if must_terms:
            and_clause = " AND ".join(must_terms)
            query_parts.append(f"({and_clause})")
        
        if should_terms:
            or_clause = " OR ".join(should_terms)
            query_parts.append(f"({or_clause})")
        
        if must_not_terms:
            not_clause = " ".join([f"NOT {term}" for term in must_not_terms])
            query_parts.append(not_clause)
        
        return " ".join(query_parts)
    
    @staticmethod
    def wildcard_search(prefix):
        """
        Wildcard/prefix search.
        
        Example: "lap*" matches "laptop", "lapdog", etc.
        """
        return f"{prefix}*"
    
    @staticmethod
    def fuzzy_search(term, edit_distance=1):
        """
        Fuzzy matching with edit distance.
        
        Example: "laptop~1" matches "lapto", "laptops", etc.
        """
        return f"{term}~{edit_distance}"
    
    @staticmethod
    def proximity_search(term1, term2, distance=5):
        """
        Terms within specified distance.
        
        Example: "azure search"~5 finds "azure" within 5 words of "search"
        """
        return f'"{term1} {term2}"~{distance}'
    
    @staticmethod
    def field_scoped_query(field, term):
        """
        Search in specific field.
        
        Example: title:laptop
        """
        return f"{field}:{term}"

# Usage
qb = AdvancedQueryBuilder()

# Exact phrase
query1 = qb.phrase_search("gaming laptop")
# Result: "gaming laptop"

# Boolean query: must have "laptop", should have "gaming", must not have "used"
query2 = qb.boolean_query(
    must_terms=["laptop"],
    should_terms=["gaming", "performance"],
    must_not_terms=["used", "refurbished"]
)
# Result: (laptop) (gaming OR performance) NOT used NOT refurbished

# Fuzzy search
query3 = qb.fuzzy_search("laptop", edit_distance=1)
# Result: laptop~1 (matches "lapto", "laptops", etc.)

# Field-specific
query4 = qb.field_scoped_query("brand", "Dell")
# Result: brand:Dell

# Combined query
complex_query = f"{qb.field_scoped_query('brand', 'Dell')} AND {qb.phrase_search('gaming laptop')}"
```

---

## Search Modes

### Simple vs Full Query Syntax

```python
class SearchModes:
    """Different search query modes."""
    
    def __init__(self, search_client):
        self.client = search_client
    
    def simple_mode_search(self, query_text):
        """
        Simple query syntax (default).
        
        Supports: +, |, -, ", (), *, ~
        
        Examples:
        - "wifi + bluetooth" → must have both
        - "wifi | bluetooth" → must have either
        - "laptop -used" → laptop but not used
        """
        results = self.client.search(
            search_text=query_text,
            query_type="simple"  # Default
        )
        return list(results)
    
    def full_lucene_search(self, query_text):
        """
        Full Lucene query syntax.
        
        Supports: AND, OR, NOT, field scoping, fuzzy, proximity, regex, etc.
        
        Examples:
        - "title:(laptop OR tablet) AND brand:Dell"
        - "description:gaming~2" → fuzzy with edit distance 2
        - "\"azure search\"~10" → terms within 10 positions
        """
        results = self.client.search(
            search_text=query_text,
            query_type="full"
        )
        return list(results)
    
    def search_mode_any_vs_all(self, query_text, search_mode="any"):
        """
        Control how multiple terms are combined.
        
        Args:
            search_mode: "any" (OR) or "all" (AND)
        
        "any": Documents match if ANY term matches (more results)
        "all": Documents match if ALL terms match (fewer, more relevant)
        """
        results = self.client.search(
            search_text=query_text,
            search_mode=search_mode
        )
        return list(results)

# Usage
modes = SearchModes(searcher.client)

# Simple mode (user-friendly)
simple_results = modes.simple_mode_search("laptop + gaming")

# Full Lucene (power users)
lucene_results = modes.full_lucene_search("title:(laptop OR tablet) AND brand:Dell")

# Search mode comparison
any_results = modes.search_mode_any_vs_all("gaming laptop", search_mode="any")
all_results = modes.search_mode_any_vs_all("gaming laptop", search_mode="all")

print(f"'any' mode: {len(any_results)} results")
print(f"'all' mode: {len(all_results)} results")
```

---

## Analyzers

### Built-in Analyzers

```python
class AnalyzerGuide:
    """Guide to Azure AI Search analyzers."""
    
    @staticmethod
    def get_language_analyzers():
        """Available language-specific analyzers."""
        return {
            'English': 'en.microsoft',
            'French': 'fr.microsoft',
            'German': 'de.microsoft',
            'Spanish': 'es.microsoft',
            'Italian': 'it.microsoft',
            'Portuguese (Brazil)': 'pt-BR.microsoft',
            'Portuguese (Portugal)': 'pt-PT.microsoft',
            'Chinese (Simplified)': 'zh-Hans.microsoft',
            'Japanese': 'ja.microsoft',
            'Korean': 'ko.microsoft',
            'Russian': 'ru.microsoft',
            'Arabic': 'ar.microsoft',
            'Dutch': 'nl.microsoft',
            'Polish': 'pl.microsoft',
            'Swedish': 'sv.microsoft'
        }
    
    @staticmethod
    def get_special_analyzers():
        """Special-purpose analyzers."""
        return {
            'keyword': 'Exact matching, no tokenization (for IDs, codes)',
            'pattern': 'Regex-based tokenization',
            'simple': 'Lowercases, splits on non-letters',
            'standard.lucene': 'Standard Lucene analyzer',
            'stop': 'Removes stop words',
            'whitespace': 'Splits only on whitespace'
        }
    
    @staticmethod
    def choose_analyzer(use_case):
        """Recommend analyzer for use case."""
        recommendations = {
            'product_names': 'en.microsoft or language-specific',
            'product_codes': 'keyword',
            'categories': 'keyword',
            'descriptions': 'en.microsoft or language-specific',
            'email_addresses': 'keyword',
            'urls': 'keyword',
            'technical_docs': 'en.microsoft',
            'legal_text': 'en.microsoft',
            'code_samples': 'whitespace or keyword'
        }
        
        return recommendations.get(use_case, 'en.microsoft')

# Usage
guide = AnalyzerGuide()

# Get language analyzers
languages = guide.get_language_analyzers()
print(f"English analyzer: {languages['English']}")

# Get recommendation
recommended = guide.choose_analyzer('product_names')
print(f"For product names, use: {recommended}")
```

### Testing Analyzers

```python
from azure.search.documents.indexes.models import AnalyzeTextOptions

class AnalyzerTester:
    """Test how analyzers tokenize text."""
    
    def __init__(self, index_client):
        self.client = index_client
    
    def analyze_text(self, text, analyzer_name, index_name):
        """
        See how an analyzer processes text.
        
        Args:
            text: Text to analyze
            analyzer_name: Analyzer to use
            index_name: Index name
            
        Returns:
            List of tokens
        """
        analyze_request = AnalyzeTextOptions(
            text=text,
            analyzer_name=analyzer_name
        )
        
        result = self.client.analyze_text(
            index_name=index_name,
            analyze_request=analyze_request
        )
        
        tokens = [token.token for token in result.tokens]
        return tokens
    
    def compare_analyzers(self, text, analyzers, index_name):
        """Compare how different analyzers process same text."""
        results = {}
        
        for analyzer in analyzers:
            tokens = self.analyze_text(text, analyzer, index_name)
            results[analyzer] = tokens
        
        return results

# Usage
from azure.search.documents.indexes import SearchIndexClient

index_client = SearchIndexClient(
    endpoint=os.getenv("SEARCH_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("SEARCH_API_KEY"))
)

tester = AnalyzerTester(index_client)

# Test single analyzer
text = "High-Performance Gaming Laptop with 16GB RAM"
tokens = tester.analyze_text(text, "en.microsoft", "products")
print(f"Tokens: {tokens}")
# Output: ['high', 'performance', 'gaming', 'laptop', '16gb', 'ram']

# Compare analyzers
comparison = tester.compare_analyzers(
    text,
    ["en.microsoft", "standard.lucene", "keyword"],
    "products"
)

for analyzer, tokens in comparison.items():
    print(f"\n{analyzer}:")
    print(f"  {tokens}")
```

---

## Scoring Profiles

### Basic Scoring Profile

```python
from azure.search.documents.indexes.models import (
    ScoringProfile,
    TextWeights,
    ScoringFunction,
    FreshnessScoringFunction,
    MagnitudeScoringFunction,
    ScoringFunctionInterpolation
)

class ScoringProfileBuilder:
    """Build scoring profiles for relevance tuning."""
    
    @staticmethod
    def create_text_weighted_profile(name="textWeighted"):
        """
        Create profile with field weights.
        
        Boosts matches in certain fields.
        """
        profile = ScoringProfile(
            name=name,
            text_weights=TextWeights(weights={
                'title': 3.0,      # Title matches 3x more important
                'tags': 2.0,       # Tag matches 2x more important
                'description': 1.0  # Description baseline weight
            })
        )
        
        return profile
    
    @staticmethod
    def create_freshness_profile(name="recentFirst"):
        """
        Boost recent documents.
        
        Uses publish date to favor newer content.
        """
        freshness_function = FreshnessScoringFunction(
            field_name="publishDate",
            boost=5.0,  # Multiply score by up to 5x
            interpolation=ScoringFunctionInterpolation.LINEAR,
            parameters={
                'boostingDuration': 'P30D'  # Boost docs from last 30 days
            }
        )
        
        profile = ScoringProfile(
            name=name,
            functions=[freshness_function],
            function_aggregation='sum'
        )
        
        return profile
    
    @staticmethod
    def create_popularity_profile(name="popularFirst"):
        """
        Boost popular items.
        
        Uses rating or view count.
        """
        magnitude_function = MagnitudeScoringFunction(
            field_name="rating",
            boost=3.0,
            interpolation=ScoringFunctionInterpolation.LINEAR,
            parameters={
                'boostingRangeStart': 4.0,  # Start boosting at 4 stars
                'boostingRangeEnd': 5.0     # Max boost at 5 stars
            }
        )
        
        profile = ScoringProfile(
            name=name,
            functions=[magnitude_function],
            function_aggregation='sum'
        )
        
        return profile
    
    @staticmethod
    def create_combined_profile(name="combined"):
        """
        Combine text weights with boosting functions.
        """
        freshness_function = FreshnessScoringFunction(
            field_name="publishDate",
            boost=2.0,
            interpolation=ScoringFunctionInterpolation.LINEAR,
            parameters={'boostingDuration': 'P90D'}
        )
        
        popularity_function = MagnitudeScoringFunction(
            field_name="rating",
            boost=2.0,
            interpolation=ScoringFunctionInterpolation.LINEAR,
            parameters={
                'boostingRangeStart': 3.5,
                'boostingRangeEnd': 5.0
            }
        )
        
        profile = ScoringProfile(
            name=name,
            text_weights=TextWeights(weights={
                'title': 2.5,
                'description': 1.0
            }),
            functions=[freshness_function, popularity_function],
            function_aggregation='sum'
        )
        
        return profile

# Usage - Add to index definition
builder = ScoringProfileBuilder()

scoring_profiles = [
    builder.create_text_weighted_profile(),
    builder.create_freshness_profile(),
    builder.create_popularity_profile(),
    builder.create_combined_profile()
]

# Update index with scoring profiles
index = SearchIndex(
    name="products",
    fields=fields,
    scoring_profiles=scoring_profiles,
    default_scoring_profile="textWeighted"  # Optional default
)
```

### Using Scoring Profiles in Queries

```python
class ScoringProfileSearcher:
    """Execute searches with scoring profiles."""
    
    def __init__(self, search_client):
        self.client = search_client
    
    def search_with_profile(self, query_text, profile_name, top=10):
        """
        Search using a specific scoring profile.
        
        Args:
            query_text: Search query
            profile_name: Name of scoring profile to use
            top: Number of results
        """
        results = self.client.search(
            search_text=query_text,
            scoring_profile=profile_name,
            top=top
        )
        
        return list(results)
    
    def compare_profiles(self, query_text, profile_names, top=5):
        """Compare results from different scoring profiles."""
        comparison = {}
        
        for profile in profile_names:
            results = self.search_with_profile(query_text, profile, top)
            comparison[profile] = [
                {
                    'id': r['id'],
                    'title': r.get('title'),
                    'score': r['@search.score']
                }
                for r in results
            ]
        
        return comparison

# Usage
profile_searcher = ScoringProfileSearcher(searcher.client)

# Search with specific profile
recent_results = profile_searcher.search_with_profile(
    "laptop",
    profile_name="recentFirst",
    top=10
)

# Compare profiles
comparison = profile_searcher.compare_profiles(
    "laptop",
    profile_names=["textWeighted", "recentFirst", "popularFirst"],
    top=5
)

for profile, results in comparison.items():
    print(f"\n{profile}:")
    for r in results:
        print(f"  {r['title']} (score: {r['score']:.2f})")
```

---

## Optimization Techniques

### Query Performance Tips

```python
class FullTextOptimization:
    """Optimization techniques for full-text search."""
    
    @staticmethod
    def optimize_query_for_performance():
        """Best practices for fast queries."""
        return {
            'use_filters': 'Filter before search for better performance',
            'limit_fields': 'Search only necessary fields',
            'use_top_parameter': 'Limit results with top parameter',
            'avoid_wildcard_prefix': 'Avoid leading wildcards (e.g., *term)',
            'use_select': 'Return only needed fields',
            'enable_caching': 'Identical queries are cached for ~1 minute'
        }
    
    @staticmethod
    def optimize_index_for_performance():
        """Index design best practices."""
        return {
            'minimize_searchable_fields': 'Only make necessary fields searchable',
            'use_appropriate_analyzers': 'Language-specific for better tokenization',
            'strategic_field_attributes': 'Set filterable, sortable only when needed',
            'avoid_large_fields': 'Split very large text into chunks',
            'use_suggester': 'Pre-build for autocomplete features'
        }

# Example optimized query
def optimized_search(search_client, query_text):
    """Optimized search query example."""
    results = search_client.search(
        search_text=query_text,
        filter="inStock eq true and price le 2000",  # Pre-filter
        search_fields="title,tags",  # Limit search fields
        select="id,title,price,rating",  # Return only needed fields
        top=20,  # Limit results
        query_type="simple",  # Simpler parsing
        search_mode="all"  # More precise
    )
    
    return list(results)
```

---

## Performance Tuning

### Benchmarking

```python
import time
from statistics import mean, median, stdev

class FullTextBenchmark:
    """Benchmark full-text search performance."""
    
    def __init__(self, search_client):
        self.client = search_client
    
    def benchmark_query(self, query_text, iterations=10):
        """
        Benchmark a single query.
        
        Args:
            query_text: Query to test
            iterations: Number of runs
            
        Returns:
            Performance statistics
        """
        latencies = []
        
        for i in range(iterations):
            start = time.time()
            results = list(self.client.search(query_text, top=10))
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
            if i == 0:
                result_count = len(results)
        
        return {
            'query': query_text,
            'iterations': iterations,
            'result_count': result_count,
            'latency_ms': {
                'min': min(latencies),
                'max': max(latencies),
                'mean': mean(latencies),
                'median': median(latencies),
                'stdev': stdev(latencies) if len(latencies) > 1 else 0,
                'p95': sorted(latencies)[int(0.95 * len(latencies))]
            }
        }
    
    def benchmark_query_variations(self, base_query, variations):
        """
        Compare performance of query variations.
        
        Args:
            base_query: Base query text
            variations: Dict of variation parameters
        """
        results = {}
        
        for name, params in variations.items():
            query = params.get('query', base_query)
            start = time.time()
            search_results = list(self.client.search(query, **params.get('options', {})))
            latency = (time.time() - start) * 1000
            
            results[name] = {
                'latency_ms': latency,
                'result_count': len(search_results)
            }
        
        return results

# Usage
benchmark = FullTextBenchmark(searcher.client)

# Benchmark single query
stats = benchmark.benchmark_query("gaming laptop", iterations=20)
print(f"Query: {stats['query']}")
print(f"Mean latency: {stats['latency_ms']['mean']:.2f}ms")
print(f"P95 latency: {stats['latency_ms']['p95']:.2f}ms")

# Compare variations
variations = {
    'simple': {
        'query': 'gaming laptop',
        'options': {'query_type': 'simple', 'search_mode': 'any'}
    },
    'simple_all': {
        'query': 'gaming laptop',
        'options': {'query_type': 'simple', 'search_mode': 'all'}
    },
    'lucene': {
        'query': 'title:(gaming AND laptop)',
        'options': {'query_type': 'full'}
    }
}

comparison = benchmark.benchmark_query_variations('gaming laptop', variations)
for name, result in comparison.items():
    print(f"{name}: {result['latency_ms']:.2f}ms ({result['result_count']} results)")
```

---

## Best Practices

Full-text search with BM25 is mature technology, but there are numerous pitfalls that can degrade relevance, performance, or user experience. These best practices are derived from production deployments handling billions of queries across diverse industries.

### Index Design and Field Configuration

**✅ DO: Use Appropriate Analyzers for Each Field Type**
- **Product titles**: Use language-specific analyzer (`en.microsoft` for English)
  - Handles stemming: "running shoes" matches "run shoe"
  - Removes stop words: "the best laptop" → "best laptop"
  
- **Categories/brands**: Use `keyword` analyzer for exact matching
  - Preserves case: "Apple" ≠ "apple"
  - No tokenization: "Surface Pro" stays as one term
  
- **SKUs/part numbers**: Use `keyword` or custom pattern analyzer
  - Exact match: "ABC-123-XYZ" must match exactly
  
- **Descriptions**: Use language analyzer with synonyms
  - Expand vocabulary: "cheap" matches "affordable", "inexpensive"

**Example:**
```python
SearchableField(
    name="title",
    type=SearchFieldDataType.String,
    analyzer_name="en.microsoft",  # ✅ Language-aware
    searchable=True
)

SearchableField(
    name="brand",
    type=SearchFieldDataType.String,
    analyzer_name="keyword",  # ✅ Exact match for brands
    searchable=True,
    filterable=True
)
```

**❌ DON'T: Make All Fields Searchable**
- **Problem**: Searching across 20 fields dilutes relevance
- **Impact**: "laptop battery" matches products with "laptop" in title OR "battery" in specs
- **Cost**: Increased index size (30-40% larger) and slower queries (2-3× latency)

**Solution: Limit to 3-5 core search fields**
```python
# BAD: 15 searchable fields
title, description, short_description, long_description, features, 
specs, category, subcategory, brand, manufacturer, tags, keywords, 
sku, model, variant  # Overkill!

# GOOD: 3-4 targeted fields
title, description, category, brand  # Focused, relevant
```

### Query Construction and Search Modes

**✅ DO: Use Search Mode "All" for Precision-Critical Scenarios**
- **Use Case**: E-commerce, legal search, medical records
- **Behavior**: Requires ALL query terms to appear in document
- **Example**: Query "wireless bluetooth headphones" requires all 3 words
- **Trade-off**: Higher precision, lower recall (fewer but more relevant results)

```python
# Precision mode
results = search_client.search(
    search_text="wireless bluetooth headphones",
    search_mode="all"  # ✅ All terms required
)
# Returns: 150 highly relevant results
```

**✅ DO: Use Search Mode "Any" for Recall-Critical Scenarios**
- **Use Case**: Knowledge base, documentation search, content discovery
- **Behavior**: Matches documents with ANY query term
- **Example**: Query "wireless bluetooth headphones" matches docs with ANY word
- **Trade-off**: Higher recall, lower precision (more results, some less relevant)

```python
# Recall mode
results = search_client.search(
    search_text="wireless bluetooth headphones",
    search_mode="any"  # ✅ Any term matches
)
# Returns: 8,500 results (includes "wireless chargers", "bluetooth speakers", etc.)
```

**❌ DON'T: Use Leading Wildcards Without Understanding Performance Impact**
- **Problem**: `*phone` (leading wildcard) scans EVERY term in the index
- **Impact**: 100-1000× slower than regular search
- **Cost**: Can timeout on large indexes (>1M documents)

**Performance Comparison:**
```python
# BAD: Leading wildcard (1,500ms latency)
results = search_client.search(search_text="*phone")  # Scans entire index

# GOOD: Trailing wildcard (15ms latency)
results = search_client.search(search_text="phone*")  # Uses term prefix structure

# BEST: Full word search (8ms latency)
results = search_client.search(search_text="phone")  # Direct term lookup
```

### Analyzer Selection and Text Processing

**✅ DO: Test Analyzers with Analyze API Before Indexing**
- **Why**: See exactly how your text will be tokenized and processed
- **Benefit**: Catch analyzer misconfigurations before indexing millions of documents

```python
from azure.search.documents.indexes import SearchIndexClient

# Test analyzer behavior
response = index_client.analyze_text(
    index_name="products",
    analyze_request={
        "text": "Samsung Galaxy S24 Ultra 5G",
        "analyzer": "en.microsoft"
    }
)

# See tokens: ["samsung", "galaxy", "s24", "ultra", "5g"]
print("Tokens:", [token.token for token in response.tokens])

# Compare with keyword analyzer
response_keyword = index_client.analyze_text(
    index_name="products",
    analyze_request={
        "text": "Samsung Galaxy S24 Ultra 5G",
        "analyzer": "keyword"
    }
)

# See tokens: ["Samsung Galaxy S24 Ultra 5G"]  # Single token
```

**❌ DON'T: Ignore Analyzer Choice (Default Analyzer Trap)**
- **Problem**: Using `standard.lucene` analyzer when language-specific is better
- **Impact**: Misses stemming, stop word removal, language-specific tokenization
- **Example**: "running shoes" doesn't match "run shoe" (no stemming)

**Analyzer Decision Tree:**
```
Field Type                | Recommended Analyzer      | Reason
--------------------------|---------------------------|---------------------------
Product names/titles      | en.microsoft (or language)| Stemming, stop words
Descriptions              | en.microsoft + synonyms   | Natural language processing
Categories/tags           | keyword                   | Exact matching
SKUs/part numbers         | keyword or pattern        | No tokenization
Email addresses           | pattern analyzer          | Special character handling
URLs                      | pattern analyzer          | Preserve structure
```

### Scoring and Relevance Tuning

**✅ DO: Implement Scoring Profiles for Business Logic**
- **Use Case**: Boost newer products, higher-rated items, promoted content
- **Benefit**: Combines BM25 relevance with business priorities

```python
scoring_profile = ScoringProfile(
    name="boost_popular_recent",
    functions=[
        # Boost products with high ratings
        ScoringFunction(
            field_name="rating",
            interpolation="linear",
            magnitude={"boostingRangeStart": 3.0, "boostingRangeEnd": 5.0, "constantBoostBeyondRange": False},
            boost=2.0  # 2× boost for highly rated products
        ),
        # Boost recently added products
        ScoringFunction(
            field_name="dateAdded",
            interpolation="linear",
            freshness={"boostingDuration": "P90D"},  # Boost products added in last 90 days
            boost=1.5  # 1.5× boost for fresh content
        )
    ]
)
```

**❌ DON'T: Over-Use Fuzzy Search (Performance and Precision Trade-off)**
- **Problem**: Fuzzy search with edit distance 2 generates 1000s of term variations
- **Impact**: 10-20× slower queries, lower precision (matches too many irrelevant terms)
- **When to use**: Autocorrect, typo tolerance on specific fields only

**Fuzzy Search Guidelines:**
```python
# BAD: Fuzzy on all fields (300ms latency)
results = search_client.search(
    search_text="lapto~2",  # Edit distance 2 on ALL searchable fields
    query_type="full"
)

# GOOD: Fuzzy on title only (35ms latency)
results = search_client.search(
    search_text="title:lapto~1",  # Edit distance 1, title field only
    query_type="full"
)

# BEST: Use fuzzy sparingly, only when user likely has typo
# Detect zero results, then retry with fuzzy
first_try = search_client.search("lapto")
if len(list(first_try)) == 0:
    # Retry with fuzzy
    results = search_client.search("title:lapto~1", query_type="full")
```

### Performance Optimization

**✅ DO: Filter Before Searching (Query Execution Order)**
- **Why**: Filters reduce the document set BEFORE scoring
- **Impact**: 2-5× faster queries on large indexes
- **Cost**: No additional cost, pure optimization

```python
# BAD: Search then filter (slower)
results = search_client.search(
    search_text="laptop",
    # Scores ALL laptops, then filters
)
# Then manually filter results

# GOOD: Filter then search (2-3× faster)
results = search_client.search(
    search_text="laptop",
    filter="price le 1500 and brand eq 'Dell'",  # ✅ Filter BEFORE scoring
    select=["title", "price", "rating"]
)
```

**✅ DO: Limit Searchable Fields to Essentials**
- **Why**: Fewer fields = smaller index = faster queries
- **Guideline**: 3-5 searchable fields for most applications
- **Impact**: 30-40% smaller index size, 20-30% faster queries

**Field Type Decision:**
```python
# For each field, ask:
# 1. Should users search for this content? → searchable=True
# 2. Should users filter by this value? → filterable=True
# 3. Should users sort by this value? → sortable=True
# 4. Should this appear in facets? → facetable=True

# Example: Product price
SimpleField(
    name="price",
    type=SearchFieldDataType.Double,
    searchable=False,  # ✅ Don't search "$19.99"
    filterable=True,   # ✅ Filter by price range
    sortable=True,     # ✅ Sort by price
    facetable=True     # ✅ Price range facets
)
```

### User Experience and Highlighting

**✅ DO: Implement Highlighting for Better UX**
- **Why**: Shows users WHERE query terms matched
- **Impact**: 15-25% increase in click-through rate
- **Cost**: Negligible performance overhead (<5ms)

```python
results = search_client.search(
    search_text="wireless headphones",
    highlight_fields="title,description",
    highlight_pre_tag="<mark>",
    highlight_post_tag="</mark>"
)

for result in results:
    # Display: "Sony <mark>Wireless</mark> Noise-Cancelling <mark>Headphones</mark>"
    if "@search.highlights" in result:
        print(result["@search.highlights"]["title"])
```

**✅ DO: Implement Autocomplete and Suggestions**
- **Use Case**: Help users formulate better queries
- **Impact**: 30% reduction in zero-result searches
- **Implementation**: Use suggester with edge n-gram tokenization

```python
# Configure suggester in index
suggester = SearchSuggester(
    name="product-suggester",
    source_fields=["title", "brand"]
)

# Query for autocomplete
suggestions = search_client.autocomplete(
    search_text="gam",
    suggester_name="product-suggester",
    mode="oneTerm"
)
# Returns: ["gaming", "game", "games"]
```

### Monitoring and Continuous Improvement

**✅ DO: Benchmark Queries Regularly**
- **Why**: Detect performance regressions as index grows
- **Frequency**: Weekly for production systems
- **Metrics**: P50, P95, P99 latency; zero-result rate; avg results per query

```python
# Automated benchmark suite
test_queries = [
    "laptop",                    # Single term
    "gaming laptop",             # Two terms
    "wireless bluetooth headphones",  # Three terms
    "Dell XPS 15 i7",           # Brand + model
    "noise cancelling",         # Phrase
]

for query in test_queries:
    stats = benchmark.benchmark_query(query, iterations=100)
    assert stats['latency_ms']['p95'] < 100, f"Query '{query}' P95 latency too high"
```

**❌ DON'T: Forget to Monitor Zero-Result Searches**
- **Why**: Indicates missing content or analyzer issues
- **Action**: Review top zero-result queries monthly
- **Fix**: Add synonyms, adjust analyzers, expand catalog

### Security and Cost Management

**✅ DO: Use API Keys with Minimal Required Permissions**
- **Query-only keys**: For client-side search (read-only)
- **Admin keys**: For index management only (server-side)
- **Rotation**: Rotate admin keys quarterly

```python
# Client-side: Use query key
search_client = SearchClient(
    endpoint=endpoint,
    index_name="products",
    credential=AzureKeyCredential(query_key)  # ✅ Query-only key
)

# Server-side: Use admin key
index_client = SearchIndexClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(admin_key)  # Admin key for index management
)
```

**✅ DO: Estimate Costs Before Scaling**
- **Index size**: ~1.5× source document size (with searchable fields, analyzers)
- **Query costs**: Free tier = 3 QPS; S1 = 100 QPS ($250/month)
- **Storage**: S1 = 25GB ($0.40/GB/month beyond tier limit)

**Cost Example:**
```
Scenario: 500,000 products, 2M queries/month
- Source data: 10GB JSON
- Index size: ~15GB (1.5× with analyzers, suggestions)
- Required tier: S1 (25GB storage, 100 QPS)
- Cost: $250/month (includes storage + queries)
- Cost per query: $0.000125 (extremely cost-effective)
```

### Summary: BM25 Best Practices Checklist

**Before Indexing:**
- [ ] Choose appropriate analyzer for each field type
- [ ] Test analyzers with Analyze API
- [ ] Limit searchable fields to 3-5 core fields
- [ ] Configure scoring profiles for business logic
- [ ] Set up suggester for autocomplete

**During Querying:**
- [ ] Use `search_mode="all"` for precision scenarios
- [ ] Filter before searching when possible
- [ ] Implement result highlighting
- [ ] Avoid leading wildcards
- [ ] Use fuzzy search sparingly

**Ongoing Monitoring:**
- [ ] Benchmark query performance weekly
- [ ] Monitor zero-result search rate
- [ ] Review top queries and adjust relevance
- [ ] Track P95 latency and set alerts
- [ ] Rotate API keys quarterly

---

## Troubleshooting

Even with proper configuration, you'll encounter relevance issues, performance problems, and unexpected search behavior. This section covers the most common full-text search issues and their solutions based on production experience.

### Issue 1: Irrelevant Results Ranking Higher Than Expected

**Symptoms:**
- Query "Dell laptop" returns HP laptops in top 3 results
- Product with exact title match appears on page 2
- Low-quality products rank higher than premium products

**Root Causes & Solutions:**

**Solution 1: Verify Field Weights in Scoring Profile**
```python
# Check current scoring profile
# Common issue: Description field weighted too heavily

# BAD: Description weight = 3.0, Title weight = 1.0
# Result: Laptops with "Dell" mentioned 5× in description outrank exact title match

# GOOD: Title weight = 3.0, Description weight = 1.0
scoring_profile = ScoringProfile(
    name="prioritize_title",
    text_weights=TextWeights(
        weights={
            "title": 3.0,      # ✅ Highest weight for title matches
            "description": 1.0,
            "category": 1.5
        }
    )
)
```

**Solution 2: Implement Boosting Functions**
```python
# Boost products with exact brand match
results = search_client.search(
    search_text="Dell laptop",
    search_fields=["title", "description"],
    filter="brand eq 'Dell'",  # ✅ Exact brand match gets priority
    scoring_profile="boost_exact_matches"
)
```

**Solution 3: Check Analyzer Configuration**
```kusto
# Analyze how "Dell laptop" is tokenized
# If using keyword analyzer on title: ["Dell laptop"] (exact match required)
# If using en.microsoft analyzer: ["dell", "laptop"] (both must appear)

# Test actual tokenization:
# Azure Portal → Index → Analyze Text → Enter "Dell XPS 15"
# Verify tokens match your expectations
```

### Issue 2: Query Performance Degrades as Index Grows

**Symptoms:**
- Queries that took 50ms now take 500ms
- P95 latency increases from 100ms to 800ms
- Timeouts on complex queries

**Root Causes & Solutions:**

**Solution 1: Reduce Number of Searchable Fields**
```python
# BAD: 15 searchable fields (searches ALL fields)
# Impact: 10× slower as index grows

# GOOD: Use search_fields parameter to limit scope
results = search_client.search(
    search_text="laptop",
    search_fields=["title", "description"],  # ✅ Only search 2 fields
    top=10
)
# Impact: 5-10× faster than searching all 15 fields
```

**Solution 2: Implement Query Result Caching**
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_search(query_hash):
    """Cache frequent queries for 5 minutes."""
    # Actual search implementation
    pass

# Generate cache key
query_hash = hashlib.md5(f"{search_text}:{filter}".encode()).hexdigest()
results = cached_search(query_hash)
```

**Solution 3: Use Filters to Reduce Search Scope**
```python
# BAD: Search all 500,000 products
results = search_client.search("laptop", top=10)
# Scores all 500K products (500ms)

# GOOD: Filter to category first
results = search_client.search(
    search_text="laptop",
    filter="category eq 'Electronics' and price le 2000",  # ✅ Reduces to 50K products
    top=10
)
# Scores only 50K products (80ms, 6× faster)
```

### Issue 3: Zero Results for Valid Queries

**Symptoms:**
- Query "laptop" returns 0 results (but products exist)
- Query "Samsung Galaxy" returns 0 results
- Phrase searches never match

**Root Causes & Solutions:**

**Solution 1: Check Search Mode (All vs Any)**
```python
# PROBLEM: search_mode="all" requires ALL terms
results = search_client.search(
    search_text="Samsung Galaxy S24 Ultra",
    search_mode="all"  # Requires: Samsung AND Galaxy AND S24 AND Ultra
)
# Returns 0 if any term missing

# SOLUTION: Use search_mode="any" for broader matching
results = search_client.search(
    search_text="Samsung Galaxy S24 Ultra",
    search_mode="any"  # Matches: Samsung OR Galaxy OR S24 OR Ultra
)
```

**Solution 2: Verify Analyzer Applied Correctly**
```python
# Test if documents were indexed with expected analyzer
from azure.search.documents.indexes import SearchIndexClient

# Analyze indexed content
response = index_client.analyze_text(
    index_name="products",
    analyze_request={
        "text": "Samsung Galaxy S24",
        "analyzer": "en.microsoft"
    }
)
# Tokens: ["samsung", "galaxy", "s24"]

# If query uses different analyzer, tokens won't match!
# FIX: Ensure index and query use SAME analyzer
```

**Solution 3: Add Synonyms for Common Variations**
```json
// Create synonym map
{
  "name": "product-synonyms",
  "synonyms": [
    "laptop, notebook, portable computer",
    "phone, mobile, smartphone, cell",
    "TV, television, display"
  ]
}
```

```python
# Apply synonym map to field
SearchableField(
    name="title",
    type=SearchFieldDataType.String,
    analyzer_name="en.microsoft",
    synonym_map_names=["product-synonyms"]  # ✅ Expand query terms
)
```

### Issue 4: Special Characters Break Searches

**Symptoms:**
- Query "C++" returns 0 results
- Query "user@example.com" doesn't match
- Hyphenated terms like "noise-cancelling" don't work

**Root Causes & Solutions:**

**Solution 1: Use Pattern Analyzer for Special Characters**
```python
# Create custom pattern analyzer
from azure.search.documents.indexes.models import PatternAnalyzer

pattern_analyzer = PatternAnalyzer(
    name="preserve_special_chars",
    pattern=r"[^\w@\.\-\+]+",  # Preserve: @ . - +
    flags=[]
)

# Apply to fields with special characters
SearchableField(
    name="sku",
    type=SearchFieldDataType.String,
    analyzer_name="preserve_special_chars",  # ✅ Handles "ABC-123-XYZ"
    searchable=True
)
```

**Solution 2: Escape Special Characters in Lucene Queries**
```python
import re

def escape_lucene_special_chars(query):
    """Escape Lucene special characters."""
    special_chars = r'[+\-&|!(){}\[\]^"~*?:\\\/]'
    return re.sub(special_chars, r'\\\g<0>', query)

# Query for "C++"
escaped_query = escape_lucene_special_chars("C++")  # "C\+\+"
results = search_client.search(escaped_query, query_type="full")
```

### Issue 5: Long Queries Timeout or Return Errors

**Symptoms:**
- Queries with 20+ terms timeout
- Error: "Query too complex"
- Wildcard queries never complete

**Root Causes & Solutions:**

**Solution 1: Simplify Query Structure**
```python
# BAD: 20-term query with wildcards (timeout)
query = "laptop notebook portable computer gaming workstation ultrabook chromebook macbook thinkpad inspiron pavilion aspire swift zenbook vivobook latitude precision xps spectre envy"

# GOOD: Extract 3-5 most important terms
important_terms = ["laptop", "gaming", "ultrabook"]
query = " ".join(important_terms)
```

**Solution 2: Remove Leading Wildcards**
```python
# BAD: Leading wildcard (scans entire index)
query = "*phone"  # Timeout on large index

# GOOD: Trailing wildcard (uses prefix structure)
query = "phone*"  # Fast

# BEST: Use suggester/autocomplete instead
suggestions = search_client.suggest(
    search_text="phon",
    suggester_name="product-suggester"
)
```

---

## Next Steps

- **[Vector Search](./09-vector-search.md)** - Semantic search with embeddings
- **[Hybrid Search](./10-hybrid-search.md)** - Combine full-text and vector
- **[Custom Analyzers](./13-custom-analyzers.md)** - Domain-specific text processing
- **[Scoring Profiles](./14-scoring-profiles.md)** - Advanced relevance tuning

---

*See also: [Query Optimization](./12-query-optimization.md) | [Performance Tuning](./12-query-optimization.md)*