# Full-Text Search (BM25)

Complete guide to implementing and optimizing keyword-based full-text search using BM25 (Best Match 25) algorithm in Azure AI Search.

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

---

## Overview

### What is BM25?

BM25 (Best Match 25) is a ranking function used by search engines to estimate the relevance of documents to a search query. It's the default algorithm in Azure AI Search for full-text search.

**Key Characteristics:**
- **Term frequency (TF)**: How often a term appears in a document
- **Inverse document frequency (IDF)**: Rarity of the term across all documents
- **Document length normalization**: Prevents bias toward longer documents
- **Configurable parameters**: k1 and b for tuning

### BM25 vs TF-IDF

```
┌──────────────────┬──────────────┬──────────────┐
│    Feature       │    BM25      │   TF-IDF     │
├──────────────────┼──────────────┼──────────────┤
│ Length norm      │ Yes (tunable)│ Basic        │
│ Saturation       │ Yes          │ No           │
│ Modern usage     │ Standard     │ Legacy       │
│ Azure default    │ Yes          │ No           │
└──────────────────┴──────────────┴──────────────┘
```

---

## BM25 Algorithm

### Mathematical Formula

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

### ✅ Do's
1. **Use appropriate analyzers** for each field type
2. **Implement scoring profiles** for relevance tuning
3. **Test queries** with analyze API before indexing
4. **Filter before searching** for better performance
5. **Limit searchable fields** to what's necessary
6. **Use search mode "all"** for precision
7. **Implement highlighting** for better UX

### ❌ Don'ts
1. **Don't** make all fields searchable
2. **Don't** use leading wildcards (performance killer)
3. **Don't** ignore analyzer choice
4. **Don't** over-use fuzzy search (expensive)
5. **Don't** forget to benchmark queries

---

## Next Steps

- **[Vector Search](./09-vector-search.md)** - Semantic search with embeddings
- **[Hybrid Search](./10-hybrid-search.md)** - Combine full-text and vector
- **[Custom Analyzers](./13-custom-analyzers.md)** - Domain-specific text processing
- **[Scoring Profiles](./14-scoring-profiles.md)** - Advanced relevance tuning

---

*See also: [Query Optimization](./12-query-optimization.md) | [Performance Tuning](./12-query-optimization.md)*