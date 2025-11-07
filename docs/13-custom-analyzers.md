# Custom Analyzers

Complete guide to building custom text analyzers in Azure AI Search for specialized tokenization, filtering, and normalization. This document provides comprehensive coverage of analyzer architecture, component selection, domain-specific configurations, testing strategies, and production best practices for achieving optimal text processing and search relevance.

## 📋 Table of Contents
- [Overview](#overview)
- [Analyzer Components](#analyzer-components)
- [Tokenizers](#tokenizers)
- [Token Filters](#token-filters)
- [Character Filters](#character-filters)
- [Custom Analyzer Implementation](#custom-analyzer-implementation)
- [Domain-Specific Analyzers](#domain-specific-analyzers)
- [Testing Analyzers](#testing-analyzers)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Analyzers are the foundation of full-text search in Azure AI Search, transforming raw text into searchable tokens through a configurable pipeline of character filters, tokenizers, and token filters. While built-in analyzers work well for general-purpose search, custom analyzers are essential for domain-specific requirements, specialized terminology, or unique text processing needs. Properly configured analyzers can improve search precision by 30-60% compared to default analyzers for specialized domains.

### The Business Case for Custom Analyzers

**Without Custom Analyzers (Default Standard Analyzer):**
- Medical search: "MI" doesn't match "myocardial infarction" (0% recall for abbreviations)
- E-commerce: "iPhone13Pro" becomes one token, misses "iPhone 13" queries (40% missed queries)
- Legal: Case citations "Brown v. Board" loses structure (poor precision)
- Code search: "getUserName" not matched by "get user name" (case-sensitive issues)

**With Custom Analyzers (Domain-Optimized):**
- Medical: Synonym expansion MI → myocardial infarction, heart attack (90% recall improvement)
- E-commerce: Word delimiter splits "iPhone13Pro" → ["iPhone", "13", "Pro"] (40% more matches)
- Legal: Preserved citation structure with custom tokenization (65% precision improvement)
- Code: CamelCase splitting enables natural language queries (3× better developer experience)

### Real-World Application Scenario

**Company**: MedSearch Pro (Medical research database)
**Scale**: 5 million medical articles, 100K healthcare professionals, 20M searches/year
**Challenge**: Standard analyzer inadequate for medical terminology, abbreviations, and synonyms

**Previous Architecture (Standard Analyzer):**
```
Index Configuration:
- Analyzer: standard (built-in)
- No medical synonym support
- No abbreviation expansion
- Case-insensitive (loses important distinctions like "US" vs "us")

Search Problems:
- "MI" search: 0 results (should find "myocardial infarction")
- "HTN" search: 0 results (should find "hypertension")
- "heart attack" search: Misses "MI", "myocardial infarction"
- Drug name variations: "acetaminophen" doesn't match "paracetamol" (same drug)
- Dosage formats: "500mg" vs "500 mg" treated differently

Performance Metrics (Baseline):
- Medical term recall: 42% (missing 58% of relevant results)
- Abbreviation matching: 8% (only direct matches)
- User satisfaction: 2.3/5 stars
- Average search refinements: 4.2 per query
- Zero-result rate: 28%
```

**Custom Analyzer Implementation:**

**Phase 1: Medical Synonym Expansion (Week 1)**

```python
# Created medical synonym map with 5,000+ mappings
medical_synonyms = [
    "MI, myocardial infarction, heart attack",
    "HTN, hypertension, high blood pressure",
    "DM, diabetes mellitus, diabetes, sugar disease",
    "COPD, chronic obstructive pulmonary disease, emphysema, chronic bronchitis",
    "CHF, congestive heart failure, heart failure",
    "CVA, cerebrovascular accident, stroke",
    "acetaminophen, paracetamol, tylenol",
    "ibuprofen, advil, motrin",
    # ... 5,000+ medical terms, abbreviations, brand names
]

# Custom analyzer configuration
medical_analyzer = {
    'tokenizer': 'standard',
    'token_filters': [
        'lowercase',
        'medical_synonym_filter',  # Expand medical terms
        'asciifolding',  # Normalize accents
        'unique'  # Remove duplicates after expansion
    ]
}
```

**Results - Phase 1**:
- Medical term recall: 42% → 78% (+86% improvement)
- Abbreviation matching: 8% → 92% (+1,050% improvement)
- Zero-result rate: 28% → 12% (-57% reduction)
- Query refinements: 4.2 → 2.1 (50% reduction)

**Phase 2: Dosage Normalization (Week 2)**

```python
# Pattern replace filter for dosage normalization
dosage_normalizer = {
    'pattern': r'(\d+)\s*(mg|ml|g|l)',  # Match "500 mg" or "500mg"
    'replacement': r'$1$2'  # Normalize to "500mg"
}

# Updated analyzer
medical_analyzer_v2 = {
    'char_filters': ['dosage_normalizer'],  # Pre-processing
    'tokenizer': 'standard',
    'token_filters': [
        'lowercase',
        'medical_synonym_filter',
        'asciifolding',
        'unique'
    ]
}
```

**Results - Phase 2**:
- Dosage query matching: 63% → 94% (+49% improvement)
- User satisfaction: 2.3 → 3.8 stars (+65% improvement)

**Phase 3: Specialized Tokenization for Compound Terms (Week 3)**

```python
# Word delimiter for compound medical terms
compound_filter = {
    'type': 'word_delimiter',
    'generate_word_parts': True,  # "COVID-19" → ["COVID", "19"]
    'generate_number_parts': True,
    'split_on_case_change': True,  # "HbA1c" → ["Hb", "A1c"]
    'preserve_original': True  # Keep original too
}

# Final production analyzer
medical_analyzer_v3 = {
    'char_filters': ['dosage_normalizer', 'html_strip'],
    'tokenizer': 'standard',
    'token_filters': [
        'compound_filter',  # Handle compound terms
        'lowercase',
        'medical_synonym_filter',
        'asciifolding',
        'unique'
    ]
}
```

**Final Performance After Full Custom Analyzer:**
```
Search Metrics:
- Medical term recall: 42% → 89% (+112% improvement)
- Precision: 68% → 91% (+34% improvement)
- Abbreviation matching: 8% → 94% (+1,075% improvement)
- Dosage query matching: 63% → 96% (+52% improvement)
- Zero-result rate: 28% → 6% (-79% reduction)
- User satisfaction: 2.3 → 4.4 stars (+91% improvement)
- Avg query refinements: 4.2 → 1.3 (-69% reduction)

User Behavior Changes:
- Search session length: 8.5 min → 3.2 min (-62%, faster finding)
- Articles viewed per session: 2.1 → 4.8 (+129%, better relevance)
- Successful searches (found what they need): 54% → 91% (+69%)

Business Impact:
- Subscription renewals: 72% → 89% (+17 percentage points)
- Customer support tickets (search-related): 1,200/month → 180/month (-85%)
- Time to find relevant articles: 8.5 min → 3.2 min (saves 5.3 min/search)
- For 100K users × 200 searches/year: 106M minutes saved/year
- Estimated value: 106M min ÷ 60 = 1.77M hours × $75/hour (professional time) = $132M/year value created
```

**Cost of Implementation:**
```
Development Costs:
- Synonym map creation: 2 weeks (sourced from medical dictionaries)
- Analyzer configuration: 1 week
- Testing and validation: 1 week
- Total: 4 weeks × $15K/week = $60K

Ongoing Costs:
- Synonym map maintenance: 4 hours/month (medical terms evolve)
- Annual maintenance: $5K/year
- Index size increase: +15% (from synonym expansion)
- Storage cost increase: $500/month → $575/month = +$75/month = $900/year

Total First Year Cost: $60K + $5K + $900 = $65,900
Annual Ongoing Cost: $5,900

ROI:
- Value created: $132M/year (time savings + better outcomes)
- Cost: $65,900 first year
- ROI: 2,002× return on investment
- Payback period: 4.5 days (!)
```

**Key Lessons from MedSearch Pro:**
1. **Domain-specific synonyms** were the single biggest improvement (+86% recall)
2. **Normalization** (dosages, abbreviations) eliminated 22% of zero-result searches
3. **Compound term handling** improved technical term matching by 49%
4. **Testing with real users** revealed issues standard metrics missed (e.g., "COVID-19" not matching "COVID19")
5. **Incremental rollout** (v1 → v2 → v3) allowed validation and refinement

This document will guide you through building similarly effective custom analyzers for your domain.

---

### What are Custom Analyzers?

```python
class AnalyzerOverview:
    """Understanding custom analyzers in Azure AI Search."""
    
    @staticmethod
    def analyzer_pipeline():
        """
        Analyzer processing pipeline:
        
        Input Text
            ↓
        Character Filters (optional)
            ↓
        Tokenizer (required)
            ↓
        Token Filters (optional)
            ↓
        Output Tokens
        """
        return {
            'character_filters': {
                'stage': 1,
                'purpose': 'Pre-process text before tokenization',
                'examples': ['Remove HTML tags', 'Map characters', 'Pattern replacement'],
                'optional': True
            },
            'tokenizer': {
                'stage': 2,
                'purpose': 'Split text into tokens',
                'examples': ['standard', 'whitespace', 'pattern', 'keyword'],
                'optional': False,
                'note': 'Exactly one tokenizer required'
            },
            'token_filters': {
                'stage': 3,
                'purpose': 'Transform, filter, or generate tokens',
                'examples': ['lowercase', 'stop words', 'stemming', 'synonyms'],
                'optional': True,
                'note': 'Applied in order specified'
            }
        }
    
    @staticmethod
    def when_to_use_custom():
        """When to use custom analyzers vs built-in."""
        return {
            'use_built_in': [
                'Standard text search in common languages',
                'Language-specific search (en.microsoft, fr.microsoft)',
                'Simple keyword matching',
                'Quick prototyping'
            ],
            'use_custom': [
                'Domain-specific terminology (medical, legal, technical)',
                'Special tokenization rules (SKUs, codes, identifiers)',
                'Custom normalization requirements',
                'Synonym expansion',
                'Phonetic matching',
                'Remove special patterns (HTML, URLs)',
                'Multi-word token generation (n-grams)'
            ]
        }

# Usage
overview = AnalyzerOverview()
pipeline = overview.analyzer_pipeline()

print("Analyzer Pipeline:")
for component, details in pipeline.items():
    print(f"\n{component} (Stage {details['stage']}):")
    print(f"  Purpose: {details['purpose']}")
    print(f"  Optional: {details['optional']}")

when_custom = overview.when_to_use_custom()
print("\nUse custom analyzers for:")
for use_case in when_custom['use_custom']:
    print(f"  • {use_case}")
```

---

## Analyzer Components

### Component Types

```python
from azure.search.documents.indexes.models import (
    LexicalAnalyzerName,
    LexicalTokenizerName,
    TokenFilterName,
    CharFilterName
)

class AnalyzerComponents:
    """Available analyzer components in Azure AI Search."""
    
    @staticmethod
    def list_tokenizers():
        """Available tokenizers."""
        return {
            'standard': {
                'description': 'Grammar-based tokenization using Unicode Text Segmentation',
                'splits_on': 'Most symbols',
                'max_token_length': 255,
                'use_case': 'General-purpose text'
            },
            'whitespace': {
                'description': 'Splits on whitespace only',
                'splits_on': 'Spaces, tabs, newlines',
                'preserves': 'Punctuation',
                'use_case': 'Codes, identifiers with punctuation'
            },
            'keyword': {
                'description': 'No tokenization - entire input is single token',
                'splits_on': 'Nothing',
                'use_case': 'Exact match fields (SKUs, IDs, categories)'
            },
            'pattern': {
                'description': 'Regex-based tokenization',
                'splits_on': 'User-defined regex pattern',
                'use_case': 'Custom splitting rules (e.g., comma-separated)'
            },
            'edge_n_gram': {
                'description': 'Generates n-grams from beginning of tokens',
                'generates': 'Prefixes of specified lengths',
                'use_case': 'Autocomplete, prefix matching'
            },
            'n_gram': {
                'description': 'Generates n-grams from tokens',
                'generates': 'Substrings of specified lengths',
                'use_case': 'Partial word matching, fuzzy search'
            },
            'path_hierarchy': {
                'description': 'Tokenizes file paths',
                'example': '/a/b/c → [/a, /a/b, /a/b/c]',
                'use_case': 'File system paths, URLs'
            }
        }
    
    @staticmethod
    def list_token_filters():
        """Available token filters."""
        return {
            'lowercase': {
                'description': 'Convert tokens to lowercase',
                'example': 'HELLO → hello',
                'use_case': 'Case-insensitive search (almost always needed)'
            },
            'uppercase': {
                'description': 'Convert tokens to uppercase',
                'example': 'hello → HELLO',
                'use_case': 'Normalized uppercase storage'
            },
            'stop': {
                'description': 'Remove stop words (the, a, an, etc.)',
                'configurable': 'Custom stop word lists',
                'use_case': 'Reduce noise in text search'
            },
            'stemmer': {
                'description': 'Reduce words to root form',
                'example': 'running, runs → run',
                'languages': '30+ languages supported',
                'use_case': 'Match word variations'
            },
            'synonym': {
                'description': 'Expand tokens with synonyms',
                'example': 'laptop → laptop, notebook',
                'configurable': 'Custom synonym maps',
                'use_case': 'Improve recall with related terms'
            },
            'phonetic': {
                'description': 'Generate phonetic encodings',
                'algorithms': 'Metaphone, Soundex',
                'example': 'Smith → SM0',
                'use_case': 'Sound-alike matching (names)'
            },
            'word_delimiter': {
                'description': 'Split words on case changes, numbers, non-alphanumeric',
                'example': 'PowerShot500 → Power, Shot, 500',
                'use_case': 'Product names, compound words'
            },
            'edge_n_gram': {
                'description': 'Generate edge n-grams from tokens',
                'example': 'quick → q, qu, qui, quic, quick',
                'use_case': 'Autocomplete suggestions'
            },
            'n_gram': {
                'description': 'Generate n-grams from tokens',
                'example': 'quick → qu, ui, ic, ck',
                'use_case': 'Partial matching'
            },
            'truncate': {
                'description': 'Truncate tokens to specified length',
                'example': 'elephant → eleph (length=5)',
                'use_case': 'Limit token size'
            },
            'unique': {
                'description': 'Remove duplicate tokens',
                'use_case': 'After synonym expansion'
            },
            'reverse': {
                'description': 'Reverse token characters',
                'example': 'hello → olleh',
                'use_case': 'Suffix matching'
            },
            'asciifolding': {
                'description': 'Convert accented characters to ASCII',
                'example': 'café → cafe',
                'use_case': 'Normalize international text'
            }
        }
    
    @staticmethod
    def list_character_filters():
        """Available character filters."""
        return {
            'html_strip': {
                'description': 'Remove HTML tags',
                'example': '<p>Hello</p> → Hello',
                'use_case': 'Index HTML documents'
            },
            'mapping': {
                'description': 'Map characters to replacements',
                'example': 'é → e, ñ → n',
                'configurable': 'Custom character mappings',
                'use_case': 'Custom normalization'
            },
            'pattern_replace': {
                'description': 'Regex-based character replacement',
                'example': 'Replace digits: \\d → #',
                'use_case': 'Complex character transformations'
            }
        }

# Usage
components = AnalyzerComponents()

tokenizers = components.list_tokenizers()
print("Available Tokenizers:")
for name, details in tokenizers.items():
    print(f"\n{name}:")
    print(f"  {details['description']}")
    print(f"  Use case: {details['use_case']}")

filters = components.list_token_filters()
print("\n\nKey Token Filters:")
for name in ['lowercase', 'stemmer', 'synonym', 'stop']:
    if name in filters:
        print(f"\n{name}:")
        print(f"  {filters[name]['description']}")
```

---

## Tokenizers

### Tokenizer Implementation

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    CustomAnalyzer,
    PatternTokenizer,
    StandardTokenizer,
    WhitespaceTokenizer,
    EdgeNGramTokenizer,
    NGramTokenizer,
    LowercaseTokenFilter
)
from azure.core.credentials import AzureKeyCredential

class CustomTokenizers:
    """Implement custom tokenizers."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_pattern_tokenizer_analyzer(self, index_name):
        """
        Create analyzer with pattern tokenizer.
        
        Example: Split on commas for tag lists.
        """
        # Define custom tokenizer
        comma_tokenizer = PatternTokenizer(
            name="comma_tokenizer",
            pattern=",",  # Split on comma
        )
        
        # Define custom analyzer using the tokenizer
        comma_analyzer = CustomAnalyzer(
            name="comma_analyzer",
            tokenizer_name="comma_tokenizer",
            token_filters=[TokenFilterName.LOWERCASE, TokenFilterName.TRIM]
        )
        
        # Define fields
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="tags",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="comma_analyzer"
            )
        ]
        
        # Create index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            tokenizers=[comma_tokenizer],
            analyzers=[comma_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)
    
    def create_sku_analyzer(self, index_name):
        """
        Create analyzer for product SKUs.
        
        SKU format: ABC-123-XYZ
        Tokenizes on hyphens while preserving parts.
        """
        # Pattern tokenizer splits on hyphens
        sku_tokenizer = PatternTokenizer(
            name="sku_tokenizer",
            pattern="-"
        )
        
        # Analyzer preserves case (no lowercase filter)
        sku_analyzer = CustomAnalyzer(
            name="sku_analyzer",
            tokenizer_name="sku_tokenizer",
            token_filters=[TokenFilterName.TRIM]
        )
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="sku",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="sku_analyzer"
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            tokenizers=[sku_tokenizer],
            analyzers=[sku_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)
    
    def create_autocomplete_analyzer(self, index_name):
        """
        Create analyzer for autocomplete using edge n-grams.
        
        Generates prefixes: "quick" → q, qu, qui, quic, quick
        """
        # Edge n-gram tokenizer
        autocomplete_tokenizer = EdgeNGramTokenizer(
            name="autocomplete_tokenizer",
            min_gram=2,
            max_gram=10,
            token_chars=["letter", "digit"]
        )
        
        # Autocomplete analyzer
        autocomplete_analyzer = CustomAnalyzer(
            name="autocomplete_analyzer",
            tokenizer_name="autocomplete_tokenizer",
            token_filters=[TokenFilterName.LOWERCASE]
        )
        
        # Search analyzer (standard, no n-grams)
        search_analyzer = CustomAnalyzer(
            name="autocomplete_search",
            tokenizer_name=LexicalTokenizerName.STANDARD,
            token_filters=[TokenFilterName.LOWERCASE]
        )
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="autocomplete_analyzer",  # Index time
                search_analyzer_name="autocomplete_search"  # Query time
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            tokenizers=[autocomplete_tokenizer],
            analyzers=[autocomplete_analyzer, search_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)
    
    def create_path_analyzer(self, index_name):
        """
        Create analyzer for file paths.
        
        Path: /docs/projects/readme.md
        Tokens: /docs, /docs/projects, /docs/projects/readme.md
        """
        from azure.search.documents.indexes.models import PathHierarchyTokenizerV2
        
        path_tokenizer = PathHierarchyTokenizerV2(
            name="path_tokenizer",
            delimiter="/",
            replacement="/",
            max_token_length=300
        )
        
        path_analyzer = CustomAnalyzer(
            name="path_analyzer",
            tokenizer_name="path_tokenizer"
        )
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="filePath",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="path_analyzer"
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            tokenizers=[path_tokenizer],
            analyzers=[path_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)

# Usage
import os

tokenizer_setup = CustomTokenizers(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Create indexes with custom tokenizers
pattern_index = tokenizer_setup.create_pattern_tokenizer_analyzer("tags-index")
print("Created pattern tokenizer analyzer for tags")

sku_index = tokenizer_setup.create_sku_analyzer("products-sku")
print("Created SKU analyzer")

autocomplete_index = tokenizer_setup.create_autocomplete_analyzer("autocomplete-index")
print("Created autocomplete analyzer with edge n-grams")

path_index = tokenizer_setup.create_path_analyzer("files-index")
print("Created path hierarchy analyzer")
```

---

## Token Filters

### Token Filter Implementation

```python
from azure.search.documents.indexes.models import (
    StopwordsTokenFilter,
    SynonymTokenFilter,
    EdgeNGramTokenFilterV2,
    NGramTokenFilterV2,
    PhoneticTokenFilter,
    WordDelimiterTokenFilter,
    SynonymMap
)

class CustomTokenFilters:
    """Implement custom token filters."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_stop_words_analyzer(self, index_name):
        """
        Create analyzer with custom stop words.
        
        Removes common words that don't add search value.
        """
        # Custom stop words filter
        custom_stop_filter = StopwordsTokenFilter(
            name="custom_stop",
            stopwords=["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"]
        )
        
        # Analyzer with stop words
        stop_analyzer = CustomAnalyzer(
            name="stop_analyzer",
            tokenizer_name=LexicalTokenizerName.STANDARD,
            token_filters=[
                TokenFilterName.LOWERCASE,
                "custom_stop"
            ]
        )
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="stop_analyzer"
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            token_filters=[custom_stop_filter],
            analyzers=[stop_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)
    
    def create_synonym_analyzer(self, index_name, synonym_map_name):
        """
        Create analyzer with synonym expansion.
        
        First create synonym map, then analyzer using it.
        """
        # Create synonym map
        synonyms = [
            "laptop, notebook, portable computer",
            "phone, smartphone, mobile",
            "cheap, affordable, budget, inexpensive",
            "fast, quick, rapid, speedy"
        ]
        
        synonym_map = SynonymMap(
            name=synonym_map_name,
            synonyms="\n".join(synonyms)
        )
        
        self.index_client.create_or_update_synonym_map(synonym_map)
        
        # Create synonym filter
        synonym_filter = SynonymTokenFilter(
            name="custom_synonym",
            synonyms=synonym_map_name,
            expand=True  # Expand all synonyms
        )
        
        # Analyzer with synonyms
        synonym_analyzer = CustomAnalyzer(
            name="synonym_analyzer",
            tokenizer_name=LexicalTokenizerName.STANDARD,
            token_filters=[
                TokenFilterName.LOWERCASE,
                "custom_synonym"
            ]
        )
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="description",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="synonym_analyzer"
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            token_filters=[synonym_filter],
            analyzers=[synonym_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)
    
    def create_word_delimiter_analyzer(self, index_name):
        """
        Create analyzer that splits compound words.
        
        Example: "PowerShot500" → "Power", "Shot", "500"
        """
        # Word delimiter filter
        word_delimiter_filter = WordDelimiterTokenFilter(
            name="word_delimiter",
            generate_word_parts=True,
            generate_number_parts=True,
            catenate_words=False,
            catenate_numbers=False,
            catenate_all=False,
            split_on_case_change=True,
            preserve_original=True
        )
        
        # Analyzer with word delimiter
        product_analyzer = CustomAnalyzer(
            name="product_analyzer",
            tokenizer_name=LexicalTokenizerName.WHITESPACE,
            token_filters=[
                "word_delimiter",
                TokenFilterName.LOWERCASE
            ]
        )
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="productName",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="product_analyzer"
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            token_filters=[word_delimiter_filter],
            analyzers=[product_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)
    
    def create_phonetic_analyzer(self, index_name):
        """
        Create analyzer with phonetic matching.
        
        Matches names that sound similar: Smith, Smyth, Smythe
        """
        # Phonetic filter
        phonetic_filter = PhoneticTokenFilter(
            name="phonetic",
            encoder="metaphone"  # or "soundex", "caverphone1", etc.
        )
        
        # Phonetic analyzer
        name_analyzer = CustomAnalyzer(
            name="name_analyzer",
            tokenizer_name=LexicalTokenizerName.STANDARD,
            token_filters=[
                TokenFilterName.LOWERCASE,
                "phonetic"
            ]
        )
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="lastName",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="name_analyzer"
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            token_filters=[phonetic_filter],
            analyzers=[name_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)

# Usage
filter_setup = CustomTokenFilters(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Create indexes with custom filters
stop_index = filter_setup.create_stop_words_analyzer("content-index")
print("Created analyzer with stop words")

synonym_index = filter_setup.create_synonym_analyzer("products-synonym", "product-synonyms")
print("Created analyzer with synonyms")

delimiter_index = filter_setup.create_word_delimiter_analyzer("products-compound")
print("Created analyzer with word delimiter")

phonetic_index = filter_setup.create_phonetic_analyzer("names-index")
print("Created analyzer with phonetic matching")
```

---

## Character Filters

### Character Filter Implementation

```python
from azure.search.documents.indexes.models import (
    MappingCharFilter,
    PatternReplaceCharFilter
)

class CustomCharFilters:
    """Implement custom character filters."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_html_strip_analyzer(self, index_name):
        """
        Create analyzer that removes HTML tags.
        
        Input: "<p>Hello <b>world</b></p>"
        Output: "Hello world"
        """
        # Analyzer with HTML strip
        html_analyzer = CustomAnalyzer(
            name="html_analyzer",
            tokenizer_name=LexicalTokenizerName.STANDARD,
            char_filters=[CharFilterName.HTML_STRIP],
            token_filters=[TokenFilterName.LOWERCASE]
        )
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="htmlContent",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="html_analyzer"
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            analyzers=[html_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)
    
    def create_accent_normalizer(self, index_name):
        """
        Create analyzer that normalizes accented characters.
        
        Input: "café", "naïve"
        Output: "cafe", "naive"
        """
        # Mapping character filter
        accent_filter = MappingCharFilter(
            name="accent_normalizer",
            mappings=[
                "á=>a", "é=>e", "í=>i", "ó=>o", "ú=>u",
                "à=>a", "è=>e", "ì=>i", "ò=>o", "ù=>u",
                "ä=>a", "ë=>e", "ï=>i", "ö=>o", "ü=>u",
                "â=>a", "ê=>e", "î=>i", "ô=>o", "û=>u",
                "ñ=>n", "ç=>c"
            ]
        )
        
        # Analyzer with accent normalization
        normalized_analyzer = CustomAnalyzer(
            name="normalized_analyzer",
            tokenizer_name=LexicalTokenizerName.STANDARD,
            char_filters=["accent_normalizer"],
            token_filters=[TokenFilterName.LOWERCASE]
        )
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="text",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="normalized_analyzer"
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            char_filters=[accent_filter],
            analyzers=[normalized_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)
    
    def create_pattern_replace_analyzer(self, index_name):
        """
        Create analyzer with regex-based character replacement.
        
        Example: Remove phone number formatting
        Input: "(555) 123-4567"
        Output: "5551234567"
        """
        # Pattern replace filter - remove phone formatting
        phone_filter = PatternReplaceCharFilter(
            name="phone_normalizer",
            pattern=r"[\s\(\)\-\.]",  # Remove spaces, parens, hyphens, dots
            replacement=""
        )
        
        # URL normalizer - remove protocols
        url_filter = PatternReplaceCharFilter(
            name="url_normalizer",
            pattern=r"https?://",
            replacement=""
        )
        
        # Analyzer with pattern replacement
        contact_analyzer = CustomAnalyzer(
            name="contact_analyzer",
            tokenizer_name=LexicalTokenizerName.KEYWORD,
            char_filters=["phone_normalizer"],
            token_filters=[TokenFilterName.LOWERCASE]
        )
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="phone",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="contact_analyzer"
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            char_filters=[phone_filter, url_filter],
            analyzers=[contact_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)

# Usage
char_filter_setup = CustomCharFilters(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Create indexes with character filters
html_index = char_filter_setup.create_html_strip_analyzer("html-content")
print("Created HTML strip analyzer")

accent_index = char_filter_setup.create_accent_normalizer("international-text")
print("Created accent normalizer")

pattern_index = char_filter_setup.create_pattern_replace_analyzer("contacts")
print("Created pattern replace analyzer")
```

---

## Custom Analyzer Implementation

### Complete Custom Analyzer Example

```python
class CompleteCustomAnalyzer:
    """Build a complete custom analyzer from scratch."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_medical_analyzer(self, index_name):
        """
        Create analyzer for medical terminology.
        
        Features:
        - Preserve medical abbreviations
        - Expand medical synonyms
        - Remove stop words
        - Case-insensitive
        """
        # Synonym map for medical terms
        medical_synonyms = SynonymMap(
            name="medical_synonyms",
            synonyms="\n".join([
                "MI, myocardial infarction, heart attack",
                "HTN, hypertension, high blood pressure",
                "DM, diabetes mellitus, diabetes",
                "COPD, chronic obstructive pulmonary disease",
                "CHF, congestive heart failure, heart failure"
            ])
        )
        
        self.index_client.create_or_update_synonym_map(medical_synonyms)
        
        # Custom stop words (medical context)
        medical_stop_filter = StopwordsTokenFilter(
            name="medical_stop",
            stopwords=["the", "a", "an", "of", "with", "without", "patient", "mg", "ml"]
        )
        
        # Synonym filter
        medical_synonym_filter = SynonymTokenFilter(
            name="medical_synonym",
            synonyms="medical_synonyms",
            expand=True
        )
        
        # Complete medical analyzer
        medical_analyzer = CustomAnalyzer(
            name="medical_analyzer",
            tokenizer_name=LexicalTokenizerName.STANDARD,
            token_filters=[
                TokenFilterName.LOWERCASE,
                "medical_stop",
                "medical_synonym",
                TokenFilterName.UNIQUE  # Remove duplicates after synonym expansion
            ]
        )
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="diagnosis",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="medical_analyzer"
            ),
            SearchField(
                name="notes",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="medical_analyzer"
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            token_filters=[medical_stop_filter, medical_synonym_filter],
            analyzers=[medical_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)
    
    def create_code_analyzer(self, index_name):
        """
        Create analyzer for source code.
        
        Features:
        - Preserve case (case-sensitive)
        - Split on camelCase and snake_case
        - Keep special characters
        """
        # Word delimiter for code
        code_delimiter = WordDelimiterTokenFilter(
            name="code_delimiter",
            generate_word_parts=True,
            generate_number_parts=True,
            split_on_case_change=True,
            split_on_numerics=True,
            preserve_original=True
        )
        
        # Code analyzer (case-sensitive)
        code_analyzer = CustomAnalyzer(
            name="code_analyzer",
            tokenizer_name=LexicalTokenizerName.WHITESPACE,
            token_filters=["code_delimiter"]  # No lowercase!
        )
        
        # Case-insensitive variant
        code_insensitive_analyzer = CustomAnalyzer(
            name="code_insensitive_analyzer",
            tokenizer_name=LexicalTokenizerName.WHITESPACE,
            token_filters=[
                "code_delimiter",
                TokenFilterName.LOWERCASE
            ]
        )
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="code",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="code_analyzer"  # Case-sensitive index
            ),
            SearchField(
                name="comments",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="code_insensitive_analyzer"
            )
        ]
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            token_filters=[code_delimiter],
            analyzers=[code_analyzer, code_insensitive_analyzer]
        )
        
        return self.index_client.create_or_update_index(index)

# Usage
complete_setup = CompleteCustomAnalyzer(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Create specialized analyzers
medical_index = complete_setup.create_medical_analyzer("medical-records")
print("Created medical terminology analyzer")

code_index = complete_setup.create_code_analyzer("source-code")
print("Created source code analyzer")
```

---

## Domain-Specific Analyzers

### Industry-Specific Examples

```python
class DomainAnalyzers:
    """Domain-specific analyzer configurations."""
    
    @staticmethod
    def legal_analyzer_config():
        """
        Legal document analyzer configuration.
        
        Features:
        - Preserve case for citations (e.g., Brown v. Board)
        - Recognize legal citations
        - Legal term synonyms
        """
        return {
            'char_filters': ['html_strip'],  # Remove formatting
            'tokenizer': 'standard',
            'token_filters': [
                'unique',  # Remove duplicate citations
                # Custom: legal synonym expansion
            ],
            'use_cases': [
                'Case law search',
                'Contract analysis',
                'Legal research'
            ]
        }
    
    @staticmethod
    def ecommerce_analyzer_config():
        """
        E-commerce product analyzer configuration.
        
        Features:
        - Word delimiter for product names (iPhone13Pro)
        - Synonym expansion (cheap, affordable, budget)
        - Stop word removal
        """
        return {
            'char_filters': [],
            'tokenizer': 'standard',
            'token_filters': [
                'word_delimiter',  # Split compound product names
                'lowercase',
                'stop',  # Remove noise words
                'synonym',  # Product synonyms
                'unique'
            ],
            'examples': {
                'input': 'iPhone13Pro Max 256GB',
                'tokens': ['iphone', '13', 'pro', 'max', '256', 'gb']
            }
        }
    
    @staticmethod
    def technical_documentation_config():
        """
        Technical documentation analyzer.
        
        Features:
        - Preserve code snippets
        - Technical acronyms
        - Version numbers
        """
        return {
            'char_filters': ['html_strip'],
            'tokenizer': 'whitespace',  # Preserve technical terms
            'token_filters': [
                'lowercase',
                # Preserve acronyms in uppercase
            ],
            'use_cases': [
                'API documentation',
                'Technical manuals',
                'Code documentation'
            ]
        }
    
    @staticmethod
    def name_search_config():
        """
        Person name analyzer configuration.
        
        Features:
        - Phonetic matching
        - Handle multi-word names
        - Accent normalization
        """
        return {
            'char_filters': ['accent_normalizer'],
            'tokenizer': 'standard',
            'token_filters': [
                'lowercase',
                'phonetic',  # Metaphone encoding
                'unique'
            ],
            'examples': {
                'matching': {
                    'Smith': 'Smyth, Smythe, Smithe',
                    'Johnson': 'Jonson, Johnsen'
                }
            }
        }

# Usage
domain = DomainAnalyzers()

legal = domain.legal_analyzer_config()
print("Legal Analyzer:")
print(f"  Use cases: {legal['use_cases']}")

ecommerce = domain.ecommerce_analyzer_config()
print("\nE-commerce Analyzer:")
print(f"  Example: {ecommerce['examples']['input']}")
print(f"  Tokens: {ecommerce['examples']['tokens']}")
```

---

## Testing Analyzers

### Analyzer Testing Tools

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import AnalyzeTextOptions

class AnalyzerTester:
    """Test and validate custom analyzers."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def analyze_text(self, text, analyzer_name):
        """
        Analyze text using specified analyzer.
        
        Returns list of tokens generated.
        """
        result = self.index_client.analyze_text(
            AnalyzeTextOptions(
                text=text,
                analyzer_name=analyzer_name
            )
        )
        
        tokens = [token.token for token in result.tokens]
        return tokens
    
    def compare_analyzers(self, text, analyzer_names):
        """
        Compare output of multiple analyzers.
        """
        results = {}
        
        for analyzer_name in analyzer_names:
            tokens = self.analyze_text(text, analyzer_name)
            results[analyzer_name] = tokens
        
        return results
    
    def test_analyzer_examples(self, analyzer_name, test_cases):
        """
        Test analyzer with multiple examples.
        
        Args:
            analyzer_name: Name of analyzer to test
            test_cases: List of (input, expected_tokens)
        """
        results = []
        
        for input_text, expected in test_cases:
            actual = self.analyze_text(input_text, analyzer_name)
            
            matches = set(actual) == set(expected)
            
            results.append({
                'input': input_text,
                'expected': expected,
                'actual': actual,
                'matches': matches
            })
        
        return results
    
    def validate_tokenization(self, text, analyzer_name, min_tokens=1, max_tokens=100):
        """
        Validate tokenization output.
        
        Checks:
        - Token count in acceptable range
        - No empty tokens
        - No excessively long tokens
        """
        tokens = self.analyze_text(text, analyzer_name)
        
        issues = []
        
        # Check token count
        if len(tokens) < min_tokens:
            issues.append(f"Too few tokens: {len(tokens)} < {min_tokens}")
        if len(tokens) > max_tokens:
            issues.append(f"Too many tokens: {len(tokens)} > {max_tokens}")
        
        # Check for empty tokens
        empty_tokens = [t for t in tokens if not t]
        if empty_tokens:
            issues.append(f"Empty tokens found: {len(empty_tokens)}")
        
        # Check token lengths
        long_tokens = [t for t in tokens if len(t) > 100]
        if long_tokens:
            issues.append(f"Excessively long tokens: {long_tokens[:3]}")
        
        return {
            'text': text,
            'token_count': len(tokens),
            'tokens': tokens,
            'valid': len(issues) == 0,
            'issues': issues
        }

# Usage
tester = AnalyzerTester(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Test analyzer
text = "The quick brown fox jumps over the lazy dog"
tokens = tester.analyze_text(text, LexicalAnalyzerName.STANDARD)
print(f"Standard analyzer tokens: {tokens}")

# Compare analyzers
comparison = tester.compare_analyzers(
    "iPhone13Pro",
    [LexicalAnalyzerName.STANDARD, LexicalAnalyzerName.KEYWORD]
)
for analyzer, tokens in comparison.items():
    print(f"{analyzer}: {tokens}")

# Test with examples
test_cases = [
    ("laptop computer", ["laptop", "computer"]),
    ("PowerShot500", ["power", "shot", "500"])
]

results = tester.test_analyzer_examples("product_analyzer", test_cases)
for result in results:
    status = "✓" if result['matches'] else "✗"
    print(f"{status} {result['input']}: {result['actual']}")

# Validate tokenization
validation = tester.validate_tokenization(
    "The quick brown fox",
    LexicalAnalyzerName.STANDARD
)
print(f"Valid: {validation['valid']}")
print(f"Token count: {validation['token_count']}")
if validation['issues']:
    print(f"Issues: {validation['issues']}")
```

---

## Best Practices

### ✅ Do's
1. **Test analyzers thoroughly** with real data before production
2. **Use built-in analyzers** when possible (better maintained)
3. **Keep analyzer chains simple** (3-5 filters maximum)
4. **Document custom analyzers** with examples and use cases
5. **Use separate analyzers** for index time vs query time when needed
6. **Version analyzer names** for easier updates (my_analyzer_v2)
7. **Monitor token generation** to avoid excessive index size

### ❌ Don'ts
1. **Don't** over-tokenize (too many tokens = large index, slow queries)
2. **Don't** use different analyzers for index/query without reason
3. **Don't** forget to normalize case (lowercase filter is usually needed)
4. **Don't** apply heavy processing** at query time (slows searches)
5. **Don't** create analyzers without testing** token output
6. **Don't** ignore analyzer performance impact
7. **Don't** use complex regex patterns** in character filters (slow)

---

## Next Steps

- **[Scoring Profiles](./14-scoring-profiles.md)** - Advanced relevance tuning
- **[Index Management](./15-index-management.md)** - Managing analyzer updates
- **[Query Optimization](./12-query-optimization.md)** - Analyzer performance impact

---

*See also: [Full-Text Search](./08-fulltext-search-bm25.md) | [Hybrid Search](./10-hybrid-search.md)*