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

Custom analyzers can dramatically improve search quality when properly configured, but poor implementation can lead to performance issues, maintenance headaches, and degraded search relevance. This section provides comprehensive guidance based on production deployments and common pitfalls.

### 1. Analyzer Selection Strategy

**DO: Start with built-in analyzers and only customize when necessary**

Built-in analyzers are battle-tested, optimized, and maintained by Azure. They handle 80% of use cases effectively.

```python
# GOOD: Use built-in when appropriate
from azure.search.documents.indexes.models import LexicalAnalyzerName

# English content - use built-in English analyzer
SearchField(
    name="description",
    type=SearchFieldDataType.String,
    searchable=True,
    analyzer_name=LexicalAnalyzerName.EN_MICROSOFT  # Optimized for English
)

# Product codes - use keyword analyzer
SearchField(
    name="sku",
    type=SearchFieldDataType.String,
    searchable=True,
    analyzer_name=LexicalAnalyzerName.KEYWORD  # Exact matching
)
```

```python
# BAD: Creating custom analyzer for standard use case
custom_english = CustomAnalyzer(
    name="my_english_analyzer",
    tokenizer_name="standard",
    token_filters=["lowercase", "stop", "porter_stem"]
    # This recreates built-in functionality - unnecessary maintenance burden
)
```

**Decision Tree: When to use custom analyzers**

| Scenario | Use Built-in | Use Custom | Reason |
|----------|-------------|------------|--------|
| General English text | ✅ | ❌ | en.microsoft handles it |
| Product SKUs (exact match) | ✅ | ❌ | keyword analyzer works |
| Medical terminology | ❌ | ✅ | Need domain synonyms |
| Compound product names | ❌ | ✅ | Need word delimiter |
| Legal citations | ❌ | ✅ | Need structure preservation |
| Autocomplete | ❌ | ✅ | Need edge n-grams |
| Multi-language content | ✅ | ❌ | Use language-specific analyzers |
| Code search (camelCase) | ❌ | ✅ | Need case-sensitive splitting |

**Performance Impact**:
- Built-in analyzers: 5-10ms per 1K tokens
- Simple custom (2-3 filters): 8-15ms per 1K tokens
- Complex custom (5+ filters): 15-30ms per 1K tokens

### 2. Tokenizer Selection

**DO: Match tokenizer to your data structure**

Different data types require different tokenization strategies.

```python
# GOOD: Appropriate tokenizer for data type

# Tags: comma-separated
pattern_tokenizer = PatternTokenizer(
    name="comma_tokenizer",
    pattern=r",\s*"  # Split on comma with optional whitespace
)
# "apple, banana, cherry" → ["apple", "banana", "cherry"]

# Product codes: preserve hyphens and numbers
keyword_tokenizer = KeywordTokenizerV2(
    name="sku_tokenizer",
    max_token_length=50
)
# "ABC-123-XYZ" → ["ABC-123-XYZ"] (preserved as single token)

# Search suggestions: edge n-grams
edge_ngram_tokenizer = EdgeNGramTokenizer(
    name="autocomplete_tokenizer",
    min_gram=2,
    max_gram=10,
    token_chars=["letter", "digit"]
)
# "search" → ["se", "sea", "sear", "searc", "search"]

# File paths: hierarchical structure
path_tokenizer = PathHierarchyTokenizerV2(
    name="path_tokenizer",
    delimiter="/"
)
# "/docs/api/search" → ["/docs", "/docs/api", "/docs/api/search"]
```

```python
# BAD: Wrong tokenizer for use case

# Using standard tokenizer for product codes
standard_tokenizer = "standard"
# "ABC-123-XYZ" → ["ABC", "123", "XYZ"] (splits on hyphens - loses structure!)

# Using keyword tokenizer for sentences
keyword_tokenizer = "keyword"
# "The quick brown fox" → ["The quick brown fox"] (no tokenization - can't match partial words!)
```

**Tokenizer Selection Guide**:

| Data Type | Best Tokenizer | Example Input | Example Output |
|-----------|---------------|---------------|----------------|
| Natural language | `standard` | "Hello world!" | ["Hello", "world"] |
| Product SKUs | `keyword` | "SKU-12345" | ["SKU-12345"] |
| Tags/Categories | `pattern` (comma) | "red, blue" | ["red", "blue"] |
| Email addresses | `pattern` (@/.) | "user@example.com" | ["user", "example", "com"] |
| File paths | `path_hierarchy` | "/a/b/c" | ["/a", "/a/b", "/a/b/c"] |
| Autocomplete | `edge_ngram` | "search" | ["s", "se", "sea", ...] |
| Code identifiers | `word_delimiter` | "getUserName" | ["get", "User", "Name"] |

### 3. Token Filter Order Optimization

**DO: Apply filters in optimal order to minimize processing**

Filter order affects both performance and results. The general principle: **reduce token count early**.

```python
# GOOD: Optimal filter order
optimal_analyzer = CustomAnalyzer(
    name="optimal_analyzer",
    tokenizer_name="standard",
    token_filters=[
        "lowercase",          # 1. Normalize case first (fast, reduces variants)
        "stop",               # 2. Remove stop words (reduces token count)
        "asciifolding",       # 3. Normalize accents (reduces variants)
        "synonym_map",        # 4. Expand synonyms (after normalization)
        "porter_stem",        # 5. Stem words (after synonym expansion)
        "unique"              # 6. Remove duplicates (after all transformations)
    ]
)

# Example: "The café's" 
# → "the café's" (lowercase)
# → "café's" (stop word removed)
# → "cafes" (asciifolding)
# → ["cafes", "coffee shop"] (synonym expansion)
# → ["cafe", "coffe", "shop"] (stemming)
# → ["cafe", "coffe", "shop"] (duplicates removed)
```

```python
# BAD: Inefficient filter order
inefficient_analyzer = CustomAnalyzer(
    name="inefficient_analyzer",
    tokenizer_name="standard",
    token_filters=[
        "synonym_map",        # ❌ WRONG: Synonyms before normalization
        "porter_stem",        # ❌ WRONG: Stemming before synonyms
        "lowercase",          # ❌ WRONG: Should be first
        "stop"                # ❌ WRONG: Should be earlier
    ]
)

# Problem: "Café" won't match synonym for "cafe" (case/accent mismatch)
# Problem: Stemmed words might not match synonym map entries
# Problem: Processing stop words through all filters (wasted work)
```

**Performance Comparison** (100K tokens):

| Filter Order | Processing Time | Token Count | Index Size |
|--------------|----------------|-------------|------------|
| Optimal (above) | 1.2s | 65K | 450 KB |
| Lowercase last | 1.8s (+50%) | 65K | 450 KB |
| No stop filter | 1.5s (+25%) | 95K (+46%) | 680 KB (+51%) |
| Synonyms before lowercase | 1.6s (+33%) | 65K | 450 KB |

**Recommended Filter Order**:

1. **Character normalization** (lowercase, asciifolding) - Fast, reduces variants
2. **Stop word removal** - Reduces token count for subsequent filters
3. **Structural filters** (word_delimiter, elision) - Change token structure
4. **Synonym expansion** - After normalization, before stemming
5. **Stemming** - After synonyms (so both original and synonym get stemmed)
6. **Cleanup** (unique, length_filter) - Remove unwanted results

### 4. Index Time vs. Query Time Analyzers

**DO: Use different analyzers for index and query when appropriate**

Some transformations should only happen at index time or query time, not both.

```python
# GOOD: Separate index and query analyzers for autocomplete

# Index analyzer: Generate all edge n-grams
autocomplete_index = CustomAnalyzer(
    name="autocomplete_index",
    tokenizer_name=EdgeNGramTokenizer(
        name="edge_ngram_tokenizer",
        min_gram=2,
        max_gram=10,
        token_chars=["letter", "digit"]
    ),
    token_filters=["lowercase"]
)

# Query analyzer: Just tokenize normally (match against n-grams)
autocomplete_query = CustomAnalyzer(
    name="autocomplete_query",
    tokenizer_name="standard",
    token_filters=["lowercase"]
)

# Field configuration
SearchField(
    name="title",
    type=SearchFieldDataType.String,
    searchable=True,
    analyzer_name="autocomplete_index",        # Index time
    search_analyzer_name="autocomplete_query"  # Query time
)

# Why this works:
# Index: "search" → ["se", "sea", "sear", "searc", "search"]
# Query: "sea" → ["sea"]
# Match: "sea" matches the "sea" token in index ✓
```

```python
# BAD: Same analyzer for both (inefficient)

# Using edge n-gram for both index and query
SearchField(
    name="title",
    analyzer_name="autocomplete_index"  # Applied to both index and query
)

# Problem:
# Index: "search" → ["se", "sea", "sear", "searc", "search"]
# Query: "sea" → ["se", "sea"]  # Unnecessary work!
# Match: "se" matches "se", "sea" matches "sea" (redundant matches)
```

**When to use different analyzers**:

| Use Case | Index Analyzer | Query Analyzer | Reason |
|----------|---------------|----------------|--------|
| Autocomplete | Edge n-gram | Standard | Generate n-grams once, query efficiently |
| Synonym expansion | With synonyms | Without synonyms | Avoid query expansion (faster, better ranking) |
| Phonetic matching | Phonetic filter | Standard | Match sound at index, query normal text |
| Stemming | Aggressive stem | Light stem | Broader matching at index, precision at query |

**Performance Impact** (1,000 queries/sec):

| Configuration | Query Latency | CPU Usage | Why |
|--------------|---------------|-----------|-----|
| Optimal (different analyzers) | 45ms | 30% | Minimal query processing |
| Same complex analyzer | 85ms (+89%) | 65% (+117%) | Re-processing at query time |

### 5. Synonym Management

**DO: Manage synonyms as versioned, external resources**

Synonyms change frequently and should be managed separately from analyzers.

```python
# GOOD: External synonym map (versioned and updatable)

from azure.search.documents.indexes.models import SynonymMap

# Create synonym map
medical_synonyms_v2 = SynonymMap(
    name="medical_synonyms_v2",  # Versioned name
    synonyms="""
MI, myocardial infarction, heart attack
HTN, hypertension, high blood pressure
DM, diabetes mellitus, diabetes
CHF, congestive heart failure, heart failure
CVA, cerebrovascular accident, stroke
    """.strip()
)

index_client.create_or_update_synonym_map(medical_synonyms_v2)

# Reference in analyzer
medical_analyzer = CustomAnalyzer(
    name="medical_analyzer",
    tokenizer_name="standard",
    token_filters=[
        "lowercase",
        SynonymTokenFilter(
            name="medical_synonym_filter",
            synonyms_map_name="medical_synonyms_v2",  # Reference by name
            expand=True  # "MI" matches all variants
        )
    ]
)
```

```python
# BAD: Hardcoded synonyms in analyzer (hard to update)

# Inline synonyms (requires index rebuild to change)
hardcoded_analyzer = CustomAnalyzer(
    name="hardcoded_analyzer",
    tokenizer_name="standard",
    token_filters=[
        SynonymTokenFilter(
            name="inline_synonyms",
            synonyms=[
                "MI=>myocardial infarction",
                "HTN=>hypertension"
            ]  # Can't update without rebuilding index!
        )
    ]
)
```

**Synonym Map Best Practices**:

1. **Version synonym maps**: `medical_synonyms_v1`, `medical_synonyms_v2`
2. **Use expand=True** for bidirectional matching: `MI, heart attack` (both match both)
3. **Use expand=False** for one-way: `MI=>myocardial infarction` (only MI expands)
4. **Organize by domain**: Separate maps for medical, legal, technical terms
5. **Document source**: Include comments with source (medical dictionary, etc.)
6. **Test before deployment**: Use Analyze API to verify behavior

**Synonym Expansion Modes**:

| Mode | Syntax | Index: "MI" | Query: "MI" | Query: "heart attack" |
|------|--------|-------------|-------------|---------------------|
| Bidirectional (expand=True) | `MI, heart attack` | ["MI", "heart", "attack"] | ["MI", "heart", "attack"] | ["heart", "attack"] |
| One-way | `MI=>heart attack` | ["heart", "attack"] | ["heart", "attack"] | ["heart", "attack"] |

**When to expand at index vs. query**:

- **Index-time expansion** (expand=True):
  - ✅ Pros: Faster queries, better ranking, simpler query processing
  - ❌ Cons: Larger index size, harder to update synonyms
  - **Best for**: Stable synonym lists, performance-critical applications

- **Query-time expansion** (expand at query only):
  - ✅ Pros: Smaller index, easier synonym updates, flexible
  - ❌ Cons: Slower queries, less precise ranking
  - **Best for**: Frequently changing synonyms, smaller datasets

### 6. Testing and Validation

**DO: Test analyzers with Analyze API before production**

Always validate tokenization behavior before indexing documents.

```python
# GOOD: Systematic analyzer testing

from azure.search.documents.indexes.models import AnalyzeTextOptions

def test_analyzer(index_client, analyzer_name, test_cases):
    """
    Test analyzer with multiple inputs and expected outputs.
    """
    results = []
    
    for input_text, expected_tokens in test_cases:
        # Analyze text
        result = index_client.analyze_text(
            AnalyzeTextOptions(
                text=input_text,
                analyzer_name=analyzer_name
            )
        )
        
        actual_tokens = [token.token for token in result.tokens]
        
        # Validate
        matches = set(actual_tokens) == set(expected_tokens)
        
        results.append({
            'input': input_text,
            'expected': expected_tokens,
            'actual': actual_tokens,
            'status': '✅ PASS' if matches else '❌ FAIL'
        })
        
        if not matches:
            print(f"❌ FAIL: {input_text}")
            print(f"  Expected: {expected_tokens}")
            print(f"  Actual:   {actual_tokens}")
    
    return results

# Test cases for medical analyzer
test_cases = [
    ("MI", ["mi", "myocardial", "infarction", "heart", "attack"]),
    ("HTN patient", ["htn", "hypertension", "high", "blood", "pressure", "patient"]),
    ("500mg", ["500mg"]),  # Dosage preserved
]

results = test_analyzer(index_client, "medical_analyzer", test_cases)

# Check pass rate
pass_rate = sum(1 for r in results if '✅' in r['status']) / len(results)
print(f"Pass rate: {pass_rate:.0%}")
assert pass_rate >= 0.90, "Analyzer validation failed"
```

```python
# BAD: No testing before production

# Deploy analyzer without validation
index_client.create_or_update_index(index)

# Index documents
# Discover issues in production when searches fail!
```

**Testing Checklist**:

- [ ] Test with real data samples (not just synthetic examples)
- [ ] Test edge cases (empty strings, special characters, very long inputs)
- [ ] Test multilingual inputs if applicable
- [ ] Validate token count (avoid over-tokenization)
- [ ] Check for unexpected token splits
- [ ] Verify case normalization
- [ ] Confirm synonym expansion behavior
- [ ] Test with production query patterns

**Common Test Cases**:

| Test Scenario | Example Input | Check For |
|--------------|---------------|-----------|
| Special characters | "user@example.com" | Preserved or split correctly |
| Numbers | "COVID-19", "500mg" | Preserved with context |
| Compound words | "PowerShot500" | Split appropriately |
| Accents | "café", "naïve" | Normalized or preserved |
| Case sensitivity | "USB" vs "usb" | Normalized consistently |
| Stop words | "the quick brown fox" | Removed or preserved |
| Long tokens | 200-character word | Truncated or rejected |

### 7. Performance Monitoring

**DO: Monitor analyzer impact on index size and query performance**

Custom analyzers can significantly affect system performance.

```python
# GOOD: Monitor analyzer metrics

import logging
from datetime import datetime

class AnalyzerPerformanceMonitor:
    """Monitor analyzer performance and index impact."""
    
    def __init__(self, index_client):
        self.index_client = index_client
    
    def analyze_with_metrics(self, text, analyzer_name):
        """Analyze text and measure performance."""
        start_time = datetime.now()
        
        result = self.index_client.analyze_text(
            AnalyzeTextOptions(text=text, analyzer_name=analyzer_name)
        )
        
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        tokens = [token.token for token in result.tokens]
        
        metrics = {
            'input_length': len(text),
            'token_count': len(tokens),
            'duration_ms': duration_ms,
            'tokens_per_second': len(tokens) / (duration_ms / 1000) if duration_ms > 0 else 0,
            'expansion_ratio': len(tokens) / len(text.split()) if text.split() else 0
        }
        
        # Log warning if slow
        if duration_ms > 100:
            logging.warning(f"Slow analyzer: {analyzer_name} took {duration_ms:.0f}ms")
        
        # Log warning if over-tokenization
        if metrics['expansion_ratio'] > 5.0:
            logging.warning(f"High token expansion: {metrics['expansion_ratio']:.1f}x")
        
        return tokens, metrics

# Usage
monitor = AnalyzerPerformanceMonitor(index_client)

sample_text = "The patient was diagnosed with MI and started on treatment"
tokens, metrics = monitor.analyze_with_metrics(sample_text, "medical_analyzer")

print(f"Tokens: {len(tokens)}")
print(f"Processing time: {metrics['duration_ms']:.1f}ms")
print(f"Expansion ratio: {metrics['expansion_ratio']:.1f}x")
```

**Performance Targets**:

| Metric | Target | Alert Threshold | Action if Exceeded |
|--------|--------|-----------------|-------------------|
| Processing time | <50ms per 1K tokens | >100ms | Simplify filter chain |
| Token count | <10 tokens per word | >15 tokens | Reduce n-gram range |
| Index size increase | <30% vs standard | >50% | Review synonym expansion |
| Query latency | <100ms P95 | >200ms | Move processing to index time |

**Index Size Impact** (1M documents, 500 words avg):

| Analyzer Configuration | Index Size | Storage Cost | Query Latency |
|----------------------|------------|--------------|---------------|
| Standard (baseline) | 2.5 GB | $50/month | 45ms |
| +Synonyms (5K terms) | 3.2 GB (+28%) | $64/month | 48ms |
| +Edge n-grams (2-10) | 4.8 GB (+92%) | $96/month | 52ms |
| +Aggressive stemming | 2.1 GB (-16%) | $42/month | 43ms |

### 8. Analyzer Versioning and Updates

**DO: Version analyzer names for safe updates**

Changing an analyzer requires reindexing. Version names to manage transitions.

```python
# GOOD: Versioned analyzer names

# Initial deployment
medical_analyzer_v1 = CustomAnalyzer(
    name="medical_analyzer_v1",
    tokenizer_name="standard",
    token_filters=["lowercase", "medical_synonyms_v1"]
)

# Create new version with improvements
medical_analyzer_v2 = CustomAnalyzer(
    name="medical_analyzer_v2",
    tokenizer_name="standard",
    token_filters=[
        "lowercase",
        "medical_synonyms_v2",  # Updated synonym map
        "porter_stem"  # Added stemming
    ]
)

# Deployment strategy:
# 1. Create v2 analyzer in index
# 2. Create new field with v2 analyzer
# 3. Reindex documents to populate new field
# 4. Test v2 field with subset of traffic
# 5. Switch traffic to v2 field
# 6. Deprecate v1 field after validation period
```

```python
# BAD: Changing analyzer in-place (breaks existing index)

# Attempt to modify existing analyzer
medical_analyzer = CustomAnalyzer(
    name="medical_analyzer",  # Same name
    # Changed configuration
)

# Problem: Can't update analyzer on existing index without full rebuild!
# Azure Search throws error: "Analyzer cannot be changed on an existing field"
```

**Safe Update Process**:

1. **Create new analyzer version**: `my_analyzer_v2`
2. **Add new field** with new analyzer to index schema
3. **Reindex documents** to populate new field
4. **A/B test** new analyzer with sample traffic
5. **Monitor metrics**: Compare precision, recall, latency
6. **Gradual rollout**: 10% → 50% → 100% traffic
7. **Deprecate old version** after validation period (30-90 days)

**Rollback Strategy**:
```python
# Keep both versions during transition
fields = [
    SearchField(name="description_v1", analyzer_name="my_analyzer_v1"),
    SearchField(name="description_v2", analyzer_name="my_analyzer_v2"),
]

# Query both fields with weighted scoring
search_client.search(
    search_text="query",
    search_fields=["description_v1^1.0", "description_v2^1.2"],  # Boost v2
    select=["id", "description_v1", "description_v2"]
)
```

---

## Troubleshooting

Common issues when implementing custom analyzers and their solutions.

### Issue 1: Analyzer Produces Too Many Tokens (Over-Tokenization)

**Symptoms**:
- Index size unexpectedly large (2-3× expected)
- Slow indexing performance
- High storage costs
- Query latency degradation

**Diagnosis**:

```python
# Check token count for sample documents
text = "Your sample document text here..."

result = index_client.analyze_text(
    AnalyzeTextOptions(text=text, analyzer_name="your_analyzer")
)

tokens = [token.token for token in result.tokens]
word_count = len(text.split())
expansion_ratio = len(tokens) / word_count

print(f"Input words: {word_count}")
print(f"Output tokens: {len(tokens)}")
print(f"Expansion ratio: {expansion_ratio:.1f}x")

if expansion_ratio > 5.0:
    print("⚠️  WARNING: Excessive tokenization!")
    print(f"Sample tokens: {tokens[:20]}")
```

**Root Causes**:

**1. Edge n-gram settings too broad**
```python
# BAD: Generates too many n-grams
edge_ngram = EdgeNGramTokenizer(
    name="autocomplete",
    min_gram=1,      # Too small
    max_gram=20,     # Too large
    token_chars=["letter", "digit", "punctuation"]  # Too permissive
)

# "search" (6 letters) generates:
# 1-gram: s, e, a, r, c, h (6 tokens)
# 2-gram: se, ea, ar, rc, ch (5 tokens)
# ... (continues through 20-gram)
# Total: ~100+ tokens for one word!
```

**Solution**:
```python
# GOOD: Constrained n-gram settings
edge_ngram = EdgeNGramTokenizer(
    name="autocomplete",
    min_gram=2,      # Minimum useful length
    max_gram=10,     # Reasonable maximum
    token_chars=["letter", "digit"]  # No punctuation
)

# "search" generates: se, sea, sear, searc, search (5 tokens) ✓
```

**2. Synonym expansion creating cascading matches**
```python
# BAD: Overly broad synonym expansion
synonyms = """
car, automobile, vehicle, transport, conveyance, motorcar, auto
vehicle, car, truck, van, SUV, automobile, transport
    """
# "car" expands to 12+ tokens!

# GOOD: Targeted synonym groups
synonyms = """
car, automobile
vehicle, transport
truck, lorry
    """
# "car" expands to 2 tokens ✓
```

**3. No maximum token length filter**
```python
# BAD: Allowing unlimited token length
analyzer = CustomAnalyzer(
    name="no_limit",
    tokenizer_name="standard"
    # No length filter - very long words become huge tokens
)

# GOOD: Add length limit
analyzer = CustomAnalyzer(
    name="with_limit",
    tokenizer_name="standard",
    token_filters=[
        LengthTokenFilter(
            name="length_limit",
            min_length=2,
            max_length=50  # Truncate very long tokens
        )
    ]
)
```

**Solutions Summary**:

| Root Cause | Fix | Impact |
|------------|-----|--------|
| Edge n-gram too broad | Set min_gram=2, max_gram=10 | -60% tokens |
| Excessive synonym expansion | Limit to 3-4 synonyms per term | -40% tokens |
| No length filter | Add max_length=50 filter | -15% tokens |
| Word delimiter + n-gram | Don't combine unless necessary | -70% tokens |

**Validation**:
```bash
# Check index statistics
# Target: <20 tokens per document on average

Total Tokens: 5,000,000
Total Documents: 100,000
Avg Tokens/Doc: 50 ✓ (acceptable)

# If >100 tokens/doc, investigate over-tokenization
```

### Issue 2: Expected Matches Not Found (Low Recall)

**Symptoms**:
- Users report relevant documents not appearing in results
- Specific product names or technical terms missed
- Queries with abbreviations return no results

**Diagnosis**:

```python
# Test specific failing query
query = "MI"  # Medical term abbreviation

# Check what tokens the query generates
query_result = index_client.analyze_text(
    AnalyzeTextOptions(text=query, analyzer_name="your_analyzer")
)
query_tokens = [token.token for token in query_result.tokens]

# Check what tokens the document generates
document = "Patient diagnosed with myocardial infarction"
doc_result = index_client.analyze_text(
    AnalyzeTextOptions(text=document, analyzer_name="your_analyzer")
)
doc_tokens = [token.token for token in doc_result.tokens]

print(f"Query tokens: {query_tokens}")
print(f"Document tokens: {doc_tokens}")

# Check for overlap
overlap = set(query_tokens) & set(doc_tokens)
print(f"Matching tokens: {overlap}")

if not overlap:
    print("❌ No matching tokens - documents won't be found!")
```

**Root Causes**:

**1. Different analyzers for index vs. query**
```python
# BAD: Mismatch between index and query analyzers
SearchField(
    name="description",
    analyzer_name="en.microsoft",        # Index analyzer
    search_analyzer_name="standard"      # Different query analyzer!
)

# Index: "running" → ["run"] (stemmed)
# Query: "running" → ["running"] (not stemmed)
# Result: No match! ❌
```

**Solution**:
```python
# GOOD: Use same analyzer (or compatible pair)
SearchField(
    name="description",
    analyzer_name="en.microsoft"  # Same for both index and query
)

# Or use compatible pair for autocomplete:
SearchField(
    name="title",
    analyzer_name="autocomplete_index",    # Edge n-grams
    search_analyzer_name="autocomplete_query"  # Standard
)
# Compatible because query tokens match subset of index tokens
```

**2. Missing synonyms**
```python
# Document contains: "myocardial infarction"
# User searches: "MI"
# No synonym map = no match

# Solution: Add synonym map
synonyms = SynonymMap(
    name="medical_synonyms",
    synonyms="MI, myocardial infarction, heart attack"
)

analyzer = CustomAnalyzer(
    name="medical_analyzer",
    tokenizer_name="standard",
    token_filters=[
        "lowercase",
        SynonymTokenFilter(
            name="medical_synonyms",
            synonyms_map_name="medical_synonyms",
            expand=True  # Bidirectional matching
        )
    ]
)
```

**3. Aggressive stemming or stop word removal**
```python
# BAD: Over-aggressive stop word removal
stop_words = StopwordsTokenFilter(
    name="aggressive_stops",
    stopwords=["a", "an", "the", "is", "at", "on", "in", "to", "of", "for"]
)

# Query: "care for elderly"
# After stop word removal: "care elderly"
# Document: "caring for the elderly" → "care elderly"
# Match: Maybe (if stemming aligns) ⚠️

# Document: "to care" → "care"
# Query: "to care" → "care"
# Different meaning, but matches! ❌
```

**Solution**:
```python
# GOOD: Use language-specific stop words (more conservative)
# or test with your query patterns first

analyzer = CustomAnalyzer(
    tokenizer_name="standard",
    token_filters=[
        "lowercase",
        LexicalTokenFilterName.STOPWORDS_ENGLISH,  # Language-specific
        "porter_stem"
    ]
)

# Better: Don't remove stop words for technical/medical content
```

**4. Case-sensitive tokenization when it shouldn't be**
```python
# BAD: No lowercase normalization
analyzer = CustomAnalyzer(
    tokenizer_name="standard"
    # No lowercase filter
)

# Query: "usb" → ["usb"]
# Document: "USB cable" → ["USB", "cable"]
# No match! ❌

# GOOD: Always lowercase (unless case matters)
analyzer = CustomAnalyzer(
    tokenizer_name="standard",
    token_filters=["lowercase"]
)

# Query: "usb" → ["usb"]
# Document: "USB cable" → ["usb", "cable"]
# Match! ✓
```

**Solutions Summary**:

| Root Cause | Fix | Test Command |
|------------|-----|--------------|
| Analyzer mismatch | Use same analyzer for index/query | Compare Analyze API output |
| Missing synonyms | Add synonym map with domain terms | Test abbreviations |
| Aggressive filtering | Use conservative stop word lists | Test specific queries |
| No case normalization | Add lowercase filter | Test with different cases |

**Validation Checklist**:
```python
# Run this for each critical query type
test_queries = ["MI", "heart attack", "COVID-19", "500mg"]

for query in test_queries:
    # 1. Check query tokenization
    query_tokens = analyze(query, analyzer="your_analyzer")
    
    # 2. Search for known relevant documents
    results = search(query, top=10)
    
    # 3. Verify known documents appear
    expected_doc_ids = ["doc1", "doc2", "doc3"]
    found_ids = [r.id for r in results]
    
    missing = set(expected_doc_ids) - set(found_ids)
    if missing:
        print(f"❌ Query '{query}' missing docs: {missing}")
```

### Issue 3: Index Build Fails After Adding Custom Analyzer

**Symptoms**:
- Error: "Analyzer configuration is invalid"
- Error: "Synonym map not found"
- Error: "Circular dependency detected"
- Index creation/update fails

**Diagnosis**:

```python
# Capture full error message
try:
    index_client.create_or_update_index(index)
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print(f"Error details: {e.__dict__}")
```

**Root Causes**:

**1. Synonym map doesn't exist yet**
```python
# BAD: Reference synonym map before creating it
analyzer = CustomAnalyzer(
    name="my_analyzer",
    token_filters=[
        SynonymTokenFilter(
            synonyms_map_name="my_synonyms"  # Doesn't exist yet!
        )
    ]
)

index = SearchIndex(name="myindex", fields=fields, analyzers=[analyzer])
index_client.create_index(index)  # ❌ Fails!
```

**Solution**:
```python
# GOOD: Create synonym map first
synonym_map = SynonymMap(
    name="my_synonyms",
    synonyms="word1, word2\nword3, word4"
)
index_client.create_or_update_synonym_map(synonym_map)  # Create first

# Then create analyzer that references it
analyzer = CustomAnalyzer(
    name="my_analyzer",
    token_filters=[
        SynonymTokenFilter(synonyms_map_name="my_synonyms")  # Now exists ✓
    ]
)

index = SearchIndex(name="myindex", fields=fields, analyzers=[analyzer])
index_client.create_index(index)  # ✓ Works
```

**2. Invalid regex pattern in pattern tokenizer**
```python
# BAD: Invalid regex
pattern_tokenizer = PatternTokenizer(
    name="invalid_pattern",
    pattern=r"[a-z"  # Unclosed bracket - invalid regex!
)

# Error: "Invalid regular expression pattern"
```

**Solution**:
```python
# GOOD: Validate regex before using
import re

pattern = r"[a-z0-9]+"

try:
    re.compile(pattern)
    print(f"✓ Pattern '{pattern}' is valid")
except re.error as e:
    print(f"❌ Invalid pattern: {e}")

# Use validated pattern
pattern_tokenizer = PatternTokenizer(
    name="valid_pattern",
    pattern=pattern
)
```

**3. Circular filter dependencies**
```python
# BAD: Analyzer references itself (edge case)
# This can happen with complex filter chains

analyzer1 = CustomAnalyzer(
    name="analyzer1",
    tokenizer_name="standard",
    char_filters=["filter_a"]
)

# If filter_a somehow depends on analyzer1 output...
# Creates circular dependency
```

**Solution**:
```python
# GOOD: Keep dependencies linear
# char_filters → tokenizer → token_filters
# No backwards references

analyzer = CustomAnalyzer(
    name="clean_analyzer",
    char_filters=["html_strip"],           # Step 1
    tokenizer_name="standard",              # Step 2
    token_filters=["lowercase", "stop"]     # Step 3
)
```

**4. Analyzer name conflicts**
```python
# BAD: Custom analyzer name same as built-in
analyzer = CustomAnalyzer(
    name="standard",  # ❌ Conflicts with built-in!
    tokenizer_name="standard"
)

# Error: "Analyzer name 'standard' is reserved"
```

**Solution**:
```python
# GOOD: Use unique prefixed names
analyzer = CustomAnalyzer(
    name="custom_standard_v1",  # ✓ Unique name
    tokenizer_name="standard"
)
```

**Solutions Summary**:

| Error | Root Cause | Fix |
|-------|-----------|-----|
| "Synonym map not found" | Create analyzer before synonym map | Create synonym map first |
| "Invalid regular expression" | Bad regex in pattern tokenizer | Test regex with `re.compile()` |
| "Circular dependency" | Filter references itself | Keep dependencies linear |
| "Analyzer name reserved" | Name conflicts with built-in | Use custom prefix |
| "Field cannot be modified" | Changing analyzer on existing field | Create new field with new analyzer |

**Deployment Checklist**:
```python
def safe_analyzer_deployment(index_client, synonym_maps, analyzers, index):
    """
    Deploy custom analyzers safely with proper error handling.
    """
    try:
        # Step 1: Create synonym maps first
        for synonym_map in synonym_maps:
            print(f"Creating synonym map: {synonym_map.name}")
            index_client.create_or_update_synonym_map(synonym_map)
        
        # Step 2: Validate analyzer configuration
        for analyzer in analyzers:
            print(f"Validating analyzer: {analyzer.name}")
            # Check for reserved names
            reserved = ["standard", "keyword", "simple", "stop", "whitespace"]
            if analyzer.name in reserved:
                raise ValueError(f"Analyzer name '{analyzer.name}' is reserved")
            
            # Validate regex patterns if any
            if hasattr(analyzer, 'tokenizer') and hasattr(analyzer.tokenizer, 'pattern'):
                try:
                    re.compile(analyzer.tokenizer.pattern)
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern: {e}")
        
        # Step 3: Create/update index with analyzers
        print(f"Creating index: {index.name}")
        index_client.create_or_update_index(index)
        
        print("✅ Deployment successful")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        # Rollback if needed
        raise
```

### Issue 4: Search Performance Degraded After Adding Custom Analyzer

**Symptoms**:
- Query latency increased significantly (2-5× slower)
- High CPU usage during searches
- Slow indexing operations

**Diagnosis**:

```python
# Measure query performance before/after
import time

def measure_query_performance(search_client, query, iterations=100):
    """Measure average query latency."""
    latencies = []
    
    for _ in range(iterations):
        start = time.time()
        results = list(search_client.search(query, top=10))
        end = time.time()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
    
    return {
        'avg_ms': avg_latency,
        'p95_ms': p95_latency,
        'min_ms': min(latencies),
        'max_ms': max(latencies)
    }

# Test with custom analyzer
perf = measure_query_performance(search_client, "test query")
print(f"Average latency: {perf['avg_ms']:.1f}ms")
print(f"P95 latency: {perf['p95_ms']:.1f}ms")

# If >100ms avg, investigate analyzer complexity
```

**Root Causes**:

**1. Expensive filters applied at query time**
```python
# BAD: Complex processing at query time
SearchField(
    name="description",
    search_analyzer_name="complex_query_analyzer"  # Used for EVERY query
)

complex_query_analyzer = CustomAnalyzer(
    name="complex_query_analyzer",
    tokenizer_name="standard",
    token_filters=[
        "lowercase",
        "synonym_expansion",     # 5K synonyms expanded per query
        "edge_ngram",            # N-grams generated per query
        "phonetic_matching",     # Expensive phonetic algorithm
        "custom_stemmer"
    ]
)

# Problem: All this processing happens for every query!
# 1000 queries/sec × 50ms processing = 50 seconds of CPU per second!
```

**Solution**:
```python
# GOOD: Move expensive processing to index time
SearchField(
    name="description",
    analyzer_name="complex_index_analyzer",      # Used once at index time
    search_analyzer_name="simple_query_analyzer"  # Fast query processing
)

# Index analyzer (expensive, but only runs once per document)
complex_index_analyzer = CustomAnalyzer(
    name="complex_index_analyzer",
    tokenizer_name="standard",
    token_filters=[
        "lowercase",
        "synonym_expansion",
        "edge_ngram",
        "phonetic_matching"
    ]
)

# Query analyzer (simple and fast)
simple_query_analyzer = CustomAnalyzer(
    name="simple_query_analyzer",
    tokenizer_name="standard",
    token_filters=["lowercase"]  # Only lowercase normalization
)

# Result: Query processing drops from 50ms to 5ms ✓
```

**2. Over-tokenization causing large posting lists**
```python
# BAD: Too many tokens = large inverted index
# Document: "search functionality"
# With edge n-grams (1-10):
# "search" → s, se, sea, sear, searc, search (6 tokens)
# "functionality" → f, fu, fun, func, ... functionality (14 tokens)
# Total: 20 tokens for 2 words!

# Problem: Every query must scan huge posting lists
```

**Solution**:
```python
# GOOD: Constrain tokenization
edge_ngram = EdgeNGramTokenizer(
    name="constrained_ngram",
    min_gram=2,      # Skip 1-character tokens
    max_gram=8,      # Reasonable max
    token_chars=["letter"]  # No digits/punctuation
)

# "search" → se, sea, sear, searc, search (5 tokens vs 6)
# "functionality" → fu, fun, func, funct, functio, function (6 tokens vs 14)
# Total: 11 tokens vs 20 (45% reduction) ✓
```

**3. Complex regex patterns in character filters**
```python
# BAD: Complex regex evaluated for every character
complex_pattern = PatternReplaceCharFilter(
    name="complex_regex",
    pattern=r"(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+",  # URL regex
    replacement=""
)

# Problem: Complex regex on large documents is slow
# 10KB document × complex regex = 100ms+ processing time
```

**Solution**:
```python
# GOOD: Simpler regex or pre-processing
simple_pattern = PatternReplaceCharFilter(
    name="simple_regex",
    pattern=r"http\S+",  # Simpler URL match
    replacement=""
)

# Or pre-process documents before indexing
# Remove URLs in application code (faster)
```

**4. Sequential filter execution (no optimization)**
```python
# While you can't parallelize filters, you can optimize order:

# BAD order (processes all tokens through expensive filters)
analyzer = CustomAnalyzer(
    tokenizer_name="standard",
    token_filters=[
        "synonym_expansion",     # Expensive, runs on all tokens
        "lowercase",
        "stop",                  # Removes tokens
        "stem"
    ]
)

# GOOD order (reduce token count early)
analyzer = CustomAnalyzer(
    tokenizer_name="standard",
    token_filters=[
        "lowercase",             # Fast
        "stop",                  # Removes tokens early
        "synonym_expansion",     # Fewer tokens to expand
        "stem"
    ]
)
```

**Performance Optimization Summary**:

| Optimization | Latency Improvement | Implementation |
|--------------|---------------------|----------------|
| Move processing to index time | -40-60ms | Use different index/query analyzers |
| Constrain n-gram generation | -20-30ms | Set min_gram=2, max_gram=8 |
| Simplify regex patterns | -10-50ms | Use simpler patterns or pre-process |
| Optimize filter order | -5-15ms | Remove tokens early in chain |
| Limit synonym expansion | -10-20ms | Cap at 3-4 synonyms per term |

**Target Performance** (P95 latency):
- Simple queries: <50ms
- Medium complexity: <100ms
- Complex queries: <200ms

If exceeding these targets, profile and optimize analyzer configuration.

### Issue 5: Can't Update Analyzer Without Reindexing

**Symptoms**:
- Error: "Analyzer cannot be changed on an existing field"
- Need to modify analyzer behavior but can't change it
- Want to add/remove filters from production analyzer

**Diagnosis**:

This is by design - Azure AI Search doesn't allow analyzer changes on existing fields to prevent index corruption.

**Root Cause**:

```python
# Attempt to update analyzer
existing_analyzer = CustomAnalyzer(
    name="my_analyzer",
    tokenizer_name="standard",
    token_filters=["lowercase"]
)

# Try to add a filter
updated_analyzer = CustomAnalyzer(
    name="my_analyzer",  # Same name
    tokenizer_name="standard",
    token_filters=["lowercase", "stop"]  # Added stop filter
)

# Update index with new analyzer
index = index_client.get_index("myindex")
index.analyzers = [updated_analyzer]
index_client.create_or_update_index(index)

# ❌ Error: "Cannot modify analyzer 'my_analyzer' because it is used by field 'description'"
```

**Solutions**:

**Option 1: Create new analyzer version and field (Recommended)**

```python
# Step 1: Create new analyzer version
analyzer_v2 = CustomAnalyzer(
    name="my_analyzer_v2",  # New version name
    tokenizer_name="standard",
    token_filters=["lowercase", "stop"]
)

# Step 2: Add new field with new analyzer
new_field = SearchField(
    name="description_v2",  # New field name
    type=SearchFieldDataType.String,
    searchable=True,
    analyzer_name="my_analyzer_v2"
)

# Step 3: Update index schema
index = index_client.get_index("myindex")
index.analyzers.append(analyzer_v2)
index.fields.append(new_field)
index_client.create_or_update_index(index)

# Step 4: Reindex documents to populate new field
# (Can happen incrementally in background)

# Step 5: Update queries to use new field
results = search_client.search(
    search_text="query",
    search_fields=["description_v2"],  # Use new field
    select=["id", "description_v2"]
)

# Step 6: Deprecate old field after validation
```

**Option 2: Create shadow index (Zero-downtime migration)**

```python
# Step 1: Create new index with updated analyzer
new_index = SearchIndex(
    name="myindex_v2",  # New index name
    fields=[
        SearchField(
            name="description",
            analyzer_name="my_analyzer_v2"  # Updated analyzer
        )
    ],
    analyzers=[analyzer_v2]
)

index_client.create_index(new_index)

# Step 2: Reindex documents to new index
# (Use indexer or bulk upload)

# Step 3: Switch alias or update application config
# to point to new index

# Step 4: Validate new index works correctly

# Step 5: Delete old index
index_client.delete_index("myindex")
```

**Option 3: Full reindex (Simplest but downtime)**

```python
# Step 1: Delete and recreate index with new analyzer
index_client.delete_index("myindex")

new_index = SearchIndex(
    name="myindex",
    fields=[...],
    analyzers=[updated_analyzer]
)

index_client.create_index(new_index)

# Step 2: Reindex all documents
# (Application is down during this process)

# ⚠️ Use only for dev/test or off-hours maintenance
```

**Decision Matrix**:

| Approach | Downtime | Complexity | Rollback | Best For |
|----------|----------|------------|----------|----------|
| New field version | None | Medium | Easy | Production (recommended) |
| Shadow index | None | High | Easy | Large indexes, critical apps |
| Full reindex | Yes | Low | Hard | Dev/test, small indexes |

**Prevention Strategy**:

```python
# Design for future changes from the start

# 1. Version analyzer names
analyzer = CustomAnalyzer(
    name="my_analyzer_v1",  # Include version
    ...
)

# 2. Use synonym maps (can update without reindex)
synonym_map = SynonymMap(
    name="my_synonyms_v1",
    synonyms="..."
)
# Synonym maps can be updated without full reindex!

# 3. Test analyzer thoroughly before production
# Run full test suite before deploying

# 4. Plan for migrations
# Document procedure for analyzer updates in runbook
```

**Synonym Map Updates (No Reindex Required)**:

```python
# ✅ CAN update synonym maps without reindex
updated_synonyms = SynonymMap(
    name="my_synonyms_v1",
    synonyms="new,synonyms\nhere"  # Updated content
)

index_client.create_or_update_synonym_map(updated_synonyms)

# Changes take effect immediately for new queries
# No reindex required! ✓
```

---

## Next Steps

- **[Scoring Profiles](./14-scoring-profiles.md)** - Advanced relevance tuning
- **[Index Management](./15-index-management.md)** - Managing analyzer updates
- **[Query Optimization](./12-query-optimization.md)** - Analyzer performance impact

---

*See also: [Full-Text Search](./08-fulltext-search-bm25.md) | [Hybrid Search](./10-hybrid-search.md)*