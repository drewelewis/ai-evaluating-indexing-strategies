# Scoring Profiles

Complete guide to advanced relevance tuning in Azure AI Search using scoring profiles for field weighting, freshness, magnitude, and distance functions.

## 📋 Table of Contents
- [Overview](#overview)
- [Scoring Profile Fundamentals](#scoring-profile-fundamentals)
- [Text Weights](#text-weights)
- [Functions](#functions)
- [Implementation](#implementation)
- [Advanced Patterns](#advanced-patterns)
- [Evaluation](#evaluation)
- [Best Practices](#best-practices)

---

## Overview

### What are Scoring Profiles?

Scoring profiles are Azure AI Search's mechanism for customizing relevance ranking beyond the default BM25 algorithm. They allow you to boost documents based on business rules, user preferences, and contextual signals—transforming generic search results into precisely tuned experiences that align with your specific requirements.

**Core Concept**: While BM25 provides excellent baseline relevance based on term frequency and document statistics, scoring profiles let you incorporate domain-specific knowledge like "recent articles are more valuable" or "products with high ratings should rank higher" or "results near the user's location are preferred."

### Real-World Impact: TechNews360

**Company Profile**: TechNews360 is a technology news aggregator serving 2.5 million monthly readers with 500K articles from 10,000+ sources. Their search handles 8 million queries per month, with users expecting the most relevant, recent, and authoritative content.

**The Challenge**: Default BM25 Ranking Falls Short

With standard Azure AI Search configuration, TechNews360 faced significant user satisfaction issues:

**Baseline Search Performance** (BM25 only):
- **Outdated Results**: 6-month-old articles ranking above yesterday's breaking news
- **Poor Source Authority**: Blog posts outranking Reuters/AP articles due to keyword density
- **Low User Engagement**:
  - Click-through rate (CTR): 12% on first result
  - Average session duration: 2.3 minutes
  - Bounce rate: 58%
  - User satisfaction: 2.8/5 stars
- **Business Impact**: 18% decline in return visitors month-over-month

**Root Cause Analysis**:

```python
# Example: Query "iPhone 15 review"
# BM25 ranking (default):

# Rank 1: Blog post from 8 months ago (high keyword density)
{
    'title': 'iPhone 15 Pro Max Complete Review Guide Tips Tricks',
    'source': 'techblog99.com',
    'published': '2024-03-15',  # 8 months old
    'author_reputation': 2.1/5,
    'bm25_score': 12.4,
    'reason': 'Keyword stuffing: "review" appears 47 times'
}

# Rank 2: Official Apple press release (low keyword match)
{
    'title': 'Apple Announces iPhone 15',
    'source': 'apple.com',
    'published': '2024-11-01',  # Recent
    'author_reputation': 5.0/5,
    'bm25_score': 8.7,
    'reason': 'Official but missing "review" keyword'
}

# Rank 3: Yesterday's professional review (perfect match)
{
    'title': 'iPhone 15 Pro Review: One Month Later',
    'source': 'theverge.com',
    'published': '2024-11-06',  # Yesterday!
    'author_reputation': 4.8/5,
    'bm25_score': 11.2,
    'reason': 'Recent, authoritative, but ranked #3'
}

# Problem: User wants #3, gets #1
```

**The Solution: Multi-Signal Scoring Profile**

TechNews360 implemented a comprehensive scoring profile combining five signals:

**Phase 1: Text Weights (Field-Level Boosting)**

```python
# Title matches are 3× more valuable than body text
text_weights = TextWeights(weights={
    "title": 3.0,        # Headline matches critical
    "summary": 1.5,      # Abstract matches valuable
    "body": 1.0,         # Body matches baseline
    "tags": 2.0          # Tag matches important
})

# Results:
# - Title keyword matches jump from position 5-8 to position 1-3
# - Precision@5 improved 34% (0.41 → 0.55)
```

**Phase 2: Freshness Function (Recency Boost)**

```python
# Boost articles published in last 7 days
freshness_function = FreshnessScoringFunction(
    field_name="publishedDate",
    boost=5.0,                    # Strong freshness signal
    parameters=FreshnessScoringParameters(
        boosting_duration="P7D"    # 7-day window
    ),
    interpolation=ScoringFunctionInterpolation.LINEAR
)

# Impact:
# - Yesterday's articles: +4.5 score boost (90% of max)
# - 3-day-old articles: +3.2 score boost (64% of max)
# - 7-day-old articles: +0.5 score boost (10% of max)
# - 8+ day-old articles: No freshness boost

# Results:
# - Average result age: 45 days → 3 days
# - Breaking news CTR: 12% → 38% (+217%)
```

**Phase 3: Source Authority (Magnitude Boost)**

```python
# Boost high-authority sources (verified journalists, major outlets)
authority_function = MagnitudeScoringFunction(
    field_name="authorReputation",
    boost=3.0,
    parameters=MagnitudeScoringParameters(
        boosting_range_start=4.0,   # Start boost at 4.0/5
        boosting_range_end=5.0,     # Max boost at 5.0/5
        constant_boost_beyond_range=False
    ),
    interpolation=ScoringFunctionInterpolation.LINEAR
)

# Authority scores:
# - Major outlets (NYT, Reuters, AP): 5.0/5 → +3.0 boost
# - Verified tech journalists: 4.5/5 → +1.5 boost
# - Established blogs: 3.5/5 → No boost
# - Unverified sources: 2.0/5 → No boost

# Results:
# - Authoritative sources in top 5: 32% → 78%
# - User trust score: 2.8/5 → 4.1/5 (+46%)
```

**Phase 4: Engagement Magnitude (Popularity Signal)**

```python
# Boost articles with high engagement (views, shares)
engagement_function = MagnitudeScoringFunction(
    field_name="engagementScore",  # Composite: views + shares + comments
    boost=2.0,
    parameters=MagnitudeScoringParameters(
        boosting_range_start=1000.0,   # 1K engagement
        boosting_range_end=10000.0,    # 10K engagement
        constant_boost_beyond_range=True  # Cap at 10K
    ),
    interpolation=ScoringFunctionInterpolation.LOGARITHMIC  # Diminishing returns
)

# Engagement tiers:
# - Viral (10K+): +2.0 boost (log scale caps runaway growth)
# - Popular (5K): +1.5 boost
# - Moderate (1K): +0.3 boost
# - Low (<1K): No boost

# Results:
# - Popular content visibility: +67%
# - Session duration: 2.3 → 4.8 minutes (+109%)
```

**Phase 5: Tag Matching (Personalization)**

```python
# Boost content matching user's topic preferences
tag_function = TagScoringFunction(
    field_name="topicTags",
    boost=2.5,
    parameters=TagScoringParameters(
        tags_parameter="userPreferences"  # Pass user's interests
    )
)

# Example: User interested in "AI", "Machine Learning", "Python"
# - Article tagged ["AI", "Deep Learning"]: +2.5 boost (1 match)
# - Article tagged ["AI", "Python", "Tutorial"]: +5.0 boost (2 matches)
# - Article tagged ["JavaScript", "React"]: No boost (0 matches)

# Results:
# - Personalized relevance: +52%
# - Return visitor rate: +41%
```

**Complete Scoring Profile**:

```python
tech_news_profile = ScoringProfile(
    name="tech_news_relevance_v2",
    text_weights=text_weights,
    functions=[
        freshness_function,      # Recency
        authority_function,      # Source credibility
        engagement_function,     # Popularity
        tag_function            # Personalization
    ],
    function_aggregation=ScoringFunctionAggregation.SUM
)

# Scoring calculation example:
# Query: "iPhone 15 review"
# Document: Verge article from yesterday

base_score = 11.2          # BM25 (good keyword match)
title_weight = +3.0        # "review" in title
freshness = +4.5           # Published yesterday (7-day window)
authority = +3.0           # The Verge (5.0/5 reputation)
engagement = +1.5          # 5K views (popular)
tag_match = +2.5           # User interested in "smartphones"

final_score = 11.2 + 3.0 + 4.5 + 3.0 + 1.5 + 2.5 = 25.7

# vs. Old blog post:
base_score = 12.4          # BM25 (keyword stuffing)
title_weight = +3.0        # "review" in title
freshness = +0.0           # 8 months old (no boost)
authority = +0.0           # Low-reputation blog (2.1/5)
engagement = +0.0          # 200 views (below threshold)
tag_match = +0.0           # Generic tech content

final_score = 12.4 + 3.0 = 15.4

# Result: Recent authoritative article (25.7) beats keyword-stuffed blog (15.4) ✓
```

**Business Results** (3 months post-deployment):

**User Engagement**:
- **Click-through rate**: 12% → 34% (+183%)
- **Time on site**: 2.3 → 4.8 minutes (+109%)
- **Bounce rate**: 58% → 31% (-47%)
- **Pages per session**: 1.8 → 3.4 (+89%)

**Search Quality**:
- **User satisfaction**: 2.8/5 → 4.4/5 (+57%)
- **Zero-result rate**: 8% → 2% (-75%)
- **Refinement rate**: 34% → 18% (-47% - users find what they need faster)
- **Precision@5**: 0.41 → 0.73 (+78%)
- **NDCG@10**: 0.58 → 0.84 (+45%)

**Business Impact**:
- **Return visitors**: -18% MoM → +28% MoM (46-point swing)
- **Ad revenue**: $420K/month → $680K/month (+62% from engagement)
- **Premium subscriptions**: 18K → 31K (+72% - better content discovery drives conversions)
- **Annual revenue impact**: +$3.8M/year
- **Implementation cost**: $28,000 (profile development, testing, tuning)
- **ROI**: 13,471% first year, 4.2-day payback period

**Key Learnings**:

1. **Freshness Matters**: 7-day boost window perfect for tech news (vs. 30 days tested initially)
2. **Authority Scaling**: Linear interpolation (4.0-5.0 range) prevented low-quality sources from gaming system
3. **Engagement Caps**: Logarithmic scaling for engagement prevented viral-but-low-quality content from dominating
4. **Aggregation Strategy**: SUM aggregation worked better than AVERAGE—multiple positive signals should compound
5. **A/B Testing Critical**: 15% of users on new profile for 2 weeks before full rollout validated improvements

**Scoring Profile Components Explained**:

```python
class ScoringProfileOverview:
    """Understanding scoring profiles in Azure AI Search."""
    
    @staticmethod
    def explain_scoring_profiles():
        """
        Scoring profiles customize relevance ranking by:
        - Boosting specific fields (title > description)
        - Applying freshness decay (newer = higher score)
        - Boosting by magnitude (rating, price range)
        - Distance-based boosting (geo-search)
        - Combining multiple signals
        """
        return {
            'purpose': 'Customize BM25 relevance scores',
            'when_applied': 'After initial BM25 scoring, before return',
            'components': {
                'text_weights': 'Boost specific searchable fields',
                'functions': 'Apply mathematical transformations (freshness, magnitude, distance, tags)',
                'function_aggregation': 'Combine multiple function scores'
            },
            'use_cases': [
                'Boost title matches over body matches',
                'Favor recent documents (news, posts)',
                'Promote highly-rated products',
                'Prioritize items within price range',
                'Boost geo-proximity results',
                'Combine multiple ranking signals'
            ]
        }
    
    @staticmethod
    def scoring_flow():
        """
        Scoring flow with profiles:
        
        1. BM25 base score calculated
        2. Scoring profile applied:
           a. Text weights applied to field matches
           b. Functions evaluated (freshness, magnitude, etc.)
           c. Function scores aggregated
        3. Final score = base_score + profile_boost
        4. Results sorted by final score
        """
        return {
            'step_1': 'BM25 base score (0 to ~10+)',
            'step_2': 'Apply text weights to field matches',
            'step_3': 'Calculate function scores',
            'step_4': 'Aggregate functions (sum, average, min, max)',
            'step_5': 'Final score = base + (weights × functions)',
            'step_6': 'Sort results by final score'
        }

# Usage
overview = ScoringProfileOverview()
explained = overview.explain_scoring_profiles()

print("Scoring Profiles:")
print(f"Purpose: {explained['purpose']}")
print(f"\nComponents:")
for component, description in explained['components'].items():
    print(f"  {component}: {description}")

print(f"\nUse cases:")
for use_case in explained['use_cases']:
    print(f"  • {use_case}")
```

---

## Scoring Profile Fundamentals

### Score Calculation

```python
class ScoringMath:
    """Understanding scoring calculations."""
    
    @staticmethod
    def base_score_explained():
        """
        BM25 base score calculation.
        
        Base = BM25(query, document)
        - Considers term frequency
        - Inverse document frequency
        - Document length normalization
        - Field-level scoring
        """
        return {
            'algorithm': 'BM25 (Best Match 25)',
            'range': '0 to unbounded (typically 0-20)',
            'factors': [
                'Term frequency in document',
                'Inverse document frequency (rarity)',
                'Document length',
                'Field where match occurs'
            ]
        }
    
    @staticmethod
    def profile_boost_formula():
        """
        Scoring profile boost formula.
        
        Final Score = Base Score + Profile Boost
        
        Profile Boost = (Text Weight Boost) + (Function Boost)
        
        Function Boost = Aggregation(
            function_1_score × boost_1,
            function_2_score × boost_2,
            ...
        )
        """
        return {
            'final_score': 'base_score + profile_boost',
            'profile_boost': 'text_weights + function_boost',
            'function_boost': 'aggregation(f1×b1, f2×b2, ...)',
            'aggregation_types': ['sum', 'average', 'minimum', 'maximum', 'firstMatching']
        }
    
    @staticmethod
    def example_calculation():
        """Example scoring calculation."""
        base_score = 5.2  # BM25 score
        
        # Text weights: title matched (weight=3.0)
        title_boost = 3.0
        
        # Freshness function: 7 days old, 30-day boost
        freshness_score = 0.7  # Normalized score
        freshness_boost = 2.0
        
        # Magnitude function: rating 4.5/5
        magnitude_score = 0.9  # Normalized score
        magnitude_boost = 1.5
        
        # Aggregate functions (sum)
        function_boost = (freshness_score * freshness_boost) + \
                        (magnitude_score * magnitude_boost)
        
        # Combined boost
        profile_boost = title_boost + function_boost
        
        # Final score
        final_score = base_score + profile_boost
        
        return {
            'base_score': base_score,
            'title_boost': title_boost,
            'freshness_contribution': freshness_score * freshness_boost,
            'magnitude_contribution': magnitude_score * magnitude_boost,
            'total_function_boost': function_boost,
            'total_profile_boost': profile_boost,
            'final_score': final_score,
            'breakdown': {
                'base': f'{base_score} (BM25)',
                'title': f'+{title_boost} (text weight)',
                'freshness': f'+{freshness_score * freshness_boost:.2f} (function)',
                'magnitude': f'+{magnitude_score * magnitude_boost:.2f} (function)',
                'final': f'= {final_score:.2f}'
            }
        }

# Usage
scoring_math = ScoringMath()

example = scoring_math.example_calculation()
print("Score Calculation Example:")
print(f"Base Score: {example['base_score']}")
print(f"Title Boost: +{example['title_boost']}")
print(f"Freshness: +{example['freshness_contribution']:.2f}")
print(f"Magnitude: +{example['magnitude_contribution']:.2f}")
print(f"Final Score: {example['final_score']:.2f}")

print("\nBreakdown:")
for component, value in example['breakdown'].items():
    print(f"  {component}: {value}")
```

---

## Text Weights

### Field Weighting

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    ScoringProfile,
    TextWeights
)
from azure.core.credentials import AzureKeyCredential

class TextWeightProfiles:
    """Implement text weight scoring profiles."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_basic_weighted_index(self, index_name):
        """
        Create index with basic text weighting.
        
        Weights:
        - title: 3.0 (highest priority)
        - tags: 2.0 (medium priority)
        - description: 1.0 (normal priority)
        """
        # Define fields
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="description", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True),
            SearchField(name="category", type=SearchFieldDataType.String, filterable=True)
        ]
        
        # Define text weights
        text_weights = TextWeights(
            weights={
                "title": 3.0,
                "tags": 2.0,
                "description": 1.0
            }
        )
        
        # Create scoring profile
        scoring_profile = ScoringProfile(
            name="weighted_fields",
            text_weights=text_weights
        )
        
        # Create index with profile
        index = SearchIndex(
            name=index_name,
            fields=fields,
            scoring_profiles=[scoring_profile]
        )
        
        result = self.index_client.create_or_update_index(index)
        print(f"Created index with text weights scoring profile")
        return result
    
    def create_multi_profile_index(self, index_name):
        """
        Create index with multiple scoring profiles.
        
        Different profiles for different scenarios:
        - general: Balanced weights
        - title_focused: Heavy title emphasis
        - content_focused: Heavy description emphasis
        """
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="description", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True)
        ]
        
        # Profile 1: Balanced
        balanced_profile = ScoringProfile(
            name="balanced",
            text_weights=TextWeights(weights={
                "title": 2.0,
                "description": 1.5,
                "content": 1.0,
                "tags": 1.5
            })
        )
        
        # Profile 2: Title-focused
        title_profile = ScoringProfile(
            name="title_focused",
            text_weights=TextWeights(weights={
                "title": 5.0,
                "description": 1.0,
                "content": 0.5,
                "tags": 1.0
            })
        )
        
        # Profile 3: Content-focused
        content_profile = ScoringProfile(
            name="content_focused",
            text_weights=TextWeights(weights={
                "title": 1.0,
                "description": 2.0,
                "content": 3.0,
                "tags": 1.0
            })
        )
        
        # Create index with multiple profiles
        index = SearchIndex(
            name=index_name,
            fields=fields,
            scoring_profiles=[balanced_profile, title_profile, content_profile],
            default_scoring_profile="balanced"  # Default profile
        )
        
        result = self.index_client.create_or_update_index(index)
        return result

# Usage
import os

text_weight_setup = TextWeightProfiles(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Create weighted index
weighted_index = text_weight_setup.create_basic_weighted_index("products-weighted")
print("Created index with text weight scoring profile")

# Create multi-profile index
multi_index = text_weight_setup.create_multi_profile_index("content-multi-profile")
print(f"Created index with {len(multi_index.scoring_profiles)} scoring profiles")
```

### Using Text Weight Profiles

```python
from azure.search.documents import SearchClient

class TextWeightSearch:
    """Search using text weight profiles."""
    
    def __init__(self, search_endpoint, index_name, search_key):
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )
    
    def search_with_profile(self, query_text, scoring_profile=None, top=10):
        """
        Execute search with scoring profile.
        
        Args:
            query_text: Search query
            scoring_profile: Name of scoring profile to use
            top: Number of results
        """
        results = self.search_client.search(
            search_text=query_text,
            scoring_profile=scoring_profile,
            top=top
        )
        
        documents = []
        for result in results:
            documents.append({
                'id': result['id'],
                'title': result.get('title'),
                'score': result['@search.score']
            })
        
        return documents
    
    def compare_profiles(self, query_text, profile_names, top=10):
        """
        Compare results across different scoring profiles.
        """
        comparisons = {}
        
        # Default (no profile)
        comparisons['default'] = self.search_with_profile(query_text, None, top)
        
        # Each profile
        for profile_name in profile_names:
            comparisons[profile_name] = self.search_with_profile(
                query_text,
                profile_name,
                top
            )
        
        return comparisons

# Usage
searcher = TextWeightSearch(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    index_name="content-multi-profile",
    search_key=os.getenv("SEARCH_API_KEY")
)

# Search with specific profile
results = searcher.search_with_profile(
    "machine learning",
    scoring_profile="title_focused",
    top=5
)

print("Results with title_focused profile:")
for doc in results:
    print(f"  {doc['title']} - Score: {doc['score']:.4f}")

# Compare profiles
comparison = searcher.compare_profiles(
    "machine learning",
    ["balanced", "title_focused", "content_focused"],
    top=5
)

print("\nProfile Comparison:")
for profile_name, results in comparison.items():
    print(f"\n{profile_name}:")
    for doc in results[:3]:
        print(f"  {doc['title']} ({doc['score']:.4f})")
```

---

## Functions

### Freshness Function

```python
from azure.search.documents.indexes.models import (
    ScoringProfile,
    FreshnessScoringFunction,
    FreshnessScoringParameters,
    ScoringFunctionAggregation,
    ScoringFunctionInterpolation
)

class FreshnessProfiles:
    """Implement freshness-based scoring."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_freshness_profile(self, index_name):
        """
        Create scoring profile with freshness function.
        
        Boosts recent documents:
        - Documents within 30 days get full boost
        - Older documents get reduced boost (linear decay)
        """
        from datetime import datetime
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="publishedDate", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SearchField(name="category", type=SearchFieldDataType.String, filterable=True)
        ]
        
        # Freshness function
        freshness_function = FreshnessScoringFunction(
            field_name="publishedDate",
            boost=5.0,  # Maximum boost multiplier
            parameters=FreshnessScoringParameters(
                boosting_duration="P30D"  # 30 days (ISO 8601 duration)
            ),
            interpolation=ScoringFunctionInterpolation.LINEAR
        )
        
        # Scoring profile with freshness
        freshness_profile = ScoringProfile(
            name="freshness_boost",
            functions=[freshness_function],
            function_aggregation=ScoringFunctionAggregation.SUM
        )
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            scoring_profiles=[freshness_profile]
        )
        
        return self.index_client.create_or_update_index(index)
    
    def create_combined_freshness_profile(self, index_name):
        """
        Combine freshness with text weights.
        
        - Title weight: 3.0
        - Freshness boost for recent items
        """
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="publishedDate", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True)
        ]
        
        # Text weights
        text_weights = TextWeights(weights={
            "title": 3.0,
            "content": 1.0
        })
        
        # Freshness function
        freshness_function = FreshnessScoringFunction(
            field_name="publishedDate",
            boost=3.0,
            parameters=FreshnessScoringParameters(
                boosting_duration="P7D"  # 7 days
            ),
            interpolation=ScoringFunctionInterpolation.LINEAR
        )
        
        # Combined profile
        combined_profile = ScoringProfile(
            name="recent_and_relevant",
            text_weights=text_weights,
            functions=[freshness_function],
            function_aggregation=ScoringFunctionAggregation.SUM
        )
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            scoring_profiles=[combined_profile]
        )
        
        return self.index_client.create_or_update_index(index)

# Usage
freshness_setup = FreshnessProfiles(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Create freshness-boosted index
freshness_index = freshness_setup.create_freshness_profile("news-articles")
print("Created freshness scoring profile (30-day boost)")

# Create combined profile
combined_index = freshness_setup.create_combined_freshness_profile("blog-posts")
print("Created combined freshness + text weight profile")
```

### Magnitude Function

```python
from azure.search.documents.indexes.models import (
    MagnitudeScoringFunction,
    MagnitudeScoringParameters
)

class MagnitudeProfiles:
    """Implement magnitude-based scoring."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_rating_boost_profile(self, index_name):
        """
        Boost by product rating.
        
        Rating range: 1.0 to 5.0
        Boost items with rating >= 4.0
        """
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="productName", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="description", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="rating", type=SearchFieldDataType.Double, filterable=True, sortable=True),
            SearchField(name="price", type=SearchFieldDataType.Double, filterable=True, sortable=True)
        ]
        
        # Magnitude function for rating
        rating_function = MagnitudeScoringFunction(
            field_name="rating",
            boost=3.0,  # Maximum boost
            parameters=MagnitudeScoringParameters(
                boosting_range_start=4.0,  # Start boosting at 4.0
                boosting_range_end=5.0,    # Maximum boost at 5.0
                constant_boost_beyond_range=False
            ),
            interpolation=ScoringFunctionInterpolation.LINEAR
        )
        
        # Scoring profile
        rating_profile = ScoringProfile(
            name="high_rated",
            functions=[rating_function],
            function_aggregation=ScoringFunctionAggregation.SUM
        )
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            scoring_profiles=[rating_profile]
        )
        
        return self.index_client.create_or_update_index(index)
    
    def create_price_range_profile(self, index_name):
        """
        Boost products in target price range.
        
        Boost items priced $500-$1500 (sweet spot for laptops).
        """
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="productName", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="price", type=SearchFieldDataType.Double, filterable=True, sortable=True),
            SearchField(name="rating", type=SearchFieldDataType.Double, filterable=True, sortable=True)
        ]
        
        # Price range function
        price_function = MagnitudeScoringFunction(
            field_name="price",
            boost=2.0,
            parameters=MagnitudeScoringParameters(
                boosting_range_start=500.0,
                boosting_range_end=1500.0,
                constant_boost_beyond_range=False  # No boost outside range
            ),
            interpolation=ScoringFunctionInterpolation.LINEAR
        )
        
        # Combined with rating
        rating_function = MagnitudeScoringFunction(
            field_name="rating",
            boost=2.5,
            parameters=MagnitudeScoringParameters(
                boosting_range_start=4.0,
                boosting_range_end=5.0,
                constant_boost_beyond_range=False
            ),
            interpolation=ScoringFunctionInterpolation.LINEAR
        )
        
        # Multi-function profile
        combined_profile = ScoringProfile(
            name="sweet_spot",
            functions=[price_function, rating_function],
            function_aggregation=ScoringFunctionAggregation.SUM
        )
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            scoring_profiles=[combined_profile]
        )
        
        return self.index_client.create_or_update_index(index)

# Usage
magnitude_setup = MagnitudeProfiles(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Create rating-boosted index
rating_index = magnitude_setup.create_rating_boost_profile("products-rated")
print("Created rating boost profile (4.0-5.0 range)")

# Create price range profile
price_index = magnitude_setup.create_price_range_profile("laptops-sweet-spot")
print("Created price range boost profile ($500-$1500)")
```

### Distance Function

```python
from azure.search.documents.indexes.models import (
    DistanceScoringFunction,
    DistanceScoringParameters
)

class DistanceProfiles:
    """Implement distance-based scoring for geo-search."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_geo_proximity_profile(self, index_name):
        """
        Boost results by geographic proximity.
        
        Boosts results within 50km of reference point.
        """
        from azure.search.documents.indexes.models import SearchableField, SimpleField
        
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="name", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="description", type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name="location",
                type=SearchFieldDataType.GeographyPoint,
                filterable=True,
                sortable=True
            ),
            SearchField(name="category", type=SearchFieldDataType.String, filterable=True)
        ]
        
        # Distance function
        distance_function = DistanceScoringFunction(
            field_name="location",
            boost=2.0,
            parameters=DistanceScoringParameters(
                reference_point_parameter="currentLocation",  # Parameter name
                boosting_distance=50  # 50 km
            ),
            interpolation=ScoringFunctionInterpolation.LINEAR
        )
        
        # Geo-proximity profile
        geo_profile = ScoringProfile(
            name="geo_proximity",
            functions=[distance_function],
            function_aggregation=ScoringFunctionAggregation.SUM
        )
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            scoring_profiles=[geo_profile]
        )
        
        return self.index_client.create_or_update_index(index)

# Usage
distance_setup = DistanceProfiles(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Create geo-proximity index
geo_index = distance_setup.create_geo_proximity_profile("stores-geo")
print("Created geo-proximity scoring profile (50km boost)")

# Search with geo-proximity
from azure.search.documents import SearchClient

geo_searcher = SearchClient(
    endpoint=os.getenv("SEARCH_ENDPOINT"),
    index_name="stores-geo",
    credential=AzureKeyCredential(os.getenv("SEARCH_API_KEY"))
)

# Search near a location (Seattle coordinates)
results = geo_searcher.search(
    search_text="coffee shop",
    scoring_profile="geo_proximity",
    scoring_parameters=["currentLocation-47.6062,-122.3321"],  # Seattle lat,lon
    top=10
)

print("\nNearby results:")
for result in results:
    print(f"  {result['name']} - Score: {result['@search.score']:.4f}")
```

### Tag Boosting Function

```python
from azure.search.documents.indexes.models import (
    TagScoringFunction,
    TagScoringParameters
)

class TagProfiles:
    """Implement tag-based scoring."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_tag_boost_profile(self, index_name):
        """
        Boost documents matching preferred tags.
        
        Example: Boost products in user's preferred categories.
        """
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="description", type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name="tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True,
                filterable=True
            ),
            SearchField(
                name="categories",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True
            )
        ]
        
        # Tag function
        tag_function = TagScoringFunction(
            field_name="tags",
            boost=4.0,
            parameters=TagScoringParameters(
                tags_parameter="preferredTags"  # Parameter name
            )
        )
        
        # Category function
        category_function = TagScoringFunction(
            field_name="categories",
            boost=2.5,
            parameters=TagScoringParameters(
                tags_parameter="preferredCategories"
            )
        )
        
        # Tag-based profile
        tag_profile = ScoringProfile(
            name="personalized",
            functions=[tag_function, category_function],
            function_aggregation=ScoringFunctionAggregation.SUM
        )
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            scoring_profiles=[tag_profile]
        )
        
        return self.index_client.create_or_update_index(index)

# Usage
tag_setup = TagProfiles(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Create tag-boosted index
tag_index = tag_setup.create_tag_boost_profile("products-personalized")
print("Created tag boost profile")

# Search with tag preferences
tag_searcher = SearchClient(
    endpoint=os.getenv("SEARCH_ENDPOINT"),
    index_name="products-personalized",
    credential=AzureKeyCredential(os.getenv("SEARCH_API_KEY"))
)

# Search with preferred tags
results = tag_searcher.search(
    search_text="laptop",
    scoring_profile="personalized",
    scoring_parameters=[
        "preferredTags-gaming,performance,high-end",
        "preferredCategories-Electronics,Computers"
    ],
    top=10
)

print("\nPersonalized results:")
for result in results:
    print(f"  {result['title']} - Score: {result['@search.score']:.4f}")
```

---

## Implementation

### Complete Scoring Profile Example

```python
class CompleteScoringProfile:
    """Build comprehensive scoring profile combining multiple signals."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_ecommerce_profile(self, index_name):
        """
        Complete e-commerce scoring profile.
        
        Combines:
        - Text weights (title > description)
        - Freshness (recent products)
        - Rating (highly rated products)
        - Price range (target price point)
        - Popularity (view count magnitude)
        """
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="description", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="brand", type=SearchFieldDataType.String, searchable=True, filterable=True),
            SearchField(name="price", type=SearchFieldDataType.Double, filterable=True, sortable=True),
            SearchField(name="rating", type=SearchFieldDataType.Double, filterable=True, sortable=True),
            SearchField(name="reviewCount", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SearchField(name="addedDate", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SearchField(
                name="tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True,
                filterable=True
            )
        ]
        
        # Text weights
        text_weights = TextWeights(weights={
            "title": 3.0,
            "brand": 2.0,
            "description": 1.0,
            "tags": 1.5
        })
        
        # Freshness (30-day boost for new products)
        freshness_fn = FreshnessScoringFunction(
            field_name="addedDate",
            boost=2.0,
            parameters=FreshnessScoringParameters(boosting_duration="P30D"),
            interpolation=ScoringFunctionInterpolation.LINEAR
        )
        
        # Rating boost (4.0-5.0)
        rating_fn = MagnitudeScoringFunction(
            field_name="rating",
            boost=3.0,
            parameters=MagnitudeScoringParameters(
                boosting_range_start=4.0,
                boosting_range_end=5.0,
                constant_boost_beyond_range=False
            ),
            interpolation=ScoringFunctionInterpolation.LINEAR
        )
        
        # Review count (popularity)
        popularity_fn = MagnitudeScoringFunction(
            field_name="reviewCount",
            boost=1.5,
            parameters=MagnitudeScoringParameters(
                boosting_range_start=50.0,
                boosting_range_end=500.0,
                constant_boost_beyond_range=True
            ),
            interpolation=ScoringFunctionInterpolation.LOGARITHMIC  # Logarithmic for popularity
        )
        
        # Combined profile
        ecommerce_profile = ScoringProfile(
            name="ecommerce_relevance",
            text_weights=text_weights,
            functions=[freshness_fn, rating_fn, popularity_fn],
            function_aggregation=ScoringFunctionAggregation.SUM
        )
        
        # Alternative: Average aggregation
        ecommerce_avg_profile = ScoringProfile(
            name="ecommerce_balanced",
            text_weights=text_weights,
            functions=[freshness_fn, rating_fn, popularity_fn],
            function_aggregation=ScoringFunctionAggregation.AVERAGE
        )
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            scoring_profiles=[ecommerce_profile, ecommerce_avg_profile],
            default_scoring_profile="ecommerce_relevance"
        )
        
        return self.index_client.create_or_update_index(index)

# Usage
complete_setup = CompleteScoringProfile(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Create comprehensive e-commerce profile
ecommerce_index = complete_setup.create_ecommerce_profile("ecommerce-products")
print("Created comprehensive e-commerce scoring profile")
print("  - Text weights (title, brand, description, tags)")
print("  - Freshness (30-day boost)")
print("  - Rating (4.0-5.0 boost)")
print("  - Popularity (review count)")
```

---

## Advanced Patterns

### Function Aggregation Strategies

```python
class AggregationStrategies:
    """Understanding function aggregation."""
    
    @staticmethod
    def aggregation_types():
        """
        Function aggregation methods:
        
        1. SUM: Add all function scores
           Best for: Additive signals (freshness + rating + popularity)
           
        2. AVERAGE: Average all function scores
           Best for: Balanced contribution from each signal
           
        3. MINIMUM: Take lowest function score
           Best for: Conservative ranking (all signals must agree)
           
        4. MAXIMUM: Take highest function score
           Best for: Optimistic ranking (any strong signal wins)
           
        5. FIRST_MATCHING: Use first function that applies
           Best for: Conditional logic (if A then boost, else if B...)
        """
        return {
            'sum': {
                'formula': 'score = f1 + f2 + f3 + ...',
                'use_case': 'Multiple independent positive signals',
                'example': 'Freshness + rating + popularity',
                'note': 'Most common aggregation'
            },
            'average': {
                'formula': 'score = (f1 + f2 + f3 + ...) / n',
                'use_case': 'Balanced contribution from each function',
                'example': 'Equal weight to freshness and rating',
                'note': 'Prevents any single function from dominating'
            },
            'minimum': {
                'formula': 'score = min(f1, f2, f3, ...)',
                'use_case': 'All criteria must be met',
                'example': 'Boost only if both recent AND highly rated',
                'note': 'Conservative, strict requirements'
            },
            'maximum': {
                'formula': 'score = max(f1, f2, f3, ...)',
                'use_case': 'Any strong signal is sufficient',
                'example': 'Boost if recent OR highly rated',
                'note': 'Optimistic, lenient requirements'
            },
            'first_matching': {
                'formula': 'score = first function that returns > 0',
                'use_case': 'Conditional boosting logic',
                'example': 'Premium tier > regular tier > basic tier',
                'note': 'Order matters!'
            }
        }
    
    @staticmethod
    def aggregation_example():
        """Example of different aggregations."""
        # Function scores
        freshness_score = 0.8  # Recent document
        rating_score = 0.6     # Good rating
        popularity_score = 0.4 # Moderate popularity
        
        # Base score
        base = 5.0
        
        # Boost multipliers
        freshness_boost = 2.0
        rating_boost = 3.0
        popularity_boost = 1.5
        
        # Calculate contributions
        f1 = freshness_score * freshness_boost    # 1.6
        f2 = rating_score * rating_boost          # 1.8
        f3 = popularity_score * popularity_boost  # 0.6
        
        return {
            'base_score': base,
            'function_contributions': {
                'freshness': f1,
                'rating': f2,
                'popularity': f3
            },
            'aggregations': {
                'sum': base + (f1 + f2 + f3),           # 5.0 + 4.0 = 9.0
                'average': base + ((f1 + f2 + f3) / 3), # 5.0 + 1.33 = 6.33
                'minimum': base + min(f1, f2, f3),      # 5.0 + 0.6 = 5.6
                'maximum': base + max(f1, f2, f3),      # 5.0 + 1.8 = 6.8
            }
        }

# Usage
agg = AggregationStrategies()

types = agg.aggregation_types()
print("Aggregation Types:")
for agg_type, details in types.items():
    print(f"\n{agg_type.upper()}:")
    print(f"  Formula: {details['formula']}")
    print(f"  Use case: {details['use_case']}")

example = agg.aggregation_example()
print("\n\nAggregation Example:")
print(f"Base score: {example['base_score']}")
print(f"Function contributions: {example['function_contributions']}")
print(f"\nFinal scores by aggregation:")
for agg_type, score in example['aggregations'].items():
    print(f"  {agg_type}: {score:.2f}")
```

---

## Evaluation

### A/B Testing Scoring Profiles

```python
class ScoringProfileEvaluator:
    """Evaluate and compare scoring profiles."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def compare_profiles(self, query_text, profiles, top=20):
        """
        Compare results across different scoring profiles.
        
        Returns ranking differences and score distributions.
        """
        results_by_profile = {}
        
        for profile_name in profiles:
            results = list(self.search_client.search(
                search_text=query_text,
                scoring_profile=profile_name if profile_name != "default" else None,
                top=top
            ))
            
            results_by_profile[profile_name] = [
                {
                    'id': r['id'],
                    'title': r.get('title'),
                    'score': r['@search.score']
                }
                for r in results
            ]
        
        return self._analyze_differences(results_by_profile)
    
    def _analyze_differences(self, results_by_profile):
        """Analyze differences between profiles."""
        analysis = {}
        
        for profile_name, results in results_by_profile.items():
            doc_ids = [r['id'] for r in results]
            scores = [r['score'] for r in results]
            
            analysis[profile_name] = {
                'top_docs': doc_ids[:5],
                'min_score': min(scores) if scores else 0,
                'max_score': max(scores) if scores else 0,
                'avg_score': sum(scores) / len(scores) if scores else 0
            }
        
        # Calculate overlap
        if len(results_by_profile) >= 2:
            profiles = list(results_by_profile.keys())
            profile_a = profiles[0]
            profile_b = profiles[1]
            
            ids_a = set(results_by_profile[profile_a][0]['id'] for _ in range(min(10, len(results_by_profile[profile_a]))))
            ids_b = set(results_by_profile[profile_b][0]['id'] for _ in range(min(10, len(results_by_profile[profile_b]))))
            
            overlap = len(ids_a & ids_b)
            analysis['overlap_top10'] = {
                'count': overlap,
                'percentage': (overlap / 10) * 100 if overlap else 0
            }
        
        return analysis
    
    def measure_impact(self, query_text, base_profile, test_profile, top=50):
        """
        Measure impact of scoring profile change.
        
        Metrics:
        - Rank changes
        - Score changes
        - Top-k stability
        """
        # Get results from both profiles
        base_results = list(self.search_client.search(
            search_text=query_text,
            scoring_profile=base_profile if base_profile != "default" else None,
            top=top
        ))
        
        test_results = list(self.search_client.search(
            search_text=query_text,
            scoring_profile=test_profile,
            top=top
        ))
        
        # Build rank maps
        base_ranks = {r['id']: i for i, r in enumerate(base_results)}
        test_ranks = {r['id']: i for i, r in enumerate(test_results)}
        
        # Calculate movements
        movements = []
        for doc_id in base_ranks:
            if doc_id in test_ranks:
                rank_change = base_ranks[doc_id] - test_ranks[doc_id]
                movements.append({
                    'id': doc_id,
                    'base_rank': base_ranks[doc_id],
                    'test_rank': test_ranks[doc_id],
                    'change': rank_change
                })
        
        # Sort by absolute movement
        movements.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return {
            'query': query_text,
            'base_profile': base_profile,
            'test_profile': test_profile,
            'top_movements': movements[:10],
            'avg_abs_movement': sum(abs(m['change']) for m in movements) / len(movements) if movements else 0
        }

# Usage
evaluator = ScoringProfileEvaluator(searcher.search_client)

# Compare profiles
comparison = evaluator.compare_profiles(
    "laptop for gaming",
    ["default", "ecommerce_relevance", "ecommerce_balanced"],
    top=20
)

print("Profile Comparison:")
for profile, stats in comparison.items():
    if profile != 'overlap_top10':
        print(f"\n{profile}:")
        print(f"  Avg score: {stats['avg_score']:.4f}")
        print(f"  Score range: {stats['min_score']:.4f} - {stats['max_score']:.4f}")

# Measure impact
impact = evaluator.measure_impact(
    "gaming laptop",
    "default",
    "ecommerce_relevance",
    top=50
)

print(f"\nImpact Analysis:")
print(f"Average rank movement: {impact['avg_abs_movement']:.2f} positions")
print(f"\nTop movements:")
for movement in impact['top_movements'][:5]:
    direction = "↑" if movement['change'] > 0 else "↓"
    print(f"  {movement['id']}: {movement['base_rank']} → {movement['test_rank']} ({direction}{abs(movement['change'])})")
```

---

## Best Practices

Effective scoring profile implementation requires careful balance between business requirements, user expectations, and technical constraints. These practices emerge from production deployments across e-commerce, content, and enterprise search applications.

### 1. Start Simple, Then Iterate

**DO: Begin with text weights only**

Text weights provide immediate value with minimal complexity and zero risk of over-complicating relevance.

```python
# GOOD: Simple first iteration
simple_profile = ScoringProfile(
    name="simple_v1",
    text_weights=TextWeights(weights={
        "title": 3.0,
        "description": 1.0
    })
)

# Deploy, measure impact, then add functions if needed
```

```python
# BAD: Starting with complex multi-function profile
complex_profile = ScoringProfile(
    name="complex_v1",
    text_weights=TextWeights(weights={
        "title": 3.0,
        "summary": 2.0,
        "body": 1.0,
        "tags": 2.5,
        "category": 1.5
    }),
    functions=[
        freshness_fn, rating_fn, popularity_fn, 
        distance_fn, tag_fn, price_fn  # Too many!
    ],
    function_aggregation=ScoringFunctionAggregation.SUM
)

# Problem: Hard to debug, tune, or understand impact of individual components
```

**Iterative Development Path**:

| Phase | Components | Measure | Decision |
|-------|-----------|---------|----------|
| 1. Baseline | BM25 only | Precision@5, NDCG@10 | Establish baseline metrics |
| 2. Text weights | title:3.0, description:1.0 | Compare to baseline | If +10% precision, keep; iterate on weights |
| 3. Single function | Add freshness OR rating | Compare to Phase 2 | If +5% NDCG, keep; else remove |
| 4. Multi-function | Add 1-2 more functions | A/B test vs Phase 3 | Validate each addition |
| 5. Tune | Adjust boost values | Fine-tune with real traffic | Optimize parameters |

**Measurement Between Phases**:
```python
def compare_phases(phase_a_profile, phase_b_profile, test_queries):
    """Compare two profile versions with metrics."""
    results = {
        'phase_a': evaluate_profile(phase_a_profile, test_queries),
        'phase_b': evaluate_profile(phase_b_profile, test_queries)
    }
    
    improvement = {
        'precision': results['phase_b']['precision'] - results['phase_a']['precision'],
        'ndcg': results['phase_b']['ndcg'] - results['phase_a']['ndcg'],
        'user_satisfaction': results['phase_b']['satisfaction'] - results['phase_a']['satisfaction']
    }
    
    # Only proceed if improvement > 5% on primary metric
    if improvement['ndcg'] < 0.05:
        print("⚠️  Insufficient improvement - reconsider Phase B changes")
    
    return improvement
```

### 2. Choose Appropriate Boost Values

**DO: Keep boost values moderate (1.5-5.0 range)**

Extreme boost values can completely overwhelm BM25 relevance, causing low-quality but boosted documents to rank above highly relevant results.

```python
# GOOD: Moderate boost values
appropriate_profile = ScoringProfile(
    name="balanced",
    text_weights=TextWeights(weights={
        "title": 3.0,      # Reasonable emphasis
        "description": 1.0
    }),
    functions=[
        FreshnessScoringFunction(
            field_name="publishedDate",
            boost=2.0,     # Modest freshness boost
            parameters=FreshnessScoringParameters(boosting_duration="P7D")
        ),
        MagnitudeScoringFunction(
            field_name="rating",
            boost=3.0,     # Moderate rating boost
            parameters=MagnitudeScoringParameters(
                boosting_range_start=4.0,
                boosting_range_end=5.0
            )
        )
    ]
)

# Result: BM25 still primary signal, functions enhance
# Example: base=8.0, freshness=+1.8, rating=+2.4 → final=12.2
```

```python
# BAD: Extreme boost values
extreme_profile = ScoringProfile(
    name="over_boosted",
    text_weights=TextWeights(weights={
        "title": 15.0,     # ❌ Too high!
        "description": 1.0
    }),
    functions=[
        FreshnessScoringFunction(
            field_name="publishedDate",
            boost=20.0,    # ❌ Freshness dominates everything
            parameters=FreshnessScoringParameters(boosting_duration="P30D")
        )
    ]
)

# Result: Relevance broken
# Example: base=8.0, freshness=+18.0 → final=26.0
# Poor-quality recent document (base=2.0, freshness=+18.0 → final=20.0)
# Outranks highly relevant older document (base=15.0, freshness=+5.0 → final=20.0)
```

**Boost Value Guidelines**:

| Signal Type | Recommended Range | Rationale |
|------------|------------------|-----------|
| Title weight | 2.0 - 4.0 | Title matches important but not dominant |
| Description weight | 1.0 (baseline) | Reference point for other fields |
| Tags/metadata | 1.5 - 2.5 | Moderate signal strength |
| Freshness | 1.5 - 5.0 | Depends on content type (news: 5.0, docs: 2.0) |
| Rating/quality | 2.0 - 4.0 | Strong signal but not overwhelming |
| Popularity | 1.5 - 3.0 | Supporting signal only |
| Distance | 2.0 - 5.0 | High for local search, low for general |
| Tag matching | 2.0 - 4.0 | Personalization signal |

**Boost Tuning Process**:
```python
def tune_boost_value(profile_name, function_name, boost_values, test_queries):
    """
    Test different boost values to find optimal setting.
    
    Args:
        profile_name: Base profile name
        function_name: Which function to tune
        boost_values: List of values to test (e.g., [1.5, 2.0, 3.0, 5.0])
        test_queries: Representative query set
    """
    results = []
    
    for boost in boost_values:
        # Create profile variant with this boost value
        profile = create_profile_variant(profile_name, function_name, boost)
        
        # Evaluate
        metrics = evaluate_profile(profile, test_queries)
        
        results.append({
            'boost': boost,
            'ndcg': metrics['ndcg'],
            'precision': metrics['precision'],
            'user_satisfaction': metrics['satisfaction']
        })
    
    # Find optimal boost (highest NDCG)
    optimal = max(results, key=lambda x: x['ndcg'])
    
    return optimal

# Example usage:
optimal_freshness = tune_boost_value(
    "ecommerce_v1",
    "freshness",
    boost_values=[1.5, 2.0, 3.0, 4.0, 5.0],
    test_queries=load_test_queries()
)

print(f"Optimal freshness boost: {optimal_freshness['boost']}")
print(f"NDCG@10: {optimal_freshness['ndcg']:.4f}")
```

### 3. Select Correct Interpolation

**DO: Match interpolation to data distribution**

Different interpolation methods produce dramatically different boost curves.

```python
# LINEAR: Best for uniformly distributed values
# Use for: Ratings (1-5), prices in narrow range, recency in fixed window

linear_freshness = FreshnessScoringFunction(
    field_name="publishedDate",
    boost=5.0,
    parameters=FreshnessScoringParameters(boosting_duration="P30D"),
    interpolation=ScoringFunctionInterpolation.LINEAR
)

# Boost curve:
# Today: 5.0 (100%)
# 15 days ago: 2.5 (50%)
# 30 days ago: 0.0 (0%)
# Smooth, predictable decay ✓

# LOGARITHMIC: Best for exponentially distributed values
# Use for: View counts, popularity metrics, social engagement

log_popularity = MagnitudeScoringFunction(
    field_name="viewCount",
    boost=3.0,
    parameters=MagnitudeScoringParameters(
        boosting_range_start=100.0,
        boosting_range_end=10000.0
    ),
    interpolation=ScoringFunctionInterpolation.LOGARITHMIC
)

# Boost curve:
# 100 views: 0.0
# 500 views: 1.05 (35%)
# 1000 views: 1.5 (50%)
# 5000 views: 2.4 (80%)
# 10000 views: 3.0 (100%)
# Diminishing returns prevent viral content from dominating ✓

# QUADRATIC: Best for accelerating at extremes
# Use for: Priority tiers, premium content, VIP boosting

quadratic_priority = MagnitudeScoringFunction(
    field_name="priority",
    boost=4.0,
    parameters=MagnitudeScoringParameters(
        boosting_range_start=1.0,
        boosting_range_end=5.0
    ),
    interpolation=ScoringFunctionInterpolation.QUADRATIC
)

# Boost curve (quadratic):
# Priority 1: 0.0 (0%)
# Priority 2: 0.4 (10%)
# Priority 3: 1.6 (40%)
# Priority 4: 2.8 (70%)
# Priority 5: 4.0 (100%)
# High-priority items get disproportionate boost ✓

# CONSTANT: All values in range get same boost
# Use for: Binary signals, membership tiers

constant_boost = MagnitudeScoringFunction(
    field_name="isPremium",
    boost=2.5,
    parameters=MagnitudeScoringParameters(
        boosting_range_start=1.0,
        boosting_range_end=1.0,
        constant_boost_beyond_range=True
    ),
    interpolation=ScoringFunctionInterpolation.CONSTANT
)

# Boost: isPremium=1 → +2.5, isPremium=0 → +0.0
# Simple on/off boost ✓
```

**Interpolation Decision Tree**:

| Data Pattern | Interpolation | Example Use Case |
|-------------|---------------|------------------|
| Uniform distribution | LINEAR | Recency (1-30 days), ratings (1-5 stars) |
| Exponential distribution | LOGARITHMIC | Views, social shares, comment counts |
| Priority tiers | QUADRATIC | VIP levels, content tiers, quality scores |
| Binary/categorical | CONSTANT | Membership status, verification badges |

### 4. Use Appropriate Function Aggregation

**DO: Choose aggregation that matches your ranking intent**

```python
# SUM: Additive signals (most common)
# Use when: Multiple independent positive signals should compound
sum_profile = ScoringProfile(
    name="additive",
    functions=[freshness_fn, rating_fn, popularity_fn],
    function_aggregation=ScoringFunctionAggregation.SUM
)

# Example: base=10, freshness=+2, rating=+3, popularity=+1 → final=16
# All positive signals add up ✓

# AVERAGE: Balanced contribution
# Use when: Preventing any single function from dominating
avg_profile = ScoringProfile(
    name="balanced",
    functions=[freshness_fn, rating_fn, popularity_fn],
    function_aggregation=ScoringFunctionAggregation.AVERAGE
)

# Example: base=10, (freshness=2 + rating=3 + popularity=1)/3 → final=12
# Each function contributes equally ✓

# MINIMUM: All criteria must be met (conservative)
# Use when: Documents must meet ALL quality bars
min_profile = ScoringProfile(
    name="strict",
    functions=[freshness_fn, rating_fn, quality_fn],
    function_aggregation=ScoringFunctionAggregation.MINIMUM
)

# Example: base=10, min(freshness=5, rating=2, quality=4) → final=12
# Weakest signal limits overall boost (conservative) ✓

# MAXIMUM: Any strong signal wins (optimistic)
# Use when: Documents excel in ANY dimension deserve boost
max_profile = ScoringProfile(
    name="opportunistic",
    functions=[freshness_fn, rating_fn, popularity_fn],
    function_aggregation=ScoringFunctionAggregation.MAXIMUM
)

# Example: base=10, max(freshness=1, rating=5, popularity=2) → final=15
# Strongest signal determines boost ✓

# FIRST_MATCHING: Conditional logic
# Use when: Prioritized fallback logic
first_profile = ScoringProfile(
    name="tiered",
    functions=[premium_fn, verified_fn, standard_fn],  # Order matters!
    function_aggregation=ScoringFunctionAggregation.FIRST_MATCHING
)

# Example: Premium content → use premium_fn boost
#          Not premium but verified → use verified_fn boost
#          Otherwise → use standard_fn boost
# Implements if/else logic ✓
```

**Aggregation Selection Guide**:

| Business Requirement | Aggregation | Scenario |
|---------------------|-------------|----------|
| "Boost recent AND highly-rated items" | SUM | E-commerce: new + popular products |
| "Balance freshness and quality equally" | AVERAGE | News: recent + authoritative |
| "Only boost if ALL criteria met" | MINIMUM | Premium search: verified + recent + rated |
| "Boost if strong in ANY dimension" | MAXIMUM | Content discovery: viral OR recent OR quality |
| "Tiered priority (premium > standard)" | FIRST_MATCHING | Membership: premium > verified > free |

### 5. Test with Real Queries Before Production

**DO: Validate profile with representative query set**

```python
class ScoringProfileValidator:
    """Systematic testing before production deployment."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def validate_profile(self, profile_name, test_queries, relevance_judgments):
        """
        Comprehensive profile validation.
        
        Args:
            profile_name: Scoring profile to test
            test_queries: List of representative queries
            relevance_judgments: Ground truth relevance (query -> [relevant_doc_ids])
        
        Returns:
            Validation metrics and pass/fail status
        """
        metrics = {
            'precision_at_5': [],
            'ndcg_at_10': [],
            'mrr': [],
            'user_satisfaction_proxy': []
        }
        
        failed_queries = []
        
        for query in test_queries:
            # Execute search with profile
            results = list(self.search_client.search(
                search_text=query['text'],
                scoring_profile=profile_name,
                top=10
            ))
            
            # Get result IDs
            result_ids = [r['id'] for r in results]
            
            # Check against ground truth
            relevant_ids = relevance_judgments.get(query['text'], [])
            
            # Calculate metrics
            precision = self._precision_at_k(result_ids[:5], relevant_ids)
            ndcg = self._ndcg_at_k(result_ids, relevant_ids, k=10)
            mrr = self._mrr(result_ids, relevant_ids)
            
            metrics['precision_at_5'].append(precision)
            metrics['ndcg_at_10'].append(ndcg)
            metrics['mrr'].append(mrr)
            
            # Check for failures (no relevant results in top 5)
            if precision == 0:
                failed_queries.append({
                    'query': query['text'],
                    'top_results': result_ids[:5],
                    'expected': relevant_ids[:5]
                })
        
        # Aggregate metrics
        avg_metrics = {
            'avg_precision_at_5': sum(metrics['precision_at_5']) / len(metrics['precision_at_5']),
            'avg_ndcg_at_10': sum(metrics['ndcg_at_10']) / len(metrics['ndcg_at_10']),
            'avg_mrr': sum(metrics['mrr']) / len(metrics['mrr']),
            'zero_result_rate': len(failed_queries) / len(test_queries)
        }
        
        # Pass/fail criteria
        passing = (
            avg_metrics['avg_precision_at_5'] >= 0.6 and
            avg_metrics['avg_ndcg_at_10'] >= 0.7 and
            avg_metrics['zero_result_rate'] <= 0.05
        )
        
        return {
            'metrics': avg_metrics,
            'passing': passing,
            'failed_queries': failed_queries
        }
    
    def _precision_at_k(self, results, relevant, k=5):
        """Calculate precision@k."""
        results_k = results[:k]
        relevant_set = set(relevant)
        matches = len([r for r in results_k if r in relevant_set])
        return matches / k if k > 0 else 0
    
    def _ndcg_at_k(self, results, relevant, k=10):
        """Calculate NDCG@k."""
        # Simplified NDCG calculation
        dcg = sum(1.0 / np.log2(i + 2) if results[i] in relevant else 0 
                  for i in range(min(k, len(results))))
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
        return dcg / ideal_dcg if ideal_dcg > 0 else 0
    
    def _mrr(self, results, relevant):
        """Calculate Mean Reciprocal Rank."""
        for i, result in enumerate(results):
            if result in relevant:
                return 1.0 / (i + 1)
        return 0.0

# Usage:
validator = ScoringProfileValidator(search_client)

test_queries = [
    {'text': 'laptop for gaming'},
    {'text': 'noise cancelling headphones'},
    {'text': 'wireless mouse'},
    # ... 50-100 more representative queries
]

relevance_judgments = {
    'laptop for gaming': ['gaming_laptop_1', 'gaming_laptop_2', 'gaming_laptop_3'],
    'noise cancelling headphones': ['headphones_1', 'headphones_2'],
    # ... ground truth for each query
}

validation = validator.validate_profile(
    'ecommerce_relevance_v2',
    test_queries,
    relevance_judgments
)

if validation['passing']:
    print("✅ Profile passed validation")
    print(f"Precision@5: {validation['metrics']['avg_precision_at_5']:.3f}")
    print(f"NDCG@10: {validation['metrics']['avg_ndcg_at_10']:.3f}")
else:
    print("❌ Profile failed validation")
    print(f"Failed queries: {len(validation['failed_queries'])}")
    for failure in validation['failed_queries'][:5]:
        print(f"  Query: {failure['query']}")
        print(f"    Expected: {failure['expected']}")
        print(f"    Got: {failure['top_results']}")
```

### 6. Version Profile Names for Safe Updates

**DO: Use versioned names to enable safe rollback**

```python
# GOOD: Versioned profile names
profile_v1 = ScoringProfile(
    name="ecommerce_relevance_v1",
    text_weights=TextWeights(weights={"title": 2.0, "description": 1.0})
)

profile_v2 = ScoringProfile(
    name="ecommerce_relevance_v2",
    text_weights=TextWeights(weights={"title": 3.0, "description": 1.0}),
    functions=[freshness_fn]  # Added freshness
)

# Deploy both profiles
index.scoring_profiles = [profile_v1, profile_v2]

# A/B test v2 vs v1
# 90% traffic → v1 (stable)
# 10% traffic → v2 (canary)

# If v2 performs better:
# - Gradual rollout: 10% → 25% → 50% → 100%
# - Keep v1 available for instant rollback

# If v2 performs worse:
# - Instant rollback to v1 (just change scoring_profile parameter)
# - Investigate issues offline
# - Create v3 with fixes
```

```python
# BAD: Modifying profile in-place
profile = ScoringProfile(
    name="ecommerce_relevance",  # No version
    # ... configuration
)

# Update profile with new config
# Problem: Can't roll back! Old config is lost.
# If new config has issues, must fix forward (risky)
```

**Version Management Strategy**:
```python
class ProfileVersionManager:
    """Manage scoring profile versions."""
    
    def __init__(self, index_client):
        self.index_client = index_client
    
    def deploy_new_version(self, base_name, version, profile_config):
        """Deploy new profile version alongside existing."""
        versioned_name = f"{base_name}_v{version}"
        
        # Create new profile
        new_profile = ScoringProfile(
            name=versioned_name,
            **profile_config
        )
        
        # Add to index (keeps existing versions)
        index = self.index_client.get_index("myindex")
        index.scoring_profiles.append(new_profile)
        self.index_client.create_or_update_index(index)
        
        return versioned_name
    
    def deprecate_version(self, base_name, version):
        """Remove old profile version after successful migration."""
        old_name = f"{base_name}_v{version}"
        
        index = self.index_client.get_index("myindex")
        index.scoring_profiles = [
            p for p in index.scoring_profiles 
            if p.name != old_name
        ]
        self.index_client.create_or_update_index(index)
        
        print(f"Deprecated profile: {old_name}")

# Usage:
manager = ProfileVersionManager(index_client)

# Deploy v3 (keeps v1, v2 available)
v3_name = manager.deploy_new_version(
    "ecommerce_relevance",
    version=3,
    profile_config={
        'text_weights': TextWeights(weights={"title": 3.5, "description": 1.0}),
        'functions': [freshness_fn, rating_fn]
    }
)

# After v3 validated and at 100% traffic, deprecate v1
manager.deprecate_version("ecommerce_relevance", version=1)
```

### 7. Monitor Score Distributions

**DO: Track score distributions to detect anomalies**

```python
import numpy as np
import matplotlib.pyplot as plt

class ScoreDistributionMonitor:
    """Monitor and alert on scoring anomalies."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def collect_score_distribution(self, queries, profile_name, top=50):
        """Collect scores across representative queries."""
        all_scores = []
        
        for query in queries:
            results = list(self.search_client.search(
                search_text=query,
                scoring_profile=profile_name,
                top=top
            ))
            
            scores = [r['@search.score'] for r in results]
            all_scores.extend(scores)
        
        return np.array(all_scores)
    
    def analyze_distribution(self, scores):
        """Analyze score distribution statistics."""
        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'p25': np.percentile(scores, 25),
            'p75': np.percentile(scores, 75),
            'p95': np.percentile(scores, 95),
            'p99': np.percentile(scores, 99)
        }
    
    def detect_anomalies(self, baseline_stats, current_stats, threshold=0.3):
        """Detect significant distribution changes."""
        anomalies = []
        
        # Check for mean shift
        mean_change = abs(current_stats['mean'] - baseline_stats['mean']) / baseline_stats['mean']
        if mean_change > threshold:
            anomalies.append({
                'metric': 'mean',
                'baseline': baseline_stats['mean'],
                'current': current_stats['mean'],
                'change_pct': mean_change * 100
            })
        
        # Check for variance change
        std_change = abs(current_stats['std'] - baseline_stats['std']) / baseline_stats['std']
        if std_change > threshold:
            anomalies.append({
                'metric': 'std',
                'baseline': baseline_stats['std'],
                'current': current_stats['std'],
                'change_pct': std_change * 100
            })
        
        # Check for range explosion
        range_baseline = baseline_stats['max'] - baseline_stats['min']
        range_current = current_stats['max'] - current_stats['min']
        range_change = abs(range_current - range_baseline) / range_baseline
        if range_change > threshold:
            anomalies.append({
                'metric': 'range',
                'baseline': range_baseline,
                'current': range_current,
                'change_pct': range_change * 100
            })
        
        return anomalies
    
    def plot_distributions(self, baseline_scores, current_scores, profile_name):
        """Visualize score distribution comparison."""
        plt.figure(figsize=(12, 5))
        
        # Histogram comparison
        plt.subplot(1, 2, 1)
        plt.hist(baseline_scores, bins=30, alpha=0.5, label='Baseline', color='blue')
        plt.hist(current_scores, bins=30, alpha=0.5, label=f'{profile_name}', color='red')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution Comparison')
        plt.legend()
        
        # Box plot comparison
        plt.subplot(1, 2, 2)
        plt.boxplot([baseline_scores, current_scores], labels=['Baseline', profile_name])
        plt.ylabel('Score')
        plt.title('Score Distribution Box Plot')
        
        plt.tight_layout()
        plt.savefig(f'score_distribution_{profile_name}.png')
        plt.close()

# Usage:
monitor = ScoreDistributionMonitor(search_client)

test_queries = ['query1', 'query2', 'query3']  # Representative set

# Collect baseline (BM25 only)
baseline_scores = monitor.collect_score_distribution(test_queries, profile_name=None)
baseline_stats = monitor.analyze_distribution(baseline_scores)

# Collect with new profile
profile_scores = monitor.collect_score_distribution(test_queries, 'ecommerce_v2')
profile_stats = monitor.analyze_distribution(profile_scores)

# Detect anomalies
anomalies = monitor.detect_anomalies(baseline_stats, profile_stats, threshold=0.3)

if anomalies:
    print("⚠️  Score distribution anomalies detected:")
    for anomaly in anomalies:
        print(f"  {anomaly['metric']}: {anomaly['change_pct']:.1f}% change")
        print(f"    Baseline: {anomaly['baseline']:.2f}")
        print(f"    Current: {anomaly['current']:.2f}")
else:
    print("✅ Score distribution within expected range")

# Visualize
monitor.plot_distributions(baseline_scores, profile_scores, 'ecommerce_v2')
```

**Score Distribution Red Flags**:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Mean score increased 50%+ | Boost values too high | Reduce boost multipliers |
| Std deviation doubled | Inconsistent boosting | Review function parameters |
| Max score >100 | Extreme boost values | Cap boost values at 5.0-10.0 |
| Bimodal distribution | Profile favoring subset | Broaden boosting criteria |
| All scores in narrow range | No differentiation | Increase boost values |

### 8. A/B Test Profile Changes

**DO: Validate improvements with real users**

```python
class ABTestFramework:
    """A/B test scoring profile changes."""
    
    def __init__(self, search_client, analytics_client):
        self.search_client = search_client
        self.analytics_client = analytics_client
    
    def route_traffic(self, user_id, control_profile, treatment_profile, treatment_pct=10):
        """
        Route traffic between control and treatment profiles.
        
        Args:
            user_id: User identifier (for consistent assignment)
            control_profile: Baseline profile
            treatment_profile: New profile being tested
            treatment_pct: Percentage of traffic to treatment (default 10%)
        
        Returns:
            Profile name to use for this user
        """
        # Consistent hashing ensures same user always gets same variant
        hash_value = hash(str(user_id)) % 100
        
        if hash_value < treatment_pct:
            return treatment_profile
        else:
            return control_profile
    
    def execute_search_with_tracking(self, user_id, query, profile_name):
        """Execute search and track metrics."""
        # Execute search
        results = list(self.search_client.search(
            search_text=query,
            scoring_profile=profile_name,
            top=20
        ))
        
        # Track assignment
        self.analytics_client.track_event('search', {
            'user_id': user_id,
            'query': query,
            'profile': profile_name,
            'result_count': len(results),
            'top_score': results[0]['@search.score'] if results else 0
        })
        
        return results
    
    def analyze_ab_test(self, control_profile, treatment_profile, min_samples=1000):
        """
        Analyze A/B test results.
        
        Returns statistical significance and metric improvements.
        """
        # Fetch metrics from analytics
        control_metrics = self.analytics_client.get_metrics(
            profile=control_profile,
            min_samples=min_samples
        )
        
        treatment_metrics = self.analytics_client.get_metrics(
            profile=treatment_profile,
            min_samples=min_samples
        )
        
        # Calculate improvements
        improvements = {
            'ctr': (treatment_metrics['ctr'] - control_metrics['ctr']) / control_metrics['ctr'] * 100,
            'session_duration': (treatment_metrics['session_duration'] - control_metrics['session_duration']) / control_metrics['session_duration'] * 100,
            'refinement_rate': (treatment_metrics['refinement_rate'] - control_metrics['refinement_rate']) / control_metrics['refinement_rate'] * 100,
        }
        
        # Statistical significance (simplified)
        significance = self._calculate_significance(control_metrics, treatment_metrics)
        
        return {
            'control': control_metrics,
            'treatment': treatment_metrics,
            'improvements': improvements,
            'significance': significance,
            'recommendation': self._make_recommendation(improvements, significance)
        }
    
    def _calculate_significance(self, control, treatment):
        """Calculate statistical significance (simplified)."""
        # In production, use proper statistical tests (t-test, chi-square, etc.)
        # This is a simplified example
        sample_size = min(control['sample_size'], treatment['sample_size'])
        
        if sample_size < 1000:
            return 'insufficient_data'
        elif sample_size >= 10000:
            return 'high_confidence'
        else:
            return 'moderate_confidence'
    
    def _make_recommendation(self, improvements, significance):
        """Make rollout recommendation."""
        if significance == 'insufficient_data':
            return 'continue_test'
        
        # Check if primary metric (CTR) improved significantly
        if improvements['ctr'] > 5.0 and significance in ['moderate_confidence', 'high_confidence']:
            return 'rollout_to_50_percent'
        elif improvements['ctr'] > 10.0 and significance == 'high_confidence':
            return 'rollout_to_100_percent'
        elif improvements['ctr'] < -5.0:
            return 'rollback_immediately'
        else:
            return 'continue_test'

# Usage:
ab_test = ABTestFramework(search_client, analytics_client)

# Route traffic
user_id = 'user_12345'
profile = ab_test.route_traffic(
    user_id,
    control_profile='ecommerce_v1',
    treatment_profile='ecommerce_v2',
    treatment_pct=10  # 10% of traffic to treatment
)

# Execute search with assigned profile
results = ab_test.execute_search_with_tracking(user_id, 'laptop', profile)

# After 2-4 weeks, analyze results
analysis = ab_test.analyze_ab_test('ecommerce_v1', 'ecommerce_v2', min_samples=5000)

print(f"A/B Test Results:")
print(f"CTR improvement: {analysis['improvements']['ctr']:.1f}%")
print(f"Recommendation: {analysis['recommendation']}")

if analysis['recommendation'] == 'rollout_to_100_percent':
    print("✅ Treatment is winner - full rollout recommended")
elif analysis['recommendation'] == 'rollback_immediately':
    print("❌ Treatment is worse - rollback immediately")
```

---

## Troubleshooting

Common scoring profile issues and their solutions.

### Issue 1: Scoring Profile Not Being Applied

**Symptoms**:
- Results identical with and without profile
- Scores don't change when using different profiles
- Functions appear to have no effect

**Diagnosis**:

```python
# Test if profile is actually being used
results_default = list(search_client.search("test query", top=10))
results_profile = list(search_client.search(
    "test query",
    scoring_profile="my_profile",
    top=10
))

# Compare scores
print("Default scores:", [r['@search.score'] for r in results_default[:5]])
print("Profile scores:", [r['@search.score'] for r in results_profile[:5]])

# If identical, profile not being applied
```

**Root Causes**:

**1. Profile name typo or doesn't exist**
```python
# BAD: Typo in profile name
results = search_client.search(
    "query",
    scoring_profile="ecommerce_relevence"  # Typo: "relevence" vs "relevance"
)

# Azure AI Search silently ignores invalid profile, uses default BM25
# No error thrown!
```

**Solution**:
```python
# GOOD: Verify profile exists before using
def verify_profile_exists(index_client, index_name, profile_name):
    """Check if scoring profile exists in index."""
    index = index_client.get_index(index_name)
    
    profile_names = [p.name for p in index.scoring_profiles]
    
    if profile_name not in profile_names:
        raise ValueError(
            f"Profile '{profile_name}' not found. "
            f"Available profiles: {profile_names}"
        )
    
    return True

# Verify before searching
verify_profile_exists(index_client, "myindex", "ecommerce_relevance")

# Then search with confidence
results = search_client.search(
    "query",
    scoring_profile="ecommerce_relevance"
)
```

**2. Profile applied but functions don't match any documents**
```python
# BAD: Freshness function on field with all NULL values
freshness_fn = FreshnessScoringFunction(
    field_name="publishedDate",  # All documents have NULL publishedDate
    boost=5.0,
    parameters=FreshnessScoringParameters(boosting_duration="P30D")
)

# Result: Function never fires, no boost applied
```

**Solution**:
```python
# GOOD: Verify field has values
def verify_function_field(search_client, field_name, sample_size=100):
    """Check if function field has non-null values."""
    results = list(search_client.search(
        search_text="*",
        select=[field_name],
        top=sample_size
    ))
    
    non_null_count = sum(1 for r in results if r.get(field_name) is not None)
    null_pct = (sample_size - non_null_count) / sample_size * 100
    
    if null_pct > 50:
        print(f"⚠️  WARNING: {field_name} is NULL in {null_pct:.0f}% of documents")
        print(f"   Function on this field will have limited impact")
    
    return non_null_count > 0

# Verify before creating profile
verify_function_field(search_client, "publishedDate")
verify_function_field(search_client, "rating")
```

**3. Functions outside boosting range**
```python
# BAD: Magnitude function with range that excludes all values
magnitude_fn = MagnitudeScoringFunction(
    field_name="price",
    boost=3.0,
    parameters=MagnitudeScoringParameters(
        boosting_range_start=5000.0,   # $5,000
        boosting_range_end=10000.0,    # $10,000
        constant_boost_beyond_range=False
    )
)

# Problem: All products priced $10-$100 → no boost applied!
```

**Solution**:
```python
# GOOD: Analyze field value distribution first
def analyze_field_distribution(search_client, field_name):
    """Analyze field value distribution."""
    results = list(search_client.search(
        search_text="*",
        select=[field_name],
        top=1000
    ))
    
    values = [r[field_name] for r in results if r.get(field_name) is not None]
    
    if not values:
        print(f"❌ No values found for {field_name}")
        return None
    
    import numpy as np
    stats = {
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'p25': np.percentile(values, 25),
        'p75': np.percentile(values, 75)
    }
    
    print(f"Field: {field_name}")
    print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
    print(f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
    print(f"  25th-75th percentile: {stats['p25']:.2f} - {stats['p75']:.2f}")
    
    # Recommend boosting range
    recommended_start = stats['p25']
    recommended_end = stats['p75']
    print(f"\n💡 Recommended boosting range: {recommended_start:.2f} - {recommended_end:.2f}")
    
    return stats

# Use analysis to set appropriate range
stats = analyze_field_distribution(search_client, "price")

# Create function with data-driven range
magnitude_fn = MagnitudeScoringFunction(
    field_name="price",
    boost=3.0,
    parameters=MagnitudeScoringParameters(
        boosting_range_start=stats['p25'],  # 25th percentile
        boosting_range_end=stats['p75'],    # 75th percentile
        constant_boost_beyond_range=False
    )
)
```

### Issue 2: Unexpected Ranking Changes

**Symptoms**:
- Low-quality documents ranking above high-quality ones
- Results seem random or inconsistent
- Specific important documents disappeared from top results

**Diagnosis**:

```python
def diagnose_ranking(search_client, query, expected_top_doc_id, profile_name):
    """Diagnose why expected document isn't ranking well."""
    
    # Search with profile
    results = list(search_client.search(
        search_text=query,
        scoring_profile=profile_name,
        top=50
    ))
    
    # Find expected document
    expected_rank = None
    expected_score = None
    
    for i, result in enumerate(results):
        if result['id'] == expected_top_doc_id:
            expected_rank = i + 1
            expected_score = result['@search.score']
            break
    
    if expected_rank is None:
        print(f"❌ Expected document '{expected_top_doc_id}' not in top 50!")
        return
    
    print(f"Expected document ranked #{expected_rank}")
    print(f"Score: {expected_score:.4f}")
    
    # Show what beat it
    print(f"\nDocuments ranked higher:")
    for i, result in enumerate(results[:expected_rank-1]):
        print(f"  #{i+1}: {result['id']} - Score: {result['@search.score']:.4f}")
        print(f"      Title: {result.get('title', 'N/A')[:60]}")
    
    # Compare to default BM25
    results_default = list(search_client.search(
        search_text=query,
        top=50
    ))
    
    default_rank = None
    for i, result in enumerate(results_default):
        if result['id'] == expected_top_doc_id:
            default_rank = i + 1
            break
    
    if default_rank:
        rank_change = default_rank - expected_rank
        if rank_change > 0:
            print(f"\n✅ Profile improved ranking: #{default_rank} → #{expected_rank} (+{rank_change})")
        else:
            print(f"\n❌ Profile worsened ranking: #{default_rank} → #{expected_rank} ({rank_change})")

# Usage:
diagnose_ranking(
    search_client,
    query="gaming laptop",
    expected_top_doc_id="product_12345",
    profile_name="ecommerce_relevance"
)
```

**Root Causes**:

**1. Boost values too aggressive**
```python
# Problem: Freshness boost overwhelming relevance
freshness_fn = FreshnessScoringFunction(
    field_name="publishedDate",
    boost=20.0,  # ❌ Too high!
    parameters=FreshnessScoringParameters(boosting_duration="P30D")
)

# Result:
# - Recent low-quality document: base=3.0, freshness=+18.0 → final=21.0
# - Older high-quality document: base=15.0, freshness=+5.0 → final=20.0
# Recent beats relevant!
```

**Solution**: Reduce boost to 2.0-5.0 range

**2. Wrong function aggregation**
```python
# BAD: Using MINIMUM when signals are independent
profile = ScoringProfile(
    name="too_strict",
    functions=[freshness_fn, rating_fn, popularity_fn],
    function_aggregation=ScoringFunctionAggregation.MINIMUM
)

# Result: Document needs to score well on ALL functions
# - Freshness: 5.0, Rating: 1.0, Popularity: 4.0 → boost=1.0 (minimum)
# Overly conservative
```

**Solution**: Use SUM for independent signals

**3. Conflicting signals**
```python
# Problem: Freshness favors new, rating favors old (established products)
profile = ScoringProfile(
    name="conflicting",
    functions=[
        FreshnessScoringFunction(..., boost=5.0),  # New products boosted
        MagnitudeScoringFunction(field_name="reviewCount", boost=5.0)  # Needs many reviews (takes time)
    ],
    function_aggregation=ScoringFunctionAggregation.SUM
)

# Result: Very new products (no reviews) and very old products (many reviews) both boosted
# Middle-aged products (moderate reviews) get no boost
```

**Solution**: Balance boost values or use different aggregation

### Issue 3: Score Distribution Anomalies

**Symptoms**:
- All scores bunched together (no differentiation)
- Scores wildly different between queries
- Scores increasing over time without code changes

**Diagnosis**:

```python
def check_score_variance(search_client, queries, profile_name):
    """Check if profile provides adequate score differentiation."""
    
    all_scores = []
    query_stats = []
    
    for query in queries:
        results = list(search_client.search(
            search_text=query,
            scoring_profile=profile_name,
            top=20
        ))
        
        scores = [r['@search.score'] for r in results]
        all_scores.extend(scores)
        
        if scores:
            query_stats.append({
                'query': query,
                'range': max(scores) - min(scores),
                'std': np.std(scores)
            })
    
    # Overall statistics
    overall_std = np.std(all_scores)
    overall_range = max(all_scores) - min(all_scores)
    
    print(f"Overall score statistics:")
    print(f"  Range: {min(all_scores):.2f} - {max(all_scores):.2f}")
    print(f"  Std deviation: {overall_std:.2f}")
    
    # Check for problems
    if overall_std < 2.0:
        print("\n⚠️  WARNING: Low score variance - results not well differentiated")
        print("   Consider increasing boost values")
    
    if overall_range > 100:
        print("\n⚠️  WARNING: Very high score range - possible runaway boosting")
        print("   Consider reducing boost values")
    
    # Per-query variance
    low_variance_queries = [q for q in query_stats if q['std'] < 1.0]
    if len(low_variance_queries) > len(queries) * 0.3:
        print(f"\n⚠️  WARNING: {len(low_variance_queries)} queries have low score variance")

# Usage:
test_queries = ['laptop', 'headphones', 'monitor', 'keyboard', 'mouse']
check_score_variance(search_client, test_queries, 'ecommerce_relevance')
```

**Solutions**:

- **Low variance**: Increase boost values (2.0 → 4.0)
- **High variance**: Reduce boost values or use AVERAGE aggregation
- **Scores increasing over time**: Freshness function without upper bound—add `boosting_duration`

### Issue 4: Distance Function Not Working

**Symptoms**:
- Geo-proximity results not prioritized
- Distance parameter seems ignored
- Results identical regardless of reference point

**Root Causes**:

**1. Reference point not passed correctly**
```python
# BAD: Wrong parameter format
results = search_client.search(
    "coffee shop",
    scoring_profile="geo_proximity",
    scoring_parameters=["currentLocation:-122.3321,47.6062"]  # ❌ Wrong order (lon,lat not lat,lon)
)

# GOOD: Correct format (lon,lat per GeoJSON spec)
results = search_client.search(
    "coffee shop",
    scoring_profile="geo_proximity",
    scoring_parameters=["currentLocation--122.3321,47.6062"]  # ✓ lon,lat
)
```

**2. Boosting distance too large**
```python
# BAD: 500km boosting distance (too broad)
distance_fn = DistanceScoringFunction(
    field_name="location",
    boost=3.0,
    parameters=DistanceScoringParameters(
        reference_point_parameter="currentLocation",
        boosting_distance=500  # 500 km - everything nearby!
    )
)

# GOOD: Appropriate distance for use case
distance_fn = DistanceScoringFunction(
    field_name="location",
    boost=3.0,
    parameters=DistanceScoringParameters(
        reference_point_parameter="currentLocation",
        boosting_distance=10  # 10 km for local search
    )
)
```

**3. Location field NULL or malformed**
```python
# Check location field coverage
def verify_location_field(search_client, field_name="location"):
    """Verify location field has valid coordinates."""
    results = list(search_client.search(
        search_text="*",
        select=[field_name],
        top=100
    ))
    
    valid_count = 0
    for r in results:
        loc = r.get(field_name)
        if loc and hasattr(loc, 'coordinates'):
            valid_count += 1
    
    coverage = (valid_count / len(results)) * 100
    print(f"Location field coverage: {coverage:.0f}%")
    
    if coverage < 80:
        print(f"⚠️  Only {coverage:.0f}% of documents have valid coordinates")
        print(f"   Distance function will have limited impact")

verify_location_field(search_client)
```

### Issue 5: Tag Function Not Matching

**Symptoms**:
- Tag boosting not working
- Documents with matching tags not boosted
- Personalization has no effect

**Root Causes**:

**1. Tag parameter format incorrect**
```python
# BAD: Wrong format for tags
results = search_client.search(
    "laptop",
    scoring_profile="personalized",
    scoring_parameters=["preferredTags-gaming"]  # ❌ Single tag
)

# GOOD: Comma-separated tags
results = search_client.search(
    "laptop",
    scoring_profile="personalized",
    scoring_parameters=["preferredTags-gaming,performance,high-end"]  # ✓ Multiple tags
)
```

**2. Tag field type mismatch**
```python
# Profile expects Collection(String) but field is String
tag_fn = TagScoringFunction(
    field_name="tags",  # Defined as String, not Collection(String)
    boost=3.0,
    parameters=TagScoringParameters(tags_parameter="preferredTags")
)

# Solution: Use Collection field type
SearchField(
    name="tags",
    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
    filterable=True
)
```

---

## Next Steps

- **[Index Management](./15-index-management.md)** - Updating scoring profiles
- **[A/B Testing](./17-ab-testing-framework.md)** - Testing profile effectiveness
- **[Evaluation Frameworks](./02-evaluation-frameworks.md)** - Measuring relevance improvements

---

*See also: [Full-Text Search](./08-fulltext-search-bm25.md) | [Query Optimization](./12-query-optimization.md)*