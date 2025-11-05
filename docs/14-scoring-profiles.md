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

### ✅ Do's
1. **Start simple** with text weights before adding functions
2. **Test profiles** with real queries before production
3. **Use appropriate aggregation** (SUM for additive signals)
4. **Monitor score distributions** to detect anomalies
5. **Version profile names** (profile_v1, profile_v2)
6. **A/B test profile changes** with real users
7. **Document profile rationale** and parameter choices

### ❌ Don'ts
1. **Don't** over-boost (keep boost values < 10.0)
2. **Don't** use too many functions (3-5 maximum)
3. **Don't** ignore base BM25 scores (profiles augment, not replace)
4. **Don't** use constant_boost_beyond_range** without understanding impact
5. **Don't** assume more boost = better relevance
6. **Don't** skip evaluation (always measure impact)
7. **Don't** use distance functions without geo-coordinates

---

## Next Steps

- **[Index Management](./15-index-management.md)** - Updating scoring profiles
- **[A/B Testing](./17-ab-testing-framework.md)** - Testing profile effectiveness
- **[Evaluation Frameworks](./02-evaluation-frameworks.md)** - Measuring relevance improvements

---

*See also: [Full-Text Search](./08-fulltext-search-bm25.md) | [Query Optimization](./12-query-optimization.md)*