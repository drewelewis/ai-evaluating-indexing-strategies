# Semantic Search - Part 2: Best Practices

## Comprehensive Best Practices for Semantic Ranking Implementation

This guide provides 8 comprehensive best practice categories for implementing and optimizing semantic search in Azure AI Search, based on production deployments and lessons learned from enterprise implementations.

---

## 1. Field Configuration Optimization

### Overview

Semantic field configuration is the foundation of semantic ranking quality. The `SemanticConfiguration` defines which fields the semantic models analyze and how they're prioritized.

### Field Priority Hierarchy

```python
from azure.search.documents.indexes.models import (
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields
)

# Optimal field configuration pattern
SemanticConfiguration(
    name="optimal-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        # TITLE: Single most important field (highest weight)
        title_field=SemanticField(field_name="title"),
        
        # CONTENT: Main body text (medium weight, multiple fields allowed)
        content_fields=[
            SemanticField(field_name="summary"),    # Prioritize concise summary
            SemanticField(field_name="description"), # Then longer description
            SemanticField(field_name="body")        # Finally full body text
        ],
        
        # KEYWORDS: Supporting context (lower weight)
        keywords_fields=[
            SemanticField(field_name="category"),
            SemanticField(field_name="tags"),
            SemanticField(field_name="metadata")
        ]
    )
)
```

### Best Practices

**✅ DO:**

1. **Use title field for primary identifier**
   ```python
   # Good: Clear, descriptive title
   title_field=SemanticField(field_name="productName")
   
   # Bad: Don't use IDs or codes as title
   # title_field=SemanticField(field_name="sku")  # ❌
   ```

2. **Order content fields by importance**
   ```python
   # Good: Most relevant first
   content_fields=[
       SemanticField(field_name="summary"),      # 200-300 words, key points
       SemanticField(field_name="description"),  # 500-1000 words, details
       SemanticField(field_name="fullText")      # Complete document
   ]
   ```

3. **Limit content fields to 3-4 maximum**
   - Semantic models have limited attention span
   - Too many fields dilute focus
   - Better: Combine fields at index time if needed

4. **Use keywords fields for categorical data**
   ```python
   keywords_fields=[
       SemanticField(field_name="category"),      # "Electronics > Laptops"
       SemanticField(field_name="brand"),         # "Dell"
       SemanticField(field_name="tags")           # ["business", "portable"]
   ]
   ```

**❌ DON'T:**

1. **Don't include non-textual fields**
   ```python
   # Bad: Numeric fields don't benefit from semantic ranking
   content_fields=[
       SemanticField(field_name="description"),
       SemanticField(field_name="price"),        # ❌ Numeric
       SemanticField(field_name="stockCount")    # ❌ Numeric
   ]
   ```

2. **Don't include very long text (>10,000 words)**
   - Semantic models truncate after ~5,000 tokens
   - Use field splitting or summarization

3. **Don't duplicate fields across categories**
   ```python
   # Bad: Field used in multiple categories
   title_field=SemanticField(field_name="title"),
   content_fields=[
       SemanticField(field_name="title"),  # ❌ Already in title_field
       SemanticField(field_name="body")
   ]
   ```

### Field Configuration Testing Framework

```python
class SemanticConfigTester:
    """Test different semantic configurations to find optimal setup."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def test_configurations(self, test_queries, relevance_judgments):
        """
        Test multiple semantic configurations.
        
        Args:
            test_queries: List of representative queries
            relevance_judgments: Dict mapping (query, doc_id) -> relevance_score
            
        Returns:
            Configuration performance comparison
        """
        configs_to_test = [
            {
                'name': 'title-only',
                'title': 'productName',
                'content': ['description'],
                'keywords': []
            },
            {
                'name': 'title-summary',
                'title': 'productName',
                'content': ['summary', 'description'],
                'keywords': ['category']
            },
            {
                'name': 'title-all-content',
                'title': 'productName',
                'content': ['summary', 'description', 'specifications', 'reviews'],
                'keywords': ['category', 'brand', 'tags']
            }
        ]
        
        results = {}
        
        for config in configs_to_test:
            ndcg_scores = []
            
            for query in test_queries:
                search_results = self.search_client.search(
                    search_text=query,
                    query_type="semantic",
                    semantic_configuration_name=config['name'],
                    top=10
                )
                
                # Calculate NDCG@10
                ndcg = self._calculate_ndcg(
                    search_results, 
                    relevance_judgments.get(query, {})
                )
                ndcg_scores.append(ndcg)
            
            results[config['name']] = {
                'avg_ndcg': sum(ndcg_scores) / len(ndcg_scores),
                'min_ndcg': min(ndcg_scores),
                'max_ndcg': max(ndcg_scores),
                'config': config
            }
        
        return results
    
    def _calculate_ndcg(self, search_results, relevance_judgments, k=10):
        """Calculate NDCG@k for search results."""
        import math
        
        dcg = 0.0
        for i, result in enumerate(list(search_results)[:k]):
            doc_id = result['id']
            relevance = relevance_judgments.get(doc_id, 0)
            dcg += (2**relevance - 1) / math.log2(i + 2)
        
        # Calculate ideal DCG
        ideal_relevances = sorted(relevance_judgments.values(), reverse=True)[:k]
        idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        return dcg / idcg if idcg > 0 else 0.0

# Usage
tester = SemanticConfigTester(search_client)

test_queries = [
    "best laptop for programming",
    "affordable gaming computer",
    "business laptop with long battery life"
]

# Relevance judgments (query -> doc_id -> score)
judgments = {
    "best laptop for programming": {
        "doc123": 3,  # Highly relevant
        "doc456": 2,  # Relevant
        "doc789": 1   # Somewhat relevant
    }
}

results = tester.test_configurations(test_queries, judgments)

print("Configuration Performance:")
for config_name, metrics in results.items():
    print(f"{config_name}: NDCG@10 = {metrics['avg_ndcg']:.4f}")
```

### Decision Framework: Field Selection

Use this decision tree for field configuration:

```
Is the field text-based?
├─ No → Exclude from semantic config
└─ Yes
    │
    ├─ Is it the primary identifier? (product name, article title, etc.)
    │  └─ Yes → Use as title_field
    │
    ├─ Does it contain main descriptive content?
    │  └─ Yes → Add to content_fields (order by importance)
    │
    └─ Is it categorical/metadata? (tags, categories, etc.)
       └─ Yes → Add to keywords_fields
```

---

## 2. Query Type Classification & Routing

### Overview

Not all queries benefit equally from semantic ranking. Intelligent query classification routes queries to the optimal search strategy, balancing relevance, latency, and cost.

### Query Classification Framework

```python
import re
from typing import Literal

QueryType = Literal['semantic_question', 'semantic_descriptive', 'keyword', 'exact_match']

class QueryClassifier:
    """Classify queries to determine optimal search strategy."""
    
    def __init__(self):
        # Question words that indicate natural language queries
        self.question_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 
            'which', 'whose', 'whom',
            'can', 'could', 'would', 'should', 'will',
            'is', 'are', 'was', 'were', 'does', 'do', 'did'
        }
        
        # Patterns for exact match queries
        self.exact_match_patterns = [
            r'^[A-Z0-9-]{8,}$',           # SKU/Product codes (ABC-12345)
            r'^\d{10,}$',                 # ISBN, UPC codes
            r'^[0-9]{3}-[0-9]{2}-[0-9]{4}$'  # Formatted IDs (123-45-6789)
        ]
    
    def classify(self, query: str) -> QueryType:
        """
        Classify query into one of four types.
        
        Returns:
            - 'semantic_question': Natural language question
            - 'semantic_descriptive': Descriptive search (benefits from semantic)
            - 'keyword': Simple keyword search
            - 'exact_match': Exact code/ID lookup
        """
        query_clean = query.strip()
        query_lower = query_clean.lower()
        words = query_lower.split()
        
        # Check for exact match patterns
        for pattern in self.exact_match_patterns:
            if re.match(pattern, query_clean):
                return 'exact_match'
        
        # Very short queries (1-2 words) are typically keyword searches
        if len(words) <= 2:
            return 'keyword'
        
        # Check for question format
        is_question = (
            words[0] in self.question_words or  # Starts with question word
            query_clean.endswith('?') or         # Ends with question mark
            self._has_question_structure(query_lower)
        )
        
        if is_question:
            return 'semantic_question'
        
        # Longer descriptive queries (3+ words, not questions)
        if len(words) >= 3:
            # Check if descriptive (adjectives, quality terms)
            if self._is_descriptive(query_lower):
                return 'semantic_descriptive'
        
        # Default to keyword search
        return 'keyword'
    
    def _has_question_structure(self, query: str) -> bool:
        """Detect question structure patterns."""
        question_patterns = [
            r'\bhow (to|do|does|can)\b',
            r'\bwhat (is|are|was|were)\b',
            r'\bwhy (is|are|do|does)\b',
            r'\bwhen (to|should|do)\b',
            r'\bwhich (is|are)\b'
        ]
        return any(re.search(pattern, query) for pattern in question_patterns)
    
    def _is_descriptive(self, query: str) -> bool:
        """Check if query contains descriptive qualifiers."""
        descriptive_words = {
            'best', 'top', 'good', 'great', 'affordable', 'cheap', 'expensive',
            'high-quality', 'reliable', 'durable', 'efficient', 'fast', 'powerful',
            'comfortable', 'stylish', 'modern', 'professional', 'advanced'
        }
        return any(word in query for word in descriptive_words)

# Usage
classifier = QueryClassifier()

queries = [
    "What is the best laptop for video editing?",      # semantic_question
    "affordable gaming laptop with RTX graphics",      # semantic_descriptive
    "Dell XPS 15",                                     # keyword
    "SKU-ABC-12345"                                    # exact_match
]

for query in queries:
    query_type = classifier.classify(query)
    print(f"{query:50} → {query_type}")
```

### Routing Strategy Implementation

```python
from azure.search.documents.models import QueryType as AzureQueryType

class SemanticRouter:
    """Route queries to optimal search strategy based on classification."""
    
    def __init__(self, search_client, semantic_config_name="default"):
        self.search_client = search_client
        self.semantic_config_name = semantic_config_name
        self.classifier = QueryClassifier()
    
    def search(self, query: str, top: int = 10):
        """
        Execute search with optimal strategy based on query type.
        
        Args:
            query: User's search query
            top: Number of results to return
            
        Returns:
            Search results with metadata about strategy used
        """
        query_type = self.classifier.classify(query)
        
        if query_type == 'exact_match':
            return self._exact_match_search(query, top)
        elif query_type == 'keyword':
            return self._keyword_search(query, top)
        elif query_type in ['semantic_question', 'semantic_descriptive']:
            return self._semantic_search(query, top, query_type)
    
    def _exact_match_search(self, query, top):
        """Exact match search (filter-based, fastest)."""
        results = self.search_client.search(
            search_text=None,  # No full-text search
            filter=f"id eq '{query}' or sku eq '{query}'",
            top=top
        )
        return {
            'results': list(results),
            'strategy': 'exact_match',
            'estimated_latency_ms': 10
        }
    
    def _keyword_search(self, query, top):
        """Standard BM25 keyword search."""
        results = self.search_client.search(
            search_text=query,
            query_type=AzureQueryType.SIMPLE,
            top=top
        )
        return {
            'results': list(results),
            'strategy': 'keyword',
            'estimated_latency_ms': 30
        }
    
    def _semantic_search(self, query, top, query_type):
        """Semantic search with captions and answers."""
        from azure.search.documents.models import (
            QueryCaptionType, QueryAnswerType
        )
        
        # Enable answers only for question queries
        enable_answers = (query_type == 'semantic_question')
        
        results = self.search_client.search(
            search_text=query,
            query_type=AzureQueryType.SEMANTIC,
            semantic_configuration_name=self.semantic_config_name,
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE if enable_answers else None,
            query_answer_count=3 if enable_answers else 0,
            top=top
        )
        
        return {
            'results': list(results),
            'strategy': f'semantic_{query_type}',
            'estimated_latency_ms': 150,
            'semantic_features': {
                'captions': True,
                'answers': enable_answers
            }
        }

# Usage
router = SemanticRouter(search_client, semantic_config_name="products-semantic")

# Different query types routed appropriately
test_queries = [
    "What are the benefits of SSD storage?",           # → semantic (question)
    "lightweight laptop with good battery life",       # → semantic (descriptive)
    "Dell XPS",                                        # → keyword
    "SKU-12345"                                        # → exact match
]

for query in test_queries:
    result = router.search(query, top=5)
    print(f"Query: {query}")
    print(f"  Strategy: {result['strategy']}")
    print(f"  Latency: ~{result['estimated_latency_ms']}ms")
    print(f"  Results: {len(result['results'])}")
    print()
```

### Routing Performance Monitoring

```python
class RoutingMetrics:
    """Monitor routing decisions and performance."""
    
    def __init__(self):
        self.metrics = {
            'exact_match': {'count': 0, 'latency_sum': 0},
            'keyword': {'count': 0, 'latency_sum': 0},
            'semantic_question': {'count': 0, 'latency_sum': 0},
            'semantic_descriptive': {'count': 0, 'latency_sum': 0}
        }
    
    def record(self, strategy: str, latency_ms: float):
        """Record routing decision and latency."""
        if strategy in self.metrics:
            self.metrics[strategy]['count'] += 1
            self.metrics[strategy]['latency_sum'] += latency_ms
    
    def get_stats(self):
        """Get routing statistics."""
        total_queries = sum(m['count'] for m in self.metrics.values())
        
        stats = {}
        for strategy, data in self.metrics.items():
            count = data['count']
            if count > 0:
                stats[strategy] = {
                    'count': count,
                    'percentage': (count / total_queries) * 100,
                    'avg_latency_ms': data['latency_sum'] / count
                }
        
        return stats
    
    def print_report(self):
        """Print routing report."""
        stats = self.get_stats()
        print("Query Routing Report")
        print("=" * 70)
        print(f"{'Strategy':<25} {'Count':<10} {'%':<10} {'Avg Latency':<15}")
        print("-" * 70)
        
        for strategy, data in stats.items():
            print(f"{strategy:<25} {data['count']:<10} "
                  f"{data['percentage']:<9.1f}% {data['avg_latency_ms']:<14.1f}ms")

# Usage with router
metrics = RoutingMetrics()

for query in user_queries:  # Process real user queries
    start_time = time.time()
    result = router.search(query)
    latency_ms = (time.time() - start_time) * 1000
    
    metrics.record(result['strategy'], latency_ms)

# Print report
metrics.print_report()

# Output:
# Query Routing Report
# ======================================================================
# Strategy                  Count      %          Avg Latency    
# ----------------------------------------------------------------------
# keyword                   4521       45.2%      32.4ms
# semantic_question         2834       28.3%      147.8ms
# semantic_descriptive      2103       21.0%      138.2ms
# exact_match               542        5.4%       12.1ms
```

### Best Practices for Routing

**✅ DO:**

1. **Monitor routing accuracy** - Sample queries and validate classification
2. **A/B test routing logic** - Measure impact of classification changes
3. **Provide fallback** - If semantic fails, fall back to keyword search
4. **Log routing decisions** - Debug classification issues
5. **Tune thresholds** - Adjust word count, pattern matching based on data

**❌ DON'T:**

1. **Don't over-optimize for edge cases** - Focus on common query patterns
2. **Don't route everything to semantic** - Wastes latency budget and cost
3. **Don't ignore latency impact** - Monitor p95 latency by route type
4. **Don't forget caching** - Cache routing decisions for common queries

---

## 3. Semantic Captions Optimization

### Overview

Semantic captions are extractive summaries highlighting why a document is relevant. Optimizing captions improves CTR and user satisfaction.

### Caption Configuration

```python
from azure.search.documents.models import QueryCaptionType

class SemanticCaptionOptimizer:
    """Optimize semantic caption generation."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def search_with_optimized_captions(
        self,
        query: str,
        max_captions: int = 3,
        highlight_pre_tag: str = "<strong>",
        highlight_post_tag: str = "</strong>",
        top: int = 10
    ):
        """
        Search with optimized caption settings.
        
        Args:
            query: Search query
            max_captions: Maximum captions per document (1-5)
            highlight_pre_tag: HTML tag before highlight
            highlight_post_tag: HTML tag after highlight
            top: Number of results
            
        Returns:
            Search results with optimized captions
        """
        results = self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="default",
            query_caption=QueryCaptionType.EXTRACTIVE,
            top=top,
            # Caption-specific parameters
            highlight_pre_tag=highlight_pre_tag,
            highlight_post_tag=highlight_post_tag
        )
        
        # Post-process captions
        processed_results = []
        for result in results:
            captions = self._process_captions(
                result.get('@search.captions', []),
                max_count=max_captions
            )
            
            processed_results.append({
                'id': result['id'],
                'title': result.get('title'),
                'score': result.get('@search.score'),
                'reranker_score': result.get('@search.rerankerScore'),
                'captions': captions
            })
        
        return processed_results
    
    def _process_captions(self, raw_captions, max_count):
        """Process and rank captions."""
        if not raw_captions:
            return []
        
        # Sort by length (prefer longer captions)
        captions = sorted(
            raw_captions,
            key=lambda c: len(c.get('text', '')),
            reverse=True
        )[:max_count]
        
        processed = []
        for caption in captions:
            processed.append({
                'text': caption.get('text', ''),
                'highlights': caption.get('highlights', caption.get('text', '')),
                'length': len(caption.get('text', ''))
            })
        
        return processed

# Usage
optimizer = SemanticCaptionOptimizer(search_client)

results = optimizer.search_with_optimized_captions(
    query="What is the best laptop for video editing?",
    max_captions=2,  # Show top 2 captions per result
    highlight_pre_tag="<mark>",
    highlight_post_tag="</mark>",
    top=10
)

for result in results:
    print(f"\nTitle: {result['title']}")
    print(f"Reranker Score: {result['reranker_score']:.4f}")
    print("Captions:")
    for i, caption in enumerate(result['captions'], 1):
        print(f"  {i}. {caption['highlights']}")
```

### Caption Quality Metrics

```python
class CaptionQualityAnalyzer:
    """Analyze semantic caption quality."""
    
    def __init__(self):
        self.caption_lengths = []
        self.highlight_ratios = []
        self.caption_counts = []
    
    def analyze_captions(self, search_results):
        """Analyze caption quality metrics."""
        for result in search_results:
            captions = result.get('@search.captions', [])
            self.caption_counts.append(len(captions))
            
            for caption in captions:
                text = caption.get('text', '')
                highlights = caption.get('highlights', '')
                
                # Caption length
                self.caption_lengths.append(len(text.split()))
                
                # Highlight ratio (highlighted words / total words)
                if text:
                    highlight_ratio = self._calculate_highlight_ratio(text, highlights)
                    self.highlight_ratios.append(highlight_ratio)
    
    def _calculate_highlight_ratio(self, text, highlights):
        """Calculate percentage of text that's highlighted."""
        # Count <em> tags in highlights
        em_count = highlights.count('<em>')
        total_words = len(text.split())
        return (em_count / total_words) if total_words > 0 else 0
    
    def get_stats(self):
        """Get caption quality statistics."""
        return {
            'avg_caption_length': sum(self.caption_lengths) / len(self.caption_lengths),
            'avg_captions_per_result': sum(self.caption_counts) / len(self.caption_counts),
            'avg_highlight_ratio': sum(self.highlight_ratios) / len(self.highlight_ratios),
            'total_results_analyzed': len(self.caption_counts)
        }

# Usage
analyzer = CaptionQualityAnalyzer()

# Analyze captions from multiple queries
for query in test_queries:
    results = search_client.search(
        search_text=query,
        query_type="semantic",
        semantic_configuration_name="default",
        query_caption=QueryCaptionType.EXTRACTIVE,
        top=10
    )
    analyzer.analyze_captions(list(results))

stats = analyzer.get_stats()
print(f"Average caption length: {stats['avg_caption_length']:.1f} words")
print(f"Average captions per result: {stats['avg_captions_per_result']:.1f}")
print(f"Average highlight ratio: {stats['avg_highlight_ratio']:.1%}")

# Target benchmarks:
# - Caption length: 30-60 words (readable snippet)
# - Captions per result: 2-3 (multiple perspectives)
# - Highlight ratio: 15-30% (enough context, not too much)
```

### Best Practices for Captions

**✅ DO:**

1. **Show 2-3 captions per result** - Multiple perspectives increase CTR
2. **Use semantic HTML for highlights** - `<mark>` or `<strong>` tags
3. **Truncate very long captions** - 60-80 words maximum for display
4. **A/B test caption presentation** - Measure impact on CTR
5. **Show caption source** - Indicate which field caption came from

**❌ DON'T:**

1. **Don't show truncated sentences** - Ensure captions are complete
2. **Don't over-highlight** - Too much highlighting reduces usefulness
3. **Don't ignore mobile display** - Test caption rendering on mobile
4. **Don't show duplicate captions** - Deduplicate similar captions

---

## 4. Semantic Answers Implementation

### Overview

Semantic answers extract direct responses to question queries, reducing search friction and improving satisfaction.

### Answer Extraction Strategy

```python
from azure.search.documents.models import QueryAnswerType

class SemanticAnswerExtractor:
    """Extract and present semantic answers."""
    
    def __init__(self, search_client, semantic_config_name="default"):
        self.search_client = search_client
        self.semantic_config_name = semantic_config_name
    
    def search_with_answers(
        self,
        query: str,
        answer_count: int = 3,
        answer_threshold: float = 0.7,
        top: int = 20
    ):
        """
        Search with semantic answer extraction.
        
        Args:
            query: Question query
            answer_count: Maximum answers to extract (1-5)
            answer_threshold: Minimum confidence score (0.0-1.0)
            top: Number of documents to search
            
        Returns:
            Dict with answers and supporting documents
        """
        results = self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=self.semantic_config_name,
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            query_answer_count=answer_count,
            top=top
        )
        
        # Extract and filter answers
        answers = []
        documents = []
        
        for result in results:
            # Process answers
            if '@search.answers' in result:
                for answer in result['@search.answers']:
                    score = answer.get('score', 0)
                    if score >= answer_threshold:
                        answers.append({
                            'text': answer.get('text', ''),
                            'highlights': answer.get('highlights', ''),
                            'score': score,
                            'source_id': result.get('id'),
                            'source_title': result.get('title')
                        })
            
            # Process documents
            documents.append({
                'id': result.get('id'),
                'title': result.get('title'),
                'reranker_score': result.get('@search.rerankerScore'),
                'captions': result.get('@search.captions', [])
            })
        
        # Sort answers by score
        answers.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'query': query,
            'answers': answers[:answer_count],
            'documents': documents,
            'has_answers': len(answers) > 0
        }

# Usage
answer_extractor = SemanticAnswerExtractor(
    search_client,
    semantic_config_name="faq-semantic"
)

result = answer_extractor.search_with_answers(
    query="How do I reset my password?",
    answer_count=2,
    answer_threshold=0.8,  # High confidence only
    top=20
)

if result['has_answers']:
    print(f"Question: {result['query']}\n")
    print("Answers:")
    for i, answer in enumerate(result['answers'], 1):
        print(f"\n{i}. {answer['highlights']}")
        print(f"   Confidence: {answer['score']:.2f}")
        print(f"   Source: {answer['source_title']}")
else:
    print("No high-confidence answers found.")
    print(f"Showing {len(result['documents'])} relevant documents.")
```

### Answer Quality Monitoring

```python
class AnswerQualityMonitor:
    """Monitor semantic answer quality and performance."""
    
    def __init__(self):
        self.total_queries = 0
        self.queries_with_answers = 0
        self.answer_scores = []
        self.answer_lengths = []
        self.false_positive_count = 0  # Wrong answers
    
    def record_query(self, query, result, user_feedback=None):
        """
        Record query result and optional user feedback.
        
        Args:
            query: Original query
            result: Search result with answers
            user_feedback: Optional user feedback ('helpful' | 'wrong' | None)
        """
        self.total_queries += 1
        
        if result['has_answers']:
            self.queries_with_answers += 1
            
            for answer in result['answers']:
                self.answer_scores.append(answer['score'])
                self.answer_lengths.append(len(answer['text'].split()))
        
        # Record user feedback
        if user_feedback == 'wrong':
            self.false_positive_count += 1
    
    def get_metrics(self):
        """Get answer quality metrics."""
        answer_rate = (self.queries_with_answers / self.total_queries) * 100 if self.total_queries > 0 else 0
        
        metrics = {
            'total_queries': self.total_queries,
            'answer_rate': answer_rate,
            'avg_answer_score': sum(self.answer_scores) / len(self.answer_scores) if self.answer_scores else 0,
            'avg_answer_length': sum(self.answer_lengths) / len(self.answer_lengths) if self.answer_lengths else 0,
            'false_positive_rate': (self.false_positive_count / self.queries_with_answers) * 100 if self.queries_with_answers > 0 else 0
        }
        
        return metrics
    
    def print_report(self):
        """Print answer quality report."""
        metrics = self.get_metrics()
        
        print("Semantic Answer Quality Report")
        print("=" * 50)
        print(f"Total queries: {metrics['total_queries']}")
        print(f"Answer rate: {metrics['answer_rate']:.1f}%")
        print(f"Avg answer confidence: {metrics['avg_answer_score']:.2f}")
        print(f"Avg answer length: {metrics['avg_answer_length']:.1f} words")
        print(f"False positive rate: {metrics['false_positive_rate']:.1f}%")
        print()
        print("Benchmarks:")
        print("  Answer rate: 30-50% (question queries)")
        print("  Avg confidence: >0.75 (high quality)")
        print("  Avg length: 20-50 words (concise)")
        print("  False positive rate: <5%")

# Usage
monitor = AnswerQualityMonitor()

# Process queries and collect user feedback
for query in question_queries:
    result = answer_extractor.search_with_answers(query)
    user_feedback = collect_user_feedback(result)  # 'helpful' | 'wrong' | None
    monitor.record_query(query, result, user_feedback)

monitor.print_report()
```

### Best Practices for Answers

**✅ DO:**

1. **Use high confidence threshold** (0.7-0.8) - Reduce false positives
2. **Show answer source** - Link to full document
3. **Enable answer voting** - Collect user feedback (helpful/not helpful)
4. **Fall back gracefully** - If no answers, show regular results
5. **Test with question queries** - Answers work best for factual questions

**❌ DON'T:**

1. **Don't show low-confidence answers** (<0.6) - Too many errors
2. **Don't hide source documents** - Users may want more context
3. **Don't assume answers are perfect** - Monitor false positive rate
4. **Don't use for non-question queries** - Answer extraction fails on keyword searches

---

## 5. Performance & Latency Optimization

### Overview

Semantic ranking adds 80-150ms latency overhead. Optimization strategies balance relevance improvements with acceptable performance.

### Latency Reduction Techniques

```python
import time
from functools import wraps

class SemanticLatencyOptimizer:
    """Optimize semantic search latency."""
    
    def __init__(self, search_client):
        self.search_client = search_client
        self.latency_cache = {}
    
    def optimized_semantic_search(
        self,
        query: str,
        use_cache: bool = True,
        top: int = 10,
        timeout_ms: int = 300
    ):
        """
        Execute semantic search with optimization.
        
        Optimizations:
        1. Result caching for common queries
        2. Timeout protection
        3. Adaptive top-k (reduce if slow)
        4. Parallel processing where possible
        """
        # Check cache
        if use_cache and query in self.latency_cache:
            return self.latency_cache[query]
        
        start_time = time.time()
        
        try:
            # Execute with timeout
            results = self._search_with_timeout(
                query=query,
                top=top,
                timeout_ms=timeout_ms
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Cache if latency acceptable
            if use_cache and latency_ms < 200:
                self.latency_cache[query] = results
            
            return {
                'results': results,
                'latency_ms': latency_ms,
                'cached': False
            }
            
        except TimeoutError:
            # Fall back to keyword search
            return self._fallback_keyword_search(query, top)
    
    def _search_with_timeout(self, query, top, timeout_ms):
        """Execute search with timeout."""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                self._execute_semantic_search,
                query, top
            )
            
            try:
                results = future.result(timeout=timeout_ms / 1000.0)
                return results
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError(f"Search exceeded {timeout_ms}ms timeout")
    
    def _execute_semantic_search(self, query, top):
        """Execute actual semantic search."""
        results = self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="default",
            query_caption=QueryCaptionType.EXTRACTIVE,
            top=top
        )
        return list(results)
    
    def _fallback_keyword_search(self, query, top):
        """Fallback to fast keyword search if semantic times out."""
        results = self.search_client.search(
            search_text=query,
            query_type="simple",
            top=top
        )
        
        return {
            'results': list(results),
            'latency_ms': 50,  # Estimated
            'cached': False,
            'fallback': True,
            'reason': 'semantic_timeout'
        }

# Usage
optimizer = SemanticLatencyOptimizer(search_client)

result = optimizer.optimized_semantic_search(
    query="What is the best laptop for programming?",
    use_cache=True,
    top=10,
    timeout_ms=250  # Fall back if exceeds 250ms
)

print(f"Latency: {result['latency_ms']:.1f}ms")
print(f"Cached: {result.get('cached', False)}")
print(f"Fallback: {result.get('fallback', False)}")
```

### Best Practices for Performance

**✅ DO:**

1. **Cache common queries** - 20-30% of queries are repeated
2. **Set reasonable timeouts** - Fall back to keyword if semantic slow
3. **Monitor p95/p99 latency** - Don't just look at averages
4. **Use CDN for static content** - Reduce network latency
5. **Optimize initial retrieval** - Fast BM25 = faster semantic

**❌ DON'T:**

1. **Don't request too many results** - Semantic re-ranks top 50 max
2. **Don't ignore latency regressions** - Monitor continuously
3. **Don't cache low-quality results** - Only cache good experiences
4. **Don't forget regional latency** - Test from multiple locations

---

## 6. Cost Management

### Overview

Azure AI Search semantic ranking requires Standard tier (~$500/month). Effective cost management ensures ROI.

### Cost Monitoring

```python
class SemanticCostTracker:
    """Track semantic search usage and costs."""
    
    def __init__(self, monthly_budget: float = 500.0):
        self.monthly_budget = monthly_budget
        self.query_counts = {
            'semantic': 0,
            'keyword': 0,
            'total': 0
        }
        self.monthly_cost = 500.0  # Standard tier base cost
    
    def record_query(self, query_type: str):
        """Record query by type."""
        self.query_counts[query_type] = self.query_counts.get(query_type, 0) + 1
        self.query_counts['total'] += 1
    
    def get_usage_stats(self):
        """Get usage statistics."""
        total = self.query_counts['total']
        semantic_count = self.query_counts.get('semantic', 0)
        
        return {
            'total_queries': total,
            'semantic_queries': semantic_count,
            'semantic_percentage': (semantic_count / total * 100) if total > 0 else 0,
            'estimated_monthly_cost': self.monthly_cost,
            'cost_per_query': (self.monthly_cost / total) if total > 0 else 0,
            'cost_per_semantic_query': (self.monthly_cost / semantic_count) if semantic_count > 0 else 0
        }
    
    def print_cost_report(self):
        """Print cost analysis report."""
        stats = self.get_usage_stats()
        
        print("Semantic Search Cost Report")
        print("=" * 60)
        print(f"Total queries: {stats['total_queries']:,}")
        print(f"Semantic queries: {stats['semantic_queries']:,} ({stats['semantic_percentage']:.1f}%)")
        print(f"Monthly cost: ${stats['estimated_monthly_cost']:.2f}")
        print(f"Cost per query: ${stats['cost_per_query']:.4f}")
        print(f"Cost per semantic query: ${stats['cost_per_semantic_query']:.4f}")
        print()
        print("Cost optimization opportunities:")
        if stats['semantic_percentage'] > 60:
            print("  ⚠️  High semantic usage (>60%) - consider query routing")
        if stats['cost_per_query'] > 0.01:
            print("  ⚠️  High cost per query - increase query volume or optimize")

# Usage
cost_tracker = SemanticCostTracker(monthly_budget=500.0)

# Record queries
for query in production_queries:
    query_type = classify_and_route(query)
    cost_tracker.record_query(query_type)

cost_tracker.print_cost_report()
```

### Best Practices for Cost Management

**✅ DO:**

1. **Use query routing** - Route only beneficial queries to semantic
2. **Monitor usage** - Track semantic query percentage (target: 30-50%)
3. **Cache aggressively** - Reduce redundant semantic calls
4. **Start with lower tier** - Upgrade only when needed
5. **Calculate ROI** - Ensure benefits exceed costs

**❌ DON'T:**

1. **Don't use semantic for all queries** - Wastes budget on simple searches
2. **Don't ignore alternative solutions** - Vector search may be more cost-effective
3. **Don't forget scale** - Costs are fixed, benefits scale with volume
4. **Don't optimize prematurely** - Validate semantic value first

---

## 7. Testing & Validation

### Overview

Systematic testing ensures semantic ranking improves relevance without introducing regressions.

### A/B Testing Framework

```python
import random
from datetime import datetime

class SemanticABTest:
    """A/B test semantic ranking vs baseline."""
    
    def __init__(self, search_client, test_name: str, split: float = 0.5):
        self.search_client = search_client
        self.test_name = test_name
        self.split = split  # Percentage in treatment group
        self.results = {
            'control': [],
            'treatment': []
        }
    
    def search(self, query: str, user_id: str, top: int = 10):
        """
        Execute search with A/B test assignment.
        
        Args:
            query: Search query
            user_id: Unique user identifier
            top: Number of results
            
        Returns:
            Search results with variant assignment
        """
        # Deterministic assignment based on user_id
        variant = self._assign_variant(user_id)
        
        if variant == 'treatment':
            results = self._semantic_search(query, top)
        else:
            results = self._keyword_search(query, top)
        
        return {
            'results': results,
            'variant': variant,
            'query': query,
            'timestamp': datetime.now()
        }
    
    def _assign_variant(self, user_id: str) -> str:
        """Deterministically assign user to variant."""
        # Hash user_id to get consistent assignment
        hash_value = hash(user_id + self.test_name)
        return 'treatment' if (hash_value % 100) < (self.split * 100) else 'control'
    
    def _semantic_search(self, query, top):
        """Execute semantic search (treatment)."""
        return list(self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="default",
            query_caption=QueryCaptionType.EXTRACTIVE,
            top=top
        ))
    
    def _keyword_search(self, query, top):
        """Execute keyword search (control)."""
        return list(self.search_client.search(
            search_text=query,
            top=top
        ))
    
    def record_interaction(self, query: str, variant: str, clicked: bool, position: int = None):
        """Record user interaction for analysis."""
        self.results[variant].append({
            'query': query,
            'clicked': clicked,
            'position': position,
            'timestamp': datetime.now()
        })
    
    def get_results(self):
        """Calculate A/B test metrics."""
        control_ctr = self._calculate_ctr(self.results['control'])
        treatment_ctr = self._calculate_ctr(self.results['treatment'])
        
        return {
            'control': {
                'queries': len(self.results['control']),
                'ctr': control_ctr
            },
            'treatment': {
                'queries': len(self.results['treatment']),
                'ctr': treatment_ctr
            },
            'improvement': ((treatment_ctr - control_ctr) / control_ctr * 100) if control_ctr > 0 else 0
        }
    
    def _calculate_ctr(self, interactions):
        """Calculate click-through rate."""
        if not interactions:
            return 0
        clicks = sum(1 for i in interactions if i['clicked'])
        return clicks / len(interactions)

# Usage
ab_test = SemanticABTest(
    search_client,
    test_name="semantic_vs_keyword_v1",
    split=0.5  # 50/50 split
)

# Execute searches
for user_query in user_queries:
    result = ab_test.search(
        query=user_query['query'],
        user_id=user_query['user_id'],
        top=10
    )
    
    # Record if user clicked
    if user_clicked:
        ab_test.record_interaction(
            query=user_query['query'],
            variant=result['variant'],
            clicked=True,
            position=clicked_position
        )

# Analyze results
metrics = ab_test.get_results()
print(f"Control CTR: {metrics['control']['ctr']:.2%}")
print(f"Treatment CTR: {metrics['treatment']['ctr']:.2%}")
print(f"Improvement: {metrics['improvement']:.1f}%")
```

### Best Practices for Testing

**✅ DO:**

1. **Run A/B tests** - Don't assume semantic is better
2. **Measure multiple metrics** - CTR, dwell time, satisfaction
3. **Test for 14+ days** - Account for weekly patterns
4. **Validate statistical significance** - Use proper hypothesis testing
5. **Segment analysis** - Check different query types separately

**❌ DON'T:**

1. **Don't trust offline metrics only** - Online behavior differs
2. **Don't test on small samples** - Need statistical power
3. **Don't ignore negative results** - Semantic may hurt some queries
4. **Don't deploy without validation** - Test first, deploy later

---

## 8. Monitoring & Maintenance

### Overview

Production semantic search requires ongoing monitoring to maintain quality and catch regressions.

### Monitoring Dashboard

```python
class SemanticMonitoring:
    """Monitor semantic search health."""
    
    def __init__(self):
        self.metrics = {
            'semantic_queries': 0,
            'keyword_queries': 0,
            'avg_semantic_latency': [],
            'avg_keyword_latency': [],
            'caption_rate': 0,
            'answer_rate': 0,
            'error_count': 0
        }
    
    def record_search(self, query_type: str, latency_ms: float, 
                     has_captions: bool = False, has_answers: bool = False):
        """Record search execution."""
        if query_type == 'semantic':
            self.metrics['semantic_queries'] += 1
            self.metrics['avg_semantic_latency'].append(latency_ms)
            if has_captions:
                self.metrics['caption_rate'] += 1
            if has_answers:
                self.metrics['answer_rate'] += 1
        else:
            self.metrics['keyword_queries'] += 1
            self.metrics['avg_keyword_latency'].append(latency_ms)
    
    def record_error(self, error_type: str):
        """Record search error."""
        self.metrics['error_count'] += 1
    
    def get_health_status(self):
        """Get system health status."""
        total_queries = self.metrics['semantic_queries'] + self.metrics['keyword_queries']
        
        avg_semantic_latency = (
            sum(self.metrics['avg_semantic_latency']) / len(self.metrics['avg_semantic_latency'])
            if self.metrics['avg_semantic_latency'] else 0
        )
        
        health = {
            'status': 'healthy',
            'total_queries': total_queries,
            'semantic_percentage': (self.metrics['semantic_queries'] / total_queries * 100) if total_queries > 0 else 0,
            'avg_semantic_latency_ms': avg_semantic_latency,
            'caption_rate': (self.metrics['caption_rate'] / self.metrics['semantic_queries'] * 100) if self.metrics['semantic_queries'] > 0 else 0,
            'answer_rate': (self.metrics['answer_rate'] / self.metrics['semantic_queries'] * 100) if self.metrics['semantic_queries'] > 0 else 0,
            'error_rate': (self.metrics['error_count'] / total_queries * 100) if total_queries > 0 else 0
        }
        
        # Determine health status
        if health['error_rate'] > 1.0:
            health['status'] = 'unhealthy'
        elif health['avg_semantic_latency_ms'] > 300:
            health['status'] = 'degraded'
        
        return health

# Usage
monitor = SemanticMonitoring()

# Record searches
monitor.record_search('semantic', latency_ms=145, has_captions=True, has_answers=True)
monitor.record_search('keyword', latency_ms=35)

# Check health
health = monitor.get_health_status()
print(f"Status: {health['status']}")
print(f"Semantic %: {health['semantic_percentage']:.1f}%")
print(f"Avg latency: {health['avg_semantic_latency_ms']:.1f}ms")
print(f"Caption rate: {health['caption_rate']:.1f}%")
print(f"Answer rate: {health['answer_rate']:.1f}%")
```

### Best Practices for Monitoring

**✅ DO:**

1. **Monitor key metrics** - Latency, error rate, semantic %, CTR
2. **Set up alerts** - Notify on latency spikes or error increases
3. **Track trends** - Weekly/monthly performance trends
4. **Log sample queries** - Debug quality issues
5. **Review regularly** - Weekly health check meetings

**❌ DON'T:**

1. **Don't ignore slow drift** - Quality can degrade over time
2. **Don't over-alert** - Too many alerts = alert fatigue
3. **Don't forget segments** - Monitor by query type, user segment
4. **Don't neglect user feedback** - Qualitative data matters

---

## Summary

These 8 best practices provide a comprehensive framework for successful semantic search implementation:

1. **Field Configuration Optimization** - Foundation of semantic quality
2. **Query Routing** - Balance relevance, latency, and cost
3. **Semantic Captions** - Improve CTR and user experience
4. **Semantic Answers** - Reduce search friction for questions
5. **Performance Optimization** - Maintain acceptable latency
6. **Cost Management** - Ensure positive ROI
7. **Testing & Validation** - Validate improvements with A/B tests
8. **Monitoring & Maintenance** - Maintain quality over time

Following these practices ensures semantic ranking delivers maximum value while managing complexity, cost, and performance trade-offs.

---

**Document Statistics:**
- **Words**: ~5,800
- **Categories**: 8 comprehensive best practices
- **Code examples**: 15+ production-ready implementations
- **Decision frameworks**: Query classification, field selection, routing strategies
