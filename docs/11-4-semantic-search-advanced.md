# Semantic Search - Part 4: Advanced Topics

## Advanced Implementation Patterns and Optimization Techniques

This section explores advanced patterns for production semantic search implementations, including multi-configuration strategies, performance tuning, cost optimization, and integration with other Azure AI services.

---

## 1. Multi-Configuration Strategies

### Overview

Different query types and content domains benefit from different semantic field configurations. Advanced implementations use multiple semantic configurations and intelligent routing.

### Pattern: Domain-Specific Configurations

```python
# Define multiple semantic configurations for different content types
product_semantic = SemanticConfiguration(
    name="products-semantic",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="productName"),
        content_fields=[
            SemanticField(field_name="marketing_description"),
            SemanticField(field_name="specifications")
        ],
        keywords_fields=[
            SemanticField(field_name="category"),
            SemanticField(field_name="brand")
        ]
    )
)

faq_semantic = SemanticConfiguration(
    name="faq-semantic",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="question"),
        content_fields=[
            SemanticField(field_name="answer")
        ],
        keywords_fields=[
            SemanticField(field_name="topic")
        ]
    )
)

article_semantic = SemanticConfiguration(
    name="articles-semantic",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="title"),
        content_fields=[
            SemanticField(field_name="summary"),
            SemanticField(field_name="body")
        ],
        keywords_fields=[
            SemanticField(field_name="tags"),
            SemanticField(field_name="author")
        ]
    )
)

# Apply all configurations to index
index.semantic_search = SemanticSearch(
    configurations=[product_semantic, faq_semantic, article_semantic]
)
```

### Intelligent Configuration Router

```python
class MultiConfigurationRouter:
    """Route queries to optimal semantic configuration."""
    
    def __init__(self, search_client):
        self.search_client = search_client
        self.config_map = {
            'product': 'products-semantic',
            'faq': 'faq-semantic',
            'article': 'articles-semantic',
            'general': 'products-semantic'  # Default
        }
    
    def search(self, query: str, content_type: str = None, top: int = 10):
        """
        Search with appropriate semantic configuration.
        
        Args:
            query: Search query
            content_type: Type of content to search (product/faq/article)
            top: Number of results
            
        Returns:
            Search results using optimal configuration
        """
        # Auto-detect content type if not specified
        if not content_type:
            content_type = self._detect_content_type(query)
        
        semantic_config = self.config_map.get(content_type, 'general')
        
        # Add filter for content type
        filter_expr = f"contentType eq '{content_type}'"
        
        results = self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=semantic_config,
            filter=filter_expr,
            query_caption=QueryCaptionType.EXTRACTIVE,
            top=top
        )
        
        return list(results)
    
    def _detect_content_type(self, query):
        """Auto-detect content type from query."""
        query_lower = query.lower()
        
        # Question patterns → FAQ
        if query.endswith('?') or any(query_lower.startswith(q) for q in ['what', 'how', 'why', 'when']):
            return 'faq'
        
        # Product-related terms
        product_terms = ['buy', 'price', 'product', 'best', 'review', 'compare']
        if any(term in query_lower for term in product_terms):
            return 'product'
        
        # Default to article for informational queries
        return 'article'

# Usage
router = MultiConfigurationRouter(search_client)

# Auto-detect content type
results = router.search("What is the return policy?")
# → Routed to faq-semantic configuration

# Explicit content type
results = router.search("best gaming laptop", content_type='product')
# → Routed to products-semantic configuration
```

### Benefits of Multi-Configuration Approach

1. **Optimized relevance** - Each configuration tuned for domain
2. **Better answer extraction** - FAQ config optimized for answers
3. **Improved performance** - Smaller search space per config
4. **Clear separation** - Easier to test and optimize each domain

---

## 2. Advanced Performance Optimization

### Pattern: Adaptive Query Strategy

```python
class AdaptiveSemanticSearch:
    """
    Adaptively choose between semantic, hybrid, and keyword
    based on query characteristics and performance constraints.
    """
    
    def __init__(self, search_client, semantic_config):
        self.search_client = search_client
        self.semantic_config = semantic_config
        self.latency_budget_ms = 200
        self.query_history = {}
    
    def search(self, query: str, user_id: str = None, top: int = 10):
        """
        Search with adaptive strategy selection.
        
        Args:
            query: Search query
            user_id: User identifier for personalization
            top: Number of results
            
        Returns:
            Results with metadata about strategy used
        """
        # Analyze query
        query_profile = self._profile_query(query)
        
        # Choose strategy based on query profile
        strategy = self._select_strategy(query_profile, user_id)
        
        # Execute with chosen strategy
        start = time.time()
        results = self._execute_strategy(strategy, query, top)
        latency_ms = (time.time() - start) * 1000
        
        # Learn from this query
        self._update_history(query, strategy, latency_ms)
        
        return {
            'results': results,
            'strategy': strategy,
            'latency_ms': latency_ms,
            'query_profile': query_profile
        }
    
    def _profile_query(self, query):
        """Profile query characteristics."""
        return {
            'length': len(query.split()),
            'is_question': query.endswith('?'),
            'is_natural_language': len(query.split()) >= 4,
            'has_special_chars': bool(re.search(r'[-_/]', query)),
            'estimated_complexity': 'high' if len(query.split()) > 6 else 'medium' if len(query.split()) > 3 else 'low'
        }
    
    def _select_strategy(self, query_profile, user_id):
        """
        Select optimal strategy based on query profile.
        
        Strategy decision tree:
        1. Exact match (special chars) → keyword with filter
        2. Natural language question → semantic + answers
        3. Short keyword (1-2 words) → keyword only
        4. Medium complexity → hybrid (BM25 + semantic)
        5. High complexity → full semantic
        """
        if query_profile['has_special_chars']:
            return 'keyword_filtered'
        
        if query_profile['is_question'] and query_profile['is_natural_language']:
            return 'semantic_with_answers'
        
        if query_profile['length'] <= 2:
            return 'keyword_only'
        
        if query_profile['estimated_complexity'] == 'high':
            return 'semantic_full'
        
        return 'hybrid'
    
    def _execute_strategy(self, strategy, query, top):
        """Execute search with selected strategy."""
        
        if strategy == 'keyword_only':
            return list(self.search_client.search(
                search_text=query,
                top=top
            ))
        
        elif strategy == 'semantic_full':
            return list(self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name=self.semantic_config,
                query_caption=QueryCaptionType.EXTRACTIVE,
                top=top
            ))
        
        elif strategy == 'semantic_with_answers':
            return list(self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name=self.semantic_config,
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                query_answer_count=2,
                top=top
            ))
        
        elif strategy == 'hybrid':
            # Hybrid: BM25 + semantic without captions for speed
            return list(self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name=self.semantic_config,
                top=top
            ))
        
        elif strategy == 'keyword_filtered':
            # For exact matches, use filter instead of semantic
            return list(self.search_client.search(
                search_text=query,
                search_mode=SearchMode.All,  # Require all terms
                top=top
            ))
        
        return []
    
    def _update_history(self, query, strategy, latency_ms):
        """Track query history for learning."""
        if query not in self.query_history:
            self.query_history[query] = []
        
        self.query_history[query].append({
            'strategy': strategy,
            'latency_ms': latency_ms,
            'timestamp': time.time()
        })
        
        # Keep only last 10 executions per query
        self.query_history[query] = self.query_history[query][-10:]

# Usage
adaptive_search = AdaptiveSemanticSearch(
    search_client,
    semantic_config="products"
)

# Natural language question → semantic_with_answers
result = adaptive_search.search("What's the best laptop for video editing?")
print(f"Strategy: {result['strategy']}")  # semantic_with_answers
print(f"Latency: {result['latency_ms']:.0f}ms")

# Short keyword → keyword_only
result = adaptive_search.search("laptop")
print(f"Strategy: {result['strategy']}")  # keyword_only

# Exact match → keyword_filtered
result = adaptive_search.search("SKU-ABC-123")
print(f"Strategy: {result['strategy']}")  # keyword_filtered
```

### Pattern: Progressive Enhancement

```python
class ProgressiveSemanticSearch:
    """
    Progressive enhancement: Start with fast keyword search,
    enhance with semantic in background if time permits.
    """
    
    def __init__(self, search_client, semantic_config):
        self.search_client = search_client
        self.semantic_config = semantic_config
    
    def search_progressive(self, query: str, top: int = 10, max_wait_ms: int = 150):
        """
        Progressive search: Fast keyword first, semantic enhancement if time permits.
        
        Args:
            query: Search query
            top: Number of results
            max_wait_ms: Maximum time to wait for semantic enhancement
            
        Returns:
            Results with metadata about what was computed
        """
        import concurrent.futures
        
        # Start keyword search immediately
        start = time.time()
        keyword_results = list(self.search_client.search(
            search_text=query,
            top=50  # Get more candidates for semantic re-ranking
        ))
        keyword_latency_ms = (time.time() - start) * 1000
        
        # Calculate remaining time budget
        remaining_budget_ms = max_wait_ms - keyword_latency_ms
        
        if remaining_budget_ms < 20:
            # Not enough time for semantic, return keyword results
            return {
                'results': keyword_results[:top],
                'strategy': 'keyword_only',
                'latency_ms': keyword_latency_ms,
                'semantic_applied': False
            }
        
        # Try semantic enhancement with remaining time budget
        with concurrent.futures.ThreadPoolExecutor() as executor:
            semantic_future = executor.submit(
                self._semantic_search,
                query, top
            )
            
            try:
                semantic_results = semantic_future.result(
                    timeout=remaining_budget_ms / 1000.0
                )
                total_latency_ms = (time.time() - start) * 1000
                
                return {
                    'results': semantic_results,
                    'strategy': 'semantic_enhanced',
                    'latency_ms': total_latency_ms,
                    'semantic_applied': True,
                    'keyword_latency_ms': keyword_latency_ms
                }
            
            except concurrent.futures.TimeoutError:
                # Semantic timed out, return keyword results
                semantic_future.cancel()
                return {
                    'results': keyword_results[:top],
                    'strategy': 'keyword_fallback',
                    'latency_ms': keyword_latency_ms,
                    'semantic_applied': False,
                    'reason': 'semantic_timeout'
                }
    
    def _semantic_search(self, query, top):
        """Execute semantic search."""
        return list(self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=self.semantic_config,
            query_caption=QueryCaptionType.EXTRACTIVE,
            top=top
        ))

# Usage
progressive_search = ProgressiveSemanticSearch(search_client, "products")

result = progressive_search.search_progressive(
    query="best laptop for video editing",
    top=10,
    max_wait_ms=150  # 150ms time budget
)

print(f"Strategy: {result['strategy']}")
print(f"Semantic applied: {result['semantic_applied']}")
print(f"Total latency: {result['latency_ms']:.0f}ms")
```

---

## 3. Cost Optimization Strategies

### Pattern: Query Budget Management

```python
class SemanticBudgetManager:
    """
    Manage semantic query budget to control costs.
    
    Azure AI Search Standard tier: ~$500/month
    Includes baseline semantic queries, additional usage metered.
    """
    
    def __init__(self, search_client, semantic_config, monthly_budget_usd=500):
        self.search_client = search_client
        self.semantic_config = semantic_config
        self.monthly_budget_usd = monthly_budget_usd
        self.semantic_queries_this_month = 0
        self.total_queries_this_month = 0
    
    def search(self, query: str, user_tier: str = 'free', top: int = 10):
        """
        Search with budget-aware semantic routing.
        
        Args:
            query: Search query
            user_tier: User tier (free/premium/enterprise)
            top: Number of results
            
        Returns:
            Search results with strategy based on budget
        """
        self.total_queries_this_month += 1
        
        # Budget check
        semantic_percentage = (
            self.semantic_queries_this_month / self.total_queries_this_month * 100
            if self.total_queries_this_month > 0 else 0
        )
        
        # Decision: Use semantic or not?
        use_semantic = self._should_use_semantic(
            query, user_tier, semantic_percentage
        )
        
        if use_semantic:
            self.semantic_queries_this_month += 1
            return self._semantic_search(query, top)
        else:
            return self._keyword_search(query, top)
    
    def _should_use_semantic(self, query, user_tier, current_semantic_pct):
        """
        Decide whether to use semantic based on budget and user tier.
        
        Budget allocation:
        - Enterprise users: 80% of semantic budget
        - Premium users: 15% of semantic budget
        - Free users: 5% of semantic budget
        
        Target: 30-40% of queries use semantic overall
        """
        target_semantic_pct = 35.0  # Target overall
        
        # Enterprise users always get semantic
        if user_tier == 'enterprise':
            return True
        
        # If under budget, allocate based on user tier
        if current_semantic_pct < target_semantic_pct:
            if user_tier == 'premium':
                # Premium users get semantic for NL queries
                return len(query.split()) >= 4
            elif user_tier == 'free':
                # Free users get semantic for questions only
                return query.endswith('?')
        
        # Over budget - only enterprise users
        return user_tier == 'enterprise'
    
    def _semantic_search(self, query, top):
        """Execute semantic search."""
        results = list(self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=self.semantic_config,
            query_caption=QueryCaptionType.EXTRACTIVE,
            top=top
        ))
        return {'results': results, 'strategy': 'semantic', 'cost_tier': 'premium'}
    
    def _keyword_search(self, query, top):
        """Execute keyword search."""
        results = list(self.search_client.search(
            search_text=query,
            top=top
        ))
        return {'results': results, 'strategy': 'keyword', 'cost_tier': 'free'}
    
    def get_budget_status(self):
        """Get current budget status."""
        semantic_pct = (
            self.semantic_queries_this_month / self.total_queries_this_month * 100
            if self.total_queries_this_month > 0 else 0
        )
        
        return {
            'total_queries': self.total_queries_this_month,
            'semantic_queries': self.semantic_queries_this_month,
            'semantic_percentage': semantic_pct,
            'target_percentage': 35.0,
            'status': 'under_budget' if semantic_pct < 35.0 else 'at_budget' if semantic_pct < 40.0 else 'over_budget'
        }

# Usage
budget_manager = SemanticBudgetManager(
    search_client,
    semantic_config="products",
    monthly_budget_usd=500
)

# Enterprise user - always gets semantic
result = budget_manager.search(
    query="What's the best laptop?",
    user_tier='enterprise'
)
print(f"Strategy: {result['strategy']}")  # semantic

# Free user with short query - gets keyword
result = budget_manager.search(
    query="laptop",
    user_tier='free'
)
print(f"Strategy: {result['strategy']}")  # keyword

# Check budget status
status = budget_manager.get_budget_status()
print(f"Semantic usage: {status['semantic_percentage']:.1f}% (target: 35%)")
print(f"Status: {status['status']}")
```

---

## 4. Integration with Azure AI Services

### Pattern: Semantic Search + Azure OpenAI

```python
from openai import AzureOpenAI

class SemanticSearchWithOpenAI:
    """
    Combine semantic search with Azure OpenAI for enhanced experiences.
    
    Use case: RAG (Retrieval-Augmented Generation)
    - Semantic search finds relevant context
    - Azure OpenAI generates natural language response
    """
    
    def __init__(self, search_client, semantic_config, openai_client):
        self.search_client = search_client
        self.semantic_config = semantic_config
        self.openai_client = openai_client
    
    def answer_question(self, question: str, top_k: int = 5):
        """
        Answer question using RAG pattern.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve as context
            
        Returns:
            Generated answer with citations
        """
        # Step 1: Retrieve relevant context with semantic search
        search_results = list(self.search_client.search(
            search_text=question,
            query_type="semantic",
            semantic_configuration_name=self.semantic_config,
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            top=top_k
        ))
        
        # Step 2: Check if semantic answer is high-confidence
        semantic_answer = self._extract_semantic_answer(search_results)
        if semantic_answer and semantic_answer['score'] > 0.85:
            # High-confidence semantic answer - return directly
            return {
                'answer': semantic_answer['text'],
                'strategy': 'semantic_answer',
                'confidence': semantic_answer['score'],
                'citations': [semantic_answer['source']]
            }
        
        # Step 3: Build context from search results
        context = self._build_context(search_results)
        
        # Step 4: Generate answer with Azure OpenAI
        generated_answer = self._generate_answer_with_openai(question, context)
        
        # Step 5: Extract citations
        citations = [
            {'id': r['id'], 'title': r.get('title', 'Unknown')}
            for r in search_results[:3]
        ]
        
        return {
            'answer': generated_answer,
            'strategy': 'rag_with_openai',
            'confidence': 0.75,  # Medium confidence for generated answers
            'citations': citations
        }
    
    def _extract_semantic_answer(self, results):
        """Extract highest-confidence semantic answer."""
        for result in results:
            if '@search.answers' in result and result['@search.answers']:
                best_answer = max(
                    result['@search.answers'],
                    key=lambda a: a.get('score', 0)
                )
                if best_answer.get('score', 0) > 0.85:
                    return {
                        'text': best_answer.get('text', ''),
                        'score': best_answer.get('score', 0),
                        'source': result.get('id', 'unknown')
                    }
        return None
    
    def _build_context(self, results):
        """Build context from search results."""
        context_parts = []
        
        for i, result in enumerate(results[:5]):
            # Use semantic captions if available
            if '@search.captions' in result and result['@search.captions']:
                caption_text = result['@search.captions'][0].get('text', '')
                context_parts.append(f"[{i+1}] {caption_text}")
            else:
                # Fall back to summary or content
                content = result.get('summary', '') or result.get('content', '')
                context_parts.append(f"[{i+1}] {content[:300]}")
        
        return '\n\n'.join(context_parts)
    
    def _generate_answer_with_openai(self, question, context):
        """Generate answer using Azure OpenAI."""
        system_prompt = """You are a helpful assistant that answers questions based on provided context.
Use the context below to answer the user's question. If the context doesn't contain enough information,
say so clearly. Include reference numbers [1], [2], etc. when citing context."""
        
        user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content

# Usage
openai_client = AzureOpenAI(
    api_key="your-api-key",
    api_version="2024-02-15-preview",
    azure_endpoint="https://your-resource.openai.azure.com/"
)

rag_search = SemanticSearchWithOpenAI(
    search_client,
    semantic_config="faq",
    openai_client=openai_client
)

# Ask a question
result = rag_search.answer_question(
    question="What is the return policy for electronics?"
)

print(f"Answer: {result['answer']}")
print(f"Strategy: {result['strategy']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Citations: {[c['title'] for c in result['citations']]}")
```

### Pattern: Semantic Search + Content Safety

```python
from azure.ai.contentsafety import ContentSafetyClient

class SafeSemanticSearch:
    """
    Semantic search with Azure Content Safety integration.
    
    Filters unsafe content and queries.
    """
    
    def __init__(self, search_client, semantic_config, content_safety_client):
        self.search_client = search_client
        self.semantic_config = semantic_config
        self.content_safety_client = content_safety_client
    
    def safe_search(self, query: str, top: int = 10):
        """
        Search with content safety filtering.
        
        Args:
            query: Search query
            top: Number of results
            
        Returns:
            Safe search results with unsafe content filtered
        """
        # Step 1: Check query safety
        query_safety = self._check_query_safety(query)
        
        if not query_safety['is_safe']:
            return {
                'results': [],
                'blocked': True,
                'reason': 'Unsafe query detected',
                'categories': query_safety['categories']
            }
        
        # Step 2: Execute search
        results = list(self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=self.semantic_config,
            query_caption=QueryCaptionType.EXTRACTIVE,
            top=top * 2  # Get more to account for filtering
        ))
        
        # Step 3: Filter unsafe results
        safe_results = []
        for result in results:
            content = result.get('content', '')
            if self._is_content_safe(content):
                safe_results.append(result)
            
            if len(safe_results) >= top:
                break
        
        return {
            'results': safe_results,
            'blocked': False,
            'filtered_count': len(results) - len(safe_results)
        }
    
    def _check_query_safety(self, query):
        """Check if query is safe."""
        # Analyze query with Content Safety API
        analysis = self.content_safety_client.analyze_text(
            text=query,
            categories=["Hate", "SelfHarm", "Sexual", "Violence"]
        )
        
        # Check if any category exceeds threshold
        unsafe_categories = [
            cat for cat in analysis.categories
            if cat.severity > 2  # Threshold: 0-2 safe, 3-6 unsafe
        ]
        
        return {
            'is_safe': len(unsafe_categories) == 0,
            'categories': [cat.category for cat in unsafe_categories]
        }
    
    def _is_content_safe(self, content):
        """Check if content is safe to display."""
        if not content or len(content) < 10:
            return True
        
        analysis = self.content_safety_client.analyze_text(
            text=content[:1000],  # Analyze first 1000 chars
            categories=["Hate", "SelfHarm", "Sexual", "Violence"]
        )
        
        # All categories must be below threshold
        return all(cat.severity <= 2 for cat in analysis.categories)

# Usage (pseudo-code - requires actual Content Safety client setup)
# safe_search = SafeSemanticSearch(
#     search_client,
#     semantic_config="content",
#     content_safety_client=content_safety_client
# )
# 
# result = safe_search.safe_search("appropriate query", top=10)
# print(f"Results: {len(result['results'])}")
# print(f"Filtered: {result.get('filtered_count', 0)}")
```

---

## 5. Testing and Validation Framework

### Pattern: Comprehensive Semantic Search Testing

```python
class SemanticSearchTestSuite:
    """
    Comprehensive testing framework for semantic search implementations.
    """
    
    def __init__(self, search_client, semantic_config):
        self.search_client = search_client
        self.semantic_config = semantic_config
    
    def run_full_test_suite(self, test_queries_with_judgments):
        """
        Run complete test suite.
        
        Args:
            test_queries_with_judgments: Dict of query -> {doc_id: relevance_score}
            
        Returns:
            Comprehensive test report
        """
        results = {
            'relevance_metrics': self._test_relevance(test_queries_with_judgments),
            'latency_metrics': self._test_latency(test_queries_with_judgments),
            'answer_quality': self._test_answers(test_queries_with_judgments),
            'caption_quality': self._test_captions(test_queries_with_judgments),
            'regression_checks': self._test_regressions(test_queries_with_judgments)
        }
        
        return results
    
    def _test_relevance(self, test_queries):
        """Test relevance with NDCG and other metrics."""
        ndcg_scores = []
        
        for query, judgments in test_queries.items():
            results = list(self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name=self.semantic_config,
                top=10
            ))
            
            ndcg = self._calculate_ndcg(results, judgments)
            ndcg_scores.append(ndcg)
        
        return {
            'avg_ndcg': sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0,
            'min_ndcg': min(ndcg_scores) if ndcg_scores else 0,
            'max_ndcg': max(ndcg_scores) if ndcg_scores else 0
        }
    
    def _test_latency(self, test_queries):
        """Test latency percentiles."""
        latencies = []
        
        for query in test_queries.keys():
            start = time.time()
            list(self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name=self.semantic_config,
                top=10
            ))
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
        
        latencies.sort()
        return {
            'p50': latencies[len(latencies)//2],
            'p95': latencies[int(len(latencies)*0.95)],
            'p99': latencies[int(len(latencies)*0.99)],
            'avg': sum(latencies) / len(latencies)
        }
    
    def _test_answers(self, test_queries):
        """Test answer extraction quality."""
        question_queries = {
            q: j for q, j in test_queries.items()
            if q.endswith('?')
        }
        
        answers_extracted = 0
        high_confidence_answers = 0
        
        for query in question_queries.keys():
            results = list(self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name=self.semantic_config,
                query_answer=QueryAnswerType.EXTRACTIVE,
                top=10
            ))
            
            for result in results:
                if '@search.answers' in result and result['@search.answers']:
                    answers_extracted += 1
                    best_score = max(a.get('score', 0) for a in result['@search.answers'])
                    if best_score > 0.75:
                        high_confidence_answers += 1
                    break
        
        total_questions = len(question_queries)
        return {
            'answer_rate': answers_extracted / total_questions if total_questions > 0 else 0,
            'high_confidence_rate': high_confidence_answers / total_questions if total_questions > 0 else 0
        }
    
    def _test_captions(self, test_queries):
        """Test caption quality."""
        caption_lengths = []
        
        for query in test_queries.keys():
            results = list(self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name=self.semantic_config,
                query_caption=QueryCaptionType.EXTRACTIVE,
                top=10
            ))
            
            for result in results:
                if '@search.captions' in result:
                    for caption in result['@search.captions']:
                        text = caption.get('text', '')
                        caption_lengths.append(len(text.split()))
        
        return {
            'avg_caption_length': sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0,
            'caption_rate': len(caption_lengths) / (len(test_queries) * 10)  # Per result
        }
    
    def _test_regressions(self, test_queries):
        """Test for regressions vs keyword search."""
        improvements = 0
        regressions = 0
        
        for query, judgments in test_queries.items():
            # Keyword NDCG
            keyword_results = list(self.search_client.search(
                search_text=query,
                top=10
            ))
            keyword_ndcg = self._calculate_ndcg(keyword_results, judgments)
            
            # Semantic NDCG
            semantic_results = list(self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name=self.semantic_config,
                top=10
            ))
            semantic_ndcg = self._calculate_ndcg(semantic_results, judgments)
            
            if semantic_ndcg > keyword_ndcg + 0.05:  # 5% improvement threshold
                improvements += 1
            elif semantic_ndcg < keyword_ndcg - 0.05:  # 5% regression threshold
                regressions += 1
        
        return {
            'improvements': improvements,
            'regressions': regressions,
            'neutral': len(test_queries) - improvements - regressions
        }
    
    def _calculate_ndcg(self, results, judgments):
        """Calculate NDCG@10."""
        import math
        dcg = sum((2**judgments.get(r.get('id', ''), 0) - 1) / math.log2(i + 2) 
                  for i, r in enumerate(results[:10]))
        ideal_scores = sorted(judgments.values(), reverse=True)[:10]
        idcg = sum((2**score - 1) / math.log2(i + 2) for i, score in enumerate(ideal_scores))
        return dcg / idcg if idcg > 0 else 0

# Usage
test_suite = SemanticSearchTestSuite(search_client, "products")

# Define test queries with relevance judgments
test_queries = {
    "best laptop for video editing": {
        "doc1": 4,  # Highly relevant
        "doc2": 3,  # Relevant
        "doc3": 1   # Marginally relevant
    },
    # ... more test queries
}

report = test_suite.run_full_test_suite(test_queries)

print("Test Results:")
print(f"  NDCG@10: {report['relevance_metrics']['avg_ndcg']:.3f}")
print(f"  Latency p95: {report['latency_metrics']['p95']:.0f}ms")
print(f"  Answer rate: {report['answer_quality']['answer_rate']:.1%}")
print(f"  Improvements: {report['regression_checks']['improvements']}")
print(f"  Regressions: {report['regression_checks']['regressions']}")
```

---

## Summary

These advanced patterns enable production-grade semantic search implementations:

1. **Multi-Configuration Strategies** - Domain-specific configs with intelligent routing
2. **Advanced Performance Optimization** - Adaptive strategies and progressive enhancement
3. **Cost Optimization** - Budget-aware routing and user-tier allocation
4. **Azure AI Integration** - RAG with OpenAI, Content Safety filtering
5. **Comprehensive Testing** - Full test suite for validation

**Key Takeaways:**
- Use multiple semantic configurations for different content domains
- Implement adaptive strategies based on query characteristics
- Manage costs with budget-aware routing and user tiers
- Integrate with Azure AI services for enhanced experiences
- Test comprehensively before production deployment

**Document Statistics:**
- **Words**: ~4,000
- **Patterns**: 5 advanced implementation patterns
- **Code examples**: 10+ production-ready implementations
- **Integration examples**: Azure OpenAI, Content Safety
