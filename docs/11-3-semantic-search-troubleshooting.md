# Semantic Search - Part 3: Troubleshooting Guide

## Common Issues, Root Causes, and Solutions

This comprehensive troubleshooting guide addresses the 5 most common issues encountered when implementing and operating semantic search in production environments.

---

## Issue 1: Low Semantic Scores / Poor Re-ranking

### Symptoms

- Semantic `@search.rerankerScore` values consistently low (<2.0)
- Semantic ranking doesn't change result order significantly vs BM25
- User feedback indicates irrelevant results despite semantic ranking
- NDCG improvement minimal or negative vs keyword-only search

**Example:**
```python
# Query: "best laptop for video editing"
results = search_client.search(
    search_text=query,
    query_type="semantic",
    semantic_configuration_name="products",
    top=10
)

# Symptom: All reranker scores < 2.0
for result in results:
    print(f"{result['title']}: {result['@search.rerankerScore']}")
    # Output:
    # Dell Laptop ABC: 1.2
    # HP Notebook XYZ: 1.1
    # Lenovo ThinkPad: 0.9
    # ... (all scores very low)
```

### Root Causes

1. **Semantic field configuration mismatch**
   - Title/content fields don't contain relevant text
   - Important fields excluded from semantic config
   - Fields too long (>10,000 words) causing truncation

2. **Poor initial retrieval (BM25)**
   - BM25 not returning relevant candidates in top 50
   - Semantic ranking can only re-rank what BM25 retrieves
   - "Garbage in, garbage out" problem

3. **Content quality issues**
   - Documents lack descriptive text
   - Heavy on technical specs, light on natural language
   - Missing context that semantic models need

4. **Query-document vocabulary mismatch**
   - Documents use different terminology than queries
   - Even semantic ranking struggles with extreme mismatches

### Diagnosis Tools

```python
class SemanticScoreDiagnostics:
    """Diagnose low semantic score issues."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def diagnose_low_scores(self, query: str, semantic_config: str):
        """
        Diagnose why semantic scores are low.
        
        Returns diagnostic report with actionable insights.
        """
        # Run both BM25 and semantic searches
        bm25_results = list(self.search_client.search(
            search_text=query,
            top=50
        ))
        
        semantic_results = list(self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=semantic_config,
            top=50
        ))
        
        # Analyze results
        diagnosis = {
            'query': query,
            'semantic_config': semantic_config,
            'issues': []
        }
        
        # Check 1: Are semantic scores universally low?
        avg_reranker_score = sum(
            r.get('@search.rerankerScore', 0) for r in semantic_results
        ) / len(semantic_results)
        
        if avg_reranker_score < 1.5:
            diagnosis['issues'].append({
                'issue': 'Low semantic scores across all results',
                'avg_score': avg_reranker_score,
                'recommendation': 'Check semantic field configuration and content quality'
            })
        
        # Check 2: Is BM25 returning relevant candidates?
        top10_bm25 = bm25_results[:10]
        if not self._has_relevant_results(top10_bm25, query):
            diagnosis['issues'].append({
                'issue': 'Poor BM25 initial retrieval',
                'recommendation': 'Improve BM25 ranking before adding semantic (tune analyzers, synonyms, scoring profiles)'
            })
        
        # Check 3: Are semantic fields populated?
        field_stats = self._analyze_field_population(semantic_results, semantic_config)
        if field_stats['empty_field_rate'] > 20:
            diagnosis['issues'].append({
                'issue': f"Semantic fields sparsely populated ({field_stats['empty_field_rate']}% empty)",
                'recommendation': 'Ensure title, content, and keyword fields contain text'
            })
        
        # Check 4: Content length
        if field_stats['avg_content_length'] < 50:
            diagnosis['issues'].append({
                'issue': f"Very short content (avg {field_stats['avg_content_length']} words)",
                'recommendation': 'Semantic ranking works best with 100+ words of natural language content'
            })
        
        return diagnosis
    
    def _has_relevant_results(self, results, query):
        """Heuristic to check if BM25 results seem relevant."""
        # Simple check: Do top results contain query words?
        query_words = set(query.lower().split())
        
        relevant_count = 0
        for result in results[:10]:
            title = result.get('title', '').lower()
            if len(query_words.intersection(set(title.split()))) >= 2:
                relevant_count += 1
        
        return relevant_count >= 5  # At least 5 of top 10 match
    
    def _analyze_field_population(self, results, semantic_config):
        """Analyze semantic field population and content length."""
        # Note: In production, read semantic_config to get field names
        # For this example, assume common fields
        
        empty_count = 0
        total_length = 0
        
        for result in results:
            title = result.get('title', '')
            content = result.get('content', '') or result.get('description', '')
            
            if not title or not content:
                empty_count += 1
            
            total_length += len(title.split()) + len(content.split())
        
        return {
            'empty_field_rate': (empty_count / len(results) * 100) if results else 0,
            'avg_content_length': (total_length / len(results)) if results else 0
        }

# Usage
diagnostics = SemanticScoreDiagnostics(search_client)

diagnosis = diagnostics.diagnose_low_scores(
    query="best laptop for video editing",
    semantic_config="products-semantic"
)

print(f"Diagnosis for: {diagnosis['query']}")
print(f"Found {len(diagnosis['issues'])} issues:\n")

for issue in diagnosis['issues']:
    print(f"❌ {issue['issue']}")
    print(f"   Recommendation: {issue['recommendation']}\n")
```

### Solutions

**Solution 1: Optimize semantic field configuration**

```python
# Bad configuration (common mistake)
SemanticConfiguration(
    name="products-bad",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="sku"),  # ❌ SKU, not descriptive
        content_fields=[
            SemanticField(field_name="specifications")  # ❌ Technical specs only
        ],
        keywords_fields=[]
    )
)

# Good configuration
SemanticConfiguration(
    name="products-good",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="productName"),  # ✅ Descriptive name
        content_fields=[
            SemanticField(field_name="marketing_description"),  # ✅ Natural language
            SemanticField(field_name="customer_reviews_summary"),  # ✅ User language
            SemanticField(field_name="specifications")  # ✅ Technical details last
        ],
        keywords_fields=[
            SemanticField(field_name="category"),
            SemanticField(field_name="use_cases")
        ]
    )
)
```

**Solution 2: Improve BM25 before adding semantic**

```python
# Step 1: Tune BM25 parameters
scoring_profile = ScoringProfile(
    name="bm25-tuned",
    text_weights=TextWeights(
        weights={
            "productName": 3.0,      # Boost title matches
            "description": 1.5,       # Moderate boost for description
            "specifications": 0.8     # Slight downrank for specs
        }
    )
)

# Step 2: Add custom analyzer for better tokenization
custom_analyzer = CustomAnalyzer(
    name="product_analyzer",
    tokenizer="standard",
    token_filters=[
        "lowercase",
        "stop_words",
        "synonym_filter"  # Add product synonyms
    ]
)

# Step 3: Validate BM25 improvements before re-enabling semantic
bm25_results = search_client.search(
    search_text=query,
    scoring_profile="bm25-tuned",
    top=50
)

# Only add semantic once BM25 NDCG@10 > 0.6
```

**Solution 3: Enrich content at index time**

```python
class ContentEnricher:
    """Enrich documents with semantic-friendly content."""
    
    def enrich_document(self, doc):
        """Add natural language descriptions for semantic ranking."""
        
        # Generate marketing description from specs
        if not doc.get('description') and doc.get('specs'):
            doc['description'] = self._generate_description_from_specs(
                doc['specs']
            )
        
        # Combine user reviews into summary
        if doc.get('reviews'):
            doc['review_summary'] = self._summarize_reviews(
                doc['reviews']
            )
        
        # Extract use cases from specs
        if doc.get('specs'):
            doc['use_cases'] = self._extract_use_cases(
                doc['specs']
            )
        
        return doc
    
    def _generate_description_from_specs(self, specs):
        """Convert specs to natural language."""
        # Example: Transform "RAM: 16GB, CPU: Intel i7" 
        # → "This laptop features 16GB of RAM and an Intel Core i7 processor"
        
        description_parts = []
        
        if 'RAM' in specs:
            description_parts.append(f"features {specs['RAM']} of RAM")
        
        if 'CPU' in specs:
            description_parts.append(f"powered by {specs['CPU']} processor")
        
        if 'Storage' in specs:
            description_parts.append(f"includes {specs['Storage']} storage")
        
        return f"This product {', '.join(description_parts)}."
    
    def _summarize_reviews(self, reviews):
        """Summarize user reviews."""
        # Extract common themes
        positive_terms = ['great', 'excellent', 'love', 'perfect', 'amazing']
        negative_terms = ['poor', 'bad', 'slow', 'broken', 'disappointed']
        
        positive_count = sum(
            any(term in review.lower() for term in positive_terms)
            for review in reviews
        )
        
        return f"Users appreciate this product ({positive_count}/{len(reviews)} positive reviews)."

# Usage: Enrich documents before indexing
enricher = ContentEnricher()

for doc in documents_to_index:
    enriched_doc = enricher.enrich_document(doc)
    index_batch.append(enriched_doc)

search_client.upload_documents(documents=index_batch)
```

### Prevention

1. **Test semantic config with sample queries** - Validate before production
2. **Monitor semantic score distribution** - Alert if avg score drops
3. **Ensure BM25 baseline quality** - NDCG@10 > 0.6 before semantic
4. **Validate content quality** - 100+ words of natural language per document

---

## Issue 2: Semantic Answers Not Extracting or Low Quality

### Symptoms

- `@search.answers` array empty or missing for question queries
- Answer extraction rate <20% (expected: 30-50%)
- Extracted answers incorrect or irrelevant
- Answer confidence scores consistently low (<0.6)

**Example:**
```python
# Query: "What is the return policy?"
results = search_client.search(
    search_text=query,
    query_type="semantic",
    semantic_configuration_name="faq",
    query_answer=QueryAnswerType.EXTRACTIVE,
    query_answer_count=3,
    top=20
)

# Symptom: No answers extracted
for result in results:
    answers = result.get('@search.answers', [])
    if not answers:
        print(f"❌ No answers found for: {query}")
    else:
        for answer in answers:
            if answer['score'] < 0.6:
                print(f"⚠️ Low confidence answer: {answer['score']:.2f}")
```

### Root Causes

1. **Query not question-like**
   - Answer extraction requires question format
   - Keyword queries don't trigger answer extraction

2. **Documents lack factual content**
   - Answer extraction needs direct, factual statements
   - Descriptive or marketing content less suitable

3. **Content structure issues**
   - Answers buried in long paragraphs
   - No clear question-answer structure
   - FAQs not formatted properly

4. **Semantic configuration problems**
   - Wrong fields included in content_fields
   - Answer-containing fields excluded

### Diagnosis Tools

```python
class AnswerExtractionDiagnostics:
    """Diagnose answer extraction issues."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def diagnose_answer_extraction(
        self,
        query: str,
        semantic_config: str,
        expected_answer_field: str = "content"
    ):
        """Diagnose why answers aren't being extracted."""
        
        results = list(self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=semantic_config,
            query_answer=QueryAnswerType.EXTRACTIVE,
            query_answer_count=3,
            top=20
        ))
        
        diagnosis = {
            'query': query,
            'issues': []
        }
        
        # Check 1: Is query question-like?
        if not self._is_question(query):
            diagnosis['issues'].append({
                'issue': 'Query is not question-format',
                'recommendation': 'Answer extraction works best for questions (What/How/When/Where/Why)'
            })
        
        # Check 2: Are any answers extracted?
        answer_count = sum(
            1 for r in results if '@search.answers' in r and r['@search.answers']
        )
        
        if answer_count == 0:
            diagnosis['issues'].append({
                'issue': 'Zero answers extracted from 20 documents',
                'recommendation': 'Check document content structure and semantic configuration'
            })
            
            # Sub-check: Is answer field populated?
            empty_content = sum(
                1 for r in results if not r.get(expected_answer_field)
            )
            if empty_content > 10:
                diagnosis['issues'].append({
                    'issue': f'{empty_content}/20 documents have empty {expected_answer_field} field',
                    'recommendation': f'Ensure {expected_answer_field} field contains factual content'
                })
        
        # Check 3: Answer confidence scores
        answer_scores = []
        for r in results:
            if '@search.answers' in r:
                for ans in r['@search.answers']:
                    answer_scores.append(ans.get('score', 0))
        
        if answer_scores:
            avg_score = sum(answer_scores) / len(answer_scores)
            if avg_score < 0.6:
                diagnosis['issues'].append({
                    'issue': f'Low average answer confidence: {avg_score:.2f}',
                    'recommendation': 'Content may not contain direct, factual answers'
                })
        
        # Check 4: Content analysis
        content_analysis = self._analyze_content_for_answers(results, expected_answer_field)
        if content_analysis['has_qa_structure'] < 0.3:
            diagnosis['issues'].append({
                'issue': 'Documents lack clear question-answer structure',
                'recommendation': 'Format content as explicit Q&A pairs or direct statements'
            })
        
        return diagnosis
    
    def _is_question(self, query):
        """Check if query is question-format."""
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'should', 'is', 'are', 'does']
        query_lower = query.lower()
        return (
            any(query_lower.startswith(q) for q in question_words) or
            query.endswith('?')
        )
    
    def _analyze_content_for_answers(self, results, field_name):
        """Analyze if content is suitable for answer extraction."""
        qa_indicators = [':', 'is', 'are', 'means', 'refers to', 'answer', 'policy']
        
        docs_with_qa_structure = 0
        
        for result in results:
            content = result.get(field_name, '')
            if any(indicator in content.lower() for indicator in qa_indicators):
                docs_with_qa_structure += 1
        
        return {
            'has_qa_structure': docs_with_qa_structure / len(results) if results else 0
        }

# Usage
answer_diagnostics = AnswerExtractionDiagnostics(search_client)

diagnosis = answer_diagnostics.diagnose_answer_extraction(
    query="What is the return policy?",
    semantic_config="faq-semantic",
    expected_answer_field="answer_text"
)

print(f"Answer Extraction Diagnosis: {diagnosis['query']}")
for issue in diagnosis['issues']:
    print(f"  ❌ {issue['issue']}")
    print(f"     → {issue['recommendation']}\n")
```

### Solutions

**Solution 1: Structure content for answer extraction**

```python
# Bad: Unstructured content
bad_document = {
    "id": "faq-1",
    "content": "Our company has a great return policy. We want customers to be satisfied. "
               "If you're not happy, you can return items. Contact customer service for details."
}

# Good: Structured Q&A format
good_document = {
    "id": "faq-1",
    "question": "What is the return policy?",
    "answer": "You can return items within 30 days of purchase for a full refund. "
              "The item must be in original condition with tags attached. "
              "To initiate a return, contact customer service at 1-800-123-4567 or visit our returns portal.",
    "category": "Returns and Refunds"
}

# Index with semantic configuration targeting answer field
semantic_config = SemanticConfiguration(
    name="faq-answers",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="question"),
        content_fields=[
            SemanticField(field_name="answer")  # Direct answers here
        ],
        keywords_fields=[
            SemanticField(field_name="category")
        ]
    )
)
```

**Solution 2: Increase answer confidence threshold**

```python
# Bad: Low threshold allows poor answers
results = search_client.search(
    search_text=query,
    query_type="semantic",
    semantic_configuration_name="faq",
    query_answer=QueryAnswerType.EXTRACTIVE,
    query_answer_count=5,  # ❌ Too many low-quality answers
    top=20
)

# Good: High threshold for quality
results = search_client.search(
    search_text=query,
    query_type="semantic",
    semantic_configuration_name="faq",
    query_answer=QueryAnswerType.EXTRACTIVE,
    query_answer_count=2,  # ✅ Fewer, higher-quality answers
    top=50  # ✅ Search more documents to find good answers
)

# Filter answers by confidence in application code
filtered_answers = []
for result in results:
    if '@search.answers' in result:
        for answer in result['@search.answers']:
            if answer.get('score', 0) >= 0.75:  # ✅ High confidence only
                filtered_answers.append(answer)

# Only show if we have at least one high-confidence answer
if filtered_answers:
    display_answer_box(filtered_answers[0])
else:
    display_regular_results(results)
```

**Solution 3: Pre-process FAQs at index time**

```python
class FAQPreprocessor:
    """Preprocess FAQs for better answer extraction."""
    
    def process_faq(self, raw_faq):
        """Transform FAQ into answer-extraction friendly format."""
        
        # Extract question and answer from combined text
        if 'question' not in raw_faq:
            question, answer = self._split_qa(raw_faq['content'])
        else:
            question = raw_faq['question']
            answer = raw_faq['answer']
        
        # Ensure answer is direct and factual
        answer = self._make_answer_direct(answer)
        
        # Add question variations for better retrieval
        question_variations = self._generate_question_variations(question)
        
        return {
            'id': raw_faq['id'],
            'question': question,
            'question_variations': question_variations,
            'answer': answer,
            'answer_summary': answer[:200],  # First 200 chars
            'category': raw_faq.get('category', 'General')
        }
    
    def _split_qa(self, content):
        """Split combined content into Q and A."""
        # Look for patterns like "Q:" or "A:"
        if 'Q:' in content and 'A:' in content:
            parts = content.split('A:', 1)
            question = parts[0].replace('Q:', '').strip()
            answer = parts[1].strip()
            return question, answer
        
        # Fallback: Assume first sentence is question
        sentences = content.split('.')
        return sentences[0], '.'.join(sentences[1:])
    
    def _make_answer_direct(self, answer):
        """Ensure answer is direct and factual."""
        # Remove marketing fluff
        fluff_phrases = [
            "We're happy to help",
            "Thank you for asking",
            "Great question"
        ]
        
        for phrase in fluff_phrases:
            answer = answer.replace(phrase, '')
        
        return answer.strip()
    
    def _generate_question_variations(self, question):
        """Generate question variations for better matching."""
        variations = [question]
        
        # Add common rephrasing
        if question.startswith("What is"):
            variations.append(question.replace("What is", "What's"))
            variations.append(question.replace("What is", "Tell me about"))
        
        if question.startswith("How do I"):
            variations.append(question.replace("How do I", "How can I"))
            variations.append(question.replace("How do I", "How to"))
        
        return variations

# Usage
preprocessor = FAQPreprocessor()

faqs_to_index = []
for raw_faq in raw_faq_data:
    processed = preprocessor.process_faq(raw_faq)
    faqs_to_index.append(processed)

search_client.upload_documents(documents=faqs_to_index)
```

### Prevention

1. **Format content as explicit Q&A** - Use question and answer fields
2. **Write direct, factual answers** - Avoid marketing language
3. **Use high confidence threshold** (0.7+) - Fewer, better answers
4. **Test with real user questions** - Validate answer extraction rate
5. **Monitor answer quality** - Collect user feedback on answers

---

## Issue 3: High Latency / Timeout Errors

### Symptoms

- Semantic search p95 latency >300ms (target: <200ms)
- Timeout errors (HTTP 408 or client-side timeouts)
- User complaints about slow search
- Latency spikes during peak hours

**Example:**
```python
import time

start = time.time()
try:
    results = search_client.search(
        search_text=query,
        query_type="semantic",
        semantic_configuration_name="products",
        top=10,
        timeout=5  # 5 second timeout
    )
    latency_ms = (time.time() - start) * 1000
    
    if latency_ms > 300:
        print(f"⚠️ High latency: {latency_ms:.0f}ms")
        
except Exception as e:
    if "timeout" in str(e).lower():
        print(f"❌ Search timed out after 5 seconds")
```

### Root Causes

1. **Large result set (top parameter too high)**
   - Semantic re-ranks top 50 documents
   - Requesting top=100 doesn't improve quality but adds latency

2. **Heavy semantic configuration**
   - Too many content fields
   - Very long content fields (>5,000 words)
   - Complex semantic processing

3. **Insufficient Search Service capacity**
   - Underpowered tier (Basic/Standard S1)
   - High query volume relative to capacity
   - Resource contention

4. **Network latency**
   - Client far from Search Service region
   - No CDN or regional failover

5. **Inefficient initial retrieval (L0)**
   - Slow BM25 query (complex filters, large index)
   - Semantic latency compounds slow base search

### Diagnosis Tools

```python
class LatencyDiagnostics:
    """Diagnose semantic search latency issues."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def diagnose_latency(self, query: str, semantic_config: str):
        """
        Measure latency breakdown and identify bottlenecks.
        
        Returns:
            Latency breakdown and recommendations
        """
        diagnosis = {
            'query': query,
            'latency_breakdown': {},
            'issues': []
        }
        
        # Measure BM25-only latency (baseline)
        start = time.time()
        bm25_results = list(self.search_client.search(
            search_text=query,
            top=50
        ))
        bm25_latency_ms = (time.time() - start) * 1000
        diagnosis['latency_breakdown']['bm25_only'] = bm25_latency_ms
        
        # Measure semantic latency
        start = time.time()
        semantic_results = list(self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=semantic_config,
            query_caption=QueryCaptionType.EXTRACTIVE,
            top=10
        ))
        semantic_latency_ms = (time.time() - start) * 1000
        diagnosis['latency_breakdown']['semantic_total'] = semantic_latency_ms
        
        # Calculate semantic overhead
        semantic_overhead = semantic_latency_ms - bm25_latency_ms
        diagnosis['latency_breakdown']['semantic_overhead'] = semantic_overhead
        
        # Identify issues
        if bm25_latency_ms > 100:
            diagnosis['issues'].append({
                'issue': f'Slow BM25 baseline: {bm25_latency_ms:.0f}ms',
                'recommendation': 'Optimize BM25 query (indexes, filters, scoring profiles) before addressing semantic'
            })
        
        if semantic_overhead > 150:
            diagnosis['issues'].append({
                'issue': f'High semantic overhead: {semantic_overhead:.0f}ms',
                'recommendation': 'Simplify semantic configuration (fewer content fields, shorter text)'
            })
        
        if semantic_latency_ms > 300:
            diagnosis['issues'].append({
                'issue': f'Total latency exceeds 300ms: {semantic_latency_ms:.0f}ms',
                'recommendation': 'Consider caching, CDN, or falling back to keyword for latency-sensitive queries'
            })
        
        # Check top parameter
        if len(semantic_results) > 20:
            diagnosis['issues'].append({
                'issue': f'Requesting {len(semantic_results)} results',
                'recommendation': 'Reduce top parameter to 10-20 for semantic queries'
            })
        
        return diagnosis

# Usage
latency_diagnostics = LatencyDiagnostics(search_client)

diagnosis = latency_diagnostics.diagnose_latency(
    query="best laptop for video editing",
    semantic_config="products-semantic"
)

print(f"Latency Diagnosis: {diagnosis['query']}")
print(f"\nLatency Breakdown:")
print(f"  BM25 baseline: {diagnosis['latency_breakdown']['bm25_only']:.0f}ms")
print(f"  Semantic overhead: {diagnosis['latency_breakdown']['semantic_overhead']:.0f}ms")
print(f"  Total: {diagnosis['latency_breakdown']['semantic_total']:.0f}ms")
print(f"\nIssues:")
for issue in diagnosis['issues']:
    print(f"  ❌ {issue['issue']}")
    print(f"     → {issue['recommendation']}\n")
```

### Solutions

**Solution 1: Optimize top parameter**

```python
# Bad: Requesting too many results
results = search_client.search(
    search_text=query,
    query_type="semantic",
    semantic_configuration_name="products",
    top=100  # ❌ Semantic only re-ranks top 50, rest is wasted
)

# Good: Optimize for semantic
results = search_client.search(
    search_text=query,
    query_type="semantic",
    semantic_configuration_name="products",
    top=10  # ✅ Request only what you need
)

# If pagination needed, fetch more results with keyword search
if need_more_results:
    additional_results = search_client.search(
        search_text=query,
        skip=10,
        top=20
    )
```

**Solution 2: Implement timeout and fallback**

```python
class SemanticSearchWithFallback:
    """Semantic search with automatic fallback on timeout."""
    
    def __init__(self, search_client, semantic_config, timeout_ms=250):
        self.search_client = search_client
        self.semantic_config = semantic_config
        self.timeout_ms = timeout_ms
    
    def search(self, query: str, top: int = 10):
        """
        Try semantic search, fall back to keyword on timeout.
        
        Args:
            query: Search query
            top: Number of results
            
        Returns:
            Results with metadata about which strategy was used
        """
        import concurrent.futures
        
        # Try semantic with timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                self._semantic_search,
                query, top
            )
            
            try:
                results = future.result(timeout=self.timeout_ms / 1000.0)
                return {
                    'results': results,
                    'strategy': 'semantic',
                    'fallback': False
                }
            except concurrent.futures.TimeoutError:
                future.cancel()
                # Fall back to fast keyword search
                return self._fallback_keyword_search(query, top)
    
    def _semantic_search(self, query, top):
        """Execute semantic search."""
        return list(self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=self.semantic_config,
            query_caption=QueryCaptionType.EXTRACTIVE,
            top=top
        ))
    
    def _fallback_keyword_search(self, query, top):
        """Fallback to keyword search."""
        results = list(self.search_client.search(
            search_text=query,
            top=top
        ))
        return {
            'results': results,
            'strategy': 'keyword',
            'fallback': True,
            'reason': 'semantic_timeout'
        }

# Usage
searcher = SemanticSearchWithFallback(
    search_client,
    semantic_config="products",
    timeout_ms=200  # 200ms timeout
)

result = searcher.search("laptop for video editing", top=10)

if result['fallback']:
    print(f"⚠️ Fell back to {result['strategy']} (reason: {result['reason']})")
    # Log for monitoring
    log_fallback_event(query, result['reason'])
```

**Solution 3: Aggressive caching**

```python
from functools import lru_cache
import hashlib

class CachedSemanticSearch:
    """Semantic search with aggressive caching."""
    
    def __init__(self, search_client, semantic_config, cache_size=1000):
        self.search_client = search_client
        self.semantic_config = semantic_config
        self.cache_size = cache_size
        self._cache = {}
    
    def search(self, query: str, top: int = 10, use_cache: bool = True):
        """
        Search with caching.
        
        Args:
            query: Search query
            top: Number of results
            use_cache: Whether to use cache
            
        Returns:
            Search results (from cache or fresh)
        """
        if not use_cache:
            return self._execute_search(query, top)
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, top)
        
        # Check cache
        if cache_key in self._cache:
            return {
                'results': self._cache[cache_key],
                'cached': True,
                'latency_ms': 5  # Cache hit is fast
            }
        
        # Execute search
        start = time.time()
        results = self._execute_search(query, top)
        latency_ms = (time.time() - start) * 1000
        
        # Cache if successful and fast enough
        if latency_ms < 300:
            self._add_to_cache(cache_key, results)
        
        return {
            'results': results,
            'cached': False,
            'latency_ms': latency_ms
        }
    
    def _generate_cache_key(self, query, top):
        """Generate cache key from query and parameters."""
        key_str = f"{query}:{top}:{self.semantic_config}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _add_to_cache(self, key, results):
        """Add to cache with size limit."""
        if len(self._cache) >= self.cache_size:
            # Evict oldest entry (simplified LRU)
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = results
    
    def _execute_search(self, query, top):
        """Execute actual search."""
        return list(self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=self.semantic_config,
            query_caption=QueryCaptionType.EXTRACTIVE,
            top=top
        ))

# Usage
cached_searcher = CachedSemanticSearch(
    search_client,
    semantic_config="products",
    cache_size=500  # Cache 500 most common queries
)

result = cached_searcher.search("best laptop", top=10)
print(f"Cached: {result['cached']}")
print(f"Latency: {result['latency_ms']:.0f}ms")

# Second call for same query - cache hit
result2 = cached_searcher.search("best laptop", top=10)
print(f"Cached: {result2['cached']}")  # True
print(f"Latency: {result2['latency_ms']:.0f}ms")  # ~5ms
```

### Prevention

1. **Set performance SLOs** - Target p95 <200ms, p99 <300ms
2. **Monitor latency continuously** - Alert on regressions
3. **Implement caching** - Cache 20-30% of repeated queries
4. **Use appropriate tier** - Standard S2+ for production semantic workloads
5. **Optimize BM25 first** - Fast baseline = fast semantic

---

## Issue 4: Inconsistent Results / Result Order Instability

### Symptoms

- Same query returns different results on repeated searches
- Result order changes between searches (within seconds)
- `@search.rerankerScore` values fluctuate for same query
- A/B testing shows high variance

### Root Causes

1. **Index updates during search**
   - Documents being added/updated/deleted
   - Changes affect initial retrieval (L0 BM25)

2. **Tie-breaking with identical scores**
   - Multiple documents with same reranker score
   - Tie-breaking is non-deterministic

3. **Concurrent index writes**
   - Real-time indexing causing instability
   - BM25 scores change as document frequencies update

4. **Distributed query execution**
   - Multiple replicas may have slight inconsistencies
   - Query routing to different replicas

### Solutions

**Solution 1: Add stable sorting tie-breaker**

```python
# Bad: No tie-breaking, unstable results
results = search_client.search(
    search_text=query,
    query_type="semantic",
    semantic_configuration_name="products",
    top=10
    # ❌ Documents with same score may shuffle
)

# Good: Stable tie-breaking by document ID
results = search_client.search(
    search_text=query,
    query_type="semantic",
    semantic_configuration_name="products",
    order_by=["@search.rerankerScore desc", "id asc"],  # ✅ Stable tie-breaker
    top=10
)
```

**Solution 2: Use sessionId for consistency**

```python
class StableSemanticSearch:
    """Ensure consistent results within a session."""
    
    def __init__(self, search_client):
        self.search_client = search_client
        self.session_cache = {}
    
    def search(self, query: str, session_id: str, top: int = 10):
        """
        Search with session-based consistency.
        
        Same query in same session returns same results.
        """
        cache_key = f"{session_id}:{query}:{top}"
        
        # Return cached results for this session
        if cache_key in self.session_cache:
            return self.session_cache[cache_key]
        
        # Execute search with stable sorting
        results = list(self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="products",
            order_by=["@search.rerankerScore desc", "id asc"],
            top=top
        ))
        
        # Cache for session
        self.session_cache[cache_key] = results
        
        return results
    
    def clear_session(self, session_id: str):
        """Clear session cache."""
        keys_to_delete = [k for k in self.session_cache if k.startswith(f"{session_id}:")]
        for key in keys_to_delete:
            del self.session_cache[key]

# Usage
stable_searcher = StableSemanticSearch(search_client)

# User session
session_id = user.session_id

# First search
results1 = stable_searcher.search("laptop", session_id, top=10)

# Repeated search in same session - returns same results
results2 = stable_searcher.search("laptop", session_id, top=10)

assert results1 == results2  # ✅ Consistent within session
```

### Prevention

1. **Use stable tie-breaking** - Always include deterministic sort field
2. **Cache results per session** - Ensure consistency within user session
3. **Batch index updates** - Avoid frequent small updates
4. **Monitor result stability** - Track result position variance

---

## Issue 5: Semantic Ranking Not Improving Relevance (Negative Impact)

### Symptoms

- NDCG@10 worse with semantic than keyword-only
- User satisfaction lower with semantic enabled
- Specific query types regress with semantic
- CTR decreases after enabling semantic

### Root Causes

1. **Wrong queries routed to semantic**
   - Exact match queries don't benefit from semantic
   - Technical queries degraded by semantic understanding

2. **Semantic optimizing for wrong field**
   - Configuration emphasizes wrong content
   - User intent misaligned with semantic fields

3. **BM25 baseline already excellent**
   - Semantic can't improve already-optimal ranking
   - Added latency without benefit

### Diagnosis & Solutions

```python
class SemanticRegressionAnalyzer:
    """Identify where semantic hurts vs helps."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def analyze_query_segments(self, test_queries_with_judgments):
        """
        Analyze which query types benefit from semantic.
        
        Args:
            test_queries_with_judgments: Dict of query -> relevance judgments
            
        Returns:
            Segment analysis showing where semantic helps/hurts
        """
        segments = {
            'question_queries': [],
            'keyword_queries': [],
            'exact_match_queries': [],
            'technical_queries': []
        }
        
        # Classify queries
        for query, judgments in test_queries_with_judgments.items():
            segment = self._classify_query(query)
            segments[segment].append((query, judgments))
        
        # Compare BM25 vs Semantic for each segment
        results = {}
        for segment, queries in segments.items():
            if not queries:
                continue
            
            bm25_ndcgs = []
            semantic_ndcgs = []
            
            for query, judgments in queries:
                # BM25
                bm25_results = list(self.search_client.search(
                    search_text=query,
                    top=10
                ))
                bm25_ndcg = self._calculate_ndcg(bm25_results, judgments)
                bm25_ndcgs.append(bm25_ndcg)
                
                # Semantic
                semantic_results = list(self.search_client.search(
                    search_text=query,
                    query_type="semantic",
                    semantic_configuration_name="default",
                    top=10
                ))
                semantic_ndcg = self._calculate_ndcg(semantic_results, judgments)
                semantic_ndcgs.append(semantic_ndcg)
            
            avg_bm25 = sum(bm25_ndcgs) / len(bm25_ndcgs)
            avg_semantic = sum(semantic_ndcgs) / len(semantic_ndcgs)
            
            results[segment] = {
                'query_count': len(queries),
                'bm25_ndcg': avg_bm25,
                'semantic_ndcg': avg_semantic,
                'improvement': ((avg_semantic - avg_bm25) / avg_bm25 * 100) if avg_bm25 > 0 else 0
            }
        
        return results
    
    def _classify_query(self, query):
        """Classify query into segment."""
        if query.endswith('?') or any(query.lower().startswith(q) for q in ['what', 'how', 'why']):
            return 'question_queries'
        elif len(query.split()) <= 2:
            return 'keyword_queries'
        elif query.isupper() or '-' in query:
            return 'exact_match_queries'
        else:
            return 'technical_queries'
    
    def _calculate_ndcg(self, results, judgments):
        """Calculate NDCG@10."""
        import math
        dcg = sum((2**judgments.get(r['id'], 0) - 1) / math.log2(i + 2) 
                  for i, r in enumerate(results[:10]))
        ideal_scores = sorted(judgments.values(), reverse=True)[:10]
        idcg = sum((2**score - 1) / math.log2(i + 2) for i, score in enumerate(ideal_scores))
        return dcg / idcg if idcg > 0 else 0
    
    def print_segment_report(self, results):
        """Print segment analysis report."""
        print("Semantic Impact by Query Segment")
        print("=" * 70)
        print(f"{'Segment':<25} {'Queries':<10} {'BM25':<10} {'Semantic':<10} {'Change':<10}")
        print("-" * 70)
        
        for segment, metrics in results.items():
            change_str = f"{metrics['improvement']:+.1f}%"
            change_indicator = "✅" if metrics['improvement'] > 5 else "❌" if metrics['improvement'] < -5 else "➖"
            
            print(f"{segment:<25} {metrics['query_count']:<10} "
                  f"{metrics['bm25_ndcg']:<10.3f} {metrics['semantic_ndcg']:<10.3f} "
                  f"{change_str:<10} {change_indicator}")

# Usage
analyzer = SemanticRegressionAnalyzer(search_client)

# Test with labeled queries
test_queries = {
    "What is machine learning?": {"doc1": 3, "doc2": 2},
    "laptop": {"doc3": 3, "doc4": 2},
    "SKU-ABC-123": {"doc5": 4},
    # ... more queries
}

segment_results = analyzer.analyze_query_segments(test_queries)
analyzer.print_segment_report(segment_results)

# Output:
# Segment                   Queries    BM25       Semantic   Change     
# ----------------------------------------------------------------------
# question_queries          45         0.654      0.782      +19.6%  ✅
# keyword_queries           120        0.721      0.698      -3.2%   ❌
# exact_match_queries       30         0.891      0.712      -20.1%  ❌
# technical_queries         55         0.645      0.701      +8.7%   ✅

# Action: Don't use semantic for keyword and exact_match queries
```

### Prevention

1. **Segment testing during evaluation** - Test on different query types
2. **Route intelligently** - Use semantic only where it helps
3. **Monitor by segment** - Track metrics per query type
4. **Accept that semantic isn't universal** - Some queries work better with keyword

---

## Summary

These 5 troubleshooting scenarios cover the most common issues encountered in production semantic search implementations:

1. **Low Semantic Scores** - Field configuration and content quality issues
2. **Poor Answer Extraction** - Content structure and query format problems
3. **High Latency** - Performance optimization and timeout handling
4. **Inconsistent Results** - Stability and tie-breaking issues
5. **Negative Impact** - Query segment analysis and selective routing

Each issue includes:
- Clear symptoms to recognize the problem
- Root cause analysis
- Diagnostic tools with code
- Step-by-step solutions
- Prevention strategies

**Document Statistics:**
- **Words**: ~6,200
- **Issues covered**: 5 comprehensive scenarios
- **Diagnostic tools**: 5 production-ready diagnostic classes
- **Solutions**: 15+ actionable fixes with code examples
