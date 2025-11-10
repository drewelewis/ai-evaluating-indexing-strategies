# Dataset Preparation

Complete guide to creating high-quality test datasets and relevance judgments for evaluating search strategies.

## 📋 Table of Contents
- [Overview](#overview)
- [Golden Dataset Design](#golden-dataset-design)
- [Query Selection](#query-selection)
- [Relevance Judgments](#relevance-judgments)
- [Dataset Generation](#dataset-generation)
- [Quality Assurance](#quality-assurance)
- [Dataset Management](#dataset-management)
- [Best Practices](#best-practices)

---

## Overview

### Real-World Scenario: LegalSearch Pro's Golden Dataset Creation

**The Challenge**

LegalSearch Pro provides case law search for 12,000 attorneys across 450 law firms. Their search index contains 2.8 million legal documents (court opinions, statutes, regulations, briefs) with 850K queries per month (avg 28 QPS, peak 85 QPS during case research deadlines).

**The Problem: No Ground Truth for Evaluation**

After two years in production, the team needed to improve search relevance but faced a critical blocker:
- **No systematic evaluation**: Relied on anecdotal attorney feedback ("search is bad")
- **No metrics baseline**: Couldn't measure if changes improved or degraded relevance
- **No test queries**: No representative set of searches to evaluate against
- **No relevance judgments**: No ground truth for what documents should match queries
- **Blind optimization**: Changed scoring profiles, analyzers with no way to validate improvements

**Previous Failed Approach: Ad-Hoc Testing**

Their first attempt used **informal testing**:

1. ❌ Engineers manually tested 10-15 queries they created
2. ❌ Picked "obviously relevant" documents (confirmation bias)
3. ❌ No coverage of real attorney query patterns
4. ❌ No inter-annotator agreement (single engineer's opinion)
5. ❌ Deployed changes that "looked better" to engineers
6. **Result**: 23% increase in support tickets after deployment ("search got worse"), attorney satisfaction dropped from 3.2/5 to 2.8/5

**Golden Dataset Solution**

The team implemented a **systematic dataset preparation process** following industry best practices:

**Phase 1: Query Mining (Week 1)**
- Analyzed 90 days of query logs (2.55M queries)
- Extracted top 500 most frequent queries (head queries, 60% of traffic)
- Sampled 300 long-tail queries (2-5 occurrences, 35% of traffic)
- Added 50 synthetic edge-case queries (complex Boolean, date ranges)
- **Total**: 850 representative queries

**Query Distribution Analysis**:
```
Query Type         | Count | % of Dataset | % of Production
-------------------|-------|--------------|----------------
Head (>100 uses)   |   200 |         24%  |            60%
Torso (10-100)     |   300 |         35%  |            25%
Long-tail (2-9)    |   300 |         35%  |            15%
Edge cases         |    50 |          6%  |           <1%
Total              |   850 |        100%  |           100%
```

**Query Intent Coverage**:
- **Informational** (45%): "what is summary judgment", "elements of negligence"
- **Navigational** (30%): "Brown v. Board of Education", "California Civil Code 1542"
- **Transactional** (15%): "download complaint template", "cite recent case"
- **Analytical** (10%): "cases citing Smith v. Jones since 2020"

**Phase 2: Document Pooling (Week 2)**
- Ran all 850 queries against 5 different search configurations:
  - **Baseline** (current production): BM25 only
  - **Enhanced BM25**: Custom legal analyzer + field weights
  - **Hybrid**: BM25 + vector search (0.7/0.3 RRF)
  - **Semantic**: Pure vector search with legal embeddings
  - **Tuned**: Hybrid + scoring profile (recency, authority)
- Pooled top-20 results from each configuration → 85,000 candidate pairs
- Deduplicated: 34,200 unique query-document pairs
- **Strategic sampling**: Selected top-10 per query (8,500 pairs for judgment)

**Pooling Strategy Benefits**:
- Captures diverse results across configurations (not biased to current system)
- Ensures judgments cover documents future configurations might surface
- Reduces annotation cost (8,500 vs 85,000 pairs, 90% reduction)

**Phase 3: Relevance Judgment Collection (Weeks 3-6)**
- Recruited **12 annotators** (6 senior attorneys, 6 paralegals with 5+ years experience)
- Developed comprehensive **judgment guidelines** (18-page manual with 45 examples)
- Used **5-point scale** (0=not relevant, 4=perfectly relevant)
- **Triple annotation**: Each query-document pair judged by 3 annotators
- Annotation interface showed:
  - Query text + detected intent
  - Document title, excerpt (first 500 words), metadata (court, date, jurisdiction)
  - Contextual information (why this document was pooled)

**Judgment Guidelines (Simplified)**:

| Score | Label | Legal Search Criteria |
|-------|-------|----------------------|
| **4** | Perfect | Directly addresses query, correct jurisdiction, binding precedent, recent |
| **3** | Highly Relevant | On-point but older, or persuasive (not binding) precedent |
| **2** | Moderately Relevant | Related legal issue but different jurisdiction or tangential |
| **1** | Slightly Relevant | Mentions topic but not substantively useful |
| **0** | Not Relevant | Unrelated legal area or irrelevant document |

**Annotation Quality Control**:
- **Training phase**: All annotators judged same 50 pairs, discussed disagreements
- **Inter-annotator agreement**: Cohen's Kappa measured weekly
  - Week 1: κ = 0.48 (moderate agreement)
  - Week 3: κ = 0.67 (substantial agreement) after calibration
  - Week 6: κ = 0.74 (substantial agreement, maintained)
- **Disagreement resolution**: Pairs with ≥2-point disagreement reviewed by senior attorney
- **Throughput**: 15-20 judgments/hour per annotator (avg 18 min/judgment for legal review)

**Phase 4: Dataset Validation (Week 7)**
- **Completeness check**: 8,500 pairs × 3 annotators = 25,500 judgments collected ✓
- **Coverage analysis**: 
  - All 850 queries have ≥8 judgments (avg 10.0)
  - 94% of queries have at least one highly relevant doc (score ≥3)
  - 6% zero-result queries (expected for exploratory/edge cases)
- **Distribution validation**:
  ```
  Score | Count  | %    | Interpretation
  ------|--------|------|----------------------------------
  4     |  3,825 | 15%  | Perfect matches (good, not too common)
  3     |  6,375 | 25%  | Highly relevant (largest group ✓)
  2     |  5,100 | 20%  | Moderately relevant (balanced)
  1     |  4,250 | 17%  | Slightly relevant (balanced)
  0     |  5,950 | 23%  | Not relevant (healthy representation)
  ```
- **Quality issues found**: 127 pairs flagged for review (0.5%), re-annotated

**Phase 5: Baseline Evaluation (Week 8)**

Evaluated current production system against golden dataset:

**Metrics (Current Production System)**:
- **Precision@5**: 0.42 (42% of top-5 results highly relevant)
- **Precision@10**: 0.35 (declining precision, poor result ordering)
- **NDCG@10**: 0.51 (mediocre ranking quality)
- **MRR**: 0.38 (first relevant result at position ~2.6 on average)
- **Zero-result rate**: 8.4% (higher than expected 6%)

**Per-Intent Performance**:
```
Intent          | P@5  | NDCG@10 | Zero-Result
----------------|------|---------|-------------
Navigational    | 0.68 |    0.72 |        2.1%  ← Good (simple lookups)
Informational   | 0.38 |    0.48 |        7.8%  ← Poor (complex concepts)
Transactional   | 0.31 |    0.41 |       11.2%  ← Very poor (need specific docs)
Analytical      | 0.22 |    0.35 |       18.5%  ← Worst (complex multi-doc)
```

**Key Insights from Baseline**:
1. **Navigational queries work well** (68% P@5) → keep current approach
2. **Informational queries struggle** (38% P@5) → need better concept matching (semantic search)
3. **Transactional queries fail** (31% P@5) → need document type filtering
4. **Analytical queries broken** (22% P@5) → need citation graph analysis

**Business Impact (First 6 Months After Dataset Creation)**

**Development Velocity**:
- Time to evaluate search change: 6 days → 4 hours (96% reduction)
- A/B test duration: 4 weeks → 1 week (confident metrics)
- Failed production deployments: 3 incidents → 0 incidents (validation catches issues)

**Search Quality Improvements** (using dataset to guide optimization):
- **Phase 1** (Legal analyzer): P@5 +12% (0.42 → 0.47)
- **Phase 2** (Hybrid search): P@5 +19% (0.47 → 0.56)
- **Phase 3** (Scoring profile): P@5 +11% (0.56 → 0.62)
- **Overall**: P@5 +48% (0.42 → 0.62), NDCG +37% (0.51 → 0.70)

**Business Results**:
- Attorney satisfaction: 2.8/5 → 4.1/5 (+46%)
- Support tickets (search issues): 340/month → 85/month (-75%)
- Research time per case: 6.2 hours → 4.4 hours (-29%, $180/hour billed = $324 saved per case)
- Firm retention (annual contracts): 89% → 96% (+7 percentage points)
- Revenue impact: +$2.8M/year (reduced churn + upsells)
- **Dataset ROI**: $95K investment (annotation cost), 2,847% ROI, 12-day payback

**Key Learnings**

1. **Query mining beats synthetic**: Real query logs revealed 73% of queries were multi-term legal phrases engineers never would have created.

2. **Domain experts essential**: Attorneys caught subtle relevance issues (e.g., binding vs persuasive precedent) generic annotators would miss.

3. **Triple annotation worth it**: 14% of pairs had ≥2-point disagreement, required expert resolution. Single annotation would have 14% error rate.

4. **Pooling saves 90% cost**: 8,500 annotated pairs vs 85,000 candidates. Still captured 94% of relevant documents.

5. **Per-intent metrics critical**: Overall NDCG@10=0.51 masked that analytical queries were completely broken (0.35). Aggregate metrics hide problems.

6. **Dataset enables confident optimization**: Before dataset, "guessed" at improvements. After dataset, measured +48% P@5 improvement across 6 iterations.

7. **Regular refresh needed**: Legal corpus changes (new cases published daily). Refreshing 10% of dataset quarterly (85 queries, 850 pairs) keeps it current for $8K vs $95K full rebuild.

### Dataset Requirements

```python
class DatasetRequirements:
    """Understanding test dataset requirements."""
    
    @staticmethod
    def dataset_components():
        """
        Essential components of a search evaluation dataset:
        
        1. Document corpus (indexed content)
        2. Query set (representative search queries)
        3. Relevance judgments (query-document pairs with ratings)
        4. Metadata (query intent, difficulty, category)
        """
        return {
            'document_corpus': {
                'purpose': 'Content to be indexed and searched',
                'size': '1,000 - 100,000+ documents',
                'requirements': [
                    'Representative of production data',
                    'Diverse content types',
                    'Realistic document length distribution',
                    'Include edge cases'
                ]
            },
            'query_set': {
                'purpose': 'Test queries representing user intent',
                'size': '100 - 1,000+ queries',
                'requirements': [
                    'Cover common query patterns',
                    'Include long-tail queries',
                    'Represent different intents',
                    'Vary in difficulty'
                ]
            },
            'relevance_judgments': {
                'purpose': 'Ground truth for evaluation',
                'size': '1,000 - 10,000+ judgments',
                'requirements': [
                    'Multiple judgments per query',
                    'Consistent rating scale',
                    'Inter-annotator agreement',
                    'Cover top-k results'
                ]
            }
        }
    
    @staticmethod
    def quality_metrics():
        """
        Dataset quality metrics to track.
        """
        return {
            'coverage': {
                'query_diversity': 'Unique query types represented',
                'document_coverage': 'Percentage of corpus with judgments',
                'intent_coverage': 'User intent categories covered'
            },
            'consistency': {
                'inter_annotator_agreement': 'Agreement between judges (Kappa)',
                'rating_distribution': 'Balance of relevance levels',
                'judgment_density': 'Judgments per query'
            },
            'realism': {
                'query_frequency_alignment': 'Match production query distribution',
                'seasonal_relevance': 'Current/timely content',
                'domain_specificity': 'Industry-appropriate terminology'
            }
        }

# Usage
requirements = DatasetRequirements()

components = requirements.dataset_components()
print("Dataset Components:")
for component, details in components.items():
    print(f"\n{component.upper()}:")
    print(f"  Purpose: {details['purpose']}")
    print(f"  Size: {details['size']}")
    print(f"  Requirements:")
    for req in details['requirements']:
        print(f"    • {req}")

quality = requirements.quality_metrics()
print("\n\nQuality Metrics:")
for category, metrics in quality.items():
    print(f"\n{category.upper()}:")
    for metric, description in metrics.items():
        print(f"  {metric}: {description}")
```

---

## Golden Dataset Design

### Dataset Structure

```python
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class Document:
    """Represents a document in the corpus."""
    id: str
    title: str
    content: str
    category: str
    tags: List[str]
    metadata: Dict[str, any]
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Query:
    """Represents a test query."""
    id: str
    text: str
    intent: str  # navigational, informational, transactional
    difficulty: str  # easy, medium, hard
    category: str
    expected_result_count: int
    metadata: Dict[str, any]
    
    def to_dict(self):
        return asdict(self)

@dataclass
class RelevanceJudgment:
    """Represents a relevance judgment for a query-document pair."""
    query_id: str
    document_id: str
    relevance: int  # 0-4 scale (0=not relevant, 4=perfectly relevant)
    judge_id: str
    timestamp: str
    notes: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

class GoldenDataset:
    """Manages a golden dataset for search evaluation."""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.queries: List[Query] = []
        self.judgments: List[RelevanceJudgment] = []
    
    def add_document(self, document: Document):
        """Add document to corpus."""
        self.documents.append(document)
    
    def add_query(self, query: Query):
        """Add query to test set."""
        self.queries.append(query)
    
    def add_judgment(self, judgment: RelevanceJudgment):
        """Add relevance judgment."""
        self.judgments.append(judgment)
    
    def get_judgments_for_query(self, query_id: str) -> List[RelevanceJudgment]:
        """Get all judgments for a specific query."""
        return [j for j in self.judgments if j.query_id == query_id]
    
    def get_relevant_documents(self, query_id: str, min_relevance: int = 2) -> List[str]:
        """Get list of relevant document IDs for a query."""
        judgments = self.get_judgments_for_query(query_id)
        return [j.document_id for j in judgments if j.relevance >= min_relevance]
    
    def export_to_json(self, output_dir: str):
        """Export dataset to JSON files."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export documents
        with open(f"{output_dir}/documents.json", 'w') as f:
            json.dump([doc.to_dict() for doc in self.documents], f, indent=2)
        
        # Export queries
        with open(f"{output_dir}/queries.json", 'w') as f:
            json.dump([q.to_dict() for q in self.queries], f, indent=2)
        
        # Export judgments
        with open(f"{output_dir}/judgments.json", 'w') as f:
            json.dump([j.to_dict() for j in self.judgments], f, indent=2)
        
        print(f"Exported dataset to {output_dir}")
        print(f"  Documents: {len(self.documents)}")
        print(f"  Queries: {len(self.queries)}")
        print(f"  Judgments: {len(self.judgments)}")
    
    def load_from_json(self, input_dir: str):
        """Load dataset from JSON files."""
        # Load documents
        with open(f"{input_dir}/documents.json", 'r') as f:
            doc_data = json.load(f)
            self.documents = [Document(**d) for d in doc_data]
        
        # Load queries
        with open(f"{input_dir}/queries.json", 'r') as f:
            query_data = json.load(f)
            self.queries = [Query(**q) for q in query_data]
        
        # Load judgments
        with open(f"{input_dir}/judgments.json", 'r') as f:
            judgment_data = json.load(f)
            self.judgments = [RelevanceJudgment(**j) for j in judgment_data]
        
        print(f"Loaded dataset from {input_dir}")
        print(f"  Documents: {len(self.documents)}")
        print(f"  Queries: {len(self.queries)}")
        print(f"  Judgments: {len(self.judgments)}")

# Usage
dataset = GoldenDataset()

# Add sample document
doc1 = Document(
    id="doc1",
    title="Introduction to Machine Learning",
    content="Machine learning is a subset of artificial intelligence...",
    category="Education",
    tags=["AI", "ML", "Tutorial"],
    metadata={"author": "John Doe", "date": "2023-01-15"}
)
dataset.add_document(doc1)

# Add sample query
query1 = Query(
    id="q1",
    text="machine learning basics",
    intent="informational",
    difficulty="easy",
    category="Education",
    expected_result_count=10,
    metadata={"source": "user_logs"}
)
dataset.add_query(query1)

# Add relevance judgment
judgment1 = RelevanceJudgment(
    query_id="q1",
    document_id="doc1",
    relevance=4,  # Highly relevant
    judge_id="judge1",
    timestamp=datetime.now().isoformat()
)
dataset.add_judgment(judgment1)

# Export dataset
# dataset.export_to_json("datasets/golden_set")
```

---

## Query Selection

### Query Mining from Logs

```python
import re
from collections import Counter
from typing import List, Tuple

class QueryMiner:
    """Mine queries from search logs."""
    
    def __init__(self):
        self.query_logs = []
    
    def load_query_logs(self, log_file: str):
        """Load query logs from file."""
        with open(log_file, 'r') as f:
            for line in f:
                # Parse log line (assuming JSON format)
                try:
                    log_entry = json.loads(line)
                    self.query_logs.append(log_entry)
                except:
                    pass
        
        print(f"Loaded {len(self.query_logs)} query log entries")
    
    def extract_top_queries(self, top_n: int = 100) -> List[Tuple[str, int]]:
        """Extract most frequent queries."""
        queries = [log['query'] for log in self.query_logs if 'query' in log]
        query_counts = Counter(queries)
        
        return query_counts.most_common(top_n)
    
    def extract_long_tail_queries(self, min_frequency: int = 2, max_frequency: int = 5) -> List[str]:
        """
        Extract long-tail queries (low frequency).
        
        These are important for comprehensive evaluation.
        """
        queries = [log['query'] for log in self.query_logs if 'query' in log]
        query_counts = Counter(queries)
        
        long_tail = [
            query for query, count in query_counts.items()
            if min_frequency <= count <= max_frequency
        ]
        
        return long_tail
    
    def categorize_queries(self, queries: List[str]) -> Dict[str, List[str]]:
        """
        Categorize queries by intent and characteristics.
        """
        categorized = {
            'short': [],      # 1-2 words
            'medium': [],     # 3-4 words
            'long': [],       # 5+ words
            'question': [],   # Contains question words
            'phrase': [],     # Quoted phrases
            'boolean': []     # Contains AND/OR/NOT
        }
        
        for query in queries:
            word_count = len(query.split())
            
            # Length categorization
            if word_count <= 2:
                categorized['short'].append(query)
            elif word_count <= 4:
                categorized['medium'].append(query)
            else:
                categorized['long'].append(query)
            
            # Intent categorization
            question_words = r'\b(what|who|where|when|why|how|is|are|can|does)\b'
            if re.search(question_words, query.lower()):
                categorized['question'].append(query)
            
            if '"' in query:
                categorized['phrase'].append(query)
            
            boolean_ops = r'\b(AND|OR|NOT)\b'
            if re.search(boolean_ops, query, re.IGNORECASE):
                categorized['boolean'].append(query)
        
        return categorized
    
    def sample_diverse_queries(self, query_list: List[str], sample_size: int = 100) -> List[str]:
        """
        Sample diverse queries using stratified sampling.
        """
        import random
        
        categorized = self.categorize_queries(query_list)
        
        # Calculate samples per category (proportional)
        total = len(query_list)
        sampled = []
        
        for category, queries in categorized.items():
            if not queries:
                continue
            
            proportion = len(queries) / total
            category_sample_size = max(1, int(sample_size * proportion))
            
            if len(queries) >= category_sample_size:
                sampled.extend(random.sample(queries, category_sample_size))
            else:
                sampled.extend(queries)
        
        # Remove duplicates and limit to sample_size
        sampled = list(set(sampled))[:sample_size]
        
        return sampled

# Usage
miner = QueryMiner()

# Simulate query logs
sample_logs = [
    {"query": "machine learning", "timestamp": "2023-01-01"},
    {"query": "python tutorial", "timestamp": "2023-01-02"},
    {"query": "machine learning", "timestamp": "2023-01-03"},
    {"query": "what is deep learning", "timestamp": "2023-01-04"},
    {"query": "azure search", "timestamp": "2023-01-05"}
]

# Save sample logs
with open("query_logs.jsonl", 'w') as f:
    for log in sample_logs:
        f.write(json.dumps(log) + '\n')

# Load and analyze
miner.load_query_logs("query_logs.jsonl")

top_queries = miner.extract_top_queries(top_n=10)
print("\nTop Queries:")
for query, count in top_queries[:5]:
    print(f"  {query}: {count}")

# Categorize queries
all_queries = [log['query'] for log in sample_logs]
categorized = miner.categorize_queries(all_queries)
print("\nQuery Categories:")
for category, queries in categorized.items():
    if queries:
        print(f"  {category}: {len(queries)}")
```

### Synthetic Query Generation

```python
class SyntheticQueryGenerator:
    """Generate synthetic queries for testing."""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
    
    def generate_keyword_queries(self, num_queries: int = 50) -> List[str]:
        """
        Generate queries from document keywords.
        
        Extract important terms and create queries.
        """
        import random
        from collections import Counter
        
        # Extract all words from documents
        all_words = []
        for doc in self.documents:
            words = doc.title.lower().split() + doc.content.lower().split()
            # Filter short words and common words
            words = [w for w in words if len(w) > 3]
            all_words.extend(words)
        
        # Get most common words
        word_counts = Counter(all_words)
        top_words = [word for word, _ in word_counts.most_common(200)]
        
        queries = []
        for _ in range(num_queries):
            # Generate 1-3 word queries
            query_length = random.randint(1, 3)
            query = ' '.join(random.sample(top_words, query_length))
            queries.append(query)
        
        return queries
    
    def generate_phrase_queries(self, num_queries: int = 20) -> List[str]:
        """
        Generate phrase queries from document titles.
        """
        import random
        
        queries = []
        for _ in range(num_queries):
            doc = random.choice(self.documents)
            # Use part of title as query
            words = doc.title.split()
            if len(words) >= 2:
                start = random.randint(0, len(words) - 2)
                length = random.randint(2, min(4, len(words) - start))
                phrase = ' '.join(words[start:start+length])
                queries.append(phrase.lower())
        
        return queries
    
    def generate_question_queries(self, num_queries: int = 20) -> List[str]:
        """
        Generate question-style queries.
        """
        import random
        
        question_templates = [
            "what is {}",
            "how to {}",
            "why {}",
            "when to use {}",
            "where to find {}",
            "which {} is best"
        ]
        
        # Extract key phrases from documents
        key_phrases = []
        for doc in self.documents:
            words = doc.title.lower().split()
            if len(words) >= 2:
                key_phrases.append(' '.join(words[:3]))
        
        queries = []
        for _ in range(num_queries):
            template = random.choice(question_templates)
            phrase = random.choice(key_phrases)
            query = template.format(phrase)
            queries.append(query)
        
        return queries

# Usage
# generator = SyntheticQueryGenerator(dataset.documents)
# 
# keyword_queries = generator.generate_keyword_queries(num_queries=30)
# print("Synthetic Keyword Queries:")
# for query in keyword_queries[:5]:
#     print(f"  {query}")
# 
# phrase_queries = generator.generate_phrase_queries(num_queries=10)
# print("\nSynthetic Phrase Queries:")
# for query in phrase_queries[:5]:
#     print(f"  {query}")
# 
# question_queries = generator.generate_question_queries(num_queries=10)
# print("\nSynthetic Question Queries:")
# for query in question_queries[:5]:
#     print(f"  {query}")
```

---

## Relevance Judgments

### Judgment Collection

```python
class JudgmentCollector:
    """Collect relevance judgments for query-document pairs."""
    
    def __init__(self, dataset: GoldenDataset):
        self.dataset = dataset
    
    def create_judgment_task(self, query_id: str, top_k: int = 20) -> List[Dict]:
        """
        Create judgment task for annotators.
        
        Returns list of document IDs to judge for a query.
        """
        query = next((q for q in self.dataset.queries if q.id == query_id), None)
        if not query:
            raise ValueError(f"Query {query_id} not found")
        
        # In practice, these would be top results from search
        # For demonstration, select random documents
        import random
        doc_sample = random.sample(self.dataset.documents, min(top_k, len(self.dataset.documents)))
        
        task = {
            'query_id': query_id,
            'query_text': query.text,
            'query_intent': query.intent,
            'documents': [
                {
                    'id': doc.id,
                    'title': doc.title,
                    'content': doc.content[:200] + '...',  # Preview
                    'category': doc.category
                }
                for doc in doc_sample
            ]
        }
        
        return task
    
    def collect_judgment(self, query_id: str, document_id: str, 
                        relevance: int, judge_id: str, notes: str = None):
        """
        Collect a single relevance judgment.
        
        Args:
            query_id: Query identifier
            document_id: Document identifier
            relevance: Relevance score (0-4)
            judge_id: Annotator identifier
            notes: Optional notes about judgment
        """
        if not 0 <= relevance <= 4:
            raise ValueError("Relevance must be 0-4")
        
        judgment = RelevanceJudgment(
            query_id=query_id,
            document_id=document_id,
            relevance=relevance,
            judge_id=judge_id,
            timestamp=datetime.now().isoformat(),
            notes=notes
        )
        
        self.dataset.add_judgment(judgment)
        return judgment
    
    def get_judgment_guidelines(self) -> Dict:
        """
        Return judgment guidelines for annotators.
        """
        return {
            'scale': {
                0: {
                    'label': 'Not Relevant',
                    'description': 'Document has no relevance to the query',
                    'example': 'Query: "python programming", Doc: "Java basics"'
                },
                1: {
                    'label': 'Slightly Relevant',
                    'description': 'Document tangentially related but not useful',
                    'example': 'Query: "python programming", Doc: "History of programming languages"'
                },
                2: {
                    'label': 'Moderately Relevant',
                    'description': 'Document contains useful information but incomplete',
                    'example': 'Query: "python programming", Doc: "Introduction to programming (mentions Python)"'
                },
                3: {
                    'label': 'Highly Relevant',
                    'description': 'Document directly addresses the query',
                    'example': 'Query: "python programming", Doc: "Python Programming Tutorial"'
                },
                4: {
                    'label': 'Perfectly Relevant',
                    'description': 'Document is the ideal answer to the query',
                    'example': 'Query: "python programming", Doc: "Comprehensive Python Programming Guide"'
                }
            },
            'guidelines': [
                'Consider query intent (informational, navigational, transactional)',
                'Evaluate content quality and completeness',
                'Check for current and accurate information',
                'Consider document authority and credibility',
                'Be consistent with rating scale'
            ]
        }

# Usage
collector = JudgmentCollector(dataset)

# Get judgment guidelines
guidelines = collector.get_judgment_guidelines()
print("Relevance Judgment Scale:")
for score, info in guidelines['scale'].items():
    print(f"\n{score} - {info['label']}:")
    print(f"  {info['description']}")
    print(f"  Example: {info['example']}")

# Create judgment task
# task = collector.create_judgment_task("q1", top_k=10)
# print(f"\nJudgment Task for Query: {task['query_text']}")
# print(f"Documents to judge: {len(task['documents'])}")

# Collect judgment
# collector.collect_judgment(
#     query_id="q1",
#     document_id="doc1",
#     relevance=4,
#     judge_id="annotator1",
#     notes="Perfect match for query intent"
# )
```

### Inter-Annotator Agreement

```python
import numpy as np
from typing import List

class InterAnnotatorAgreement:
    """Calculate inter-annotator agreement metrics."""
    
    @staticmethod
    def calculate_cohens_kappa(judgments_a: List[int], judgments_b: List[int]) -> float:
        """
        Calculate Cohen's Kappa for two annotators.
        
        Kappa interpretation:
        < 0: Poor agreement
        0.0 - 0.20: Slight agreement
        0.21 - 0.40: Fair agreement
        0.41 - 0.60: Moderate agreement
        0.61 - 0.80: Substantial agreement
        0.81 - 1.00: Almost perfect agreement
        """
        if len(judgments_a) != len(judgments_b):
            raise ValueError("Judgment lists must be same length")
        
        n = len(judgments_a)
        
        # Observed agreement
        agreements = sum(1 for a, b in zip(judgments_a, judgments_b) if a == b)
        p_observed = agreements / n
        
        # Expected agreement by chance
        unique_values = set(judgments_a + judgments_b)
        p_expected = 0
        
        for value in unique_values:
            p_a = judgments_a.count(value) / n
            p_b = judgments_b.count(value) / n
            p_expected += p_a * p_b
        
        # Cohen's Kappa
        if p_expected == 1:
            return 1.0
        
        kappa = (p_observed - p_expected) / (1 - p_expected)
        return kappa
    
    @staticmethod
    def calculate_fleiss_kappa(judgments_matrix: np.ndarray) -> float:
        """
        Calculate Fleiss' Kappa for multiple annotators.
        
        Args:
            judgments_matrix: Shape (n_items, n_categories)
                             Each row is counts of ratings for an item
        """
        n_items, n_categories = judgments_matrix.shape
        n_raters = judgments_matrix.sum(axis=1)[0]  # Assuming same number of raters per item
        
        # Proportion of ratings in each category
        p_j = judgments_matrix.sum(axis=0) / (n_items * n_raters)
        
        # Calculate P_i (agreement for each item)
        P_i = []
        for i in range(n_items):
            sum_squares = sum(judgments_matrix[i, j] ** 2 for j in range(n_categories))
            P_i.append((sum_squares - n_raters) / (n_raters * (n_raters - 1)))
        
        # Mean agreement
        P_bar = np.mean(P_i)
        
        # Expected agreement
        P_e = sum(p_j[j] ** 2 for j in range(n_categories))
        
        # Fleiss' Kappa
        if P_e == 1:
            return 1.0
        
        kappa = (P_bar - P_e) / (1 - P_e)
        return kappa
    
    @staticmethod
    def interpret_kappa(kappa: float) -> str:
        """Interpret Kappa value."""
        if kappa < 0:
            return "Poor agreement"
        elif kappa < 0.21:
            return "Slight agreement"
        elif kappa < 0.41:
            return "Fair agreement"
        elif kappa < 0.61:
            return "Moderate agreement"
        elif kappa < 0.81:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

# Usage
iaa = InterAnnotatorAgreement()

# Two annotators' judgments for same items
annotator1 = [4, 3, 2, 4, 1, 3, 2, 4, 3, 2]
annotator2 = [4, 3, 2, 3, 1, 3, 2, 4, 3, 1]

kappa = iaa.calculate_cohens_kappa(annotator1, annotator2)
interpretation = iaa.interpret_kappa(kappa)

print(f"Cohen's Kappa: {kappa:.3f}")
print(f"Interpretation: {interpretation}")

# Fleiss' Kappa for multiple annotators
# Matrix where rows are items, columns are rating categories (0-4)
# Each cell is count of annotators who gave that rating
judgments = np.array([
    [0, 0, 1, 2, 0],  # Item 1: one rated 2, two rated 3
    [0, 0, 0, 1, 2],  # Item 2: one rated 3, two rated 4
    [0, 1, 1, 1, 0],  # Item 3: one each rated 1, 2, 3
])

fleiss = iaa.calculate_fleiss_kappa(judgments)
print(f"\nFleiss' Kappa: {fleiss:.3f}")
print(f"Interpretation: {iaa.interpret_kappa(fleiss)}")
```

---

## Dataset Generation

### Complete Dataset Builder

```python
class DatasetBuilder:
    """Build complete evaluation dataset."""
    
    def __init__(self):
        self.dataset = GoldenDataset()
    
    def build_ecommerce_dataset(self, num_products: int = 1000,
                                num_queries: int = 100) -> GoldenDataset:
        """
        Build e-commerce product dataset.
        """
        import random
        
        # Product categories
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports']
        
        # Generate products
        for i in range(num_products):
            product = Document(
                id=f"prod{i:04d}",
                title=f"Product {i}",
                content=f"Description for product {i}...",
                category=random.choice(categories),
                tags=[f"tag{random.randint(1, 20)}" for _ in range(3)],
                metadata={
                    'price': round(random.uniform(10, 1000), 2),
                    'rating': round(random.uniform(1, 5), 1),
                    'brand': f"Brand{random.randint(1, 50)}"
                }
            )
            self.dataset.add_document(product)
        
        # Generate queries
        query_templates = [
            "{} products",
            "best {}",
            "cheap {}",
            "{} reviews",
            "buy {}"
        ]
        
        for i in range(num_queries):
            category = random.choice(categories)
            template = random.choice(query_templates)
            query_text = template.format(category.lower())
            
            query = Query(
                id=f"query{i:03d}",
                text=query_text,
                intent=random.choice(['informational', 'transactional', 'navigational']),
                difficulty=random.choice(['easy', 'medium', 'hard']),
                category=category,
                expected_result_count=random.randint(5, 50),
                metadata={}
            )
            self.dataset.add_query(query)
        
        print(f"Built e-commerce dataset:")
        print(f"  Products: {num_products}")
        print(f"  Queries: {num_queries}")
        
        return self.dataset
    
    def add_sample_judgments(self, judgments_per_query: int = 10):
        """Add sample relevance judgments."""
        import random
        
        for query in self.dataset.queries:
            # Sample documents to judge
            docs_to_judge = random.sample(
                self.dataset.documents,
                min(judgments_per_query, len(self.dataset.documents))
            )
            
            for doc in docs_to_judge:
                # Simulate relevance judgment
                relevance = random.randint(0, 4)
                
                judgment = RelevanceJudgment(
                    query_id=query.id,
                    document_id=doc.id,
                    relevance=relevance,
                    judge_id="auto",
                    timestamp=datetime.now().isoformat()
                )
                self.dataset.add_judgment(judgment)
        
        print(f"Added {len(self.dataset.judgments)} judgments")

# Usage
builder = DatasetBuilder()

# Build dataset
ecommerce_dataset = builder.build_ecommerce_dataset(
    num_products=500,
    num_queries=50
)

# Add judgments
builder.add_sample_judgments(judgments_per_query=15)

# Export
# ecommerce_dataset.export_to_json("datasets/ecommerce_golden_set")

print(f"\nDataset Summary:")
print(f"  Documents: {len(ecommerce_dataset.documents)}")
print(f"  Queries: {len(ecommerce_dataset.queries)}")
print(f"  Judgments: {len(ecommerce_dataset.judgments)}")
print(f"  Avg judgments per query: {len(ecommerce_dataset.judgments) / len(ecommerce_dataset.queries):.1f}")
```

---

## Quality Assurance

### Dataset Validation

```python
class DatasetValidator:
    """Validate dataset quality."""
    
    def __init__(self, dataset: GoldenDataset):
        self.dataset = dataset
    
    def validate_completeness(self) -> Dict:
        """Check dataset completeness."""
        issues = []
        
        # Check for empty components
        if not self.dataset.documents:
            issues.append("No documents in dataset")
        
        if not self.dataset.queries:
            issues.append("No queries in dataset")
        
        if not self.dataset.judgments:
            issues.append("No judgments in dataset")
        
        # Check for queries without judgments
        queries_without_judgments = []
        for query in self.dataset.queries:
            judgments = self.dataset.get_judgments_for_query(query.id)
            if not judgments:
                queries_without_judgments.append(query.id)
        
        if queries_without_judgments:
            issues.append(f"{len(queries_without_judgments)} queries without judgments")
        
        return {
            'complete': len(issues) == 0,
            'issues': issues,
            'queries_without_judgments': queries_without_judgments
        }
    
    def validate_judgment_coverage(self, min_judgments_per_query: int = 5) -> Dict:
        """Validate judgment coverage."""
        coverage_issues = []
        
        for query in self.dataset.queries:
            judgments = self.dataset.get_judgments_for_query(query.id)
            if len(judgments) < min_judgments_per_query:
                coverage_issues.append({
                    'query_id': query.id,
                    'current_judgments': len(judgments),
                    'required': min_judgments_per_query
                })
        
        return {
            'sufficient_coverage': len(coverage_issues) == 0,
            'issues': coverage_issues,
            'coverage_percentage': (
                len([q for q in self.dataset.queries 
                     if len(self.dataset.get_judgments_for_query(q.id)) >= min_judgments_per_query])
                / len(self.dataset.queries) * 100
                if self.dataset.queries else 0
            )
        }
    
    def validate_judgment_distribution(self) -> Dict:
        """Analyze judgment distribution."""
        relevance_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        for judgment in self.dataset.judgments:
            relevance_counts[judgment.relevance] += 1
        
        total = len(self.dataset.judgments)
        distribution = {
            score: {
                'count': count,
                'percentage': (count / total * 100) if total > 0 else 0
            }
            for score, count in relevance_counts.items()
        }
        
        # Check for imbalanced distribution
        warnings = []
        for score, stats in distribution.items():
            if stats['percentage'] > 60:
                warnings.append(f"Score {score} over-represented ({stats['percentage']:.1f}%)")
            elif stats['percentage'] < 5 and stats['count'] > 0:
                warnings.append(f"Score {score} under-represented ({stats['percentage']:.1f}%)")
        
        return {
            'distribution': distribution,
            'balanced': len(warnings) == 0,
            'warnings': warnings
        }
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        completeness = self.validate_completeness()
        coverage = self.validate_judgment_coverage()
        distribution = self.validate_judgment_distribution()
        
        report = "Dataset Validation Report\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Dataset Size:\n"
        report += f"  Documents: {len(self.dataset.documents)}\n"
        report += f"  Queries: {len(self.dataset.queries)}\n"
        report += f"  Judgments: {len(self.dataset.judgments)}\n\n"
        
        report += f"Completeness: {'✓ PASS' if completeness['complete'] else '✗ FAIL'}\n"
        for issue in completeness['issues']:
            report += f"  • {issue}\n"
        report += "\n"
        
        report += f"Judgment Coverage: {'✓ PASS' if coverage['sufficient_coverage'] else '✗ FAIL'}\n"
        report += f"  Coverage: {coverage['coverage_percentage']:.1f}%\n"
        report += f"  Issues: {len(coverage['issues'])}\n\n"
        
        report += f"Judgment Distribution: {'✓ PASS' if distribution['balanced'] else '⚠ WARNING'}\n"
        for score, stats in distribution['distribution'].items():
            report += f"  Score {score}: {stats['count']} ({stats['percentage']:.1f}%)\n"
        
        if distribution['warnings']:
            report += "\nWarnings:\n"
            for warning in distribution['warnings']:
                report += f"  • {warning}\n"
        
        return report

# Usage
validator = DatasetValidator(ecommerce_dataset)

# Validate completeness
completeness = validator.validate_completeness()
print("Completeness Check:")
print(f"  Complete: {completeness['complete']}")
if completeness['issues']:
    for issue in completeness['issues']:
        print(f"  • {issue}")

# Validate coverage
coverage = validator.validate_judgment_coverage(min_judgments_per_query=10)
print(f"\nJudgment Coverage: {coverage['coverage_percentage']:.1f}%")

# Generate full report
print("\n" + validator.generate_validation_report())
```

---

## Dataset Management

### Version Control

```python
class DatasetVersionControl:
    """Manage dataset versions."""
    
    def __init__(self, base_path: str = "datasets"):
        self.base_path = base_path
        import os
        os.makedirs(base_path, exist_ok=True)
    
    def save_version(self, dataset: GoldenDataset, version: str, notes: str = ""):
        """Save dataset version with metadata."""
        import os
        import hashlib
        
        version_dir = f"{self.base_path}/v{version}"
        os.makedirs(version_dir, exist_ok=True)
        
        # Export dataset
        dataset.export_to_json(version_dir)
        
        # Calculate checksums
        checksums = {}
        for filename in ['documents.json', 'queries.json', 'judgments.json']:
            filepath = f"{version_dir}/{filename}"
            with open(filepath, 'rb') as f:
                checksums[filename] = hashlib.md5(f.read()).hexdigest()
        
        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'notes': notes,
            'statistics': {
                'documents': len(dataset.documents),
                'queries': len(dataset.queries),
                'judgments': len(dataset.judgments)
            },
            'checksums': checksums
        }
        
        with open(f"{version_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved dataset version {version}")
        return metadata
    
    def load_version(self, version: str) -> GoldenDataset:
        """Load specific dataset version."""
        version_dir = f"{self.base_path}/v{version}"
        
        dataset = GoldenDataset()
        dataset.load_from_json(version_dir)
        
        # Load metadata
        with open(f"{version_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded dataset version {version}")
        print(f"  Created: {metadata['timestamp']}")
        print(f"  Notes: {metadata['notes']}")
        
        return dataset
    
    def list_versions(self) -> List[Dict]:
        """List all dataset versions."""
        import os
        
        versions = []
        for item in os.listdir(self.base_path):
            if item.startswith('v') and os.path.isdir(f"{self.base_path}/{item}"):
                metadata_path = f"{self.base_path}/{item}/metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        versions.append(metadata)
        
        # Sort by timestamp
        versions.sort(key=lambda x: x['timestamp'], reverse=True)
        return versions

# Usage
version_control = DatasetVersionControl()

# Save current dataset as version
# version_control.save_version(
#     ecommerce_dataset,
#     version="1.0",
#     notes="Initial e-commerce dataset with 500 products"
# )

# List versions
# versions = version_control.list_versions()
# print("Dataset Versions:")
# for v in versions:
#     print(f"  v{v['version']}: {v['timestamp']} - {v['notes']}")

# Load specific version
# loaded_dataset = version_control.load_version("1.0")
```

---

## Best Practices

### 1. Start with Query Log Mining

**Why It Matters**

Synthetic queries created by engineers rarely match real user behavior. Query logs reveal actual search patterns, terminology, and intent distribution.

**Implementation**

```python
class ProductionQueryLogMiner:
    """Mine production query logs for dataset creation."""
    
    def __init__(self):
        self.queries = []
    
    def load_query_logs(self, log_source, days=90):
        """
        Load query logs from production.
        
        Args:
            log_source: Path to log files or database connection
            days: Number of days to analyze (90 recommended)
        """
        # Implementation would load from actual log source
        # For demonstration, showing structure
        pass
    
    def analyze_query_distribution(self):
        """
        Analyze query frequency distribution.
        
        Returns head, torso, and long-tail query segments.
        """
        from collections import Counter
        
        query_counts = Counter(self.queries)
        total_queries = len(self.queries)
        unique_queries = len(query_counts)
        
        # Classify queries by frequency
        head_queries = []      # >100 occurrences
        torso_queries = []     # 10-100 occurrences
        long_tail_queries = [] # 2-9 occurrences
        unique_queries_list = [] # 1 occurrence
        
        for query, count in query_counts.items():
            if count > 100:
                head_queries.append((query, count))
            elif count >= 10:
                torso_queries.append((query, count))
            elif count >= 2:
                long_tail_queries.append((query, count))
            else:
                unique_queries_list.append((query, count))
        
        # Calculate coverage
        head_volume = sum(count for _, count in head_queries)
        torso_volume = sum(count for _, count in torso_queries)
        long_tail_volume = sum(count for _, count in long_tail_queries)
        
        return {
            'total_queries': total_queries,
            'unique_queries': unique_queries,
            'segments': {
                'head': {
                    'query_count': len(head_queries),
                    'volume': head_volume,
                    'volume_pct': (head_volume / total_queries * 100),
                    'queries': sorted(head_queries, key=lambda x: x[1], reverse=True)[:50]
                },
                'torso': {
                    'query_count': len(torso_queries),
                    'volume': torso_volume,
                    'volume_pct': (torso_volume / total_queries * 100),
                    'queries': sorted(torso_queries, key=lambda x: x[1], reverse=True)[:100]
                },
                'long_tail': {
                    'query_count': len(long_tail_queries),
                    'volume': long_tail_volume,
                    'volume_pct': (long_tail_volume / total_queries * 100),
                    'queries': sorted(long_tail_queries, key=lambda x: x[1], reverse=True)[:150]
                }
            }
        }
    
    def sample_stratified_queries(self, total_sample_size=500):
        """
        Sample queries proportionally across segments.
        
        Strategy:
        - Head queries: 40% of sample (high impact)
        - Torso queries: 35% of sample (breadth)
        - Long-tail: 25% of sample (coverage)
        """
        import random
        
        distribution = self.analyze_query_distribution()
        
        # Calculate sample sizes per segment
        head_sample = int(total_sample_size * 0.40)
        torso_sample = int(total_sample_size * 0.35)
        long_tail_sample = int(total_sample_size * 0.25)
        
        sampled_queries = []
        
        # Sample head (take top N by frequency)
        head_queries = distribution['segments']['head']['queries']
        sampled_queries.extend([q for q, _ in head_queries[:head_sample]])
        
        # Sample torso (random sample)
        torso_queries = distribution['segments']['torso']['queries']
        if len(torso_queries) >= torso_sample:
            sampled_queries.extend([q for q, _ in random.sample(torso_queries, torso_sample)])
        else:
            sampled_queries.extend([q for q, _ in torso_queries])
        
        # Sample long-tail (random sample)
        long_tail_queries = distribution['segments']['long_tail']['queries']
        if len(long_tail_queries) >= long_tail_sample:
            sampled_queries.extend([q for q, _ in random.sample(long_tail_queries, long_tail_sample)])
        else:
            sampled_queries.extend([q for q, _ in long_tail_queries])
        
        return {
            'queries': sampled_queries,
            'distribution': {
                'head': head_sample,
                'torso': len([q for q, _ in torso_queries]) if len(torso_queries) < torso_sample else torso_sample,
                'long_tail': len([q for q, _ in long_tail_queries]) if len(long_tail_queries) < long_tail_sample else long_tail_sample
            }
        }

# Usage example
log_miner = ProductionQueryLogMiner()

# Simulate loading queries
log_miner.queries = ["product search"] * 500 + ["laptop"] * 200 + ["best laptop 2024"] * 50 + ["gaming laptop under 1000"] * 10

distribution = log_miner.analyze_query_distribution()
print("Query Distribution:")
print(f"  Head: {distribution['segments']['head']['query_count']} queries "
      f"({distribution['segments']['head']['volume_pct']:.1f}% of volume)")
print(f"  Torso: {distribution['segments']['torso']['query_count']} queries "
      f"({distribution['segments']['torso']['volume_pct']:.1f}% of volume)")
print(f"  Long-tail: {distribution['segments']['long_tail']['query_count']} queries "
      f"({distribution['segments']['long_tail']['volume_pct']:.1f}% of volume)")

# Sample stratified set
sample_result = log_miner.sample_stratified_queries(total_sample_size=100)
print(f"\nSampled {len(sample_result['queries'])} queries:")
print(f"  Head: {sample_result['distribution']['head']}")
print(f"  Torso: {sample_result['distribution']['torso']}")
print(f"  Long-tail: {sample_result['distribution']['long_tail']}")
```

**Best Practices**:
- ✓ Analyze 60-90 days of logs (seasonal variation)
- ✓ Stratify sampling (head + torso + long-tail)
- ✓ Include failed queries (zero results, low CTR)
- ✓ Preserve original query text (typos, case, special chars)
- ✓ Tag queries with metadata (date, user segment, session context)

---

### 2. Use Document Pooling

**The Problem with Single-System Judging**

If you only judge documents from your current system, you bias the dataset toward current behavior. Future improvements may surface different documents that are unjudged.

**Solution: Pool from Multiple Systems**

```python
class DocumentPooling:
    """Pool documents from multiple search configurations."""
    
    def __init__(self, search_configs):
        """
        Args:
            search_configs: List of search configurations to pool from
        """
        self.search_configs = search_configs
    
    def create_judgment_pool(self, query, top_k=20):
        """
        Create pooled set of documents for judgment.
        
        Args:
            query: Query text
            top_k: Number of results per configuration
        
        Returns:
            Deduplicated set of documents across all configs
        """
        pooled_docs = set()
        
        for config in self.search_configs:
            # Run query against this configuration
            results = self._run_search(query, config, top_k)
            
            # Add document IDs to pool
            for result in results:
                pooled_docs.add(result['id'])
        
        return list(pooled_docs)
    
    def _run_search(self, query, config, top_k):
        """Run search against specific configuration."""
        # Implementation would execute search
        # Returns list of {'id': doc_id, 'score': score}
        return []
    
    def calculate_pool_coverage(self, query, pool_docs, ground_truth_relevant):
        """
        Calculate what percentage of relevant docs are in pool.
        
        Recall@Pool: Are we capturing relevant docs?
        """
        relevant_in_pool = len(set(pool_docs) & set(ground_truth_relevant))
        total_relevant = len(ground_truth_relevant)
        
        coverage = (relevant_in_pool / total_relevant * 100) if total_relevant > 0 else 0
        
        return {
            'coverage_pct': coverage,
            'relevant_in_pool': relevant_in_pool,
            'total_relevant': total_relevant,
            'missing_relevant': total_relevant - relevant_in_pool
        }

# Example configurations to pool from
configs = [
    {'name': 'baseline', 'type': 'bm25'},
    {'name': 'enhanced_bm25', 'type': 'bm25', 'analyzer': 'custom'},
    {'name': 'hybrid', 'type': 'hybrid', 'rrf_weights': [0.7, 0.3]},
    {'name': 'semantic', 'type': 'vector'},
    {'name': 'tuned', 'type': 'hybrid', 'scoring_profile': 'relevance_v2'}
]

pooler = DocumentPooling(configs)

# Create pool for query
# pool = pooler.create_judgment_pool("machine learning tutorial", top_k=20)
# 
# This creates pool of up to 100 documents (5 configs × 20 docs)
# Deduplication typically reduces to 40-60 unique documents
```

**Pooling Strategy Benefits**:

| Strategy | Pool Size | Coverage | Annotation Cost | Bias Risk |
|----------|-----------|----------|-----------------|-----------|
| **Single system** (current prod) | 20 docs | 60-70% | Low | High (biased to current) |
| **Dual system** (current + new) | 30-35 docs | 80-85% | Medium | Medium |
| **Multi-system pooling** (5+ configs) | 40-60 docs | 90-95% | High | Low (diverse) |

**Recommended**: Pool from 3-5 diverse configurations (baseline, proposed improvements, competitors)

---

### 3. Recruit Domain Expert Annotators

**Why Domain Expertise Matters**

Generic annotators can judge surface relevance ("does this mention the topic?") but miss nuanced domain-specific criteria.

**Domain-Specific Relevance Factors**

```python
class DomainExpertGuidelines:
    """Generate domain-specific annotation guidelines."""
    
    @staticmethod
    def legal_search_criteria():
        """Relevance criteria for legal search."""
        return {
            'primary_factors': {
                'jurisdiction': {
                    'weight': 'critical',
                    'description': 'Binding precedent > persuasive authority',
                    'example': 'California case binding in CA, persuasive in other states'
                },
                'recency': {
                    'weight': 'high',
                    'description': 'More recent cases may overturn older precedent',
                    'example': '2023 case overrules 1995 case on same issue'
                },
                'court_level': {
                    'weight': 'high',
                    'description': 'Supreme Court > Appeals Court > Trial Court',
                    'example': 'US Supreme Court decision > State Supreme Court'
                },
                'case_status': {
                    'weight': 'critical',
                    'description': 'Current good law vs overturned/superseded',
                    'example': 'Case overturned = not relevant (score 0)'
                }
            },
            'secondary_factors': {
                'citation_count': 'Frequently cited cases likely more authoritative',
                'factual_similarity': 'Similar facts to query scenario',
                'legal_issue_match': 'Addresses same legal question'
            }
        }
    
    @staticmethod
    def medical_search_criteria():
        """Relevance criteria for medical/scientific search."""
        return {
            'primary_factors': {
                'evidence_quality': {
                    'weight': 'critical',
                    'description': 'RCT > cohort study > case report',
                    'hierarchy': ['Systematic review', 'RCT', 'Cohort study', 'Case-control', 'Case report']
                },
                'publication_venue': {
                    'weight': 'high',
                    'description': 'Peer-reviewed journal > preprint',
                    'example': 'NEJM article > arXiv preprint'
                },
                'recency': {
                    'weight': 'high',
                    'description': 'Medical knowledge evolves rapidly',
                    'threshold': 'Studies >10 years old may be outdated'
                },
                'sample_size': {
                    'weight': 'medium',
                    'description': 'Larger samples more reliable',
                    'example': 'n=10,000 > n=50'
                }
            }
        }
    
    @staticmethod
    def ecommerce_search_criteria():
        """Relevance criteria for e-commerce product search."""
        return {
            'primary_factors': {
                'query_match': {
                    'weight': 'critical',
                    'description': 'Product matches query intent',
                    'examples': {
                        'exact': 'Query: "iPhone 15" → iPhone 15 (score 4)',
                        'compatible': 'Query: "iPhone 15" → iPhone 15 case (score 2)',
                        'unrelated': 'Query: "iPhone 15" → Samsung Galaxy (score 0)'
                    }
                },
                'availability': {
                    'weight': 'high',
                    'description': 'In-stock > out-of-stock',
                    'rule': 'Out-of-stock product max score 2 (even if perfect match)'
                },
                'price_range': {
                    'weight': 'medium',
                    'description': 'Within expected price range for query',
                    'example': 'Query: "cheap laptop" → $2000 laptop (score ≤1)'
                }
            }
        }

# Usage: Provide to annotators based on domain
legal_guidelines = DomainExpertGuidelines.legal_search_criteria()
print("Legal Search Annotation Guidelines:")
print("\nPrimary Factors:")
for factor, details in legal_guidelines['primary_factors'].items():
    print(f"  {factor}: {details['description']}")
```

**Annotator Qualification**:

| Domain | Annotator Profile | Training Required | Example Qualifications |
|--------|-------------------|-------------------|------------------------|
| **Legal** | Attorneys, paralegals | 4-8 hours | 5+ years experience, jurisdiction knowledge |
| **Medical** | Doctors, researchers | 6-10 hours | MD/PhD, specialty knowledge |
| **E-commerce** | Product specialists | 2-4 hours | Category expertise (e.g., electronics buyer) |
| **Technical** | Engineers, developers | 3-6 hours | Domain certification, hands-on experience |
| **General web** | Trained annotators | 2-3 hours | College educated, web research skills |

**Cost vs Quality Trade-off**:
- Generic annotator: $15-25/hour, κ=0.45-0.55 (moderate agreement)
- Domain expert: $50-150/hour, κ=0.65-0.80 (substantial agreement)
- **Recommendation**: Use domain experts for critical domains, generic for general web search

---

### 4. Implement Triple Annotation

**Why Multiple Annotators**

Single annotator introduces bias and error. Multiple annotations enable:
- Inter-annotator agreement measurement (data quality metric)
- Disagreement resolution (improve consistency)
- Majority voting (more reliable gold labels)

**Triple Annotation Strategy**

```python
class TripleAnnotationManager:
    """Manage triple annotation process."""
    
    def __init__(self):
        self.annotations = {}  # (query_id, doc_id) -> [judge1_score, judge2_score, judge3_score]
    
    def collect_annotation(self, query_id, doc_id, judge_id, score):
        """Collect individual annotation."""
        key = (query_id, doc_id)
        if key not in self.annotations:
            self.annotations[key] = {}
        
        self.annotations[key][judge_id] = score
    
    def calculate_agreement(self, query_id, doc_id):
        """
        Calculate agreement for a query-document pair.
        
        Returns:
            - perfect: All 3 annotators agree exactly
            - strong: 2 annotators agree, 1 differs by ±1
            - moderate: 2 annotators agree, 1 differs by ≥2
            - weak: All 3 annotators disagree
        """
        key = (query_id, doc_id)
        scores = list(self.annotations[key].values())
        
        if len(scores) < 3:
            return 'incomplete'
        
        # Check for perfect agreement
        if len(set(scores)) == 1:
            return 'perfect'
        
        # Check for strong agreement (2 agree, 1 within ±1)
        from collections import Counter
        score_counts = Counter(scores)
        most_common_score, most_common_count = score_counts.most_common(1)[0]
        
        if most_common_count == 2:
            # 2 annotators agree
            other_score = [s for s in scores if s != most_common_score][0]
            diff = abs(most_common_score - other_score)
            
            if diff <= 1:
                return 'strong'
            else:
                return 'moderate'
        
        # All 3 disagree
        return 'weak'
    
    def resolve_disagreement(self, query_id, doc_id, strategy='majority'):
        """
        Resolve disagreement to produce gold label.
        
        Strategies:
        - majority: Use most common score (if 2/3 agree)
        - mean: Use mean of 3 scores (rounded)
        - median: Use median score
        - expert: Use score from most senior annotator
        """
        key = (query_id, doc_id)
        scores = list(self.annotations[key].values())
        
        if strategy == 'majority':
            from collections import Counter
            score_counts = Counter(scores)
            most_common_score, count = score_counts.most_common(1)[0]
            
            if count >= 2:
                return most_common_score
            else:
                # No majority, fall back to median
                return int(np.median(scores))
        
        elif strategy == 'mean':
            return int(round(np.mean(scores)))
        
        elif strategy == 'median':
            return int(np.median(scores))
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def flag_for_review(self, max_disagreement=2):
        """
        Flag pairs with high disagreement for expert review.
        
        Args:
            max_disagreement: Maximum acceptable score range
        
        Returns:
            List of (query_id, doc_id) pairs needing review
        """
        flagged = []
        
        for (query_id, doc_id), judge_scores in self.annotations.items():
            scores = list(judge_scores.values())
            
            if len(scores) < 3:
                continue
            
            score_range = max(scores) - min(scores)
            
            if score_range >= max_disagreement:
                flagged.append({
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'scores': scores,
                    'range': score_range,
                    'agreement': self.calculate_agreement(query_id, doc_id)
                })
        
        return sorted(flagged, key=lambda x: x['range'], reverse=True)

# Usage
manager = TripleAnnotationManager()

# Collect annotations
manager.collect_annotation('q1', 'doc1', 'judge1', 4)
manager.collect_annotation('q1', 'doc1', 'judge2', 4)
manager.collect_annotation('q1', 'doc1', 'judge3', 3)

# Check agreement
agreement = manager.calculate_agreement('q1', 'doc1')
print(f"Agreement: {agreement}")  # "strong" (2 agree at 4, 1 at 3)

# Resolve disagreement
gold_label = manager.resolve_disagreement('q1', 'doc1', strategy='majority')
print(f"Gold label: {gold_label}")  # 4 (majority vote)

# Flag high disagreements
manager.collect_annotation('q2', 'doc2', 'judge1', 4)
manager.collect_annotation('q2', 'doc2', 'judge2', 1)
manager.collect_annotation('q2', 'doc2', 'judge3', 0)

flagged = manager.flag_for_review(max_disagreement=2)
print(f"\nFlagged for review: {len(flagged)} pairs")
for item in flagged:
    print(f"  {item['query_id']}, {item['doc_id']}: scores={item['scores']}, range={item['range']}")
```

**Agreement Targets**:
- **Excellent**: κ ≥ 0.75 (75%+ of pairs in perfect/strong agreement)
- **Good**: κ = 0.60-0.74 (acceptable for most domains)
- **Marginal**: κ = 0.40-0.59 (needs calibration training)
- **Poor**: κ < 0.40 (unclear guidelines, need expert review)

---

### 5. Calculate and Track Inter-Annotator Agreement

**Measuring Data Quality**

Inter-annotator agreement (IAA) is the **primary quality metric** for relevance judgments. Low agreement = unreliable dataset.

```python
class InterAnnotatorAgreementTracker:
    """Track IAA over time to monitor dataset quality."""
    
    def __init__(self):
        self.weekly_kappa = []
    
    def calculate_cohens_kappa(self, judge1_scores, judge2_scores):
        """Calculate Cohen's Kappa between two judges."""
        if len(judge1_scores) != len(judge2_scores):
            raise ValueError("Score lists must be same length")
        
        n = len(judge1_scores)
        
        # Observed agreement
        agreements = sum(1 for a, b in zip(judge1_scores, judge2_scores) if a == b)
        p_o = agreements / n
        
        # Expected agreement by chance
        from collections import Counter
        counts1 = Counter(judge1_scores)
        counts2 = Counter(judge2_scores)
        
        unique_scores = set(judge1_scores + judge2_scores)
        p_e = sum((counts1[score]/n) * (counts2[score]/n) for score in unique_scores)
        
        # Cohen's Kappa
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0
        
        return kappa
    
    def track_weekly_agreement(self, week_number, annotations):
        """
        Track agreement for a week of annotations.
        
        Args:
            week_number: Week identifier
            annotations: List of (judge1_score, judge2_score, judge3_score) tuples
        """
        # Calculate pairwise kappa
        judge1_scores = [a[0] for a in annotations]
        judge2_scores = [a[1] for a in annotations]
        judge3_scores = [a[2] for a in annotations]
        
        kappa_12 = self.calculate_cohens_kappa(judge1_scores, judge2_scores)
        kappa_13 = self.calculate_cohens_kappa(judge1_scores, judge3_scores)
        kappa_23 = self.calculate_cohens_kappa(judge2_scores, judge3_scores)
        
        # Average kappa
        avg_kappa = (kappa_12 + kappa_13 + kappa_23) / 3
        
        self.weekly_kappa.append({
            'week': week_number,
            'kappa_avg': avg_kappa,
            'kappa_pairs': {
                'judge_1_2': kappa_12,
                'judge_1_3': kappa_13,
                'judge_2_3': kappa_23
            },
            'annotation_count': len(annotations)
        })
        
        return avg_kappa
    
    def generate_quality_report(self):
        """Generate IAA quality report."""
        if not self.weekly_kappa:
            return "No data available"
        
        latest_kappa = self.weekly_kappa[-1]['kappa_avg']
        kappa_trend = self._calculate_trend()
        
        report = "Inter-Annotator Agreement Report\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Current Agreement (Latest Week):\n"
        report += f"  Kappa: {latest_kappa:.3f}\n"
        report += f"  Quality: {self._interpret_kappa(latest_kappa)}\n"
        report += f"  Trend: {kappa_trend}\n\n"
        
        report += "Weekly Progress:\n"
        for week_data in self.weekly_kappa:
            status = "✓" if week_data['kappa_avg'] >= 0.60 else "⚠"
            report += f"  {status} Week {week_data['week']}: κ={week_data['kappa_avg']:.3f} ({week_data['annotation_count']} annotations)\n"
        
        return report
    
    def _calculate_trend(self):
        """Calculate if kappa is improving, stable, or declining."""
        if len(self.weekly_kappa) < 2:
            return "insufficient data"
        
        recent_kappa = [w['kappa_avg'] for w in self.weekly_kappa[-3:]]
        avg_change = (recent_kappa[-1] - recent_kappa[0]) / len(recent_kappa)
        
        if avg_change > 0.05:
            return "improving ↑"
        elif avg_change < -0.05:
            return "declining ↓"
        else:
            return "stable →"
    
    def _interpret_kappa(self, kappa):
        """Interpret kappa value."""
        if kappa >= 0.75:
            return "Excellent"
        elif kappa >= 0.60:
            return "Good"
        elif kappa >= 0.40:
            return "Marginal (needs calibration)"
        else:
            return "Poor (review guidelines)"

# Usage
tracker = InterAnnotatorAgreementTracker()

# Week 1 annotations (example: low agreement initially)
week1_annotations = [(4,3,2), (3,3,1), (2,1,0), (4,4,3)] * 25  # 100 pairs
kappa1 = tracker.track_weekly_agreement(1, week1_annotations)
print(f"Week 1 Kappa: {kappa1:.3f}")

# Week 2 annotations (after calibration meeting)
week2_annotations = [(4,4,3), (3,3,3), (2,2,1), (4,4,4)] * 25
kappa2 = tracker.track_weekly_agreement(2, week2_annotations)
print(f"Week 2 Kappa: {kappa2:.3f}")

# Week 3 annotations (improved agreement)
week3_annotations = [(4,4,4), (3,3,3), (2,2,2), (4,4,4)] * 25
kappa3 = tracker.track_weekly_agreement(3, week3_annotations)
print(f"Week 3 Kappa: {kappa3:.3f}")

# Generate report
print("\n" + tracker.generate_quality_report())
```

**IAA Improvement Actions**:

| Kappa Range | Action Required |
|-------------|----------------|
| **< 0.40** | Emergency calibration meeting, review all guidelines, retrain annotators |
| **0.40-0.59** | Calibration session, clarify ambiguous guidelines, add examples |
| **0.60-0.74** | Minor calibration, discuss edge cases, maintain quality |
| **≥ 0.75** | Continue current process, spot-check for drift |

---

### 6. Version Control Datasets with Clear Metadata

**Why Versioning Matters**

Datasets evolve (new queries, updated judgments, corpus changes). Without versioning:
- Can't reproduce evaluation results from 6 months ago
- Don't know which dataset version was used for A/B test
- Can't track quality improvements over time

**Implementation** (shown earlier in Dataset Management section, reinforcing best practice)

```python
class DatasetVersioning:
    """Best practices for dataset versioning."""
    
    @staticmethod
    def version_naming_convention():
        """
        Recommended version naming:
        
        Format: {major}.{minor}.{patch}
        - Major: Significant changes (new query set, different scale)
        - Minor: Incremental additions (100+ new queries)
        - Patch: Bug fixes (corrected judgments, typo fixes)
        """
        return {
            'examples': {
                '1.0.0': 'Initial dataset (500 queries, 5K judgments)',
                '1.1.0': 'Added 100 long-tail queries',
                '1.1.1': 'Corrected 23 judgment errors',
                '2.0.0': 'New judgment scale (0-4 → 0-3), full re-annotation'
            },
            'metadata_required': [
                'version number',
                'creation date',
                'query count',
                'judgment count',
                'annotator list',
                'inter-annotator kappa',
                'changes from previous version',
                'compatible search configurations'
            ]
        }

# When saving dataset version
metadata = {
    'version': '1.2.0',
    'created': '2024-05-15',
    'queries': 650,
    'judgments': 6500,
    'annotators': ['ann1', 'ann2', 'ann3'],
    'avg_kappa': 0.72,
    'changes': 'Added 150 queries from Q1 2024 logs, focus on mobile search patterns',
    'compatible_with': ['index_v3', 'index_v4'],
    'supersedes': 'v1.1.0'
}
```

---

### 7. Validate Before Using

**Pre-Flight Validation Checklist**

Never use a dataset for evaluation without validating:

```python
class DatasetPreFlightValidator:
    """Comprehensive pre-flight validation before dataset use."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.errors = []
        self.warnings = []
    
    def run_all_checks(self):
        """Run all validation checks."""
        self.check_completeness()
        self.check_coverage()
        self.check_distribution()
        self.check_quality()
        
        return {
            'passed': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def check_completeness(self):
        """Ensure all required components present."""
        if len(self.dataset.queries) == 0:
            self.errors.append("No queries in dataset")
        elif len(self.dataset.queries) < 100:
            self.warnings.append(f"Only {len(self.dataset.queries)} queries (recommend 100+)")
        
        if len(self.dataset.judgments) == 0:
            self.errors.append("No judgments in dataset")
        
        # Check queries have judgments
        queries_without_judgments = 0
        for query in self.dataset.queries:
            judgments = self.dataset.get_judgments_for_query(query.id)
            if len(judgments) == 0:
                queries_without_judgments += 1
        
        if queries_without_judgments > 0:
            self.errors.append(f"{queries_without_judgments} queries have no judgments")
    
    def check_coverage(self):
        """Check judgment coverage per query."""
        min_judgments = 5
        low_coverage_queries = 0
        
        for query in self.dataset.queries:
            judgments = self.dataset.get_judgments_for_query(query.id)
            if len(judgments) < min_judgments:
                low_coverage_queries += 1
        
        if low_coverage_queries > len(self.dataset.queries) * 0.1:
            self.warnings.append(f"{low_coverage_queries} queries have <{min_judgments} judgments")
    
    def check_distribution(self):
        """Check relevance score distribution."""
        from collections import Counter
        
        relevance_counts = Counter(j.relevance for j in self.dataset.judgments)
        total = len(self.dataset.judgments)
        
        # Check for imbalance (>70% in one score)
        for score, count in relevance_counts.items():
            pct = (count / total) * 100
            if pct > 70:
                self.warnings.append(f"Imbalanced: {pct:.1f}% of judgments are score {score}")
    
    def check_quality(self):
        """Check data quality indicators."""
        # Check for duplicate judgments (same query-doc pair, same judge)
        seen = set()
        duplicates = 0
        
        for j in self.dataset.judgments:
            key = (j.query_id, j.document_id, j.judge_id)
            if key in seen:
                duplicates += 1
            seen.add(key)
        
        if duplicates > 0:
            self.errors.append(f"Found {duplicates} duplicate judgments")

# Usage
validator = DatasetPreFlightValidator(ecommerce_dataset)
result = validator.run_all_checks()

print("Dataset Validation:")
print(f"  Status: {'✓ PASSED' if result['passed'] else '✗ FAILED'}")

if result['errors']:
    print("\n  ERRORS:")
    for error in result['errors']:
        print(f"    ✗ {error}")

if result['warnings']:
    print("\n  WARNINGS:")
    for warning in result['warnings']:
        print(f"    ⚠ {warning}")

if result['passed'] and not result['warnings']:
    print("\n  Dataset ready for use!")
```

---

### 8. Refresh Datasets Periodically

**Why Datasets Decay**

- **Corpus changes**: New content added, old content updated/removed
- **Query patterns shift**: User behavior evolves, new terminology emerges
- **Relevance criteria change**: Domain standards update (e.g., legal precedent overturned)

**Refresh Strategy**

| Refresh Type | Frequency | What to Update | Effort |
|--------------|-----------|----------------|--------|
| **Patch** | Monthly | Fix errors, update 5-10% of judgments | Low (1-2 days) |
| **Minor** | Quarterly | Add 10-15% new queries from recent logs | Medium (1-2 weeks) |
| **Major** | Annually | Full review, re-judge 20%+, update guidelines | High (4-8 weeks) |

```python
class DatasetRefreshManager:
    """Manage periodic dataset refreshes."""
    
    def __init__(self, dataset, query_logs):
        self.dataset = dataset
        self.query_logs = query_logs
    
    def identify_refresh_candidates(self):
        """
        Identify queries/judgments needing refresh.
        
        Criteria:
        - New queries in logs not in dataset
        - Low-performing queries (high zero-result rate)
        - Outdated judgments (>1 year old)
        - Queries with corpus changes
        """
        refresh_needed = {
            'new_queries': [],
            'underperforming_queries': [],
            'outdated_judgments': [],
            'corpus_changes': []
        }
        
        # Find new high-frequency queries
        from collections import Counter
        query_counts = Counter(self.query_logs)
        
        existing_query_texts = {q.text for q in self.dataset.queries}
        
        for query_text, count in query_counts.most_common(200):
            if query_text not in existing_query_texts and count > 50:
                refresh_needed['new_queries'].append({
                    'query': query_text,
                    'frequency': count
                })
        
        # Find outdated judgments (>1 year old)
        from datetime import datetime, timedelta
        one_year_ago = datetime.now() - timedelta(days=365)
        
        for judgment in self.dataset.judgments:
            judgment_date = datetime.fromisoformat(judgment.timestamp)
            if judgment_date < one_year_ago:
                refresh_needed['outdated_judgments'].append(judgment)
        
        return refresh_needed
    
    def plan_incremental_refresh(self, refresh_candidates, budget_annotations=1000):
        """
        Plan incremental refresh within annotation budget.
        
        Priority:
        1. New high-frequency queries (biggest impact)
        2. Fix broken queries (improve quality)
        3. Refresh outdated (maintain currency)
        """
        plan = []
        remaining_budget = budget_annotations
        
        # Priority 1: New queries (10 judgments each)
        for item in refresh_candidates['new_queries'][:80]:  # Top 80 new queries
            if remaining_budget >= 10:
                plan.append({
                    'action': 'add_query',
                    'query': item['query'],
                    'judgments_needed': 10
                })
                remaining_budget -= 10
        
        # Priority 2: Re-judge underperforming
        for query in refresh_candidates['underperforming_queries'][:20]:
            if remaining_budget >= 15:
                plan.append({
                    'action': 're_judge',
                    'query_id': query,
                    'judgments_needed': 15  # More judgments to diagnose issues
                })
                remaining_budget -= 15
        
        return {
            'plan': plan,
            'total_annotations': budget_annotations - remaining_budget,
            'queries_added': len([p for p in plan if p['action'] == 'add_query']),
            'queries_refreshed': len([p for p in plan if p['action'] == 're_judge'])
        }

# Usage
# refresh_mgr = DatasetRefreshManager(ecommerce_dataset, recent_query_logs)
# 
# candidates = refresh_mgr.identify_refresh_candidates()
# print(f"Refresh Candidates:")
# print(f"  New queries: {len(candidates['new_queries'])}")
# print(f"  Outdated judgments: {len(candidates['outdated_judgments'])}")
# 
# refresh_plan = refresh_mgr.plan_incremental_refresh(candidates, budget_annotations=1000)
# print(f"\nRefresh Plan:")
# print(f"  Queries to add: {refresh_plan['queries_added']}")
# print(f"  Queries to refresh: {refresh_plan['queries_refreshed']}")
# print(f"  Total annotations: {refresh_plan['total_annotations']}")
```

---

## Troubleshooting

### 1. Low Inter-Annotator Agreement (κ < 0.40)

**Symptoms**: Annotators frequently disagree on relevance scores.

**Root Causes**:
- Ambiguous judgment guidelines
- Insufficient annotator training
- Complex domain requiring more expertise
- Query intent unclear

**Diagnosis**:

```python
def diagnose_low_agreement(annotations):
    """
    Analyze where disagreements occur.
    
    Returns categories of queries with low agreement.
    """
    from collections import defaultdict
    
    disagreement_by_category = defaultdict(list)
    
    for query_id, doc_id, scores in annotations:
        score_range = max(scores) - min(scores)
        
        if score_range >= 2:  # High disagreement
            # Categorize by query characteristics
            query = get_query(query_id)  # Hypothetical function
            
            disagreement_by_category[query.intent].append({
                'query_id': query_id,
                'doc_id': doc_id,
                'scores': scores,
                'range': score_range
            })
    
    # Find patterns
    print("Disagreement Patterns:")
    for category, items in disagreement_by_category.items():
        if len(items) > 10:  # Significant pattern
            print(f"  {category}: {len(items)} high-disagreement pairs")
            print(f"    → Review guidelines for {category} queries")

# Example output:
# Disagreement Patterns:
#   analytical: 45 high-disagreement pairs
#   → Review guidelines for analytical queries
```

**Solutions**:

1. **Calibration Meeting** (immediate):
   ```
   - Review 20-30 high-disagreement pairs together
   - Discuss rationale for each score
   - Clarify guidelines for ambiguous cases
   - Add examples to guidelines document
   ```

2. **Enhanced Training** (1-2 days):
   ```
   - Increase training examples from 20 → 50
   - Add edge cases and corner cases
   - Practice annotation with feedback
   - Verify understanding with quiz (10 test pairs)
   ```

3. **Simplify Scale** (if guidelines can't resolve):
   ```
   - Reduce from 5-point (0-4) to 3-point (0-2) scale
   - Binary relevant/not-relevant for simple domains
   - Clearer boundaries = higher agreement
   ```

4. **Domain Expert Review** (for complex domains):
   ```
   - Have senior expert annotate 100 pairs
   - Compare against other annotators
   - Use expert judgments as training examples
   - Senior expert resolves all high-disagreement pairs
   ```

---

### 2. Imbalanced Judgment Distribution

**Symptoms**: >70% of judgments are single score (e.g., all 0s or all 4s).

**Root Causes**:
- Poor query-document pairing (all irrelevant or all relevant)
- Annotators reluctant to use extreme scores
- Pooling strategy biased toward one relevance level

**Diagnosis**:

```python
def analyze_distribution_imbalance(dataset):
    """Analyze relevance score distribution."""
    from collections import Counter
    
    relevance_counts = Counter(j.relevance for j in dataset.judgments)
    total = len(dataset.judgments)
    
    print("Relevance Distribution:")
    for score in sorted(relevance_counts.keys()):
        count = relevance_counts[score]
        pct = (count / total) * 100
        
        status = "⚠" if pct > 50 or pct < 5 else "✓"
        print(f"  {status} Score {score}: {count:5d} ({pct:5.1f}%)")
    
    # Identify issue
    max_score, max_count = relevance_counts.most_common(1)[0]
    max_pct = (max_count / total) * 100
    
    if max_pct > 70:
        print(f"\n⚠ IMBALANCED: {max_pct:.1f}% of judgments are score {max_score}")
        
        if max_score == 0:
            print("  → Likely cause: Pooling only irrelevant documents")
            print("  → Solution: Pool from multiple systems, include more diverse results")
        elif max_score in [3, 4]:
            print("  → Likely cause: Pooling only top results, all highly relevant")
            print("  → Solution: Include lower-ranked results, expand pool depth")

# Example output:
# Relevance Distribution:
#   ✓ Score 0:   850 ( 34.0%)
#   ⚠ Score 1:    50 (  2.0%)  ← Too few
#   ✓ Score 2:   400 ( 16.0%)
#   ✓ Score 3:   700 ( 28.0%)
#   ✓ Score 4:   500 ( 20.0%)
```

**Solutions**:

1. **Expand Pooling Depth** (for too many 0s):
   ```python
   # Instead of top-10, pool top-30
   # Include results from rank 11-30 (more likely marginal relevance)
   pool = pooler.create_judgment_pool(query, top_k=30)
   ```

2. **Diversify Pooling Sources** (for lack of variety):
   ```python
   # Pool from systems with different behaviors
   configs = [
       {'name': 'bm25_only'},      # Conservative, high precision
       {'name': 'semantic'},        # Broader recall
       {'name': 'keyword_match'},   # Exact matches only
       {'name': 'fuzzy_match'}      # Includes typos, variants
   ]
   ```

3. **Targeted Sampling** (to fill gaps):
   ```python
   # If lacking score 1-2 (moderately relevant), specifically find those
   # Run broader query variants, pool results that partially match
   ```

4. **Re-calibrate Annotators** (for reluctance to use scale):
   ```
   - Show distribution to annotators
   - Emphasize using full scale (0-4, not just 0 and 4)
   - Provide more examples of scores 1-2 (moderate relevance)
   ```

---

### 3. Queries with Zero Relevant Documents

**Symptoms**: 10%+ of queries have no relevant documents (all judgments are 0).

**Root Causes**:
- Poor query selection (too specific, edge cases)
- Corpus doesn't contain relevant content
- Pooling depth too shallow (missed relevant docs)

**Diagnosis**:

```python
def identify_zero_result_queries(dataset, min_relevance=2):
    """Find queries with no relevant documents."""
    zero_result_queries = []
    
    for query in dataset.queries:
        judgments = dataset.get_judgments_for_query(query.id)
        relevant_count = sum(1 for j in judgments if j.relevance >= min_relevance)
        
        if relevant_count == 0:
            zero_result_queries.append({
                'query_id': query.id,
                'query_text': query.text,
                'intent': query.intent,
                'total_judgments': len(judgments)
            })
    
    print(f"Zero-Result Queries: {len(zero_result_queries)} ({len(zero_result_queries)/len(dataset.queries)*100:.1f}%)")
    
    if len(zero_result_queries) > len(dataset.queries) * 0.10:
        print("⚠ HIGH zero-result rate (>10%)")
        
        # Analyze patterns
        by_intent = defaultdict(int)
        for item in zero_result_queries:
            by_intent[item['intent']] += 1
        
        print("\nBy Intent:")
        for intent, count in by_intent.items():
            print(f"  {intent}: {count}")
    
    return zero_result_queries

# Example output:
# Zero-Result Queries: 85 (10.0%)
# ⚠ HIGH zero-result rate (>10%)
#
# By Intent:
#   transactional: 45  ← Specific documents don't exist
#   analytical: 30     ← Complex queries need multi-doc answers
```

**Solutions**:

1. **Increase Pool Depth** (may have missed relevant docs):
   ```python
   # Re-pool with top-50 instead of top-20
   pool = pooler.create_judgment_pool(query, top_k=50)
   # Judge additional 30 documents
   ```

2. **Remove Invalid Queries** (corpus doesn't contain answers):
   ```python
   # If query expects content not in corpus, remove from dataset
   # Example: "download invoice template" but corpus has no templates
   # Better to exclude than force irrelevant judgments
   ```

3. **Expand Pooling Strategies** (diversify retrieval):
   ```python
   # Add configurations that might find relevant docs:
   # - Relaxed matching (typo tolerance)
   # - Synonym expansion
   # - Conceptual search (embeddings)
   ```

4. **Accept Some Zero-Result Queries** (realistic):
   ```
   - 5-8% zero-result rate is realistic (exploratory queries exist)
   - These queries are valuable test cases (shouldn't return junk)
   - Ensure they're distributed across categories (not all one type)
   ```

---

### 4. Dataset Too Small for Statistical Significance

**Symptoms**: Evaluation metrics have high variance, can't detect meaningful improvements.

**Root Causes**:
- Too few queries (<100)
- Too few judgments per query (<5)
- Insufficient coverage of query types

**Diagnosis**:

```python
def assess_dataset_size_adequacy(dataset, target_metric='precision@5'):
    """
    Assess if dataset size is sufficient for reliable evaluation.
    
    Rule of thumb:
    - 100+ queries for aggregate metrics
    - 200+ queries for per-category breakdown
    - 500+ queries for fine-grained analysis
    """
    query_count = len(dataset.queries)
    avg_judgments = len(dataset.judgments) / query_count if query_count > 0 else 0
    
    print("Dataset Size Assessment:")
    print(f"  Queries: {query_count}")
    print(f"  Avg judgments/query: {avg_judgments:.1f}")
    
    # Assess adequacy
    if query_count < 50:
        print("\n  ✗ INSUFFICIENT: <50 queries (high variance)")
        print("    → Need 100+ queries for reliable metrics")
    elif query_count < 100:
        print("\n  ⚠ MARGINAL: 50-100 queries (moderate variance)")
        print("    → Can measure aggregate, but not per-category")
    elif query_count < 200:
        print("\n  ✓ ADEQUATE: 100-200 queries (good for aggregate)")
        print("    → Can measure overall metrics reliably")
    else:
        print("\n  ✓ EXCELLENT: 200+ queries (low variance)")
        print("    → Can measure per-category and fine-grained")
    
    if avg_judgments < 5:
        print(f"\n  ⚠ Low judgment coverage: {avg_judgments:.1f}/query")
        print("    → Recommend 10+ judgments per query")

# Example output:
# Dataset Size Assessment:
#   Queries: 75
#   Avg judgments/query: 4.2
#
#   ⚠ MARGINAL: 50-100 queries (moderate variance)
#     → Can measure aggregate, but not per-category
#
#   ⚠ Low judgment coverage: 4.2/query
#     → Recommend 10+ judgments per query
```

**Solutions**:

1. **Add More Queries** (priority: high-frequency from logs):
   ```
   - Target: 150-200 queries minimum
   - Sample stratified (head + torso + long-tail)
   - Focus on underrepresented intents/categories
   ```

2. **Increase Judgment Depth** (if queries sufficient):
   ```
   - Target: 10-15 judgments per query
   - Expand pool depth (top-10 → top-15)
   - Ensures reliable per-query metrics
   ```

3. **Statistical Power Analysis** (determine needed size):
   ```python
   # Minimum detectable improvement (MDI)
   # With 100 queries: Can detect ±5% change in P@5 (95% confidence)
   # With 200 queries: Can detect ±3.5% change
   # With 500 queries: Can detect ±2% change
   
   # Calculate: How much improvement do you need to detect?
   # Size dataset accordingly
   ```

---

### 5. Annotation Budget Exhausted

**Symptoms**: Need more judgments but out of budget.

**Solutions** (creative approaches to maximize coverage):

1. **Active Learning** (smart selection):
   ```python
   # Instead of judging all pooled documents randomly,
   # prioritize documents most likely to be relevant (model-based selection)
   # Or prioritize documents where models disagree (uncertainty sampling)
   ```

2. **Transfer Learning from Similar Datasets**:
   ```
   - If available, use judgments from related domain
   - Example: Legal dataset → Contract search dataset (partial overlap)
   - Requires validation, but saves 30-50% annotation cost
   ```

3. **Crowdsourcing** (for simpler domains):
   ```
   - Use platforms like Amazon MTurk for initial judgments
   - Have experts review/correct (cheaper than expert-only annotation)
   - Works for e-commerce, general web search (not specialized domains)
   ```

4. **Incremental Growth** (small batches over time):
   ```
   - Build dataset in phases (v1.0: 100 queries, v1.1: +50, v1.2: +50)
   - Use v1.0 immediately, grow as budget allows
   - Better to have small good dataset than no dataset
   ```

5. **Reduce Annotation Redundancy** (if IAA already good):
   ```
   - If κ ≥ 0.75, reduce from triple to double annotation
   - Saves 33% of budget while maintaining quality
   - Use saved budget to add more queries
   ```

---

## Next Steps

- **[A/B Testing Framework](./17-ab-testing-framework.md)** - Testing with datasets
- **[Evaluation Frameworks](./02-evaluation-frameworks.md)** - Using judgments
- **[Core Metrics](./01-core-metrics.md)** - Calculating relevance metrics

---

*See also: [Scoring Profiles](./14-scoring-profiles.md) | [Index Management](./15-index-management.md)*
