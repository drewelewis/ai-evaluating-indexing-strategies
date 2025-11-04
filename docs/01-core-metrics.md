# Core Metrics for Search Relevance and Accuracy

This document outlines the essential metrics for quantifying how well search results align with user intent across different indexing strategies.

## 📊 Primary Relevance Metrics

### 1. Precision & Recall
The foundational metrics for search quality assessment.

#### Precision
```
Precision = Relevant results retrieved ÷ Total results retrieved
```
- **Purpose**: Measures accuracy by focusing on reducing irrelevant results
- **Range**: 0.0 to 1.0 (higher is better)
- **When to use**: When false positives are costly
- **Target**: > 0.7 for most applications

#### Recall
```
Recall = Relevant results retrieved ÷ Total relevant results in corpus
```
- **Purpose**: Measures completeness of retrieval
- **Range**: 0.0 to 1.0 (higher is better)
- **When to use**: When missing relevant results is costly
- **Target**: > 0.8 for comprehensive search scenarios

#### F1-Score
```
F1 = 2 × (Precision × Recall) ÷ (Precision + Recall)
```
- **Purpose**: Harmonic mean balancing precision and recall
- **Best for**: Overall system performance assessment

### 2. Precision@K
Evaluates ranking quality by measuring relevant results in top K positions.

```
Precision@K = Relevant results in top K ÷ K
```

**Common K values and use cases:**
- **Precision@1**: Critical for single-answer queries (Q&A, navigation)
- **Precision@5**: Standard for search result pages
- **Precision@10**: Full first page evaluation
- **Precision@20**: Extended result evaluation

**Target values:**
- Precision@1: > 0.8 for Q&A systems
- Precision@5: > 0.7 for general search
- Precision@10: > 0.6 for exploratory search

### 3. Mean Average Precision (MAP)
Considers ranking order of relevant results across multiple queries.

```
MAP = (1/Q) × Σ(Average Precision for each query)
```

Where Average Precision for a single query:
```
AP = (1/R) × Σ(Precision@k × rel(k))
```
- **R**: Total relevant documents for the query
- **rel(k)**: 1 if item at rank k is relevant, 0 otherwise

**Characteristics:**
- **Range**: 0.0 to 1.0
- **Strengths**: Rewards systems that rank relevant results higher
- **Target**: > 0.6 for most applications
- **Best for**: Comprehensive ranking evaluation

### 4. Mean Reciprocal Rank (MRR)
Rewards systems that return the first relevant result quickly.

```
MRR = (1/Q) × Σ(1/rank of first relevant result)
```

**Use cases:**
- Q&A systems where first result matters most
- Navigational queries (finding specific pages)
- Auto-complete suggestions
- **Target**: > 0.6 for Q&A, > 0.8 for navigation

### 5. Normalized Discounted Cumulative Gain (nDCG)
Accounts for graded relevance and position weighting.

```
DCG@k = Σ(rel_i / log2(i + 1)) for i = 1 to k
nDCG@k = DCG@k / IDCG@k
```

Where:
- **rel_i**: Relevance score at position i (typically 0-3 scale)
- **IDCG@k**: Ideal DCG (perfect ranking of same items)

**Relevance scale example:**
- **3**: Highly relevant (perfect match)
- **2**: Relevant (good match)
- **1**: Somewhat relevant (partial match)
- **0**: Not relevant

**Advantages:**
- Handles graded relevance (not just binary)
- Position-aware (higher positions weighted more)
- Normalized for cross-query comparison
- **Target**: > 0.8 for high-quality systems

## 🎯 Behavioral Metrics

### Click-through Rate (CTR)
```
CTR = Clicks on search results ÷ Total search sessions
```
- **Purpose**: Real-world relevance indicator
- **Target**: Baseline + 15% improvement
- **Best for**: A/B testing different indexing strategies

### Conversion Rate
```
Conversion Rate = Successful actions ÷ Total search sessions
```
- **Actions**: Purchases, downloads, form submissions
- **Purpose**: Business impact measurement
- **Best for**: E-commerce and goal-oriented searches

### Dwell Time
Average time spent on clicked results.
- **Higher dwell time**: Indicates content relevance
- **Lower dwell time**: May suggest poor result quality
- **Threshold**: > 30 seconds for content consumption

## 📈 Advanced Metrics

### Success@K
```
Success@K = Queries with at least one relevant result in top K ÷ Total queries
```
- **Purpose**: Query satisfaction rate
- **Target**: > 0.9 for Success@10

### Expected Reciprocal Rank (ERR)
Probabilistic metric assuming users may stop at any relevant result:
```
ERR = Σ(1/r × P(user stops at position r))
```
- **Best for**: Modeling realistic user behavior
- **Range**: 0.0 to 1.0

## 🔧 Implementation Considerations

### Metric Selection Guidelines

| Scenario | Primary Metrics | Secondary Metrics |
|----------|----------------|-------------------|
| **Q&A Systems** | MRR, Precision@1 | nDCG@5, Success@1 |
| **E-commerce Search** | nDCG@10, CTR | Precision@5, Conversion Rate |
| **Document Retrieval** | MAP, Recall | nDCG@20, Precision@10 |
| **RAG Systems** | Precision@5, MRR | nDCG@10, Success@5 |
| **Navigation** | MRR, Success@1 | Precision@1, CTR |

### Evaluation Setup
1. **Ground Truth**: Create relevance judgments (explicit or implicit)
2. **Query Sets**: Include diverse query types (navigational, informational, transactional)
3. **Test Splits**: Use temporal splits to avoid data leakage
4. **Statistical Significance**: Use paired t-tests or bootstrap methods

### Common Pitfalls
- **Evaluation bias**: Using training queries for testing
- **Limited query diversity**: Not representing real user queries
- **Binary relevance only**: Missing nuanced relevance grades
- **Position bias**: Not accounting for user browsing patterns
- **Corpus changes**: Evaluating on outdated document sets

## 📊 Metric Reporting Template

```python
# Example evaluation report structure
evaluation_report = {
    "dataset": "product_search_2024",
    "queries": 1000,
    "indexing_strategy": "hybrid_bm25_vector",
    "metrics": {
        "precision_at_1": 0.82,
        "precision_at_5": 0.74,
        "precision_at_10": 0.68,
        "recall": 0.85,
        "map": 0.67,
        "mrr": 0.79,
        "ndcg_at_10": 0.83,
        "success_at_10": 0.92
    },
    "behavioral_metrics": {
        "ctr": 0.45,
        "avg_dwell_time": 45.2,
        "conversion_rate": 0.12
    }
}
```

---
*Next: [Evaluation Frameworks](./02-evaluation-frameworks.md)*