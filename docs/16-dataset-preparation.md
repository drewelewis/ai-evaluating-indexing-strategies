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

### ✅ Do's
1. **Start with 100+ queries** covering diverse intents
2. **Include long-tail queries** (not just popular ones)
3. **Use multiple annotators** (2-3 minimum) for reliability
4. **Calculate inter-annotator agreement** (target Kappa > 0.6)
5. **Version control datasets** with clear metadata
6. **Validate before using** (completeness, coverage, distribution)
7. **Document judgment guidelines** clearly
8. **Sample diverse documents** for judgment tasks
9. **Track dataset statistics** over time
10. **Refresh datasets** periodically (6-12 months)

### ❌ Don'ts
1. **Don't** use only synthetic queries
2. **Don't** skip validation steps
3. **Don't** rely on single annotator
4. **Don't** ignore inter-annotator disagreements
5. **Don't** create imbalanced judgments (all 4s or all 0s)
6. **Don't** use outdated data
7. **Don't** forget to version datasets
8. **Don't** judge only top-1 results (judge top-10 or top-20)
9. **Don't** mix judgment scales across versions
10. **Don't** skip documentation

---

## Next Steps

- **[A/B Testing Framework](./17-ab-testing-framework.md)** - Testing with datasets
- **[Evaluation Frameworks](./02-evaluation-frameworks.md)** - Using judgments
- **[Core Metrics](./01-core-metrics.md)** - Calculating relevance metrics

---

*See also: [Scoring Profiles](./14-scoring-profiles.md) | [Index Management](./15-index-management.md)*
