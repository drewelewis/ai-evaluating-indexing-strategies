# Implementation Guide: Building Search Evaluation Pipelines

This guide provides step-by-step instructions for implementing and evaluating different indexing strategies in production systems.

## 🚀 Quick Start Implementation

### Phase 1: Environment Setup (Week 1)

#### 1.1 Project Structure
```bash
search-evaluation/
├── src/
│   ├── engines/          # Search engine implementations
│   ├── evaluation/       # Metrics and evaluation frameworks
│   ├── data/            # Dataset management
│   └── utils/           # Helper functions
├── configs/             # Configuration files
├── datasets/           # Test datasets and golden sets
├── experiments/        # Experiment results and notebooks
├── tests/              # Unit and integration tests
└── requirements.txt    # Dependencies
```

#### 1.2 Core Dependencies
```python
# requirements.txt
# Search engines
elasticsearch==8.11.0
pinecone-client==2.2.4
weaviate-client==3.25.0
faiss-cpu==1.7.4

# ML and embeddings
sentence-transformers==2.2.2
transformers==4.35.0
torch==2.1.0
scikit-learn==1.3.0

# Evaluation and metrics
numpy==1.24.3
pandas==2.0.3
scipy==1.11.3

# Utilities
python-dotenv==1.0.0
tqdm==4.66.0
pyyaml==6.0.1
requests==2.31.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0
```

#### 1.3 Configuration Management
```python
# configs/base_config.yaml
evaluation:
  metrics:
    - precision_at_k: [1, 5, 10]
    - recall
    - map
    - mrr
    - ndcg_at_k: [5, 10, 20]
  
  datasets:
    training: "datasets/train_queries.json"
    validation: "datasets/val_queries.json" 
    test: "datasets/test_queries.json"

search_engines:
  elasticsearch:
    host: "localhost"
    port: 9200
    index_name: "search_index"
  
  pinecone:
    api_key: "${PINECONE_API_KEY}"
    environment: "us-west1-gcp"
    index_name: "search-vectors"
    dimension: 768

models:
  embedding_model: "sentence-transformers/all-mpnet-base-v2"
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### Phase 2: Core Implementation (Week 1-2)

#### 2.1 Base Search Engine Interface
```python
# src/engines/base_engine.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SearchResult:
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = None

class BaseSearchEngine(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_indexed = False
    
    @abstractmethod
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index a collection of documents."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for documents matching the query."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics and health metrics."""
        pass
```

#### 2.2 BM25 Keyword Engine (Elasticsearch)
```python
# src/engines/keyword_engine.py
from elasticsearch import Elasticsearch
from .base_engine import BaseSearchEngine, SearchResult

class KeywordSearchEngine(BaseSearchEngine):
    def __init__(self, config):
        super().__init__(config)
        self.es = Elasticsearch(
            hosts=[f"{config['host']}:{config['port']}"],
            timeout=30
        )
        self.index_name = config['index_name']
    
    def index_documents(self, documents):
        # Create index with optimized settings
        index_config = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "custom_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "stop",
                                "snowball"
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "custom_analyzer"
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "custom_analyzer",
                        "boost": 2.0
                    },
                    "metadata": {
                        "type": "object"
                    }
                }
            }
        }
        
        # Create or recreate index
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        
        self.es.indices.create(index=self.index_name, body=index_config)
        
        # Bulk index documents
        from elasticsearch.helpers import bulk
        
        def doc_generator():
            for doc in documents:
                yield {
                    "_index": self.index_name,
                    "_id": doc["id"],
                    "_source": {
                        "content": doc["content"],
                        "title": doc.get("title", ""),
                        "metadata": doc.get("metadata", {})
                    }
                }
        
        bulk(self.es, doc_generator(), chunk_size=1000)
        self.es.indices.refresh(index=self.index_name)
        self.is_indexed = True
    
    def search(self, query, top_k=10):
        if not self.is_indexed:
            raise ValueError("Documents must be indexed before searching")
        
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "title^2"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "size": top_k,
            "_source": ["content", "title", "metadata"]
        }
        
        response = self.es.search(index=self.index_name, body=search_body)
        
        results = []
        for hit in response["hits"]["hits"]:
            result = SearchResult(
                doc_id=hit["_id"],
                content=hit["_source"]["content"],
                score=hit["_score"],
                metadata=hit["_source"].get("metadata", {})
            )
            results.append(result)
        
        return results
    
    def get_stats(self):
        stats = self.es.indices.stats(index=self.index_name)
        return {
            "total_docs": stats["indices"][self.index_name]["total"]["docs"]["count"],
            "index_size_mb": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"] / (1024*1024)
        }
```

#### 2.3 Vector Search Engine (FAISS/Pinecone)
```python
# src/engines/vector_engine.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .base_engine import BaseSearchEngine, SearchResult

class VectorSearchEngine(BaseSearchEngine):
    def __init__(self, config):
        super().__init__(config)
        self.model = SentenceTransformer(config['embedding_model'])
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        self.doc_embeddings = None
    
    def index_documents(self, documents):
        self.documents = documents
        
        # Extract text content for embedding
        texts = [doc["content"] for doc in documents]
        
        # Generate embeddings
        print("Generating embeddings...")
        self.doc_embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings)
        
        self.is_indexed = True
        print(f"Indexed {len(documents)} documents")
    
    def search(self, query, top_k=10):
        if not self.is_indexed:
            raise ValueError("Documents must be indexed before searching")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Convert to SearchResult objects
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                doc = self.documents[idx]
                result = SearchResult(
                    doc_id=doc["id"],
                    content=doc["content"],
                    score=float(score),
                    metadata=doc.get("metadata", {})
                )
                results.append(result)
        
        return results
    
    def get_stats(self):
        if self.index is None:
            return {"total_docs": 0, "dimension": self.dimension}
        
        return {
            "total_docs": self.index.ntotal,
            "dimension": self.dimension,
            "model_name": self.model._modules['0'].auto_model.name_or_path
        }
```

#### 2.4 Hybrid Search Engine
```python
# src/engines/hybrid_engine.py
from .base_engine import BaseSearchEngine, SearchResult
from typing import List
import numpy as np

class HybridSearchEngine(BaseSearchEngine):
    def __init__(self, config):
        super().__init__(config)
        self.keyword_engine = None
        self.vector_engine = None
        self.alpha = config.get('alpha', 0.5)  # Weight for keyword vs vector
        self.fusion_method = config.get('fusion_method', 'rrf')  # 'rrf' or 'weighted'
    
    def set_engines(self, keyword_engine, vector_engine):
        self.keyword_engine = keyword_engine
        self.vector_engine = vector_engine
        self.is_indexed = keyword_engine.is_indexed and vector_engine.is_indexed
    
    def index_documents(self, documents):
        if self.keyword_engine and self.vector_engine:
            self.keyword_engine.index_documents(documents)
            self.vector_engine.index_documents(documents)
            self.is_indexed = True
    
    def search(self, query, top_k=10):
        if not self.is_indexed:
            raise ValueError("Engines must be set and indexed before searching")
        
        # Get results from both engines
        keyword_results = self.keyword_engine.search(query, top_k=top_k*2)
        vector_results = self.vector_engine.search(query, top_k=top_k*2)
        
        # Fuse results
        if self.fusion_method == 'rrf':
            fused_results = self._reciprocal_rank_fusion(
                keyword_results, vector_results, k=60
            )
        else:  # weighted fusion
            fused_results = self._weighted_score_fusion(
                keyword_results, vector_results, self.alpha
            )
        
        return fused_results[:top_k]
    
    def _reciprocal_rank_fusion(self, list1, list2, k=60):
        # Create score dictionary
        scores = {}
        
        # Process first list
        for rank, result in enumerate(list1):
            scores[result.doc_id] = 1.0 / (rank + k)
        
        # Process second list
        for rank, result in enumerate(list2):
            if result.doc_id in scores:
                scores[result.doc_id] += 1.0 / (rank + k)
            else:
                scores[result.doc_id] = 1.0 / (rank + k)
        
        # Sort by fused score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Reconstruct SearchResult objects
        doc_dict = {}
        for result in list1 + list2:
            if result.doc_id not in doc_dict:
                doc_dict[result.doc_id] = result
        
        fused_results = []
        for doc_id, score in sorted_docs:
            if doc_id in doc_dict:
                result = doc_dict[doc_id]
                result.score = score
                fused_results.append(result)
        
        return fused_results
    
    def _weighted_score_fusion(self, keyword_results, vector_results, alpha):
        # Normalize scores to [0,1] range
        def normalize_scores(results):
            if not results:
                return {}
            
            scores = [r.score for r in results]
            min_score, max_score = min(scores), max(scores)
            
            if max_score == min_score:
                return {r.doc_id: 0.5 for r in results}
            
            return {
                r.doc_id: (r.score - min_score) / (max_score - min_score) 
                for r in results
            }
        
        kw_scores = normalize_scores(keyword_results)
        vec_scores = normalize_scores(vector_results)
        
        # Combine scores
        all_docs = set(kw_scores.keys()) | set(vec_scores.keys())
        fused_scores = {}
        
        for doc_id in all_docs:
            kw_score = kw_scores.get(doc_id, 0)
            vec_score = vec_scores.get(doc_id, 0)
            fused_scores[doc_id] = alpha * kw_score + (1 - alpha) * vec_score
        
        # Sort and reconstruct results
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create document lookup
        doc_dict = {}
        for result in keyword_results + vector_results:
            if result.doc_id not in doc_dict:
                doc_dict[result.doc_id] = result
        
        fused_results = []
        for doc_id, score in sorted_docs:
            if doc_id in doc_dict:
                result = doc_dict[doc_id]
                result.score = score
                fused_results.append(result)
        
        return fused_results
    
    def get_stats(self):
        stats = {}
        if self.keyword_engine:
            stats['keyword'] = self.keyword_engine.get_stats()
        if self.vector_engine:
            stats['vector'] = self.vector_engine.get_stats()
        stats['fusion_method'] = self.fusion_method
        stats['alpha'] = self.alpha
        return stats
```

### Phase 3: Evaluation Framework (Week 2)

#### 3.1 Metrics Implementation
```python
# src/evaluation/metrics.py
import numpy as np
from typing import List, Dict, Set
from dataclasses import dataclass

@dataclass
class QueryResult:
    query_id: str
    retrieved_docs: List[str]
    relevant_docs: Set[str]
    scores: List[float] = None

class EvaluationMetrics:
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """Calculate Recall@K."""
        if not relevant_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_in_top_k / len(relevant_docs)
    
    @staticmethod
    def average_precision(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """Calculate Average Precision (AP)."""
        if not relevant_docs:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs) if relevant_docs else 0.0
    
    @staticmethod
    def reciprocal_rank(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """Calculate Reciprocal Rank (RR)."""
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def dcg_at_k(retrieved_docs: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """Calculate Discounted Cumulative Gain at K."""
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            relevance = relevance_scores.get(doc, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """Calculate Normalized DCG at K."""
        dcg = EvaluationMetrics.dcg_at_k(retrieved_docs, relevance_scores, k)
        
        # Calculate IDCG (Ideal DCG)
        ideal_order = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        ideal_docs = [doc for doc, _ in ideal_order[:k]]
        idcg = EvaluationMetrics.dcg_at_k(ideal_docs, relevance_scores, k)
        
        return dcg / idcg if idcg > 0 else 0.0

class MetricsCalculator:
    def __init__(self, metrics_config: List[str]):
        self.metrics_config = metrics_config
    
    def evaluate_query(self, query_result: QueryResult) -> Dict[str, float]:
        """Evaluate a single query result."""
        metrics = {}
        
        retrieved = query_result.retrieved_docs
        relevant = query_result.relevant_docs
        
        # Precision@K and Recall@K
        for metric in self.metrics_config:
            if metric.startswith('precision_at_'):
                k = int(metric.split('_')[-1])
                metrics[f'precision_at_{k}'] = EvaluationMetrics.precision_at_k(retrieved, relevant, k)
            elif metric.startswith('recall_at_'):
                k = int(metric.split('_')[-1])
                metrics[f'recall_at_{k}'] = EvaluationMetrics.recall_at_k(retrieved, relevant, k)
            elif metric == 'map':
                metrics['map'] = EvaluationMetrics.average_precision(retrieved, relevant)
            elif metric == 'mrr':
                metrics['mrr'] = EvaluationMetrics.reciprocal_rank(retrieved, relevant)
        
        return metrics
    
    def evaluate_queries(self, query_results: List[QueryResult]) -> Dict[str, float]:
        """Evaluate multiple queries and return averaged metrics."""
        all_metrics = []
        
        for query_result in query_results:
            query_metrics = self.evaluate_query(query_result)
            all_metrics.append(query_metrics)
        
        # Average metrics across all queries
        if not all_metrics:
            return {}
        
        averaged_metrics = {}
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            averaged_metrics[metric_name] = np.mean(values) if values else 0.0
        
        return averaged_metrics
```

#### 3.2 Evaluation Pipeline
```python
# src/evaluation/evaluator.py
import json
import time
from typing import Dict, List, Any
from ..engines.base_engine import BaseSearchEngine
from .metrics import MetricsCalculator, QueryResult

class SearchEvaluator:
    def __init__(self, metrics_config: List[str]):
        self.metrics_calculator = MetricsCalculator(metrics_config)
        self.results_history = []
    
    def load_golden_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load golden dataset with queries and relevance judgments."""
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        return dataset
    
    def evaluate_engine(self, 
                       engine: BaseSearchEngine, 
                       dataset: List[Dict[str, Any]], 
                       top_k: int = 100) -> Dict[str, Any]:
        """Evaluate a search engine on a dataset."""
        
        query_results = []
        latencies = []
        
        print(f"Evaluating {len(dataset)} queries...")
        
        for i, item in enumerate(dataset):
            query = item['query']
            relevant_docs = set(item['relevant_docs'])
            
            # Measure search latency
            start_time = time.time()
            search_results = engine.search(query, top_k=top_k)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            # Extract retrieved document IDs
            retrieved_docs = [result.doc_id for result in search_results]
            
            query_result = QueryResult(
                query_id=item.get('query_id', str(i)),
                retrieved_docs=retrieved_docs,
                relevant_docs=relevant_docs
            )
            query_results.append(query_result)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(dataset)} queries")
        
        # Calculate metrics
        metrics = self.metrics_calculator.evaluate_queries(query_results)
        
        # Add performance metrics
        metrics['avg_latency_ms'] = np.mean(latencies)
        metrics['p95_latency_ms'] = np.percentile(latencies, 95)
        metrics['p99_latency_ms'] = np.percentile(latencies, 99)
        
        # Add engine stats
        engine_stats = engine.get_stats()
        
        evaluation_result = {
            'metrics': metrics,
            'engine_stats': engine_stats,
            'query_count': len(dataset),
            'timestamp': time.time()
        }
        
        self.results_history.append(evaluation_result)
        
        return evaluation_result
    
    def compare_engines(self, 
                       engines: Dict[str, BaseSearchEngine], 
                       dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple engines on the same dataset."""
        
        comparison_results = {}
        
        for engine_name, engine in engines.items():
            print(f"\nEvaluating {engine_name}...")
            result = self.evaluate_engine(engine, dataset)
            comparison_results[engine_name] = result
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(comparison_results)
        
        return {
            'individual_results': comparison_results,
            'summary': summary
        }
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary comparing different engines."""
        summary = {}
        
        # Extract key metrics for comparison
        key_metrics = ['precision_at_5', 'recall_at_10', 'map', 'mrr', 'avg_latency_ms']
        
        for metric in key_metrics:
            metric_values = {}
            for engine_name, result in results.items():
                if metric in result['metrics']:
                    metric_values[engine_name] = result['metrics'][metric]
            
            if metric_values:
                best_engine = max(metric_values.items(), key=lambda x: x[1])
                summary[metric] = {
                    'values': metric_values,
                    'best': best_engine[0],
                    'best_value': best_engine[1]
                }
        
        return summary
    
    def export_results(self, results: Dict[str, Any], output_path: str):
        """Export evaluation results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results exported to {output_path}")
```

### Phase 4: Example Usage and Testing (Week 2)

#### 4.1 Complete Example Script
```python
# examples/complete_evaluation.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.engines.keyword_engine import KeywordSearchEngine
from src.engines.vector_engine import VectorSearchEngine
from src.engines.hybrid_engine import HybridSearchEngine
from src.evaluation.evaluator import SearchEvaluator
import yaml

def load_config():
    with open('configs/base_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_sample_documents():
    """Load sample documents for testing."""
    return [
        {
            "id": "doc_1",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "title": "Introduction to Machine Learning",
            "metadata": {"category": "AI", "author": "John Doe"}
        },
        {
            "id": "doc_2", 
            "content": "Deep learning uses neural networks with multiple layers to model complex patterns.",
            "title": "Deep Learning Fundamentals",
            "metadata": {"category": "AI", "author": "Jane Smith"}
        },
        {
            "id": "doc_3",
            "content": "Natural language processing enables computers to understand human language.",
            "title": "NLP Basics",
            "metadata": {"category": "AI", "author": "Bob Johnson"}
        },
        # Add more documents...
    ]

def create_sample_queries():
    """Create sample queries with relevance judgments."""
    return [
        {
            "query_id": "q1",
            "query": "machine learning algorithms",
            "relevant_docs": ["doc_1", "doc_2"]
        },
        {
            "query_id": "q2", 
            "query": "neural networks",
            "relevant_docs": ["doc_2"]
        },
        {
            "query_id": "q3",
            "query": "understanding human language",
            "relevant_docs": ["doc_3"]
        }
    ]

def main():
    # Load configuration
    config = load_config()
    
    # Load data
    documents = load_sample_documents()
    queries = create_sample_queries()
    
    # Initialize engines
    keyword_engine = KeywordSearchEngine(config['search_engines']['elasticsearch'])
    vector_engine = VectorSearchEngine({
        'embedding_model': config['models']['embedding_model']
    })
    hybrid_engine = HybridSearchEngine({
        'alpha': 0.6,
        'fusion_method': 'rrf'
    })
    
    # Index documents
    print("Indexing documents...")
    keyword_engine.index_documents(documents)
    vector_engine.index_documents(documents)
    
    # Set up hybrid engine
    hybrid_engine.set_engines(keyword_engine, vector_engine)
    
    # Initialize evaluator
    evaluator = SearchEvaluator(['precision_at_5', 'recall_at_10', 'map', 'mrr'])
    
    # Compare engines
    engines = {
        'keyword': keyword_engine,
        'vector': vector_engine,
        'hybrid': hybrid_engine
    }
    
    print("\nStarting evaluation...")
    results = evaluator.compare_engines(engines, queries)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for engine_name, result in results['individual_results'].items():
        print(f"\n{engine_name.upper()} ENGINE:")
        metrics = result['metrics']
        print(f"  Precision@5: {metrics.get('precision_at_5', 0):.3f}")
        print(f"  Recall@10:   {metrics.get('recall_at_10', 0):.3f}")
        print(f"  MAP:         {metrics.get('map', 0):.3f}")
        print(f"  MRR:         {metrics.get('mrr', 0):.3f}")
        print(f"  Avg Latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
    
    # Export results
    evaluator.export_results(results, 'experiments/evaluation_results.json')

if __name__ == "__main__":
    main()
```

#### 4.2 A/B Testing Implementation
```python
# src/evaluation/ab_testing.py
import random
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

@dataclass
class ABTestConfig:
    test_name: str
    hypothesis: str
    success_metrics: List[str]
    guardrail_metrics: List[str]
    traffic_split: Dict[str, float]
    duration_days: int
    minimum_sample_size: int
    significance_level: float = 0.05

@dataclass
class UserInteraction:
    user_id: str
    session_id: str
    query: str
    results_shown: int
    clicks: List[int]
    dwell_times: Dict[int, float]
    timestamp: datetime
    experiment_group: str

class ABTestFramework:
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.interactions = []
        self.start_time = datetime.now()
        
    def assign_user_to_group(self, user_id: str) -> str:
        """Consistently assign user to experiment group."""
        hash_input = f"{user_id}_{self.config.test_name}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Use cumulative probability for assignment
        rand_value = (hash_value % 10000) / 10000.0
        cumulative_prob = 0.0
        
        for group, probability in self.config.traffic_split.items():
            cumulative_prob += probability
            if rand_value <= cumulative_prob:
                return group
        
        # Fallback to control group
        return "control"
    
    def log_interaction(self, interaction: UserInteraction):
        """Log user interaction for analysis."""
        interaction.experiment_group = self.assign_user_to_group(interaction.user_id)
        self.interactions.append(interaction)
    
    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each experiment group."""
        if not self.interactions:
            return {}
        
        # Group interactions by experiment group
        grouped_interactions = {}
        for interaction in self.interactions:
            group = interaction.experiment_group
            if group not in grouped_interactions:
                grouped_interactions[group] = []
            grouped_interactions[group].append(interaction)
        
        # Calculate metrics for each group
        results = {}
        for group, interactions in grouped_interactions.items():
            metrics = {}
            
            # Click-through rate
            total_sessions = len(interactions)
            sessions_with_clicks = sum(1 for i in interactions if len(i.clicks) > 0)
            metrics['ctr'] = sessions_with_clicks / total_sessions if total_sessions > 0 else 0
            
            # Mean Reciprocal Rank
            mrr_scores = []
            for interaction in interactions:
                if interaction.clicks:
                    first_click_position = min(interaction.clicks)
                    mrr_scores.append(1.0 / (first_click_position + 1))
                else:
                    mrr_scores.append(0.0)
            metrics['mrr'] = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
            
            # Average dwell time
            all_dwell_times = []
            for interaction in interactions:
                all_dwell_times.extend(interaction.dwell_times.values())
            metrics['avg_dwell_time'] = sum(all_dwell_times) / len(all_dwell_times) if all_dwell_times else 0
            
            # Session success rate (at least one click with >30s dwell time)
            successful_sessions = 0
            for interaction in interactions:
                if any(dwell_time > 30 for dwell_time in interaction.dwell_times.values()):
                    successful_sessions += 1
            metrics['session_success_rate'] = successful_sessions / total_sessions if total_sessions > 0 else 0
            
            results[group] = metrics
        
        return results
    
    def statistical_significance_test(self, metric_name: str) -> Dict[str, Any]:
        """Perform statistical significance test for a metric."""
        from scipy import stats
        
        metrics = self.calculate_metrics()
        if len(metrics) < 2:
            return {"error": "Need at least 2 groups for comparison"}
        
        groups = list(metrics.keys())
        if len(groups) != 2:
            return {"error": "Currently supports only 2-group comparison"}
        
        group1, group2 = groups
        
        # Extract metric values for each group
        group1_interactions = [i for i in self.interactions if i.experiment_group == group1]
        group2_interactions = [i for i in self.interactions if i.experiment_group == group2]
        
        if metric_name == 'ctr':
            # Two-proportion z-test for CTR
            n1, n2 = len(group1_interactions), len(group2_interactions)
            x1 = sum(1 for i in group1_interactions if len(i.clicks) > 0)
            x2 = sum(1 for i in group2_interactions if len(i.clicks) > 0)
            
            # Calculate proportions
            p1, p2 = x1/n1 if n1 > 0 else 0, x2/n2 if n2 > 0 else 0
            
            # Pooled proportion
            p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0
            
            # Standard error
            se = (p_pool * (1 - p_pool) * (1/n1 + 1/n2)) ** 0.5 if (n1 > 0 and n2 > 0) else 0
            
            # Z-statistic
            z_stat = (p1 - p2) / se if se > 0 else 0
            
            # P-value (two-tailed)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return {
                "metric": metric_name,
                "group1_value": p1,
                "group2_value": p2,
                "z_statistic": z_stat,
                "p_value": p_value,
                "significant": p_value < self.config.significance_level,
                "sample_sizes": {"group1": n1, "group2": n2}
            }
        
        else:
            # T-test for continuous metrics
            return {"error": f"Statistical test not implemented for {metric_name}"}
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive A/B test report."""
        metrics = self.calculate_metrics()
        
        report = {
            "test_config": asdict(self.config),
            "test_duration": (datetime.now() - self.start_time).days,
            "total_interactions": len(self.interactions),
            "group_metrics": metrics,
            "significance_tests": {}
        }
        
        # Run significance tests for success metrics
        for metric in self.config.success_metrics:
            if metric in ['ctr']:  # Add more as implemented
                test_result = self.statistical_significance_test(metric)
                report["significance_tests"][metric] = test_result
        
        return report
    
    def export_interactions(self, filepath: str):
        """Export interaction data for further analysis."""
        interaction_dicts = []
        for interaction in self.interactions:
            interaction_dict = asdict(interaction)
            interaction_dict['timestamp'] = interaction.timestamp.isoformat()
            interaction_dicts.append(interaction_dict)
        
        with open(filepath, 'w') as f:
            json.dump(interaction_dicts, f, indent=2)

# Example usage
def run_ab_test_example():
    # Configure A/B test
    config = ABTestConfig(
        test_name="vector_vs_keyword_search",
        hypothesis="Vector search improves user engagement",
        success_metrics=["ctr", "session_success_rate"],
        guardrail_metrics=["avg_latency"],
        traffic_split={"control": 0.5, "treatment": 0.5},
        duration_days=14,
        minimum_sample_size=1000,
        significance_level=0.05
    )
    
    # Initialize test framework
    ab_test = ABTestFramework(config)
    
    # Simulate some interactions
    for i in range(100):
        interaction = UserInteraction(
            user_id=f"user_{i}",
            session_id=f"session_{i}",
            query="machine learning",
            results_shown=10,
            clicks=[0, 2] if i % 3 == 0 else [],
            dwell_times={0: 45.0, 2: 30.0} if i % 3 == 0 else {},
            timestamp=datetime.now(),
            experiment_group=""  # Will be assigned automatically
        )
        ab_test.log_interaction(interaction)
    
    # Generate report
    report = ab_test.generate_report()
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    run_ab_test_example()
```

## 🚀 Next Steps

### Week 3-4: Advanced Features
1. **Cross-encoder re-ranking** implementation
2. **Query expansion** with LLMs
3. **Real-time monitoring** dashboard
4. **Automated hyperparameter** tuning

### Week 5-8: Production Deployment
1. **Containerization** with Docker
2. **API endpoints** for search services
3. **Load testing** and optimization
4. **CI/CD pipeline** setup

### Week 9-12: Continuous Improvement
1. **Feedback loop** implementation
2. **Online learning** integration
3. **Advanced analytics** and reporting
4. **Cost optimization** strategies

---
*Next: [Azure AI Search Evaluation](./azure-ai-search-evaluation.md)*