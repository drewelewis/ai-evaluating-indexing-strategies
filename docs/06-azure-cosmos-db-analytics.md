# Azure Cosmos DB for Analytics

Complete guide to using Azure Cosmos DB for storing search evaluation results, query analytics, and performance metrics.

## 📋 Table of Contents
- [Overview](#overview)
- [Setup and Configuration](#setup-and-configuration)
- [Schema Design](#schema-design)
- [Data Storage Patterns](#data-storage-patterns)
- [Querying and Analysis](#querying-and-analysis)
- [Performance Optimization](#performance-optimization)
- [Cost Management](#cost-management)
- [Integration with Search](#integration-with-search)

---

## Overview

### Why Cosmos DB for Search Analytics?

Azure Cosmos DB is ideal for search evaluation and analytics because it provides:
- **Low-latency writes**: Sub-10ms writes for real-time query logging
- **Global distribution**: Multi-region analytics and reporting
- **Flexible schema**: Easily evolve evaluation metrics over time
- **Rich querying**: SQL API for complex analytics queries
- **Change feed**: Real-time processing of search events
- **Time-to-live (TTL)**: Automatic cleanup of old logs

### Architecture Pattern

```
┌──────────────────┐
│  Search Queries  │
└────────┬─────────┘
         │
         ▼
┌────────────────────────┐
│  Azure AI Search       │
└────────┬───────────────┘
         │ Log query & results
         ▼
┌────────────────────────┐
│  Azure Cosmos DB       │
│  - Query logs          │
│  - Evaluation results  │
│  - Performance metrics │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Analytics Dashboard   │
│  (Power BI / Custom)   │
└────────────────────────┘
```

---

## Setup and Configuration

### Step 1: Create Cosmos DB Account

#### Using Azure CLI

```bash
# Variables
RESOURCE_GROUP="rg-search-evaluation"
LOCATION="eastus"
ACCOUNT_NAME="cosmos-search-analytics"
DATABASE_NAME="search_evaluation"

# Create Cosmos DB account (NoSQL API)
az cosmosdb create \
  --name $ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --locations regionName=$LOCATION failoverPriority=0 \
  --default-consistency-level "Session" \
  --enable-automatic-failover false \
  --enable-analytical-storage true

# Create database
az cosmosdb sql database create \
  --account-name $ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --name $DATABASE_NAME \
  --throughput 400

# Get connection string
CONNECTION_STRING=$(az cosmosdb keys list \
  --name $ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --type connection-strings \
  --query "connectionStrings[0].connectionString" \
  --output tsv)

echo "Connection String: $CONNECTION_STRING"
```

#### Using Python SDK

```python
from azure.cosmos import CosmosClient, PartitionKey, exceptions
import os

class CosmosDBSetup:
    """Setup Azure Cosmos DB for search analytics."""
    
    def __init__(self, endpoint, key):
        self.client = CosmosClient(endpoint, key)
        self.database_name = "search_evaluation"
        self.database = None
    
    def create_database(self, throughput=400):
        """Create database with specified throughput."""
        try:
            self.database = self.client.create_database(
                id=self.database_name,
                offer_throughput=throughput
            )
            print(f"✅ Database '{self.database_name}' created")
        except exceptions.CosmosResourceExistsError:
            self.database = self.client.get_database_client(self.database_name)
            print(f"ℹ️ Database '{self.database_name}' already exists")
        
        return self.database
    
    def create_containers(self):
        """Create containers for different data types."""
        containers_config = [
            {
                'id': 'query_logs',
                'partition_key': '/search_service',
                'default_ttl': 2592000  # 30 days
            },
            {
                'id': 'evaluation_results',
                'partition_key': '/experiment_id',
                'default_ttl': -1  # Never expire
            },
            {
                'id': 'performance_metrics',
                'partition_key': '/metric_type',
                'default_ttl': 7776000  # 90 days
            },
            {
                'id': 'ab_test_results',
                'partition_key': '/test_id',
                'default_ttl': -1
            }
        ]
        
        for config in containers_config:
            try:
                container = self.database.create_container(
                    id=config['id'],
                    partition_key=PartitionKey(path=config['partition_key']),
                    default_ttl=config['default_ttl']
                )
                print(f"✅ Container '{config['id']}' created")
            except exceptions.CosmosResourceExistsError:
                print(f"ℹ️ Container '{config['id']}' already exists")

# Usage
setup = CosmosDBSetup(
    endpoint=os.getenv("COSMOS_ENDPOINT"),
    key=os.getenv("COSMOS_KEY")
)

database = setup.create_database(throughput=400)
setup.create_containers()
```

---

## Schema Design

### Query Log Schema

```python
from datetime import datetime
from typing import List, Dict, Any

class QueryLogSchema:
    """Schema for search query logs."""
    
    @staticmethod
    def create_query_log(
        query_text: str,
        search_service: str,
        index_name: str,
        results: List[Dict],
        user_id: str = None,
        session_id: str = None,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        Create a query log document.
        
        Returns document matching this schema:
        {
            "id": "unique-guid",
            "search_service": "search-prod",  # Partition key
            "index_name": "products",
            "query_text": "laptop computers",
            "query_type": "simple|full|semantic",
            "timestamp": "2024-01-15T10:30:00Z",
            "user_id": "user123",
            "session_id": "session456",
            "results": {
                "count": 10,
                "top_results": [...],
                "search_score_range": {"min": 0.1, "max": 0.9}
            },
            "performance": {
                "duration_ms": 45,
                "ru_charge": 2.5
            },
            "metadata": {...}
        }
        """
        import uuid
        
        # Calculate result statistics
        scores = [r.get('search_score', 0) for r in results]
        
        return {
            'id': str(uuid.uuid4()),
            'search_service': search_service,
            'index_name': index_name,
            'query_text': query_text,
            'query_type': metadata.get('query_type', 'simple') if metadata else 'simple',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'user_id': user_id,
            'session_id': session_id,
            'results': {
                'count': len(results),
                'top_results': results[:10],  # Store top 10
                'search_score_range': {
                    'min': min(scores) if scores else 0,
                    'max': max(scores) if scores else 0
                }
            },
            'performance': {
                'duration_ms': metadata.get('duration_ms', 0) if metadata else 0,
                'ru_charge': 0  # Will be populated by Cosmos DB
            },
            'metadata': metadata or {}
        }

# Usage
log_doc = QueryLogSchema.create_query_log(
    query_text="laptop computers",
    search_service="search-prod",
    index_name="products",
    results=[
        {'id': '1', 'search_score': 0.85},
        {'id': '2', 'search_score': 0.72}
    ],
    user_id="user123",
    session_id="session456",
    metadata={'query_type': 'semantic', 'duration_ms': 45}
)
```

### Evaluation Result Schema

```python
class EvaluationResultSchema:
    """Schema for evaluation experiment results."""
    
    @staticmethod
    def create_evaluation_result(
        experiment_id: str,
        index_config: Dict,
        metrics: Dict[str, float],
        test_queries: List[Dict],
        summary: str = None
    ) -> Dict[str, Any]:
        """
        Create an evaluation result document.
        
        Schema:
        {
            "id": "eval-2024-01-15-001",
            "experiment_id": "exp-hybrid-vs-fulltext",  # Partition key
            "experiment_name": "Hybrid Search Evaluation",
            "timestamp": "2024-01-15T10:00:00Z",
            "index_config": {
                "search_type": "hybrid",
                "vector_weight": 0.7,
                "text_weight": 0.3
            },
            "metrics": {
                "precision_at_10": 0.85,
                "recall_at_10": 0.72,
                "ndcg_at_10": 0.78,
                "map": 0.76
            },
            "query_results": [
                {
                    "query": "laptop",
                    "expected_results": ["doc1", "doc2"],
                    "actual_results": ["doc1", "doc2", "doc3"],
                    "metrics": {...}
                }
            ],
            "summary": "Hybrid search outperformed full-text by 15%"
        }
        """
        import uuid
        
        return {
            'id': f"eval-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}",
            'experiment_id': experiment_id,
            'experiment_name': index_config.get('name', experiment_id),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'index_config': index_config,
            'metrics': metrics,
            'query_results': test_queries,
            'summary': summary,
            'status': 'completed'
        }
```

### Performance Metrics Schema

```python
class PerformanceMetricsSchema:
    """Schema for performance tracking."""
    
    @staticmethod
    def create_performance_metric(
        metric_type: str,
        value: float,
        dimensions: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Create a performance metric document.
        
        Schema:
        {
            "id": "perf-2024-01-15-001",
            "metric_type": "query_latency",  # Partition key
            "metric_name": "p95_latency_ms",
            "value": 125.5,
            "timestamp": "2024-01-15T10:00:00Z",
            "dimensions": {
                "search_service": "search-prod",
                "index_name": "products",
                "query_type": "semantic"
            },
            "aggregation_period": "1hour"
        }
        """
        import uuid
        
        return {
            'id': f"perf-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}",
            'metric_type': metric_type,
            'metric_name': dimensions.get('metric_name', metric_type) if dimensions else metric_type,
            'value': value,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'dimensions': dimensions or {},
            'aggregation_period': dimensions.get('aggregation_period', '1hour') if dimensions else '1hour'
        }
```

---

## Data Storage Patterns

### Query Log Writer

```python
class QueryLogWriter:
    """Write query logs to Cosmos DB."""
    
    def __init__(self, cosmos_client, database_name, container_name='query_logs'):
        self.client = cosmos_client
        self.database = self.client.get_database_client(database_name)
        self.container = self.database.get_container_client(container_name)
    
    def log_query(self, query_log: Dict) -> Dict:
        """
        Write query log to Cosmos DB.
        
        Args:
            query_log: Query log document
            
        Returns:
            Created document with RU charge
        """
        try:
            response = self.container.create_item(body=query_log)
            
            # Track RU consumption
            ru_charge = self.container.client_connection.last_response_headers.get(
                'x-ms-request-charge', 0
            )
            
            print(f"✅ Query logged. RU charge: {ru_charge}")
            return response
            
        except exceptions.CosmosHttpResponseError as e:
            print(f"❌ Error logging query: {e}")
            raise
    
    def log_query_batch(self, query_logs: List[Dict]) -> int:
        """
        Write multiple query logs efficiently.
        
        Args:
            query_logs: List of query log documents
            
        Returns:
            Number of successfully written logs
        """
        success_count = 0
        total_ru = 0
        
        for log in query_logs:
            try:
                self.container.create_item(body=log)
                success_count += 1
                
                ru_charge = float(self.container.client_connection.last_response_headers.get(
                    'x-ms-request-charge', 0
                ))
                total_ru += ru_charge
                
            except exceptions.CosmosHttpResponseError as e:
                print(f"Error logging query {log.get('id')}: {e}")
        
        print(f"✅ Logged {success_count}/{len(query_logs)} queries. Total RU: {total_ru:.2f}")
        return success_count

# Usage
log_writer = QueryLogWriter(
    cosmos_client=CosmosClient(endpoint, key),
    database_name="search_evaluation"
)

query_log = QueryLogSchema.create_query_log(
    query_text="azure search",
    search_service="search-prod",
    index_name="docs",
    results=[...]
)

log_writer.log_query(query_log)
```

### Evaluation Results Storage

```python
class EvaluationResultsStore:
    """Store and retrieve evaluation results."""
    
    def __init__(self, cosmos_client, database_name):
        self.client = cosmos_client
        self.database = self.client.get_database_client(database_name)
        self.container = self.database.get_container_client('evaluation_results')
    
    def save_evaluation(self, evaluation_result: Dict) -> str:
        """Save evaluation result."""
        try:
            response = self.container.create_item(body=evaluation_result)
            print(f"✅ Evaluation saved: {response['id']}")
            return response['id']
        except exceptions.CosmosHttpResponseError as e:
            print(f"❌ Error saving evaluation: {e}")
            raise
    
    def get_evaluation(self, experiment_id: str, evaluation_id: str) -> Dict:
        """Retrieve specific evaluation result."""
        try:
            return self.container.read_item(
                item=evaluation_id,
                partition_key=experiment_id
            )
        except exceptions.CosmosResourceNotFoundError:
            print(f"Evaluation {evaluation_id} not found")
            return None
    
    def get_experiment_results(self, experiment_id: str) -> List[Dict]:
        """Get all results for an experiment."""
        query = "SELECT * FROM c WHERE c.experiment_id = @experiment_id ORDER BY c.timestamp DESC"
        
        parameters = [
            {"name": "@experiment_id", "value": experiment_id}
        ]
        
        results = list(self.container.query_items(
            query=query,
            parameters=parameters,
            partition_key=experiment_id
        ))
        
        return results
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict:
        """Compare metrics across multiple experiments."""
        comparison = {
            'experiments': [],
            'metric_comparison': {}
        }
        
        for exp_id in experiment_ids:
            results = self.get_experiment_results(exp_id)
            if results:
                latest = results[0]  # Most recent result
                comparison['experiments'].append({
                    'experiment_id': exp_id,
                    'timestamp': latest['timestamp'],
                    'metrics': latest['metrics']
                })
        
        # Calculate metric comparisons
        if comparison['experiments']:
            all_metrics = set()
            for exp in comparison['experiments']:
                all_metrics.update(exp['metrics'].keys())
            
            for metric in all_metrics:
                values = [
                    exp['metrics'].get(metric, 0)
                    for exp in comparison['experiments']
                ]
                comparison['metric_comparison'][metric] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'best_experiment': comparison['experiments'][values.index(max(values))]['experiment_id']
                }
        
        return comparison

# Usage
eval_store = EvaluationResultsStore(
    cosmos_client=CosmosClient(endpoint, key),
    database_name="search_evaluation"
)

# Save evaluation
eval_result = EvaluationResultSchema.create_evaluation_result(
    experiment_id="exp-hybrid-001",
    index_config={'search_type': 'hybrid'},
    metrics={'precision_at_10': 0.85, 'recall_at_10': 0.72}
)
eval_store.save_evaluation(eval_result)

# Compare experiments
comparison = eval_store.compare_experiments(['exp-hybrid-001', 'exp-fulltext-001'])
print(f"Best for precision: {comparison['metric_comparison']['precision_at_10']['best_experiment']}")
```

---

## Querying and Analysis

### Query Analytics

```python
class QueryAnalytics:
    """Analyze query logs for insights."""
    
    def __init__(self, cosmos_client, database_name):
        self.client = cosmos_client
        self.database = self.client.get_database_client(database_name)
        self.container = self.database.get_container_client('query_logs')
    
    def get_top_queries(self, limit=10, time_range_hours=24):
        """Get most frequent queries."""
        from datetime import timedelta
        
        cutoff_time = (datetime.utcnow() - timedelta(hours=time_range_hours)).isoformat() + 'Z'
        
        query = """
        SELECT c.query_text, COUNT(1) as query_count
        FROM c
        WHERE c.timestamp > @cutoff_time
        GROUP BY c.query_text
        ORDER BY COUNT(1) DESC
        OFFSET 0 LIMIT @limit
        """
        
        parameters = [
            {"name": "@cutoff_time", "value": cutoff_time},
            {"name": "@limit", "value": limit}
        ]
        
        results = list(self.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        return results
    
    def get_zero_result_queries(self, limit=20):
        """Find queries with no results."""
        query = """
        SELECT c.query_text, c.timestamp, c.user_id
        FROM c
        WHERE c.results.count = 0
        ORDER BY c.timestamp DESC
        OFFSET 0 LIMIT @limit
        """
        
        results = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@limit", "value": limit}],
            enable_cross_partition_query=True
        ))
        
        return results
    
    def get_slow_queries(self, threshold_ms=1000, limit=20):
        """Find slow queries."""
        query = """
        SELECT c.query_text, c.performance.duration_ms, c.timestamp
        FROM c
        WHERE c.performance.duration_ms > @threshold
        ORDER BY c.performance.duration_ms DESC
        OFFSET 0 LIMIT @limit
        """
        
        results = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@threshold", "value": threshold_ms},
                {"name": "@limit", "value": limit}
            ],
            enable_cross_partition_query=True
        ))
        
        return results
    
    def get_query_performance_stats(self, time_range_hours=24):
        """Calculate query performance statistics."""
        from datetime import timedelta
        
        cutoff_time = (datetime.utcnow() - timedelta(hours=time_range_hours)).isoformat() + 'Z'
        
        query = """
        SELECT 
            AVG(c.performance.duration_ms) as avg_duration,
            MIN(c.performance.duration_ms) as min_duration,
            MAX(c.performance.duration_ms) as max_duration,
            COUNT(1) as total_queries
        FROM c
        WHERE c.timestamp > @cutoff_time
        """
        
        results = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@cutoff_time", "value": cutoff_time}],
            enable_cross_partition_query=True
        ))
        
        return results[0] if results else None

# Usage
analytics = QueryAnalytics(
    cosmos_client=CosmosClient(endpoint, key),
    database_name="search_evaluation"
)

# Get insights
top_queries = analytics.get_top_queries(limit=10)
print("Top 10 queries:")
for q in top_queries:
    print(f"  {q['query_text']}: {q['query_count']} searches")

zero_results = analytics.get_zero_result_queries()
print(f"\nFound {len(zero_results)} queries with no results")

performance = analytics.get_query_performance_stats()
print(f"\nAvg query time: {performance['avg_duration']:.2f}ms")
```

### Time-Series Analysis

```python
class TimeSeriesAnalytics:
    """Time-series analysis of search metrics."""
    
    def __init__(self, cosmos_client, database_name):
        self.client = cosmos_client
        self.database = self.client.get_database_client(database_name)
        self.metrics_container = self.database.get_container_client('performance_metrics')
    
    def get_metric_timeseries(self, metric_type: str, hours: int = 24):
        """Get time series data for a metric."""
        from datetime import timedelta
        
        cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + 'Z'
        
        query = """
        SELECT c.timestamp, c.value, c.dimensions
        FROM c
        WHERE c.metric_type = @metric_type 
          AND c.timestamp > @cutoff_time
        ORDER BY c.timestamp ASC
        """
        
        results = list(self.metrics_container.query_items(
            query=query,
            parameters=[
                {"name": "@metric_type", "value": metric_type},
                {"name": "@cutoff_time", "value": cutoff_time}
            ],
            partition_key=metric_type
        ))
        
        return results
    
    def calculate_trend(self, metric_type: str, hours: int = 24):
        """Calculate trend for a metric."""
        import numpy as np
        from datetime import datetime as dt
        
        data = self.get_metric_timeseries(metric_type, hours)
        
        if len(data) < 2:
            return None
        
        # Convert timestamps to hours since start
        start_time = dt.fromisoformat(data[0]['timestamp'].replace('Z', ''))
        x = []
        y = []
        
        for point in data:
            timestamp = dt.fromisoformat(point['timestamp'].replace('Z', ''))
            hours_since_start = (timestamp - start_time).total_seconds() / 3600
            x.append(hours_since_start)
            y.append(point['value'])
        
        # Linear regression
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
        
        return {
            'metric_type': metric_type,
            'slope': slope,
            'intercept': intercept,
            'trend': 'increasing' if slope > 0 else 'decreasing',
            'data_points': len(data),
            'current_value': y[-1],
            'predicted_next_hour': slope * (x[-1] + 1) + intercept
        }

# Usage
ts_analytics = TimeSeriesAnalytics(
    cosmos_client=CosmosClient(endpoint, key),
    database_name="search_evaluation"
)

trend = ts_analytics.calculate_trend('query_latency', hours=24)
print(f"Latency trend: {trend['trend']}")
print(f"Current: {trend['current_value']:.2f}ms")
print(f"Predicted next hour: {trend['predicted_next_hour']:.2f}ms")
```

---

## Performance Optimization

### Indexing Strategy

```python
class CosmosDBOptimization:
    """Optimize Cosmos DB performance."""
    
    @staticmethod
    def create_optimized_indexing_policy():
        """Create optimized indexing policy for query logs."""
        return {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [
                {
                    "path": "/query_text/*"
                },
                {
                    "path": "/timestamp/*"
                },
                {
                    "path": "/user_id/*"
                },
                {
                    "path": "/results/count/*"
                }
            ],
            "excludedPaths": [
                {
                    "path": "/results/top_results/*"  # Large array, not queried
                },
                {
                    "path": "/metadata/*"  # Variable schema
                }
            ],
            "compositeIndexes": [
                [
                    {
                        "path": "/timestamp",
                        "order": "descending"
                    },
                    {
                        "path": "/query_text",
                        "order": "ascending"
                    }
                ]
            ]
        }
    
    @staticmethod
    def estimate_ru_usage(operation_type: str, document_size_kb: float):
        """
        Estimate RU usage for operations.
        
        Args:
            operation_type: 'create', 'read', 'query', 'update'
            document_size_kb: Document size in KB
            
        Returns:
            Estimated RU charge
        """
        # Approximate RU costs
        ru_costs = {
            'read': 1.0,  # 1 RU per 1 KB
            'create': 5.0,  # ~5 RU per 1 KB
            'update': 10.0,  # ~10 RU per 1 KB
            'query': 2.5  # Varies, ~2.5 RU per KB scanned
        }
        
        base_cost = ru_costs.get(operation_type, 5.0)
        return base_cost * document_size_kb

# Usage
indexing_policy = CosmosDBOptimization.create_optimized_indexing_policy()

# Apply to container
container = database.create_container(
    id='optimized_query_logs',
    partition_key=PartitionKey(path='/search_service'),
    indexing_policy=indexing_policy
)
```

---

## Cost Management

### RU Consumption Tracking

```python
class CostTracker:
    """Track Cosmos DB RU consumption and costs."""
    
    def __init__(self, cosmos_client, database_name):
        self.client = cosmos_client
        self.database = self.client.get_database_client(database_name)
        self.total_ru = 0
        self.operation_counts = {}
    
    def track_operation(self, container_name: str, operation: str, response):
        """Track RU consumption for an operation."""
        try:
            ru_charge = float(response.headers.get('x-ms-request-charge', 0))
            self.total_ru += ru_charge
            
            key = f"{container_name}_{operation}"
            if key not in self.operation_counts:
                self.operation_counts[key] = {'count': 0, 'total_ru': 0}
            
            self.operation_counts[key]['count'] += 1
            self.operation_counts[key]['total_ru'] += ru_charge
            
        except Exception as e:
            print(f"Error tracking RU: {e}")
    
    def get_cost_summary(self, ru_price_per_100=0.008):
        """
        Get cost summary.
        
        Args:
            ru_price_per_100: Price per 100 RUs ($0.008 default)
            
        Returns:
            Cost summary dict
        """
        total_cost = (self.total_ru / 100) * ru_price_per_100
        
        summary = {
            'total_ru': self.total_ru,
            'estimated_cost_usd': total_cost,
            'operations': {}
        }
        
        for operation, stats in self.operation_counts.items():
            avg_ru = stats['total_ru'] / stats['count'] if stats['count'] > 0 else 0
            summary['operations'][operation] = {
                'count': stats['count'],
                'total_ru': stats['total_ru'],
                'avg_ru_per_operation': avg_ru,
                'cost_usd': (stats['total_ru'] / 100) * ru_price_per_100
            }
        
        return summary

# Usage
tracker = CostTracker(cosmos_client, "search_evaluation")

# Track operations
container = database.get_container_client('query_logs')
response = container.create_item(body=query_log)
tracker.track_operation('query_logs', 'create', response)

# Get summary
summary = tracker.get_cost_summary()
print(f"Total RU: {summary['total_ru']:.2f}")
print(f"Estimated cost: ${summary['estimated_cost_usd']:.4f}")
```

---

## Integration with Search

### End-to-End Example

```python
class SearchWithAnalytics:
    """Azure AI Search with Cosmos DB analytics integration."""
    
    def __init__(self, search_client, cosmos_client, database_name):
        self.search_client = search_client
        self.log_writer = QueryLogWriter(cosmos_client, database_name)
    
    def search_and_log(self, query_text: str, **search_params):
        """Execute search and log to Cosmos DB."""
        import time
        
        # Execute search
        start_time = time.time()
        results = self.search_client.search(query_text, **search_params)
        duration_ms = (time.time() - start_time) * 1000
        
        # Convert results to list
        result_list = []
        for result in results:
            result_list.append({
                'id': result.get('id'),
                'search_score': result.get('@search.score'),
                'title': result.get('title')
            })
        
        # Create log
        query_log = QueryLogSchema.create_query_log(
            query_text=query_text,
            search_service="search-prod",
            index_name=self.search_client._index_name,
            results=result_list,
            metadata={'duration_ms': duration_ms}
        )
        
        # Write to Cosmos DB
        self.log_writer.log_query(query_log)
        
        return result_list

# Usage
from azure.search.documents import SearchClient

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name="products",
    credential=AzureKeyCredential(search_key)
)

search_analytics = SearchWithAnalytics(
    search_client=search_client,
    cosmos_client=CosmosClient(cosmos_endpoint, cosmos_key),
    database_name="search_evaluation"
)

results = search_analytics.search_and_log("laptop computers", top=10)
```

---

## Next Steps

- **[Azure Monitor & Logging](./07-azure-monitor-logging.md)** - Advanced monitoring
- **[A/B Testing Framework](./17-ab-testing-framework.md)** - Run experiments
- **[Cost Analysis](./19-cost-analysis.md)** - Optimize costs

---

*See also: [Azure AI Search Setup](./04-azure-ai-search-setup.md) | [Performance Optimization](./12-query-optimization.md)*