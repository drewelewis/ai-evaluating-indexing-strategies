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

**Scenario**: You're running an e-commerce platform with 50,000 daily searches across three regions (US, Europe, Asia). Each search needs to be logged for analytics: query text, results returned, user behavior, performance metrics. You need to analyze this data to improve search relevance, identify trending products, and detect performance degradation—all while maintaining sub-second query response times.

**The challenge**: Traditional relational databases struggle with this workload. SQL databases require rigid schemas that don't adapt well to evolving metrics. They can't handle the write-heavy nature of real-time logging (50,000+ writes/day) while simultaneously supporting complex analytical queries. Multi-region replication is expensive and complex to configure.

**The solution**: Azure Cosmos DB provides a purpose-built platform for search analytics that eliminates these constraints.

#### Why Cosmos DB Excels at Search Analytics

Azure Cosmos DB is ideal for search evaluation and analytics because it provides:

**1. Low-Latency Writes (Sub-10ms Performance)**
- **Real-time query logging**: Write search events as they occur without impacting search performance
- **Single-digit millisecond latency**: 99th percentile writes complete in <10ms globally
- **No bottlenecks**: Elastic scaling handles traffic spikes during peak shopping hours
- **Real-world impact**: Your search logging doesn't slow down the user experience

**2. Global Distribution (Multi-Region Analytics)**
- **Multi-region writes**: Log searches in the user's region for minimal latency
- **Automatic replication**: Analytics dashboards in one region query data from all regions
- **Regional compliance**: Keep EU customer data in EU regions while aggregating globally
- **Consistency options**: Choose between strong consistency (accurate counts) or eventual (lower cost)

**3. Flexible Schema (Schema Evolution)**
- **No migrations required**: Add new metrics (e.g., "voice_search_used": true) without ALTER TABLE
- **Nested structures**: Store complex objects like search filters, facets, and result metadata
- **Heterogeneous data**: Different document types (query logs, evaluation results, A/B tests) in one container
- **Version compatibility**: Old logs remain readable even as your tracking evolves

**4. Rich Querying (SQL API for Analytics)**
- **SQL-like syntax**: Query with SELECT, JOIN, GROUP BY—no need to learn a new language
- **Aggregations**: Calculate P95 latency, average result count, query frequency with built-in functions
- **Cross-partition queries**: Analyze data across all customers, products, or time periods
- **Subqueries and joins**: Complex analytics like "queries with zero results that led to cart abandonment"

**5. Change Feed (Real-Time Processing)**
- **Event-driven analytics**: Trigger Azure Functions when new search logs arrive
- **Real-time dashboards**: Update Power BI visualizations as searches occur
- **Anomaly detection**: Alert when zero-result queries spike above normal thresholds
- **Integration**: Feed data to Azure Stream Analytics, Azure Synapse, or custom processors

**6. Time-to-Live (TTL) (Automatic Cleanup)**
- **Automatic deletion**: Set query logs to expire after 30 days—no cleanup jobs needed
- **Cost optimization**: Keep recent data hot, archive old data, or delete automatically
- **Granular control**: Set TTL at document level (keep critical logs longer, delete test data sooner)
- **No manual maintenance**: TTL runs in background consuming minimal RUs

#### When to Choose Cosmos DB vs Alternatives

**Choose Cosmos DB for search analytics when:**
- ✅ You need real-time query logging with low latency (<10ms writes)
- ✅ Your application spans multiple geographic regions
- ✅ You're tracking evolving metrics that don't fit rigid schemas
- ✅ You need to run complex analytics queries on recent data (last 30-90 days)
- ✅ You want change feed integration for real-time processing
- ✅ You need automatic data expiration (TTL) for compliance or cost control

**Consider alternatives when:**
- ❌ **Azure Log Analytics**: Best for long-term log retention (>90 days) and cross-service correlation, but higher query latency
- ❌ **Azure Synapse Analytics**: Best for petabyte-scale data warehousing and batch analytics, but not real-time
- ❌ **Azure SQL Database**: Best for transactional workloads with fixed schemas and strong relational integrity
- ❌ **Azure Table Storage**: Best for simple key-value lookups with minimal querying needs

#### Real-World Cost Comparison

**Scenario**: 50,000 daily searches, 30-day retention, 2KB average log size

**Cosmos DB** (400 RU/s shared throughput):
- Storage: 50K × 30 days × 2KB = 3GB → $0.75/month
- Throughput: 400 RU/s → $24/month
- **Total: ~$25/month**

**Azure SQL Database** (Basic tier):
- Database: $5/month (2GB limit—need Standard for 3GB)
- Standard S0: $15/month
- Egress for global reads: +$5/month
- **Total: ~$20/month** (but no global distribution, requires schema management, slower writes)

**Azure Log Analytics** (Pay-as-you-go):
- Ingestion: 3GB/month × $2.76/GB = $8.28/month
- Queries: ~100 queries/day × 30 days = minimal cost
- **Total: ~$9/month** (but higher query latency, no change feed, limited data types)

### Architecture Pattern

**The analytics pipeline**: Understanding how data flows from search queries to actionable insights.

```
┌──────────────────┐
│  Search Queries  │ ← User searches: "wireless headphones"
└────────┬─────────┘
         │ 1. Execute search
         ▼
┌────────────────────────┐
│  Azure AI Search       │ ← Returns results with scores, facets
│  - Execute query       │    Measures: latency, result count
│  - Calculate scores    │
│  - Return results      │
└────────┬───────────────┘
         │ 2. Log query + results + performance
         ▼
┌────────────────────────┐
│  Azure Cosmos DB       │ ← Write query log (5-10ms latency)
│  - Query logs          │    Containers:
│  - Evaluation results  │    • query_logs (30-day TTL)
│  - Performance metrics │    • evaluation_results (permanent)
│  - A/B test results    │    • performance_metrics (90-day TTL)
└────────┬───────────────┘
         │ 3a. Change feed triggers real-time processing
         ▼
┌────────────────────────┐
│  Azure Functions       │ ← Processes new logs
│  - Detect anomalies    │    • Spikes in zero-result queries
│  - Update aggregates   │    • Slow query detection
│  - Send alerts         │    • Update real-time counters
└────────────────────────┘
         │
         │ 3b. Query for analytics
         ▼
┌────────────────────────┐
│  Analytics Dashboard   │ ← Visualize insights
│  (Power BI / Custom)   │    • Top queries by volume
│  - Query trends        │    • P95 latency trends
│  - Performance graphs  │    • Zero-result rate
│  - A/B test results    │    • Search quality metrics
└────────────────────────┘
```

#### Pipeline Flow Explained

**Step 1: Search Execution**
- User submits query through your application
- Azure AI Search processes the query (BM25, vector, or hybrid)
- Search returns results with metadata: scores, highlights, facets
- Application measures: query latency, result count, filter usage

**Step 2: Log to Cosmos DB**
- Application writes structured log document (JSON format)
- Cosmos DB writes complete in 5-10ms (doesn't block user response)
- Document includes: query text, results, performance, user context
- Partitioned by `search_service` for efficient queries per service

**Step 3a: Real-Time Processing (Change Feed)**
- Cosmos DB change feed notifies Azure Functions of new documents
- Function processes logs for:
  - **Anomaly detection**: Is zero-result rate >10% (normally 3%)?
  - **Performance alerts**: Is P95 latency >500ms (normally 200ms)?
  - **Aggregate updates**: Update hourly query count counters
- Alerts sent via email, Teams, or incident management systems

**Step 3b: Analytics Queries**
- Analysts query Cosmos DB for insights:
  - **Top queries**: What are users searching for most?
  - **Zero-result queries**: Where is search failing?
  - **Performance trends**: Is latency increasing over time?
  - **A/B test analysis**: Which algorithm performs better?
- Power BI connects directly to Cosmos DB for visualizations
- Custom dashboards query via SQL API

#### Consistency Level Decision: Impact on Analytics Accuracy

Cosmos DB offers five consistency levels that trade off between **accuracy** and **cost/performance**:

| Consistency Level | Analytics Use Case | Accuracy | Cost | Latency |
|------------------|-------------------|----------|------|---------|
| **Strong** | A/B test final reports (exact counts required) | 100% accurate | Highest (2× RUs) | Highest |
| **Bounded Staleness** | Real-time dashboards (max 5-min lag acceptable) | Eventually accurate | High (1.5× RUs) | Medium |
| **Session** | User's own query history | Accurate for session | Medium (1× RUs) | Low |
| **Consistent Prefix** | Query trend analysis | Reads never go backwards | Medium (1× RUs) | Low |
| **Eventual** | Query logging (analytics run later) | Eventually accurate | Lowest (0.5× RUs) | Lowest |

**Recommended for search analytics**: **Session** consistency
- **Why**: Query logs written in one session are immediately readable in that session
- **Cost**: Standard RU pricing (no premium for stronger consistency)
- **Accuracy**: Analytics queries see consistent snapshots, no partial updates
- **Latency**: Single-digit millisecond reads globally

**When to use Strong consistency**:
- Final A/B test reports where exact counts matter (e.g., "Variant A had 5,234 queries")
- Compliance requirements for audit trails (ensure no logs are missed)
- Real-time leaderboards or quota enforcement

**When to use Eventual consistency**:
- High-volume logging where eventual aggregation is acceptable
- Cost optimization for read-heavy analytics workloads
- Non-critical metrics like search suggestion usage

---

## Setup and Configuration

### Understanding Cosmos DB Provisioning Models

Before creating your Cosmos DB account, choose the right provisioning model for your search analytics workload:

#### Provisioned Throughput vs Serverless vs Autoscale

| Model | Best For | Cost Structure | Use Case for Search Analytics |
|-------|----------|----------------|-------------------------------|
| **Provisioned Throughput** | Predictable workloads | Pay for reserved RU/s | **Production search logging** with steady traffic (e.g., 10K-100K queries/day) |
| **Serverless** | Unpredictable/intermittent | Pay per request | **Development/testing** or low-volume analytics (<10K requests/day) |
| **Autoscale** | Variable workloads | Pay for max RU/s used | **Seasonal search** (e.g., retail peak during holidays) or A/B testing phases |

**Real-world decision matrix**:

**Choose Provisioned Throughput (400-1000 RU/s) if:**
- ✅ Your search volume is consistent day-to-day (±20% variance)
- ✅ You're logging >10,000 queries/day continuously
- ✅ You need predictable monthly costs for budgeting
- ✅ You can share throughput across multiple containers (cost savings)
- **Cost example**: 400 RU/s = $24/month, handles ~50K queries/day at 8 RUs/write

**Choose Serverless if:**
- ✅ You're in development/testing phase
- ✅ Search traffic is <10,000 queries/day or intermittent (batch processing)
- ✅ You prioritize zero fixed costs over per-request costs
- ✅ Storage <50GB (serverless limit)
- **Cost example**: 50 RU/operation × 10K operations/day × 30 days = 15M RUs/month = $3.75/month

**Choose Autoscale (400-4000 RU/s range) if:**
- ✅ Your search traffic varies significantly (10× difference peak vs off-peak)
- ✅ You run periodic batch analytics jobs (nightly aggregations)
- ✅ You're running A/B tests with variable query load
- ✅ You want protection against traffic spikes without manual scaling
- **Cost example**: Autoscale 400-4000 RU/s = ~$36/month (scales to 4000 during peaks, bills for actual usage)

#### Consistency Level Impact on Analytics

When creating your account, selecting the right default consistency level is critical:

**For Search Analytics, use Session consistency** (recommended):
- **Write latency**: <10ms (no consistency overhead on writes)
- **Read latency**: <10ms (reads from nearest replica)
- **Analytics accuracy**: Queries see their own writes immediately
- **Cost**: Standard RU pricing (1× multiplier)
- **Use case**: Query logs written by application are immediately queryable for that session

**When to override with Strong consistency**:
```python
# Use Strong consistency for critical A/B test final counts
ab_test_results = container.read_item(
    item=test_id,
    partition_key=test_id,
    consistency_level='Strong'  # Ensure all regions see exact same count
)
```

**When to use Eventual consistency**:
```python
# Use Eventual for cost-optimized analytics queries
query = "SELECT COUNT(1) as total_queries FROM c WHERE c.timestamp > @yesterday"
results = container.query_items(
    query=query,
    parameters=[...],
    enable_cross_partition_query=True,
    consistency_level='Eventual'  # 50% RU discount on reads
)
```

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

#### Key Configuration Decisions Explained

**1. `--enable-analytical-storage true`**
- **What it does**: Creates a column-store replica of your data optimized for analytics
- **When to use**: If you plan to query with Azure Synapse Analytics for complex aggregations
- **Cost**: Analytical storage costs $0.03/GB/month (vs $0.25/GB for transactional)
- **Search analytics use case**: Enable if you're aggregating >1M query logs for deep analysis (e.g., "show me all queries containing 'wireless' grouped by hour for the last year")
- **Skip if**: You're only querying recent data (<90 days) with simple SQL queries

**2. `--default-consistency-level "Session"`**
- **Why Session**: Balances consistency with performance for analytics
- **Alternatives**: 
  - `"Strong"` if you need globally consistent A/B test counts (2× cost)
  - `"Eventual"` if you're optimizing for cost and can tolerate slight delays (0.5× cost on reads)

**3. `--enable-automatic-failover false`**
- **Why disabled for single-region**: Automatic failover requires multiple regions
- **Enable for production**: If you're replicating to multiple regions for global analytics
- **How to enable**:
  ```bash
  az cosmosdb update \
    --name $ACCOUNT_NAME \
    --resource-group $RESOURCE_GROUP \
    --locations regionName=eastus failoverPriority=0 \
                regionName=westus failoverPriority=1 \
    --enable-automatic-failover true
  ```

**4. `--throughput 400`**
- **Minimum for shared throughput**: 400 RU/s is the minimum for database-level provisioning
- **Shared across containers**: All containers in this database share the 400 RU/s pool
- **Cost**: $24/month (400 RU/s × $0.00008/hour × 730 hours)
- **When to increase**: If you're logging >50K queries/day or running frequent analytics
  - 800 RU/s → $48/month (handles ~100K queries/day)
  - 1000 RU/s → $60/month (handles ~125K queries/day)

#### Multi-Region Setup for Global Analytics

If your search application spans multiple regions, replicate Cosmos DB to each region:

```bash
# Add read region (Europe)
az cosmosdb update \
  --name $ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --locations regionName=eastus failoverPriority=0 isZoneRedundant=false \
              regionName=northeurope failoverPriority=1 isZoneRedundant=false

# Enable multi-region writes (for logging in each region)
az cosmosdb update \
  --name $ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --enable-multiple-write-locations true
```

**Multi-region cost impact**:
- **Storage**: Replicated to each region (2 regions = 2× storage cost)
- **Throughput**: Shared across regions (400 RU/s globally, not per region)
- **Writes**: Multi-region writes cost +1 RU per additional region
  - Single region: 5 RU/write
  - Two regions: 6 RU/write (5 base + 1 for replication)
- **Reads**: Reads from local region (no additional cost)

**When multi-region makes sense**:
- ✅ Users distributed globally (reduce write latency from 100ms to 10ms)
- ✅ Compliance requires regional data residency (EU logs stay in EU)
- ✅ High availability requirements (automatic failover if region goes down)
- ❌ Single-region application (unnecessary cost overhead)

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

#### Partition Key Selection: Critical for Performance

Choosing the right partition key is the **most important decision** for Cosmos DB performance. A poor partition key choice causes hot partitions, throttling, and high costs.

**The golden rule**: Your partition key should:
1. **High cardinality**: Many unique values (ideally >100)
2. **Even distribution**: Data spread evenly across partitions
3. **Query-aligned**: Most queries filter by the partition key

**Partition Key Analysis for Search Analytics**:

| Container | Partition Key | Cardinality | Why This Choice | Query Pattern |
|-----------|---------------|-------------|-----------------|---------------|
| **query_logs** | `/search_service` | Low-Medium (10-100 services) | Queries often filtered by service | `WHERE c.search_service = 'prod-search'` |
| **evaluation_results** | `/experiment_id` | Medium (100-1000 experiments) | Each experiment queried independently | `WHERE c.experiment_id = 'exp-001'` |
| **performance_metrics** | `/metric_type` | Low (10-50 metric types) | Metrics queried by type for charting | `WHERE c.metric_type = 'query_latency'` |
| **ab_test_results** | `/test_id` | Medium (100-1000 tests) | Each test is a separate analysis | `WHERE c.test_id = 'test-20240115'` |

**Alternative partition key considerations**:

**For query_logs, why not `/user_id`?**
- ✅ **Pros**: High cardinality (millions of users), even distribution
- ❌ **Cons**: Most analytics queries are NOT filtered by user ("show all queries for product X")
- ❌ **Impact**: Every analytics query becomes a cross-partition query (higher RU cost, slower)
- **Verdict**: Use `/user_id` only if your primary use case is "user query history"

**For query_logs, why not `/query_text`?**
- ✅ **Pros**: High cardinality (thousands of unique queries)
- ❌ **Cons**: Skewed distribution (popular queries like "laptop" dominate)
- ❌ **Impact**: Hot partitions for popular queries cause throttling
- **Verdict**: Avoid. Query text distribution is inherently uneven.

**For query_logs, what about a synthetic key `/date_hour`?**
- Example: `"2024-01-15-10"` (date + hour)
- ✅ **Pros**: Even distribution across time
- ✅ **Pros**: Time-range queries stay within few partitions
- ❌ **Cons**: Older partitions become "cold" and can't be queried efficiently
- **Verdict**: Good for append-only time-series data with recent-focused queries

**Recommended approach for high-volume query logs**: **Hierarchical Partition Keys (HPK)**

```python
# Use hierarchical partition key for better distribution
containers_config = [
    {
        'id': 'query_logs_hpk',
        'partition_key': {
            'paths': ['/search_service', '/date'],  # Hierarchical: service > date
            'kind': 'MultiHash'
        },
        'default_ttl': 2592000
    }
]

# Documents structured with both levels
query_log = {
    'id': 'log-001',
    'search_service': 'prod-search',  # Level 1
    'date': '2024-01-15',              # Level 2
    'query_text': 'laptop',
    # ... other fields
}

# Queries can target specific partition
results = container.query_items(
    query="SELECT * FROM c WHERE c.query_text LIKE '%laptop%'",
    partition_key=['prod-search', '2024-01-15'],  # Query single partition
    enable_cross_partition_query=False
)
```

**HPK benefits**:
- Overcomes 20GB partition limit (each date becomes a sub-partition)
- Queries filtered by both service AND date stay in one partition (lower RU cost)
- Better distribution as data grows over time

#### TTL (Time-to-Live) Configuration Strategy

TTL automatically deletes old documents, reducing storage costs and improving query performance.

**TTL configuration per container**:

```python
# query_logs: 30-day retention (compliance requirement)
'default_ttl': 2592000  # 30 days in seconds

# evaluation_results: permanent (historical comparison)
'default_ttl': -1  # Never expire

# performance_metrics: 90-day retention (trend analysis)
'default_ttl': 7776000  # 90 days

# ab_test_results: permanent (audit trail)
'default_ttl': -1
```

**Document-level TTL override**:

```python
# Keep critical query logs longer than default
critical_log = {
    'id': 'log-001',
    'query_text': 'security breach search',
    'search_service': 'prod',
    'ttl': 31536000  # Override: keep for 1 year (365 days)
}

# Expire test data immediately
test_log = {
    'id': 'test-log-001',
    'query_text': 'test query',
    'search_service': 'dev',
    'ttl': 60  # Expire in 60 seconds
}

# Prevent specific document from expiring
important_eval = {
    'id': 'eval-baseline',
    'experiment_id': 'baseline',
    'ttl': -1  # Never expire, even if container default is 30 days
}
```

**TTL cost impact**:

**Scenario**: 50K queries/day, 2KB per log, 30-day retention

**Without TTL** (manual cleanup):
- Month 1: 3GB storage = $0.75
- Month 2: 6GB storage = $1.50 (forgot to delete old logs)
- Month 3: 9GB storage = $2.25 (still not deleted)
- **Total year**: ~$15-20 in unnecessary storage costs
- **Effort**: Write cleanup scripts, schedule jobs, monitor execution

**With TTL** (automatic):
- Every month: 3GB storage = $0.75 (consistent)
- **Total year**: $9 storage cost
- **Effort**: Zero maintenance
- **Bonus**: No RU cost for deletions (TTL runs in background)

---

## Schema Design

### Understanding Schema Flexibility in Cosmos DB

Unlike relational databases, Cosmos DB doesn't enforce schemas—documents in the same container can have completely different structures. This flexibility is powerful for search analytics where metrics evolve over time.

**Schema evolution example**:

**Week 1**: Simple query logging
```json
{
  "id": "log-001",
  "query_text": "laptop",
  "timestamp": "2024-01-15T10:00:00Z"
}
```

**Week 2**: Add performance tracking (no migration needed!)
```json
{
  "id": "log-002",
  "query_text": "wireless mouse",
  "timestamp": "2024-01-22T10:00:00Z",
  "performance": {
    "duration_ms": 45
  }
}
```

**Week 3**: Add user context and filters
```json
{
  "id": "log-003",
  "query_text": "gaming keyboard",
  "timestamp": "2024-01-29T10:00:00Z",
  "performance": {
    "duration_ms": 52,
    "ru_charge": 2.5
  },
  "user_id": "user123",
  "filters": ["category:electronics", "price:<100"]
}
```

**Key insight**: Old documents (log-001, log-002) remain queryable. New fields are optional. Your analytics queries handle missing fields gracefully:

```sql
-- Query works across all schema versions
SELECT c.query_text, c.performance.duration_ms ?? 0 as latency
FROM c
WHERE c.timestamp > '2024-01-01'
```

**Best practices for schema flexibility**:
1. **Add fields, don't change types**: Adding `filters` array is safe; changing `timestamp` from string to number breaks queries
2. **Use nested objects for related fields**: Group `performance.*` fields together for clarity
3. **Document your schema**: Even though Cosmos DB doesn't enforce schemas, your team needs to know the structure
4. **Version your documents**: Add `schema_version: 1` to enable future migrations if needed

### Query Log Schema

**Purpose**: Track every search query, its results, performance, and user context for analytics.

**Schema design principles**:
- **Denormalize for performance**: Store top results directly in log (avoid joins)
- **Optimize for common queries**: Include pre-calculated fields (result count, score range)
- **Balance detail vs cost**: Store top 10 results, not all 1000 (reduces document size → lower RU cost)

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

#### Query Log Schema Design Decisions

**1. Why store `top_results` instead of all results?**

**Problem**: A search returning 1,000 results creates a document >50KB
- **Storage cost**: 50KB × 50K queries/day × 30 days = 75GB storage → $18.75/month
- **RU cost**: Writing 50KB document costs ~25 RU (vs 5 RU for 2KB document)
- **Query performance**: Reading large documents slows analytics queries

**Solution**: Store only top 10 results
- **Storage**: 2KB × 50K queries/day × 30 days = 3GB → $0.75/month (25× savings)
- **RU cost**: 5 RU per write (5× savings)
- **Insight**: Analytics rarely need beyond top 10 results

**When to store more results**:
- Relevance tuning: Store top 20-50 to analyze ranking quality
- Click-through analysis: Store all results user saw before clicking
- Compliance: Legal requirement to log all returned data

**2. Why pre-calculate `search_score_range`?**

**Without pre-calculation**, finding score range requires aggregation:
```sql
SELECT MIN(r.search_score) as min, MAX(r.search_score) as max
FROM c
JOIN r IN c.results.top_results
WHERE c.timestamp > '2024-01-15'
```
- **Cost**: Cross-partition query, scans all documents → 50+ RUs
- **Performance**: 500-1000ms query time

**With pre-calculation**:
```sql
SELECT c.results.search_score_range.min, c.results.search_score_range.max
FROM c
WHERE c.timestamp > '2024-01-15'
```
- **Cost**: Simple projection → 2-5 RUs
- **Performance**: <50ms query time

**Trade-off**: Slight increase in write complexity (calculate min/max) for massive read performance gain

**3. Why separate `performance` object?**

Grouping related metrics makes queries cleaner and schema evolution easier:

```python
# Good: grouped performance metrics
{
  "performance": {
    "duration_ms": 45,
    "ru_charge": 2.5,
    "index_used": "vector",
    "cache_hit": true
  }
}

# Avoid: flat structure
{
  "performance_duration_ms": 45,
  "performance_ru_charge": 2.5,
  "performance_index_used": "vector",
  "performance_cache_hit": true
}
```

**Benefits of grouping**:
- **Query clarity**: `SELECT c.performance FROM c` returns all performance metrics
- **Schema evolution**: Add `performance.cpu_ms` without polluting top level
- **Indexing efficiency**: Exclude entire `performance` object from index if not queried frequently

### Evaluation Result Schema

**Purpose**: Store results from search evaluation experiments to compare different indexing strategies, ranking algorithms, or configuration changes.

**Key differences from query logs**:
- **Permanent storage** (TTL = -1): Historical experiments used for future comparisons
- **Partition by experiment_id**: Each experiment queried independently
- **Rich metrics**: Stores precision, recall, NDCG, MAP—not just raw queries
- **Query-level detail**: Includes expected vs actual results for each test query

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

### Understanding Indexing in Cosmos DB

Cosmos DB **automatically indexes every field** in every document by default. For search analytics, this is often wasteful—indexing large result arrays or metadata objects consumes RUs and slows writes without improving query performance.

**Default indexing behavior**:
```json
{
  "id": "log-001",
  "query_text": "laptop",
  "results": {
    "top_results": [
      {"id": "doc1", "title": "Gaming Laptop", "description": "..."},
      {"id": "doc2", "title": "Business Laptop", "description": "..."},
      // 8 more results...
    ]
  }
}
```

**Problem**: Cosmos DB indexes `results.top_results[0].title`, `results.top_results[1].title`, etc.
- **Write RU cost**: 10-15 RUs per query log (vs 5 RUs without indexing)
- **Index size**: Large indexes slow queries and increase storage
- **Wasted effort**: You never query `WHERE c.results.top_results[5].description = "..."`

**Solution**: Selective indexing policy that indexes only queried fields

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

#### Indexing Policy Decisions Explained

**1. Included Paths: Index only queried fields**

```python
"includedPaths": [
    {"path": "/query_text/*"},  # Filtered: WHERE c.query_text = "laptop"
    {"path": "/timestamp/*"},   # Sorted: ORDER BY c.timestamp
    {"path": "/user_id/*"},     # Filtered: WHERE c.user_id = "user123"
    {"path": "/results/count/*"}  # Filtered: WHERE c.results.count = 0
]
```

**Common analytics queries**:
- "Show me all searches for 'laptop' today" → needs `query_text` index
- "List queries with zero results" → needs `results.count` index
- "Get user's recent queries" → needs `user_id` and `timestamp` indexes

**2. Excluded Paths: Skip indexing large, unqueried data**

```python
"excludedPaths": [
    {"path": "/results/top_results/*"},  # Never queried by field
    {"path": "/metadata/*"}               # Variable schema, rarely filtered
]
```

**Why exclude `results.top_results`**:
- **Never queried**: You don't filter by `WHERE c.results.top_results[0].title = "..."`
- **Retrieved whole**: Queries retrieve entire `top_results` array, not individual elements
- **Large data**: Array can be 1-2KB, indexing it adds 2-5 RUs per write

**RU savings**: 10 RU/write without exclusion → 5 RU/write with exclusion (50% savings)

**3. Composite Indexes: Optimize common query patterns**

```python
"compositeIndexes": [
    [
        {"path": "/timestamp", "order": "descending"},
        {"path": "/query_text", "order": "ascending"}
    ]
]
```

**Enables efficient queries like**:
```sql
SELECT * FROM c
WHERE c.timestamp > '2024-01-15'
ORDER BY c.timestamp DESC, c.query_text ASC
```

**Without composite index**: Cross-partition ORDER BY costs 50-100 RUs
**With composite index**: ORDER BY served from index → 5-10 RUs (10× savings)

**When to add composite indexes**:
- ✅ Frequently run query with ORDER BY multiple fields
- ✅ Query filters + sorts on same fields
- ❌ Rarely run queries (not worth the write overhead)
- ❌ Single-field ORDER BY (range index sufficient)

#### Indexing Policy Impact: Real-World Example

**Scenario**: 50,000 query logs/day, 2KB per document

**Default indexing** (index everything):
- Write cost: 10 RU/write × 50K = 500K RU/day
- Provisioned throughput needed: 500K RU/day ÷ 86,400 sec = 5.8 RU/s → provision 400 RU/s (minimum)
- Monthly cost: $24/month (400 RU/s)
- **Problem**: Wasting 4-5 RUs per write on indexing unused fields

**Optimized indexing** (exclude large arrays):
- Write cost: 5 RU/write × 50K = 250K RU/day
- Provisioned throughput needed: 250K RU/day ÷ 86,400 sec = 2.9 RU/s → provision 400 RU/s (minimum)
- Monthly cost: $24/month (400 RU/s)
- **Benefit**: 2× RU capacity headroom for analytics queries

**At scale** (200,000 query logs/day):
- Default: 2M RU/day → need 800 RU/s → $48/month
- Optimized: 1M RU/day → need 400 RU/s → $24/month
- **Savings**: $24/month (50% cost reduction)

### Query Optimization Techniques

**1. Partition Key Targeting**

**Inefficient** (cross-partition query):
```python
query = "SELECT * FROM c WHERE c.query_text = 'laptop'"
results = container.query_items(
    query=query,
    enable_cross_partition_query=True  # Scans ALL partitions
)
# Cost: 20-50 RUs (depends on partition count)
```

**Efficient** (single partition query):
```python
query = "SELECT * FROM c WHERE c.search_service = @service AND c.query_text = @text"
parameters = [
    {"name": "@service", "value": "prod-search"},
    {"name": "@text", "value": "laptop"}
]
results = container.query_items(
    query=query,
    parameters=parameters,
    partition_key="prod-search"  # Query only this partition
)
# Cost: 2-5 RUs (10× faster)
```

**2. Projection: SELECT only needed fields**

**Inefficient** (retrieve all fields):
```python
query = "SELECT * FROM c WHERE c.timestamp > @cutoff"
# Returns entire 2KB document including top_results array
```

**Efficient** (project specific fields):
```python
query = "SELECT c.id, c.query_text, c.timestamp FROM c WHERE c.timestamp > @cutoff"
# Returns only needed fields (~200 bytes per result)
# Cost: 5× less RUs, 10× faster over network
```

**3. Pagination: Use OFFSET/LIMIT**

**Inefficient** (retrieve all results):
```python
query = "SELECT * FROM c WHERE c.timestamp > @cutoff"
results = list(container.query_items(query=query))  # Could be 50K+ documents
# Cost: 100+ RUs, 10+ seconds
```

**Efficient** (paginate results):
```python
query = "SELECT * FROM c WHERE c.timestamp > @cutoff ORDER BY c.timestamp DESC OFFSET 0 LIMIT 100"
results = list(container.query_items(query=query))  # Only 100 documents
# Cost: 5-10 RUs, <1 second
# User fetches next page only if needed
```

**4. Aggregate in Query, Not in Code**

**Inefficient** (aggregate in Python):
```python
query = "SELECT c.performance.duration_ms FROM c"
results = list(container.query_items(query=query))  # Retrieves 50K documents
avg_latency = sum(r['duration_ms'] for r in results) / len(results)
# Cost: 50+ RUs, 5+ seconds, transfers 100MB+ over network
```

**Efficient** (aggregate in Cosmos DB):
```python
query = "SELECT AVG(c.performance.duration_ms) as avg_latency FROM c"
results = list(container.query_items(query=query))  # Returns 1 document
avg_latency = results[0]['avg_latency']
# Cost: 5-10 RUs, <1 second, transfers <1KB
```

---

## Cost Management

### Understanding Cosmos DB Pricing

**Two cost components**:

**1. Throughput (RU/s)**:
- **Provisioned**: Pay for reserved RU/s ($0.00008/hour per RU/s)
  - 400 RU/s = $24/month (minimum for shared throughput)
  - 1000 RU/s = $60/month
- **Serverless**: Pay per request ($0.25 per million RUs)
  - 1M RUs/month = $0.25 (cost-effective for <10K requests/day)

**2. Storage**:
- **Transactional storage**: $0.25/GB/month
- **Analytical storage**: $0.03/GB/month (if enabled)

**Real-world cost example** (50K queries/day, 30-day retention):

**Throughput cost**:
- Writes: 50K queries × 5 RU/write = 250K RU/day
- Reads (analytics): 1K queries × 5 RU/query = 5K RU/day
- Total: 255K RU/day ÷ 86,400 seconds = 2.95 RU/s average
- **Provision**: 400 RU/s (minimum) = $24/month

**Storage cost**:
- Data: 50K queries × 2KB × 30 days = 3GB
- **Cost**: 3GB × $0.25/GB = $0.75/month

**Total**: $24.75/month

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

#### Cost Optimization Strategies

**1. Use Serverless for Low-Volume Workloads**

**When to switch to serverless**:
- <10,000 requests/day (writes + reads combined)
- Development/testing environments
- Irregular workloads (batch processing once per week)

**Cost comparison** (5,000 requests/day):

**Provisioned** (400 RU/s minimum):
- Throughput: $24/month (always running)
- Storage: 1GB = $0.25/month
- **Total**: $24.25/month

**Serverless**:
- Requests: 5K req × 5 RU/req = 25K RU/day × 30 days = 750K RU/month
- Cost: 750K ÷ 1M × $0.25 = $0.19/month
- Storage: 1GB = $0.25/month
- **Total**: $0.44/month (98% savings)

**2. Share Throughput Across Containers**

**Dedicated throughput** (each container separate):
- query_logs: 400 RU/s = $24/month
- evaluation_results: 400 RU/s = $24/month
- performance_metrics: 400 RU/s = $24/month
- **Total**: $72/month

**Shared throughput** (database-level):
- Database: 400 RU/s shared across all containers = $24/month
- **Savings**: $48/month (67% reduction)

**When shared throughput works**:
- ✅ Containers have complementary usage patterns (writes during day, analytics at night)
- ✅ No single container consistently uses >50% of throughput
- ✅ Cost savings are more important than guaranteed per-container RUs

**When to use dedicated throughput**:
- ❌ One container has high, constant load (query logs from high-traffic site)
- ❌ Need guaranteed RUs for SLA compliance
- ❌ Containers require different consistency levels or geo-replication settings

**3. Optimize Document Size**

**Bloated document** (5KB):
```python
{
  "id": "log-001",
  "query_text": "laptop",
  "results": {
    "top_results": [
      {
        "id": "doc1",
        "title": "Gaming Laptop XPS 15",
        "description": "500 words of product description...",
        "full_specs": {...},  # Not needed for analytics
        "reviews": [...]       # Not needed for analytics
      },
      # 9 more results with full details
    ]
  }
}
```
- **Write cost**: 25 RU (5KB × 5 RU/KB)
- **Storage**: 50K queries × 5KB × 30 days = 7.5GB = $1.88/month

**Optimized document** (2KB):
```python
{
  "id": "log-001",
  "query_text": "laptop",
  "results": {
    "count": 10,
    "top_results": [
      {"id": "doc1", "search_score": 0.85},  # Only ID and score
      {"id": "doc2", "search_score": 0.72},
      # ...
    ]
  }
}
```
- **Write cost**: 10 RU (2KB × 5 RU/KB)
- **Storage**: 50K queries × 2KB × 30 days = 3GB = $0.75/month
- **Savings**: 15 RU/write + $1.13/month storage (60% reduction)

**4. Use TTL Aggressively**

**Conservative** (90-day retention):
- Storage: 50K × 2KB × 90 days = 9GB = $2.25/month

**Optimized** (30-day retention with selective long-term storage):
- Hot data (30 days): 50K × 2KB × 30 days = 3GB = $0.75/month
- Archive critical queries: Export to Azure Blob Storage (cold tier) = $0.01/GB/month
- **Total**: $0.75 + ($0.01 × 3GB × 3 months) = $0.84/month
- **Savings**: $1.41/month (63% reduction)

**5. Batch Reads with Continuation Tokens**

**Inefficient** (multiple small queries):
```python
for hour in range(24):
    query = f"SELECT * FROM c WHERE c.timestamp > '{hour}:00:00'"
    results = list(container.query_items(query=query))
# Cost: 24 queries × 5 RU = 120 RUs
```

**Efficient** (single query with continuation):
```python
query = "SELECT * FROM c WHERE c.timestamp > @start"
items = container.query_items(query=query, partition_key="prod-search")

# Process results with continuation token (automatic pagination)
for item in items:  # SDK handles pagination automatically
    process(item)
# Cost: 1 query × 5 RU = 5 RUs (24× savings)
```

---

## Integration with Search

### Real-World Integration Patterns

#### Pattern 1: Synchronous Logging (User Waits)

**Use when**: Logging is critical for compliance (financial transactions, healthcare)

```python
def search_with_mandatory_logging(query_text):
    try:
        # Execute search
        results = search_client.search(query_text)
        
        # Log to Cosmos DB (blocks until written)
        log_doc = create_query_log(query_text, results)
        cosmos_container.create_item(body=log_doc)
        
        return results
    except Exception as e:
        # If logging fails, search fails (compliance requirement)
        raise
```

**Pros**: Guaranteed logging, audit trail
**Cons**: Adds 10-20ms to search latency
**RU cost**: 5 RU/search

#### Pattern 2: Asynchronous Logging (Fire and Forget)

**Use when**: Search performance is critical, occasional log loss acceptable

```python
import asyncio

async def search_with_async_logging(query_text):
    # Execute search (returns immediately)
    results = search_client.search(query_text)
    
    # Log asynchronously (doesn't block)
    asyncio.create_task(log_to_cosmos(query_text, results))
    
    return results  # User gets results without waiting for log

async def log_to_cosmos(query_text, results):
    try:
        log_doc = create_query_log(query_text, results)
        cosmos_container.create_item(body=log_doc)
    except:
        # Log failed, but user's search succeeded
        pass
```

**Pros**: Zero impact on search latency
**Cons**: Logs may be lost if app crashes before write completes
**RU cost**: 5 RU/search (same, but doesn't affect user experience)

#### Pattern 3: Batch Logging (Queue and Flush)

**Use when**: High-volume searches (>1000/sec), cost optimization critical

```python
import queue
import threading

class BatchQueryLogger:
    def __init__(self, cosmos_container, batch_size=100, flush_interval=5):
        self.container = cosmos_container
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.queue = queue.Queue()
        
        # Background thread flushes queue periodically
        self.flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.flush_thread.start()
    
    def log_query(self, query_text, results):
        """Add query to batch (non-blocking)."""
        log_doc = create_query_log(query_text, results)
        self.queue.put(log_doc)
    
    def _flush_loop(self):
        """Flush queue to Cosmos DB periodically."""
        while True:
            time.sleep(self.flush_interval)
            self._flush_batch()
    
    def _flush_batch(self):
        """Write batched logs to Cosmos DB."""
        batch = []
        while not self.queue.empty() and len(batch) < self.batch_size:
            batch.append(self.queue.get())
        
        if batch:
            # Write all logs in batch (could optimize with bulk executor)
            for log_doc in batch:
                try:
                    self.container.create_item(body=log_doc)
                except:
                    # Re-queue failed items
                    self.queue.put(log_doc)

# Usage
logger = BatchQueryLogger(cosmos_container)

def search_with_batched_logging(query_text):
    results = search_client.search(query_text)
    logger.log_query(query_text, results)  # Non-blocking, queued
    return results
```

**Pros**: Zero search latency impact, handles traffic spikes
**Cons**: Logs delayed by 5 seconds, more complex code
**RU cost**: Same 5 RU/search, but smoother throughput consumption

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

## Best Practices for Search Analytics

### 1. Partition Key Design

**✅ DO**:
- Choose partition keys with 100+ unique values (e.g., `search_service`, `experiment_id`)
- Ensure even data distribution across partitions
- Align partition key with your most common query filters
- Use hierarchical partition keys for high-volume containers (>20GB per partition)

**❌ DON'T**:
- Use low-cardinality keys like `index_name` (only 5-10 values) → hot partitions
- Use skewed keys like `query_text` (popular queries dominate) → throttling
- Use keys that don't match query patterns (forces cross-partition queries)

**Example hot partition scenario**:
```python
# Bad: partition by result count (most queries return 10 results)
partition_key = "/results/count"
# Result: 90% of documents in partition "10" → hot partition, throttling

# Good: partition by search service (even distribution)
partition_key = "/search_service"
# Result: prod-search (40%), dev-search (30%), test-search (30%) → balanced
```

### 2. Schema Evolution Strategy

**✅ DO**:
- Add optional fields, never require them (old documents still valid)
- Use nested objects for related fields (`performance.*`, `results.*`)
- Include `schema_version` field for future migrations
- Document your schema in code comments and team wiki

**❌ DON'T**:
- Change field data types (string → number breaks existing queries)
- Remove fields that old code depends on (breaks backward compatibility)
- Store huge arrays (>100 elements) without pagination

**Schema versioning example**:
```python
# Version 1 documents
{"id": "log-001", "schema_version": 1, "query_text": "laptop"}

# Version 2 documents (added performance tracking)
{"id": "log-002", "schema_version": 2, "query_text": "mouse", "performance": {...}}

# Queries handle both versions
query = """
SELECT c.query_text, c.performance.duration_ms ?? 0 as latency
FROM c
WHERE c.timestamp > @cutoff
```

### 3. Indexing Policy Optimization

**✅ DO**:
- Exclude large, unqueried fields from indexing (`results.top_results`, `metadata`)
- Create composite indexes for frequently used ORDER BY queries
- Use range indexes (default) for timestamp and numeric fields
- Monitor RU consumption and adjust indexing based on actual query patterns

**❌ DON'T**:
- Index everything (default behavior wastes RUs on every write)
- Create composite indexes for rarely-used queries (adds write overhead)
- Over-index (100+ indexed paths slows writes and increases storage)

**Indexing decision matrix**:
| Field | Queried Frequency | Include in Index? | Reason |
|-------|------------------|-------------------|--------|
| `query_text` | Every analytics query | ✅ Yes | Filtered frequently |
| `timestamp` | Every analytics query | ✅ Yes | Filtered + sorted frequently |
| `user_id` | 20% of queries | ✅ Yes | Used for user-specific analytics |
| `results.top_results` | Never filtered by | ❌ No | Retrieved whole, not filtered |
| `metadata.*` | 5% of queries | ❌ No | Variable schema, rare filters |

### 4. Query Performance Optimization

**✅ DO**:
- Specify partition key in queries whenever possible (10× faster)
- Use `SELECT` projection to retrieve only needed fields
- Paginate results with `OFFSET`/`LIMIT` (don't retrieve 10K+ documents)
- Run aggregations in Cosmos DB (AVG, COUNT, SUM) not in application code

**❌ DON'T**:
- Use `enable_cross_partition_query=True` without filtering by partition key
- Retrieve entire documents when you only need 2-3 fields
- Run `SELECT *` for large result sets
- Fetch all results and aggregate in Python (network + CPU waste)

### 5. Cost Management

**✅ DO**:
- Use serverless for workloads <10K requests/day
- Share throughput across containers when usage patterns don't overlap
- Set aggressive TTL on query logs (30 days typical, 7 days for high-volume)
- Monitor RU consumption per operation type and optimize expensive queries
- Use Eventual consistency for non-critical analytics (50% RU savings on reads)

**❌ DON'T**:
- Provision dedicated throughput for every container (wastes money)
- Keep query logs forever without archival strategy
- Use Strong consistency for all queries (2× RU cost)
- Ignore hot partitions (leads to over-provisioning)

### 6. Multi-Region Considerations

**✅ DO**:
- Replicate to user regions for low-latency writes (Europe, Asia)
- Use multi-region writes for global applications (each region writes locally)
- Configure automatic failover for high availability
- Use Session consistency for regional write/read accuracy

**❌ DON'T**:
- Replicate to every Azure region (unnecessary storage cost)
- Use Strong consistency globally (high latency + 2× RU cost)
- Enable multi-region writes for single-region applications (adds 1 RU/write)

### 7. Security and Compliance

**✅ DO**:
- Use managed identity for authentication (avoid connection strings in code)
- Enable Azure Private Link for VNet-isolated Cosmos DB access
- Set TTL based on compliance requirements (GDPR: 30-90 days for logs)
- Encrypt sensitive fields at application level before storing
- Use Azure Key Vault for storing Cosmos DB keys

**❌ DON'T**:
- Hardcode connection strings in source code
- Store PII (personally identifiable information) without encryption
- Disable firewall rules in production
- Use same Cosmos DB account for dev/test and production

---

## Troubleshooting Common Issues

### Issue 1: High RU Consumption / Throttling (429 Errors)

**Symptoms**:
```
azure.cosmos.exceptions.CosmosHttpResponseError: (429) Request rate too large
```

**Causes**:
1. **Hot partition**: One partition consumes all RUs while others idle
2. **Inefficient queries**: Cross-partition queries without filters
3. **Under-provisioned throughput**: Actual load exceeds provisioned RU/s
4. **Large documents**: Writing 10KB documents costs 50 RU each

**Solutions**:

**Check partition distribution**:
```sql
-- Find partition sizes
SELECT c.search_service, COUNT(1) as doc_count
FROM c
GROUP BY c.search_service
```
- **If skewed**: Redesign partition key (e.g., use hierarchical partition key)

**Optimize expensive queries**:
```python
# Before: Cross-partition query (50+ RUs)
query = "SELECT * FROM c WHERE c.query_text = 'laptop'"
results = container.query_items(query, enable_cross_partition_query=True)

# After: Single-partition query (2-5 RUs)
query = "SELECT * FROM c WHERE c.search_service = @service AND c.query_text = @text"
results = container.query_items(query, partition_key="prod-search")
```

**Increase throughput** (temporary fix):
```bash
az cosmosdb sql database throughput update \
  --account-name $ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --name $DATABASE_NAME \
  --throughput 800  # Double from 400 to 800 RU/s
```

**Reduce document size**:
```python
# Before: 5KB document (25 RU/write)
log = {"query_text": "...", "results": [full_details_array]}

# After: 2KB document (10 RU/write)
log = {"query_text": "...", "results": [{"id": "...", "score": 0.85}]}
```

### Issue 2: Slow Analytics Queries (>5 seconds)

**Symptoms**:
- Dashboard queries take 10-30 seconds to load
- Simple COUNT queries consume 50+ RUs

**Causes**:
1. **Missing composite indexes**: ORDER BY multiple fields without composite index
2. **Cross-partition queries**: Querying without partition key
3. **Large result sets**: Retrieving 10K+ documents without pagination
4. **Unoptimized indexing**: Indexing huge arrays unnecessarily

**Solutions**:

**Add composite indexes**:
```python
# Query that's slow
query = "SELECT * FROM c ORDER BY c.timestamp DESC, c.query_text ASC"

# Add composite index
indexing_policy = {
    "compositeIndexes": [
        [
            {"path": "/timestamp", "order": "descending"},
            {"path": "/query_text", "order": "ascending"}
        ]
    ]
}
# Result: 50 RU → 5 RU query cost
```

**Use pagination**:
```python
# Before: Retrieve all 50K documents (100+ RUs, 30+ seconds)
query = "SELECT * FROM c"
results = list(container.query_items(query))

# After: Retrieve first 100 (5 RUs, <1 second)
query = "SELECT * FROM c ORDER BY c.timestamp DESC OFFSET 0 LIMIT 100"
results = list(container.query_items(query))
```

**Optimize indexing policy**:
```python
# Exclude large arrays from indexing
"excludedPaths": [
    {"path": "/results/top_results/*"},  # Large array, never filtered
    {"path": "/metadata/*"}               # Variable schema, rarely queried
]
# Result: 10 RU → 5 RU write cost, faster reads
```

### Issue 3: Data Not Appearing in Queries

**Symptoms**:
- Document created successfully, but not returned in queries
- Analytics showing 0 results for recent logs

**Causes**:
1. **Wrong partition key**: Document written to different partition than query targets
2. **TTL expiration**: Document expired before query executed
3. **Consistency level**: Using Eventual consistency, data not yet replicated
4. **Indexing delay**: Document not yet indexed (rare with consistent indexing)

**Solutions**:

**Verify partition key**:
```python
# Write document
doc = {"id": "log-001", "search_service": "prod-search", "query_text": "laptop"}
container.create_item(body=doc)

# Query must match partition key
query = "SELECT * FROM c WHERE c.id = 'log-001'"
results = list(container.query_items(
    query=query,
    partition_key="prod-search"  # Must match!
))
```

**Check TTL**:
```python
# Document with TTL
doc = {"id": "log-001", "ttl": 60}  # Expires in 60 seconds
container.create_item(body=doc)

# If queried 61 seconds later → document gone
time.sleep(61)
results = container.read_item(item="log-001", partition_key="...")  # Not found
```

**Use higher consistency level**:
```python
# Write to Cosmos DB
container.create_item(body=doc)

# Read immediately with Session consistency (default)
result = container.read_item(item=doc['id'], partition_key=doc['search_service'])
# ✅ Works: Session consistency sees own writes immediately

# Read with Eventual consistency
result = container.read_item(
    item=doc['id'],
    partition_key=doc['search_service'],
    consistency_level='Eventual'
)
# ❌ May fail: Data not yet replicated in multi-region setup
```

### Issue 4: High Storage Costs

**Symptoms**:
- Storage costs growing unexpectedly ($10+/month for small workload)
- Database size increasing despite TTL configuration

**Causes**:
1. **TTL not enabled**: Old documents never deleted
2. **TTL not working**: Documents missing `ttl` field or container default TTL disabled
3. **Large documents**: Storing unnecessary data (full product descriptions, images)
4. **Analytical storage enabled**: Paying for both transactional + analytical storage

**Solutions**:

**Enable container-level TTL**:
```python
# Check if TTL enabled
container_properties = container.read()
print(container_properties['defaultTtl'])  # Should be 2592000 (30 days), not None

# Enable TTL if disabled
container_properties['defaultTtl'] = 2592000  # 30 days
database.replace_container(container=container, partition_key=..., default_ttl=2592000)
```

**Verify documents have TTL**:
```python
# Query documents to check TTL field
query = "SELECT c.id, c.ttl, c.timestamp FROM c"
results = list(container.query_items(query))

for doc in results:
    if 'ttl' not in doc or doc['ttl'] == -1:
        print(f"Document {doc['id']} will never expire!")
```

**Reduce document size**:
```python
# Check average document size
import sys

query = "SELECT TOP 100 * FROM c"
results = list(container.query_items(query))
total_size = sum(sys.getsizeof(str(doc)) for doc in results)
avg_size = total_size / len(results)
print(f"Average document size: {avg_size/1024:.2f} KB")

# If >3KB, review what data is unnecessary and remove it
```

**Disable analytical storage if unused**:
```bash
# Check if analytical storage is enabled
az cosmosdb show --name $ACCOUNT_NAME --resource-group $RESOURCE_GROUP \
  --query "enableAnalyticalStorage"

# If true but you're not using Azure Synapse Link, you're paying for both storage types
# Note: Can't disable after enabling, must create new account
```

---

## Next Steps

- **[Azure Monitor & Logging](./07-azure-monitor-logging.md)** - Set up comprehensive monitoring and alerting for your search analytics pipeline
- **[A/B Testing Framework](./17-ab-testing-framework.md)** - Use Cosmos DB evaluation results to run controlled experiments
- **[Cost Analysis](./19-cost-analysis.md)** - Deep dive into optimizing costs across all Azure services
- **[Multi-Region Deployment](./20-multi-region-deployment.md)** - Expand your analytics to global scale with multi-region Cosmos DB

---

*See also: [Azure AI Search Setup](./04-azure-ai-search-setup.md) | [Query Optimization](./12-query-optimization.md) | [Change Feed Processing](https://learn.microsoft.com/azure/cosmos-db/change-feed)*