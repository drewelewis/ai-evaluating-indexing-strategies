# Azure Monitor & Logging

Complete guide to monitoring, logging, and diagnostics for Azure AI Search using Azure Monitor, Application Insights, and custom analytics.

## 📋 Table of Contents
- [Overview](#overview)
- [Azure Monitor Setup](#azure-monitor-setup)
- [Diagnostic Logging](#diagnostic-logging)
- [Metrics and KPIs](#metrics-and-kpis)
- [KQL Queries](#kql-queries)
- [Dashboards](#dashboards)
- [Alerting](#alerting)
- [Application Insights Integration](#application-insights-integration)
- [Custom Logging](#custom-logging)

---

## Overview

### Why Azure Monitor for Search?

**Scenario**: Your e-commerce platform's search is experiencing intermittent slowdowns. Users complain that searches for "wireless headphones" sometimes take 5+ seconds, while other queries return instantly. Product managers want to know which queries are most popular, developers need to identify performance bottlenecks, and executives want SLA compliance reports.

**The challenge**: Without proper monitoring, you're flying blind:
- **Performance issues go undetected** until users complain
- **Root cause analysis takes hours** (which query? which index? which time?)
- **No proactive alerting** for degraded performance or errors
- **Manual log analysis** is time-consuming and incomplete

**The solution**: Azure Monitor provides end-to-end observability for your Azure AI Search service.

#### What Azure Monitor Provides

Azure Monitor provides comprehensive observability across three key pillars:

**1. Real-Time Metrics (Performance Monitoring)**
- **Query latency**: P50, P95, P99 latency in milliseconds
- **Throughput**: Queries per second (QPS), indexing rate
- **Error rates**: Failed queries, throttling percentage
- **Resource utilization**: Storage consumption, replica health
- **Granularity**: 1-minute intervals for near real-time visibility

**Use case**: Dashboard showing current P95 latency (120ms) with threshold alert at 500ms

**2. Diagnostic Logs (Detailed Query Data)**
- **Query execution logs**: Every search query, filters used, results returned
- **Indexing logs**: Documents indexed, failed, deleted
- **Operation logs**: Index creation, deletion, updates
- **Performance data**: Duration, result count, client request ID
- **Retention**: 30-90 days in Log Analytics (configurable)

**Use case**: KQL query to find "show me all searches for 'laptop' that took >1 second in the last 24 hours"

**3. Alerting (Proactive Issue Detection)**
- **Metric-based alerts**: Trigger when latency >500ms for 5 minutes
- **Log-based alerts**: Trigger when error rate >5% in 10-minute window
- **Action groups**: Email, SMS, webhook, Azure Function, Logic App
- **Severity levels**: Critical, Error, Warning, Information
- **Smart detection**: ML-based anomaly detection (Application Insights)

**Use case**: SMS alert to on-call engineer when throttling >10% for 5 minutes

#### Why Azure Monitor vs Alternatives

**Azure Monitor** (Recommended for Azure AI Search):
- ✅ Native integration (diagnostic settings enable with one click)
- ✅ Rich query language (KQL) for complex analytics
- ✅ Real-time metrics with 1-minute granularity
- ✅ Integrated alerting with action groups
- ✅ Cost-effective for moderate volumes (<100GB/month ingestion)
- ✅ Unified platform for all Azure resources

**Application Insights** (Complement, not replacement):
- ✅ End-to-end transaction tracing (search query → API call → database)
- ✅ Dependency tracking (Azure Search latency as part of request)
- ✅ User session tracking (which user ran which searches)
- ❌ Doesn't capture Azure Search resource logs automatically
- **Best used together**: Application Insights for app telemetry + Azure Monitor for Search resource logs

**Azure Cosmos DB** (For custom analytics):
- ✅ Store logs for >90 days with custom schema
- ✅ Real-time queries with low latency
- ❌ Requires custom code to ingest logs (Change Feed from Monitor)
- ❌ Higher cost for high-volume logging
- **Use case**: Long-term query analytics, A/B test results storage (see [Cosmos DB Analytics](./06-azure-cosmos-db-analytics.md))

**Third-party tools** (Datadog, Splunk, Elastic):
- ✅ Cross-cloud monitoring (if you use AWS, GCP, on-prem)
- ✅ Advanced ML-based anomaly detection
- ❌ Higher cost (typically 2-3× Azure Monitor)
- ❌ Requires additional integration setup
- **Use case**: Multi-cloud environments, advanced analytics requirements

### Monitoring Architecture

**The complete observability stack**: How data flows from Azure AI Search to insights and actions.

```
┌─────────────────────┐
│  Azure AI Search    │ ← Resource generating metrics and logs
│  (Resource Logs)    │    • Query operations
│  - Query.Search     │    • Index operations  
│  - Index.Create     │    • Throttling events
│  - Document.Index   │
└──────────┬──────────┘
           │ 1. Diagnostic Settings enabled
           │    (sends logs & metrics)
           ▼
┌─────────────────────────────┐
│  Azure Monitor              │ ← Centralized monitoring platform
│  - Metrics Database         │    Stores time-series metrics
│  - Diagnostic Logs          │    Receives raw logs
│  - Activity Logs            │    Management operations
└──────────┬──────────────────┘
           │
           ├──► 2a. Metrics → Metrics Explorer
           │                  • Real-time charts
           │                  • QPS, latency, throttling
           │                  • 1-minute granularity
           │                  • 93-day retention
           │
           ├──► 2b. Logs → Log Analytics Workspace
           │                • KQL query engine
           │                • Full-text search on logs
           │                • Join, aggregate, analyze
           │                • 30-365 day retention
           │
           ├──► 2c. Logs → Application Insights
           │                • Distributed tracing
           │                • User session correlation
           │                • Dependency tracking
           │                • Smart detection (ML)
           │
           └──► 2d. Logs → Event Hub (optional)
                            • Real-time streaming
                            • Custom processing
                            • Integration with SIEM
                            • Archive to Blob Storage
           
           ▼
┌─────────────────────────────┐
│  3. Alerts & Actions        │ ← Proactive monitoring
│  - Metric Alerts            │
│    • Latency >500ms         │
│    • Throttling >10%        │
│  - Log Query Alerts         │
│    • Error rate >5%         │
│    • Zero-result spike      │
│  - Action Groups            │
│    → Email: dev-team@...    │
│    → SMS: +1-555-...        │
│    → Webhook: PagerDuty     │
│    → Azure Function         │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  4. Visualization           │ ← Insights and reporting
│  - Azure Dashboards         │    • Executive overview
│  - Workbooks                │    • Interactive analysis
│  - Power BI                 │    • Custom reports
│  - Grafana                  │    • Cross-platform dashboards
└─────────────────────────────┘
```

#### Architecture Flow Explained

**Step 1: Data Collection**
- **Diagnostic settings**: One-time configuration on Azure AI Search service
- **What's collected**: 
  - Metrics: Latency, QPS, throttling (automatically, no config needed)
  - Logs: Query details, indexing operations, errors (requires diagnostic settings)
- **Frequency**: Metrics every 1 minute, logs in near real-time (<30 seconds delay)

**Step 2a: Metrics Analysis**
- Stored in Azure Monitor Metrics database (separate from logs)
- **Retention**: 93 days (automatic, no cost for storage)
- **Query**: Metrics Explorer UI or Azure Monitor API
- **Use case**: "Show me P95 latency for the last 7 days"

**Step 2b: Log Analytics**
- Logs sent to Log Analytics workspace
- **Storage**: Columnar format optimized for KQL queries
- **Retention**: 30 days default (configurable to 730 days)
- **Cost**: $2.76/GB ingested (first 5GB/month free per workspace)
- **Use case**: "Show me all queries for 'laptop' that returned zero results"

**Step 2c: Application Insights**
- Optional integration for application-level telemetry
- **Correlation**: Link search queries to user sessions, API requests
- **Smart detection**: ML-based anomaly detection
- **Use case**: "Show me the entire request chain when search took >2 seconds"

**Step 2d: Event Hub Streaming** (Advanced)
- Real-time log streaming to external systems
- **Use cases**: 
  - SIEM integration (Splunk, QRadar)
  - Long-term archival to Azure Blob Storage ($0.01/GB/month)
  - Custom real-time processing (Azure Stream Analytics)

**Step 3: Alerting**
- Rules evaluate metrics/logs every 1-5 minutes
- **Trigger conditions**: Latency threshold, error rate spike, throttling detected
- **Actions**: Email, SMS, webhook, runbook, Logic App, Function
- **Smart grouping**: Combine related alerts to reduce noise

**Step 4: Visualization**
- **Azure Dashboards**: Pin charts from Metrics Explorer and Log Analytics
- **Workbooks**: Interactive analysis with parameters (time range, index name)
- **Power BI**: Business intelligence reports with drill-down
- **Grafana**: Open-source dashboards (requires plugin)

#### Monitoring Strategy Decision Matrix

| Requirement | Solution | Why |
|------------|----------|-----|
| Real-time latency monitoring | Metrics Explorer | 1-min granularity, no query cost |
| "Which queries are slow?" | Log Analytics + KQL | Full query details retained |
| Alert on P95 latency >500ms | Metric Alert | Fast evaluation, low latency |
| Alert on error rate >5% | Log Query Alert | Flexible logic (errors/total) |
| Executive SLA report | Azure Dashboard | Shareable, embeddable charts |
| Interactive investigation | Workbooks | Parameterized queries |
| Long-term trend analysis (>93 days) | Log Analytics + Archive | Metrics retained 93 days only |
| Correlate search with API calls | Application Insights | Distributed tracing |
| Feed to SIEM (Splunk, etc.) | Event Hub streaming | Real-time export |

Azure Monitor provides comprehensive observability across three key pillars:
- **Real-time metrics**: Query latency, throughput, error rates
- **Diagnostic logs**: Detailed query execution data
- **Alerting**: Proactive issue detection
- **Dashboards**: Visual performance tracking
- **Integration**: Seamless with Azure AI Search

### Monitoring Architecture

```
┌─────────────────────┐
│  Azure AI Search    │
│  (Resource Logs)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────┐
│  Azure Monitor              │
│  - Metrics                  │
│  - Diagnostic Logs          │
│  - Activity Logs            │
└──────────┬──────────────────┘
           │
           ├──► Log Analytics Workspace
           │    (KQL Queries)
           │
           ├──► Application Insights
           │    (APM & Telemetry)
           │
           └──► Alerts & Dashboards
```

---

## Azure Monitor Setup

### Understanding Log Analytics Pricing

Before creating a workspace, understand the cost model to avoid surprises:

**Pricing tiers**:

| Tier | Cost | Best For | Commitment |
|------|------|----------|------------|
| **Pay-as-you-go** | $2.76/GB ingested | Variable workloads, <100GB/month | None |
| **Commitment: 100GB/day** | $196/day = $1.96/GB | Predictable >100GB/month ingestion | 31 days |
| **Commitment: 200GB/day** | $360/day = $1.80/GB | High volume >200GB/month | 31 days |

**Typical Azure AI Search ingestion volumes**:

| Search Traffic | Log Volume/Day | Cost/Month (Pay-as-you-go) |
|----------------|----------------|---------------------------|
| 10K queries/day | 50MB | $4.14/month |
| 100K queries/day | 500MB | $41.40/month |
| 1M queries/day | 5GB | $414/month |
| 10M queries/day | 50GB | $4,140/month (consider commitment tier) |

**Cost optimization strategies**:
1. **Set retention to 30 days** (not 90) if you don't need historical analysis
2. **Filter logs**: Don't log every query if volume is high (sample 10%)
3. **Use Basic logs** for rarely-queried data ($0.50/GB vs $2.76/GB)
4. **Archive to storage**: Export old logs to Blob Storage ($0.01/GB/month)

### Step 1: Create Log Analytics Workspace

#### Using Azure CLI

```bash
# Variables
RESOURCE_GROUP="rg-search-evaluation"
LOCATION="eastus"
WORKSPACE_NAME="law-search-analytics"

# Create Log Analytics Workspace
az monitor log-analytics workspace create \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $WORKSPACE_NAME \
  --location $LOCATION \
  --retention-time 30

# Get workspace ID
WORKSPACE_ID=$(az monitor log-analytics workspace show \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $WORKSPACE_NAME \
  --query customerId \
  --output tsv)

echo "Workspace ID: $WORKSPACE_ID"
```

#### Configuration Decisions Explained

**1. `--retention-time 30`**
- **What it means**: Logs are automatically deleted after 30 days
- **Cost impact**: Retention beyond 30 days costs $0.12/GB/month
  - Example: 10GB logs × 90 days retention = 10GB × 60 extra days × $0.12 = $72/month
- **When to increase**: 
  - Compliance requirements (SOC 2, HIPAA may require 90-365 days)
  - Long-term trend analysis (year-over-year query patterns)
  - Incident investigation (need 60+ days of history)
- **Alternative**: Archive to Blob Storage after 30 days (much cheaper for cold data)

**2. `--location eastus`**
- **Latency impact**: Logs sent from Search service to workspace
  - Same region: <50ms ingestion latency
  - Cross-region: 100-300ms ingestion latency (doesn't affect search performance)
- **Cost impact**: No data transfer charges within same region
- **Best practice**: Place workspace in same region as Search service

**3. Pricing tier** (not specified = Pay-as-you-go)
- **Default**: Pay-as-you-go ($2.76/GB)
- **Change to commitment tier**:
  ```bash
  az monitor log-analytics workspace update \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $WORKSPACE_NAME \
    --capacity-reservation-level 100  # 100GB/day commitment
  ```
- **When to switch**: If you're consistently ingesting >100GB/month

#### Using Python SDK

```python
from azure.mgmt.loganalytics import LogAnalyticsManagementClient
from azure.identity import DefaultAzureCredential
from azure.mgmt.loganalytics.models import Workspace

class LogAnalyticsSetup:
    """Setup Log Analytics for Azure Search monitoring."""
    
    def __init__(self, subscription_id, resource_group):
        self.credential = DefaultAzureCredential()
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.client = LogAnalyticsManagementClient(
            credential=self.credential,
            subscription_id=subscription_id
        )
    
    def create_workspace(self, workspace_name, location, retention_days=30):
        """Create Log Analytics workspace."""
        workspace_params = Workspace(
            location=location,
            retention_in_days=retention_days,
            sku={
                'name': 'PerGB2018'  # Pay-as-you-go pricing
            }
        )
        
        workspace = self.client.workspaces.begin_create_or_update(
            resource_group_name=self.resource_group,
            workspace_name=workspace_name,
            parameters=workspace_params
        ).result()
        
        print(f"✅ Workspace created: {workspace.name}")
        print(f"   Workspace ID: {workspace.customer_id}")
        
        return workspace

# Usage
import os

setup = LogAnalyticsSetup(
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group="rg-search-evaluation"
)

workspace = setup.create_workspace(
    workspace_name="law-search-analytics",
    location="eastus",
    retention_days=30
)
```

### Step 2: Enable Diagnostic Settings

**Critical step**: Diagnostic settings tell Azure Monitor *what* to collect and *where* to send it.

**Before enabling**: Azure AI Search generates metrics automatically, but **logs are NOT collected** until diagnostic settings are configured.

```bash
# Variables
SEARCH_SERVICE="search-prod"
WORKSPACE_ID="/subscriptions/{sub-id}/resourceGroups/{rg}/providers/Microsoft.OperationalInsights/workspaces/law-search-analytics"

# Enable diagnostic settings
az monitor diagnostic-settings create \
  --name search-diagnostics \
  --resource "/subscriptions/{subscription-id}/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Search/searchServices/$SEARCH_SERVICE" \
  --workspace $WORKSPACE_ID \
  --logs '[
    {
      "category": "OperationLogs",
      "enabled": true,
      "retentionPolicy": {
        "enabled": true,
        "days": 30
      }
    }
  ]' \
  --metrics '[
    {
      "category": "AllMetrics",
      "enabled": true,
      "retentionPolicy": {
        "enabled": true,
        "days": 30
      }
    }
  ]'
```

#### Diagnostic Settings Decisions

**1. Log Categories**
- **OperationLogs**: Only category available for Azure AI Search
  - Contains: Query.Search, Index.Create, Document.Index, etc.
  - **Enable**: Yes (this is why you're setting up monitoring)

**2. Metrics Categories**
- **AllMetrics**: Latency, QPS, throttling, storage, document count
  - **Enable**: Yes (low volume, high value)
  - **Cost**: Metrics sent to workspace cost ~$0.10/month (negligible)
  - **Why enable**: Allows KQL queries joining metrics + logs (e.g., "show queries when latency was >500ms")

**3. Retention Policy**
- **Days: 30**: How long logs stay in Log Analytics workspace
  - **Workspace retention**: 30 days (from Step 1)
  - **Diagnostic retention**: 30 days (must be ≤ workspace retention)
  - **Why they differ**: Diagnostic retention is deprecated (use workspace retention)
  - **Best practice**: Set both to same value to avoid confusion

**4. Destinations** (where to send logs)

You can send logs to multiple destinations simultaneously:

| Destination | Use Case | Cost | Retention |
|-------------|----------|------|-----------|
| **Log Analytics** | KQL queries, alerting, dashboards | $2.76/GB ingested | 30-730 days |
| **Storage Account** | Long-term archive, compliance | $0.01/GB/month (cool tier) | Unlimited |
| **Event Hub** | Real-time streaming, SIEM integration | $0.028/million events | N/A (streaming) |
| **Partner Solution** | Datadog, Elastic, Splunk | Varies | Varies |

**Typical configuration**:
```bash
# Send to both Log Analytics (analysis) and Storage (archive)
az monitor diagnostic-settings create \
  --name search-diagnostics \
  --resource $SEARCH_RESOURCE_ID \
  --workspace $WORKSPACE_ID \
  --storage-account $STORAGE_ACCOUNT_ID \
  --logs '[{"category": "OperationLogs", "enabled": true}]' \
  --metrics '[{"category": "AllMetrics", "enabled": true}]'
```

**Cost example** (100K queries/day):
- Log Analytics: 500MB/day × 30 days × $2.76/GB = $41.40/month
- Storage archive: 500MB/day × 365 days × $0.01/GB/month = $5.48/year (for 1 year archive)
- **Total**: $41.40/month + $0.46/month = $41.86/month

#### Using Python

```python
from azure.mgmt.monitor import MonitorManagementClient
from azure.mgmt.monitor.models import DiagnosticSettingsResource, LogSettings, MetricSettings

class DiagnosticSettingsSetup:
    """Configure diagnostic settings for Azure Search."""
    
    def __init__(self, subscription_id, credential):
        self.client = MonitorManagementClient(credential, subscription_id)
    
    def enable_diagnostics(self, resource_id, workspace_id, retention_days=30):
        """Enable diagnostic settings for search service."""
        diagnostic_settings = DiagnosticSettingsResource(
            workspace_id=workspace_id,
            logs=[
                LogSettings(
                    category='OperationLogs',
                    enabled=True,
                    retention_policy={
                        'enabled': True,
                        'days': retention_days
                    }
                )
            ],
            metrics=[
                MetricSettings(
                    category='AllMetrics',
                    enabled=True,
                    retention_policy={
                        'enabled': True,
                        'days': retention_days
                    }
                )
            ]
        )
        
        result = self.client.diagnostic_settings.create_or_update(
            resource_uri=resource_id,
            name='search-diagnostics',
            parameters=diagnostic_settings
        )
        
        print(f"✅ Diagnostic settings enabled")
        return result

# Usage
from azure.identity import DefaultAzureCredential

diag_setup = DiagnosticSettingsSetup(
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    credential=DefaultAzureCredential()
)

search_resource_id = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Search/searchServices/{search_service}"
workspace_id = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.OperationalInsights/workspaces/law-search-analytics"

diag_setup.enable_diagnostics(search_resource_id, workspace_id)
```

---

## Diagnostic Logging

Once diagnostic settings are enabled, Azure AI Search begins sending logs to your Log Analytics workspace. Understanding what's logged, how to filter it, and how to optimize logging costs is critical for effective monitoring.

### What Gets Logged

Azure AI Search **OperationLogs** category captures all operations:

| Operation Type | Description | Typical Volume | Example Use Case |
|----------------|-------------|----------------|------------------|
| **Query.Search** | Search query execution | 95% of logs | Find slow queries, analyze search patterns |
| **Document.Index** | Document indexing operations | 3% of logs | Track indexing failures, measure indexing speed |
| **Index.Create** | Index creation/modification | <1% of logs | Audit schema changes |
| **Index.Delete** | Index deletion | <1% of logs | Audit compliance |
| **Indexer.Run** | Indexer execution | <1% of logs | Debug data source connectivity |
| **Skillset.Execute** | AI enrichment pipeline | <1% of logs | Debug enrichment failures |

**Real-world example** (e-commerce site, 1M queries/day):
- **Query.Search**: 950,000 logs/day (95%)
- **Document.Index**: 30,000 logs/day (3%)
- **Other operations**: 20,000 logs/day (2%)
- **Total**: 1,000,000 logs/day ≈ 500 MB/day

**Cost impact**: 500 MB/day × 30 days × $2.76/GB = **$41.40/month**

### Log Schema

Each log entry contains:

```json
{
  "time": "2024-01-15T10:23:45.123Z",
  "resourceId": "/subscriptions/{sub-id}/resourceGroups/{rg}/providers/Microsoft.Search/searchServices/search-prod",
  "operationName": "Query.Search",
  "operationVersion": "2023-11-01",
  "category": "OperationLogs",
  "resultType": "Success",
  "resultSignature": "200",
  "durationMs": 145,
  "properties": {
    "Description": "GET /indexes/products/docs",
    "Query": "search=laptop&$filter=price lt 1000&$top=10",
    "IndexName": "products",
    "Documents": 10,
    "SearchMode": "any",
    "ScoringProfile": "",
    "ApiVersion": "2023-11-01"
  },
  "callerIpAddress": "40.112.233.144",
  "correlationId": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "identity": {
    "authorization": {
      "evidence": {
        "principalId": "user@domain.com",
        "principalType": "User"
      }
    }
  }
}
```

**Key fields for analysis**:
- **durationMs**: Query latency (alert if >500ms for P95)
- **resultType**: Success/Failure (calculate error rate)
- **operationName**: Query vs indexing vs admin operations
- **properties.Query**: Actual search query text (identify problematic queries)
- **properties.Documents**: Number of results returned
- **callerIpAddress**: Client IP (identify traffic patterns, potential abuse)
- **correlationId**: Trace requests across services

### Filtering Strategies

**Problem**: At high query volumes, logging everything can be expensive.

**Solution**: Filter logs before they reach Log Analytics.

#### Strategy 1: Sample Logs (Reduce Volume)

**Use case**: You have 10M queries/day (5GB/day × $2.76/GB = $414/month). You only need representative sample for analysis.

**Approach**: Log only 10% of queries randomly.

```python
import random
from azure.search.documents import SearchClient
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

# Configure sampling logger
sampling_logger = logging.getLogger('search_sampling')
sampling_logger.addHandler(AzureLogHandler(
    connection_string=f'InstrumentationKey={app_insights_key}'
))

def search_with_sampling(search_client, query, sampling_rate=0.10):
    """Execute search with probabilistic logging."""
    start_time = time.time()
    results = search_client.search(query)
    duration_ms = (time.time() - start_time) * 1000
    
    # Log only 10% of queries
    if random.random() < sampling_rate:
        sampling_logger.info(
            "Query executed",
            extra={
                'custom_dimensions': {
                    'query': query,
                    'duration_ms': duration_ms,
                    'sampled': True,
                    'sampling_rate': sampling_rate
                }
            }
        )
    
    return results
```

**Cost savings**: 5GB/day × 10% = 0.5GB/day
- Before: $414/month
- After: $41.40/month
- **Savings**: 90% ($372.60/month)

**Trade-off**: You lose detail on 90% of queries, but statistical analysis is still valid (e.g., P95 latency calculated from 10% sample is representative).

#### Strategy 2: Log Only Slow Queries

**Use case**: You only care about queries >500ms (performance troubleshooting).

```python
def search_with_conditional_logging(search_client, query, latency_threshold_ms=500):
    """Log only queries exceeding latency threshold."""
    start_time = time.time()
    results = search_client.search(query)
    duration_ms = (time.time() - start_time) * 1000
    
    # Log only slow queries
    if duration_ms > latency_threshold_ms:
        logging.warning(
            f"Slow query detected: {duration_ms}ms",
            extra={
                'custom_dimensions': {
                    'query': query,
                    'duration_ms': duration_ms,
                    'threshold_ms': latency_threshold_ms,
                    'slow_query': True
                }
            }
        )
    
    return results
```

**Cost savings**: If only 5% of queries are slow:
- Before: 5GB/day × $2.76/GB × 30 days = $414/month
- After: 5GB/day × 5% × $2.76/GB × 30 days = $20.70/month
- **Savings**: 95% ($393.30/month)

#### Strategy 3: Log Errors + Sample Success

**Use case**: Log all errors (low volume) + sample successful queries.

```python
def search_with_error_priority_logging(search_client, query, success_sampling_rate=0.10):
    """Log all errors, sample successes."""
    start_time = time.time()
    
    try:
        results = search_client.search(query)
        duration_ms = (time.time() - start_time) * 1000
        
        # Sample successful queries
        if random.random() < success_sampling_rate:
            logging.info(
                "Query succeeded",
                extra={'custom_dimensions': {
                    'query': query,
                    'duration_ms': duration_ms,
                    'result_type': 'Success'
                }}
            )
        
        return results
    
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        # Always log errors
        logging.error(
            f"Query failed: {str(e)}",
            extra={'custom_dimensions': {
                'query': query,
                'duration_ms': duration_ms,
                'error': str(e),
                'result_type': 'Failure'
            }}
        )
        raise
```

**Cost profile**:
- Errors: 0.1% of queries (1,000/day) = 0.5MB/day
- Sampled success: 10% of 999,000 queries = 99,900/day = 49.95MB/day
- **Total**: 50.45MB/day × 30 days × $2.76/GB = **$4.18/month** (99% savings)

### Log Enrichment

**Problem**: Default logs lack business context (e.g., which user, which tenant, which A/B test variant).

**Solution**: Add custom dimensions to logs.

```python
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace

# Configure OpenTelemetry with Azure Monitor
configure_azure_monitor(
    connection_string=f"InstrumentationKey={app_insights_key}"
)

tracer = trace.get_tracer(__name__)

def search_with_enrichment(search_client, query, user_id, tenant_id, experiment_variant):
    """Add business context to logs."""
    with tracer.start_as_current_span("search_query") as span:
        # Add custom attributes
        span.set_attribute("user_id", user_id)
        span.set_attribute("tenant_id", tenant_id)
        span.set_attribute("experiment_variant", experiment_variant)
        span.set_attribute("query", query)
        
        start_time = time.time()
        results = search_client.search(query)
        duration_ms = (time.time() - start_time) * 1000
        
        span.set_attribute("duration_ms", duration_ms)
        span.set_attribute("result_count", len(list(results)))
        
        return results
```

**Benefits**:
- **User analysis**: "Which users experience slow queries?"
- **Multi-tenancy**: "Which tenant is causing high load?"
- **A/B testing**: "Does variant B have higher latency than variant A?"

**KQL query example**:
```kql
traces
| where customDimensions.experiment_variant == "B"
| summarize P95Latency = percentile(customDimensions.duration_ms, 95) by bin(timestamp, 1h)
| render timechart
```

### Retention Strategy

**Question**: How long should you keep logs?

| Use Case | Retention | Why |
|----------|-----------|-----|
| **Real-time monitoring** | 7 days | Recent data for alerts, dashboards |
| **Troubleshooting** | 30 days | Investigate issues reported in last month |
| **Trend analysis** | 90 days | Seasonal patterns (quarterly comparison) |
| **Compliance/Audit** | 1-7 years | Regulatory requirements (HIPAA, PCI-DSS) |

**Cost impact** (for 500MB/day ingestion):

| Retention | Log Analytics Cost | Storage Archive Cost | Total Monthly Cost |
|-----------|-------------------|---------------------|-------------------|
| 7 days | $9.66 | - | $9.66 |
| 30 days | $41.40 | - | $41.40 |
| 90 days | $124.20 | - | $124.20 |
| 1 year | $41.40 + archive | $1.50 (365 days × 500MB × $0.01/GB) | $42.90 |

**Best practice**: Use **tiered retention**:
1. **Hot (30 days)**: Keep recent logs in Log Analytics for KQL queries
2. **Archive (1 year)**: Move older logs to Azure Storage (cool tier)
3. **Delete (>1 year)**: Remove logs unless compliance requires longer retention

**Implementation**:
```bash
# Configure Log Analytics workspace with 30-day retention
az monitor log-analytics workspace update \
  --resource-group $RESOURCE_GROUP \
  --workspace-name law-search-analytics \
  --retention-time 30

# Configure diagnostic settings to also send to storage (archive)
az monitor diagnostic-settings create \
  --name search-diagnostics-archive \
  --resource $SEARCH_RESOURCE_ID \
  --storage-account $STORAGE_ACCOUNT_ID \
  --logs '[{"category": "OperationLogs", "enabled": true}]'
```

### Available Log Categories

```python
class SearchLogCategories:
    """Azure Search diagnostic log categories."""
    
    OPERATION_LOGS = "OperationLogs"  # Query execution, indexing operations
    
    @staticmethod
    def get_log_schema():
        """Get expected log schema for each category."""
        return {
            'OperationLogs': {
                'fields': [
                    'TimeGenerated',
                    'OperationName',  # Query, Index, etc.
                    'ResultType',     # Success, Failure
                    'DurationMs',
                    'Query',
                    'IndexName',
                    'Documents',
                    'ResultSignature',
                    'Properties'
                ],
                'examples': [
                    'Query execution',
                    'Document indexing',
                    'Index creation',
                    'Suggester queries',
                    'Autocomplete requests'
                ]
            }
        }
```

### Query Log Structure

```python
# Example log entry structure
example_query_log = {
    "TimeGenerated": "2024-01-15T10:30:00.000Z",
    "OperationName": "Query.Search",
    "ResultType": "Success",
    "DurationMs": 45,
    "Query": "laptop computers",
    "IndexName": "products",
    "Documents": 15,
    "ResultSignature": "200",
    "Properties": {
        "QueryType": "simple",
        "SearchMode": "any",
        "Top": 10,
        "Skip": 0,
        "ClientRequestId": "client-guid"
    }
}
```

---

## Metrics and KPIs

Azure Monitor automatically collects metrics from Azure AI Search every minute. Understanding which metrics matter, how to interpret them, and how to track KPIs is essential for maintaining search service health.

### Built-in Metrics Overview

Azure AI Search exposes metrics across 3 categories:

#### 1. Query Performance Metrics

| Metric | Description | Target Value | Alert Threshold |
|--------|-------------|--------------|-----------------|
| **SearchLatency** | Average query latency | <200ms (P50), <500ms (P95) | >500ms sustained |
| **SearchQPS** | Queries per second | Varies by tier | >80% of tier limit |
| **ThrottledSearchQueriesPercentage** | % of queries throttled | 0% | >1% |

**Why these matter**:
- **SearchLatency**: User experience degrades above 500ms
- **SearchQPS**: Indicates load, helps with capacity planning
- **ThrottledSearchQueriesPercentage**: Indicates need to scale up

**Real-world example**:
```
Tier: Standard S1 (100 QPS limit)
Current SearchQPS: 85 QPS (85% utilization)
Action: Plan to scale to S2 (200 QPS) when utilization hits 90%
```

#### 2. Indexing Metrics

| Metric | Description | Target Value | Alert Threshold |
|--------|-------------|--------------|-----------------|
| **DocumentsProcessedCount** | Successful indexing operations | Varies | N/A (trend only) |
| **DocumentsFailedCount** | Failed indexing operations | 0 | >10 failures in 5 min |

**Why these matter**:
- **DocumentsFailedCount**: Indicates data quality issues, schema mismatches, or indexer errors
- High failure rates can prevent index updates, leading to stale data

#### 3. Resource Metrics

| Metric | Description | Target Value | Alert Threshold |
|--------|-------------|--------------|-----------------|
| **StorageSize** | Total index size (GB) | <80% of tier limit | >90% of tier limit |
| **DocumentCount** | Total indexed documents | <tier limit | >90% of tier limit |

**Tier limits example** (Standard S1):
- Max documents: 15 million
- Max storage: 25 GB
- Alert threshold: 20 GB (80% utilization)

### Tracking KPIs

KPIs provide measurable targets for search service performance. Here's how to track the most critical KPIs.

#### KPI 1: P95 Query Latency (SLA Compliance)

**Goal**: 95% of queries complete in <500ms.

**How to measure**: Use Log Analytics to calculate P95 latency from diagnostic logs.

```kql
// P95 latency by hour (last 7 days)
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.SEARCH"
| where OperationName == "Query.Search"
| where TimeGenerated > ago(7d)
| summarize P95Latency = percentile(DurationMs, 95) by bin(TimeGenerated, 1h)
| render timechart
```

**Interpretation**:
- **Green**: P95 <300ms (excellent)
- **Yellow**: P95 300-500ms (acceptable)
- **Red**: P95 >500ms (requires action)

**Actions when red**:
1. Identify slow queries using KQL (see KQL Queries section)
2. Review index design (reduce searchable fields, optimize scoring profiles)
3. Scale up service tier (add replicas for query performance)

#### KPI 2: Error Rate

**Goal**: <0.1% of queries result in errors.

```kql
// Error rate by hour
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.SEARCH"
| where OperationName == "Query.Search"
| where TimeGenerated > ago(7d)
| summarize 
    TotalQueries = count(),
    ErrorQueries = countif(ResultType == "ClientError" or ResultType == "ServerError"),
    ErrorRate = 100.0 * countif(ResultType == "ClientError" or ResultType == "ServerError") / count()
    by bin(TimeGenerated, 1h)
| render timechart
```

**Common error types**:
- **ClientError (400)**: Invalid query syntax, missing index
- **ServerError (500)**: Search service overload, transient failures
- **Throttling (503)**: QPS limit exceeded

#### KPI 3: Storage Utilization

**Goal**: Keep storage <80% of tier limit to allow for growth.

```python
# Calculate storage utilization percentage
tier_limits = {
    'Basic': 2 * 1024 * 1024 * 1024,    # 2 GB in bytes
    'S1': 25 * 1024 * 1024 * 1024,      # 25 GB
    'S2': 100 * 1024 * 1024 * 1024,     # 100 GB
    'S3': 200 * 1024 * 1024 * 1024      # 200 GB
}

current_size_bytes = metrics_client.query_metrics('StorageSize')
tier = 'S1'
utilization_pct = (current_size_bytes / tier_limits[tier]) * 100

if utilization_pct > 80:
    print(f"⚠️ Storage at {utilization_pct:.1f}% - consider scaling")
```

**Actions at different thresholds**:
- **<50%**: Normal operation
- **50-80%**: Monitor growth rate, plan future scaling
- **80-90%**: Review index design, purge old data, or scale to larger tier
- **>90%**: Urgent action required (indexing operations may start failing)

### Built-in Metrics (Code Reference)

```python
class SearchMetrics:
    """Azure Search available metrics."""
    
    # Query Metrics
    SEARCH_LATENCY = "SearchLatency"              # Average query latency (ms)
    SEARCH_QUERIES_PER_SECOND = "SearchQPS"       # Queries per second
    THROTTLED_SEARCH_QUERIES = "ThrottledSearchQueriesPercentage"
    
    # Indexing Metrics
    INDEXING_DOCUMENTS_SUCCESS = "DocumentsProcessedCount"
    INDEXING_DOCUMENTS_FAILED = "DocumentsFailedCount"
    
    # Resource Metrics
    SEARCH_STORAGE_SIZE = "StorageSize"           # GB used
    SEARCH_DOCUMENT_COUNT = "DocumentCount"
    
    @staticmethod
    def get_all_metrics():
        """Get list of all available metrics."""
        return {
            'Query Performance': [
                'SearchLatency',
                'SearchQPS',
                'ThrottledSearchQueriesPercentage'
            ],
            'Indexing': [
                'DocumentsProcessedCount',
                'DocumentsFailedCount'
            ],
            'Resource Utilization': [
                'StorageSize',
                'DocumentCount'
            ]
        }
```

### Query Metrics Programmatically

```python
from azure.monitor.query import MetricsQueryClient, MetricAggregationType
from datetime import datetime, timedelta

class MetricsAnalyzer:
    """Query Azure Monitor metrics for Search service."""
    
    def __init__(self, credential):
        self.client = MetricsQueryClient(credential)
    
    def get_search_latency(self, resource_id, hours=24):
        """Get search latency metrics."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        response = self.client.query_resource(
            resource_uri=resource_id,
            metric_names=["SearchLatency"],
            timespan=(start_time, end_time),
            granularity=timedelta(minutes=5),
            aggregations=[MetricAggregationType.AVERAGE, MetricAggregationType.MAXIMUM]
        )
        
        results = []
        for metric in response.metrics:
            for timeseries in metric.timeseries:
                for data_point in timeseries.data:
                    results.append({
                        'timestamp': data_point.timestamp,
                        'average': data_point.average,
                        'maximum': data_point.maximum
                    })
        
        return results
    
    def get_qps(self, resource_id, hours=24):
        """Get queries per second."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        response = self.client.query_resource(
            resource_uri=resource_id,
            metric_names=["SearchQPS"],
            timespan=(start_time, end_time),
            granularity=timedelta(minutes=5),
            aggregations=[MetricAggregationType.AVERAGE, MetricAggregationType.MAXIMUM]
        )
        
        results = []
        for metric in response.metrics:
            for timeseries in metric.timeseries:
                for data_point in timeseries.data:
                    results.append({
                        'timestamp': data_point.timestamp,
                        'average_qps': data_point.average,
                        'peak_qps': data_point.maximum
                    })
        
        return results
    
    def get_throttling_percentage(self, resource_id, hours=24):
        """Get throttled query percentage."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        response = self.client.query_resource(
            resource_uri=resource_id,
            metric_names=["ThrottledSearchQueriesPercentage"],
            timespan=(start_time, end_time),
            granularity=timedelta(minutes=5),
            aggregations=[MetricAggregationType.AVERAGE]
        )
        
        throttling_data = []
        for metric in response.metrics:
            for timeseries in metric.timeseries:
                for data_point in timeseries.data:
                    if data_point.average and data_point.average > 0:
                        throttling_data.append({
                            'timestamp': data_point.timestamp,
                            'throttle_percentage': data_point.average
                        })
        
        return throttling_data
    
    def get_performance_summary(self, resource_id, hours=24):
        """Get comprehensive performance summary."""
        latency = self.get_search_latency(resource_id, hours)
        qps = self.get_qps(resource_id, hours)
        throttling = self.get_throttling_percentage(resource_id, hours)
        
        # Calculate statistics
        avg_latency = sum(d['average'] for d in latency if d['average']) / len(latency) if latency else 0
        max_latency = max((d['maximum'] for d in latency if d['maximum']), default=0)
        avg_qps = sum(d['average_qps'] for d in qps if d['average_qps']) / len(qps) if qps else 0
        peak_qps = max((d['peak_qps'] for d in qps if d['peak_qps']), default=0)
        
        return {
            'period_hours': hours,
            'latency': {
                'average_ms': avg_latency,
                'maximum_ms': max_latency,
                'p95_ms': self._calculate_percentile(latency, 'maximum', 95)
            },
            'throughput': {
                'average_qps': avg_qps,
                'peak_qps': peak_qps
            },
            'throttling': {
                'events': len(throttling),
                'max_percentage': max((d['throttle_percentage'] for d in throttling), default=0)
            }
        }
    
    def _calculate_percentile(self, data, field, percentile):
        """Calculate percentile from data."""
        import numpy as np
        values = [d[field] for d in data if d.get(field) is not None]
        if not values:
            return 0
        return np.percentile(values, percentile)

# Usage
from azure.identity import DefaultAzureCredential

analyzer = MetricsAnalyzer(DefaultAzureCredential())

resource_id = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Search/searchServices/{search_service}"

# Get 24-hour summary
summary = analyzer.get_performance_summary(resource_id, hours=24)

print(f"Average Latency: {summary['latency']['average_ms']:.2f}ms")
print(f"P95 Latency: {summary['latency']['p95_ms']:.2f}ms")
print(f"Average QPS: {summary['throughput']['average_qps']:.2f}")
print(f"Peak QPS: {summary['throughput']['peak_qps']:.2f}")
```

---

## KQL Queries

Kusto Query Language (KQL) is the primary tool for analyzing Azure AI Search logs in Log Analytics. Mastering KQL enables you to troubleshoot issues, identify trends, and optimize search performance.

### Why KQL Matters

Unlike Azure Metrics (which provide aggregated values), **KQL queries give you raw log access** to:
- Identify specific slow queries (not just average latency)
- Analyze query patterns (which searches return zero results)
- Trace user behavior (query refinement patterns)
- Debug errors (exact error messages, not just error counts)

**Example**: Metrics show "average latency 200ms." KQL reveals:
- 95% of queries < 150ms (fast)
- 5% of queries > 2000ms (very slow - due to complex filters on `description` field)
- **Action**: Optimize `description` field indexing or remove from searchable fields

### KQL Query Patterns

#### Pattern 1: Top-N Analysis

**Use case**: "What are the most popular search terms?"

```kql
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.SEARCH"
| where OperationName == "Query.Search"
| where TimeGenerated > ago(7d)
| extend QueryText = tostring(properties.Query)
| summarize QueryCount = count() by QueryText
| top 20 by QueryCount desc
```

**Actionable insight**: If "laptop" appears 10,000 times but returns zero results, you may be missing product catalog data.

#### Pattern 2: Percentile Analysis

**Use case**: "What's our P95 latency by index?"

```kql
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.SEARCH"
| where OperationName == "Query.Search"
| where TimeGenerated > ago(24h)
| summarize 
    P50 = percentile(DurationMs, 50),
    P95 = percentile(DurationMs, 95),
    P99 = percentile(DurationMs, 99),
    QueryCount = count()
    by IndexName
| order by P95 desc
```

**Actionable insight**: If `products` index has P95 > 800ms but `articles` index has P95 < 200ms, `products` index needs optimization.

#### Pattern 3: Time Series Analysis

**Use case**: "Did latency spike during the deploy?"

```kql
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.SEARCH"
| where OperationName == "Query.Search"
| where TimeGenerated > ago(7d)
| summarize P95Latency = percentile(DurationMs, 95) by bin(TimeGenerated, 1h)
| render timechart
```

**Actionable insight**: Spike from 200ms to 800ms at 2024-01-15 14:00 correlates with deployment → rollback deploy.

### Performance Optimization for KQL Queries

**Problem**: KQL queries on large log volumes (500MB/day = 15GB/month) can be slow and expensive.

#### Solution 1: Use Time Filters Aggressively

```kql
// BAD: Scans all logs (expensive)
AzureDiagnostics
| where OperationName == "Query.Search"
| summarize count()

// GOOD: Scans only last 24 hours
AzureDiagnostics
| where TimeGenerated > ago(24h)
| where OperationName == "Query.Search"
| summarize count()
```

**Impact**: 30× faster (15GB → 500MB scanned), 30× cheaper.

#### Solution 2: Filter Early in Query Pipeline

```kql
// BAD: Filters after parsing (slow)
AzureDiagnostics
| extend QueryText = tostring(properties.Query)
| where QueryText contains "laptop"

// GOOD: Filters before parsing (fast)
AzureDiagnostics
| where properties contains "laptop"
| extend QueryText = tostring(properties.Query)
```

**Impact**: 5× faster by reducing rows before expensive operations.

#### Solution 3: Limit Result Set Early

```kql
// BAD: Orders entire dataset then takes 20
AzureDiagnostics
| where TimeGenerated > ago(7d)
| order by DurationMs desc
| take 20

// GOOD: Uses top (optimized for large datasets)
AzureDiagnostics
| where TimeGenerated > ago(7d)
| top 20 by DurationMs desc
```

**Impact**: 10× faster when querying millions of rows.

### Essential KQL Queries

```python
class SearchKQLQueries:
    """KQL query library for Azure Search analysis."""
    
    @staticmethod
    def top_queries(limit=20, hours=24):
        """Most frequent queries."""
        return f"""
AzureDiagnostics
| where ResourceType == "SEARCHSERVICES"
| where OperationName == "Query.Search"
| where TimeGenerated > ago({hours}h)
| extend QueryText = tostring(parse_json(properties_s).Query)
| summarize QueryCount = count() by QueryText
| top {limit} by QueryCount desc
"""
    
    @staticmethod
    def slow_queries(threshold_ms=1000, limit=20):
        """Queries exceeding latency threshold."""
        return f"""
AzureDiagnostics
| where ResourceType == "SEARCHSERVICES"
| where OperationName == "Query.Search"
| where DurationMs > {threshold_ms}
| extend QueryText = tostring(parse_json(properties_s).Query)
| project TimeGenerated, QueryText, DurationMs, IndexName
| order by DurationMs desc
| take {limit}
"""
    
    @staticmethod
    def zero_result_queries(hours=24):
        """Queries returning no results."""
        return f"""
AzureDiagnostics
| where ResourceType == "SEARCHSERVICES"
| where OperationName == "Query.Search"
| where TimeGenerated > ago({hours}h)
| extend QueryText = tostring(parse_json(properties_s).Query)
| extend DocumentCount = toint(Documents)
| where DocumentCount == 0
| summarize QueryCount = count() by QueryText
| order by QueryCount desc
"""
    
    @staticmethod
    def error_rate_by_operation():
        """Error rate by operation type."""
        return """
AzureDiagnostics
| where ResourceType == "SEARCHSERVICES"
| where TimeGenerated > ago(24h)
| summarize 
    Total = count(),
    Errors = countif(ResultType == "Error" or ResultSignature >= 400)
    by OperationName
| extend ErrorRate = (Errors * 100.0) / Total
| order by ErrorRate desc
"""
    
    @staticmethod
    def latency_percentiles_by_index():
        """Latency percentiles per index."""
        return """
AzureDiagnostics
| where ResourceType == "SEARCHSERVICES"
| where OperationName == "Query.Search"
| where TimeGenerated > ago(24h)
| summarize 
    p50 = percentile(DurationMs, 50),
    p75 = percentile(DurationMs, 75),
    p95 = percentile(DurationMs, 95),
    p99 = percentile(DurationMs, 99),
    QueryCount = count()
    by IndexName
| order by p95 desc
"""
    
    @staticmethod
    def query_volume_timechart():
        """Query volume over time."""
        return """
AzureDiagnostics
| where ResourceType == "SEARCHSERVICES"
| where OperationName == "Query.Search"
| where TimeGenerated > ago(24h)
| summarize QueryCount = count() by bin(TimeGenerated, 5m)
| render timechart
"""
    
    @staticmethod
    def index_usage_statistics():
        """Index usage statistics."""
        return """
AzureDiagnostics
| where ResourceType == "SEARCHSERVICES"
| where OperationName == "Query.Search"
| where TimeGenerated > ago(24h)
| summarize 
    QueryCount = count(),
    AvgLatency = avg(DurationMs),
    UniqueQueries = dcount(tostring(parse_json(properties_s).Query))
    by IndexName
| order by QueryCount desc
"""
    
    @staticmethod
    def throttling_events():
        """Throttling event analysis."""
        return """
AzureDiagnostics
| where ResourceType == "SEARCHSERVICES"
| where ResultSignature == "503" or ResultType =~ "Throttled"
| where TimeGenerated > ago(24h)
| summarize ThrottleCount = count() by bin(TimeGenerated, 5m), OperationName
| render timechart
"""
```

### Query Log Analytics with Python

```python
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from datetime import timedelta

class LogAnalyticsQuerier:
    """Query Log Analytics for search insights."""
    
    def __init__(self, credential, workspace_id):
        self.client = LogsQueryClient(credential)
        self.workspace_id = workspace_id
    
    def execute_query(self, kql_query, timespan_hours=24):
        """Execute KQL query."""
        response = self.client.query_workspace(
            workspace_id=self.workspace_id,
            query=kql_query,
            timespan=timedelta(hours=timespan_hours)
        )
        
        if response.status == LogsQueryStatus.SUCCESS:
            results = []
            for table in response.tables:
                for row in table.rows:
                    row_dict = {
                        table.columns[i].name: row[i]
                        for i in range(len(table.columns))
                    }
                    results.append(row_dict)
            return results
        else:
            print(f"Query failed: {response.status}")
            return []
    
    def get_top_queries(self, limit=20, hours=24):
        """Get most frequent queries."""
        query = SearchKQLQueries.top_queries(limit, hours)
        return self.execute_query(query, hours)
    
    def get_slow_queries(self, threshold_ms=1000, limit=20):
        """Get slow queries."""
        query = SearchKQLQueries.slow_queries(threshold_ms, limit)
        return self.execute_query(query, 24)
    
    def get_zero_result_queries(self, hours=24):
        """Get queries with no results."""
        query = SearchKQLQueries.zero_result_queries(hours)
        return self.execute_query(query, hours)
    
    def get_performance_report(self):
        """Generate comprehensive performance report."""
        report = {
            'top_queries': self.get_top_queries(10, 24),
            'slow_queries': self.get_slow_queries(1000, 10),
            'zero_results': self.get_zero_result_queries(24),
            'latency_percentiles': self.execute_query(
                SearchKQLQueries.latency_percentiles_by_index(), 24
            ),
            'error_rates': self.execute_query(
                SearchKQLQueries.error_rate_by_operation(), 24
            )
        }
        return report

# Usage
from azure.identity import DefaultAzureCredential

querier = LogAnalyticsQuerier(
    credential=DefaultAzureCredential(),
    workspace_id=workspace_id
)

# Get insights
top_queries = querier.get_top_queries(limit=10)
print("Top 10 queries:")
for q in top_queries:
    print(f"  {q['QueryText']}: {q['QueryCount']} times")

slow_queries = querier.get_slow_queries(threshold_ms=500)
print(f"\nSlow queries (>500ms): {len(slow_queries)}")

# Full report
report = querier.get_performance_report()
```

---

## Dashboards

Dashboards provide real-time visibility into search service health. Well-designed dashboards enable teams to spot issues quickly and track KPIs without running manual queries.

### Dashboard Design Principles

#### 1. Hierarchy of Information

**Top row**: Most critical metrics (current health)
- P95 latency (last 5 minutes)
- Error rate (last hour)
- QPS (current load)
- Throttling events (last 24 hours)

**Middle rows**: Trend analysis (historical patterns)
- Latency trend chart (last 7 days)
- Query volume chart (last 24 hours)
- Error rate trend (last 7 days)

**Bottom rows**: Detailed analysis (drill-down)
- Top 20 queries (volume)
- Slowest queries (latency)
- Zero-result queries (content gaps)

#### 2. Color Coding

Use consistent color semantics:
- **Green**: Healthy (P95 < 300ms, error rate < 0.05%)
- **Yellow**: Warning (P95 300-500ms, error rate 0.05-0.1%)
- **Red**: Critical (P95 > 500ms, error rate > 0.1%)

#### 3. Refresh Intervals

Match refresh rate to use case:
- **Real-time monitoring**: 1-minute refresh (operational dashboards)
- **Daily review**: 5-minute refresh (management dashboards)
- **Weekly reports**: Manual refresh (executive dashboards)

### Dashboard Types

#### Type 1: Operational Dashboard (NOC/DevOps Teams)

**Purpose**: Real-time monitoring, incident detection

**Key widgets**:
1. **Current P95 Latency** (large number, color-coded)
2. **QPS vs Tier Limit** (gauge chart showing utilization %)
3. **Error Rate (last hour)** (line chart with threshold annotations)
4. **Active Alerts** (list of firing alerts)
5. **Latency Heatmap** (hourly bins, 7-day view)

**Refresh**: 1 minute

**Use case**: Displayed on wall monitors in operations center, alerts team to issues requiring immediate action.

#### Type 2: Performance Dashboard (Engineering Teams)

**Purpose**: Trend analysis, optimization opportunities

**Key widgets**:
1. **P95 Latency Trend** (7-day line chart)
2. **Top 20 Slowest Queries** (table with query text, avg latency)
3. **Query Volume by Index** (bar chart)
4. **Zero-Result Queries** (table showing content gaps)
5. **Index Storage Utilization** (stacked bar chart by index)

**Refresh**: 5 minutes

**Use case**: Engineers review daily to identify optimization opportunities (e.g., "products index consistently slower → needs tuning").

#### Type 3: Executive Dashboard (Leadership)

**Purpose**: High-level KPIs, business impact

**Key widgets**:
1. **SLA Compliance** (% of queries <500ms, goal 95%)
2. **Availability** (% uptime, goal 99.9%)
3. **Monthly Query Volume** (trend line with growth %)
4. **Cost per 1000 Queries** (efficiency metric)
5. **User Experience Score** (derived from latency + error rate)

**Refresh**: Manual (generated weekly)

**Use case**: Executive reviews showing search service health contributing to business KPIs.

### Create Azure Dashboard

```python
import json

class SearchDashboard:
    """Create Azure Monitor dashboard for search analytics."""
    
    @staticmethod
    def create_dashboard_json(workspace_id, search_resource_id):
        """Generate dashboard JSON definition."""
        dashboard = {
            "properties": {
                "lenses": [
                    # Query Volume Chart
                    {
                        "order": 0,
                        "parts": [
                            {
                                "position": {"x": 0, "y": 0, "colSpan": 6, "rowSpan": 4},
                                "metadata": {
                                    "type": "Extension/Microsoft_OperationsManagementSuite_Workspace/PartType/LogsDashboardPart",
                                    "inputs": [
                                        {
                                            "name": "Query",
                                            "value": SearchKQLQueries.query_volume_timechart()
                                        },
                                        {
                                            "name": "WorkspaceId",
                                            "value": workspace_id
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    # Latency Chart
                    {
                        "order": 1,
                        "parts": [
                            {
                                "position": {"x": 6, "y": 0, "colSpan": 6, "rowSpan": 4},
                                "metadata": {
                                    "type": "Extension/HubsExtension/PartType/MonitorChartPart",
                                    "inputs": [
                                        {
                                            "name": "resourceId",
                                            "value": search_resource_id
                                        },
                                        {
                                            "name": "metricName",
                                            "value": "SearchLatency"
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    # Top Queries Table
                    {
                        "order": 2,
                        "parts": [
                            {
                                "position": {"x": 0, "y": 4, "colSpan": 12, "rowSpan": 4},
                                "metadata": {
                                    "type": "Extension/Microsoft_OperationsManagementSuite_Workspace/PartType/LogsDashboardPart",
                                    "inputs": [
                                        {
                                            "name": "Query",
                                            "value": SearchKQLQueries.top_queries(20, 24)
                                        },
                                        {
                                            "name": "WorkspaceId",
                                            "value": workspace_id
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            }
        }
        
        return json.dumps(dashboard, indent=2)
    
    @staticmethod
    def save_dashboard(dashboard_json, output_file="search-dashboard.json"):
        """Save dashboard definition to file."""
        with open(output_file, 'w') as f:
            f.write(dashboard_json)
        print(f"✅ Dashboard saved to {output_file}")

# Usage
dashboard_json = SearchDashboard.create_dashboard_json(
    workspace_id=workspace_id,
    search_resource_id=search_resource_id
)

SearchDashboard.save_dashboard(dashboard_json)
```

---

## Alerting

Proactive alerting enables teams to detect and resolve issues before they impact users. Well-configured alerts balance sensitivity (catching real issues) with specificity (avoiding false positives).

### Alerting Strategy

#### Alert Severity Levels

| Severity | Description | Response Time | Example |
|----------|-------------|---------------|---------|
| **Sev 0 (Critical)** | Service down, revenue impact | <5 minutes | 100% error rate, service unreachable |
| **Sev 1 (High)** | Degraded performance, user impact | <15 minutes | P95 latency >1000ms, >10% throttling |
| **Sev 2 (Medium)** | Performance warning, trending issues | <1 hour | P95 latency >500ms, >1% throttling |
| **Sev 3 (Low)** | Informational, no immediate action | <24 hours | Storage >80%, unusual query patterns |

#### Alert Types

**1. Metric Alerts** (based on Azure Monitor metrics)
- **Pros**: Low latency (<1 minute), simple configuration
- **Cons**: Limited to predefined metrics (SearchLatency, SearchQPS, ThrottledSearchQueriesPercentage)
- **Use case**: Real-time performance monitoring

**2. Log Alerts** (based on KQL queries)
- **Pros**: Flexible (any KQL query), rich context in alert
- **Cons**: Higher latency (5-15 minutes), more complex configuration
- **Use case**: Complex conditions (e.g., "alert if >50 slow queries in 5 minutes")

#### Alerting Best Practices

**1. Use Percentiles, Not Averages**

```
❌ BAD: Alert when average latency > 500ms
Reason: Average hides outliers (90% fast + 10% very slow = acceptable average)

✅ GOOD: Alert when P95 latency > 500ms
Reason: Ensures 95% of users have good experience
```

**2. Multi-Window Evaluation**

```
❌ BAD: Alert on single spike
Evaluation: 1-minute window
Result: Alert fires on transient spike (false positive)

✅ GOOD: Alert on sustained degradation
Evaluation: 5-minute window, 2 consecutive violations
Result: Alert fires only when issue persists
```

**3. Time-of-Day Awareness**

```
❌ BAD: Static thresholds
Alert: QPS > 80 (triggers at 3 AM when normal QPS is 5)

✅ GOOD: Dynamic thresholds
Alert: QPS > 2× baseline for hour of day
Result: 160 QPS at 2 PM (normal), 10 QPS at 3 AM (alert - unusual spike)
```

**4. Alert Fatigue Prevention**

**Problem**: Too many alerts → ignored alerts → missed incidents

**Solutions**:
- **Aggregate related alerts**: Instead of 50 alerts for "index1 slow", "index2 slow", create 1 alert "multiple indexes slow"
- **Cooldown period**: After alert fires, suppress duplicates for 30 minutes
- **Escalation policy**: Send Sev 3 to Slack, Sev 1 to PagerDuty

### Common Alert Configurations

#### Alert 1: High P95 Latency

**Trigger**: P95 latency > 500ms for 5 minutes

**KQL query**:
```kql
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.SEARCH"
| where OperationName == "Query.Search"
| where TimeGenerated > ago(5m)
| summarize P95 = percentile(DurationMs, 95)
| where P95 > 500
```

**Action**: Page on-call engineer, auto-scale replicas

**Severity**: Sev 2 (if P95 > 500ms), Sev 1 (if P95 > 1000ms)

#### Alert 2: Throttling Detected

**Trigger**: >1% of queries throttled in 5-minute window

**Metric alert**:
- Metric: `ThrottledSearchQueriesPercentage`
- Condition: `> 1`
- Window: 5 minutes
- Frequency: 1 minute

**Action**: Auto-scale service tier, notify capacity planning team

**Severity**: Sev 1 (immediate revenue impact)

#### Alert 3: Error Rate Spike

**Trigger**: Error rate >0.5% (10× normal baseline of 0.05%)

**KQL query**:
```kql
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.SEARCH"
| where OperationName == "Query.Search"
| where TimeGenerated > ago(5m)
| summarize 
    Total = count(),
    Errors = countif(ResultType != "Success")
| extend ErrorRate = (Errors * 100.0) / Total
| where ErrorRate > 0.5
```

**Action**: Alert engineering team, check for deployment correlation

**Severity**: Sev 2

#### Alert 4: Storage Nearing Limit

**Trigger**: Storage >90% of tier limit

**Metric alert**:
- Metric: `StorageSize`
- Condition: `> 22.5 GB` (for S1 with 25 GB limit)
- Window: 15 minutes
- Frequency: 5 minutes

**Action**: Purge old data, plan tier upgrade

**Severity**: Sev 3 (preventive)

### Action Groups

Action groups define **who gets notified and how** when an alert fires.

**Notification channels**:
1. **Email/SMS**: For Sev 2-3 alerts (non-urgent)
2. **Webhook**: For integration with Slack, Microsoft Teams
3. **Azure Function**: For automated remediation (e.g., auto-scale)
4. **Logic App**: For complex workflows (e.g., create incident ticket, notify multiple teams)
5. **ITSM connector**: For ServiceNow, Jira integration

**Example escalation policy**:
```
Sev 3: Email to engineering-search@company.com
Sev 2: Email + Slack message to #search-ops
Sev 1: PagerDuty page + SMS to on-call engineer + Slack to #incidents
Sev 0: PagerDuty page entire SRE team + auto-create war room
```

### Create Alert Rules

```python
from azure.mgmt.monitor import MonitorManagementClient
from azure.mgmt.monitor.models import (
    MetricAlertResource,
    MetricAlertCriteria,
    MetricCriteria,
    MetricAlertAction
)

class SearchAlerts:
    """Configure alerts for Azure Search."""
    
    def __init__(self, subscription_id, credential):
        self.client = MonitorManagementClient(credential, subscription_id)
    
    def create_latency_alert(
        self,
        resource_group,
        alert_name,
        search_resource_id,
        threshold_ms=1000,
        action_group_id=None
    ):
        """Create alert for high search latency."""
        alert = MetricAlertResource(
            location='global',
            description=f'Alert when search latency exceeds {threshold_ms}ms',
            severity=2,
            enabled=True,
            scopes=[search_resource_id],
            evaluation_frequency='PT1M',  # 1 minute
            window_size='PT5M',  # 5 minute window
            criteria=MetricAlertCriteria(
                all_of=[
                    MetricCriteria(
                        name='HighLatency',
                        metric_name='SearchLatency',
                        metric_namespace='Microsoft.Search/searchServices',
                        operator='GreaterThan',
                        threshold=threshold_ms,
                        time_aggregation='Average'
                    )
                ]
            ),
            actions=[
                MetricAlertAction(action_group_id=action_group_id)
            ] if action_group_id else []
        )
        
        result = self.client.metric_alerts.create_or_update(
            resource_group_name=resource_group,
            rule_name=alert_name,
            parameters=alert
        )
        
        print(f"✅ Latency alert created: {alert_name}")
        return result
    
    def create_throttling_alert(
        self,
        resource_group,
        alert_name,
        search_resource_id,
        threshold_percentage=5,
        action_group_id=None
    ):
        """Create alert for query throttling."""
        alert = MetricAlertResource(
            location='global',
            description=f'Alert when throttling exceeds {threshold_percentage}%',
            severity=1,  # Higher severity
            enabled=True,
            scopes=[search_resource_id],
            evaluation_frequency='PT1M',
            window_size='PT5M',
            criteria=MetricAlertCriteria(
                all_of=[
                    MetricCriteria(
                        name='HighThrottling',
                        metric_name='ThrottledSearchQueriesPercentage',
                        metric_namespace='Microsoft.Search/searchServices',
                        operator='GreaterThan',
                        threshold=threshold_percentage,
                        time_aggregation='Average'
                    )
                ]
            ),
            actions=[
                MetricAlertAction(action_group_id=action_group_id)
            ] if action_group_id else []
        )
        
        result = self.client.metric_alerts.create_or_update(
            resource_group_name=resource_group,
            rule_name=alert_name,
            parameters=alert
        )
        
        print(f"✅ Throttling alert created: {alert_name}")
        return result

# Usage
alerts = SearchAlerts(
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    credential=DefaultAzureCredential()
)

# Create latency alert
alerts.create_latency_alert(
    resource_group="rg-search-evaluation",
    alert_name="search-high-latency",
    search_resource_id=search_resource_id,
    threshold_ms=1000
)

# Create throttling alert
alerts.create_throttling_alert(
    resource_group="rg-search-evaluation",
    alert_name="search-throttling",
    search_resource_id=search_resource_id,
    threshold_percentage=5
)
```

---

## Application Insights Integration

While Azure Monitor and Log Analytics provide comprehensive logging for Azure AI Search operations, **Application Insights** adds a critical layer of distributed tracing and end-to-end correlation across your entire application stack. This is especially valuable when search is one component in a multi-service architecture.

### When to Use Application Insights vs Log Analytics

**Use Log Analytics for:**
- Search service-level diagnostics (query performance, indexing operations)
- Cost-effective log storage with long retention periods
- KQL-based analysis of search-specific patterns
- Compliance and audit requirements

**Use Application Insights for:**
- End-to-end request tracing across multiple services (web app → API → Azure AI Search)
- Correlation of search queries with user sessions and business outcomes
- Real-time performance profiling and dependency tracking
- Application-level custom events (user clicks, conversions)

**Best Practice: Use Both**
- Send Azure AI Search diagnostic logs to **Log Analytics** ($2.76/GB)
- Send application telemetry to **Application Insights** ($2.30/GB)
- Correlate using **operation_id** to trace requests end-to-end

### Distributed Tracing with Correlation IDs

The most powerful feature of Application Insights is the ability to trace a single user request across multiple services. When a user submits a search query, you can follow that request from the web browser → web server → Azure AI Search → Cosmos DB → back to the user.

**Correlation ID Propagation:**
```python
from applicationinsights import TelemetryClient
from applicationinsights.requests import WSGIApplication
import uuid

class SearchTelemetry:
    """Custom telemetry for search operations with correlation."""
    
    def __init__(self, instrumentation_key):
        self.client = TelemetryClient(instrumentation_key)
    
    def track_query(self, query_text, duration_ms, result_count, 
                     correlation_id=None, user_id=None, session_id=None, success=True):
        """Track search query with full correlation context."""
        
        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Set operation context for distributed tracing
        self.client.context.operation.id = correlation_id
        self.client.context.operation.name = f"Search: {query_text[:50]}"
        
        # Add user context
        if user_id:
            self.client.context.user.id = user_id
        if session_id:
            self.client.context.session.id = session_id
        
        # Track custom event with rich properties
        properties = {
            'query': query_text,
            'result_count': result_count,
            'success': success,
            'correlation_id': correlation_id  # For cross-service correlation
        }
        
        measurements = {
            'duration_ms': duration_ms,
            'results': result_count
        }
        
        self.client.track_event(
            'SearchQuery',
            properties=properties,
            measurements=measurements
        )
    
    def track_dependency(self, name, duration_ms, success=True, correlation_id=None):
        """Track external dependency call (e.g., Azure AI Search API)."""
        if correlation_id:
            self.client.context.operation.id = correlation_id
        
        self.client.track_dependency(
            name=name,
            data='Azure AI Search',
            type='HTTP',
            duration=duration_ms,
            success=success
        )
    
    def flush(self):
        """Flush telemetry data."""
        self.client.flush()

# Usage in Web Application
telemetry = SearchTelemetry(
    instrumentation_key=os.getenv("APPINSIGHTS_INSTRUMENTATION_KEY")
)

# Generate correlation ID for this request
correlation_id = str(uuid.uuid4())

# Track search query with correlation
import time
start = time.time()
results = search_client.search("laptop")
duration = (time.time() - start) * 1000

telemetry.track_query(
    query_text="laptop",
    duration_ms=duration,
    result_count=len(list(results)),
    correlation_id=correlation_id,  # Same ID used across all services
    user_id="user123",
    session_id="session456",
    success=True
)

telemetry.flush()
```

**How Correlation Works:**
1. User submits search from web app
2. Web app generates `correlation_id` and includes it in all telemetry
3. Web app calls Azure AI Search API (logs dependency with `correlation_id`)
4. Azure AI Search executes query (diagnostic logs include `operationId`)
5. Results returned to user (tracked with same `correlation_id`)

**Result in Application Insights:**
- Single **End-to-End Transaction** view showing all steps
- Total latency breakdown: Web (20ms) + Search (150ms) + Rendering (30ms) = 200ms
- Identify bottlenecks visually in dependency graph

### Custom Events for Business Metrics

Beyond technical performance, Application Insights can track business outcomes:

```python
# Track search result clicks (conversion tracking)
telemetry.client.track_event(
    'SearchResultClick',
    properties={
        'query': 'laptop',
        'clicked_position': 3,  # User clicked 3rd result
        'product_id': 'prod-789',
        'correlation_id': correlation_id
    }
)

# Track zero-result searches (opportunity for catalog expansion)
if result_count == 0:
    telemetry.client.track_event(
        'ZeroResultSearch',
        properties={
            'query': query_text,
            'user_id': user_id,
            'session_id': session_id
        }
    )
```

### Performance Profiling Integration

Application Insights Profiler provides code-level insights into performance bottlenecks:

**Enable Profiler:**
```bash
# Install Application Insights Profiler for Python
pip install applicationinsights-profiler

# Enable in your application
from applicationinsights.profiler import Profiler
profiler = Profiler(instrumentation_key=os.getenv("APPINSIGHTS_INSTRUMENTATION_KEY"))
profiler.start()
```

**Use Case: Identify Slow Search Client Calls**
- Profiler captures stack traces when search latency exceeds threshold
- See exact line of code causing delays (e.g., synchronous result iteration)
- Optimize based on real production data

---

## Custom Logging

While Azure Monitor provides built-in diagnostic logging, custom logging gives you fine-grained control over what, when, and how you log search operations. This is essential for debugging complex issues, tracking business metrics, and maintaining audit trails.

### Structured Logging Best Practices

**Why Structured Logging Matters:**
- **Machine-readable**: JSON format enables automated log analysis
- **Consistent schema**: Standardized fields across all log entries
- **Rich context**: Include correlation IDs, user context, business metadata
- **Efficient querying**: Filter and aggregate logs without parsing text

**Structured Logging Implementation:**
```python
import logging
import json
from datetime import datetime
import random

class SearchLogger:
    """Structured logging for search operations."""
    
    def __init__(self, log_file="search_operations.log"):
        self.logger = logging.getLogger("SearchLogger")
        self.logger.setLevel(logging.INFO)
        
        # File handler with JSON formatting
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log_query(self, query_text, index_name, duration_ms, result_count, 
                  user_id=None, correlation_id=None, metadata=None):
        """Log search query with structured data."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'search_query',
            'level': 'INFO',
            'query': query_text,
            'index': index_name,
            'duration_ms': duration_ms,
            'result_count': result_count,
            'user_id': user_id,
            'correlation_id': correlation_id,
            'metadata': metadata or {}
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, operation, error_message, context=None, correlation_id=None):
        """Log error with context."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'error',
            'level': 'ERROR',
            'operation': operation,
            'error': error_message,
            'correlation_id': correlation_id,
            'context': context or {}
        }
        
        self.logger.error(json.dumps(log_entry))
    
    def should_sample(self, sample_rate=0.1):
        """Probabilistic sampling for high-volume logging."""
        return random.random() < sample_rate

# Usage
logger = SearchLogger()

# Log with full context
logger.log_query(
    query_text="azure search",
    index_name="products",
    duration_ms=45.5,
    result_count=12,
    user_id="user123",
    correlation_id="corr-456",
    metadata={'experiment': 'relevance-v2', 'tenant': 'contoso'}
)
```

### Log Levels and When to Use Them

**INFO: Routine Operations (Sample 10%)**
- Successful search queries
- Indexing operations completed
- Configuration changes applied

**WARNING: Degraded Performance**
- Query latency >500ms (slow but successful)
- Throttling events (approaching limits)
- Deprecated API usage

**ERROR: Failures**
- Search queries that returned errors (4xx, 5xx)
- Failed indexing operations
- Authentication failures

**Example:**
```python
# INFO: Log sampled successful queries
if logger.should_sample(sample_rate=0.1):
    logger.log_query(query_text="laptop", ...)

# WARNING: Log all slow queries
if duration_ms > 500:
    logger.logger.warning(json.dumps({
        'event_type': 'slow_query',
        'query': query_text,
        'duration_ms': duration_ms
    }))

# ERROR: Log all failures
except Exception as e:
    logger.log_error(
        operation='search_query',
        error_message=str(e),
        context={'query': query_text}
    )
```

### Correlation IDs for Request Tracing

Correlation IDs enable you to trace a single user request across multiple log entries and services:

```python
import uuid

def handle_search_request(query_text, user_id):
    """Handle search request with correlation ID."""
    correlation_id = str(uuid.uuid4())
    
    try:
        # Log request received
        logger.logger.info(json.dumps({
            'event_type': 'request_received',
            'correlation_id': correlation_id,
            'query': query_text,
            'user_id': user_id
        }))
        
        # Execute search
        start = time.time()
        results = search_client.search(query_text)
        duration = (time.time() - start) * 1000
        
        # Log search completed
        logger.log_query(
            query_text=query_text,
            index_name="products",
            duration_ms=duration,
            result_count=len(list(results)),
            user_id=user_id,
            correlation_id=correlation_id
        )
        
        # Call downstream service (e.g., fetch product details)
        fetch_product_details(results, correlation_id)
        
    except Exception as e:
        # Log error with same correlation ID
        logger.log_error(
            operation='handle_search_request',
            error_message=str(e),
            correlation_id=correlation_id
        )
```

**Result: All log entries share same `correlation_id`, enabling end-to-end tracing:**
```json
{"event_type": "request_received", "correlation_id": "abc-123", ...}
{"event_type": "search_query", "correlation_id": "abc-123", "duration_ms": 45.5}
{"event_type": "fetch_products", "correlation_id": "abc-123", ...}
```

### Sampling Strategies for High-Volume Applications

For applications with 100K+ QPS, logging every event is cost-prohibitive. Implement intelligent sampling:

```python
class SamplingLogger(SearchLogger):
    """Logger with multiple sampling strategies."""
    
    def log_query_sampled(self, query_text, duration_ms, result_count, **kwargs):
        """Log query with conditional sampling."""
        
        # ALWAYS log errors
        if kwargs.get('error'):
            self.log_error(operation='search_query', error_message=kwargs['error'])
            return
        
        # ALWAYS log slow queries
        if duration_ms > 500:
            self.log_query(query_text, duration_ms=duration_ms, 
                          result_count=result_count, **kwargs)
            return
        
        # ALWAYS log zero-result searches (business insight)
        if result_count == 0:
            self.logger.warning(json.dumps({
                'event_type': 'zero_result_search',
                'query': query_text,
                'user_id': kwargs.get('user_id')
            }))
            return
        
        # Sample 10% of routine successful queries
        if self.should_sample(sample_rate=0.1):
            self.log_query(query_text, duration_ms=duration_ms, 
                          result_count=result_count, **kwargs)
```

**Expected Outcome:**
- 100% coverage for failures, slow queries, zero-result searches
- 10% sampling for routine queries
- 90% cost reduction while maintaining diagnostic coverage

---

## Best Practices

Effective monitoring of Azure AI Search requires thoughtful configuration that balances observability needs with cost management. The following best practices are organized by key concern areas to help you build a robust, cost-effective monitoring solution.

### Logging Strategy

**✅ DO: Sample High-Volume Operations**
- Implement probabilistic sampling for routine queries (10% sample rate)
- Log all errors and slow queries (>500ms) regardless of sampling
- Use conditional logging to capture edge cases without overwhelming storage
- Expected outcome: 90% cost reduction while maintaining diagnostic coverage

**❌ DON'T: Log Everything at Full Volume**
- Logging every query at 100K QPS = $414/month for standard queries alone
- Full logging creates noise that obscures important signals
- High ingestion costs often exceed search service costs
- Risk: Alert fatigue from processing millions of routine success logs

**Example Decision Matrix:**
```
Query Type          | Sample Rate | Reasoning
--------------------|-------------|--------------------------------------------
Success <500ms      | 10%         | Representative sample for trend analysis
Success >500ms      | 100%        | Investigate all slow queries
Error (4xx/5xx)     | 100%        | Always log failures
Throttling (503)    | 100%        | Critical for capacity planning
```

### Cost Optimization

**✅ DO: Use Tiered Retention Policies**
- Keep recent logs (7 days) in **Hot tier** ($2.76/GB) for interactive queries
- Archive compliance logs (90+ days) to **Storage Account** ($0.01/GB)
- Delete non-critical logs after retention period expires
- Expected outcome: 60-80% reduction in long-term storage costs

**❌ DON'T: Keep All Logs in Hot Tier Indefinitely**
- 7-day retention for 100K QPS: ~$41/month
- 1-year retention for same workload: ~$2,400/month
- Most queries analyze only the last 24-48 hours
- Risk: Paying premium storage prices for logs that are rarely accessed

**Cost Optimization Example:**
```python
# BAD: Single retention policy
retention_days = 365  # $2,400/month for 100K QPS

# GOOD: Tiered retention
hot_tier_days = 7      # $41/month (interactive queries)
archive_days = 90      # $9/month (compliance, rarely accessed)
delete_after_days = 365  # Purge old data
# Total: ~$50/month vs $2,400/month (96% savings)
```

### Alert Design

**✅ DO: Use Percentiles for Latency Alerts**
- Alert on **P95 or P99 latency**, not average latency
- Multi-window evaluation: Require 2 consecutive 5-minute windows to trigger
- Time-of-day awareness: Scale thresholds by 2× during peak hours
- Expected outcome: 90% reduction in false positive alerts

**❌ DON'T: Alert on Average Latency**
- Averages hide outliers (95% fast + 5% very slow = acceptable average)
- Single data point triggers cause false positives from transient spikes
- Fixed thresholds don't account for traffic patterns
- Risk: Alert fatigue leads to ignoring critical alerts

**Alert Configuration Comparison:**
```
Metric          | Bad Approach              | Good Approach
----------------|---------------------------|----------------------------------
Latency         | Avg > 200ms               | P95 > 500ms (2 consecutive 5-min)
Threshold       | Fixed 200ms 24/7          | Dynamic (2× baseline during peak)
Evaluation      | Single 1-min window       | Multi-window (reduce false +)
Action          | Email every trigger       | Aggregate, then escalate
```

### Dashboard Design

**✅ DO: Follow Hierarchy of Information**
- **Top row**: Critical real-time metrics (P95 latency, active alerts, error rate)
- **Middle row**: Trend analysis (7-day latency trends, query volume patterns)
- **Bottom row**: Drill-down details (slowest queries, zero-result searches)
- Use color coding: Green (<300ms), Yellow (300-500ms), Red (>500ms)
- Expected outcome: 10-second glanceability for on-call engineers

**❌ DON'T: Create Overcrowded Dashboards**
- More than 12 widgets = cognitive overload
- Mixing operational and executive metrics confuses audience
- Auto-refresh every 5 seconds overwhelms users
- Risk: Important signals lost in visual noise

**Dashboard Examples:**
```
// BAD: Overcrowded Dashboard
Widgets: 28
Refresh: 5 seconds
Audience: Everyone
Result: Information overload, unused

// GOOD: Purpose-Built Dashboards
1. Operational (NOC):    8 widgets, 1-min refresh, real-time focus
2. Performance (Eng):    10 widgets, 5-min refresh, trend analysis
3. Executive (CxO):      6 widgets, manual refresh, KPIs only
```

### KQL Performance Optimization

**✅ DO: Filter Early and Aggressively**
- Apply time filter first: `| where TimeGenerated > ago(1h)`
- Filter before parse/extend: Reduce rows before expensive operations
- Use `top` instead of `sort | take`: 10× faster on large result sets
- Expected outcome: 30× faster queries, 30× lower costs

**❌ DON'T: Scan Entire Log Tables**
- Querying 30 days without time filter: Scans 15GB+ (30× slower, $41.40 query cost)
- Parsing before filtering: Processes unnecessary rows
- Using `sort | take` on millions of rows: Materializes entire dataset
- Risk: Slow dashboards, high query costs, timeout errors

**KQL Performance Examples:**
```kusto
// BAD: Scan everything, then filter (30× slower)
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.SEARCH"
| where Category == "OperationLogs"
| where TimeGenerated > ago(1h)  // Filter last!

// GOOD: Filter first, then process (30× faster)
AzureDiagnostics
| where TimeGenerated > ago(1h)  // Filter first!
| where ResourceProvider == "MICROSOFT.SEARCH"
| where Category == "OperationLogs"
```

### Security and Access Control

**✅ DO: Restrict Log Analytics Workspace Access**
- Use **Azure RBAC** to limit who can query logs
- Grant **Log Analytics Reader** role (read-only) to most users
- Grant **Log Analytics Contributor** role only to administrators
- Audit access regularly using Azure Activity Logs
- Expected outcome: Compliance with least-privilege principle

**❌ DON'T: Share Workspace Connection Strings**
- Connection strings grant full query access
- No audit trail for who queried what
- Credentials can be leaked in code repositories
- Risk: Unauthorized access to sensitive query data, compliance violations

**Access Control Example:**
```bash
# BAD: Share workspace ID + key in code
WORKSPACE_ID="abc123..."
WORKSPACE_KEY="sharedSecret..."  # Anyone can query!

# GOOD: Use Azure AD authentication
az login
az monitor log-analytics workspace show \
  --resource-group rg-search \
  --workspace-name law-search-prod
# Access controlled by Azure RBAC, full audit trail
```

### Retention and Compliance

**✅ DO: Archive Logs for Compliance Requirements**
- Configure **Archive tier** for compliance logs (SOX, HIPAA, GDPR)
- Use **Azure Storage immutable blobs** for tamper-proof audit trails
- Retain query logs for forensic analysis (90-365 days)
- Document retention policy in compliance documentation
- Expected outcome: Meet regulatory requirements at minimal cost

**❌ DON'T: Delete Audit Logs Prematurely**
- Compliance often requires 7-year retention for financial records
- Deleted logs cannot be recovered for audits or investigations
- Storage costs are minimal compared to compliance penalties
- Risk: Regulatory fines, failed audits, legal exposure

**Compliance Retention Example:**
```
Log Type               | Retention   | Storage Tier | Cost/Month (100K QPS)
-----------------------|-------------|--------------|---------------------
Operational (debug)    | 7 days      | Hot          | $41
Performance (trends)   | 90 days     | Hot          | $248
Compliance (audit)     | 7 years     | Archive      | $108
Security (forensics)   | 1 year      | Archive      | $12
                                     | Total        | $409/month
```

### Summary: Monitoring Maturity Model

As your Azure AI Search implementation evolves, your monitoring strategy should progress through these maturity stages:

**Level 1: Basic (Day 1)**
- Enable diagnostic logging with 7-day retention
- Create 2-3 critical alerts (latency, throttling, errors)
- Single operational dashboard with 6-8 widgets
- Cost: ~$50/month for moderate traffic

**Level 2: Intermediate (Month 1-3)**
- Implement sampling strategy (90% cost reduction)
- Add tiered retention (Hot + Archive)
- Create role-specific dashboards (Ops, Eng, Exec)
- Advanced KQL queries for troubleshooting
- Cost: ~$100/month with optimizations

**Level 3: Advanced (Month 3+)**
- Application Insights correlation for distributed tracing
- Custom enrichment (tenant_id, experiment_variant)
- Automated anomaly detection with Machine Learning
- Compliance-ready retention with immutable archives
- Cost: ~$150/month with full observability

By following these best practices, you'll build a monitoring solution that provides deep visibility into your Azure AI Search service while controlling costs and maintaining compliance with regulatory requirements.

---

## Troubleshooting Common Monitoring Issues

Even with proper configuration, you may encounter issues with Azure Monitor, Log Analytics, or Application Insights. This section covers the most common monitoring problems and their solutions.

### Issue 1: Logs Not Appearing in Log Analytics

**Symptoms:**
- Diagnostic settings configured, but no logs appear in Log Analytics workspace
- Queries return zero results even though search service is active
- `AzureDiagnostics` table exists but has no recent data

**Root Causes & Solutions:**

**Solution 1: Verify Diagnostic Settings Configuration**
```bash
# Check diagnostic settings are enabled
az monitor diagnostic-settings list \
  --resource /subscriptions/{sub-id}/resourceGroups/rg-search/providers/Microsoft.Search/searchServices/search-prod

# Look for:
# - "logs": [{"category": "OperationLogs", "enabled": true}]
# - "workspaceId": Points to correct Log Analytics workspace
```

**Solution 2: Confirm Workspace Connection**
```kusto
# In Log Analytics, verify workspace is receiving any data
Heartbeat
| where TimeGenerated > ago(1h)
| summarize count()

# If zero results, workspace connectivity issue (not search-specific)
# If >0 results, search service logs not being sent
```

**Solution 3: Check Retention Policy**
- Default retention: 30 days
- If querying older data, logs may have been purged
- Verify retention: Azure Portal → Log Analytics Workspace → Usage and estimated costs → Data Retention

**Solution 4: Wait for Initial Delay (New Configuration)**
- First logs appear 10-15 minutes after enabling diagnostic settings
- Full ingestion pipeline: Up to 30 minutes
- If >1 hour with no logs, investigate configuration

### Issue 2: High Log Analytics Costs

**Symptoms:**
- Log Analytics bill exceeds search service costs
- Unexpected charges for data ingestion (>$500/month)
- Approaching monthly budget limits

**Root Causes & Solutions:**

**Solution 1: Implement Sampling (90% Cost Reduction)**
```python
import random

def should_log_query(query_duration_ms, result_count, error=None):
    """Decide whether to log this query."""
    
    # ALWAYS log errors
    if error:
        return True
    
    # ALWAYS log slow queries
    if query_duration_ms > 500:
        return True
    
    # ALWAYS log zero-result searches
    if result_count == 0:
        return True
    
    # Sample 10% of routine queries
    return random.random() < 0.1

# Before logging:
if should_log_query(duration_ms, result_count, error):
    # Send to Log Analytics
```

**Solution 2: Filter Unnecessary Log Categories**
```bash
# Disable verbose categories you don't need
az monitor diagnostic-settings update \
  --name "search-diagnostics" \
  --resource {search-service-id} \
  --logs '[
    {"category": "OperationLogs", "enabled": true},
    {"category": "AllMetrics", "enabled": false}
  ]'
# Metrics are free via Azure Monitor; don't pay to ingest them to Log Analytics
```

**Solution 3: Reduce Retention Period**
```bash
# Change from 90 days to 7 days (reduces costs by 92%)
az monitor log-analytics workspace update \
  --resource-group rg-search \
  --workspace-name law-search \
  --retention-time 7
```

**Solution 4: Use Archive Tier for Compliance Logs**
- Configure diagnostic settings to send logs to **Storage Account** ($0.01/GB)
- Keep only 7 days in Log Analytics for interactive queries
- Archive older logs for compliance

**Cost Comparison:**
```
Scenario                    | Monthly Cost (100K QPS)
----------------------------|------------------------
Before: 90-day retention    | $2,400
After: 7-day + sampling     | $41 (98% savings)
```

### Issue 3: Alerts Not Firing

**Symptoms:**
- Alert rule configured, but never triggers
- Known performance issues (high latency) don't generate alerts
- Action group configured but no emails/notifications received

**Root Causes & Solutions:**

**Solution 1: Verify Alert Threshold and Aggregation**
```kusto
# Test the alert query manually
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.SEARCH"
| where Category == "OperationLogs"
| where TimeGenerated > ago(5m)
| summarize P95Latency = percentile(DurationMs, 95)
| where P95Latency > 500  // Alert threshold

# If query returns results, alert should fire
# If query returns nothing, threshold may be too high
```

**Solution 2: Check Evaluation Frequency**
- Alert frequency: How often query runs (default: 5 minutes)
- Time window: How much data to analyze (default: 5 minutes)
- **Issue**: If evaluation frequency > time window, you miss data
- **Fix**: Set evaluation frequency ≤ time window

```bash
# Update alert rule
az monitor metrics alert update \
  --name "high-latency-alert" \
  --resource-group rg-search \
  --evaluation-frequency 5m \  # Run every 5 minutes
  --window-size 5m              # Look at last 5 minutes
```

**Solution 3: Validate Action Group**
```bash
# Test action group
az monitor action-group test-notifications create \
  --action-group-name "search-alerts" \
  --resource-group rg-search \
  --notification-type Email \
  --receivers "engineer@contoso.com"

# Check email spam folder for test notification
```

### Issue 4: Dashboard Not Updating or Showing Errors

**Symptoms:**
- Dashboard widgets show "No data" or stale data
- Charts display error: "Query timeout" or "Insufficient permissions"
- Dashboard loads slowly (>30 seconds)

**Root Causes & Solutions:**

**Solution 1: Verify Query Permissions**
```bash
# Grant yourself Log Analytics Reader role
az role assignment create \
  --assignee user@contoso.com \
  --role "Log Analytics Reader" \
  --scope /subscriptions/{sub-id}/resourceGroups/rg-search/providers/Microsoft.OperationalInsights/workspaces/law-search
```

**Solution 2: Optimize Slow KQL Queries**
```kusto
// BAD: Slow query (scans all data)
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.SEARCH"
| summarize count()

// GOOD: Fast query (time filter first)
AzureDiagnostics
| where TimeGenerated > ago(1h)  // Filter first!
| where ResourceProvider == "MICROSOFT.SEARCH"
| summarize count()
```

**Solution 3: Reduce Auto-Refresh Frequency**
- Dashboard refreshing every 10 seconds = excessive query load
- Operational dashboards: 1-minute refresh
- Performance dashboards: 5-minute refresh
- Executive dashboards: Manual refresh

**Solution 4: Check Data Retention**
- If dashboard queries look at last 90 days but retention is 30 days = "No data"
- Align query time range with retention policy

---

## Next Steps

- **[Query Optimization](./12-query-optimization.md)** - Use metrics to optimize
- **[Monitoring & Alerting](./23-monitoring-alerting.md)** - Production monitoring
- **[Troubleshooting Guide](./27-troubleshooting-guide.md)** - Debug with logs

---

*See also: [Azure AI Search Setup](./04-azure-ai-search-setup.md) | [Cost Analysis](./19-cost-analysis.md)*