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

Azure Monitor provides comprehensive observability:
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

### Built-in Metrics

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

### Setup Application Insights

```python
from applicationinsights import TelemetryClient
from applicationinsights.requests import WSGIApplication

class SearchTelemetry:
    """Custom telemetry for search operations."""
    
    def __init__(self, instrumentation_key):
        self.client = TelemetryClient(instrumentation_key)
    
    def track_query(self, query_text, duration_ms, result_count, success=True):
        """Track search query execution."""
        properties = {
            'query': query_text,
            'result_count': result_count,
            'success': success
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
    
    def track_dependency(self, name, duration_ms, success=True):
        """Track external dependency call."""
        self.client.track_dependency(
            name=name,
            data='Azure AI Search',
            duration=duration_ms,
            success=success
        )
    
    def flush(self):
        """Flush telemetry data."""
        self.client.flush()

# Usage
telemetry = SearchTelemetry(
    instrumentation_key=os.getenv("APPINSIGHTS_INSTRUMENTATION_KEY")
)

# Track query
import time
start = time.time()
results = search_client.search("laptop")
duration = (time.time() - start) * 1000

telemetry.track_query(
    query_text="laptop",
    duration_ms=duration,
    result_count=len(list(results)),
    success=True
)

telemetry.flush()
```

---

## Custom Logging

### Structured Logging

```python
import logging
import json
from datetime import datetime

class SearchLogger:
    """Structured logging for search operations."""
    
    def __init__(self, log_file="search_operations.log"):
        self.logger = logging.getLogger("SearchLogger")
        self.logger.setLevel(logging.INFO)
        
        # File handler with JSON formatting
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log_query(self, query_text, index_name, duration_ms, result_count, user_id=None):
        """Log search query with structured data."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'search_query',
            'query': query_text,
            'index': index_name,
            'duration_ms': duration_ms,
            'result_count': result_count,
            'user_id': user_id
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, operation, error_message, context=None):
        """Log error with context."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'error',
            'operation': operation,
            'error': error_message,
            'context': context or {}
        }
        
        self.logger.error(json.dumps(log_entry))

# Usage
logger = SearchLogger()

logger.log_query(
    query_text="azure search",
    index_name="products",
    duration_ms=45.5,
    result_count=12,
    user_id="user123"
)
```

---

## Best Practices

### ✅ Do's
1. **Enable diagnostic logging** immediately for all search services
2. **Set appropriate retention** (30-90 days for production)
3. **Create alerts** for latency, throttling, and errors
4. **Use KQL queries** for deep analysis
5. **Monitor costs** of Log Analytics ingestion
6. **Export logs** for long-term storage if needed

### ❌ Don'ts
1. **Don't** log sensitive data in queries
2. **Don't** ignore throttling alerts
3. **Don't** skip setting up alerting
4. **Don't** use excessive retention without cost analysis
5. **Don't** forget to correlate with Application Insights

---

## Next Steps

- **[Query Optimization](./12-query-optimization.md)** - Use metrics to optimize
- **[Monitoring & Alerting](./23-monitoring-alerting.md)** - Production monitoring
- **[Troubleshooting Guide](./27-troubleshooting-guide.md)** - Debug with logs

---

*See also: [Azure AI Search Setup](./04-azure-ai-search-setup.md) | [Cost Analysis](./19-cost-analysis.md)*