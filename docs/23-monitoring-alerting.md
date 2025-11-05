# Monitoring & Alerting for Azure AI Search

## Table of Contents

- [Overview](#overview)
- [Azure Monitor Integration](#azure-monitor-integration)
- [Metrics & KPIs](#metrics--kpis)
- [Log Analytics](#log-analytics)
- [Alerting](#alerting)
- [Dashboards](#dashboards)
- [Application Insights](#application-insights)
- [Anomaly Detection](#anomaly-detection)
- [Incident Response](#incident-response)
- [SLA Tracking](#sla-tracking)
- [Best Practices](#best-practices)

---

## Overview

### Monitoring Strategy

Comprehensive monitoring ensures search service reliability and performance.

```
Metrics → Logs → Alerts → Dashboards → Incident Response
```

**Key Components**:
- **Azure Monitor**: Platform metrics and logs
- **Log Analytics**: Query and analyze telemetry
- **Alerts**: Proactive notifications
- **Dashboards**: Visual monitoring
- **Application Insights**: Application-level telemetry

### Monitoring Pillars

1. **Availability**: Service uptime and health
2. **Performance**: Latency, throughput, QPS
3. **Errors**: Failures, throttling, exceptions
4. **Cost**: Resource consumption and spending
5. **Usage**: Query patterns and trends

---

## Azure Monitor Integration

### Diagnostic Settings

```python
from azure.mgmt.monitor import MonitorManagementClient
from azure.mgmt.monitor.models import (
    DiagnosticSettingsResource,
    LogSettings,
    MetricSettings,
    RetentionPolicy
)
from azure.identity import DefaultAzureCredential

class MonitoringSetup:
    """Configure Azure Monitor for search service."""
    
    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        search_service_name: str
    ):
        self.credential = DefaultAzureCredential()
        self.monitor_client = MonitorManagementClient(
            credential=self.credential,
            subscription_id=subscription_id
        )
        
        self.resource_id = (
            f"/subscriptions/{subscription_id}"
            f"/resourceGroups/{resource_group}"
            f"/providers/Microsoft.Search/searchServices/{search_service_name}"
        )
    
    def configure_diagnostic_settings(
        self,
        workspace_id: str,
        storage_account_id: str = None,
        retention_days: int = 90
    ):
        """
        Configure diagnostic settings to send logs to Log Analytics.
        
        Args:
            workspace_id: Log Analytics workspace resource ID
            storage_account_id: Optional storage account for long-term retention
            retention_days: Log retention period
        """
        # Define log categories
        log_categories = [
            LogSettings(
                category='OperationLogs',
                enabled=True,
                retention_policy=RetentionPolicy(
                    enabled=True,
                    days=retention_days
                )
            )
        ]
        
        # Define metrics
        metrics = [
            MetricSettings(
                category='AllMetrics',
                enabled=True,
                retention_policy=RetentionPolicy(
                    enabled=True,
                    days=retention_days
                )
            )
        ]
        
        # Create diagnostic settings
        diagnostic_settings = DiagnosticSettingsResource(
            workspace_id=workspace_id,
            storage_account_id=storage_account_id,
            logs=log_categories,
            metrics=metrics
        )
        
        result = self.monitor_client.diagnostic_settings.create_or_update(
            resource_uri=self.resource_id,
            name='search-diagnostics',
            parameters=diagnostic_settings
        )
        
        print(f"Diagnostic settings configured: {result.name}")
        
        return result
    
    def enable_metrics(self):
        """Enable platform metrics collection."""
        # Metrics are enabled by default for Azure AI Search
        # Available metrics:
        # - SearchLatency
        # - SearchQueriesPerSecond
        # - ThrottledSearchQueriesPercentage
        # - DocumentsProcessedCount
        # - SkillExecutionCount
        
        print("Platform metrics enabled")
```

### Platform Metrics

```python
from azure.monitor.query import MetricsQueryClient, MetricAggregationType
from datetime import datetime, timedelta

class MetricsCollector:
    """Collect and analyze Azure Monitor metrics."""
    
    def __init__(self, subscription_id: str):
        self.credential = DefaultAzureCredential()
        self.metrics_client = MetricsQueryClient(self.credential)
        self.subscription_id = subscription_id
    
    def query_search_latency(
        self,
        resource_group: str,
        search_service_name: str,
        hours_back: int = 24
    ) -> dict:
        """
        Query search latency metrics.
        
        Args:
            resource_group: Resource group name
            search_service_name: Search service name
            hours_back: Hours of historical data
            
        Returns:
            Latency statistics
        """
        resource_id = (
            f"/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{resource_group}"
            f"/providers/Microsoft.Search/searchServices/{search_service_name}"
        )
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Query latency metric
        response = self.metrics_client.query_resource(
            resource_uri=resource_id,
            metric_names=['SearchLatency'],
            timespan=(start_time, end_time),
            granularity=timedelta(minutes=5),
            aggregations=[
                MetricAggregationType.AVERAGE,
                MetricAggregationType.MAXIMUM,
                MetricAggregationType.MINIMUM
            ]
        )
        
        # Process results
        latencies = []
        for metric in response.metrics:
            for timeseries in metric.timeseries:
                for data_point in timeseries.data:
                    if data_point.average:
                        latencies.append(data_point.average)
        
        if latencies:
            return {
                'average_ms': sum(latencies) / len(latencies),
                'p50_ms': self._percentile(latencies, 50),
                'p95_ms': self._percentile(latencies, 95),
                'p99_ms': self._percentile(latencies, 99),
                'max_ms': max(latencies),
                'min_ms': min(latencies)
            }
        
        return {}
    
    def query_search_qps(
        self,
        resource_group: str,
        search_service_name: str,
        hours_back: int = 24
    ) -> dict:
        """Query queries per second metric."""
        resource_id = (
            f"/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{resource_group}"
            f"/providers/Microsoft.Search/searchServices/{search_service_name}"
        )
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        response = self.metrics_client.query_resource(
            resource_uri=resource_id,
            metric_names=['SearchQueriesPerSecond'],
            timespan=(start_time, end_time),
            granularity=timedelta(minutes=5),
            aggregations=[MetricAggregationType.AVERAGE, MetricAggregationType.MAXIMUM]
        )
        
        qps_values = []
        for metric in response.metrics:
            for timeseries in metric.timeseries:
                for data_point in timeseries.data:
                    if data_point.average:
                        qps_values.append(data_point.average)
        
        if qps_values:
            return {
                'average_qps': sum(qps_values) / len(qps_values),
                'peak_qps': max(qps_values),
                'min_qps': min(qps_values)
            }
        
        return {}
    
    def query_throttling_rate(
        self,
        resource_group: str,
        search_service_name: str,
        hours_back: int = 24
    ) -> float:
        """Query throttled queries percentage."""
        resource_id = (
            f"/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{resource_group}"
            f"/providers/Microsoft.Search/searchServices/{search_service_name}"
        )
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        response = self.metrics_client.query_resource(
            resource_uri=resource_id,
            metric_names=['ThrottledSearchQueriesPercentage'],
            timespan=(start_time, end_time),
            granularity=timedelta(minutes=5),
            aggregations=[MetricAggregationType.AVERAGE]
        )
        
        throttle_percentages = []
        for metric in response.metrics:
            for timeseries in metric.timeseries:
                for data_point in timeseries.data:
                    if data_point.average:
                        throttle_percentages.append(data_point.average)
        
        if throttle_percentages:
            return sum(throttle_percentages) / len(throttle_percentages)
        
        return 0.0
    
    def _percentile(self, data: list, percentile: int) -> float:
        """Calculate percentile value."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
```

---

## Metrics & KPIs

### Key Performance Indicators

```python
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

@dataclass
class SearchServiceKPIs:
    """Key performance indicators for search service."""
    
    # Availability
    uptime_percentage: float  # Target: 99.9%
    
    # Performance
    avg_latency_ms: float  # Target: < 200ms
    p95_latency_ms: float  # Target: < 500ms
    p99_latency_ms: float  # Target: < 1000ms
    
    # Throughput
    avg_qps: float
    peak_qps: float
    
    # Errors
    error_rate_percentage: float  # Target: < 1%
    throttle_rate_percentage: float  # Target: < 5%
    
    # Resource Utilization
    storage_used_gb: float
    storage_percentage: float
    
    # Cost
    estimated_monthly_cost: float
    cost_per_query: float
    
    timestamp: datetime

class KPITracker:
    """Track and report KPIs."""
    
    def __init__(self):
        self.kpi_history: List[SearchServiceKPIs] = []
    
    def calculate_kpis(
        self,
        metrics: dict,
        target_uptime: float = 99.9,
        target_p95_latency: float = 500
    ) -> SearchServiceKPIs:
        """
        Calculate KPIs from metrics.
        
        Args:
            metrics: Collected metrics dictionary
            target_uptime: Target uptime percentage
            target_p95_latency: Target P95 latency in ms
            
        Returns:
            SearchServiceKPIs object
        """
        kpis = SearchServiceKPIs(
            uptime_percentage=metrics.get('uptime_percentage', 0),
            avg_latency_ms=metrics.get('average_latency_ms', 0),
            p95_latency_ms=metrics.get('p95_latency_ms', 0),
            p99_latency_ms=metrics.get('p99_latency_ms', 0),
            avg_qps=metrics.get('average_qps', 0),
            peak_qps=metrics.get('peak_qps', 0),
            error_rate_percentage=metrics.get('error_rate', 0),
            throttle_rate_percentage=metrics.get('throttle_rate', 0),
            storage_used_gb=metrics.get('storage_used_gb', 0),
            storage_percentage=metrics.get('storage_percentage', 0),
            estimated_monthly_cost=metrics.get('monthly_cost', 0),
            cost_per_query=metrics.get('cost_per_query', 0),
            timestamp=datetime.utcnow()
        )
        
        self.kpi_history.append(kpis)
        
        return kpis
    
    def check_sla_compliance(self, kpis: SearchServiceKPIs) -> dict:
        """Check if KPIs meet SLA targets."""
        
        violations = []
        
        if kpis.uptime_percentage < 99.9:
            violations.append({
                'metric': 'Uptime',
                'target': 99.9,
                'actual': kpis.uptime_percentage,
                'severity': 'critical'
            })
        
        if kpis.p95_latency_ms > 500:
            violations.append({
                'metric': 'P95 Latency',
                'target': 500,
                'actual': kpis.p95_latency_ms,
                'severity': 'high'
            })
        
        if kpis.error_rate_percentage > 1:
            violations.append({
                'metric': 'Error Rate',
                'target': 1,
                'actual': kpis.error_rate_percentage,
                'severity': 'high'
            })
        
        if kpis.throttle_rate_percentage > 5:
            violations.append({
                'metric': 'Throttle Rate',
                'target': 5,
                'actual': kpis.throttle_rate_percentage,
                'severity': 'medium'
            })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
    
    def generate_kpi_report(self, days: int = 7) -> dict:
        """Generate KPI summary report."""
        
        recent_kpis = [
            kpi for kpi in self.kpi_history
            if (datetime.utcnow() - kpi.timestamp).days <= days
        ]
        
        if not recent_kpis:
            return {}
        
        return {
            'period_days': days,
            'avg_uptime': sum(k.uptime_percentage for k in recent_kpis) / len(recent_kpis),
            'avg_latency_ms': sum(k.avg_latency_ms for k in recent_kpis) / len(recent_kpis),
            'avg_p95_latency_ms': sum(k.p95_latency_ms for k in recent_kpis) / len(recent_kpis),
            'avg_error_rate': sum(k.error_rate_percentage for k in recent_kpis) / len(recent_kpis),
            'total_cost': sum(k.estimated_monthly_cost for k in recent_kpis),
            'kpi_count': len(recent_kpis)
        }
```

---

## Log Analytics

### KQL Queries

```python
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from datetime import timedelta

class LogAnalytics:
    """Query Log Analytics workspace."""
    
    def __init__(self, workspace_id: str):
        self.credential = DefaultAzureCredential()
        self.logs_client = LogsQueryClient(self.credential)
        self.workspace_id = workspace_id
    
    def query_operation_logs(
        self,
        hours_back: int = 24,
        operation_name: str = None
    ) -> list:
        """
        Query operation logs.
        
        Args:
            hours_back: Hours of historical data
            operation_name: Filter by operation name
            
        Returns:
            List of log entries
        """
        query = """
        AzureDiagnostics
        | where ResourceProvider == "MICROSOFT.SEARCH"
        | where TimeGenerated > ago({hours}h)
        """.format(hours=hours_back)
        
        if operation_name:
            query += f"\n| where OperationName == '{operation_name}'"
        
        query += """
        | project TimeGenerated, OperationName, ResultType, DurationMs, 
                  CallerIpAddress, Resource, properties_s
        | order by TimeGenerated desc
        """
        
        response = self.logs_client.query_workspace(
            workspace_id=self.workspace_id,
            query=query,
            timespan=timedelta(hours=hours_back)
        )
        
        if response.status == LogsQueryStatus.SUCCESS:
            return [dict(row) for row in response.tables[0].rows]
        
        return []
    
    def query_failed_requests(self, hours_back: int = 24) -> list:
        """Query failed search requests."""
        
        query = """
        AzureDiagnostics
        | where ResourceProvider == "MICROSOFT.SEARCH"
        | where TimeGenerated > ago({hours}h)
        | where ResultType != "Success"
        | project TimeGenerated, OperationName, ResultType, 
                  ResultDescription, CallerIpAddress
        | order by TimeGenerated desc
        """.format(hours=hours_back)
        
        response = self.logs_client.query_workspace(
            workspace_id=self.workspace_id,
            query=query,
            timespan=timedelta(hours=hours_back)
        )
        
        if response.status == LogsQueryStatus.SUCCESS:
            return [dict(row) for row in response.tables[0].rows]
        
        return []
    
    def query_slow_queries(
        self,
        threshold_ms: int = 1000,
        hours_back: int = 24
    ) -> list:
        """Query slow search requests."""
        
        query = """
        AzureDiagnostics
        | where ResourceProvider == "MICROSOFT.SEARCH"
        | where TimeGenerated > ago({hours}h)
        | where OperationName == "Query.Search"
        | where DurationMs > {threshold}
        | project TimeGenerated, DurationMs, Query_s, 
                  IndexName_s, CallerIpAddress
        | order by DurationMs desc
        | take 100
        """.format(hours=hours_back, threshold=threshold_ms)
        
        response = self.logs_client.query_workspace(
            workspace_id=self.workspace_id,
            query=query,
            timespan=timedelta(hours=hours_back)
        )
        
        if response.status == LogsQueryStatus.SUCCESS:
            return [dict(row) for row in response.tables[0].rows]
        
        return []
    
    def analyze_query_patterns(self, hours_back: int = 24) -> dict:
        """Analyze search query patterns."""
        
        query = """
        AzureDiagnostics
        | where ResourceProvider == "MICROSOFT.SEARCH"
        | where TimeGenerated > ago({hours}h)
        | where OperationName == "Query.Search"
        | summarize 
            QueryCount = count(),
            AvgDuration = avg(DurationMs),
            P95Duration = percentile(DurationMs, 95)
          by IndexName_s
        | order by QueryCount desc
        """.format(hours=hours_back)
        
        response = self.logs_client.query_workspace(
            workspace_id=self.workspace_id,
            query=query,
            timespan=timedelta(hours=hours_back)
        )
        
        if response.status == LogsQueryStatus.SUCCESS:
            results = {}
            for row in response.tables[0].rows:
                index_name = row[0]
                results[index_name] = {
                    'query_count': row[1],
                    'avg_duration_ms': row[2],
                    'p95_duration_ms': row[3]
                }
            return results
        
        return {}
```

---

## Alerting

### Alert Rules

```python
from azure.mgmt.monitor.models import (
    MetricAlertResource,
    MetricAlertCriteria,
    MetricAlertSingleResourceMultipleMetricCriteria,
    MetricCriteria,
    DynamicMetricCriteria
)

class AlertManager:
    """Manage Azure Monitor alert rules."""
    
    def __init__(
        self,
        subscription_id: str,
        resource_group: str
    ):
        self.credential = DefaultAzureCredential()
        self.monitor_client = MonitorManagementClient(
            credential=self.credential,
            subscription_id=subscription_id
        )
        self.resource_group = resource_group
        self.subscription_id = subscription_id
    
    def create_latency_alert(
        self,
        search_service_name: str,
        threshold_ms: float = 500,
        severity: int = 2,
        action_group_id: str = None
    ):
        """
        Create alert for high search latency.
        
        Args:
            search_service_name: Search service name
            threshold_ms: Latency threshold in milliseconds
            severity: Alert severity (0-4, 0=Critical)
            action_group_id: Action group resource ID for notifications
        """
        resource_id = (
            f"/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Search/searchServices/{search_service_name}"
        )
        
        criteria = MetricAlertSingleResourceMultipleMetricCriteria(
            all_of=[
                MetricCriteria(
                    name='HighLatency',
                    metric_name='SearchLatency',
                    metric_namespace='Microsoft.Search/searchServices',
                    time_aggregation='Average',
                    operator='GreaterThan',
                    threshold=threshold_ms
                )
            ]
        )
        
        alert = MetricAlertResource(
            location='global',
            description=f'Alert when search latency exceeds {threshold_ms}ms',
            severity=severity,
            enabled=True,
            scopes=[resource_id],
            evaluation_frequency=timedelta(minutes=5),
            window_size=timedelta(minutes=5),
            criteria=criteria,
            actions=[{'action_group_id': action_group_id}] if action_group_id else []
        )
        
        result = self.monitor_client.metric_alerts.create_or_update(
            resource_group_name=self.resource_group,
            rule_name=f'{search_service_name}-high-latency',
            parameters=alert
        )
        
        print(f"Created latency alert: {result.name}")
        
        return result
    
    def create_availability_alert(
        self,
        search_service_name: str,
        threshold_percent: float = 99.9,
        action_group_id: str = None
    ):
        """Create alert for service availability."""
        
        resource_id = (
            f"/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Search/searchServices/{search_service_name}"
        )
        
        # Use dynamic threshold for availability
        criteria = MetricAlertSingleResourceMultipleMetricCriteria(
            all_of=[
                DynamicMetricCriteria(
                    name='LowAvailability',
                    metric_name='Availability',
                    metric_namespace='Microsoft.Search/searchServices',
                    time_aggregation='Average',
                    operator='LessThan',
                    alert_sensitivity='Medium',
                    failing_periods={
                        'number_of_evaluation_periods': 4,
                        'min_failing_periods_to_alert': 3
                    }
                )
            ]
        )
        
        alert = MetricAlertResource(
            location='global',
            description='Alert when service availability drops',
            severity=0,  # Critical
            enabled=True,
            scopes=[resource_id],
            evaluation_frequency=timedelta(minutes=5),
            window_size=timedelta(minutes=15),
            criteria=criteria,
            actions=[{'action_group_id': action_group_id}] if action_group_id else []
        )
        
        result = self.monitor_client.metric_alerts.create_or_update(
            resource_group_name=self.resource_group,
            rule_name=f'{search_service_name}-low-availability',
            parameters=alert
        )
        
        print(f"Created availability alert: {result.name}")
        
        return result
    
    def create_throttling_alert(
        self,
        search_service_name: str,
        threshold_percent: float = 5,
        action_group_id: str = None
    ):
        """Create alert for query throttling."""
        
        resource_id = (
            f"/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Search/searchServices/{search_service_name}"
        )
        
        criteria = MetricAlertSingleResourceMultipleMetricCriteria(
            all_of=[
                MetricCriteria(
                    name='HighThrottling',
                    metric_name='ThrottledSearchQueriesPercentage',
                    metric_namespace='Microsoft.Search/searchServices',
                    time_aggregation='Average',
                    operator='GreaterThan',
                    threshold=threshold_percent
                )
            ]
        )
        
        alert = MetricAlertResource(
            location='global',
            description=f'Alert when throttling exceeds {threshold_percent}%',
            severity=1,  # High
            enabled=True,
            scopes=[resource_id],
            evaluation_frequency=timedelta(minutes=5),
            window_size=timedelta(minutes=15),
            criteria=criteria,
            actions=[{'action_group_id': action_group_id}] if action_group_id else []
        )
        
        result = self.monitor_client.metric_alerts.create_or_update(
            resource_group_name=self.resource_group,
            rule_name=f'{search_service_name}-high-throttling',
            parameters=alert
        )
        
        print(f"Created throttling alert: {result.name}")
        
        return result
    
    def create_cost_alert(
        self,
        monthly_budget: float,
        threshold_percent: int = 80,
        action_group_id: str = None
    ):
        """
        Create budget alert for cost management.
        
        Args:
            monthly_budget: Monthly budget in USD
            threshold_percent: Alert threshold percentage
            action_group_id: Action group for notifications
        """
        # Cost alerts are configured through Azure Cost Management
        # This is a simplified example
        
        print(f"Cost alert configured for ${monthly_budget}/month at {threshold_percent}%")
```

### Action Groups

```python
from azure.mgmt.monitor.models import (
    ActionGroupResource,
    EmailReceiver,
    SmsReceiver,
    WebhookReceiver,
    AzureFunctionReceiver
)

class ActionGroupManager:
    """Manage alert action groups."""
    
    def __init__(self, subscription_id: str, resource_group: str):
        self.credential = DefaultAzureCredential()
        self.monitor_client = MonitorManagementClient(
            credential=self.credential,
            subscription_id=subscription_id
        )
        self.resource_group = resource_group
    
    def create_action_group(
        self,
        name: str,
        short_name: str,
        email_receivers: list = None,
        sms_receivers: list = None,
        webhook_url: str = None
    ) -> ActionGroupResource:
        """
        Create action group for alert notifications.
        
        Args:
            name: Action group name
            short_name: Short name (max 12 chars)
            email_receivers: List of email addresses
            sms_receivers: List of (name, country_code, phone_number) tuples
            webhook_url: Optional webhook URL
            
        Returns:
            Created action group
        """
        email_receiver_list = []
        if email_receivers:
            for i, email in enumerate(email_receivers):
                email_receiver_list.append(
                    EmailReceiver(
                        name=f'email{i}',
                        email_address=email,
                        use_common_alert_schema=True
                    )
                )
        
        sms_receiver_list = []
        if sms_receivers:
            for name, country_code, phone in sms_receivers:
                sms_receiver_list.append(
                    SmsReceiver(
                        name=name,
                        country_code=country_code,
                        phone_number=phone
                    )
                )
        
        webhook_receiver_list = []
        if webhook_url:
            webhook_receiver_list.append(
                WebhookReceiver(
                    name='webhook',
                    service_uri=webhook_url,
                    use_common_alert_schema=True
                )
            )
        
        action_group = ActionGroupResource(
            location='global',
            group_short_name=short_name[:12],
            enabled=True,
            email_receivers=email_receiver_list,
            sms_receivers=sms_receiver_list,
            webhook_receivers=webhook_receiver_list
        )
        
        result = self.monitor_client.action_groups.create_or_update(
            resource_group_name=self.resource_group,
            action_group_name=name,
            action_group=action_group
        )
        
        print(f"Created action group: {result.name}")
        
        return result
```

---

## Dashboards

### Azure Dashboard

```python
import json

class DashboardGenerator:
    """Generate Azure Dashboard configurations."""
    
    def create_search_dashboard(
        self,
        search_service_resource_id: str,
        workspace_id: str
    ) -> dict:
        """
        Create comprehensive search monitoring dashboard.
        
        Args:
            search_service_resource_id: Search service resource ID
            workspace_id: Log Analytics workspace ID
            
        Returns:
            Dashboard JSON configuration
        """
        dashboard = {
            "properties": {
                "lenses": [
                    # Latency chart
                    {
                        "order": 0,
                        "parts": [
                            {
                                "position": {"x": 0, "y": 0, "colSpan": 6, "rowSpan": 4},
                                "metadata": {
                                    "type": "Extension/Microsoft_Azure_Monitoring/PartType/MetricsChartPart",
                                    "inputs": [{
                                        "name": "resourceId",
                                        "value": search_service_resource_id
                                    }],
                                    "settings": {
                                        "title": "Search Latency (P95)",
                                        "metrics": [{
                                            "name": "SearchLatency",
                                            "aggregationType": 95,
                                            "namespace": "Microsoft.Search/searchServices"
                                        }]
                                    }
                                }
                            }
                        ]
                    },
                    # QPS chart
                    {
                        "order": 1,
                        "parts": [
                            {
                                "position": {"x": 6, "y": 0, "colSpan": 6, "rowSpan": 4},
                                "metadata": {
                                    "type": "Extension/Microsoft_Azure_Monitoring/PartType/MetricsChartPart",
                                    "inputs": [{
                                        "name": "resourceId",
                                        "value": search_service_resource_id
                                    }],
                                    "settings": {
                                        "title": "Queries Per Second",
                                        "metrics": [{
                                            "name": "SearchQueriesPerSecond",
                                            "aggregationType": 4,  # Average
                                            "namespace": "Microsoft.Search/searchServices"
                                        }]
                                    }
                                }
                            }
                        ]
                    },
                    # Error rate
                    {
                        "order": 2,
                        "parts": [
                            {
                                "position": {"x": 0, "y": 4, "colSpan": 6, "rowSpan": 4},
                                "metadata": {
                                    "type": "Extension/Microsoft_Azure_Monitoring/PartType/LogsChartPart",
                                    "inputs": [{
                                        "name": "workspaceId",
                                        "value": workspace_id
                                    }],
                                    "settings": {
                                        "title": "Error Rate",
                                        "query": """
                                        AzureDiagnostics
                                        | where ResourceProvider == "MICROSOFT.SEARCH"
                                        | summarize 
                                            Total = count(),
                                            Errors = countif(ResultType != "Success")
                                          by bin(TimeGenerated, 5m)
                                        | extend ErrorRate = (Errors * 100.0) / Total
                                        | project TimeGenerated, ErrorRate
                                        """
                                    }
                                }
                            }
                        ]
                    }
                ]
            }
        }
        
        return dashboard
    
    def export_dashboard_json(self, dashboard: dict, filename: str):
        """Export dashboard to JSON file."""
        with open(filename, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        print(f"Dashboard exported to {filename}")
```

---

## Application Insights

### Custom Telemetry

```python
from applicationinsights import TelemetryClient
from applicationinsights.logging import LoggingHandler
import logging

class SearchTelemetry:
    """Application Insights integration for search applications."""
    
    def __init__(self, instrumentation_key: str):
        self.telemetry_client = TelemetryClient(instrumentation_key)
        
        # Configure logging
        logging_handler = LoggingHandler(instrumentation_key)
        logging.getLogger().addHandler(logging_handler)
    
    def track_search_query(
        self,
        query: str,
        index_name: str,
        duration_ms: float,
        result_count: int,
        user_id: str = None
    ):
        """Track search query execution."""
        
        properties = {
            'query': query,
            'index': index_name,
            'resultCount': result_count
        }
        
        if user_id:
            properties['userId'] = user_id
        
        self.telemetry_client.track_event(
            'SearchQuery',
            properties=properties,
            measurements={'durationMs': duration_ms}
        )
    
    def track_search_error(
        self,
        query: str,
        error_message: str,
        error_type: str
    ):
        """Track search errors."""
        
        self.telemetry_client.track_exception(
            properties={
                'query': query,
                'errorType': error_type,
                'errorMessage': error_message
            }
        )
    
    def track_metric(self, name: str, value: float):
        """Track custom metric."""
        self.telemetry_client.track_metric(name, value)
    
    def flush(self):
        """Flush telemetry data."""
        self.telemetry_client.flush()
```

---

## Anomaly Detection

### Statistical Anomaly Detection

```python
import numpy as np
from typing import List, Tuple
from datetime import datetime

class AnomalyDetector:
    """Detect anomalies in search metrics."""
    
    def __init__(self, sensitivity: float = 3.0):
        """
        Initialize anomaly detector.
        
        Args:
            sensitivity: Standard deviation multiplier for anomaly threshold
        """
        self.sensitivity = sensitivity
    
    def detect_anomalies_zscore(
        self,
        values: List[float],
        timestamps: List[datetime] = None
    ) -> List[Tuple[int, float]]:
        """
        Detect anomalies using Z-score method.
        
        Args:
            values: Metric values
            timestamps: Optional timestamps
            
        Returns:
            List of (index, value) tuples for anomalies
        """
        if len(values) < 3:
            return []
        
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std == 0:
            return []
        
        z_scores = np.abs((arr - mean) / std)
        anomalies = []
        
        for i, z_score in enumerate(z_scores):
            if z_score > self.sensitivity:
                anomalies.append((i, values[i]))
        
        return anomalies
    
    def detect_anomalies_moving_average(
        self,
        values: List[float],
        window_size: int = 10,
        threshold_factor: float = 2.0
    ) -> List[Tuple[int, float]]:
        """
        Detect anomalies using moving average method.
        
        Args:
            values: Metric values
            window_size: Moving average window
            threshold_factor: Deviation threshold multiplier
            
        Returns:
            List of (index, value) tuples for anomalies
        """
        if len(values) < window_size:
            return []
        
        anomalies = []
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            moving_avg = np.mean(window)
            moving_std = np.std(window)
            
            if moving_std > 0:
                deviation = abs(values[i] - moving_avg) / moving_std
                
                if deviation > threshold_factor:
                    anomalies.append((i, values[i]))
        
        return anomalies
    
    def detect_sudden_change(
        self,
        values: List[float],
        change_threshold_percent: float = 50
    ) -> List[Tuple[int, float, float]]:
        """
        Detect sudden percentage changes.
        
        Args:
            values: Metric values
            change_threshold_percent: Threshold for sudden change
            
        Returns:
            List of (index, previous_value, current_value) tuples
        """
        changes = []
        
        for i in range(1, len(values)):
            if values[i-1] > 0:
                percent_change = abs((values[i] - values[i-1]) / values[i-1]) * 100
                
                if percent_change > change_threshold_percent:
                    changes.append((i, values[i-1], values[i]))
        
        return changes
```

---

## Incident Response

### Incident Management

```python
from enum import Enum
from dataclasses import dataclass
from typing import List
from datetime import datetime

class IncidentSeverity(Enum):
    """Incident severity levels."""
    CRITICAL = 1  # Service down
    HIGH = 2      # Significant degradation
    MEDIUM = 3    # Moderate impact
    LOW = 4       # Minor issues

class IncidentStatus(Enum):
    """Incident status."""
    DETECTED = 'detected'
    INVESTIGATING = 'investigating'
    IDENTIFIED = 'identified'
    RESOLVING = 'resolving'
    RESOLVED = 'resolved'
    CLOSED = 'closed'

@dataclass
class Incident:
    """Incident record."""
    id: str
    title: str
    severity: IncidentSeverity
    status: IncidentStatus
    detected_at: datetime
    resolved_at: datetime = None
    description: str = ""
    root_cause: str = ""
    remediation: str = ""
    affected_services: List[str] = None

class IncidentManager:
    """Manage incidents and response procedures."""
    
    def __init__(self):
        self.incidents: List[Incident] = []
    
    def create_incident(
        self,
        title: str,
        severity: IncidentSeverity,
        description: str,
        affected_services: List[str] = None
    ) -> Incident:
        """Create new incident."""
        
        incident = Incident(
            id=f"INC-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            title=title,
            severity=severity,
            status=IncidentStatus.DETECTED,
            detected_at=datetime.utcnow(),
            description=description,
            affected_services=affected_services or []
        )
        
        self.incidents.append(incident)
        
        # Escalate based on severity
        self._escalate_incident(incident)
        
        return incident
    
    def update_incident_status(
        self,
        incident_id: str,
        status: IncidentStatus,
        notes: str = ""
    ):
        """Update incident status."""
        
        for incident in self.incidents:
            if incident.id == incident_id:
                incident.status = status
                
                if status == IncidentStatus.RESOLVED:
                    incident.resolved_at = datetime.utcnow()
                
                print(f"Incident {incident_id} updated to {status.value}")
                break
    
    def _escalate_incident(self, incident: Incident):
        """Escalate incident based on severity."""
        
        if incident.severity == IncidentSeverity.CRITICAL:
            # Page on-call engineer immediately
            print(f"🚨 CRITICAL INCIDENT: {incident.title}")
            print("Paging on-call engineer...")
            # Send PagerDuty/OpsGenie alert
            
        elif incident.severity == IncidentSeverity.HIGH:
            # Notify team via Slack/Teams
            print(f"⚠️ HIGH SEVERITY: {incident.title}")
            print("Notifying engineering team...")
        
        else:
            # Create ticket for investigation
            print(f"ℹ️ New incident logged: {incident.title}")
    
    def generate_postmortem(self, incident: Incident) -> dict:
        """Generate incident postmortem report."""
        
        duration = None
        if incident.resolved_at:
            duration = (incident.resolved_at - incident.detected_at).total_seconds() / 60
        
        return {
            'incident_id': incident.id,
            'title': incident.title,
            'severity': incident.severity.name,
            'detected_at': incident.detected_at.isoformat(),
            'resolved_at': incident.resolved_at.isoformat() if incident.resolved_at else None,
            'duration_minutes': duration,
            'root_cause': incident.root_cause,
            'remediation': incident.remediation,
            'affected_services': incident.affected_services,
            'lessons_learned': [],
            'action_items': []
        }
```

---

## SLA Tracking

### SLA Calculator

```python
from datetime import datetime, timedelta
from typing import List, Dict

class SLATracker:
    """Track SLA compliance."""
    
    def __init__(
        self,
        target_uptime: float = 99.9,
        target_p95_latency: float = 500
    ):
        self.target_uptime = target_uptime
        self.target_p95_latency = target_p95_latency
        self.downtime_events: List[Dict] = []
    
    def calculate_uptime(
        self,
        period_days: int = 30
    ) -> dict:
        """
        Calculate uptime percentage.
        
        Args:
            period_days: Calculation period
            
        Returns:
            Uptime statistics
        """
        total_minutes = period_days * 24 * 60
        
        # Calculate downtime from events
        downtime_minutes = 0
        for event in self.downtime_events:
            if event['end_time']:
                duration = (event['end_time'] - event['start_time']).total_seconds() / 60
                downtime_minutes += duration
        
        uptime_minutes = total_minutes - downtime_minutes
        uptime_percentage = (uptime_minutes / total_minutes) * 100
        
        # SLA credit calculation (example)
        sla_credit = 0
        if uptime_percentage < 99.9:
            sla_credit = 10  # 10% credit
        if uptime_percentage < 99:
            sla_credit = 25  # 25% credit
        
        return {
            'period_days': period_days,
            'total_minutes': total_minutes,
            'uptime_minutes': uptime_minutes,
            'downtime_minutes': downtime_minutes,
            'uptime_percentage': uptime_percentage,
            'target_uptime': self.target_uptime,
            'meets_sla': uptime_percentage >= self.target_uptime,
            'sla_credit_percentage': sla_credit
        }
    
    def record_downtime(
        self,
        start_time: datetime,
        end_time: datetime = None,
        reason: str = ""
    ):
        """Record downtime event."""
        
        self.downtime_events.append({
            'start_time': start_time,
            'end_time': end_time,
            'reason': reason
        })
    
    def get_sla_report(self, months: int = 1) -> dict:
        """Generate SLA compliance report."""
        
        uptime_stats = self.calculate_uptime(months * 30)
        
        return {
            'reporting_period_months': months,
            'uptime_percentage': uptime_stats['uptime_percentage'],
            'target_uptime': self.target_uptime,
            'sla_compliant': uptime_stats['meets_sla'],
            'downtime_events': len(self.downtime_events),
            'total_downtime_minutes': uptime_stats['downtime_minutes'],
            'sla_credit': uptime_stats['sla_credit_percentage']
        }
```

---

## Best Practices

### 1. Monitoring Strategy

- **Monitor Key Metrics**: Latency, QPS, throttling, error rate
- **Set Baseline**: Establish normal performance baselines
- **Use Multiple Channels**: Metrics + Logs + Application Insights
- **Enable Diagnostic Settings**: Always send logs to Log Analytics
- **Retain Data**: Keep logs for 90+ days for trend analysis

### 2. Alerting

- **Alert on Symptoms**: Focus on user-facing issues
- **Set Meaningful Thresholds**: Based on SLA targets
- **Reduce Noise**: Avoid alert fatigue with proper thresholds
- **Use Action Groups**: Configure multiple notification channels
- **Test Alerts**: Regularly verify alerting is working

### 3. Dashboards

- **Create Role-Specific Dashboards**: Different views for different teams
- **Include Context**: Show baselines and targets
- **Update Regularly**: Evolve dashboards with system changes
- **Share Widely**: Make dashboards accessible to all stakeholders

### 4. Incident Response

- **Define Severity Levels**: Clear criteria for each level
- **Establish Runbooks**: Document response procedures
- **Practice**: Conduct incident response drills
- **Learn**: Write postmortems for all incidents
- **Track Metrics**: MTTR, MTTD, incident frequency

### 5. SLA Management

- **Define Clear SLAs**: Specific uptime and performance targets
- **Measure Continuously**: Track SLA compliance in real-time
- **Report Transparently**: Share SLA reports with stakeholders
- **Plan for Failure**: SLA targets should account for planned maintenance

---

For CI/CD integration with monitoring, see [CI/CD Pipelines (Page 22)](./22-cicd-pipelines.md).

For security monitoring, see [Security & Compliance (Page 21)](./21-security-compliance.md).
