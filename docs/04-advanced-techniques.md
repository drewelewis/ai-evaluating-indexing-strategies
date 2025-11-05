# Azure Advanced Search Techniques

This document covers advanced techniques for optimizing Azure AI Search performance, including monitoring, troubleshooting, and enterprise-scale deployment patterns.

## 🔍 Advanced Azure AI Search Patterns

### 1. Query Optimization and Performance Tuning

#### Query Analysis with Azure Monitor
```python
import json
from azure.monitor.query import LogsQueryClient
from azure.identity import DefaultAzureCredential

class AzureSearchQueryAnalyzer:
    def __init__(self, workspace_id):
        credential = DefaultAzureCredential()
        self.logs_client = LogsQueryClient(credential)
        self.workspace_id = workspace_id
    
    def analyze_slow_queries(self, days=7, threshold_ms=1000):
        """Identify and analyze slow-performing queries."""
        
        kusto_query = f"""
        AzureDiagnostics
        | where Category == "OperationLogs"
        | where TimeGenerated > ago({days}d)
        | where DurationMs > {threshold_ms}
        | extend QueryType = extract("queryType=([^&]*)", 1, requestUri_s)
        | extend SearchText = extract("search=([^&]*)", 1, requestUri_s)
        | summarize 
            AvgDuration = avg(DurationMs),
            MaxDuration = max(DurationMs),
            QueryCount = count()
        by SearchText, QueryType
        | order by AvgDuration desc
        | take 50
        """
        
        response = self.logs_client.query_workspace(
            workspace_id=self.workspace_id,
            query=kusto_query
        )
        
        slow_queries = []
        for row in response.tables[0].rows:
            slow_queries.append({
                "query": row[0],
                "query_type": row[1],
                "avg_duration_ms": row[2],
                "max_duration_ms": row[3],
                "frequency": row[4]
            })
        
        return slow_queries
    
    def analyze_query_patterns(self, days=30):
        """Analyze query patterns and usage trends."""
        
        kusto_query = f"""
        AzureDiagnostics
        | where Category == "OperationLogs"
        | where TimeGenerated > ago({days}d)
        | extend QueryType = extract("queryType=([^&]*)", 1, requestUri_s)
        | extend Hour = bin(TimeGenerated, 1h)
        | summarize 
            QueryCount = count(),
            AvgDuration = avg(DurationMs),
            UniqueQueries = dcount(requestUri_s)
        by Hour, QueryType
        | order by Hour desc
        """
        
        response = self.logs_client.query_workspace(
            workspace_id=self.workspace_id,
            query=kusto_query
        )
        
        return self._format_query_patterns(response.tables[0].rows)
    
    def _format_query_patterns(self, rows):
        """Format query pattern analysis results."""
        patterns = []
        for row in rows:
            patterns.append({
                "hour": row[0],
                "query_type": row[1],
                "query_count": row[2],
                "avg_duration_ms": row[3],
                "unique_queries": row[4]
            })
        return patterns
```

#### Intelligent Query Routing
```python
class AzureSearchRouter:
    def __init__(self, search_client, embedder):
        self.search_client = search_client
        self.embedder = embedder
        self.routing_rules = self._load_routing_rules()
    
    def route_query(self, query, user_context=None):
        """Intelligently route queries to optimal search strategy."""
        
        query_features = self._extract_query_features(query, user_context)
        optimal_strategy = self._predict_optimal_strategy(query_features)
        
        if optimal_strategy == "keyword":
            return self._keyword_search(query)
        elif optimal_strategy == "vector":
            return self._vector_search(query)
        elif optimal_strategy == "hybrid":
            return self._hybrid_search(query)
        elif optimal_strategy == "semantic":
            return self._semantic_search(query)
        else:
            # Default to hybrid
            return self._hybrid_search(query)
    
    def _extract_query_features(self, query, user_context):
        """Extract features for routing decision."""
        features = {
            "query_length": len(query.split()),
            "has_quotes": '"' in query,
            "has_wildcards": '*' in query or '?' in query,
            "question_words": len([w for w in query.lower().split() 
                                 if w in ["what", "how", "why", "when", "where", "who"]]),
            "is_question": query.strip().endswith("?"),
            "has_numbers": any(char.isdigit() for char in query),
            "is_short": len(query.split()) <= 3,
            "user_type": user_context.get("user_type", "general") if user_context else "general"
        }
        
        return features
    
    def _predict_optimal_strategy(self, features):
        """Predict optimal search strategy based on features."""
        
        # Rule-based routing (could be replaced with ML model)
        if features["has_quotes"] or features["has_wildcards"]:
            return "keyword"
        
        if features["question_words"] > 0 or features["is_question"]:
            return "semantic"
        
        if features["is_short"] and features["has_numbers"]:
            return "keyword"
        
        if features["query_length"] > 8:
            return "semantic"
        
        # Default to hybrid for balanced performance
        return "hybrid"
```

### 2. Advanced Indexing Strategies

#### Dynamic Index Management
```python
class AzureDynamicIndexManager:
    def __init__(self, search_admin_client):
        self.admin_client = search_admin_client
    
    def create_time_partitioned_indexes(self, base_index_name, time_periods):
        """Create time-partitioned indexes for large datasets."""
        
        created_indexes = []
        
        for period in time_periods:
            index_name = f"{base_index_name}-{period}"
            
            index_schema = {
                "name": index_name,
                "fields": [
                    {
                        "name": "id",
                        "type": "Edm.String",
                        "key": True,
                        "searchable": False
                    },
                    {
                        "name": "content",
                        "type": "Edm.String",
                        "searchable": True,
                        "analyzer": "en.microsoft"
                    },
                    {
                        "name": "timestamp",
                        "type": "Edm.DateTimeOffset",
                        "searchable": False,
                        "filterable": True,
                        "sortable": True
                    },
                    {
                        "name": "content_vector",
                        "type": "Collection(Edm.Single)",
                        "searchable": True,
                        "vectorSearchDimensions": 1536,
                        "vectorSearchProfileName": "vector-profile"
                    }
                ],
                "vectorSearch": {
                    "algorithms": [
                        {
                            "name": "hnsw-algorithm",
                            "kind": "hnsw",
                            "hnswParameters": {
                                "metric": "cosine",
                                "m": 16,
                                "efConstruction": 400,
                                "efSearch": 500
                            }
                        }
                    ],
                    "profiles": [
                        {
                            "name": "vector-profile",
                            "algorithm": "hnsw-algorithm"
                        }
                    ]
                }
            }
            
            try:
                self.admin_client.create_index(index_schema)
                created_indexes.append(index_name)
                print(f"Created index: {index_name}")
            except Exception as e:
                print(f"Error creating index {index_name}: {e}")
        
        return created_indexes
    
    def setup_index_aliases(self, alias_name, target_indexes):
        """Set up index aliases for seamless index switching."""
        
        # Azure AI Search doesn't have direct alias support
        # This would be implemented through application-level routing
        alias_config = {
            "alias": alias_name,
            "targets": target_indexes,
            "routing_strategy": "round_robin",
            "failover_enabled": True
        }
        
        return alias_config
```

#### Custom Analyzers for Domain-Specific Content
```python
def create_medical_analyzer():
    """Create custom analyzer for medical content."""
    return {
        "name": "medical_analyzer",
        "@odata.type": "#Microsoft.Azure.Search.CustomAnalyzer",
        "tokenizer": "standard_v2",
        "tokenFilters": [
            "lowercase",
            "medical_stop_filter",
            "medical_synonym_filter",
            "stemmer"
        ],
        "charFilters": [
            "medical_char_filter"
        ]
    }

def create_legal_analyzer():
    """Create custom analyzer for legal content."""
    return {
        "name": "legal_analyzer", 
        "@odata.type": "#Microsoft.Azure.Search.CustomAnalyzer",
        "tokenizer": "keyword_v2",
        "tokenFilters": [
            "lowercase",
            "legal_acronym_filter",
            "citation_filter"
        ]
    }

def create_financial_analyzer():
    """Create custom analyzer for financial content."""
    return {
        "name": "financial_analyzer",
        "@odata.type": "#Microsoft.Azure.Search.CustomAnalyzer", 
        "tokenizer": "standard_v2",
        "tokenFilters": [
            "lowercase",
            "financial_number_filter",
            "currency_filter",
            "financial_stop_filter"
        ]
    }
```

### 3. Enterprise Monitoring and Alerting

#### Comprehensive Monitoring Setup
```python
class AzureSearchMonitoring:
    def __init__(self, resource_group, search_service_name):
        self.resource_group = resource_group
        self.search_service_name = search_service_name
    
    def setup_performance_alerts(self):
        """Set up comprehensive performance monitoring alerts."""
        
        alert_rules = [
            {
                "name": "HighLatencyAlert",
                "condition": "AverageDuration > 1000ms",
                "threshold": 1000,
                "frequency": "PT5M",  # Check every 5 minutes
                "action": "email_and_webhook"
            },
            {
                "name": "LowSuccessRateAlert", 
                "condition": "SuccessRate < 95%",
                "threshold": 0.95,
                "frequency": "PT1M",
                "action": "critical_alert"
            },
            {
                "name": "HighErrorRateAlert",
                "condition": "ErrorRate > 5%", 
                "threshold": 0.05,
                "frequency": "PT1M",
                "action": "immediate_notification"
            },
            {
                "name": "StorageQuotaAlert",
                "condition": "StorageUsed > 80%",
                "threshold": 0.8,
                "frequency": "PT1H",
                "action": "capacity_planning"
            }
        ]
        
        return alert_rules
    
    def create_custom_dashboard(self):
        """Create custom Azure Monitor dashboard for search analytics."""
        
        dashboard_config = {
            "name": "AzureSearchAnalytics",
            "tiles": [
                {
                    "title": "Query Volume",
                    "type": "line_chart",
                    "query": "AzureDiagnostics | where Category == 'OperationLogs' | summarize count() by bin(TimeGenerated, 1h)",
                    "time_range": "24h"
                },
                {
                    "title": "Average Latency",
                    "type": "line_chart", 
                    "query": "AzureDiagnostics | where Category == 'OperationLogs' | summarize avg(DurationMs) by bin(TimeGenerated, 5m)",
                    "time_range": "24h"
                },
                {
                    "title": "Top Slow Queries",
                    "type": "table",
                    "query": "AzureDiagnostics | where DurationMs > 500 | top 10 by DurationMs",
                    "time_range": "1h"
                },
                {
                    "title": "Error Distribution",
                    "type": "pie_chart",
                    "query": "AzureDiagnostics | where resultType_s != 'Success' | summarize count() by resultType_s",
                    "time_range": "24h"
                }
            ]
        }
        
        return dashboard_config
```

#### Real-time Performance Monitoring
```python
class RealTimeMonitor:
    def __init__(self, search_client):
        self.search_client = search_client
        self.metrics_buffer = []
    
    async def monitor_search_performance(self, sample_queries, interval_seconds=60):
        """Monitor search performance in real-time."""
        
        while True:
            performance_data = await self._collect_performance_metrics(sample_queries)
            self.metrics_buffer.append(performance_data)
            
            # Keep only last 100 measurements
            if len(self.metrics_buffer) > 100:
                self.metrics_buffer.pop(0)
            
            # Check for anomalies
            anomalies = self._detect_anomalies(performance_data)
            if anomalies:
                await self._send_alerts(anomalies)
            
            await asyncio.sleep(interval_seconds)
    
    async def _collect_performance_metrics(self, queries):
        """Collect current performance metrics."""
        
        metrics = {
            "timestamp": datetime.utcnow(),
            "query_latencies": [],
            "success_rate": 0,
            "total_queries": len(queries)
        }
        
        successful_queries = 0
        
        for query in queries:
            start_time = time.time()
            try:
                results = list(self.search_client.search(query, top=10))
                latency = (time.time() - start_time) * 1000  # Convert to ms
                metrics["query_latencies"].append(latency)
                successful_queries += 1
            except Exception as e:
                print(f"Query failed: {query}, Error: {e}")
        
        metrics["success_rate"] = successful_queries / len(queries)
        metrics["avg_latency"] = sum(metrics["query_latencies"]) / len(metrics["query_latencies"]) if metrics["query_latencies"] else 0
        
        return metrics
    
    def _detect_anomalies(self, current_metrics):
        """Detect performance anomalies."""
        
        if len(self.metrics_buffer) < 10:
            return []  # Need baseline
        
        anomalies = []
        
        # Calculate baselines from recent history
        recent_latencies = [m["avg_latency"] for m in self.metrics_buffer[-10:]]
        recent_success_rates = [m["success_rate"] for m in self.metrics_buffer[-10:]]
        
        avg_latency = sum(recent_latencies) / len(recent_latencies)
        avg_success_rate = sum(recent_success_rates) / len(recent_success_rates)
        
        # Check for latency spikes
        if current_metrics["avg_latency"] > avg_latency * 2:
            anomalies.append({
                "type": "latency_spike",
                "current": current_metrics["avg_latency"],
                "baseline": avg_latency,
                "severity": "high" if current_metrics["avg_latency"] > avg_latency * 3 else "medium"
            })
        
        # Check for success rate drops
        if current_metrics["success_rate"] < avg_success_rate * 0.9:
            anomalies.append({
                "type": "success_rate_drop",
                "current": current_metrics["success_rate"],
                "baseline": avg_success_rate,
                "severity": "high" if current_metrics["success_rate"] < avg_success_rate * 0.8 else "medium"
            })
        
        return anomalies
```

### 4. Multi-Region Deployment and Failover

#### Geo-Distributed Search Architecture
```python
class AzureSearchGeoDistribution:
    def __init__(self, regions_config):
        self.regions = regions_config
        self.primary_region = regions_config["primary"]
        self.secondary_regions = regions_config["secondary"]
        self.search_clients = self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize search clients for all regions."""
        clients = {}
        
        for region_name, config in self.regions.items():
            clients[region_name] = SearchClient(
                endpoint=config["endpoint"],
                index_name=config["index_name"],
                credential=AzureKeyCredential(config["api_key"])
            )
        
        return clients
    
    async def distributed_search(self, query, preferred_region=None):
        """Perform search with automatic failover."""
        
        # Determine search order
        search_order = self._get_search_order(preferred_region)
        
        for region in search_order:
            try:
                client = self.search_clients[region]
                
                # Add circuit breaker logic
                if self._is_region_healthy(region):
                    start_time = time.time()
                    results = list(client.search(query, top=10))
                    latency = (time.time() - start_time) * 1000
                    
                    # Update region health metrics
                    self._update_region_metrics(region, latency, True)
                    
                    return {
                        "results": results,
                        "region": region,
                        "latency_ms": latency
                    }
            
            except Exception as e:
                print(f"Search failed in region {region}: {e}")
                self._update_region_metrics(region, 0, False)
                continue
        
        raise Exception("All regions failed")
    
    def _get_search_order(self, preferred_region):
        """Determine optimal search order based on health and preference."""
        
        if preferred_region and self._is_region_healthy(preferred_region):
            order = [preferred_region]
            order.extend([r for r in self.regions.keys() if r != preferred_region])
            return order
        
        # Sort by health and latency
        return sorted(self.regions.keys(), 
                     key=lambda r: (not self._is_region_healthy(r), self._get_avg_latency(r)))
    
    def _is_region_healthy(self, region):
        """Check if region is healthy based on recent metrics."""
        # Implementation would check recent success rates and latencies
        return True  # Simplified
    
    def _update_region_metrics(self, region, latency, success):
        """Update region health metrics."""
        # Implementation would update metrics store
        pass
    
    def _get_avg_latency(self, region):
        """Get average latency for region."""
        # Implementation would return recent average latency
        return 50  # Simplified
```

#### Index Synchronization
```python
class AzureSearchIndexSync:
    def __init__(self, primary_client, secondary_clients):
        self.primary_client = primary_client
        self.secondary_clients = secondary_clients
    
    async def sync_indexes(self, batch_size=1000):
        """Synchronize indexes across regions."""
        
        # Get all documents from primary
        all_docs = self._get_all_documents(self.primary_client)
        
        # Sync to each secondary region
        sync_results = {}
        
        for region_name, client in self.secondary_clients.items():
            try:
                result = await self._sync_to_region(client, all_docs, batch_size)
                sync_results[region_name] = result
            except Exception as e:
                sync_results[region_name] = {"error": str(e)}
        
        return sync_results
    
    def _get_all_documents(self, client):
        """Get all documents from an index."""
        documents = []
        
        # Use search with wildcard to get all documents
        results = client.search("*", include_total_count=True)
        
        for result in results:
            documents.append(result)
        
        return documents
    
    async def _sync_to_region(self, client, documents, batch_size):
        """Sync documents to a specific region."""
        
        total_docs = len(documents)
        processed = 0
        errors = 0
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Upload batch to target region
                upload_results = client.upload_documents(batch)
                
                # Check for errors in batch
                for result in upload_results:
                    if not result.succeeded:
                        errors += 1
                        print(f"Failed to upload document {result.key}: {result.error_message}")
                
                processed += len(batch)
                
            except Exception as e:
                print(f"Batch upload failed: {e}")
                errors += len(batch)
        
        return {
            "total_documents": total_docs,
            "processed": processed,
            "errors": errors,
            "success_rate": (processed - errors) / processed if processed > 0 else 0
        }
```

### 5. Cost Optimization and Resource Management

#### Intelligent Resource Scaling
```python
class AzureSearchResourceOptimizer:
    def __init__(self, subscription_id, resource_group, search_service):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.search_service = search_service
    
    def analyze_usage_patterns(self, days=30):
        """Analyze usage patterns to recommend optimal scaling."""
        
        # Query Azure Monitor for usage data
        usage_query = f"""
        AzureDiagnostics
        | where Category == "OperationLogs"
        | where TimeGenerated > ago({days}d)
        | extend Hour = bin(TimeGenerated, 1h)
        | summarize 
            QueryCount = count(),
            AvgLatency = avg(DurationMs),
            P95Latency = percentile(DurationMs, 95),
            MaxConcurrentQueries = max(todouble(properties_s.ConcurrentQueries))
        by Hour
        | order by Hour desc
        """
        
        # Get usage statistics
        usage_stats = self._execute_usage_query(usage_query)
        
        # Analyze patterns
        recommendations = self._generate_scaling_recommendations(usage_stats)
        
        return recommendations
    
    def _generate_scaling_recommendations(self, usage_stats):
        """Generate scaling recommendations based on usage patterns."""
        
        # Calculate key metrics
        avg_queries_per_hour = sum(stat["QueryCount"] for stat in usage_stats) / len(usage_stats)
        peak_queries_per_hour = max(stat["QueryCount"] for stat in usage_stats)
        avg_latency = sum(stat["AvgLatency"] for stat in usage_stats) / len(usage_stats)
        p95_latency = max(stat["P95Latency"] for stat in usage_stats)
        
        recommendations = {
            "current_analysis": {
                "avg_queries_per_hour": avg_queries_per_hour,
                "peak_queries_per_hour": peak_queries_per_hour,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency
            },
            "recommendations": []
        }
        
        # Scaling recommendations
        if avg_latency > 500:
            recommendations["recommendations"].append({
                "type": "scale_up",
                "reason": "High average latency detected",
                "suggested_tier": "Standard S2",
                "estimated_cost_impact": "+$3000/month"
            })
        
        if peak_queries_per_hour > 1000:
            recommendations["recommendations"].append({
                "type": "add_replicas",
                "reason": "High query volume during peak hours",
                "suggested_replicas": 3,
                "estimated_cost_impact": "+$2000/month"
            })
        
        if p95_latency > 1000:
            recommendations["recommendations"].append({
                "type": "optimize_indexes",
                "reason": "P95 latency indicates query optimization needed",
                "suggested_actions": ["Review slow queries", "Optimize index configuration"]
            })
        
        return recommendations
    
    def estimate_cost_savings(self, optimization_plan):
        """Estimate potential cost savings from optimization."""
        
        current_cost = self._get_current_monthly_cost()
        
        savings_analysis = {
            "current_monthly_cost": current_cost,
            "optimization_opportunities": []
        }
        
        # Analyze different optimization scenarios
        if optimization_plan.get("consolidate_indexes"):
            index_savings = current_cost * 0.15  # 15% savings from consolidation
            savings_analysis["optimization_opportunities"].append({
                "optimization": "Index consolidation",
                "monthly_savings": index_savings,
                "implementation_effort": "Medium"
            })
        
        if optimization_plan.get("optimize_replicas"):
            replica_savings = current_cost * 0.25  # 25% savings from replica optimization
            savings_analysis["optimization_opportunities"].append({
                "optimization": "Replica optimization",
                "monthly_savings": replica_savings,
                "implementation_effort": "Low"
            })
        
        if optimization_plan.get("query_optimization"):
            query_savings = current_cost * 0.10  # 10% savings from query optimization
            savings_analysis["optimization_opportunities"].append({
                "optimization": "Query optimization",
                "monthly_savings": query_savings,
                "implementation_effort": "High"
            })
        
        total_savings = sum(opp["monthly_savings"] for opp in savings_analysis["optimization_opportunities"])
        savings_analysis["total_potential_savings"] = total_savings
        savings_analysis["roi_percentage"] = (total_savings / current_cost) * 100
        
        return savings_analysis
```

## 🔧 Production Deployment Checklist

### Pre-Production Validation
```python
class ProductionReadinessValidator:
    def __init__(self, search_service_config):
        self.config = search_service_config
    
    def validate_production_readiness(self):
        """Comprehensive production readiness validation."""
        
        validation_results = {
            "security": self._validate_security(),
            "performance": self._validate_performance(),
            "monitoring": self._validate_monitoring(),
            "backup": self._validate_backup_strategy(),
            "scaling": self._validate_scaling_configuration(),
            "compliance": self._validate_compliance()
        }
        
        overall_score = self._calculate_readiness_score(validation_results)
        validation_results["overall_readiness_score"] = overall_score
        validation_results["ready_for_production"] = overall_score >= 80
        
        return validation_results
    
    def _validate_security(self):
        """Validate security configuration."""
        checks = {
            "api_key_rotation": False,  # Check if API keys are rotated regularly
            "rbac_configured": False,   # Check if RBAC is properly configured
            "network_security": False, # Check if network access is restricted
            "data_encryption": True,    # Azure AI Search encrypts by default
            "audit_logging": False      # Check if audit logging is enabled
        }
        
        score = (sum(checks.values()) / len(checks)) * 100
        return {"score": score, "checks": checks}
    
    def _validate_performance(self):
        """Validate performance configuration."""
        checks = {
            "load_testing_completed": False,
            "performance_baseline_established": False,
            "auto_scaling_configured": False,
            "query_optimization_applied": False,
            "index_optimization_applied": False
        }
        
        score = (sum(checks.values()) / len(checks)) * 100
        return {"score": score, "checks": checks}
    
    def _validate_monitoring(self):
        """Validate monitoring and alerting setup."""
        checks = {
            "azure_monitor_enabled": False,
            "custom_alerts_configured": False,
            "dashboard_created": False,
            "log_analytics_connected": False,
            "application_insights_integrated": False
        }
        
        score = (sum(checks.values()) / len(checks)) * 100
        return {"score": score, "checks": checks}
```

---
*Next: [Integration Patterns](./05-azure-integration-patterns.md)*