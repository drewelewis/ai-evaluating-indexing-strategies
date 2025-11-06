# Azure AI Search Evaluation Guide

This guide provides specific instructions for evaluating indexing strategies using Azure AI Search (formerly Azure Cognitive Search) and Azure-specific tools.

## 🔧 Azure AI Search Setup

### Prerequisites
```bash
# Install Azure CLI and Azure AI Search SDK
pip install azure-search-documents==11.4.0
pip install azure-identity==1.15.0
pip install azure-ai-textanalytics==5.3.0
pip install azure-storage-blob==12.19.0
```

### Authentication Configuration
```python
# src/azure/auth.py
import os
from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.core.credentials import AzureKeyCredential

class AzureAuthManager:
    def __init__(self):
        self.search_key = os.getenv('AZURE_SEARCH_KEY')
        self.search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        
    def get_search_credential(self):
        """Get credential for Azure AI Search."""
        if self.search_key:
            return AzureKeyCredential(self.search_key)
        else:
            # Use managed identity or Azure CLI
            return DefaultAzureCredential()
    
    def get_openai_credential(self):
        """Get credential for Azure OpenAI (for embeddings)."""
        return DefaultAzureCredential()
```

### Environment Configuration
```bash
# .env file
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your-search-admin-key
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com
AZURE_OPENAI_KEY=your-openai-key
AZURE_OPENAI_API_VERSION=2024-02-01
```

## 🔍 Azure AI Search Engine Implementation

### Basic Search Engine
```python
# src/engines/azure_search_engine.py
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.search.documents.models import VectorizedQuery
from ..azure.auth import AzureAuthManager
from .base_engine import BaseSearchEngine, SearchResult

class AzureSearchEngine(BaseSearchEngine):
    def __init__(self, config):
        super().__init__(config)
        
        self.auth_manager = AzureAuthManager()
        self.endpoint = self.auth_manager.search_endpoint
        self.credential = self.auth_manager.get_search_credential()
        
        self.index_name = config.get('index_name', 'search-index')
        self.index_client = SearchIndexClient(self.endpoint, self.credential)
        self.search_client = SearchClient(self.endpoint, self.index_name, self.credential)
        
        # Initialize embedding model if using vector search
        if config.get('enable_vector_search', False):
            self.embedding_model = self._init_embedding_model(config)
    
    def _init_embedding_model(self, config):
        """Initialize Azure OpenAI embedding model."""
        from openai import AzureOpenAI
        
        return AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
        )
    
    def create_index(self):
        """Create Azure AI Search index with both text and vector fields."""
        
        # Define fields
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="metadata", type=SearchFieldDataType.String),
        ]
        
        # Add vector field if vector search is enabled
        if self.config.get('enable_vector_search', False):
            vector_field = VectorSearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # OpenAI text-embedding-ada-002 dimensions
                vector_search_profile_name="my-vector-config"
            )
            fields.append(vector_field)
        
        # Vector search configuration
        vector_search = None
        if self.config.get('enable_vector_search', False):
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="my-hnsw-config",
                        parameters=HnswParameters(
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                            metric=VectorSearchAlgorithmMetric.COSINE
                        )
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="my-vector-config",
                        algorithm_configuration_name="my-hnsw-config"
                    )
                ]
            )
        
        # Semantic search configuration
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")]
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # Create index
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        # Delete index if it exists
        try:
            self.index_client.delete_index(self.index_name)
        except:
            pass
        
        # Create new index
        self.index_client.create_index(index)
        print(f"Created index: {self.index_name}")
    
    def index_documents(self, documents):
        """Index documents in Azure AI Search."""
        
        # Create index first
        self.create_index()
        
        # Prepare documents for indexing
        search_documents = []
        
        for doc in documents:
            search_doc = {
                "id": doc["id"],
                "content": doc["content"],
                "title": doc.get("title", ""),
                "category": doc.get("metadata", {}).get("category", ""),
                "metadata": str(doc.get("metadata", {}))
            }
            
            # Generate vector embeddings if enabled
            if self.config.get('enable_vector_search', False):
                content_vector = self._generate_embedding(doc["content"])
                search_doc["content_vector"] = content_vector
            
            search_documents.append(search_doc)
        
        # Upload documents in batches
        batch_size = 100
        for i in range(0, len(search_documents), batch_size):
            batch = search_documents[i:i + batch_size]
            result = self.search_client.upload_documents(documents=batch)
            print(f"Uploaded batch {i//batch_size + 1}, {len(batch)} documents")
        
        self.is_indexed = True
    
    def _generate_embedding(self, text):
        """Generate embedding using Azure OpenAI."""
        response = self.embedding_model.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    
    def search(self, query, top_k=10, search_mode="hybrid"):
        """Search documents using different modes."""
        
        if not self.is_indexed:
            raise ValueError("Documents must be indexed before searching")
        
        search_results = []
        
        if search_mode == "keyword":
            search_results = self._keyword_search(query, top_k)
        elif search_mode == "vector":
            search_results = self._vector_search(query, top_k)
        elif search_mode == "hybrid":
            search_results = self._hybrid_search(query, top_k)
        elif search_mode == "semantic":
            search_results = self._semantic_search(query, top_k)
        else:
            raise ValueError(f"Unknown search mode: {search_mode}")
        
        return search_results
    
    def _keyword_search(self, query, top_k):
        """Pure keyword search using BM25."""
        results = self.search_client.search(
            search_text=query,
            top=top_k,
            include_total_count=True
        )
        
        search_results = []
        for result in results:
            search_result = SearchResult(
                doc_id=result["id"],
                content=result["content"],
                score=result["@search.score"],
                metadata={"title": result.get("title", "")}
            )
            search_results.append(search_result)
        
        return search_results
    
    def _vector_search(self, query, top_k):
        """Pure vector search."""
        if not self.config.get('enable_vector_search', False):
            raise ValueError("Vector search not enabled")
        
        # Generate query vector
        query_vector = self._generate_embedding(query)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )
        
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k
        )
        
        search_results = []
        for result in results:
            search_result = SearchResult(
                doc_id=result["id"],
                content=result["content"],
                score=result["@search.score"],
                metadata={"title": result.get("title", "")}
            )
            search_results.append(search_result)
        
        return search_results
    
    def _hybrid_search(self, query, top_k):
        """Hybrid search combining keyword and vector."""
        if not self.config.get('enable_vector_search', False):
            return self._keyword_search(query, top_k)
        
        # Generate query vector
        query_vector = self._generate_embedding(query)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )
        
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top_k
        )
        
        search_results = []
        for result in results:
            search_result = SearchResult(
                doc_id=result["id"],
                content=result["content"],
                score=result["@search.score"],
                metadata={"title": result.get("title", "")}
            )
            search_results.append(search_result)
        
        return search_results
    
    def _semantic_search(self, query, top_k):
        """Semantic search using Azure's built-in semantic ranking."""
        results = self.search_client.search(
            search_text=query,
            top=top_k,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="my-semantic-config",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE
        )
        
        search_results = []
        for result in results:
            # Get semantic captions if available
            captions = []
            if "@search.captions" in result:
                captions = [caption.text for caption in result["@search.captions"]]
            
            search_result = SearchResult(
                doc_id=result["id"],
                content=result["content"],
                score=result["@search.score"],
                metadata={
                    "title": result.get("title", ""),
                    "captions": captions,
                    "reranker_score": result.get("@search.reranker_score")
                }
            )
            search_results.append(search_result)
        
        return search_results
    
    def get_stats(self):
        """Get index statistics."""
        try:
            stats = self.index_client.get_index_statistics(self.index_name)
            return {
                "document_count": stats.document_count,
                "storage_size_mb": stats.storage_size / (1024 * 1024),
                "vector_index_size_mb": getattr(stats, 'vector_index_size', 0) / (1024 * 1024)
            }
        except Exception as e:
            return {"error": str(e)}
```

## 📊 Azure-Specific Evaluation Pipeline

### Azure Cognitive Services Integration
```python
# src/azure/cognitive_services.py
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

class AzureCognitiveServices:
    def __init__(self):
        endpoint = os.getenv('AZURE_TEXT_ANALYTICS_ENDPOINT')
        key = os.getenv('AZURE_TEXT_ANALYTICS_KEY')
        
        self.text_client = TextAnalyticsClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key)
        )
    
    def analyze_query_intent(self, queries):
        """Analyze query intent using Azure Text Analytics."""
        
        # Extract key phrases
        key_phrases_result = self.text_client.extract_key_phrases(queries)
        
        # Analyze sentiment
        sentiment_result = self.text_client.analyze_sentiment(queries)
        
        analysis = []
        for i, query in enumerate(queries):
            analysis.append({
                "query": query,
                "key_phrases": key_phrases_result[i].key_phrases if not key_phrases_result[i].is_error else [],
                "sentiment": sentiment_result[i].sentiment if not sentiment_result[i].is_error else "neutral",
                "confidence": sentiment_result[i].confidence_scores.positive if not sentiment_result[i].is_error else 0.5
            })
        
        return analysis
    
    def categorize_queries(self, queries):
        """Categorize queries by intent type."""
        analysis = self.analyze_query_intent(queries)
        
        categories = {
            "navigational": [],
            "informational": [],
            "transactional": []
        }
        
        for item in analysis:
            # Simple heuristic categorization
            key_phrases = [phrase.lower() for phrase in item["key_phrases"]]
            
            if any(phrase in ["buy", "purchase", "price", "cost", "order"] for phrase in key_phrases):
                categories["transactional"].append(item)
            elif any(phrase in ["how", "what", "why", "when", "where"] for phrase in key_phrases):
                categories["informational"].append(item)
            else:
                categories["navigational"].append(item)
        
        return categories
```

### Azure-Optimized Evaluation Framework
```python
# src/azure/azure_evaluator.py
from ..evaluation.evaluator import SearchEvaluator
from .cognitive_services import AzureCognitiveServices
import azure.monitor.query as monitor_query
from azure.identity import DefaultAzureCredential

class AzureSearchEvaluator(SearchEvaluator):
    def __init__(self, metrics_config, search_service_name):
        super().__init__(metrics_config)
        self.search_service_name = search_service_name
        self.cognitive_services = AzureCognitiveServices()
        
        # Initialize Azure Monitor client for performance metrics
        self.monitor_client = monitor_query.LogsQueryClient(DefaultAzureCredential())
    
    def evaluate_with_intent_analysis(self, engine, dataset):
        """Evaluate engine with Azure Cognitive Services intent analysis."""
        
        # Analyze query intents
        queries = [item['query'] for item in dataset]
        categorized_queries = self.cognitive_services.categorize_queries(queries)
        
        # Evaluate each category separately
        results_by_category = {}
        
        for category, query_items in categorized_queries.items():
            if not query_items:
                continue
                
            # Create dataset for this category
            category_dataset = []
            for query_item in query_items:
                # Find matching dataset item
                for dataset_item in dataset:
                    if dataset_item['query'] == query_item['query']:
                        category_dataset.append(dataset_item)
                        break
            
            if category_dataset:
                category_results = self.evaluate_engine(engine, category_dataset)
                results_by_category[category] = category_results
        
        # Overall evaluation
        overall_results = self.evaluate_engine(engine, dataset)
        
        return {
            "overall": overall_results,
            "by_category": results_by_category,
            "query_analysis": categorized_queries
        }
    
    def get_azure_performance_metrics(self, time_range_hours=24):
        """Get performance metrics from Azure Monitor."""
        
        # KQL query for search service metrics
        query = f"""
        AzureDiagnostics
        | where TimeGenerated > ago({time_range_hours}h)
        | where ResourceProvider == "MICROSOFT.SEARCH"
        | where Resource == "{self.search_service_name}"
        | summarize 
            AvgLatency = avg(DurationMs),
            P95Latency = percentile(DurationMs, 95),
            P99Latency = percentile(DurationMs, 99),
            TotalRequests = count(),
            ErrorRate = countif(ResultSignature != "200") * 100.0 / count()
        by bin(TimeGenerated, 1h)
        | order by TimeGenerated desc
        """
        
        try:
            response = self.monitor_client.query_workspace(
                workspace_id=os.getenv('AZURE_LOG_ANALYTICS_WORKSPACE_ID'),
                query=query,
                timespan=f"PT{time_range_hours}H"
            )
            
            metrics = []
            for row in response.tables[0].rows:
                metrics.append({
                    "timestamp": row[0],
                    "avg_latency_ms": row[1],
                    "p95_latency_ms": row[2], 
                    "p99_latency_ms": row[3],
                    "total_requests": row[4],
                    "error_rate_percent": row[5]
                })
            
            return metrics
            
        except Exception as e:
            print(f"Failed to retrieve Azure metrics: {e}")
            return []
    
    def run_azure_load_test(self, engine, test_queries, concurrent_users=10, duration_minutes=5):
        """Run load test against Azure AI Search."""
        import asyncio
        import aiohttp
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        results = {
            "start_time": time.time(),
            "concurrent_users": concurrent_users,
            "duration_minutes": duration_minutes,
            "response_times": [],
            "errors": [],
            "total_requests": 0
        }
        
        def search_worker():
            """Worker function for load testing."""
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            worker_stats = {
                "response_times": [],
                "errors": [],
                "requests": 0
            }
            
            while time.time() < end_time:
                query = test_queries[worker_stats["requests"] % len(test_queries)]
                
                try:
                    request_start = time.time()
                    engine.search(query, top_k=10)
                    response_time = (time.time() - request_start) * 1000
                    
                    worker_stats["response_times"].append(response_time)
                    worker_stats["requests"] += 1
                    
                except Exception as e:
                    worker_stats["errors"].append(str(e))
                
                # Small delay to prevent overwhelming the service
                time.sleep(0.1)
            
            return worker_stats
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(search_worker) for _ in range(concurrent_users)]
            
            # Collect results
            for future in futures:
                worker_result = future.result()
                results["response_times"].extend(worker_result["response_times"])
                results["errors"].extend(worker_result["errors"])
                results["total_requests"] += worker_result["requests"]
        
        # Calculate summary statistics
        if results["response_times"]:
            results["avg_response_time_ms"] = sum(results["response_times"]) / len(results["response_times"])
            results["p95_response_time_ms"] = sorted(results["response_times"])[int(0.95 * len(results["response_times"]))]
            results["p99_response_time_ms"] = sorted(results["response_times"])[int(0.99 * len(results["response_times"]))]
        
        results["error_rate"] = len(results["errors"]) / results["total_requests"] if results["total_requests"] > 0 else 0
        results["requests_per_second"] = results["total_requests"] / (duration_minutes * 60)
        
        return results
```

## 🔄 Azure-Specific Optimization Strategies

### Index Optimization
```python
# src/azure/index_optimizer.py
class AzureIndexOptimizer:
    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.index_client = search_engine.index_client
        self.index_name = search_engine.index_name
    
    def optimize_for_query_patterns(self, query_log):
        """Optimize index based on query patterns."""
        
        # Analyze query patterns
        analysis = self._analyze_query_patterns(query_log)
        
        optimizations = []
        
        # Suggest field boosts based on query focus
        if analysis["title_focused"] > 0.3:
            optimizations.append({
                "type": "field_boost",
                "field": "title",
                "suggested_boost": 3.0,
                "reason": f"{analysis['title_focused']:.1%} of queries focus on titles"
            })
        
        # Suggest custom analyzers for common terms
        if analysis["common_terms"]:
            optimizations.append({
                "type": "custom_analyzer",
                "terms": analysis["common_terms"][:10],
                "reason": "Frequent terms could benefit from custom analysis"
            })
        
        # Suggest vector search if semantic queries are common
        if analysis["semantic_queries"] > 0.4:
            optimizations.append({
                "type": "enable_vector_search",
                "reason": f"{analysis['semantic_queries']:.1%} of queries are semantic"
            })
        
        return optimizations
    
    def _analyze_query_patterns(self, query_log):
        """Analyze patterns in query log."""
        from collections import Counter
        
        all_terms = []
        title_queries = 0
        semantic_queries = 0
        
        for query in query_log:
            terms = query.lower().split()
            all_terms.extend(terms)
            
            # Check if query focuses on titles
            if any(term in query.lower() for term in ["title", "name", "called"]):
                title_queries += 1
            
            # Check if query is semantic (contains question words, longer phrases)
            if any(term in query.lower() for term in ["how", "what", "why", "when", "where", "explain"]) or len(terms) > 5:
                semantic_queries += 1
        
        term_counts = Counter(all_terms)
        
        return {
            "total_queries": len(query_log),
            "title_focused": title_queries / len(query_log) if query_log else 0,
            "semantic_queries": semantic_queries / len(query_log) if query_log else 0,
            "common_terms": [term for term, count in term_counts.most_common(20)]
        }
    
    def apply_optimizations(self, optimizations):
        """Apply suggested optimizations to the index."""
        
        for opt in optimizations:
            if opt["type"] == "field_boost":
                self._update_field_boost(opt["field"], opt["suggested_boost"])
            elif opt["type"] == "custom_analyzer":
                self._add_custom_analyzer(opt["terms"])
            elif opt["type"] == "enable_vector_search":
                self._enable_vector_search()
    
    def _update_field_boost(self, field_name, boost_value):
        """Update field boost in search profile."""
        # This would require updating the index definition
        print(f"Would update {field_name} boost to {boost_value}")
    
    def _add_custom_analyzer(self, common_terms):
        """Add custom analyzer for common terms."""
        # This would require updating the index definition with custom analyzer
        print(f"Would add custom analyzer for terms: {common_terms[:5]}")
    
    def _enable_vector_search(self):
        """Enable vector search capabilities."""
        print("Would enable vector search in index configuration")
```

### Cost Optimization
```python
# src/azure/cost_optimizer.py
class AzureCostOptimizer:
    def __init__(self, search_service_name, subscription_id):
        self.search_service_name = search_service_name
        self.subscription_id = subscription_id
    
    def analyze_usage_patterns(self, usage_data):
        """Analyze usage patterns for cost optimization."""
        
        recommendations = []
        
        # Analyze query volume patterns
        hourly_queries = self._group_queries_by_hour(usage_data)
        peak_hours = self._identify_peak_hours(hourly_queries)
        
        if len(peak_hours) < 8:  # Less than 8 hours of peak usage
            recommendations.append({
                "type": "scaling_schedule",
                "description": "Consider auto-scaling during peak hours only",
                "estimated_savings": "20-40%",
                "peak_hours": peak_hours
            })
        
        # Analyze search unit utilization
        avg_utilization = sum(usage_data["cpu_usage"]) / len(usage_data["cpu_usage"])
        if avg_utilization < 0.3:
            recommendations.append({
                "type": "downsize_tier",
                "description": "Consider downsizing to a smaller search tier", 
                "estimated_savings": "30-50%",
                "current_utilization": f"{avg_utilization:.1%}"
            })
        
        # Analyze index size vs query patterns
        index_size_mb = usage_data.get("index_size_mb", 0)
        queries_per_gb = usage_data.get("total_queries", 0) / (index_size_mb / 1024) if index_size_mb > 0 else 0
        
        if queries_per_gb < 100:  # Low query density
            recommendations.append({
                "type": "index_optimization",
                "description": "Consider removing unused fields or optimizing index structure",
                "estimated_savings": "10-20%",
                "query_density": f"{queries_per_gb:.1f} queries/GB"
            })
        
        return recommendations
    
    def _group_queries_by_hour(self, usage_data):
        """Group queries by hour of day."""
        from collections import defaultdict
        import datetime
        
        hourly_queries = defaultdict(int)
        
        for timestamp, query_count in zip(usage_data["timestamps"], usage_data["query_counts"]):
            hour = datetime.datetime.fromtimestamp(timestamp).hour
            hourly_queries[hour] += query_count
        
        return hourly_queries
    
    def _identify_peak_hours(self, hourly_queries):
        """Identify peak usage hours."""
        if not hourly_queries:
            return []
        
        avg_queries = sum(hourly_queries.values()) / len(hourly_queries)
        threshold = avg_queries * 1.5  # 50% above average
        
        peak_hours = [hour for hour, count in hourly_queries.items() if count > threshold]
        return sorted(peak_hours)
    
    def estimate_cost_savings(self, recommendations):
        """Estimate potential cost savings from recommendations."""
        
        total_savings_percent = 0
        monthly_cost_estimate = 1000  # Base estimate in USD
        
        for rec in recommendations:
            if rec["type"] == "scaling_schedule":
                total_savings_percent += 30
            elif rec["type"] == "downsize_tier":
                total_savings_percent += 40
            elif rec["type"] == "index_optimization":
                total_savings_percent += 15
        
        # Cap total savings at 70%
        total_savings_percent = min(total_savings_percent, 70)
        
        estimated_monthly_savings = monthly_cost_estimate * (total_savings_percent / 100)
        
        return {
            "current_monthly_estimate": monthly_cost_estimate,
            "potential_savings_percent": total_savings_percent,
            "estimated_monthly_savings": estimated_monthly_savings,
            "annual_savings": estimated_monthly_savings * 12
        }
```

## 📈 Azure Monitoring and Alerting

### Custom Monitoring Dashboard
```python
# src/azure/monitoring.py
import azure.monitor.query as monitor_query
from azure.identity import DefaultAzureCredential
import json

class AzureSearchMonitoring:
    def __init__(self, workspace_id, search_service_name):
        self.workspace_id = workspace_id
        self.search_service_name = search_service_name
        self.logs_client = monitor_query.LogsQueryClient(DefaultAzureCredential())
    
    def get_search_metrics_dashboard(self, hours=24):
        """Get comprehensive search metrics for dashboard."""
        
        queries = {
            "query_volume": f"""
                AzureDiagnostics
                | where TimeGenerated > ago({hours}h)
                | where Resource == "{self.search_service_name}"
                | summarize QueryCount = count() by bin(TimeGenerated, 1h)
                | order by TimeGenerated
            """,
            
            "latency_percentiles": f"""
                AzureDiagnostics  
                | where TimeGenerated > ago({hours}h)
                | where Resource == "{self.search_service_name}"
                | summarize 
                    P50 = percentile(DurationMs, 50),
                    P95 = percentile(DurationMs, 95),
                    P99 = percentile(DurationMs, 99)
                by bin(TimeGenerated, 1h)
                | order by TimeGenerated
            """,
            
            "error_analysis": f"""
                AzureDiagnostics
                | where TimeGenerated > ago({hours}h)
                | where Resource == "{self.search_service_name}"
                | summarize 
                    TotalRequests = count(),
                    Errors = countif(ResultSignature != "200"),
                    ErrorRate = countif(ResultSignature != "200") * 100.0 / count()
                by ResultSignature
                | order by TotalRequests desc
            """,
            
            "query_patterns": f"""
                AzureDiagnostics
                | where TimeGenerated > ago({hours}h)
                | where Resource == "{self.search_service_name}"
                | extend QueryText = tostring(parse_json(properties_s).Query)
                | summarize QueryCount = count() by QueryText
                | order by QueryCount desc
                | take 20
            """
        }
        
        dashboard_data = {}
        
        for metric_name, query in queries.items():
            try:
                response = self.logs_client.query_workspace(
                    workspace_id=self.workspace_id,
                    query=query,
                    timespan=f"PT{hours}H"
                )
                
                # Convert response to JSON-serializable format
                data = []
                if response.tables:
                    table = response.tables[0]
                    for row in table.rows:
                        row_dict = {}
                        for i, column in enumerate(table.columns):
                            row_dict[column.name] = row[i]
                        data.append(row_dict)
                
                dashboard_data[metric_name] = data
                
            except Exception as e:
                dashboard_data[metric_name] = {"error": str(e)}
        
        return dashboard_data
    
    def setup_alerts(self, alert_config):
        """Setup Azure Monitor alerts for search service."""
        
        # This would use Azure Monitor Alert Rules API
        # For now, return the configuration that would be applied
        
        alert_rules = []
        
        if "high_latency" in alert_config:
            alert_rules.append({
                "name": f"{self.search_service_name}-high-latency",
                "description": "Alert when search latency is high",
                "condition": {
                    "metric": "search_latency_p95",
                    "threshold": alert_config["high_latency"]["threshold_ms"],
                    "operator": "GreaterThan"
                },
                "actions": alert_config["high_latency"]["actions"]
            })
        
        if "high_error_rate" in alert_config:
            alert_rules.append({
                "name": f"{self.search_service_name}-high-error-rate", 
                "description": "Alert when error rate is high",
                "condition": {
                    "metric": "error_rate_percent",
                    "threshold": alert_config["high_error_rate"]["threshold_percent"],
                    "operator": "GreaterThan"
                },
                "actions": alert_config["high_error_rate"]["actions"]
            })
        
        return alert_rules

# Example usage
def setup_azure_monitoring():
    monitor = AzureSearchMonitoring(
        workspace_id=os.getenv('AZURE_LOG_ANALYTICS_WORKSPACE_ID'),
        search_service_name="my-search-service"
    )
    
    # Get dashboard data
    dashboard = monitor.get_search_metrics_dashboard(hours=24)
    
    # Setup alerts
    alert_config = {
        "high_latency": {
            "threshold_ms": 1000,
            "actions": ["email:admin@company.com"]
        },
        "high_error_rate": {
            "threshold_percent": 5.0,
            "actions": ["webhook:https://alerts.company.com/webhook"]
        }
    }
    
    alerts = monitor.setup_alerts(alert_config)
    
    return {
        "dashboard": dashboard,
        "alert_rules": alerts
    }
```

## 🚀 Complete Azure Example

```python
# examples/azure_complete_evaluation.py
def main():
    # Configure Azure Search Engine
    azure_config = {
        'index_name': 'product-search-index',
        'enable_vector_search': True,
        'enable_semantic_search': True
    }
    
    # Initialize engine
    azure_engine = AzureSearchEngine(azure_config)
    
    # Load and index documents
    documents = load_product_catalog()
    azure_engine.index_documents(documents)
    
    # Load test queries
    test_queries = load_golden_dataset('datasets/azure_test_queries.json')
    
    # Initialize evaluator with Azure-specific features
    evaluator = AzureSearchEvaluator(['precision_at_5', 'mrr', 'ndcg_at_10'], 'my-search-service')
    
    # Run comprehensive evaluation
    print("Running Azure AI Search evaluation...")
    
    # Test different search modes
    search_modes = ['keyword', 'vector', 'hybrid', 'semantic']
    results = {}
    
    for mode in search_modes:
        print(f"\nEvaluating {mode} search...")
        
        # Temporarily set search mode
        original_search = azure_engine.search
        azure_engine.search = lambda q, k=10: original_search(q, k, search_mode=mode)
        
        # Evaluate with intent analysis
        mode_results = evaluator.evaluate_with_intent_analysis(azure_engine, test_queries)
        results[mode] = mode_results
        
        # Restore original search method
        azure_engine.search = original_search
    
    # Run load test
    print("\nRunning load test...")
    load_test_queries = [item['query'] for item in test_queries[:50]]
    load_results = evaluator.run_azure_load_test(
        azure_engine, 
        load_test_queries,
        concurrent_users=5,
        duration_minutes=2
    )
    
    # Get Azure performance metrics
    azure_metrics = evaluator.get_azure_performance_metrics(hours=1)
    
    # Optimize index based on query patterns
    optimizer = AzureIndexOptimizer(azure_engine)
    optimizations = optimizer.optimize_for_query_patterns(load_test_queries)
    
    # Analyze costs
    cost_optimizer = AzureCostOptimizer('my-search-service', 'subscription-id')
    cost_recommendations = cost_optimizer.analyze_usage_patterns({
        "timestamps": [time.time()],
        "query_counts": [len(test_queries)],
        "cpu_usage": [0.4],  # 40% utilization
        "index_size_mb": 1024,
        "total_queries": len(test_queries)
    })
    
    # Generate comprehensive report
    report = {
        "evaluation_results": results,
        "load_test": load_results,
        "azure_metrics": azure_metrics,
        "optimizations": optimizations,
        "cost_analysis": cost_recommendations,
        "timestamp": time.time()
    }
    
    # Export results
    with open('azure_evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\nEvaluation complete! Report saved to azure_evaluation_report.json")

if __name__ == "__main__":
    main()
```

---
*Next: [RAG Scenario Evaluation](./rag-scenario-evaluation.md)*