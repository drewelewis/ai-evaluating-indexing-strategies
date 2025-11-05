# Cost Analysis

Complete guide to Azure AI Search cost optimization and total cost of ownership (TCO) analysis.

## 📋 Table of Contents
- [Overview](#overview)
- [Cost Components](#cost-components)
- [TCO Calculator](#tco-calculator)
- [Optimization Strategies](#optimization-strategies)
- [Cost Monitoring](#cost-monitoring)
- [Reserved Capacity](#reserved-capacity)
- [Best Practices](#best-practices)

---

## Overview

### Azure AI Search Pricing Model

```python
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class SearchServicePricing:
    """Azure AI Search pricing structure."""
    tier: str
    base_cost_per_hour: float
    storage_included_gb: float
    storage_overage_cost_per_gb: float
    qps_per_replica_partition: int
    
    def calculate_hourly_cost(self, replicas: int = 1, partitions: int = 1) -> float:
        """Calculate hourly cost for configuration."""
        return self.base_cost_per_hour * replicas * partitions
    
    def calculate_monthly_cost(self, replicas: int = 1, partitions: int = 1) -> float:
        """Calculate monthly cost (730 hours)."""
        return self.calculate_hourly_cost(replicas, partitions) * 730
    
    def calculate_annual_cost(self, replicas: int = 1, partitions: int = 1) -> float:
        """Calculate annual cost."""
        return self.calculate_monthly_cost(replicas, partitions) * 12

class AzureSearchPricing:
    """Azure AI Search pricing for all tiers."""
    
    TIERS = {
        'Free': SearchServicePricing(
            tier='Free',
            base_cost_per_hour=0.0,
            storage_included_gb=0.05,  # 50 MB
            storage_overage_cost_per_gb=0.0,
            qps_per_replica_partition=3
        ),
        'Basic': SearchServicePricing(
            tier='Basic',
            base_cost_per_hour=0.10,
            storage_included_gb=2.0,
            storage_overage_cost_per_gb=0.40,
            qps_per_replica_partition=15
        ),
        'S1': SearchServicePricing(
            tier='S1',
            base_cost_per_hour=0.34,
            storage_included_gb=25.0,
            storage_overage_cost_per_gb=0.40,
            qps_per_replica_partition=15
        ),
        'S2': SearchServicePricing(
            tier='S2',
            base_cost_per_hour=1.37,
            storage_included_gb=100.0,
            storage_overage_cost_per_gb=0.40,
            qps_per_replica_partition=60
        ),
        'S3': SearchServicePricing(
            tier='S3',
            base_cost_per_hour=2.74,
            storage_included_gb=200.0,
            storage_overage_cost_per_gb=0.40,
            qps_per_replica_partition=60
        ),
        'S3 HD': SearchServicePricing(
            tier='S3 HD',
            base_cost_per_hour=2.74,
            storage_included_gb=200.0,
            storage_overage_cost_per_gb=0.40,
            qps_per_replica_partition=60
        ),
        'L1': SearchServicePricing(
            tier='L1',
            base_cost_per_hour=1.37,
            storage_included_gb=1000.0,
            storage_overage_cost_per_gb=0.40,
            qps_per_replica_partition=60
        ),
        'L2': SearchServicePricing(
            tier='L2',
            base_cost_per_hour=2.74,
            storage_included_gb=2000.0,
            storage_overage_cost_per_gb=0.40,
            qps_per_replica_partition=60
        )
    }
    
    @classmethod
    def get_pricing(cls, tier: str) -> SearchServicePricing:
        """Get pricing for a specific tier."""
        if tier not in cls.TIERS:
            raise ValueError(f"Unknown tier: {tier}")
        return cls.TIERS[tier]
    
    @classmethod
    def compare_tiers(cls, tiers: list, replicas: int = 1, 
                     partitions: int = 1) -> Dict:
        """Compare costs across multiple tiers."""
        comparison = {}
        
        for tier in tiers:
            pricing = cls.get_pricing(tier)
            comparison[tier] = {
                'monthly_cost': pricing.calculate_monthly_cost(replicas, partitions),
                'annual_cost': pricing.calculate_annual_cost(replicas, partitions),
                'total_qps': pricing.qps_per_replica_partition * replicas * partitions,
                'total_storage_gb': pricing.storage_included_gb * partitions,
                'cost_per_qps': pricing.calculate_monthly_cost(replicas, partitions) / 
                               (pricing.qps_per_replica_partition * replicas * partitions)
            }
        
        return comparison

# Usage
pricing = AzureSearchPricing()

# Get S2 pricing
s2_pricing = pricing.get_pricing('S2')
print(f"S2 Tier Pricing:")
print(f"  Base: ${s2_pricing.base_cost_per_hour:.2f}/hour")
print(f"  Monthly (1 replica, 1 partition): ${s2_pricing.calculate_monthly_cost(1, 1):.2f}")
print(f"  Monthly (3 replicas, 2 partitions): ${s2_pricing.calculate_monthly_cost(3, 2):.2f}")

# Compare tiers
comparison = pricing.compare_tiers(['S1', 'S2', 'S3'], replicas=3, partitions=1)
print(f"\n\nTier Comparison (3 replicas, 1 partition):")
for tier, costs in comparison.items():
    print(f"\n{tier}:")
    print(f"  Monthly: ${costs['monthly_cost']:.2f}")
    print(f"  Total QPS: {costs['total_qps']}")
    print(f"  Cost per QPS: ${costs['cost_per_qps']:.2f}")
```

---

## Cost Components

### OpenAI Embeddings Cost

```python
@dataclass
class OpenAIEmbeddingPricing:
    """OpenAI embedding API pricing."""
    model: str
    cost_per_1k_tokens: float
    max_tokens_per_call: int
    
class OpenAICostCalculator:
    """Calculate OpenAI embedding costs."""
    
    MODELS = {
        'text-embedding-ada-002': OpenAIEmbeddingPricing(
            model='text-embedding-ada-002',
            cost_per_1k_tokens=0.0001,
            max_tokens_per_call=8191
        ),
        'text-embedding-3-small': OpenAIEmbeddingPricing(
            model='text-embedding-3-small',
            cost_per_1k_tokens=0.00002,
            max_tokens_per_call=8191
        ),
        'text-embedding-3-large': OpenAIEmbeddingPricing(
            model='text-embedding-3-large',
            cost_per_1k_tokens=0.00013,
            max_tokens_per_call=8191
        )
    }
    
    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """
        Estimate token count.
        Rough estimate: 1 token ≈ 4 characters
        """
        return len(text) // 4
    
    @classmethod
    def calculate_embedding_cost(cls, num_documents: int,
                                 avg_document_length: int,
                                 model: str = 'text-embedding-ada-002',
                                 fields_to_embed: int = 1) -> dict:
        """
        Calculate cost to generate embeddings.
        
        Args:
            num_documents: Number of documents to embed
            avg_document_length: Average length in characters
            model: OpenAI model name
            fields_to_embed: Number of fields per document to embed
        
        Returns:
            Cost breakdown
        """
        if model not in cls.MODELS:
            raise ValueError(f"Unknown model: {model}")
        
        pricing = cls.MODELS[model]
        
        # Estimate tokens per document
        tokens_per_doc = cls.estimate_tokens(' ' * avg_document_length)
        total_tokens = num_documents * tokens_per_doc * fields_to_embed
        
        # Calculate cost
        total_cost = (total_tokens / 1000) * pricing.cost_per_1k_tokens
        
        return {
            'model': model,
            'num_documents': num_documents,
            'avg_document_length': avg_document_length,
            'fields_to_embed': fields_to_embed,
            'tokens_per_document': tokens_per_doc,
            'total_tokens': total_tokens,
            'cost_per_1k_tokens': pricing.cost_per_1k_tokens,
            'total_cost': total_cost,
            'cost_per_document': total_cost / num_documents if num_documents > 0 else 0
        }
    
    @classmethod
    def calculate_query_cost(cls, queries_per_month: int,
                            avg_query_length: int,
                            model: str = 'text-embedding-ada-002') -> dict:
        """
        Calculate monthly cost for query embeddings.
        """
        if model not in cls.MODELS:
            raise ValueError(f"Unknown model: {model}")
        
        pricing = cls.MODELS[model]
        
        tokens_per_query = cls.estimate_tokens(' ' * avg_query_length)
        total_tokens = queries_per_month * tokens_per_query
        
        total_cost = (total_tokens / 1000) * pricing.cost_per_1k_tokens
        
        return {
            'model': model,
            'queries_per_month': queries_per_month,
            'avg_query_length': avg_query_length,
            'tokens_per_query': tokens_per_query,
            'total_tokens': total_tokens,
            'total_monthly_cost': total_cost
        }

# Usage
openai_calc = OpenAICostCalculator()

# Calculate indexing cost
indexing_cost = openai_calc.calculate_embedding_cost(
    num_documents=100000,
    avg_document_length=500,  # characters
    model='text-embedding-ada-002',
    fields_to_embed=2  # title + description
)

print(f"OpenAI Embedding Cost (Indexing):")
print(f"  Documents: {indexing_cost['num_documents']:,}")
print(f"  Fields: {indexing_cost['fields_to_embed']}")
print(f"  Total tokens: {indexing_cost['total_tokens']:,}")
print(f"  Total cost: ${indexing_cost['total_cost']:.2f}")
print(f"  Cost per document: ${indexing_cost['cost_per_document']:.4f}")

# Calculate query cost
query_cost = openai_calc.calculate_query_cost(
    queries_per_month=100000,
    avg_query_length=50,
    model='text-embedding-ada-002'
)

print(f"\nOpenAI Embedding Cost (Queries):")
print(f"  Queries/month: {query_cost['queries_per_month']:,}")
print(f"  Monthly cost: ${query_cost['total_monthly_cost']:.2f}")
```

### Cosmos DB Cost

```python
@dataclass
class CosmosDBPricing:
    """Cosmos DB pricing structure."""
    provisioning_model: str  # 'provisioned' or 'serverless'
    
    # Provisioned throughput pricing (per 100 RU/s)
    RU_COST_PER_HOUR = 0.008
    
    # Serverless pricing (per million RUs)
    SERVERLESS_COST_PER_MILLION_RUS = 0.25
    
    # Storage pricing
    TRANSACTIONAL_STORAGE_COST_PER_GB = 0.25
    ANALYTICAL_STORAGE_COST_PER_GB = 0.03
    
    @classmethod
    def calculate_provisioned_cost(cls, read_rus: int, write_rus: int,
                                   storage_gb: float,
                                   analytical_storage_gb: float = 0) -> dict:
        """
        Calculate provisioned throughput cost.
        
        Args:
            read_rus: Provisioned read RU/s
            write_rus: Provisioned write RU/s  
            storage_gb: Transactional storage in GB
            analytical_storage_gb: Analytical storage in GB
        """
        # Total provisioned RU/s
        total_rus = read_rus + write_rus
        
        # Throughput cost (per hour)
        throughput_cost_per_hour = (total_rus / 100) * cls.RU_COST_PER_HOUR
        throughput_cost_per_month = throughput_cost_per_hour * 730
        
        # Storage cost (per month)
        storage_cost = storage_gb * cls.TRANSACTIONAL_STORAGE_COST_PER_GB
        analytical_cost = analytical_storage_gb * cls.ANALYTICAL_STORAGE_COST_PER_GB
        
        total_monthly_cost = throughput_cost_per_month + storage_cost + analytical_cost
        
        return {
            'provisioning_model': 'provisioned',
            'read_rus': read_rus,
            'write_rus': write_rus,
            'total_rus': total_rus,
            'throughput_cost_monthly': throughput_cost_per_month,
            'storage_gb': storage_gb,
            'storage_cost_monthly': storage_cost,
            'analytical_storage_gb': analytical_storage_gb,
            'analytical_cost_monthly': analytical_cost,
            'total_monthly_cost': total_monthly_cost
        }
    
    @classmethod
    def calculate_serverless_cost(cls, reads_per_month: int,
                                  writes_per_month: int,
                                  storage_gb: float,
                                  avg_read_rus: int = 1,
                                  avg_write_rus: int = 5) -> dict:
        """
        Calculate serverless cost.
        
        Args:
            reads_per_month: Number of read operations
            writes_per_month: Number of write operations
            storage_gb: Storage in GB
            avg_read_rus: Average RUs per read operation
            avg_write_rus: Average RUs per write operation
        """
        # Total RUs consumed
        total_read_rus = reads_per_month * avg_read_rus
        total_write_rus = writes_per_month * avg_write_rus
        total_rus = total_read_rus + total_write_rus
        
        # RU cost
        ru_cost = (total_rus / 1_000_000) * cls.SERVERLESS_COST_PER_MILLION_RUS
        
        # Storage cost
        storage_cost = storage_gb * cls.TRANSACTIONAL_STORAGE_COST_PER_GB
        
        total_monthly_cost = ru_cost + storage_cost
        
        return {
            'provisioning_model': 'serverless',
            'reads_per_month': reads_per_month,
            'writes_per_month': writes_per_month,
            'total_rus_consumed': total_rus,
            'ru_cost_monthly': ru_cost,
            'storage_gb': storage_gb,
            'storage_cost_monthly': storage_cost,
            'total_monthly_cost': total_monthly_cost
        }

# Usage
cosmos_pricing = CosmosDBPricing()

# Provisioned throughput example
provisioned_cost = cosmos_pricing.calculate_provisioned_cost(
    read_rus=1000,  # 1000 RU/s for reads
    write_rus=500,  # 500 RU/s for writes
    storage_gb=100,  # 100 GB storage
    analytical_storage_gb=100  # 100 GB analytical
)

print(f"Cosmos DB - Provisioned Throughput:")
print(f"  Read RU/s: {provisioned_cost['read_rus']}")
print(f"  Write RU/s: {provisioned_cost['write_rus']}")
print(f"  Throughput cost: ${provisioned_cost['throughput_cost_monthly']:.2f}/month")
print(f"  Storage cost: ${provisioned_cost['storage_cost_monthly']:.2f}/month")
print(f"  Total: ${provisioned_cost['total_monthly_cost']:.2f}/month")

# Serverless example
serverless_cost = cosmos_pricing.calculate_serverless_cost(
    reads_per_month=10_000_000,  # 10M reads
    writes_per_month=1_000_000,  # 1M writes
    storage_gb=50,
    avg_read_rus=1,
    avg_write_rus=5
)

print(f"\n\nCosmos DB - Serverless:")
print(f"  Reads: {serverless_cost['reads_per_month']:,}")
print(f"  Writes: {serverless_cost['writes_per_month']:,}")
print(f"  Total RUs: {serverless_cost['total_rus_consumed']:,}")
print(f"  RU cost: ${serverless_cost['ru_cost_monthly']:.2f}/month")
print(f"  Storage cost: ${serverless_cost['storage_cost_monthly']:.2f}/month")
print(f"  Total: ${serverless_cost['total_monthly_cost']:.2f}/month")
```

### Azure Monitor Cost

```python
class AzureMonitorPricing:
    """Azure Monitor (Log Analytics) pricing."""
    
    # Log ingestion cost (per GB)
    INGESTION_COST_PER_GB = 2.76
    
    # Data retention (first 31 days free, then per GB/month)
    RETENTION_COST_PER_GB_MONTH = 0.12
    
    # Query cost (per GB scanned)
    QUERY_COST_PER_GB = 0.0055
    
    @classmethod
    def calculate_log_analytics_cost(cls, daily_ingestion_gb: float,
                                     retention_days: int = 90,
                                     query_gb_per_month: float = 0) -> dict:
        """
        Calculate Azure Monitor / Log Analytics cost.
        
        Args:
            daily_ingestion_gb: Daily log ingestion in GB
            retention_days: Log retention period
            query_gb_per_month: Data scanned by queries per month
        """
        # Monthly ingestion
        monthly_ingestion_gb = daily_ingestion_gb * 30
        ingestion_cost = monthly_ingestion_gb * cls.INGESTION_COST_PER_GB
        
        # Retention cost (beyond 31 days)
        if retention_days > 31:
            # Average storage over the month
            avg_storage_gb = monthly_ingestion_gb * (retention_days / 30)
            
            # Only charged for days beyond 31
            billable_days = retention_days - 31
            retention_cost = avg_storage_gb * (billable_days / 30) * cls.RETENTION_COST_PER_GB_MONTH
        else:
            retention_cost = 0
        
        # Query cost
        query_cost = query_gb_per_month * cls.QUERY_COST_PER_GB
        
        total_cost = ingestion_cost + retention_cost + query_cost
        
        return {
            'daily_ingestion_gb': daily_ingestion_gb,
            'monthly_ingestion_gb': monthly_ingestion_gb,
            'ingestion_cost': ingestion_cost,
            'retention_days': retention_days,
            'retention_cost': retention_cost,
            'query_gb_per_month': query_gb_per_month,
            'query_cost': query_cost,
            'total_monthly_cost': total_cost
        }

# Usage
monitor_cost = AzureMonitorPricing.calculate_log_analytics_cost(
    daily_ingestion_gb=5.0,  # 5 GB per day
    retention_days=90,  # 90 days retention
    query_gb_per_month=50.0  # 50 GB scanned by queries
)

print(f"Azure Monitor Cost:")
print(f"  Daily ingestion: {monitor_cost['daily_ingestion_gb']} GB")
print(f"  Monthly ingestion: {monitor_cost['monthly_ingestion_gb']:.1f} GB")
print(f"  Ingestion cost: ${monitor_cost['ingestion_cost']:.2f}/month")
print(f"  Retention cost: ${monitor_cost['retention_cost']:.2f}/month")
print(f"  Query cost: ${monitor_cost['query_cost']:.2f}/month")
print(f"  Total: ${monitor_cost['total_monthly_cost']:.2f}/month")
```

---

## TCO Calculator

### Complete TCO Analysis

```python
class TCOCalculator:
    """Total Cost of Ownership calculator for search solution."""
    
    def __init__(self):
        self.components = {}
    
    def add_search_service(self, tier: str, replicas: int, partitions: int):
        """Add Azure AI Search service cost."""
        pricing = AzureSearchPricing.get_pricing(tier)
        monthly_cost = pricing.calculate_monthly_cost(replicas, partitions)
        
        self.components['search_service'] = {
            'tier': tier,
            'replicas': replicas,
            'partitions': partitions,
            'monthly_cost': monthly_cost,
            'annual_cost': monthly_cost * 12
        }
    
    def add_openai_indexing(self, num_documents: int, avg_doc_length: int,
                           fields_to_embed: int, model: str = 'text-embedding-ada-002'):
        """Add OpenAI embedding cost for indexing."""
        calc = OpenAICostCalculator()
        cost = calc.calculate_embedding_cost(
            num_documents, avg_doc_length, model, fields_to_embed
        )
        
        # One-time indexing cost
        self.components['openai_indexing'] = {
            'one_time_cost': cost['total_cost'],
            'model': model,
            'documents': num_documents
        }
    
    def add_openai_queries(self, queries_per_month: int, avg_query_length: int,
                          model: str = 'text-embedding-ada-002'):
        """Add OpenAI embedding cost for queries."""
        calc = OpenAICostCalculator()
        cost = calc.calculate_query_cost(queries_per_month, avg_query_length, model)
        
        self.components['openai_queries'] = {
            'monthly_cost': cost['total_monthly_cost'],
            'annual_cost': cost['total_monthly_cost'] * 12,
            'queries_per_month': queries_per_month
        }
    
    def add_cosmos_db(self, provisioning_model: str, **kwargs):
        """Add Cosmos DB cost."""
        pricing = CosmosDBPricing()
        
        if provisioning_model == 'provisioned':
            cost = pricing.calculate_provisioned_cost(**kwargs)
        else:
            cost = pricing.calculate_serverless_cost(**kwargs)
        
        self.components['cosmos_db'] = {
            'provisioning_model': provisioning_model,
            'monthly_cost': cost['total_monthly_cost'],
            'annual_cost': cost['total_monthly_cost'] * 12
        }
    
    def add_azure_monitor(self, daily_ingestion_gb: float, retention_days: int):
        """Add Azure Monitor cost."""
        pricing = AzureMonitorPricing()
        cost = pricing.calculate_log_analytics_cost(daily_ingestion_gb, retention_days)
        
        self.components['azure_monitor'] = {
            'monthly_cost': cost['total_monthly_cost'],
            'annual_cost': cost['total_monthly_cost'] * 12,
            'daily_ingestion_gb': daily_ingestion_gb
        }
    
    def calculate_total(self) -> dict:
        """Calculate total cost."""
        monthly_recurring = 0
        annual_recurring = 0
        one_time = 0
        
        for component, details in self.components.items():
            if 'monthly_cost' in details:
                monthly_recurring += details['monthly_cost']
            if 'annual_cost' in details:
                annual_recurring += details['annual_cost']
            if 'one_time_cost' in details:
                one_time += details['one_time_cost']
        
        # 3-year TCO
        three_year_tco = (annual_recurring * 3) + one_time
        
        return {
            'one_time_costs': one_time,
            'monthly_recurring': monthly_recurring,
            'annual_recurring': annual_recurring,
            'three_year_tco': three_year_tco,
            'breakdown': self.components
        }
    
    def print_summary(self):
        """Print cost summary."""
        total = self.calculate_total()
        
        print("=" * 60)
        print("TOTAL COST OF OWNERSHIP (TCO)")
        print("=" * 60)
        
        print(f"\nOne-Time Costs:")
        print(f"  ${total['one_time_costs']:,.2f}")
        
        print(f"\nRecurring Costs:")
        print(f"  Monthly: ${total['monthly_recurring']:,.2f}")
        print(f"  Annual: ${total['annual_recurring']:,.2f}")
        
        print(f"\n3-Year TCO: ${total['three_year_tco']:,.2f}")
        
        print(f"\n\nCost Breakdown by Component:")
        print("-" * 60)
        
        for component, details in total['breakdown'].items():
            print(f"\n{component.replace('_', ' ').title()}:")
            
            if 'one_time_cost' in details:
                print(f"  One-time: ${details['one_time_cost']:,.2f}")
            
            if 'monthly_cost' in details:
                monthly = details['monthly_cost']
                annual = details['annual_cost']
                pct = (annual / total['annual_recurring'] * 100) if total['annual_recurring'] > 0 else 0
                
                print(f"  Monthly: ${monthly:,.2f}")
                print(f"  Annual: ${annual:,.2f} ({pct:.1f}% of total)")

# Usage
tco = TCOCalculator()

# Add components
tco.add_search_service('S2', replicas=3, partitions=1)
tco.add_openai_indexing(
    num_documents=1_000_000,
    avg_doc_length=500,
    fields_to_embed=2,
    model='text-embedding-ada-002'
)
tco.add_openai_queries(
    queries_per_month=500_000,
    avg_query_length=50,
    model='text-embedding-ada-002'
)
tco.add_cosmos_db(
    provisioning_model='serverless',
    reads_per_month=10_000_000,
    writes_per_month=1_000_000,
    storage_gb=100
)
tco.add_azure_monitor(
    daily_ingestion_gb=2.0,
    retention_days=90
)

# Print summary
tco.print_summary()
```

---

## Optimization Strategies

### Search Service Optimization

```python
class SearchCostOptimizer:
    """Optimize Azure AI Search costs."""
    
    @staticmethod
    def optimize_tier_selection(target_qps: float, storage_gb: float,
                               safety_margin: float = 1.5) -> list:
        """
        Find cost-optimal tier configurations.
        """
        required_qps = target_qps * safety_margin
        recommendations = []
        
        for tier_name, pricing in AzureSearchPricing.TIERS.items():
            if tier_name == 'Free':
                continue
            
            max_replicas = 12 if tier_name != 'Basic' else 3
            max_partitions = 12 if tier_name != 'Basic' else 1
            
            # Find minimum configuration meeting requirements
            for partitions in range(1, max_partitions + 1):
                # Check storage
                total_storage = pricing.storage_included_gb * partitions
                if total_storage < storage_gb:
                    continue
                
                # Check QPS
                for replicas in range(1, max_replicas + 1):
                    total_qps = pricing.qps_per_replica_partition * replicas * partitions
                    
                    if total_qps >= required_qps:
                        monthly_cost = pricing.calculate_monthly_cost(replicas, partitions)
                        
                        recommendations.append({
                            'tier': tier_name,
                            'replicas': replicas,
                            'partitions': partitions,
                            'total_qps': total_qps,
                            'total_storage_gb': total_storage,
                            'monthly_cost': monthly_cost,
                            'cost_per_qps': monthly_cost / total_qps
                        })
                        break
        
        # Sort by cost
        recommendations.sort(key=lambda x: x['monthly_cost'])
        
        return recommendations
    
    @staticmethod
    def recommend_replica_count(peak_qps: float, tier: str,
                               partitions: int = 1) -> dict:
        """
        Recommend optimal replica count for availability and performance.
        """
        pricing = AzureSearchPricing.get_pricing(tier)
        qps_per_unit = pricing.qps_per_replica_partition
        
        # Minimum 2 replicas for 99.9% SLA
        min_replicas_for_sla = 2
        
        # Replicas needed for QPS
        replicas_for_qps = max(1, int(np.ceil(peak_qps / (qps_per_unit * partitions))))
        
        # Recommended: max of SLA and QPS requirements
        recommended_replicas = max(min_replicas_for_sla, replicas_for_qps)
        
        return {
            'tier': tier,
            'partitions': partitions,
            'peak_qps': peak_qps,
            'qps_per_replica_partition': qps_per_unit,
            'replicas_for_qps': replicas_for_qps,
            'replicas_for_sla': min_replicas_for_sla,
            'recommended_replicas': recommended_replicas,
            'total_capacity_qps': recommended_replicas * partitions * qps_per_unit,
            'monthly_cost': pricing.calculate_monthly_cost(recommended_replicas, partitions)
        }

# Usage
optimizer = SearchCostOptimizer()

# Find optimal tier
import numpy as np

recommendations = optimizer.optimize_tier_selection(
    target_qps=100,
    storage_gb=150,
    safety_margin=1.5
)

print("Cost-Optimal Configurations:")
print("\nTop 3 Options:")
for i, rec in enumerate(recommendations[:3], 1):
    print(f"\n{i}. {rec['tier']}")
    print(f"   Config: {rec['replicas']} replicas × {rec['partitions']} partitions")
    print(f"   Capacity: {rec['total_qps']} QPS, {rec['total_storage_gb']} GB")
    print(f"   Cost: ${rec['monthly_cost']:.2f}/month (${rec['cost_per_qps']:.2f} per QPS)")

# Recommend replicas
replica_rec = optimizer.recommend_replica_count(
    peak_qps=120,
    tier='S2',
    partitions=1
)

print(f"\n\nReplica Recommendation:")
print(f"  Tier: {replica_rec['tier']}")
print(f"  Peak QPS: {replica_rec['peak_qps']}")
print(f"  Recommended replicas: {replica_rec['recommended_replicas']}")
print(f"  Total capacity: {replica_rec['total_capacity_qps']} QPS")
print(f"  Monthly cost: ${replica_rec['monthly_cost']:.2f}")
```

### Embedding Cost Optimization

```python
class EmbeddingCostOptimizer:
    """Optimize OpenAI embedding costs."""
    
    @staticmethod
    def calculate_caching_savings(queries_per_month: int,
                                  cache_hit_rate: float,
                                  avg_query_length: int,
                                  model: str = 'text-embedding-ada-002') -> dict:
        """
        Calculate savings from query caching.
        
        With caching, only cache misses require new embeddings.
        """
        calc = OpenAICostCalculator()
        
        # Cost without caching
        cost_no_cache = calc.calculate_query_cost(
            queries_per_month, avg_query_length, model
        )
        
        # Cost with caching (only cache misses)
        cache_misses = queries_per_month * (1 - cache_hit_rate)
        cost_with_cache = calc.calculate_query_cost(
            int(cache_misses), avg_query_length, model
        )
        
        savings = cost_no_cache['total_monthly_cost'] - cost_with_cache['total_monthly_cost']
        savings_pct = (savings / cost_no_cache['total_monthly_cost'] * 100) if cost_no_cache['total_monthly_cost'] > 0 else 0
        
        return {
            'queries_per_month': queries_per_month,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': int(queries_per_month * cache_hit_rate),
            'cache_misses': int(cache_misses),
            'cost_without_cache': cost_no_cache['total_monthly_cost'],
            'cost_with_cache': cost_with_cache['total_monthly_cost'],
            'monthly_savings': savings,
            'annual_savings': savings * 12,
            'savings_percent': savings_pct
        }
    
    @staticmethod
    def compare_embedding_models(num_documents: int,
                                avg_doc_length: int,
                                fields_to_embed: int = 1) -> dict:
        """
        Compare costs across different embedding models.
        """
        calc = OpenAICostCalculator()
        comparison = {}
        
        for model_name in calc.MODELS.keys():
            cost = calc.calculate_embedding_cost(
                num_documents, avg_doc_length, model_name, fields_to_embed
            )
            
            comparison[model_name] = {
                'total_cost': cost['total_cost'],
                'cost_per_1k_tokens': cost['cost_per_1k_tokens'],
                'cost_per_document': cost['cost_per_document']
            }
        
        return comparison

# Usage
emb_optimizer = EmbeddingCostOptimizer()

# Calculate caching savings
caching_savings = emb_optimizer.calculate_caching_savings(
    queries_per_month=1_000_000,
    cache_hit_rate=0.80,  # 80% cache hit rate
    avg_query_length=50
)

print("Embedding Caching Savings:")
print(f"  Queries/month: {caching_savings['queries_per_month']:,}")
print(f"  Cache hit rate: {caching_savings['cache_hit_rate']:.0%}")
print(f"  Cache hits: {caching_savings['cache_hits']:,}")
print(f"  Cache misses: {caching_savings['cache_misses']:,}")
print(f"\n  Cost without cache: ${caching_savings['cost_without_cache']:.2f}/month")
print(f"  Cost with cache: ${caching_savings['cost_with_cache']:.2f}/month")
print(f"  Monthly savings: ${caching_savings['monthly_savings']:.2f} ({caching_savings['savings_percent']:.1f}%)")
print(f"  Annual savings: ${caching_savings['annual_savings']:.2f}")

# Compare models
model_comparison = emb_optimizer.compare_embedding_models(
    num_documents=1_000_000,
    avg_doc_length=500,
    fields_to_embed=2
)

print("\n\nEmbedding Model Comparison:")
for model, costs in model_comparison.items():
    print(f"\n{model}:")
    print(f"  Total cost: ${costs['total_cost']:.2f}")
    print(f"  Cost per 1k tokens: ${costs['cost_per_1k_tokens']:.6f}")
    print(f"  Cost per document: ${costs['cost_per_document']:.6f}")
```

---

## Cost Monitoring

### Budget Alerts

```python
from azure.mgmt.costmanagement import CostManagementClient
from azure.mgmt.consumption import ConsumptionManagementClient

class CostMonitor:
    """Monitor and alert on Azure costs."""
    
    def __init__(self, subscription_id: str, credential):
        self.subscription_id = subscription_id
        self.credential = credential
        self.cost_client = CostManagementClient(credential)
        self.consumption_client = ConsumptionManagementClient(credential, subscription_id)
    
    def setup_budget_alert(self, budget_name: str, monthly_budget: float,
                          alert_thresholds: list = [50, 80, 100]) -> dict:
        """
        Setup budget alerts.
        
        Args:
            budget_name: Budget name
            monthly_budget: Monthly budget in USD
            alert_thresholds: Alert at these percentages (e.g., [50, 80, 100])
        """
        # This would use Azure Cost Management API to create budget
        # Simplified example
        
        alerts = []
        for threshold in alert_thresholds:
            alert_amount = monthly_budget * (threshold / 100)
            alerts.append({
                'threshold_percent': threshold,
                'threshold_amount': alert_amount,
                'alert_type': 'email',  # or 'action_group'
                'enabled': True
            })
        
        budget_config = {
            'budget_name': budget_name,
            'monthly_budget': monthly_budget,
            'alerts': alerts,
            'scope': f'/subscriptions/{self.subscription_id}'
        }
        
        print(f"Budget Configuration: {budget_name}")
        print(f"  Monthly Budget: ${monthly_budget:,.2f}")
        print(f"  Alerts:")
        for alert in alerts:
            print(f"    {alert['threshold_percent']}% (${alert['threshold_amount']:,.2f})")
        
        return budget_config
    
    def get_current_spend(self, resource_group: str = None) -> dict:
        """
        Get current month-to-date spending.
        """
        from datetime import datetime, timedelta
        
        # This would query actual costs from Azure
        # Simplified example
        
        # Current month
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Simulate cost data
        current_spend = {
            'period_start': month_start.isoformat(),
            'period_end': now.isoformat(),
            'total_cost': 1250.50,  # Simulated
            'breakdown': {
                'search_service': 800.00,
                'openai': 250.50,
                'cosmos_db': 150.00,
                'azure_monitor': 50.00
            },
            'currency': 'USD'
        }
        
        return current_spend
    
    def forecast_end_of_month_cost(self, current_spend: float,
                                   days_elapsed: int,
                                   days_in_month: int) -> dict:
        """
        Forecast end-of-month cost based on current trend.
        """
        daily_avg = current_spend / days_elapsed if days_elapsed > 0 else 0
        forecasted_total = daily_avg * days_in_month
        
        return {
            'current_spend': current_spend,
            'days_elapsed': days_elapsed,
            'days_in_month': days_in_month,
            'daily_average': daily_avg,
            'forecasted_total': forecasted_total
        }

# Usage
# monitor = CostMonitor(
#     subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
#     credential=DefaultAzureCredential()
# )

# Setup budget
# budget = monitor.setup_budget_alert(
#     budget_name="search_service_budget",
#     monthly_budget=3000.00,
#     alert_thresholds=[50, 75, 90, 100]
# )

# Get current spend
# spend = monitor.get_current_spend()
# print(f"\nCurrent Month-to-Date Spend: ${spend['total_cost']:,.2f}")

# Forecast
from datetime import datetime

now = datetime.now()
days_elapsed = now.day
days_in_month = 30  # Simplified

# forecast = monitor.forecast_end_of_month_cost(
#     current_spend=1250.50,
#     days_elapsed=days_elapsed,
#     days_in_month=days_in_month
# )
# 
# print(f"\nCost Forecast:")
# print(f"  Days elapsed: {forecast['days_elapsed']}/{forecast['days_in_month']}")
# print(f"  Daily average: ${forecast['daily_average']:.2f}")
# print(f"  Forecasted EOM: ${forecast['forecasted_total']:.2f}")
```

---

## Reserved Capacity

### Reserved Instance Pricing

```python
class ReservedCapacityAnalyzer:
    """Analyze reserved capacity pricing."""
    
    # Reserved instance discounts
    RESERVED_DISCOUNTS = {
        '1_year': 0.15,  # 15% discount
        '3_year': 0.35   # 35% discount
    }
    
    @classmethod
    def calculate_savings(cls, tier: str, replicas: int, partitions: int,
                         commitment_years: int) -> dict:
        """
        Calculate savings from reserved capacity.
        
        Args:
            tier: Search service tier
            replicas: Number of replicas
            partitions: Number of partitions
            commitment_years: 1 or 3 years
        """
        pricing = AzureSearchPricing.get_pricing(tier)
        
        # Pay-as-you-go cost
        payg_monthly = pricing.calculate_monthly_cost(replicas, partitions)
        payg_total = payg_monthly * 12 * commitment_years
        
        # Reserved capacity discount
        discount_key = f'{commitment_years}_year'
        if discount_key not in cls.RESERVED_DISCOUNTS:
            raise ValueError(f"Commitment must be 1 or 3 years")
        
        discount = cls.RESERVED_DISCOUNTS[discount_key]
        reserved_monthly = payg_monthly * (1 - discount)
        reserved_total = reserved_monthly * 12 * commitment_years
        
        savings = payg_total - reserved_total
        savings_pct = (savings / payg_total * 100) if payg_total > 0 else 0
        
        # Break-even point (months)
        upfront_cost = reserved_total
        monthly_savings = payg_monthly - reserved_monthly
        breakeven_months = upfront_cost / (payg_monthly) if payg_monthly > 0 else 0
        
        return {
            'tier': tier,
            'replicas': replicas,
            'partitions': partitions,
            'commitment_years': commitment_years,
            'discount_percent': discount * 100,
            'payg': {
                'monthly': payg_monthly,
                'total': payg_total
            },
            'reserved': {
                'monthly': reserved_monthly,
                'total': reserved_total
            },
            'savings': {
                'total': savings,
                'percent': savings_pct,
                'monthly': monthly_savings
            },
            'breakeven_months': min(breakeven_months, commitment_years * 12)
        }
    
    @classmethod
    def compare_commitments(cls, tier: str, replicas: int, partitions: int) -> dict:
        """
        Compare 1-year vs 3-year commitment.
        """
        one_year = cls.calculate_savings(tier, replicas, partitions, 1)
        three_year = cls.calculate_savings(tier, replicas, partitions, 3)
        
        return {
            'pay_as_you_go': {
                '1_year_cost': one_year['payg']['total'],
                '3_year_cost': three_year['payg']['total']
            },
            '1_year_reserved': {
                'cost': one_year['reserved']['total'],
                'savings': one_year['savings']['total'],
                'savings_percent': one_year['savings']['percent']
            },
            '3_year_reserved': {
                'cost': three_year['reserved']['total'],
                'savings': three_year['savings']['total'],
                'savings_percent': three_year['savings']['percent']
            }
        }

# Usage
reserved_analyzer = ReservedCapacityAnalyzer()

# Calculate 3-year reserved savings
savings_3yr = reserved_analyzer.calculate_savings(
    tier='S2',
    replicas=3,
    partitions=2,
    commitment_years=3
)

print("Reserved Capacity Analysis (S2, 3 replicas, 2 partitions):")
print(f"\n3-Year Commitment:")
print(f"  Discount: {savings_3yr['discount_percent']:.0f}%")
print(f"  Pay-as-you-go total: ${savings_3yr['payg']['total']:,.2f}")
print(f"  Reserved total: ${savings_3yr['reserved']['total']:,.2f}")
print(f"  Total savings: ${savings_3yr['savings']['total']:,.2f} ({savings_3yr['savings']['percent']:.1f}%)")
print(f"  Monthly savings: ${savings_3yr['savings']['monthly']:.2f}")

# Compare commitments
comparison = reserved_analyzer.compare_commitments('S2', 3, 2)

print(f"\n\nCommitment Comparison:")
print(f"\nPay-as-you-go:")
print(f"  1-year: ${comparison['pay_as_you_go']['1_year_cost']:,.2f}")
print(f"  3-year: ${comparison['pay_as_you_go']['3_year_cost']:,.2f}")

print(f"\n1-Year Reserved:")
print(f"  Cost: ${comparison['1_year_reserved']['cost']:,.2f}")
print(f"  Savings: ${comparison['1_year_reserved']['savings']:,.2f} ({comparison['1_year_reserved']['savings_percent']:.1f}%)")

print(f"\n3-Year Reserved:")
print(f"  Cost: ${comparison['3_year_reserved']['cost']:,.2f}")
print(f"  Savings: ${comparison['3_year_reserved']['savings']:,.2f} ({comparison['3_year_reserved']['savings_percent']:.1f}%)")
```

---

## Best Practices

### ✅ Cost Optimization Checklist

1. **Right-size your tier**
   - Start with S1, scale up only when needed
   - Don't over-provision QPS capacity
   - Monitor actual usage vs capacity

2. **Optimize embeddings**
   - Cache query embeddings (80%+ hit rate saves ~80%)
   - Use cheaper models when accuracy allows
   - Batch embedding generation

3. **Use reserved capacity for production**
   - 35% savings with 3-year commitment
   - Only for stable, long-term workloads
   - Evaluate at 6-month mark

4. **Minimize logging costs**
   - Only log what you need
   - Use sampling for high-volume logs
   - Appropriate retention (90 days max for most)

5. **Cosmos DB optimization**
   - Use serverless for unpredictable workloads
   - Optimize partition keys to avoid hot partitions
   - Enable TTL for temporary data

6. **Monitor and alert**
   - Set budgets with 50%, 80%, 100% alerts
   - Review costs weekly
   - Track cost per query metric

7. **Review quarterly**
   - Analyze actual vs forecasted usage
   - Adjust tier/replicas based on trends
   - Re-evaluate embedding strategy

---

## Next Steps

- **[Multi-Region Deployment](./20-multi-region-deployment.md)** - Geographic distribution
- **[Monitoring & Alerting](./23-monitoring-alerting.md)** - Production monitoring
- **[Azure Service Tiers](./28-azure-service-tiers.md)** - Detailed tier comparison

---

*See also: [Load Testing](./18-load-testing.md) | [A/B Testing](./17-ab-testing-framework.md)*
