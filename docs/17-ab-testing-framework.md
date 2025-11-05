# A/B Testing Framework

Complete guide to designing, implementing, and analyzing A/B tests for search relevance experiments.

## 📋 Table of Contents
- [Overview](#overview)
- [Experimental Design](#experimental-design)
- [Implementation](#implementation)
- [Statistical Analysis](#statistical-analysis)
- [Azure Integration](#azure-integration)
- [Results Tracking](#results-tracking)
- [Analysis Tools](#analysis-tools)
- [Best Practices](#best-practices)

---

## Overview

### A/B Testing Fundamentals

```python
class ABTestingOverview:
    """Understanding A/B testing for search."""
    
    @staticmethod
    def testing_framework():
        """
        A/B testing framework components:
        
        1. Hypothesis: What you're testing
        2. Variants: Control (A) vs Treatment (B)
        3. Randomization: User assignment to variants
        4. Metrics: Success criteria
        5. Sample size: Statistical power
        6. Duration: Test run time
        7. Analysis: Statistical significance
        """
        return {
            'hypothesis_examples': [
                'New scoring profile improves relevance',
                'Hybrid search outperforms full-text search',
                'Semantic ranking increases click-through rate',
                'Custom analyzer reduces zero-result queries'
            ],
            'variant_types': {
                'control': 'Baseline/current implementation',
                'treatment': 'New implementation being tested',
                'multi_variant': '2+ treatments (A/B/C/D testing)'
            },
            'key_metrics': {
                'relevance': ['NDCG@10', 'MRR', 'MAP'],
                'engagement': ['CTR', 'time_on_page', 'bounce_rate'],
                'business': ['conversion_rate', 'revenue_per_search', 'cart_adds']
            },
            'statistical_requirements': {
                'significance_level': 'α = 0.05 (95% confidence)',
                'power': 'β = 0.80 (80% power to detect effect)',
                'minimum_detectable_effect': 'MDE typically 2-5%'
            }
        }
    
    @staticmethod
    def experiment_lifecycle():
        """
        A/B test lifecycle stages.
        """
        return {
            '1_design': {
                'tasks': [
                    'Define hypothesis',
                    'Select metrics',
                    'Calculate sample size',
                    'Determine duration',
                    'Plan randomization'
                ],
                'duration': '1-3 days'
            },
            '2_implementation': {
                'tasks': [
                    'Build variants',
                    'Implement randomization',
                    'Setup tracking',
                    'Validate instrumentation',
                    'Run A/A test'
                ],
                'duration': '3-7 days'
            },
            '3_execution': {
                'tasks': [
                    'Launch experiment',
                    'Monitor metrics',
                    'Check for issues',
                    'Ensure balanced traffic'
                ],
                'duration': '1-4 weeks'
            },
            '4_analysis': {
                'tasks': [
                    'Statistical testing',
                    'Segment analysis',
                    'Validate results',
                    'Document findings'
                ],
                'duration': '1-2 days'
            },
            '5_decision': {
                'tasks': [
                    'Review results',
                    'Make decision (ship/iterate/abandon)',
                    'Communicate results',
                    'Plan rollout if shipping'
                ],
                'duration': '1 day'
            }
        }

# Usage
overview = ABTestingOverview()

framework = overview.testing_framework()
print("A/B Testing Framework:")
print("\nHypothesis Examples:")
for hypothesis in framework['hypothesis_examples']:
    print(f"  • {hypothesis}")

print("\nKey Metrics:")
for category, metrics in framework['key_metrics'].items():
    print(f"  {category}: {', '.join(metrics)}")

lifecycle = overview.experiment_lifecycle()
print("\n\nExperiment Lifecycle:")
for stage, details in lifecycle.items():
    print(f"\n{stage}:")
    print(f"  Duration: {details['duration']}")
    print(f"  Tasks: {len(details['tasks'])}")
```

---

## Experimental Design

### Sample Size Calculation

```python
import math
from scipy import stats

class SampleSizeCalculator:
    """Calculate required sample size for A/B tests."""
    
    @staticmethod
    def calculate_for_proportions(p1: float, p2: float, 
                                   alpha: float = 0.05,
                                   power: float = 0.80) -> int:
        """
        Calculate sample size for proportion metrics (CTR, conversion rate).
        
        Args:
            p1: Baseline proportion (e.g., 0.25 for 25% CTR)
            p2: Expected proportion in treatment
            alpha: Significance level (default 0.05)
            power: Statistical power (default 0.80)
        
        Returns:
            Required sample size per variant
        """
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
        z_beta = stats.norm.ppf(power)
        
        # Pooled proportion
        p_pooled = (p1 + p2) / 2
        
        # Sample size calculation
        numerator = (z_alpha + z_beta) ** 2 * 2 * p_pooled * (1 - p_pooled)
        denominator = (p2 - p1) ** 2
        
        n = math.ceil(numerator / denominator)
        return n
    
    @staticmethod
    def calculate_for_means(mean1: float, mean2: float, 
                           std: float,
                           alpha: float = 0.05,
                           power: float = 0.80) -> int:
        """
        Calculate sample size for continuous metrics (NDCG, MRR).
        
        Args:
            mean1: Baseline mean
            mean2: Expected mean in treatment
            std: Standard deviation
            alpha: Significance level
            power: Statistical power
        
        Returns:
            Required sample size per variant
        """
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        effect_size = abs(mean2 - mean1) / std
        
        n = math.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2)
        return n
    
    @staticmethod
    def calculate_test_duration(sample_size: int, 
                               daily_users: int,
                               allocation_percent: float = 0.5) -> int:
        """
        Calculate test duration in days.
        
        Args:
            sample_size: Required sample size per variant
            daily_users: Average daily users
            allocation_percent: Percentage allocated to each variant (0.5 = 50%)
        
        Returns:
            Duration in days
        """
        daily_per_variant = daily_users * allocation_percent
        days = math.ceil(sample_size / daily_per_variant)
        return days

# Usage
calculator = SampleSizeCalculator()

# Example 1: CTR improvement (25% → 27%)
sample_size_ctr = calculator.calculate_for_proportions(
    p1=0.25,  # Current 25% CTR
    p2=0.27,  # Target 27% CTR (8% relative increase)
    alpha=0.05,
    power=0.80
)

print("Sample Size for CTR Test:")
print(f"  Baseline CTR: 25%")
print(f"  Target CTR: 27%")
print(f"  Sample size per variant: {sample_size_ctr:,}")
print(f"  Total sample size: {sample_size_ctr * 2:,}")

# Example 2: NDCG improvement
sample_size_ndcg = calculator.calculate_for_means(
    mean1=0.75,  # Current NDCG@10
    mean2=0.78,  # Target NDCG@10 (4% relative increase)
    std=0.15,    # Estimated standard deviation
    alpha=0.05,
    power=0.80
)

print(f"\nSample Size for NDCG Test:")
print(f"  Baseline NDCG@10: 0.75")
print(f"  Target NDCG@10: 0.78")
print(f"  Sample size per variant: {sample_size_ndcg:,}")

# Example 3: Test duration
duration = calculator.calculate_test_duration(
    sample_size=sample_size_ctr,
    daily_users=10000,
    allocation_percent=0.5
)

print(f"\nTest Duration:")
print(f"  Daily users: 10,000")
print(f"  Allocation: 50% per variant")
print(f"  Duration: {duration} days")
```

### Randomization Strategy

```python
import hashlib
import random

class Randomizer:
    """Handle user randomization for A/B tests."""
    
    def __init__(self, experiment_id: str, variants: list, 
                 allocations: list = None):
        """
        Initialize randomizer.
        
        Args:
            experiment_id: Unique experiment identifier
            variants: List of variant names (e.g., ['control', 'treatment'])
            allocations: List of allocation percentages (e.g., [0.5, 0.5])
                        If None, defaults to equal allocation
        """
        self.experiment_id = experiment_id
        self.variants = variants
        
        if allocations is None:
            self.allocations = [1.0 / len(variants)] * len(variants)
        else:
            if abs(sum(allocations) - 1.0) > 0.001:
                raise ValueError("Allocations must sum to 1.0")
            self.allocations = allocations
        
        # Calculate cumulative allocations for bucket assignment
        self.cumulative_allocations = []
        cumsum = 0
        for alloc in self.allocations:
            cumsum += alloc
            self.cumulative_allocations.append(cumsum)
    
    def assign_variant(self, user_id: str) -> str:
        """
        Assign user to variant using consistent hashing.
        
        Same user_id always gets same variant for this experiment.
        """
        # Create hash of user_id + experiment_id
        hash_input = f"{user_id}:{self.experiment_id}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        
        # Convert to number between 0 and 1
        hash_int = int(hash_value, 16)
        bucket = (hash_int % 10000) / 10000.0
        
        # Assign to variant based on bucket
        for i, cumsum in enumerate(self.cumulative_allocations):
            if bucket < cumsum:
                return self.variants[i]
        
        return self.variants[-1]
    
    def get_allocation_stats(self, user_ids: list) -> dict:
        """
        Get actual allocation statistics for a list of users.
        """
        variant_counts = {v: 0 for v in self.variants}
        
        for user_id in user_ids:
            variant = self.assign_variant(user_id)
            variant_counts[variant] += 1
        
        total = len(user_ids)
        return {
            'counts': variant_counts,
            'percentages': {
                v: count / total * 100 
                for v, count in variant_counts.items()
            },
            'expected': {
                v: self.allocations[i] * 100
                for i, v in enumerate(self.variants)
            }
        }

# Usage
randomizer = Randomizer(
    experiment_id="scoring_profile_test_001",
    variants=['control', 'treatment'],
    allocations=[0.5, 0.5]
)

# Assign users
user1_variant = randomizer.assign_variant("user123")
user2_variant = randomizer.assign_variant("user456")

print(f"User assignments:")
print(f"  user123 → {user1_variant}")
print(f"  user456 → {user2_variant}")

# Verify consistency
print(f"\nConsistency check (same user, same variant):")
print(f"  user123 → {randomizer.assign_variant('user123')}")
print(f"  user123 → {randomizer.assign_variant('user123')}")

# Check allocation balance
test_users = [f"user{i}" for i in range(10000)]
stats = randomizer.get_allocation_stats(test_users)

print(f"\nAllocation Statistics (10,000 users):")
for variant in randomizer.variants:
    expected = stats['expected'][variant]
    actual = stats['percentages'][variant]
    print(f"  {variant}: {actual:.2f}% (expected {expected:.0f}%)")
```

---

## Implementation

### Experiment Tracking

```python
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from datetime import datetime
import json

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    experiment_id: str
    name: str
    hypothesis: str
    variants: list
    allocations: list
    metrics: list
    start_date: str
    end_date: Optional[str]
    status: str  # draft, running, completed, stopped
    metadata: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ExperimentEvent:
    """Single experiment event (query, click, etc.)."""
    event_id: str
    experiment_id: str
    user_id: str
    variant: str
    timestamp: str
    event_type: str  # query, click, conversion, etc.
    metrics: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)

class ExperimentTracker:
    """Track A/B test experiments and events."""
    
    def __init__(self, storage_path: str = "experiments"):
        self.storage_path = storage_path
        import os
        os.makedirs(storage_path, exist_ok=True)
    
    def create_experiment(self, config: ExperimentConfig):
        """Create new experiment."""
        # Save configuration
        config_path = f"{self.storage_path}/{config.experiment_id}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        print(f"Created experiment: {config.name}")
        print(f"  ID: {config.experiment_id}")
        print(f"  Variants: {config.variants}")
        print(f"  Allocations: {config.allocations}")
        
        return config
    
    def log_event(self, event: ExperimentEvent):
        """Log experiment event."""
        events_path = f"{self.storage_path}/{event.experiment_id}_events.jsonl"
        
        with open(events_path, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')
    
    def get_experiment_config(self, experiment_id: str) -> ExperimentConfig:
        """Load experiment configuration."""
        config_path = f"{self.storage_path}/{experiment_id}_config.json"
        
        with open(config_path, 'r') as f:
            data = json.load(f)
            return ExperimentConfig(**data)
    
    def get_events(self, experiment_id: str) -> list:
        """Load all events for an experiment."""
        events_path = f"{self.storage_path}/{experiment_id}_events.jsonl"
        events = []
        
        try:
            with open(events_path, 'r') as f:
                for line in f:
                    event_data = json.loads(line)
                    events.append(ExperimentEvent(**event_data))
        except FileNotFoundError:
            pass
        
        return events

# Usage
tracker = ExperimentTracker()

# Create experiment
experiment = ExperimentConfig(
    experiment_id="exp_semantic_001",
    name="Semantic Search vs BM25",
    hypothesis="Semantic search improves NDCG@10 by 5%",
    variants=["control_bm25", "treatment_semantic"],
    allocations=[0.5, 0.5],
    metrics=["ndcg@10", "mrr", "ctr"],
    start_date=datetime.now().isoformat(),
    end_date=None,
    status="running",
    metadata={
        "created_by": "data_team",
        "min_sample_size": 5000
    }
)

tracker.create_experiment(experiment)

# Log events
event1 = ExperimentEvent(
    event_id="evt_001",
    experiment_id="exp_semantic_001",
    user_id="user123",
    variant="control_bm25",
    timestamp=datetime.now().isoformat(),
    event_type="query",
    metrics={
        "ndcg@10": 0.78,
        "mrr": 0.85,
        "query_text": "machine learning tutorial"
    }
)

tracker.log_event(event1)
print(f"\nLogged event: {event1.event_type} for {event1.variant}")
```

### Search Variant Execution

```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

class ABTestSearchExecutor:
    """Execute search queries with A/B variant logic."""
    
    def __init__(self, search_endpoint: str, index_name: str, 
                 search_key: str, randomizer: Randomizer):
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )
        self.randomizer = randomizer
    
    def execute_search(self, user_id: str, query_text: str, 
                      variant_configs: dict, top: int = 10) -> dict:
        """
        Execute search with appropriate variant configuration.
        
        Args:
            user_id: User identifier for variant assignment
            query_text: Search query
            variant_configs: Dict mapping variant names to search parameters
            top: Number of results
        
        Returns:
            Search results with variant information
        """
        # Assign variant
        variant = self.randomizer.assign_variant(user_id)
        
        # Get variant configuration
        config = variant_configs.get(variant, {})
        
        # Execute search with variant parameters
        results = self.search_client.search(
            search_text=query_text,
            top=top,
            **config
        )
        
        # Collect results
        documents = []
        for result in results:
            documents.append({
                'id': result.get('id'),
                'score': result.get('@search.score'),
                'title': result.get('title')
            })
        
        return {
            'variant': variant,
            'query': query_text,
            'results': documents,
            'result_count': len(documents)
        }

# Usage
import os

# Setup A/B test variants
variant_configs = {
    'control_bm25': {
        # Standard BM25 search
        'scoring_profile': None,
        'query_type': 'simple'
    },
    'treatment_semantic': {
        # Semantic search
        'query_type': 'semantic',
        'semantic_configuration_name': 'my-semantic-config'
    }
}

executor = ABTestSearchExecutor(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    index_name="products",
    search_key=os.getenv("SEARCH_API_KEY"),
    randomizer=randomizer
)

# Execute search for user
# result = executor.execute_search(
#     user_id="user123",
#     query_text="laptop",
#     variant_configs=variant_configs,
#     top=10
# )
# 
# print(f"Variant: {result['variant']}")
# print(f"Results: {result['result_count']}")
```

---

## Statistical Analysis

### Hypothesis Testing

```python
from scipy import stats
import numpy as np

class HypothesisTester:
    """Perform statistical hypothesis tests."""
    
    @staticmethod
    def two_sample_t_test(control_values: list, treatment_values: list,
                         alpha: float = 0.05) -> dict:
        """
        Perform two-sample t-test for continuous metrics.
        
        Returns p-value and confidence interval.
        """
        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(control_values, treatment_values)
        
        # Calculate means and standard deviations
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(control_values) + np.var(treatment_values)) / 2
        )
        cohens_d = (treatment_mean - control_mean) / pooled_std
        
        # Confidence interval for difference
        diff = treatment_mean - control_mean
        se_diff = np.sqrt(
            np.var(control_values) / len(control_values) +
            np.var(treatment_values) / len(treatment_values)
        )
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'difference': diff,
            'relative_change': (diff / control_mean * 100) if control_mean != 0 else 0,
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cohens_d': cohens_d,
            'confidence_interval': (ci_lower, ci_upper)
        }
    
    @staticmethod
    def two_proportion_z_test(control_successes: int, control_total: int,
                              treatment_successes: int, treatment_total: int,
                              alpha: float = 0.05) -> dict:
        """
        Perform two-proportion z-test for binary metrics (CTR, conversion).
        """
        # Calculate proportions
        p1 = control_successes / control_total
        p2 = treatment_successes / treatment_total
        
        # Pooled proportion
        p_pooled = (control_successes + treatment_successes) / (control_total + treatment_total)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
        
        # Z-statistic
        z_stat = (p2 - p1) / se
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Confidence interval
        se_diff = np.sqrt(p1 * (1-p1) / control_total + p2 * (1-p2) / treatment_total)
        diff = p2 - p1
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        return {
            'control_rate': p1,
            'treatment_rate': p2,
            'difference': diff,
            'relative_change': (diff / p1 * 100) if p1 != 0 else 0,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'confidence_interval': (ci_lower, ci_upper)
        }

# Usage
tester = HypothesisTester()

# Example 1: Continuous metric (NDCG@10)
control_ndcg = [0.75, 0.78, 0.72, 0.80, 0.76, 0.74, 0.79, 0.77, 0.73, 0.78]
treatment_ndcg = [0.78, 0.81, 0.79, 0.83, 0.80, 0.77, 0.82, 0.79, 0.80, 0.81]

t_test_result = tester.two_sample_t_test(control_ndcg, treatment_ndcg)

print("T-Test Results (NDCG@10):")
print(f"  Control mean: {t_test_result['control_mean']:.4f}")
print(f"  Treatment mean: {t_test_result['treatment_mean']:.4f}")
print(f"  Difference: {t_test_result['difference']:.4f} ({t_test_result['relative_change']:.2f}%)")
print(f"  P-value: {t_test_result['p_value']:.4f}")
print(f"  Significant: {t_test_result['significant']}")
print(f"  95% CI: ({t_test_result['confidence_interval'][0]:.4f}, {t_test_result['confidence_interval'][1]:.4f})")

# Example 2: Proportion metric (CTR)
z_test_result = tester.two_proportion_z_test(
    control_successes=250,    # 250 clicks
    control_total=1000,       # 1000 searches
    treatment_successes=285,  # 285 clicks
    treatment_total=1000,     # 1000 searches
    alpha=0.05
)

print("\n\nZ-Test Results (CTR):")
print(f"  Control CTR: {z_test_result['control_rate']:.2%}")
print(f"  Treatment CTR: {z_test_result['treatment_rate']:.2%}")
print(f"  Difference: {z_test_result['difference']:.2%} ({z_test_result['relative_change']:.2f}%)")
print(f"  P-value: {z_test_result['p_value']:.4f}")
print(f"  Significant: {z_test_result['significant']}")
```

---

## Azure Integration

### Cosmos DB Storage

```python
from azure.cosmos import CosmosClient, PartitionKey

class CosmosDBExperimentStore:
    """Store experiment data in Cosmos DB."""
    
    def __init__(self, endpoint: str, key: str, 
                 database_name: str = "experiments",
                 container_name: str = "ab_tests"):
        self.client = CosmosClient(endpoint, key)
        
        # Create database if not exists
        self.database = self.client.create_database_if_not_exists(database_name)
        
        # Create container if not exists
        self.container = self.database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/experiment_id")
        )
    
    def store_event(self, event: ExperimentEvent):
        """Store experiment event in Cosmos DB."""
        self.container.upsert_item(event.to_dict())
    
    def query_events(self, experiment_id: str, variant: str = None) -> list:
        """Query events for an experiment."""
        query = f"SELECT * FROM c WHERE c.experiment_id = '{experiment_id}'"
        
        if variant:
            query += f" AND c.variant = '{variant}'"
        
        items = list(self.container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        return items
    
    def get_variant_metrics(self, experiment_id: str) -> dict:
        """
        Aggregate metrics by variant.
        """
        query = f"""
        SELECT 
            c.variant,
            COUNT(1) as event_count,
            AVG(c.metrics.ndcg_at_10) as avg_ndcg,
            AVG(c.metrics.mrr) as avg_mrr
        FROM c 
        WHERE c.experiment_id = '{experiment_id}'
        GROUP BY c.variant
        """
        
        results = list(self.container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        return {r['variant']: r for r in results}

# Usage
# cosmos_store = CosmosDBExperimentStore(
#     endpoint=os.getenv("COSMOS_ENDPOINT"),
#     key=os.getenv("COSMOS_KEY")
# )
# 
# # Store event
# cosmos_store.store_event(event1)
# 
# # Query events
# events = cosmos_store.query_events("exp_semantic_001", variant="control_bm25")
# print(f"Events for control: {len(events)}")
# 
# # Get aggregated metrics
# metrics = cosmos_store.get_variant_metrics("exp_semantic_001")
# for variant, stats in metrics.items():
#     print(f"{variant}: {stats['event_count']} events, avg NDCG = {stats['avg_ndcg']:.4f}")
```

---

## Results Tracking

### Results Dashboard

```python
class ExperimentDashboard:
    """Generate experiment results dashboard."""
    
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
    
    def generate_summary(self, experiment_id: str) -> dict:
        """Generate experiment summary."""
        config = self.tracker.get_experiment_config(experiment_id)
        events = self.tracker.get_events(experiment_id)
        
        # Group by variant
        variant_events = {}
        for event in events:
            if event.variant not in variant_events:
                variant_events[event.variant] = []
            variant_events[event.variant].append(event)
        
        # Calculate metrics per variant
        variant_stats = {}
        for variant, events_list in variant_events.items():
            ndcg_values = [e.metrics.get('ndcg@10') for e in events_list if 'ndcg@10' in e.metrics]
            mrr_values = [e.metrics.get('mrr') for e in events_list if 'mrr' in e.metrics]
            
            variant_stats[variant] = {
                'event_count': len(events_list),
                'avg_ndcg': np.mean(ndcg_values) if ndcg_values else 0,
                'avg_mrr': np.mean(mrr_values) if mrr_values else 0
            }
        
        return {
            'experiment': config.name,
            'status': config.status,
            'total_events': len(events),
            'variants': variant_stats
        }
    
    def print_summary(self, experiment_id: str):
        """Print formatted experiment summary."""
        summary = self.generate_summary(experiment_id)
        
        print(f"Experiment: {summary['experiment']}")
        print(f"Status: {summary['status']}")
        print(f"Total Events: {summary['total_events']}")
        print(f"\nVariant Results:")
        
        for variant, stats in summary['variants'].items():
            print(f"\n  {variant}:")
            print(f"    Events: {stats['event_count']}")
            print(f"    Avg NDCG@10: {stats['avg_ndcg']:.4f}")
            print(f"    Avg MRR: {stats['avg_mrr']:.4f}")

# Usage
# dashboard = ExperimentDashboard(tracker)
# dashboard.print_summary("exp_semantic_001")
```

---

## Analysis Tools

### Complete Analysis Pipeline

```python
class ExperimentAnalyzer:
    """Complete A/B test analysis pipeline."""
    
    def __init__(self, experiment_id: str, tracker: ExperimentTracker):
        self.experiment_id = experiment_id
        self.tracker = tracker
        self.config = tracker.get_experiment_config(experiment_id)
        self.events = tracker.get_events(experiment_id)
    
    def run_analysis(self) -> dict:
        """Run complete analysis."""
        results = {
            'experiment_id': self.experiment_id,
            'config': self.config.to_dict(),
            'sample_sizes': self._calculate_sample_sizes(),
            'metrics': self._analyze_metrics(),
            'statistical_tests': self._run_statistical_tests(),
            'recommendation': self._make_recommendation()
        }
        
        return results
    
    def _calculate_sample_sizes(self) -> dict:
        """Calculate actual sample sizes per variant."""
        variant_counts = {}
        for event in self.events:
            variant = event.variant
            variant_counts[variant] = variant_counts.get(variant, 0) + 1
        
        return variant_counts
    
    def _analyze_metrics(self) -> dict:
        """Analyze metrics by variant."""
        variant_metrics = {}
        
        for variant in self.config.variants:
            variant_events = [e for e in self.events if e.variant == variant]
            
            metrics = {}
            for metric_name in self.config.metrics:
                values = [
                    e.metrics.get(metric_name)
                    for e in variant_events
                    if metric_name in e.metrics and e.metrics[metric_name] is not None
                ]
                
                if values:
                    metrics[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            variant_metrics[variant] = metrics
        
        return variant_metrics
    
    def _run_statistical_tests(self) -> dict:
        """Run statistical tests comparing variants."""
        if len(self.config.variants) != 2:
            return {'error': 'Statistical tests require exactly 2 variants'}
        
        control = self.config.variants[0]
        treatment = self.config.variants[1]
        
        control_events = [e for e in self.events if e.variant == control]
        treatment_events = [e for e in self.events if e.variant == treatment]
        
        test_results = {}
        tester = HypothesisTester()
        
        for metric_name in self.config.metrics:
            control_values = [
                e.metrics.get(metric_name)
                for e in control_events
                if metric_name in e.metrics and e.metrics[metric_name] is not None
            ]
            
            treatment_values = [
                e.metrics.get(metric_name)
                for e in treatment_events
                if metric_name in e.metrics and e.metrics[metric_name] is not None
            ]
            
            if control_values and treatment_values:
                result = tester.two_sample_t_test(control_values, treatment_values)
                test_results[metric_name] = result
        
        return test_results
    
    def _make_recommendation(self) -> dict:
        """Make recommendation based on results."""
        tests = self._run_statistical_tests()
        
        if not tests:
            return {'decision': 'insufficient_data'}
        
        # Check if primary metric is significant
        primary_metric = self.config.metrics[0] if self.config.metrics else None
        
        if primary_metric and primary_metric in tests:
            result = tests[primary_metric]
            
            if result['significant'] and result['difference'] > 0:
                return {
                    'decision': 'ship_treatment',
                    'reason': f"{primary_metric} significantly improved",
                    'improvement': f"{result['relative_change']:.2f}%"
                }
            elif result['significant'] and result['difference'] < 0:
                return {
                    'decision': 'keep_control',
                    'reason': f"{primary_metric} significantly degraded",
                    'degradation': f"{result['relative_change']:.2f}%"
                }
            else:
                return {
                    'decision': 'no_significant_difference',
                    'reason': f"No significant change in {primary_metric}"
                }
        
        return {'decision': 'manual_review_required'}
    
    def print_report(self):
        """Print formatted analysis report."""
        analysis = self.run_analysis()
        
        print("=" * 60)
        print(f"A/B Test Analysis Report")
        print("=" * 60)
        print(f"\nExperiment: {self.config.name}")
        print(f"Hypothesis: {self.config.hypothesis}")
        print(f"Status: {self.config.status}")
        
        print(f"\nSample Sizes:")
        for variant, count in analysis['sample_sizes'].items():
            print(f"  {variant}: {count:,}")
        
        print(f"\nMetrics by Variant:")
        for variant, metrics in analysis['metrics'].items():
            print(f"\n  {variant}:")
            for metric_name, stats in metrics.items():
                print(f"    {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print(f"\nStatistical Tests:")
        for metric_name, result in analysis['statistical_tests'].items():
            print(f"\n  {metric_name}:")
            print(f"    Difference: {result['difference']:.4f} ({result['relative_change']:.2f}%)")
            print(f"    P-value: {result['p_value']:.4f}")
            print(f"    Significant: {'✓ YES' if result['significant'] else '✗ NO'}")
            print(f"    95% CI: ({result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f})")
        
        print(f"\nRecommendation:")
        rec = analysis['recommendation']
        print(f"  Decision: {rec['decision'].replace('_', ' ').title()}")
        if 'reason' in rec:
            print(f"  Reason: {rec['reason']}")

# Usage
# analyzer = ExperimentAnalyzer("exp_semantic_001", tracker)
# analyzer.print_report()
```

---

## Best Practices

### ✅ Do's
1. **Run A/A test first** to validate instrumentation
2. **Calculate sample size** before starting
3. **Use consistent hashing** for user assignment
4. **Monitor for** imbalanced allocation
5. **Wait for significance** before making decisions
6. **Document everything** (hypothesis, config, results)
7. **Segment analysis** by user type, query type
8. **Check for novelty effects** (first week bias)
9. **Run for full week** to capture weekly patterns
10. **Validate results** before shipping

### ❌ Don'ts
1. **Don't** peek early and stop tests based on p-values
2. **Don't** run multiple tests on same traffic simultaneously
3. **Don't** change allocation mid-experiment
4. **Don't** ignore confidence intervals (look beyond p-value)
5. **Don't** test too many variants (reduces power)
6. **Don't** run experiments indefinitely
7. **Don't** forget to account for multiple testing
8. **Don't** ignore segment differences
9. **Don't** ship without validation
10. **Don't** forget to turn off experiment code after

---

## Next Steps

- **[Load Testing](./18-load-testing.md)** - Performance testing
- **[Dataset Preparation](./16-dataset-preparation.md)** - Creating test data
- **[Monitoring & Alerting](./23-monitoring-alerting.md)** - Production monitoring

---

*See also: [Scoring Profiles](./14-scoring-profiles.md) | [Index Management](./15-index-management.md)*
