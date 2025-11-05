# Load Testing

Comprehensive guide to performance and scalability testing for Azure AI Search.

## 📋 Table of Contents
- [Overview](#overview)
- [Azure Load Testing](#azure-load-testing)
- [JMeter Configuration](#jmeter-configuration)
- [Performance Benchmarks](#performance-benchmarks)
- [Scalability Testing](#scalability-testing)
- [Analysis & Optimization](#analysis--optimization)
- [Best Practices](#best-practices)

---

## Overview

### Load Testing Fundamentals

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class LoadTestType(Enum):
    """Types of load tests."""
    BASELINE = "baseline"          # Normal load
    STRESS = "stress"              # Peak load (2-3x normal)
    SPIKE = "spike"                # Sudden burst (10x normal)
    ENDURANCE = "endurance"        # Sustained load (24hr+)
    SCALABILITY = "scalability"    # Incremental increase

@dataclass
class LoadTestConfig:
    """Load test configuration."""
    test_name: str
    test_type: LoadTestType
    virtual_users: int             # Concurrent users
    ramp_up_time: int             # Seconds to reach full load
    duration: int                  # Test duration in seconds
    target_qps: float             # Queries per second target
    query_distribution: Dict[str, float]  # Query type percentages
    
    def to_dict(self):
        return {
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'virtual_users': self.virtual_users,
            'ramp_up_time': self.ramp_up_time,
            'duration': self.duration,
            'target_qps': self.target_qps,
            'query_distribution': self.query_distribution
        }

class LoadTestScenarios:
    """Common load test scenarios."""
    
    @staticmethod
    def baseline_test() -> LoadTestConfig:
        """Normal production load."""
        return LoadTestConfig(
            test_name="baseline_normal_load",
            test_type=LoadTestType.BASELINE,
            virtual_users=100,
            ramp_up_time=60,
            duration=600,  # 10 minutes
            target_qps=50.0,
            query_distribution={
                'simple_search': 0.60,
                'faceted_search': 0.25,
                'autocomplete': 0.15
            }
        )
    
    @staticmethod
    def stress_test() -> LoadTestConfig:
        """Peak load (Black Friday, product launches)."""
        return LoadTestConfig(
            test_name="stress_peak_load",
            test_type=LoadTestType.STRESS,
            virtual_users=300,
            ramp_up_time=120,
            duration=1800,  # 30 minutes
            target_qps=150.0,
            query_distribution={
                'simple_search': 0.55,
                'faceted_search': 0.30,
                'autocomplete': 0.15
            }
        )
    
    @staticmethod
    def spike_test() -> LoadTestConfig:
        """Sudden traffic burst."""
        return LoadTestConfig(
            test_name="spike_burst",
            test_type=LoadTestType.SPIKE,
            virtual_users=500,
            ramp_up_time=10,  # Quick ramp
            duration=300,  # 5 minutes
            target_qps=250.0,
            query_distribution={
                'simple_search': 0.70,
                'faceted_search': 0.20,
                'autocomplete': 0.10
            }
        )
    
    @staticmethod
    def endurance_test() -> LoadTestConfig:
        """Long-running stability test."""
        return LoadTestConfig(
            test_name="endurance_24hr",
            test_type=LoadTestType.ENDURANCE,
            virtual_users=150,
            ramp_up_time=300,
            duration=86400,  # 24 hours
            target_qps=75.0,
            query_distribution={
                'simple_search': 0.60,
                'faceted_search': 0.25,
                'autocomplete': 0.15
            }
        )

# Usage
scenarios = LoadTestScenarios()

baseline = scenarios.baseline_test()
print(f"Baseline Test Configuration:")
print(f"  Virtual Users: {baseline.virtual_users}")
print(f"  Target QPS: {baseline.target_qps}")
print(f"  Duration: {baseline.duration}s ({baseline.duration/60:.0f} min)")
print(f"  Query Distribution:")
for query_type, percentage in baseline.query_distribution.items():
    print(f"    {query_type}: {percentage:.0%}")

stress = scenarios.stress_test()
print(f"\nStress Test Configuration:")
print(f"  Virtual Users: {stress.virtual_users} ({stress.virtual_users/baseline.virtual_users:.1f}x baseline)")
print(f"  Target QPS: {stress.target_qps} ({stress.target_qps/baseline.target_qps:.1f}x baseline)")
```

---

## Azure Load Testing

### Service Setup

```python
from azure.identity import DefaultAzureCredential
from azure.mgmt.loadtesting import LoadTestMgmtClient
import os

class AzureLoadTestSetup:
    """Setup Azure Load Testing service."""
    
    def __init__(self, subscription_id: str, resource_group: str):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.credential = DefaultAzureCredential()
        self.client = LoadTestMgmtClient(self.credential, subscription_id)
    
    def create_load_test_resource(self, name: str, location: str) -> dict:
        """
        Create Azure Load Testing resource.
        
        Args:
            name: Resource name
            location: Azure region (e.g., 'eastus')
        
        Returns:
            Resource details
        """
        parameters = {
            'location': location,
            'tags': {
                'purpose': 'search_performance_testing',
                'environment': 'dev'
            }
        }
        
        poller = self.client.load_tests.begin_create_or_update(
            resource_group_name=self.resource_group,
            load_test_name=name,
            load_test_resource=parameters
        )
        
        result = poller.result()
        
        return {
            'name': result.name,
            'location': result.location,
            'data_plane_uri': result.data_plane_uri,
            'provisioning_state': result.provisioning_state
        }
    
    def configure_vnet_injection(self, test_resource_name: str,
                                 vnet_id: str, subnet_id: str):
        """
        Configure VNet injection for private endpoint testing.
        
        Required when testing private search endpoints.
        """
        # This would typically be done via ARM template or Azure Portal
        # as it requires network configuration during resource creation
        
        print(f"VNet Configuration for {test_resource_name}:")
        print(f"  VNet ID: {vnet_id}")
        print(f"  Subnet ID: {subnet_id}")
        print(f"\nNote: VNet injection must be configured during resource creation")
        print(f"Use ARM template or Azure Portal for VNet-injected load test resources")

# Usage
# setup = AzureLoadTestSetup(
#     subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
#     resource_group="search-perf-rg"
# )
# 
# # Create load test resource
# resource = setup.create_load_test_resource(
#     name="search-loadtest",
#     location="eastus"
# )
# 
# print(f"Created Load Test Resource:")
# print(f"  Name: {resource['name']}")
# print(f"  Data Plane URI: {resource['data_plane_uri']}")
```

### Test Upload & Execution

```python
from azure.developer.loadtesting import LoadTestRunClient
from azure.core.credentials import AzureKeyCredential
import time

class LoadTestExecutor:
    """Execute load tests using Azure Load Testing."""
    
    def __init__(self, endpoint: str, credential):
        self.client = LoadTestRunClient(endpoint, credential)
    
    def upload_test_plan(self, test_id: str, jmx_file_path: str):
        """
        Upload JMeter test plan (.jmx file).
        """
        with open(jmx_file_path, 'rb') as f:
            self.client.upload_test_file(
                test_id=test_id,
                file_name=os.path.basename(jmx_file_path),
                body=f
            )
        
        print(f"Uploaded test plan: {jmx_file_path}")
    
    def upload_test_data(self, test_id: str, data_file_path: str):
        """
        Upload test data file (e.g., CSV with queries).
        """
        with open(data_file_path, 'rb') as f:
            self.client.upload_test_file(
                test_id=test_id,
                file_name=os.path.basename(data_file_path),
                body=f
            )
        
        print(f"Uploaded test data: {data_file_path}")
    
    def create_test(self, test_id: str, config: LoadTestConfig):
        """
        Create load test configuration.
        """
        test_config = {
            'testId': test_id,
            'displayName': config.test_name,
            'description': f'{config.test_type.value} test',
            'loadTestConfig': {
                'engineInstances': self._calculate_engine_instances(config.virtual_users),
                'splitAllCSVs': False
            },
            'passFailCriteria': {
                'passFailMetrics': {
                    'avg_response_time': {
                        'clientmetric': 'response_time_ms',
                        'aggregate': 'avg',
                        'condition': '<',
                        'value': 500.0
                    },
                    'error_rate': {
                        'clientmetric': 'error',
                        'aggregate': 'percentage',
                        'condition': '<',
                        'value': 5.0
                    }
                }
            }
        }
        
        result = self.client.create_or_update_test(test_id, test_config)
        print(f"Created test: {test_id}")
        return result
    
    def _calculate_engine_instances(self, virtual_users: int) -> int:
        """
        Calculate required engine instances.
        Each instance supports ~250 virtual users.
        """
        return max(1, (virtual_users + 249) // 250)
    
    def run_test(self, test_id: str, run_id: str = None) -> dict:
        """
        Execute load test.
        """
        if run_id is None:
            run_id = f"{test_id}_run_{int(time.time())}"
        
        run_config = {
            'testId': test_id,
            'displayName': f'Run {run_id}'
        }
        
        # Start test run
        result = self.client.begin_test_run(run_id, run_config)
        
        print(f"Started test run: {run_id}")
        print(f"Status: {result.status}")
        
        return {
            'run_id': run_id,
            'test_id': test_id,
            'status': result.status,
            'start_time': result.start_date_time
        }
    
    def monitor_test_run(self, run_id: str, poll_interval: int = 30):
        """
        Monitor test run progress.
        """
        print(f"Monitoring test run: {run_id}")
        
        while True:
            run = self.client.get_test_run(run_id)
            status = run.status
            
            print(f"Status: {status} - {run.test_run_statistics}")
            
            if status in ['DONE', 'FAILED', 'CANCELLED']:
                break
            
            time.sleep(poll_interval)
        
        return run
    
    def get_test_results(self, run_id: str) -> dict:
        """
        Get test run results.
        """
        run = self.client.get_test_run(run_id)
        
        return {
            'run_id': run_id,
            'status': run.status,
            'start_time': run.start_date_time,
            'end_time': run.end_date_time,
            'duration': run.duration,
            'virtual_users': run.virtual_users,
            'statistics': run.test_run_statistics,
            'results_url': run.test_result_url
        }

# Usage
# executor = LoadTestExecutor(
#     endpoint="https://search-loadtest.westus2.loadtest.azure.com",
#     credential=DefaultAzureCredential()
# )
# 
# # Upload files
# executor.upload_test_plan("search_baseline_test", "search_test.jmx")
# executor.upload_test_data("search_baseline_test", "search_queries.csv")
# 
# # Create and run test
# baseline_config = LoadTestScenarios.baseline_test()
# executor.create_test("search_baseline_test", baseline_config)
# 
# run = executor.run_test("search_baseline_test")
# results = executor.monitor_test_run(run['run_id'])
```

---

## JMeter Configuration

### Test Plan Creation

```python
class JMeterTestPlanGenerator:
    """Generate JMeter test plans for search testing."""
    
    @staticmethod
    def generate_basic_search_test(
        search_endpoint: str,
        index_name: str,
        api_key: str,
        queries_csv: str,
        thread_count: int,
        ramp_up: int,
        duration: int
    ) -> str:
        """
        Generate JMeter .jmx file for basic search testing.
        
        Returns XML content for .jmx file.
        """
        jmx_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2" properties="5.0" jmeter="5.5">
  <hashTree>
    <TestPlan guiclass="TestPlanGui" testclass="TestPlan" testname="Azure Search Performance Test">
      <elementProp name="TestPlan.user_defined_variables" elementType="Arguments">
        <collectionProp name="Arguments.arguments">
          <elementProp name="SEARCH_ENDPOINT" elementType="Argument">
            <stringProp name="Argument.name">SEARCH_ENDPOINT</stringProp>
            <stringProp name="Argument.value">{search_endpoint}</stringProp>
          </elementProp>
          <elementProp name="INDEX_NAME" elementType="Argument">
            <stringProp name="Argument.name">INDEX_NAME</stringProp>
            <stringProp name="Argument.value">{index_name}</stringProp>
          </elementProp>
          <elementProp name="API_KEY" elementType="Argument">
            <stringProp name="Argument.name">API_KEY</stringProp>
            <stringProp name="Argument.value">{api_key}</stringProp>
          </elementProp>
        </collectionProp>
      </elementProp>
    </TestPlan>
    <hashTree>
      <!-- Thread Group -->
      <ThreadGroup guiclass="ThreadGroupGui" testclass="ThreadGroup" testname="Search Users">
        <intProp name="ThreadGroup.num_threads">{thread_count}</intProp>
        <intProp name="ThreadGroup.ramp_time">{ramp_up}</intProp>
        <longProp name="ThreadGroup.duration">{duration}</longProp>
        <boolProp name="ThreadGroup.scheduler">true</boolProp>
        <stringProp name="ThreadGroup.on_sample_error">continue</stringProp>
      </ThreadGroup>
      <hashTree>
        <!-- CSV Data Set -->
        <CSVDataSet guiclass="TestBeanGUI" testclass="CSVDataSet" testname="Query Data">
          <stringProp name="filename">{queries_csv}</stringProp>
          <stringProp name="fileEncoding">UTF-8</stringProp>
          <stringProp name="variableNames">query_text,top_k</stringProp>
          <boolProp name="recycle">true</boolProp>
          <boolProp name="stopThread">false</boolProp>
          <stringProp name="shareMode">shareMode.all</stringProp>
        </CSVDataSet>
        <hashTree/>
        
        <!-- HTTP Request -->
        <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="Search Request">
          <stringProp name="HTTPSampler.domain">${{SEARCH_ENDPOINT}}</stringProp>
          <stringProp name="HTTPSampler.path">/indexes/${{INDEX_NAME}}/docs</stringProp>
          <stringProp name="HTTPSampler.method">GET</stringProp>
          <boolProp name="HTTPSampler.use_keepalive">true</boolProp>
          <elementProp name="HTTPsampler.Arguments" elementType="Arguments">
            <collectionProp name="Arguments.arguments">
              <elementProp name="search" elementType="HTTPArgument">
                <boolProp name="HTTPArgument.always_encode">true</boolProp>
                <stringProp name="Argument.value">${{query_text}}</stringProp>
                <stringProp name="Argument.metadata">=</stringProp>
                <stringProp name="Argument.name">search</stringProp>
              </elementProp>
              <elementProp name="$top" elementType="HTTPArgument">
                <boolProp name="HTTPArgument.always_encode">false</boolProp>
                <stringProp name="Argument.value">${{top_k}}</stringProp>
                <stringProp name="Argument.metadata">=</stringProp>
                <stringProp name="Argument.name">$top</stringProp>
              </elementProp>
              <elementProp name="api-version" elementType="HTTPArgument">
                <stringProp name="Argument.value">2023-11-01</stringProp>
                <stringProp name="Argument.name">api-version</stringProp>
              </elementProp>
            </collectionProp>
          </elementProp>
          <elementProp name="HTTPSampler.header_manager" elementType="HeaderManager">
            <collectionProp name="HeaderManager.headers">
              <elementProp name="" elementType="Header">
                <stringProp name="Header.name">api-key</stringProp>
                <stringProp name="Header.value">${{API_KEY}}</stringProp>
              </elementProp>
              <elementProp name="" elementType="Header">
                <stringProp name="Header.name">Content-Type</stringProp>
                <stringProp name="Header.value">application/json</stringProp>
              </elementProp>
            </collectionProp>
          </elementProp>
        </HTTPSamplerProxy>
        <hashTree>
          <!-- Response Assertion -->
          <ResponseAssertion guiclass="AssertionGui" testclass="ResponseAssertion" testname="HTTP 200">
            <collectionProp name="Asserion.test_strings">
              <stringProp name="49586">200</stringProp>
            </collectionProp>
            <stringProp name="Assertion.test_field">Assertion.response_code</stringProp>
          </ResponseAssertion>
          <hashTree/>
        </hashTree>
        
        <!-- Listeners -->
        <ResultCollector guiclass="SummaryReport" testclass="ResultCollector" testname="Summary Report"/>
        <hashTree/>
      </hashTree>
    </hashTree>
  </hashTree>
</jmeterTestPlan>"""
        
        return jmx_template
    
    @staticmethod
    def save_test_plan(xml_content: str, output_path: str):
        """Save JMeter test plan to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        print(f"Saved JMeter test plan: {output_path}")

# Usage
generator = JMeterTestPlanGenerator()

jmx_content = generator.generate_basic_search_test(
    search_endpoint="your-search-service.search.windows.net",
    index_name="products",
    api_key="${__GetSecret(api-key)}",  # Reference to secret
    queries_csv="search_queries.csv",
    thread_count=100,
    ramp_up=60,
    duration=600
)

# generator.save_test_plan(jmx_content, "search_load_test.jmx")
```

### Test Data Generation

```python
import csv
import random

class TestDataGenerator:
    """Generate test data for load testing."""
    
    @staticmethod
    def generate_query_dataset(output_file: str, query_count: int = 1000):
        """
        Generate CSV file with search queries.
        """
        # Sample query templates
        query_templates = [
            "laptop",
            "wireless headphones",
            "running shoes",
            "coffee maker",
            "smartphone case",
            "desk lamp",
            "ergonomic keyboard",
            "water bottle",
            "backpack",
            "yoga mat",
            "gaming mouse",
            "bluetooth speaker",
            "fitness tracker",
            "electric kettle",
            "standing desk"
        ]
        
        queries = []
        for _ in range(query_count):
            query_text = random.choice(query_templates)
            top_k = random.choice([10, 20, 50])
            queries.append({
                'query_text': query_text,
                'top_k': top_k
            })
        
        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['query_text', 'top_k'])
            writer.writeheader()
            writer.writerows(queries)
        
        print(f"Generated {query_count} queries in {output_file}")
        return output_file
    
    @staticmethod
    def generate_realistic_query_distribution(
        output_file: str,
        query_count: int = 10000,
        head_queries: list = None,
        tail_queries: list = None
    ):
        """
        Generate queries following realistic distribution (Zipf's law).
        
        - 20% of queries are head queries (80% frequency)
        - 80% of queries are tail queries (20% frequency)
        """
        if head_queries is None:
            head_queries = [
                "laptop", "phone", "headphones", "keyboard", "mouse"
            ]
        
        if tail_queries is None:
            tail_queries = [
                "ergonomic wireless keyboard",
                "noise cancelling headphones bluetooth",
                "gaming laptop RTX",
                "mechanical keyboard RGB",
                "ultrawide monitor 34 inch"
            ]
        
        queries = []
        
        # 80% of queries from head (popular queries)
        head_count = int(query_count * 0.8)
        for _ in range(head_count):
            queries.append({
                'query_text': random.choice(head_queries),
                'top_k': random.choice([10, 20])
            })
        
        # 20% of queries from tail (long-tail queries)
        tail_count = query_count - head_count
        for _ in range(tail_count):
            queries.append({
                'query_text': random.choice(tail_queries),
                'top_k': random.choice([10, 20, 50])
            })
        
        # Shuffle
        random.shuffle(queries)
        
        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['query_text', 'top_k'])
            writer.writeheader()
            writer.writerows(queries)
        
        print(f"Generated {query_count} queries with realistic distribution:")
        print(f"  Head queries: {head_count} ({head_count/query_count:.0%})")
        print(f"  Tail queries: {tail_count} ({tail_count/query_count:.0%})")
        
        return output_file

# Usage
data_gen = TestDataGenerator()

# Basic dataset
# data_gen.generate_query_dataset("search_queries_basic.csv", query_count=1000)

# Realistic distribution
# data_gen.generate_realistic_query_distribution(
#     "search_queries_realistic.csv",
#     query_count=10000
# )
```

---

## Performance Benchmarks

### Azure Search Tier Limits

```python
class AzureSearchTierLimits:
    """Performance limits for Azure AI Search tiers."""
    
    TIER_LIMITS = {
        'Free': {
            'max_qps': 3,
            'max_storage_gb': 0.05,  # 50 MB
            'max_indexes': 3,
            'max_indexers': 3,
            'max_replicas': 1,
            'max_partitions': 1,
            'cost_per_hour': 0.0
        },
        'Basic': {
            'max_qps': 15,  # Per replica-partition
            'max_storage_gb': 2,
            'max_indexes': 15,
            'max_indexers': 15,
            'max_replicas': 3,
            'max_partitions': 1,
            'cost_per_hour': 0.10
        },
        'S1': {
            'max_qps': 15,  # Per replica-partition
            'max_storage_gb': 25,  # Per partition
            'max_indexes': 50,
            'max_indexers': 50,
            'max_replicas': 12,
            'max_partitions': 12,
            'cost_per_hour': 0.34
        },
        'S2': {
            'max_qps': 60,  # Per replica-partition
            'max_storage_gb': 100,  # Per partition
            'max_indexes': 200,
            'max_indexers': 200,
            'max_replicas': 12,
            'max_partitions': 12,
            'cost_per_hour': 1.37
        },
        'S3': {
            'max_qps': 60,  # Per replica-partition
            'max_storage_gb': 200,  # Per partition
            'max_indexes': 200,
            'max_indexers': 200,
            'max_replicas': 12,
            'max_partitions': 12,
            'cost_per_hour': 2.74
        },
        'L1': {
            'max_qps': 60,
            'max_storage_gb': 1000,  # Per partition
            'max_indexes': 10,
            'max_indexers': 10,
            'max_replicas': 12,
            'max_partitions': 12,
            'cost_per_hour': 1.37
        },
        'L2': {
            'max_qps': 60,
            'max_storage_gb': 2000,  # Per partition
            'max_indexes': 10,
            'max_indexers': 10,
            'max_replicas': 12,
            'max_partitions': 12,
            'cost_per_hour': 2.74
        }
    }
    
    @classmethod
    def get_tier_capacity(cls, tier: str, replicas: int, partitions: int) -> dict:
        """
        Calculate total capacity for tier configuration.
        """
        limits = cls.TIER_LIMITS.get(tier)
        if not limits:
            raise ValueError(f"Unknown tier: {tier}")
        
        total_qps = limits['max_qps'] * replicas * partitions
        total_storage = limits['max_storage_gb'] * partitions
        monthly_cost = limits['cost_per_hour'] * 730 * replicas * partitions
        
        return {
            'tier': tier,
            'replicas': replicas,
            'partitions': partitions,
            'total_qps': total_qps,
            'total_storage_gb': total_storage,
            'monthly_cost_usd': round(monthly_cost, 2),
            'per_replica_partition': {
                'qps': limits['max_qps'],
                'storage_gb': limits['max_storage_gb']
            }
        }
    
    @classmethod
    def recommend_tier_for_qps(cls, target_qps: float, 
                               safety_margin: float = 1.5) -> list:
        """
        Recommend tier configurations for target QPS.
        """
        required_qps = target_qps * safety_margin
        recommendations = []
        
        for tier, limits in cls.TIER_LIMITS.items():
            if tier == 'Free':
                continue
            
            qps_per_unit = limits['max_qps']
            max_replicas = limits['max_replicas']
            max_partitions = limits['max_partitions']
            
            # Try different configurations
            for partitions in range(1, max_partitions + 1):
                for replicas in range(1, max_replicas + 1):
                    total_qps = qps_per_unit * replicas * partitions
                    
                    if total_qps >= required_qps:
                        capacity = cls.get_tier_capacity(tier, replicas, partitions)
                        recommendations.append(capacity)
                        break
        
        # Sort by cost
        recommendations.sort(key=lambda x: x['monthly_cost_usd'])
        
        return recommendations

# Usage
limits = AzureSearchTierLimits()

# Get capacity for specific configuration
s2_capacity = limits.get_tier_capacity('S2', replicas=3, partitions=2)
print("S2 with 3 replicas, 2 partitions:")
print(f"  Total QPS: {s2_capacity['total_qps']}")
print(f"  Total Storage: {s2_capacity['total_storage_gb']} GB")
print(f"  Monthly Cost: ${s2_capacity['monthly_cost_usd']}")

# Get recommendations for target QPS
print("\n\nRecommendations for 200 QPS:")
recommendations = limits.recommend_tier_for_qps(200)
for i, rec in enumerate(recommendations[:3], 1):
    print(f"\nOption {i}: {rec['tier']}")
    print(f"  Configuration: {rec['replicas']} replicas × {rec['partitions']} partitions")
    print(f"  Capacity: {rec['total_qps']} QPS")
    print(f"  Cost: ${rec['monthly_cost_usd']}/month")
```

### Latency Targets

```python
class PerformanceTargets:
    """Performance SLA targets."""
    
    # Latency targets (milliseconds)
    LATENCY_TARGETS = {
        'p50': 100,   # 50th percentile: 100ms
        'p95': 500,   # 95th percentile: 500ms
        'p99': 1000   # 99th percentile: 1000ms
    }
    
    # Availability target
    AVAILABILITY_TARGET = 99.9  # 99.9% uptime
    
    # Error rate target
    ERROR_RATE_TARGET = 1.0  # < 1% error rate
    
    @staticmethod
    def evaluate_latency(latencies: list) -> dict:
        """
        Evaluate latencies against targets.
        """
        import numpy as np
        
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        targets = PerformanceTargets.LATENCY_TARGETS
        
        return {
            'p50': {
                'value': p50,
                'target': targets['p50'],
                'meets_target': p50 <= targets['p50']
            },
            'p95': {
                'value': p95,
                'target': targets['p95'],
                'meets_target': p95 <= targets['p95']
            },
            'p99': {
                'value': p99,
                'target': targets['p99'],
                'meets_target': p99 <= targets['p99']
            }
        }
    
    @staticmethod
    def evaluate_availability(total_requests: int, successful_requests: int) -> dict:
        """
        Evaluate availability against target.
        """
        availability = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'availability_percent': availability,
            'target': PerformanceTargets.AVAILABILITY_TARGET,
            'meets_target': availability >= PerformanceTargets.AVAILABILITY_TARGET,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': total_requests - successful_requests
        }

# Usage
targets = PerformanceTargets()

# Example latencies (in milliseconds)
sample_latencies = [
    85, 92, 110, 95, 88, 105, 450, 520, 98, 102,
    890, 95, 105, 115, 88, 92, 780, 105, 98, 950
]

latency_eval = targets.evaluate_latency(sample_latencies)
print("Latency Evaluation:")
for percentile, metrics in latency_eval.items():
    status = "✓" if metrics['meets_target'] else "✗"
    print(f"  {percentile}: {metrics['value']:.0f}ms (target: {metrics['target']}ms) {status}")

# Availability evaluation
avail_eval = targets.evaluate_availability(
    total_requests=100000,
    successful_requests=99950
)
print(f"\nAvailability: {avail_eval['availability_percent']:.3f}%")
print(f"Target: {avail_eval['target']}%")
print(f"Meets Target: {'✓ YES' if avail_eval['meets_target'] else '✗ NO'}")
```

---

## Scalability Testing

### Vertical Scaling Tests

```python
class ScalabilityTester:
    """Test vertical and horizontal scaling."""
    
    def __init__(self, executor: LoadTestExecutor):
        self.executor = executor
    
    def test_vertical_scaling(self, tiers: list, 
                             config: LoadTestConfig) -> dict:
        """
        Test different service tiers (vertical scaling).
        
        Run same load test on different tiers to compare performance.
        """
        results = {}
        
        for tier in tiers:
            print(f"\nTesting tier: {tier}")
            
            # In practice, you would:
            # 1. Create search service at this tier
            # 2. Index documents
            # 3. Run load test
            # 4. Collect metrics
            # 5. Delete service
            
            # Simulated for example
            test_id = f"vertical_scale_{tier.lower()}"
            
            # Run test (simulated)
            print(f"  Running load test on {tier}...")
            # run_result = self.executor.run_test(test_id)
            # test_results = self.executor.get_test_results(run_result['run_id'])
            
            # Simulated results
            limits = AzureSearchTierLimits.get_tier_capacity(tier, 1, 1)
            
            results[tier] = {
                'tier': tier,
                'max_qps': limits['total_qps'],
                'avg_latency_ms': 150,  # Simulated
                'p95_latency_ms': 450,  # Simulated
                'cost_per_month': limits['monthly_cost_usd']
            }
        
        return results
    
    def test_horizontal_scaling(self, tier: str,
                               replica_configs: list,
                               config: LoadTestConfig) -> dict:
        """
        Test different replica counts (horizontal scaling).
        """
        results = {}
        
        for replicas in replica_configs:
            print(f"\nTesting {tier} with {replicas} replicas")
            
            capacity = AzureSearchTierLimits.get_tier_capacity(tier, replicas, 1)
            
            # Run load test (simulated)
            test_id = f"horizontal_scale_{tier}_{replicas}r"
            
            results[f"{replicas}_replicas"] = {
                'replicas': replicas,
                'max_qps': capacity['total_qps'],
                'avg_latency_ms': 150 / replicas,  # Simulated improvement
                'cost_per_month': capacity['monthly_cost_usd']
            }
        
        return results
    
    def analyze_scaling_efficiency(self, results: dict) -> dict:
        """
        Analyze how efficiently system scales.
        
        Perfect linear scaling: 2x resources = 2x throughput
        """
        configs = list(results.keys())
        if len(configs) < 2:
            return {}
        
        baseline_config = configs[0]
        baseline = results[baseline_config]
        
        efficiency = {}
        
        for config in configs[1:]:
            current = results[config]
            
            qps_ratio = current['max_qps'] / baseline['max_qps']
            cost_ratio = current['cost_per_month'] / baseline['cost_per_month']
            
            # Perfect scaling: QPS ratio = cost ratio
            # Efficiency = actual QPS gain / cost increase
            scaling_efficiency = (qps_ratio / cost_ratio) * 100 if cost_ratio > 0 else 0
            
            efficiency[config] = {
                'qps_increase': f"{qps_ratio:.2f}x",
                'cost_increase': f"{cost_ratio:.2f}x",
                'efficiency_percent': scaling_efficiency,
                'efficiency_rating': (
                    'Excellent' if scaling_efficiency >= 90 else
                    'Good' if scaling_efficiency >= 75 else
                    'Fair' if scaling_efficiency >= 60 else
                    'Poor'
                )
            }
        
        return efficiency

# Usage
# executor = LoadTestExecutor(...)
# tester = ScalabilityTester(executor)

# Test vertical scaling
# tiers = ['S1', 'S2', 'S3']
# baseline_config = LoadTestScenarios.baseline_test()
# vertical_results = tester.test_vertical_scaling(tiers, baseline_config)

# Test horizontal scaling
# horizontal_results = tester.test_horizontal_scaling(
#     tier='S2',
#     replica_configs=[1, 2, 4, 6],
#     config=baseline_config
# )

# Analyze efficiency
# efficiency = tester.analyze_scaling_efficiency(horizontal_results)
```

---

## Analysis & Optimization

### Metrics Collection

```python
from azure.monitor.query import LogsQueryClient
from datetime import timedelta

class PerformanceMetricsCollector:
    """Collect performance metrics from Azure Monitor."""
    
    def __init__(self, workspace_id: str, credential):
        self.client = LogsQueryClient(credential)
        self.workspace_id = workspace_id
    
    def query_search_metrics(self, search_service_name: str,
                            timespan: timedelta = timedelta(hours=1)) -> dict:
        """
        Query search service metrics from Azure Monitor.
        """
        query = f"""
        AzureDiagnostics
        | where ResourceProvider == "MICROSOFT.SEARCH"
        | where Resource == "{search_service_name}"
        | where TimeGenerated > ago({int(timespan.total_seconds())}s)
        | summarize
            TotalQueries = count(),
            AvgLatency = avg(Duration_d),
            P95Latency = percentile(Duration_d, 95),
            P99Latency = percentile(Duration_d, 99),
            ErrorCount = countif(ResultType != "Success")
            by bin(TimeGenerated, 5m)
        | order by TimeGenerated desc
        """
        
        response = self.client.query_workspace(
            workspace_id=self.workspace_id,
            query=query,
            timespan=timespan
        )
        
        metrics = []
        for table in response.tables:
            for row in table.rows:
                metrics.append({
                    'timestamp': row[0],
                    'total_queries': row[1],
                    'avg_latency_ms': row[2],
                    'p95_latency_ms': row[3],
                    'p99_latency_ms': row[4],
                    'error_count': row[5]
                })
        
        return metrics
    
    def detect_throttling(self, search_service_name: str,
                         timespan: timedelta = timedelta(hours=1)) -> list:
        """
        Detect 429 (throttling) responses.
        """
        query = f"""
        AzureDiagnostics
        | where ResourceProvider == "MICROSOFT.SEARCH"
        | where Resource == "{search_service_name}"
        | where TimeGenerated > ago({int(timespan.total_seconds())}s)
        | where ResultType == "TooManyRequests" or httpStatusCode_d == 429
        | summarize ThrottledRequests = count() by bin(TimeGenerated, 1m)
        | order by TimeGenerated desc
        """
        
        response = self.client.query_workspace(
            workspace_id=self.workspace_id,
            query=query,
            timespan=timespan
        )
        
        throttling_events = []
        for table in response.tables:
            for row in table.rows:
                throttling_events.append({
                    'timestamp': row[0],
                    'throttled_requests': row[1]
                })
        
        return throttling_events

# Usage
# metrics_collector = PerformanceMetricsCollector(
#     workspace_id=os.getenv("LOG_ANALYTICS_WORKSPACE_ID"),
#     credential=DefaultAzureCredential()
# )
# 
# # Collect metrics
# metrics = metrics_collector.query_search_metrics(
#     search_service_name="my-search-service",
#     timespan=timedelta(hours=1)
# )
# 
# print(f"Collected {len(metrics)} metric snapshots")
# if metrics:
#     latest = metrics[0]
#     print(f"\nLatest metrics:")
#     print(f"  Queries: {latest['total_queries']}")
#     print(f"  Avg Latency: {latest['avg_latency_ms']:.0f}ms")
#     print(f"  P95 Latency: {latest['p95_latency_ms']:.0f}ms")
```

---

## Best Practices

### ✅ Do's
1. **Establish baseline** before optimization
2. **Test incrementally** (don't change multiple variables)
3. **Use realistic query distribution** (head vs tail queries)
4. **Monitor for throttling** (429 responses)
5. **Test with production-like data volume**
6. **Run during off-peak hours** initially
7. **Gradually increase load** (ramp-up period)
8. **Monitor resource utilization** (CPU, memory, disk)
9. **Test failover scenarios**
10. **Document all test configurations**

### ❌ Don'ts
1. **Don't** test in production without approval
2. **Don't** skip baseline testing
3. **Don't** ignore warmup period (JIT compilation, caching)
4. **Don't** run load tests from single IP (use distributed)
5. **Don't** forget to clean up test resources
6. **Don't** use synthetic data only (test with real queries)
7. **Don't** ignore error logs
8. **Don't** exceed tier limits without monitoring
9. **Don't** forget to test degradation scenarios
10. **Don't** skip cost analysis

---

## Next Steps

- **[Cost Analysis](./19-cost-analysis.md)** - TCO and optimization
- **[Multi-Region Deployment](./20-multi-region-deployment.md)** - Geographic distribution
- **[Monitoring & Alerting](./23-monitoring-alerting.md)** - Production monitoring

---

*See also: [A/B Testing](./17-ab-testing-framework.md) | [Scoring Profiles](./14-scoring-profiles.md)*
