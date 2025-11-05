# Multi-Region Deployment

Comprehensive guide to deploying Azure AI Search across multiple geographic regions for global scale and disaster recovery.

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture Patterns](#architecture-patterns)
- [Region Deployment](#region-deployment)
- [Data Synchronization](#data-synchronization)
- [Traffic Management](#traffic-management)
- [Disaster Recovery](#disaster-recovery)
- [Cost Analysis](#cost-analysis)
- [Best Practices](#best-practices)

---

## Overview

### Multi-Region Benefits

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class DeploymentPattern(Enum):
    """Multi-region deployment patterns."""
    ACTIVE_ACTIVE = "active-active"
    ACTIVE_PASSIVE = "active-passive"
    MULTI_WRITE = "multi-write"

@dataclass
class RegionConfig:
    """Configuration for a region deployment."""
    region: str
    tier: str
    replicas: int
    partitions: int
    is_primary: bool
    traffic_weight: float  # 0.0 to 1.0

class MultiRegionBenefits:
    """Document multi-region deployment benefits."""
    
    @staticmethod
    def get_benefits() -> dict:
        """
        Benefits of multi-region deployment.
        """
        return {
            'performance': {
                'reduced_latency': 'Users connect to nearest region',
                'improved_throughput': 'Distribute load across regions',
                'local_data_residency': 'Keep data in required geography'
            },
            'availability': {
                'high_availability': '99.99%+ uptime with multi-region',
                'disaster_recovery': 'Automatic failover on region failure',
                'planned_maintenance': 'Zero-downtime updates'
            },
            'compliance': {
                'data_sovereignty': 'EU data stays in EU, etc.',
                'regulatory_compliance': 'GDPR, HIPAA, etc.',
                'audit_trail': 'Per-region access logs'
            },
            'scalability': {
                'geographic_expansion': 'Add regions as you grow',
                'load_distribution': 'Balance traffic globally',
                'peak_handling': 'Regional spikes isolated'
            }
        }
    
    @staticmethod
    def calculate_latency_improvement(user_region: str, 
                                     single_region: str,
                                     multi_regions: List[str]) -> dict:
        """
        Estimate latency improvement with multi-region.
        
        Typical inter-region latencies (ms):
        - Same region: 5-10ms
        - Same continent: 20-50ms
        - Cross-continent: 100-200ms
        """
        # Simplified latency matrix
        latency_matrix = {
            ('us-east', 'us-east'): 5,
            ('us-east', 'us-west'): 50,
            ('us-east', 'eu-west'): 100,
            ('us-east', 'asia-east'): 200,
            ('us-west', 'us-west'): 5,
            ('us-west', 'eu-west'): 150,
            ('us-west', 'asia-east'): 120,
            ('eu-west', 'eu-west'): 5,
            ('eu-west', 'asia-east'): 180,
            ('asia-east', 'asia-east'): 5
        }
        
        # Latency to single region
        single_latency = latency_matrix.get((user_region, single_region), 150)
        
        # Latency to nearest multi-region
        multi_latencies = [
            latency_matrix.get((user_region, r), 150) 
            for r in multi_regions
        ]
        nearest_latency = min(multi_latencies) if multi_latencies else single_latency
        
        improvement = single_latency - nearest_latency
        improvement_pct = (improvement / single_latency * 100) if single_latency > 0 else 0
        
        return {
            'user_region': user_region,
            'single_region_latency_ms': single_latency,
            'multi_region_latency_ms': nearest_latency,
            'improvement_ms': improvement,
            'improvement_percent': improvement_pct,
            'nearest_region': multi_regions[multi_latencies.index(nearest_latency)] if multi_latencies else None
        }

# Usage
benefits = MultiRegionBenefits()

all_benefits = benefits.get_benefits()
print("Multi-Region Deployment Benefits:\n")
for category, items in all_benefits.items():
    print(f"{category.upper()}:")
    for benefit, description in items.items():
        print(f"  • {benefit.replace('_', ' ').title()}: {description}")
    print()

# Calculate latency improvement
latency = benefits.calculate_latency_improvement(
    user_region='asia-east',
    single_region='us-east',
    multi_regions=['us-east', 'eu-west', 'asia-east']
)

print(f"Latency Improvement Example:")
print(f"  User in: {latency['user_region']}")
print(f"  Single region (us-east): {latency['single_region_latency_ms']}ms")
print(f"  Multi-region (nearest): {latency['multi_region_latency_ms']}ms")
print(f"  Improvement: {latency['improvement_ms']}ms ({latency['improvement_percent']:.1f}%)")
print(f"  Nearest region: {latency['nearest_region']}")
```

---

## Architecture Patterns

### Active-Active Pattern

```python
class ActiveActiveArchitecture:
    """
    Active-Active: All regions serve traffic simultaneously.
    
    Best for:
    - Global applications
    - Read-heavy workloads
    - Maximum availability
    """
    
    def __init__(self, regions: List[RegionConfig]):
        self.regions = regions
    
    def get_architecture_details(self) -> dict:
        """
        Describe active-active architecture.
        """
        return {
            'pattern': 'Active-Active',
            'description': 'All regions actively serve user traffic',
            'characteristics': {
                'write_pattern': 'Writes go to primary, replicated to secondaries',
                'read_pattern': 'Reads served from nearest region',
                'failover': 'Automatic (traffic manager routes around failures)',
                'consistency': 'Eventually consistent across regions'
            },
            'pros': [
                'Lowest latency for global users',
                'Maximum availability (no single point of failure)',
                'Automatic load balancing',
                'Seamless failover'
            ],
            'cons': [
                'Higher cost (all regions fully provisioned)',
                'Data synchronization complexity',
                'Potential for stale reads',
                'Cross-region network costs'
            ],
            'use_cases': [
                'Global e-commerce',
                'SaaS applications',
                'Content delivery',
                'Customer support systems'
            ]
        }
    
    def calculate_capacity(self) -> dict:
        """
        Calculate total capacity across regions.
        """
        total_qps = 0
        total_storage = 0
        
        for region in self.regions:
            # Simplified - would use actual tier limits
            region_qps = 60 * region.replicas * region.partitions  # Assuming S2
            region_storage = 100 * region.partitions  # 100 GB per partition
            
            total_qps += region_qps
            total_storage += region_storage
        
        return {
            'total_regions': len(self.regions),
            'total_qps': total_qps,
            'total_storage_gb': total_storage,
            'regions': [
                {
                    'region': r.region,
                    'qps': 60 * r.replicas * r.partitions,
                    'storage_gb': 100 * r.partitions,
                    'traffic_weight': r.traffic_weight
                }
                for r in self.regions
            ]
        }

# Usage
regions = [
    RegionConfig('us-east', 'S2', replicas=3, partitions=2, is_primary=True, traffic_weight=0.4),
    RegionConfig('eu-west', 'S2', replicas=3, partitions=2, is_primary=False, traffic_weight=0.3),
    RegionConfig('asia-east', 'S2', replicas=3, partitions=2, is_primary=False, traffic_weight=0.3)
]

active_active = ActiveActiveArchitecture(regions)
arch_details = active_active.get_architecture_details()

print("Active-Active Architecture:")
print(f"\nDescription: {arch_details['description']}")
print(f"\nCharacteristics:")
for key, value in arch_details['characteristics'].items():
    print(f"  {key.replace('_', ' ').title()}: {value}")

print(f"\nPros:")
for pro in arch_details['pros']:
    print(f"  ✓ {pro}")

print(f"\nCons:")
for con in arch_details['cons']:
    print(f"  ⚠ {con}")

capacity = active_active.calculate_capacity()
print(f"\n\nTotal Capacity:")
print(f"  Regions: {capacity['total_regions']}")
print(f"  Total QPS: {capacity['total_qps']}")
print(f"  Total Storage: {capacity['total_storage_gb']} GB")
```

### Active-Passive Pattern

```python
class ActivePassiveArchitecture:
    """
    Active-Passive: Primary region serves traffic, secondary for DR.
    
    Best for:
    - Regional applications
    - Cost optimization
    - Disaster recovery focus
    """
    
    def __init__(self, primary: RegionConfig, secondary: RegionConfig):
        self.primary = primary
        self.secondary = secondary
    
    def get_architecture_details(self) -> dict:
        """
        Describe active-passive architecture.
        """
        return {
            'pattern': 'Active-Passive',
            'description': 'Primary region serves all traffic, secondary standby for DR',
            'characteristics': {
                'write_pattern': 'All writes to primary',
                'read_pattern': 'All reads from primary (secondary standby)',
                'failover': 'Manual or automatic on primary failure',
                'consistency': 'Primary is source of truth'
            },
            'pros': [
                'Lower cost (secondary minimal configuration)',
                'Simpler data synchronization',
                'No cross-region routing complexity',
                'Disaster recovery capability'
            ],
            'cons': [
                'Higher latency for distant users',
                'Failover downtime (minutes)',
                'Secondary capacity may be insufficient for full load',
                'Wasted capacity in normal operation'
            ],
            'use_cases': [
                'Regional services',
                'Compliance-driven DR',
                'Cost-sensitive applications',
                'Low-traffic services'
            ]
        }
    
    def calculate_rto_rpo(self, 
                         replication_lag_seconds: int = 300,
                         failover_time_seconds: int = 600) -> dict:
        """
        Calculate Recovery Time Objective and Recovery Point Objective.
        
        Args:
            replication_lag_seconds: Time lag in data replication
            failover_time_seconds: Time to detect failure and failover
        """
        return {
            'rpo_seconds': replication_lag_seconds,
            'rpo_minutes': replication_lag_seconds / 60,
            'rpo_description': f'Up to {replication_lag_seconds/60:.0f} minutes of data loss',
            'rto_seconds': failover_time_seconds,
            'rto_minutes': failover_time_seconds / 60,
            'rto_description': f'Service restored within {failover_time_seconds/60:.0f} minutes',
            'availability_impact': '~10 minutes downtime per year (99.998%)'
        }

# Usage
active_passive = ActivePassiveArchitecture(
    primary=RegionConfig('us-east', 'S2', replicas=3, partitions=2, is_primary=True, traffic_weight=1.0),
    secondary=RegionConfig('us-west', 'S2', replicas=2, partitions=2, is_primary=False, traffic_weight=0.0)
)

arch_details = active_passive.get_architecture_details()
print("Active-Passive Architecture:")
print(f"\nDescription: {arch_details['description']}")

rto_rpo = active_passive.calculate_rto_rpo(
    replication_lag_seconds=300,  # 5 minutes
    failover_time_seconds=600     # 10 minutes
)

print(f"\n\nDisaster Recovery Metrics:")
print(f"  RPO: {rto_rpo['rpo_minutes']:.0f} minutes ({rto_rpo['rpo_description']})")
print(f"  RTO: {rto_rpo['rto_minutes']:.0f} minutes ({rto_rpo['rto_description']})")
print(f"  Availability Impact: {rto_rpo['availability_impact']}")
```

---

## Region Deployment

### Search Service Deployment

```python
from azure.mgmt.search import SearchManagementClient
from azure.mgmt.search.models import SearchService, Sku
from azure.identity import DefaultAzureCredential

class MultiRegionDeployment:
    """Deploy search services across multiple regions."""
    
    def __init__(self, subscription_id: str, resource_group: str):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.credential = DefaultAzureCredential()
        self.client = SearchManagementClient(self.credential, subscription_id)
    
    def deploy_search_service(self, service_name: str, region: str,
                              tier: str, replicas: int, partitions: int) -> dict:
        """
        Deploy Azure AI Search service in a region.
        
        Args:
            service_name: Search service name (must be globally unique)
            region: Azure region (e.g., 'eastus', 'westeurope')
            tier: Service tier (e.g., 'S2')
            replicas: Number of replicas
            partitions: Number of partitions
        """
        # Create search service
        service_params = SearchService(
            location=region,
            sku=Sku(name=tier),
            replica_count=replicas,
            partition_count=partitions,
            hosting_mode='default',
            tags={
                'environment': 'production',
                'deployment_pattern': 'multi-region'
            }
        )
        
        # This would actually deploy the service
        # For example purposes, we'll return the configuration
        
        deployment_config = {
            'service_name': service_name,
            'region': region,
            'tier': tier,
            'replicas': replicas,
            'partitions': partitions,
            'endpoint': f'https://{service_name}.search.windows.net',
            'resource_id': f'/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Search/searchServices/{service_name}'
        }
        
        print(f"Deploying search service: {service_name}")
        print(f"  Region: {region}")
        print(f"  Tier: {tier}")
        print(f"  Configuration: {replicas} replicas × {partitions} partitions")
        print(f"  Endpoint: {deployment_config['endpoint']}")
        
        return deployment_config
    
    def deploy_multi_region(self, base_name: str, 
                           regions_config: List[RegionConfig]) -> List[dict]:
        """
        Deploy search services across multiple regions.
        """
        deployments = []
        
        for i, config in enumerate(regions_config):
            # Generate unique service name per region
            service_name = f"{base_name}-{config.region}"
            
            deployment = self.deploy_search_service(
                service_name=service_name,
                region=config.region,
                tier=config.tier,
                replicas=config.replicas,
                partitions=config.partitions
            )
            
            deployment['is_primary'] = config.is_primary
            deployment['traffic_weight'] = config.traffic_weight
            
            deployments.append(deployment)
            
            print(f"✓ Deployed {i+1}/{len(regions_config)}\n")
        
        return deployments

# Usage
# deployer = MultiRegionDeployment(
#     subscription_id="your-subscription-id",
#     resource_group="search-global-rg"
# )

# Define regions
regions_config = [
    RegionConfig('eastus', 'S2', replicas=3, partitions=2, is_primary=True, traffic_weight=0.4),
    RegionConfig('westeurope', 'S2', replicas=3, partitions=2, is_primary=False, traffic_weight=0.3),
    RegionConfig('southeastasia', 'S2', replicas=3, partitions=2, is_primary=False, traffic_weight=0.3)
]

# Deploy across regions
# deployments = deployer.deploy_multi_region("products-search", regions_config)

print("Multi-Region Deployment Plan:")
for config in regions_config:
    print(f"\n{config.region}:")
    print(f"  Tier: {config.tier}")
    print(f"  Configuration: {config.replicas}R × {config.partitions}P")
    print(f"  Primary: {config.is_primary}")
    print(f"  Traffic Weight: {config.traffic_weight:.0%}")
```

---

## Data Synchronization

### Index Schema Replication

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

class IndexReplicationManager:
    """Manage index schema replication across regions."""
    
    def __init__(self):
        self.clients = {}
    
    def add_region(self, region_name: str, endpoint: str, api_key: str):
        """Add a region to manage."""
        self.clients[region_name] = SearchIndexClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
    
    def replicate_index_schema(self, source_region: str, 
                              target_regions: List[str],
                              index_name: str):
        """
        Replicate index schema from source to target regions.
        """
        # Get source index definition
        source_client = self.clients[source_region]
        source_index = source_client.get_index(index_name)
        
        print(f"Replicating index schema: {index_name}")
        print(f"  Source: {source_region}")
        print(f"  Targets: {', '.join(target_regions)}")
        
        results = []
        
        for target_region in target_regions:
            target_client = self.clients[target_region]
            
            try:
                # Create or update index in target region
                target_client.create_or_update_index(source_index)
                
                results.append({
                    'region': target_region,
                    'status': 'success',
                    'index_name': index_name
                })
                
                print(f"  ✓ {target_region}: Schema replicated")
                
            except Exception as e:
                results.append({
                    'region': target_region,
                    'status': 'failed',
                    'error': str(e)
                })
                
                print(f"  ✗ {target_region}: Failed - {str(e)}")
        
        return results
    
    def compare_schemas(self, index_name: str) -> dict:
        """
        Compare index schemas across all regions.
        """
        schemas = {}
        
        for region, client in self.clients.items():
            try:
                index = client.get_index(index_name)
                
                # Get schema signature
                schema_sig = {
                    'fields_count': len(index.fields),
                    'field_names': sorted([f.name for f in index.fields]),
                    'scoring_profiles': len(index.scoring_profiles or [])
                }
                
                schemas[region] = {
                    'status': 'found',
                    'signature': schema_sig
                }
                
            except Exception as e:
                schemas[region] = {
                    'status': 'not_found',
                    'error': str(e)
                }
        
        # Check if all schemas match
        signatures = [
            s['signature'] 
            for s in schemas.values() 
            if s['status'] == 'found'
        ]
        
        all_match = len(set(str(sig) for sig in signatures)) == 1
        
        return {
            'index_name': index_name,
            'regions': schemas,
            'all_schemas_match': all_match
        }

# Usage
# replication_mgr = IndexReplicationManager()

# Add regions
# replication_mgr.add_region(
#     'us-east',
#     endpoint='https://products-search-eastus.search.windows.net',
#     api_key='...'
# )
# replication_mgr.add_region(
#     'eu-west',
#     endpoint='https://products-search-westeurope.search.windows.net',
#     api_key='...'
# )

# Replicate schema
# results = replication_mgr.replicate_index_schema(
#     source_region='us-east',
#     target_regions=['eu-west', 'asia-east'],
#     index_name='products'
# )

# Compare schemas
# comparison = replication_mgr.compare_schemas('products')
# print(f"\nSchema Comparison:")
# print(f"  All regions match: {comparison['all_schemas_match']}")
```

### Document Synchronization

```python
import asyncio
from azure.search.documents import SearchClient

class DocumentReplicationManager:
    """Manage document replication across regions."""
    
    def __init__(self):
        self.clients = {}
    
    def add_region(self, region_name: str, endpoint: str, 
                   index_name: str, api_key: str):
        """Add a region's search client."""
        self.clients[region_name] = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )
    
    async def replicate_documents_async(self, 
                                       source_region: str,
                                       target_regions: List[str],
                                       batch_size: int = 1000) -> dict:
        """
        Replicate documents from source to target regions.
        
        Args:
            source_region: Source region name
            target_regions: List of target region names
            batch_size: Documents per batch
        """
        source_client = self.clients[source_region]
        
        # Get all documents from source (simplified - in practice use continuation)
        # source_docs = source_client.search("*", select="*")
        
        # For example, simulate with some docs
        print(f"Replicating documents from {source_region} to {len(target_regions)} regions")
        
        total_docs = 0
        batch = []
        
        # In real implementation, iterate through source_docs
        # For now, just demonstrate the pattern
        
        replication_results = {
            'source_region': source_region,
            'target_regions': target_regions,
            'documents_replicated': total_docs,
            'status': 'success'
        }
        
        return replication_results
    
    def setup_change_feed_replication(self, 
                                     source_data: str,
                                     target_regions: List[str]) -> dict:
        """
        Setup change feed-based replication.
        
        Uses Azure Event Grid or Cosmos DB change feed to trigger
        replication when source data changes.
        """
        replication_config = {
            'strategy': 'change_feed',
            'source': source_data,
            'targets': target_regions,
            'trigger': 'azure_event_grid',
            'components': {
                '1_change_detection': 'Cosmos DB Change Feed or Event Grid',
                '2_event_routing': 'Azure Function triggered on change',
                '3_indexing': 'Trigger indexer run on all target regions',
                '4_monitoring': 'Track replication lag per region'
            },
            'benefits': [
                'Near real-time replication',
                'Event-driven (no polling)',
                'Automatic retry on failure',
                'Scalable'
            ]
        }
        
        print("Change Feed Replication Configuration:")
        print(f"  Source: {source_data}")
        print(f"  Targets: {', '.join(target_regions)}")
        print(f"\nComponents:")
        for step, component in replication_config['components'].items():
            print(f"  {step}: {component}")
        
        return replication_config

# Usage
# doc_replication = DocumentReplicationManager()

# Setup change feed replication
change_feed_config = doc_replication.setup_change_feed_replication(
    source_data='cosmos_db://products',
    target_regions=['us-east', 'eu-west', 'asia-east']
)
```

---

## Traffic Management

### Azure Traffic Manager

```python
from azure.mgmt.trafficmanager import TrafficManagerManagementClient
from azure.mgmt.trafficmanager.models import Profile, Endpoint

class TrafficManagerConfig:
    """Configure Azure Traffic Manager for multi-region search."""
    
    def __init__(self, subscription_id: str, resource_group: str):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.credential = DefaultAzureCredential()
        self.client = TrafficManagerManagementClient(self.credential, subscription_id)
    
    def create_traffic_manager_profile(self, 
                                      profile_name: str,
                                      routing_method: str,
                                      endpoints: List[dict]) -> dict:
        """
        Create Traffic Manager profile.
        
        Args:
            profile_name: Traffic Manager profile name
            routing_method: 'Performance', 'Geographic', 'Priority', 'Weighted'
            endpoints: List of endpoint configurations
        """
        # Traffic Manager configuration
        profile_config = {
            'name': profile_name,
            'routing_method': routing_method,
            'dns_name': f'{profile_name}.trafficmanager.net',
            'endpoints': endpoints,
            'monitor_config': {
                'protocol': 'HTTPS',
                'port': 443,
                'path': '/indexes/products/docs?search=*&$top=1&api-version=2023-11-01',
                'interval_seconds': 30,
                'timeout_seconds': 10,
                'tolerated_failures': 3
            }
        }
        
        print(f"Traffic Manager Profile: {profile_name}")
        print(f"  DNS: {profile_config['dns_name']}")
        print(f"  Routing: {routing_method}")
        print(f"  Endpoints: {len(endpoints)}")
        
        return profile_config
    
    def configure_performance_routing(self, endpoints: List[dict]) -> dict:
        """
        Configure Performance routing (lowest latency).
        
        Automatically routes to endpoint with lowest latency for user.
        """
        profile = self.create_traffic_manager_profile(
            profile_name='search-global-perf',
            routing_method='Performance',
            endpoints=endpoints
        )
        
        profile['routing_details'] = {
            'method': 'Performance',
            'description': 'Routes to endpoint with lowest latency',
            'use_case': 'Global applications prioritizing speed'
        }
        
        return profile
    
    def configure_geographic_routing(self, endpoints: List[dict]) -> dict:
        """
        Configure Geographic routing (data residency).
        
        Routes based on user's geographic location.
        """
        profile = self.create_traffic_manager_profile(
            profile_name='search-global-geo',
            routing_method='Geographic',
            endpoints=endpoints
        )
        
        # Add geographic mappings
        for endpoint in endpoints:
            if 'geographic_mapping' in endpoint:
                print(f"\n{endpoint['name']}:")
                print(f"  Serves: {', '.join(endpoint['geographic_mapping'])}")
        
        profile['routing_details'] = {
            'method': 'Geographic',
            'description': 'Routes based on user location',
            'use_case': 'Data residency requirements (GDPR, etc.)'
        }
        
        return profile
    
    def configure_weighted_routing(self, endpoints: List[dict]) -> dict:
        """
        Configure Weighted routing (A/B testing, gradual rollout).
        """
        profile = self.create_traffic_manager_profile(
            profile_name='search-global-weighted',
            routing_method='Weighted',
            endpoints=endpoints
        )
        
        total_weight = sum(e['weight'] for e in endpoints)
        
        print("\nTraffic Distribution:")
        for endpoint in endpoints:
            percentage = (endpoint['weight'] / total_weight * 100) if total_weight > 0 else 0
            print(f"  {endpoint['name']}: {percentage:.1f}% (weight: {endpoint['weight']})")
        
        profile['routing_details'] = {
            'method': 'Weighted',
            'description': 'Distributes traffic by weight',
            'use_case': 'A/B testing, gradual rollout'
        }
        
        return profile

# Usage
# tm_config = TrafficManagerConfig(
#     subscription_id="your-subscription-id",
#     resource_group="search-global-rg"
# )

# Define endpoints
endpoints = [
    {
        'name': 'us-east-endpoint',
        'target': 'products-search-eastus.search.windows.net',
        'region': 'East US',
        'weight': 40,
        'geographic_mapping': ['United States', 'Canada', 'Mexico']
    },
    {
        'name': 'eu-west-endpoint',
        'target': 'products-search-westeurope.search.windows.net',
        'region': 'West Europe',
        'weight': 30,
        'geographic_mapping': ['Europe', 'Middle East', 'Africa']
    },
    {
        'name': 'asia-east-endpoint',
        'target': 'products-search-southeastasia.search.windows.net',
        'region': 'Southeast Asia',
        'weight': 30,
        'geographic_mapping': ['Asia Pacific']
    }
]

# Performance routing
# perf_profile = tm_config.configure_performance_routing(endpoints)

# Geographic routing
# geo_profile = tm_config.configure_geographic_routing(endpoints)

# Weighted routing
# weighted_profile = tm_config.configure_weighted_routing(endpoints)

print("Traffic Manager Configuration:")
print("\nEndpoints:")
for ep in endpoints:
    print(f"\n{ep['name']}:")
    print(f"  Target: {ep['target']}")
    print(f"  Weight: {ep['weight']}%")
    if 'geographic_mapping' in ep:
        print(f"  Serves: {', '.join(ep['geographic_mapping'])}")
```

### Azure Front Door

```python
class FrontDoorConfig:
    """Configure Azure Front Door for global load balancing."""
    
    def __init__(self):
        self.config = {}
    
    def create_front_door_config(self, name: str, backends: List[dict]) -> dict:
        """
        Create Azure Front Door configuration.
        
        Azure Front Door provides:
        - Global load balancing
        - SSL offloading
        - WAF (Web Application Firewall)
        - Caching
        - URL-based routing
        """
        config = {
            'name': name,
            'frontend_endpoints': [
                {
                    'name': 'default',
                    'host_name': f'{name}.azurefd.net'
                }
            ],
            'backend_pools': [
                {
                    'name': 'search-backend',
                    'backends': backends,
                    'load_balancing': {
                        'sample_size': 4,
                        'successful_samples_required': 2
                    },
                    'health_probe': {
                        'path': '/indexes/products/docs?search=*&$top=1&api-version=2023-11-01',
                        'protocol': 'Https',
                        'interval_seconds': 30
                    }
                }
            ],
            'routing_rules': [
                {
                    'name': 'search-routing',
                    'frontend_endpoints': ['default'],
                    'accepted_protocols': ['Https'],
                    'patterns_to_match': ['/*'],
                    'backend_pool': 'search-backend'
                }
            ]
        }
        
        print(f"Azure Front Door Configuration: {name}")
        print(f"  Frontend: {config['frontend_endpoints'][0]['host_name']}")
        print(f"  Backends: {len(backends)}")
        
        return config
    
    def configure_waf_policy(self) -> dict:
        """
        Configure Web Application Firewall policy.
        """
        waf_config = {
            'mode': 'Prevention',  # or 'Detection'
            'rules': {
                'sql_injection': 'enabled',
                'xss': 'enabled',
                'protocol_enforcement': 'enabled',
                'rate_limiting': {
                    'enabled': True,
                    'requests_per_minute': 100,
                    'duration_seconds': 60
                }
            },
            'custom_rules': [
                {
                    'name': 'block-api-key-in-url',
                    'priority': 1,
                    'rule_type': 'MatchRule',
                    'match_conditions': [
                        {
                            'match_variable': 'QueryString',
                            'operator': 'Contains',
                            'match_values': ['api-key']
                        }
                    ],
                    'action': 'Block'
                }
            ]
        }
        
        print("\nWAF Policy Configuration:")
        print(f"  Mode: {waf_config['mode']}")
        print(f"  Rate Limiting: {waf_config['rules']['rate_limiting']['requests_per_minute']} req/min")
        
        return waf_config

# Usage
fd_config = FrontDoorConfig()

backends = [
    {
        'address': 'products-search-eastus.search.windows.net',
        'http_port': 80,
        'https_port': 443,
        'priority': 1,
        'weight': 50,
        'enabled': True
    },
    {
        'address': 'products-search-westeurope.search.windows.net',
        'http_port': 80,
        'https_port': 443,
        'priority': 1,
        'weight': 30,
        'enabled': True
    },
    {
        'address': 'products-search-southeastasia.search.windows.net',
        'http_port': 80,
        'https_port': 443,
        'priority': 1,
        'weight': 20,
        'enabled': True
    }
]

front_door = fd_config.create_front_door_config('search-global-fd', backends)
waf = fd_config.configure_waf_policy()
```

---

## Disaster Recovery

### Failover Planning

```python
class DisasterRecoveryPlan:
    """Disaster recovery planning for multi-region search."""
    
    def __init__(self, regions: List[RegionConfig]):
        self.regions = regions
        self.primary = next((r for r in regions if r.is_primary), None)
        self.secondaries = [r for r in regions if not r.is_primary]
    
    def calculate_rpo_rto(self, replication_strategy: str) -> dict:
        """
        Calculate Recovery Point Objective and Recovery Time Objective.
        
        Args:
            replication_strategy: 'real-time', 'scheduled', 'manual'
        """
        rpo_rto = {
            'real-time': {
                'rpo_seconds': 60,  # 1 minute data loss max
                'rto_seconds': 120,  # 2 minutes to failover
                'description': 'Change feed with automatic failover'
            },
            'scheduled': {
                'rpo_seconds': 3600,  # 1 hour data loss max
                'rto_seconds': 600,   # 10 minutes to failover
                'description': 'Hourly replication with manual failover'
            },
            'manual': {
                'rpo_seconds': 86400,  # 24 hours data loss max
                'rto_seconds': 3600,   # 1 hour to failover
                'description': 'Daily backup with manual recovery'
            }
        }
        
        strategy_metrics = rpo_rto.get(replication_strategy, rpo_rto['scheduled'])
        
        return {
            'replication_strategy': replication_strategy,
            'rpo': {
                'seconds': strategy_metrics['rpo_seconds'],
                'minutes': strategy_metrics['rpo_seconds'] / 60,
                'hours': strategy_metrics['rpo_seconds'] / 3600
            },
            'rto': {
                'seconds': strategy_metrics['rto_seconds'],
                'minutes': strategy_metrics['rto_seconds'] / 60,
                'hours': strategy_metrics['rto_seconds'] / 3600
            },
            'description': strategy_metrics['description']
        }
    
    def create_failover_runbook(self) -> dict:
        """
        Create step-by-step failover runbook.
        """
        runbook = {
            'scenario': 'Primary Region Failure',
            'detection': {
                'step': 1,
                'actions': [
                    'Monitor detects primary region unavailable',
                    'Health checks fail for >3 minutes',
                    'Alert sent to on-call engineer'
                ],
                'automation': 'Azure Monitor alert triggers Logic App'
            },
            'validation': {
                'step': 2,
                'actions': [
                    'Confirm primary region is down (not transient)',
                    'Check secondary region health',
                    'Verify data replication lag acceptable'
                ],
                'automation': 'Automated health checks'
            },
            'failover': {
                'step': 3,
                'actions': [
                    'Update Traffic Manager endpoint priority',
                    'Or update Front Door backend weights',
                    'Redirect traffic to secondary region',
                    'Verify traffic is flowing'
                ],
                'automation': 'Can be automated or manual approval'
            },
            'communication': {
                'step': 4,
                'actions': [
                    'Update status page',
                    'Notify stakeholders',
                    'Document incident in ticket'
                ],
                'automation': 'Automated notifications'
            },
            'recovery': {
                'step': 5,
                'actions': [
                    'Investigate root cause',
                    'Wait for primary region recovery',
                    'Sync data if needed',
                    'Failback to primary when ready'
                ],
                'automation': 'Manual decision'
            }
        }
        
        return runbook
    
    def test_disaster_recovery(self) -> dict:
        """
        DR testing procedure.
        """
        test_plan = {
            'frequency': 'Quarterly',
            'test_types': {
                'table_top': {
                    'description': 'Walk through runbook with team',
                    'duration': '1 hour',
                    'frequency': 'Monthly'
                },
                'simulated_failover': {
                    'description': 'Failover to secondary (off-peak hours)',
                    'duration': '2 hours',
                    'frequency': 'Quarterly'
                },
                'full_dr_drill': {
                    'description': 'Complete DR scenario with rollback',
                    'duration': '4 hours',
                    'frequency': 'Annually'
                }
            },
            'success_criteria': [
                'Failover completes within RTO',
                'Data loss within RPO',
                'All services functional in secondary',
                'Runbook accurately reflects procedure',
                'Team members know their roles'
            ]
        }
        
        return test_plan

# Usage
dr_plan = DisasterRecoveryPlan(regions_config)

# Calculate RPO/RTO
rpo_rto = dr_plan.calculate_rpo_rto('real-time')
print("Disaster Recovery Metrics:")
print(f"  Replication: {rpo_rto['replication_strategy']}")
print(f"  RPO: {rpo_rto['rpo']['minutes']:.0f} minutes")
print(f"  RTO: {rto_rpo['rto']['minutes']:.0f} minutes")
print(f"  Description: {rpo_rto['description']}")

# Get failover runbook
runbook = dr_plan.create_failover_runbook()
print(f"\n\nFailover Runbook: {runbook['scenario']}")
for phase, details in runbook.items():
    if phase == 'scenario':
        continue
    print(f"\n{phase.upper()} (Step {details['step']}):")
    for action in details['actions']:
        print(f"  • {action}")

# DR testing
test_plan = dr_plan.test_disaster_recovery()
print(f"\n\nDR Testing Plan:")
print(f"  Frequency: {test_plan['frequency']}")
print(f"\nTest Types:")
for test_type, details in test_plan['test_types'].items():
    print(f"\n  {test_type.replace('_', ' ').title()}:")
    print(f"    {details['description']}")
    print(f"    Duration: {details['duration']}")
    print(f"    Frequency: {details['frequency']}")
```

---

## Cost Analysis

### Multi-Region Cost Calculator

```python
class MultiRegionCostCalculator:
    """Calculate costs for multi-region deployment."""
    
    @staticmethod
    def calculate_search_service_costs(regions: List[RegionConfig]) -> dict:
        """
        Calculate total search service costs across regions.
        """
        from ..cost_analysis import AzureSearchPricing
        
        total_monthly = 0
        region_costs = []
        
        for region in regions:
            pricing = AzureSearchPricing.get_pricing(region.tier)
            monthly_cost = pricing.calculate_monthly_cost(
                region.replicas, 
                region.partitions
            )
            
            total_monthly += monthly_cost
            
            region_costs.append({
                'region': region.region,
                'tier': region.tier,
                'configuration': f'{region.replicas}R × {region.partitions}P',
                'monthly_cost': monthly_cost,
                'annual_cost': monthly_cost * 12
            })
        
        return {
            'regions': region_costs,
            'total_monthly': total_monthly,
            'total_annual': total_monthly * 12
        }
    
    @staticmethod
    def calculate_traffic_management_costs(routing_method: str,
                                          queries_per_month: int) -> dict:
        """
        Calculate Traffic Manager or Front Door costs.
        """
        if routing_method == 'traffic_manager':
            # Traffic Manager: $0.54 per million DNS queries
            # + $0.36 per enabled endpoint
            
            dns_queries = queries_per_month
            dns_cost = (dns_queries / 1_000_000) * 0.54
            
            # Assume 3 endpoints
            endpoint_cost = 3 * 0.36
            
            monthly_cost = dns_cost + endpoint_cost
            
            return {
                'service': 'Azure Traffic Manager',
                'dns_queries': dns_queries,
                'dns_cost': dns_cost,
                'endpoint_cost': endpoint_cost,
                'total_monthly': monthly_cost
            }
        
        else:  # front_door
            # Front Door: Base + outbound data transfer
            # Base: $35/month
            # Outbound data: $0.12/GB for first 10 TB
            
            base_cost = 35.0
            
            # Estimate outbound data (10 KB avg response × queries)
            avg_response_kb = 10
            outbound_gb = (queries_per_month * avg_response_kb) / (1024 * 1024)
            outbound_cost = outbound_gb * 0.12
            
            monthly_cost = base_cost + outbound_cost
            
            return {
                'service': 'Azure Front Door',
                'base_cost': base_cost,
                'outbound_data_gb': outbound_gb,
                'outbound_data_cost': outbound_cost,
                'total_monthly': monthly_cost
            }
    
    @staticmethod
    def calculate_replication_bandwidth_costs(
        regions: int,
        monthly_index_updates_gb: float
    ) -> dict:
        """
        Calculate inter-region data transfer costs.
        
        Intra-region: Free
        Inter-region: $0.02-0.12/GB depending on regions
        """
        # Assume $0.05/GB average for inter-region
        cost_per_gb = 0.05
        
        # Each update goes to (regions - 1) other regions
        total_transfer_gb = monthly_index_updates_gb * (regions - 1)
        
        monthly_cost = total_transfer_gb * cost_per_gb
        
        return {
            'regions': regions,
            'index_updates_per_region_gb': monthly_index_updates_gb,
            'total_transfer_gb': total_transfer_gb,
            'cost_per_gb': cost_per_gb,
            'monthly_cost': monthly_cost
        }

# Usage
cost_calc = MultiRegionCostCalculator()

# Calculate search service costs
search_costs = cost_calc.calculate_search_service_costs(regions_config)
print("Search Service Costs:")
for region in search_costs['regions']:
    print(f"\n{region['region']}:")
    print(f"  {region['tier']} ({region['configuration']})")
    print(f"  Monthly: ${region['monthly_cost']:,.2f}")

print(f"\nTotal Monthly: ${search_costs['total_monthly']:,.2f}")
print(f"Total Annual: ${search_costs['total_annual']:,.2f}")

# Traffic management costs
tm_costs = cost_calc.calculate_traffic_management_costs(
    routing_method='traffic_manager',
    queries_per_month=10_000_000
)

print(f"\n\n{tm_costs['service']} Costs:")
print(f"  DNS queries: {tm_costs['dns_queries']:,}")
print(f"  DNS cost: ${tm_costs['dns_cost']:.2f}")
print(f"  Endpoint cost: ${tm_costs['endpoint_cost']:.2f}")
print(f"  Total monthly: ${tm_costs['total_monthly']:.2f}")

# Bandwidth costs
bandwidth_costs = cost_calc.calculate_replication_bandwidth_costs(
    regions=3,
    monthly_index_updates_gb=100
)

print(f"\n\nReplication Bandwidth Costs:")
print(f"  Regions: {bandwidth_costs['regions']}")
print(f"  Updates per region: {bandwidth_costs['index_updates_per_region_gb']} GB")
print(f"  Total transfer: {bandwidth_costs['total_transfer_gb']} GB")
print(f"  Monthly cost: ${bandwidth_costs['monthly_cost']:.2f}")
```

---

## Best Practices

### ✅ Do's
1. **Start with active-passive** for DR, expand to active-active for scale
2. **Use Traffic Manager Performance routing** for lowest latency
3. **Replicate index schema** before documents
4. **Test disaster recovery quarterly**
5. **Monitor replication lag** across regions
6. **Use geographic routing** for data residency compliance
7. **Document failover procedures** thoroughly
8. **Set up health probes** on all endpoints
9. **Consider Front Door** for advanced features (WAF, caching)
10. **Plan for cross-region costs** (bandwidth, data transfer)

### ❌ Don'ts
1. **Don't** deploy to all regions initially (start with 2-3)
2. **Don't** forget to test failover procedures
3. **Don't** assume instant replication (monitor lag)
4. **Don't** ignore data sovereignty requirements
5. **Don't** over-provision all regions equally
6. **Don't** use same admin keys across regions
7. **Don't** forget to update DNS TTL for faster failover
8. **Don't** skip monitoring in secondary regions
9. **Don't** ignore cross-region bandwidth costs
10. **Don't** deploy without runbooks

---

## Next Steps

- **[Security & Compliance](./21-security-compliance.md)** - Security best practices
- **[CI/CD Pipelines](./22-cicd-pipelines.md)** - Infrastructure as Code
- **[Monitoring & Alerting](./23-monitoring-alerting.md)** - Production monitoring

---

*See also: [Cost Analysis](./19-cost-analysis.md) | [Load Testing](./18-load-testing.md)*
