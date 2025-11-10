# Index Management

Complete guide to managing Azure AI Search indexes including aliases, versioning, schema updates, lifecycle management, and zero-downtime deployments.

## 📋 Table of Contents
- [Overview](#overview)
- [Index Aliases](#index-aliases)
- [Versioning Strategies](#versioning-strategies)
- [Schema Changes](#schema-changes)
- [Lifecycle Management](#lifecycle-management)
- [Capacity Planning](#capacity-planning)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)

---

## Overview

### Real-World Scenario: GlobalRetail's Zero-Downtime Migration

**The Challenge**

GlobalRetail operates a high-traffic e-commerce platform serving 15 million customers across 40 countries. Their product catalog index contains 8.2 million SKUs with real-time inventory updates (450K updates/hour during peak). The search service handles 2.8 million queries per day (avg 32 QPS, peak 280 QPS during flash sales).

**The Problem: Breaking Change Required**

After 18 months in production, the search team needed to implement critical changes:
- **Analyzer upgrade**: Switch from standard analyzer to custom domain-specific analyzer (30% relevance improvement in testing)
- **Schema enhancement**: Add vector embeddings for hybrid search (128-dimension vectors, ~0.5KB per document → +4.1GB storage)
- **Field type correction**: Change `price` from String to Double (was preventing numerical range filters)
- **Scoring profile overhaul**: Implement multi-signal relevance with freshness, popularity, ratings

These changes required a **complete index rebuild** — historically a 6-8 hour process causing:
- 6-8 hours of degraded search quality (old index)
- Risk of data loss during migration
- Customer-facing downtime during alias switching
- No rollback capability if issues discovered

**Previous Failed Approach**

Their first attempt used **in-place rebuild**:

1. ❌ Stopped indexer (inventory updates paused)
2. ❌ Deleted and recreated index (6.2 hour downtime)
3. ❌ Re-indexed all documents (2.1 hour lag before search available)
4. ❌ Discovered analyzer bug 30 minutes after go-live
5. ❌ No rollback option → manual fixes under pressure
6. **Result**: 8.3 hours total outage, $420K revenue loss (avg $50.6K/hour), 14% spike in support tickets

**Blue-Green Deployment Solution**

The team implemented a **zero-downtime blue-green deployment** with automated validation:

**Phase 1: Preparation (Week 1)**
- Created index versioning convention: `products-v{major}-{YYYYMMDD}`
- Established alias: `products-prod` → `products-v1-20240415`
- All application code updated to use alias (not direct index name)
- Automated schema export to Git (version control for index definitions)

**Phase 2: Build Green Index (Week 2)**
- Created new index: `products-v2-20240513` (green)
- New schema with all improvements (analyzer, vectors, field fixes, scoring)
- Parallel indexing: Maintained live index (blue) while building new (green)
- 4.5-hour initial load (8.2M documents) running in background

**Phase 3: Validation (Week 3)**
- Automated validation suite ran 2,847 test queries against green index:
  - **Precision@5**: 0.72 (baseline: 0.61) ✅ +18%
  - **NDCG@10**: 0.78 (baseline: 0.65) ✅ +20%
  - **Zero-result rate**: 2.8% (baseline: 8.4%) ✅ -67%
  - **Avg query latency**: 42ms (baseline: 38ms) ⚠️ +11% (acceptable)
- Discovered 3 critical issues during validation:
  - Vector search timeout on complex queries → tuned HNSW parameters
  - Price filter returning null for 0.03% of products → data cleanup
  - Synonym map incompatibility → updated to new analyzer format
- **Fixed all issues before production traffic** (in blue-green, blue still serving 100%)

**Phase 4: Gradual Rollout (Week 4)**
- 10% canary: Updated alias to split traffic 90% blue, 10% green (used application-level routing)
  - Monitored for 24 hours: CTR +12%, session duration +8%, zero incidents
- 50% rollout: Increased to 50/50 split
  - Monitored for 48 hours: Sustained improvements, latency within acceptable range
- 100% rollout: Updated alias to point 100% to green (`products-v2-20240513`)
  - **Total production downtime: 0 seconds** (alias update is atomic)
  - Kept blue index available for 7 days (instant rollback if needed)

**Phase 5: Cleanup (Week 5)**
- Monitoring confirmed success (no rollback needed)
- Deleted old blue index (`products-v1-20240415`)
- Saved 4.1GB storage costs ($82/month on S2 tier)

**Complete Migration Timeline**

```
Week 1 (Prep):
├─ Mon-Tue: Create v2 schema, setup alias infrastructure
├─ Wed-Thu: Update app code to use alias, deploy to staging
└─ Fri: Validate alias routing in staging, schema review

Week 2 (Build):
├─ Mon 00:00: Start v2 index build (background)
│  ├─ Initial load: 8.2M docs, 4.5 hours
│  └─ Real-time sync: Keep v2 current with v1 changes
├─ Tue-Fri: Monitor build progress, verify data quality
└─ Status: v2 index complete, in sync with v1

Week 3 (Validation):
├─ Mon-Wed: Automated validation suite (2,847 test queries)
│  ├─ Found 3 issues: Vector timeout, price nulls, synonym errors
│  └─ Fixed all issues in v2 (v1 still serving 100% of traffic)
├─ Thu: Re-validation after fixes → all tests pass ✅
└─ Fri: Stakeholder review, go-live approval

Week 4 (Rollout):
├─ Mon 10:00: 10% canary (alias routing split)
│  └─ 24hr monitoring: CTR +12%, duration +8%, 0 errors
├─ Wed 10:00: 50% rollout
│  └─ 48hr monitoring: Sustained improvements
├─ Fri 14:00: 100% cutover (alias points to v2)
│  └─ Downtime: 0 seconds (atomic alias update)
└─ Keep v1 available for 7-day rollback window

Week 5 (Cleanup):
├─ Mon: Final metrics review
├─ Wed: Delete v1 index (no rollback needed)
└─ Fri: Document lessons learned, update runbooks
```

**Business Impact (First 30 Days)**

**Search Quality Improvements**:
- Click-through rate: 18.4% → 21.7% (+18%)
- Zero-result queries: 8.4% → 2.8% (-67%)
- Average session duration: 4.2 min → 5.1 min (+21%)
- Product discovery (>3 products viewed): 34% → 48% (+41%)
- Conversion rate: 3.8% → 4.6% (+21%)

**Technical Wins**:
- **Zero downtime**: 0 seconds outage (vs 8.3 hours previous attempt)
- **Risk mitigation**: 7-day rollback window (vs no rollback capability)
- **Safe validation**: Fixed 3 critical bugs before production impact
- **Gradual rollout**: Caught issues at 10% traffic (not 100%)
- **Index build time**: 4.5 hours (vs 8.3 hours with downtime)

**Financial Results**:
- Revenue during migration: $0 lost (vs $420K previous attempt)
- Additional revenue from improvements: +$1.2M/month (higher conversion)
- Implementation cost: $45K (engineering time, testing infrastructure)
- **ROI**: 2,567% first year, 1.1-day payback period
- Storage savings: -$82/month (cleanup of old indexes)

**Key Learnings**

1. **Aliases are non-negotiable**: Never point applications directly to index names. Aliases enable instant cutover and rollback.

2. **Validate before 100% rollout**: Caught 3 critical bugs in validation phase that would have impacted 2.8M queries/day.

3. **Gradual rollout de-risks deployment**: 10% canary → 50% → 100% allows catching issues with limited blast radius.

4. **Keep old version during rollback window**: The ability to instantly revert to `products-v1` reduced deployment anxiety and enabled confident rollout.

5. **Automate everything**: Automated validation suite (2,847 queries) caught issues humans would have missed.

6. **Monitor business metrics, not just technical**: CTR and session duration revealed improvements latency metrics didn't show.

7. **Version naming matters**: `products-v2-20240513` clearly shows version and deployment date for audit trail.

**Index Management Fundamentals**

```python
class IndexManagementOverview:
    """Understanding index management in Azure AI Search."""
    
    @staticmethod
    def management_scenarios():
        """
        Common index management scenarios:
        
        1. Schema updates (adding/modifying fields)
        2. Scoring profile changes
        3. Analyzer modifications
        4. Zero-downtime deployments
        5. Index rebuilds
        6. Capacity scaling
        7. Data migration
        """
        return {
            'schema_updates': {
                'additive_changes': 'Add new fields (no rebuild)',
                'breaking_changes': 'Modify existing fields (rebuild required)',
                'safe_updates': 'Scoring profiles, synonyms (no rebuild)'
            },
            'deployment_strategies': {
                'blue_green': 'Maintain two indexes, switch alias',
                'rolling_update': 'Update in-place with versioning',
                'canary': 'Gradual rollout to subset of traffic'
            },
            'operations': {
                'create': 'Initial index creation',
                'update': 'Modify existing index',
                'rebuild': 'Recreate from scratch',
                'delete': 'Remove old indexes'
            }
        }
    
    @staticmethod
    def update_impact_matrix():
        """
        Impact of different update types.
        
        Returns matrix showing rebuild requirements.
        """
        return {
            'no_rebuild_required': [
                'Add new field',
                'Update scoring profile',
                'Update synonym map',
                'Add suggester',
                'Update CORS options'
            ],
            'rebuild_required': [
                'Modify field type',
                'Change field analyzer',
                'Remove field',
                'Change key field',
                'Modify field properties (searchable, filterable, etc.)'
            ],
            'rebuild_recommended': [
                'Major scoring changes',
                'Analyzer changes affecting existing data',
                'Performance optimization'
            ]
        }

# Usage
overview = IndexManagementOverview()

scenarios = overview.management_scenarios()
print("Index Management Scenarios:")
for category, operations in scenarios.items():
    print(f"\n{category}:")
    if isinstance(operations, dict):
        for op, desc in operations.items():
            print(f"  {op}: {desc}")

impact = overview.update_impact_matrix()
print("\n\nUpdate Impact:")
print("\nNo rebuild required:")
for change in impact['no_rebuild_required']:
    print(f"  ✓ {change}")

print("\nRebuild required:")
for change in impact['rebuild_required']:
    print(f"  ⚠ {change}")
```

---

## Index Aliases

### Creating and Managing Aliases

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchAlias
from azure.core.credentials import AzureKeyCredential

class IndexAliasManager:
    """Manage index aliases for zero-downtime updates."""
    
    def __init__(self, search_endpoint, admin_key):
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_alias(self, alias_name, target_index):
        """
        Create an alias pointing to an index.
        
        Aliases enable:
        - Zero-downtime deployments
        - Easy rollback
        - A/B testing
        - Blue-green deployments
        """
        alias = SearchAlias(
            name=alias_name,
            indexes=[target_index]
        )
        
        result = self.index_client.create_or_update_alias(alias)
        print(f"Created alias '{alias_name}' → '{target_index}'")
        return result
    
    def update_alias(self, alias_name, new_target_index):
        """
        Update alias to point to different index.
        
        This is atomic - applications using the alias
        immediately start using the new index.
        """
        alias = SearchAlias(
            name=alias_name,
            indexes=[new_target_index]
        )
        
        result = self.index_client.create_or_update_alias(alias)
        print(f"Updated alias '{alias_name}' → '{new_target_index}'")
        return result
    
    def get_alias_target(self, alias_name):
        """Get current target of an alias."""
        try:
            alias = self.index_client.get_alias(alias_name)
            return alias.indexes[0] if alias.indexes else None
        except Exception as e:
            print(f"Error getting alias: {e}")
            return None
    
    def delete_alias(self, alias_name):
        """Delete an alias."""
        self.index_client.delete_alias(alias_name)
        print(f"Deleted alias '{alias_name}'")
    
    def list_aliases(self):
        """List all aliases."""
        aliases = self.index_client.list_aliases()
        return [
            {
                'name': alias.name,
                'target': alias.indexes[0] if alias.indexes else None
            }
            for alias in aliases
        ]

# Usage
import os

alias_manager = IndexAliasManager(
    search_endpoint=os.getenv("SEARCH_ENDPOINT"),
    admin_key=os.getenv("SEARCH_ADMIN_KEY")
)

# Create alias for production index
alias_manager.create_alias(
    alias_name="products-prod",
    target_index="products-v1"
)

# List all aliases
aliases = alias_manager.list_aliases()
print("\nCurrent aliases:")
for alias in aliases:
    print(f"  {alias['name']} → {alias['target']}")

# Get specific alias target
current_target = alias_manager.get_alias_target("products-prod")
print(f"\nCurrent production index: {current_target}")
```

### Blue-Green Deployment Pattern

```python
class BlueGreenDeployment:
    """Implement blue-green deployment pattern."""
    
    def __init__(self, index_client, alias_manager):
        self.index_client = index_client
        self.alias_manager = alias_manager
    
    def deploy_new_version(self, alias_name, new_index_definition, populate_data_func):
        """
        Deploy new index version using blue-green pattern.
        
        Steps:
        1. Get current index (blue)
        2. Create new index (green)
        3. Populate new index
        4. Validate new index
        5. Switch alias to new index
        6. Keep old index for rollback
        
        Args:
            alias_name: Production alias name
            new_index_definition: New index configuration
            populate_data_func: Function to populate index
        """
        # Step 1: Get current index
        current_index = self.alias_manager.get_alias_target(alias_name)
        print(f"Current index (blue): {current_index}")
        
        # Step 2: Create new index
        new_index_name = self._generate_new_index_name(current_index)
        new_index_definition.name = new_index_name
        
        print(f"Creating new index (green): {new_index_name}")
        self.index_client.create_or_update_index(new_index_definition)
        
        # Step 3: Populate new index
        print(f"Populating new index...")
        try:
            populate_data_func(new_index_name)
            print(f"Data population complete")
        except Exception as e:
            print(f"Error populating index: {e}")
            self.index_client.delete_index(new_index_name)
            raise
        
        # Step 4: Validate new index
        print(f"Validating new index...")
        validation_passed = self._validate_index(new_index_name)
        
        if not validation_passed:
            print(f"Validation failed, aborting deployment")
            self.index_client.delete_index(new_index_name)
            return False
        
        # Step 5: Switch alias (atomic operation)
        print(f"Switching alias to new index...")
        self.alias_manager.update_alias(alias_name, new_index_name)
        
        print(f"Deployment complete!")
        print(f"  Old index (blue): {current_index}")
        print(f"  New index (green): {new_index_name}")
        print(f"  Alias: {alias_name} → {new_index_name}")
        
        return True
    
    def _generate_new_index_name(self, current_index_name):
        """Generate new index name with version increment."""
        import re
        from datetime import datetime
        
        if not current_index_name:
            # First deployment
            return f"products-v1-{datetime.now().strftime('%Y%m%d')}"
        
        # Extract version from current name (products-v1 → v1)
        version_match = re.search(r'-v(\d+)', current_index_name)
        if version_match:
            current_version = int(version_match.group(1))
            new_version = current_version + 1
            base_name = current_index_name[:version_match.start()]
            return f"{base_name}-v{new_version}-{datetime.now().strftime('%Y%m%d')}"
        else:
            # No version in current name
            return f"{current_index_name}-v2-{datetime.now().strftime('%Y%m%d')}"
    
    def _validate_index(self, index_name):
        """
        Validate new index before switching.
        
        Checks:
        - Document count > 0
        - Schema is correct
        - Sample queries work
        """
        from azure.search.documents import SearchClient
        
        # Get index statistics
        stats = self.index_client.get_index_statistics(index_name)
        doc_count = stats.document_count
        
        print(f"  Document count: {doc_count:,}")
        
        if doc_count == 0:
            print(f"  ✗ No documents in index")
            return False
        
        # Test sample query
        search_client = SearchClient(
            endpoint=self.index_client._endpoint,
            index_name=index_name,
            credential=self.index_client._credential
        )
        
        try:
            results = list(search_client.search("*", top=1))
            print(f"  ✓ Sample query successful")
            return True
        except Exception as e:
            print(f"  ✗ Sample query failed: {e}")
            return False
    
    def rollback(self, alias_name, previous_index_name):
        """
        Rollback to previous index version.
        
        Simply update alias to point back to old index.
        """
        print(f"Rolling back alias '{alias_name}' to '{previous_index_name}'")
        self.alias_manager.update_alias(alias_name, previous_index_name)
        print(f"Rollback complete")

# Usage
from azure.search.documents.indexes.models import SearchIndex, SearchField, SearchFieldDataType

blue_green = BlueGreenDeployment(
    index_client=alias_manager.index_client,
    alias_manager=alias_manager
)

# Define new index (with improvements)
new_index = SearchIndex(
    name="products-v2",  # Will be renamed during deployment
    fields=[
        SearchField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="description", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="price", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        # New field in v2
        SearchField(name="rating", type=SearchFieldDataType.Double, filterable=True, sortable=True)
    ]
)

# Data population function
def populate_index(index_name):
    """Populate index with data."""
    from azure.search.documents import SearchClient
    
    search_client = SearchClient(
        endpoint=os.getenv("SEARCH_ENDPOINT"),
        index_name=index_name,
        credential=AzureKeyCredential(os.getenv("SEARCH_API_KEY"))
    )
    
    # Sample documents
    documents = [
        {"id": "1", "title": "Product A", "description": "Description A", "price": 99.99, "rating": 4.5},
        {"id": "2", "title": "Product B", "description": "Description B", "price": 149.99, "rating": 4.8},
        {"id": "3", "title": "Product C", "description": "Description C", "price": 79.99, "rating": 4.2}
    ]
    
    result = search_client.upload_documents(documents)
    print(f"  Uploaded {len([r for r in result if r.succeeded])} documents")

# Deploy new version
# blue_green.deploy_new_version(
#     alias_name="products-prod",
#     new_index_definition=new_index,
#     populate_data_func=populate_index
# )

# If needed, rollback
# blue_green.rollback("products-prod", "products-v1-20231105")
```

---

## Versioning Strategies

### Version Naming Conventions

```python
class IndexVersioning:
    """Index versioning strategies."""
    
    @staticmethod
    def naming_conventions():
        """
        Recommended naming patterns:
        
        1. Semantic versioning: products-v1, products-v2, products-v3
        2. Date-based: products-20231105, products-20231112
        3. Combined: products-v1-20231105
        4. Feature-based: products-semantic-v1, products-hybrid-v2
        """
        return {
            'semantic': {
                'pattern': '{base}-v{major}.{minor}',
                'example': 'products-v1.0, products-v1.1, products-v2.0',
                'use_case': 'Track major/minor changes',
                'pros': 'Clear change magnitude',
                'cons': 'Manual version tracking'
            },
            'date_based': {
                'pattern': '{base}-{YYYYMMDD}',
                'example': 'products-20231105',
                'use_case': 'Daily/weekly deployments',
                'pros': 'Automatic chronological ordering',
                'cons': 'No semantic meaning'
            },
            'combined': {
                'pattern': '{base}-v{version}-{YYYYMMDD}',
                'example': 'products-v2-20231105',
                'use_case': 'Production deployments',
                'pros': 'Version + timestamp',
                'cons': 'Longer names'
            },
            'feature_based': {
                'pattern': '{base}-{feature}-v{version}',
                'example': 'products-semantic-v1, products-hybrid-v2',
                'use_case': 'Multiple parallel variants',
                'pros': 'Self-documenting',
                'cons': 'Complex namespace'
            }
        }
    
    @staticmethod
    def version_metadata():
        """
        Store version metadata for tracking.
        """
        return {
            'index_name': 'products-v2-20231105',
            'version': '2.0',
            'created_date': '2023-11-05T10:00:00Z',
            'created_by': 'deployment-pipeline',
            'changes': [
                'Added rating field',
                'Updated scoring profile',
                'Improved analyzer configuration'
            ],
            'previous_version': 'products-v1-20231025',
            'rollback_available': True,
            'status': 'active'
        }

# Usage
versioning = IndexVersioning()

conventions = versioning.naming_conventions()
print("Index Naming Conventions:")
for conv_type, details in conventions.items():
    print(f"\n{conv_type.upper()}:")
    print(f"  Pattern: {details['pattern']}")
    print(f"  Example: {details['example']}")
    print(f"  Use case: {details['use_case']}")
    print(f"  Pros: {details['pros']}")
    print(f"  Cons: {details['cons']}")

metadata = versioning.version_metadata()
print(f"\n\nExample Version Metadata:")
print(f"Index: {metadata['index_name']}")
print(f"Version: {metadata['version']}")
print(f"Changes:")
for change in metadata['changes']:
    print(f"  • {change}")
```

### Version Management

```python
class VersionManager:
    """Manage index versions and lifecycle."""
    
    def __init__(self, index_client):
        self.index_client = index_client
    
    def list_index_versions(self, base_name):
        """
        List all versions of an index.
        
        Finds all indexes matching base name pattern.
        """
        import re
        
        all_indexes = self.index_client.list_indexes()
        versions = []
        
        pattern = re.compile(f"^{re.escape(base_name)}-v(\\d+)")
        
        for index in all_indexes:
            match = pattern.match(index.name)
            if match:
                version_num = int(match.group(1))
                stats = self.index_client.get_index_statistics(index.name)
                
                versions.append({
                    'name': index.name,
                    'version': version_num,
                    'document_count': stats.document_count,
                    'storage_size': stats.storage_size
                })
        
        # Sort by version number
        versions.sort(key=lambda x: x['version'], reverse=True)
        return versions
    
    def cleanup_old_versions(self, base_name, keep_count=3):
        """
        Delete old index versions, keeping most recent N versions.
        
        Safety: Never delete index currently pointed to by alias.
        """
        versions = self.list_index_versions(base_name)
        
        if len(versions) <= keep_count:
            print(f"Only {len(versions)} versions exist, nothing to clean up")
            return
        
        # Get active aliases
        active_indexes = self._get_aliased_indexes()
        
        # Delete old versions
        to_delete = versions[keep_count:]
        deleted_count = 0
        
        for version in to_delete:
            if version['name'] in active_indexes:
                print(f"Skipping {version['name']} (in use by alias)")
                continue
            
            print(f"Deleting {version['name']} (version {version['version']})")
            self.index_client.delete_index(version['name'])
            deleted_count += 1
        
        print(f"Deleted {deleted_count} old versions")
        return deleted_count
    
    def _get_aliased_indexes(self):
        """Get set of index names currently pointed to by aliases."""
        aliases = self.index_client.list_aliases()
        aliased_indexes = set()
        
        for alias in aliases:
            if alias.indexes:
                aliased_indexes.add(alias.indexes[0])
        
        return aliased_indexes
    
    def get_version_info(self, index_name):
        """Get detailed information about an index version."""
        try:
            index = self.index_client.get_index(index_name)
            stats = self.index_client.get_index_statistics(index_name)
            
            return {
                'name': index.name,
                'fields': len(index.fields),
                'scoring_profiles': len(index.scoring_profiles) if index.scoring_profiles else 0,
                'document_count': stats.document_count,
                'storage_size_bytes': stats.storage_size,
                'storage_size_mb': stats.storage_size / (1024 * 1024),
                'analyzers': len(index.analyzers) if index.analyzers else 0
            }
        except Exception as e:
            print(f"Error getting index info: {e}")
            return None

# Usage
version_manager = VersionManager(alias_manager.index_client)

# List versions
versions = version_manager.list_index_versions("products")
print("Index Versions:")
for v in versions:
    print(f"  v{v['version']}: {v['name']} ({v['document_count']:,} docs, {v['storage_size']:,} bytes)")

# Get detailed info
info = version_manager.get_version_info("products-v2-20231105")
if info:
    print(f"\nVersion Details:")
    print(f"  Name: {info['name']}")
    print(f"  Fields: {info['fields']}")
    print(f"  Documents: {info['document_count']:,}")
    print(f"  Storage: {info['storage_size_mb']:.2f} MB")

# Cleanup old versions (keep 3 most recent)
# version_manager.cleanup_old_versions("products", keep_count=3)
```

---

## Schema Changes

### Safe Schema Updates

```python
class SchemaUpdater:
    """Handle schema updates safely."""
    
    def __init__(self, index_client):
        self.index_client = index_client
    
    def add_field(self, index_name, new_field):
        """
        Add new field to existing index (no rebuild required).
        
        This is a safe operation that doesn't require reindexing.
        """
        # Get current index
        index = self.index_client.get_index(index_name)
        
        # Add new field
        index.fields.append(new_field)
        
        # Update index
        result = self.index_client.create_or_update_index(index)
        print(f"Added field '{new_field.name}' to index '{index_name}'")
        return result
    
    def update_scoring_profile(self, index_name, new_scoring_profile):
        """
        Update or add scoring profile (no rebuild required).
        """
        index = self.index_client.get_index(index_name)
        
        # Find and update existing profile, or add new one
        profile_updated = False
        if index.scoring_profiles:
            for i, profile in enumerate(index.scoring_profiles):
                if profile.name == new_scoring_profile.name:
                    index.scoring_profiles[i] = new_scoring_profile
                    profile_updated = True
                    break
        
        if not profile_updated:
            if not index.scoring_profiles:
                index.scoring_profiles = []
            index.scoring_profiles.append(new_scoring_profile)
        
        result = self.index_client.create_or_update_index(index)
        action = "Updated" if profile_updated else "Added"
        print(f"{action} scoring profile '{new_scoring_profile.name}'")
        return result
    
    def update_synonym_map(self, synonym_map_name, new_synonyms):
        """
        Update synonym map (no rebuild required).
        
        Changes take effect immediately for new queries.
        """
        from azure.search.documents.indexes.models import SynonymMap
        
        synonym_map = SynonymMap(
            name=synonym_map_name,
            synonyms="\n".join(new_synonyms)
        )
        
        result = self.index_client.create_or_update_synonym_map(synonym_map)
        print(f"Updated synonym map '{synonym_map_name}'")
        return result
    
    def validate_schema_change(self, index_name, new_index_definition):
        """
        Validate if schema change requires rebuild.
        
        Returns:
            (requires_rebuild, breaking_changes)
        """
        current_index = self.index_client.get_index(index_name)
        
        breaking_changes = []
        
        # Check for field modifications
        current_fields = {f.name: f for f in current_index.fields}
        new_fields = {f.name: f for f in new_index_definition.fields}
        
        for field_name, current_field in current_fields.items():
            if field_name in new_fields:
                new_field = new_fields[field_name]
                
                # Check for breaking changes
                if current_field.type != new_field.type:
                    breaking_changes.append(f"Field '{field_name}' type changed")
                
                if hasattr(current_field, 'analyzer') and hasattr(new_field, 'analyzer'):
                    if current_field.analyzer != new_field.analyzer:
                        breaking_changes.append(f"Field '{field_name}' analyzer changed")
            else:
                breaking_changes.append(f"Field '{field_name}' removed")
        
        requires_rebuild = len(breaking_changes) > 0
        
        return requires_rebuild, breaking_changes

# Usage
schema_updater = SchemaUpdater(alias_manager.index_client)

# Add new field (safe operation)
new_field = SearchField(
    name="reviewCount",
    type=SearchFieldDataType.Int32,
    filterable=True,
    sortable=True
)

# schema_updater.add_field("products-v2", new_field)

# Update scoring profile (safe operation)
from azure.search.documents.indexes.models import (
    ScoringProfile,
    TextWeights,
    MagnitudeScoringFunction,
    MagnitudeScoringParameters,
    ScoringFunctionInterpolation,
    ScoringFunctionAggregation
)

updated_profile = ScoringProfile(
    name="relevance",
    text_weights=TextWeights(weights={"title": 3.0, "description": 1.5}),
    functions=[
        MagnitudeScoringFunction(
            field_name="rating",
            boost=2.0,
            parameters=MagnitudeScoringParameters(
                boosting_range_start=4.0,
                boosting_range_end=5.0,
                constant_boost_beyond_range=False
            ),
            interpolation=ScoringFunctionInterpolation.LINEAR
        )
    ],
    function_aggregation=ScoringFunctionAggregation.SUM
)

# schema_updater.update_scoring_profile("products-v2", updated_profile)

# Validate schema change
new_definition = SearchIndex(
    name="products-v2",
    fields=[
        SearchField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
        # Changed type - breaking change!
        SearchField(name="price", type=SearchFieldDataType.Int32, filterable=True)
    ]
)

requires_rebuild, changes = schema_updater.validate_schema_change(
    "products-v2",
    new_definition
)

print(f"Requires rebuild: {requires_rebuild}")
if changes:
    print("Breaking changes:")
    for change in changes:
        print(f"  • {change}")
```

---

## Lifecycle Management

### Index Lifecycle Automation

```python
class IndexLifecycleManager:
    """Automate index lifecycle management."""
    
    def __init__(self, index_client):
        self.index_client = index_client
    
    def get_index_age(self, index_name):
        """
        Calculate index age based on name or creation date.
        
        Assumes date-based naming: products-v1-20231105
        """
        import re
        from datetime import datetime, timedelta
        
        # Try to extract date from name
        date_match = re.search(r'(\d{8})$', index_name)
        if date_match:
            date_str = date_match.group(1)
            try:
                created_date = datetime.strptime(date_str, '%Y%m%d')
                age = datetime.now() - created_date
                return age.days
            except ValueError:
                pass
        
        return None
    
    def archive_old_indexes(self, base_name, max_age_days=90):
        """
        Identify indexes older than max_age_days.
        
        Note: Azure AI Search doesn't support archival,
        so this identifies candidates for deletion or export.
        """
        versions = self._list_all_versions(base_name)
        old_indexes = []
        
        for version in versions:
            age = self.get_index_age(version['name'])
            if age and age > max_age_days:
                old_indexes.append({
                    'name': version['name'],
                    'age_days': age,
                    'document_count': version['document_count']
                })
        
        return old_indexes
    
    def _list_all_versions(self, base_name):
        """List all versions of a base index."""
        import re
        
        all_indexes = self.index_client.list_indexes()
        versions = []
        
        for index in all_indexes:
            if index.name.startswith(base_name):
                stats = self.index_client.get_index_statistics(index.name)
                versions.append({
                    'name': index.name,
                    'document_count': stats.document_count,
                    'storage_size': stats.storage_size
                })
        
        return versions
    
    def export_index_schema(self, index_name, output_path):
        """
        Export index schema to JSON file for backup/versioning.
        """
        import json
        
        index = self.index_client.get_index(index_name)
        
        # Convert to serializable format
        schema = {
            'name': index.name,
            'fields': [
                {
                    'name': f.name,
                    'type': str(f.type),
                    'searchable': getattr(f, 'searchable', None),
                    'filterable': getattr(f, 'filterable', None),
                    'sortable': getattr(f, 'sortable', None),
                    'facetable': getattr(f, 'facetable', None),
                    'key': getattr(f, 'key', None)
                }
                for f in index.fields
            ],
            'scoring_profiles': [
                {'name': p.name}
                for p in (index.scoring_profiles or [])
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)
        
        print(f"Exported schema for '{index_name}' to {output_path}")
        return schema

# Usage
lifecycle_manager = IndexLifecycleManager(alias_manager.index_client)

# Get index age
age = lifecycle_manager.get_index_age("products-v1-20231025")
if age:
    print(f"Index age: {age} days")

# Find old indexes
old_indexes = lifecycle_manager.archive_old_indexes("products", max_age_days=30)
print(f"\nIndexes older than 30 days:")
for idx in old_indexes:
    print(f"  {idx['name']} ({idx['age_days']} days, {idx['document_count']:,} docs)")

# Export schema
# lifecycle_manager.export_index_schema(
#     "products-v2-20231105",
#     "schemas/products-v2-schema.json"
# )
```

---

## Capacity Planning

### Document and Storage Limits

```python
class CapacityPlanner:
    """Plan index capacity and scaling."""
    
    @staticmethod
    def tier_limits():
        """
        Azure AI Search tier limits.
        
        Returns document and storage limits by tier.
        """
        return {
            'free': {
                'max_indexes': 3,
                'max_documents_per_index': 10000,
                'max_storage_mb': 50,
                'max_replicas': 0,
                'max_partitions': 0
            },
            'basic': {
                'max_indexes': 15,
                'max_documents_per_index': 1000000,
                'max_storage_gb': 2,
                'max_replicas': 3,
                'max_partitions': 1
            },
            's1': {
                'max_indexes': 50,
                'max_documents_per_index': 15000000,
                'max_storage_gb': 25,
                'max_replicas': 12,
                'max_partitions': 12
            },
            's2': {
                'max_indexes': 200,
                'max_documents_per_index': 60000000,
                'max_storage_gb': 100,
                'max_replicas': 12,
                'max_partitions': 12
            },
            's3': {
                'max_indexes': 200,
                'max_documents_per_index': 60000000,
                'max_storage_gb': 200,  # Per partition
                'max_replicas': 12,
                'max_partitions': 12
            }
        }
    
    @staticmethod
    def calculate_storage_requirements(avg_doc_size_kb, num_documents, replicas=1):
        """
        Calculate storage requirements.
        
        Args:
            avg_doc_size_kb: Average document size in KB
            num_documents: Number of documents
            replicas: Number of replicas (1 = no replicas)
        
        Returns:
            Storage needed in GB
        """
        # Calculate base storage
        total_kb = avg_doc_size_kb * num_documents
        total_gb = total_kb / (1024 * 1024)
        
        # Account for replicas
        total_with_replicas = total_gb * replicas
        
        # Add 20% overhead for indexes, metadata
        with_overhead = total_with_replicas * 1.2
        
        return {
            'base_storage_gb': total_gb,
            'with_replicas_gb': total_with_replicas,
            'with_overhead_gb': with_overhead,
            'recommended_tier': CapacityPlanner._recommend_tier(with_overhead, num_documents)
        }
    
    @staticmethod
    def _recommend_tier(storage_gb, num_documents):
        """Recommend tier based on storage and document count."""
        limits = CapacityPlanner.tier_limits()
        
        if num_documents <= limits['free']['max_documents_per_index'] and \
           storage_gb <= limits['free']['max_storage_mb'] / 1024:
            return 'free'
        
        if num_documents <= limits['basic']['max_documents_per_index'] and \
           storage_gb <= limits['basic']['max_storage_gb']:
            return 'basic'
        
        if num_documents <= limits['s1']['max_documents_per_index'] and \
           storage_gb <= limits['s1']['max_storage_gb']:
            return 's1'
        
        if num_documents <= limits['s2']['max_documents_per_index'] and \
           storage_gb <= limits['s2']['max_storage_gb']:
            return 's2'
        
        return 's3'
    
    @staticmethod
    def calculate_qps_capacity(tier, replicas):
        """
        Estimate QPS (queries per second) capacity.
        
        Approximate QPS limits by tier and replica count.
        """
        base_qps = {
            'free': 3,
            'basic': 15,
            's1': 60,
            's2': 120,
            's3': 240
        }
        
        tier_qps = base_qps.get(tier.lower(), 15)
        
        # Each replica adds capacity
        # Not perfectly linear, but approximate
        total_qps = tier_qps * max(1, replicas)
        
        return {
            'tier': tier,
            'replicas': replicas,
            'estimated_qps': total_qps,
            'recommended_max_qps': total_qps * 0.8  # 80% of max for safety
        }

# Usage
planner = CapacityPlanner()

# Get tier limits
limits = planner.tier_limits()
print("Tier Limits:")
for tier, limits_data in limits.items():
    print(f"\n{tier.upper()}:")
    print(f"  Max indexes: {limits_data['max_indexes']}")
    print(f"  Max documents: {limits_data['max_documents_per_index']:,}")

# Calculate storage requirements
storage_req = planner.calculate_storage_requirements(
    avg_doc_size_kb=10,  # 10 KB per document
    num_documents=5000000,  # 5 million documents
    replicas=3  # 3 replicas for HA
)

print("\n\nStorage Requirements (5M docs, 10KB avg, 3 replicas):")
print(f"  Base storage: {storage_req['base_storage_gb']:.2f} GB")
print(f"  With replicas: {storage_req['with_replicas_gb']:.2f} GB")
print(f"  With overhead: {storage_req['with_overhead_gb']:.2f} GB")
print(f"  Recommended tier: {storage_req['recommended_tier'].upper()}")

# Calculate QPS capacity
qps = planner.calculate_qps_capacity('s1', replicas=3)
print(f"\n\nQPS Capacity (S1 tier, 3 replicas):")
print(f"  Estimated max QPS: {qps['estimated_qps']}")
print(f"  Recommended target: {qps['recommended_max_qps']:.0f} QPS")
```

---

## Performance Optimization

### Replica and Partition Management

```python
class ReplicaPartitionManager:
    """Manage replicas and partitions for performance."""
    
    @staticmethod
    def replica_guidelines():
        """
        Replica guidelines:
        
        Replicas provide:
        - High availability (2+ replicas for 99.9% SLA)
        - Query throughput (each replica handles queries)
        - Read scalability
        """
        return {
            'minimum_for_ha': {
                'replicas': 2,
                'reason': 'Achieve 99.9% SLA',
                'cost_multiplier': '2x'
            },
            'high_query_volume': {
                'replicas': '3-6',
                'reason': 'Scale query throughput',
                'qps_increase': 'Linear with replicas'
            },
            'global_distribution': {
                'replicas': '3+',
                'reason': 'Reduce latency for distributed users',
                'note': 'Consider multi-region instead'
            }
        }
    
    @staticmethod
    def partition_guidelines():
        """
        Partition guidelines:
        
        Partitions provide:
        - Horizontal storage scaling
        - Increased indexing throughput
        - Parallel query execution
        """
        return {
            'large_index': {
                'partitions': '2-12',
                'reason': 'Exceed storage limit of single partition',
                'note': 'S3: 200GB per partition, S2: 100GB per partition'
            },
            'high_indexing_volume': {
                'partitions': '2-6',
                'reason': 'Parallelize document ingestion',
                'throughput_increase': 'Near-linear with partitions'
            },
            'complex_queries': {
                'partitions': '2-4',
                'reason': 'Parallel query execution',
                'note': 'Diminishing returns beyond 4 partitions'
            }
        }
    
    @staticmethod
    def calculate_optimal_replicas(target_qps, avg_query_latency_ms=50):
        """
        Calculate optimal replica count for target QPS.
        
        Assumptions:
        - Single replica handles ~15-60 QPS (tier-dependent)
        - Query latency ~50ms average
        """
        # Conservative estimate: 1 replica = 30 QPS
        qps_per_replica = 1000 / avg_query_latency_ms
        
        required_replicas = max(1, int(target_qps / qps_per_replica) + 1)
        
        # Cap at 12 (Azure limit)
        optimal_replicas = min(required_replicas, 12)
        
        return {
            'target_qps': target_qps,
            'qps_per_replica': qps_per_replica,
            'required_replicas': required_replicas,
            'optimal_replicas': optimal_replicas,
            'actual_capacity_qps': optimal_replicas * qps_per_replica
        }

# Usage
replica_mgr = ReplicaPartitionManager()

# Get guidelines
replica_guide = replica_mgr.replica_guidelines()
print("Replica Guidelines:")
for scenario, guide in replica_guide.items():
    print(f"\n{scenario}:")
    print(f"  Replicas: {guide['replicas']}")
    print(f"  Reason: {guide['reason']}")

partition_guide = replica_mgr.partition_guidelines()
print("\n\nPartition Guidelines:")
for scenario, guide in partition_guide.items():
    print(f"\n{scenario}:")
    print(f"  Partitions: {guide['partitions']}")
    print(f"  Reason: {guide['reason']}")

# Calculate optimal replicas for 500 QPS
replica_calc = replica_mgr.calculate_optimal_replicas(target_qps=500)
print("\n\nOptimal Replicas for 500 QPS:")
print(f"  Required: {replica_calc['required_replicas']}")
print(f"  Recommended: {replica_calc['optimal_replicas']}")
print(f"  Actual capacity: {replica_calc['actual_capacity_qps']:.0f} QPS")
```

---

## Best Practices

### 1. Always Use Aliases for Production Indexes

**Why It Matters**

Direct index references create deployment coupling and eliminate rollback capability. If applications point to `products-v2-20240513`, you cannot switch to a different index without code deployment. Aliases provide abstraction layer enabling instant cutover.

**Implementation**

```python
class AliasBasedDeployment:
    """Best practice: Never reference indexes directly."""
    
    def __init__(self, search_endpoint, admin_key):
        from azure.search.documents.indexes import SearchIndexClient
        from azure.core.credentials import AzureKeyCredential
        
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def setup_production_alias(self, alias_name, initial_index):
        """
        Setup production alias (one-time operation).
        
        After this, all apps use alias name, never index name.
        """
        from azure.search.documents.indexes.models import SearchAlias
        
        alias = SearchAlias(name=alias_name, indexes=[initial_index])
        self.index_client.create_or_update_alias(alias)
        
        print(f"✓ Created alias '{alias_name}' → '{initial_index}'")
        print(f"✓ Update all application code to use '{alias_name}'")
        print(f"✓ Never use '{initial_index}' directly in app code")
        
        return alias

# Application code (GOOD)
search_client = SearchClient(
    endpoint="https://mysearch.search.windows.net",
    index_name="products-prod",  # ✓ Alias name
    credential=credential
)

# Application code (BAD)
search_client = SearchClient(
    endpoint="https://mysearch.search.windows.net",
    index_name="products-v2-20240513",  # ✗ Direct index reference
    credential=credential
)
```

**Benefits**:
- **Instant cutover**: Update alias, all traffic switches immediately (atomic operation)
- **Zero code deployment**: Switch indexes without redeploying applications
- **Easy rollback**: Point alias back to previous index in <5 seconds
- **A/B testing**: Application-level routing can split traffic between indexes

**When to Create Aliases**:
- ✅ Before first production deployment
- ✅ For every environment (dev-alias, staging-alias, prod-alias)
- ✅ For any index that serves user-facing traffic

---

### 2. Test Schema Changes in Non-Production First

**The Validation Pipeline**

Never deploy schema changes directly to production. Follow this progression:

```python
class SchemaChangeValidation:
    """Multi-environment schema validation pipeline."""
    
    def __init__(self):
        self.environments = ['dev', 'staging', 'production']
    
    def deploy_schema_change(self, new_schema):
        """
        Deploy schema through validation pipeline.
        
        Stages:
        1. Dev: Initial testing, rapid iteration
        2. Staging: Full validation suite, load testing
        3. Production: Gradual rollout with monitoring
        """
        results = {}
        
        for env in self.environments:
            print(f"\n{'='*60}")
            print(f"Deploying to {env.upper()}")
            print(f"{'='*60}")
            
            if env == 'dev':
                # Dev: Quick smoke test
                results[env] = self._dev_validation(new_schema)
            elif env == 'staging':
                # Staging: Comprehensive validation
                results[env] = self._staging_validation(new_schema)
            elif env == 'production':
                # Production: Gradual rollout
                results[env] = self._production_deployment(new_schema)
            
            if not results[env]['passed']:
                print(f"✗ Deployment failed in {env}")
                print(f"  Stopping pipeline. Fix issues before proceeding.")
                return results
        
        return results
    
    def _dev_validation(self, schema):
        """Dev environment validation."""
        checks = {
            'schema_valid': self._validate_schema_syntax(schema),
            'sample_query': self._test_sample_queries(schema, count=10),
            'data_loads': self._test_data_loading(schema, doc_count=1000)
        }
        
        passed = all(checks.values())
        
        print(f"Dev Validation:")
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}")
        
        return {'passed': passed, 'checks': checks}
    
    def _staging_validation(self, schema):
        """Staging environment validation."""
        checks = {
            'full_data_load': self._load_production_sample(schema),
            'query_suite': self._run_query_suite(schema, count=1000),
            'performance': self._benchmark_queries(schema),
            'relevance': self._measure_relevance_metrics(schema),
            'load_test': self._run_load_test(schema, qps=50, duration_min=10)
        }
        
        passed = all(checks.values())
        
        print(f"Staging Validation:")
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}")
        
        return {'passed': passed, 'checks': checks}
    
    def _production_deployment(self, schema):
        """Production deployment with gradual rollout."""
        print("Production Deployment:")
        
        # 10% canary
        print("  1. Deploy to 10% of traffic...")
        canary_result = self._deploy_canary(schema, percent=10)
        if not canary_result['passed']:
            return canary_result
        
        # 50% rollout
        print("  2. Increase to 50% of traffic...")
        half_result = self._deploy_canary(schema, percent=50)
        if not half_result['passed']:
            return half_result
        
        # 100% rollout
        print("  3. Full cutover to 100%...")
        full_result = self._deploy_full(schema)
        
        return full_result
    
    def _validate_schema_syntax(self, schema):
        """Check schema is syntactically valid."""
        # Implementation would validate schema structure
        return True
    
    def _test_sample_queries(self, schema, count):
        """Run sample queries."""
        return True
    
    def _test_data_loading(self, schema, doc_count):
        """Test loading sample data."""
        return True
    
    def _load_production_sample(self, schema):
        """Load representative production data."""
        return True
    
    def _run_query_suite(self, schema, count):
        """Run comprehensive query suite."""
        return True
    
    def _benchmark_queries(self, schema):
        """Measure query performance."""
        return True
    
    def _measure_relevance_metrics(self, schema):
        """Calculate precision, NDCG, etc."""
        return True
    
    def _run_load_test(self, schema, qps, duration_min):
        """Load test at target QPS."""
        return True
    
    def _deploy_canary(self, schema, percent):
        """Deploy to percentage of traffic."""
        return {'passed': True, 'metrics': {}}
    
    def _deploy_full(self, schema):
        """Deploy to 100% of traffic."""
        return {'passed': True, 'metrics': {}}
```

**Validation Checklist**:

**Dev Environment**:
- ✓ Schema syntax valid
- ✓ Sample queries work (10-20 queries)
- ✓ Data loads successfully (1K-10K documents)
- ✓ Basic functionality confirmed

**Staging Environment**:
- ✓ Full production-scale data (or representative sample)
- ✓ Comprehensive query suite (1,000+ queries with ground truth)
- ✓ Performance benchmarks meet SLA (latency, throughput)
- ✓ Relevance metrics exceed baseline (precision, NDCG)
- ✓ Load testing at production QPS for 10+ minutes
- ✓ All edge cases tested (special characters, empty fields, etc.)

**Production Environment**:
- ✓ 10% canary for 24-48 hours with metrics monitoring
- ✓ 50% rollout for 24-48 hours if canary successful
- ✓ 100% cutover after validation
- ✓ Previous version kept for 7-day rollback window

---

### 3. Keep 2-3 Recent Versions for Quick Rollback

**Version Retention Strategy**

```python
class VersionRetentionManager:
    """Manage index version lifecycle with retention policy."""
    
    def __init__(self, index_client):
        self.index_client = index_client
        self.retention_policy = {
            'active': 1,           # Current production index
            'rollback': 2,         # Keep 2 previous versions for rollback
            'archive_days': 30     # Delete versions >30 days old after leaving rollback pool
        }
    
    def enforce_retention_policy(self, base_name, current_alias):
        """
        Enforce retention policy for index versions.
        
        Strategy:
        1. Keep current production index (pointed to by alias)
        2. Keep 2 most recent previous versions (rollback capability)
        3. Delete older versions after retention period
        """
        import re
        from datetime import datetime, timedelta
        
        # Get all versions
        all_indexes = list(self.index_client.list_indexes())
        versions = []
        
        pattern = re.compile(f"^{re.escape(base_name)}-v(\\d+)")
        
        for index in all_indexes:
            match = pattern.match(index.name)
            if match:
                # Extract date from name (if exists)
                date_match = re.search(r'(\\d{8})$', index.name)
                created_date = None
                if date_match:
                    try:
                        created_date = datetime.strptime(date_match.group(1), '%Y%m%d')
                    except ValueError:
                        pass
                
                versions.append({
                    'name': index.name,
                    'version': int(match.group(1)),
                    'created_date': created_date
                })
        
        # Sort by version (newest first)
        versions.sort(key=lambda x: x['version'], reverse=True)
        
        # Get current production index
        from azure.search.documents.indexes.models import SearchAlias
        try:
            alias = self.index_client.get_alias(current_alias)
            current_index = alias.indexes[0] if alias.indexes else None
        except:
            current_index = None
        
        # Determine what to keep and delete
        to_keep = set()
        to_delete = []
        
        # Always keep current production
        if current_index:
            to_keep.add(current_index)
        
        # Keep N most recent rollback versions
        rollback_count = 0
        for v in versions:
            if v['name'] == current_index:
                continue  # Already in to_keep
            
            if rollback_count < self.retention_policy['rollback']:
                to_keep.add(v['name'])
                rollback_count += 1
            else:
                # Check if old enough to delete
                if v['created_date']:
                    age = datetime.now() - v['created_date']
                    if age.days > self.retention_policy['archive_days']:
                        to_delete.append(v)
        
        # Execute deletion
        print(f"Version Retention Policy:")
        print(f"  Current production: {current_index}")
        print(f"  Keeping for rollback: {rollback_count} versions")
        print(f"  To keep: {', '.join(to_keep)}")
        
        if to_delete:
            print(f"\n  Deleting {len(to_delete)} old versions:")
            for v in to_delete:
                age_days = (datetime.now() - v['created_date']).days if v['created_date'] else '?'
                print(f"    • {v['name']} (age: {age_days} days)")
                # Uncomment to actually delete:
                # self.index_client.delete_index(v['name'])
        else:
            print(f"  No versions to delete (all within retention window)")
        
        return {
            'kept': list(to_keep),
            'deleted': [v['name'] for v in to_delete]
        }

# Usage
retention_mgr = VersionRetentionManager(index_client)

# Run retention policy enforcement (e.g., daily cron job)
result = retention_mgr.enforce_retention_policy(
    base_name="products",
    current_alias="products-prod"
)
```

**Retention Guidelines**:

| Scenario | Active | Rollback Versions | Archive Period |
|----------|--------|-------------------|----------------|
| **Low-risk index** (rarely changes) | 1 | 1 | 14 days |
| **Standard index** (weekly/monthly updates) | 1 | 2 | 30 days |
| **High-change index** (daily updates) | 1 | 3 | 7 days |
| **Critical index** (needs audit trail) | 1 | 5 | 90 days |

**Storage Cost Consideration**:
- S1 tier: $250/month for 25GB
- 8M doc index at 10KB avg = ~80GB
- 3 versions = 240GB = ~$2,400/month
- Keeping unnecessary old versions wastes ~$800/month per version

---

### 4. Document Version Changes in Metadata

**Version Documentation Pattern**

```python
class IndexVersionDocumentation:
    """Document index version changes for audit and rollback."""
    
    def __init__(self):
        self.version_log = []
    
    def create_version_record(self, version_info):
        """
        Create comprehensive version record.
        
        Includes:
        - Version identifier and timestamp
        - Schema changes (what changed and why)
        - Performance impact
        - Rollback procedure
        - Deployment metadata
        """
        record = {
            'version': {
                'number': version_info['version_number'],
                'index_name': version_info['index_name'],
                'deployed_date': version_info['deployed_date'],
                'deployed_by': version_info['deployed_by'],
                'deployment_duration_min': version_info['deployment_duration']
            },
            'changes': {
                'schema': version_info.get('schema_changes', []),
                'scoring_profiles': version_info.get('scoring_changes', []),
                'analyzers': version_info.get('analyzer_changes', []),
                'breaking_changes': version_info.get('breaking_changes', [])
            },
            'testing': {
                'test_queries_run': version_info.get('test_query_count', 0),
                'validation_passed': version_info.get('validation_passed', False),
                'staging_duration_hours': version_info.get('staging_hours', 0)
            },
            'performance': {
                'baseline_metrics': version_info.get('baseline_metrics', {}),
                'new_metrics': version_info.get('new_metrics', {}),
                'comparison': version_info.get('metric_comparison', {})
            },
            'rollback': {
                'previous_version': version_info.get('previous_version'),
                'rollback_procedure': version_info.get('rollback_procedure', ''),
                'rollback_window_days': version_info.get('rollback_window', 7)
            },
            'notes': version_info.get('notes', '')
        }
        
        self.version_log.append(record)
        return record
    
    def export_version_log(self, output_file):
        """Export version log to JSON for version control."""
        import json
        from datetime import datetime
        
        with open(output_file, 'w') as f:
            json.dump({
                'export_date': datetime.now().isoformat(),
                'versions': self.version_log
            }, f, indent=2)
        
        print(f"Exported version log to {output_file}")

# Usage example
doc_mgr = IndexVersionDocumentation()

# Document v2 deployment
v2_record = doc_mgr.create_version_record({
    'version_number': '2.0',
    'index_name': 'products-v2-20240513',
    'deployed_date': '2024-05-13T14:30:00Z',
    'deployed_by': 'jane.doe@company.com',
    'deployment_duration': 6,
    'schema_changes': [
        'Added vector field (embeddings, 128-dim)',
        'Changed price field from String to Double',
        'Added rating field (Double, filterable, sortable)'
    ],
    'analyzer_changes': [
        'Updated product_name analyzer: standard → custom_product_analyzer',
        'Added synonym map for brand names'
    ],
    'breaking_changes': [
        'price field type change (requires application update for range filters)'
    ],
    'test_query_count': 2847,
    'validation_passed': True,
    'staging_hours': 72,
    'baseline_metrics': {
        'precision_at_5': 0.61,
        'ndcg_at_10': 0.65,
        'zero_result_rate': 0.084
    },
    'new_metrics': {
        'precision_at_5': 0.72,
        'ndcg_at_10': 0.78,
        'zero_result_rate': 0.028
    },
    'metric_comparison': {
        'precision_at_5': '+18%',
        'ndcg_at_10': '+20%',
        'zero_result_rate': '-67%'
    },
    'previous_version': 'products-v1-20240415',
    'rollback_procedure': 'Update alias products-prod to point to products-v1-20240415',
    'rollback_window': 7,
    'notes': 'Gradual rollout: 10% → 50% → 100% over 5 days. No issues detected.'
})

print("Version Record Created:")
print(f"  Version: {v2_record['version']['number']}")
print(f"  Index: {v2_record['version']['index_name']}")
print(f"  Changes: {len(v2_record['changes']['schema'])} schema, "
      f"{len(v2_record['changes']['analyzer_changes'])} analyzer")
print(f"  Performance: Precision@5 {v2_record['performance']['comparison']['precision_at_5']}")

# Export to version control
# doc_mgr.export_version_log('version-history/products-index-versions.json')
```

**What to Document**:
- ✓ Version number and index name
- ✓ Deployment date/time and deployer
- ✓ All schema changes (fields added/modified/removed)
- ✓ Scoring profile changes
- ✓ Analyzer modifications
- ✓ Breaking changes (require app updates)
- ✓ Testing results (query count, pass/fail)
- ✓ Performance metrics (before/after)
- ✓ Rollback procedure and window
- ✓ Any issues encountered and resolutions

**Storage Location**:
- Git repository (JSON files in `docs/version-history/`)
- Wiki or documentation system
- Deployment automation tool (e.g., Azure DevOps, GitHub Actions)

---

### 5. Automate Deployments with Blue-Green Pattern

**Automated Blue-Green Deployment**

```python
class AutomatedBlueGreenDeployment:
    """Fully automated blue-green deployment pipeline."""
    
    def __init__(self, index_client, alias_manager):
        self.index_client = index_client
        self.alias_manager = alias_manager
    
    def execute_deployment(self, config):
        """
        Execute complete automated deployment.
        
        Steps:
        1. Validate configuration
        2. Create green index
        3. Load data
        4. Run validation suite
        5. Gradual rollout (10% → 50% → 100%)
        6. Cleanup old versions
        """
        print(f"Starting automated deployment: {config['new_index_name']}")
        
        try:
            # Step 1: Validate configuration
            print("\\n[1/6] Validating configuration...")
            self._validate_config(config)
            print("  ✓ Configuration valid")
            
            # Step 2: Create green index
            print("\\n[2/6] Creating green index...")
            self._create_green_index(config)
            print(f"  ✓ Created index: {config['new_index_name']}")
            
            # Step 3: Load data
            print("\\n[3/6] Loading data...")
            self._load_data(config)
            print(f"  ✓ Loaded {config.get('document_count', '?')} documents")
            
            # Step 4: Validation
            print("\\n[4/6] Running validation suite...")
            validation_result = self._run_validation(config)
            if not validation_result['passed']:
                raise Exception(f"Validation failed: {validation_result['errors']}")
            print(f"  ✓ All {validation_result['test_count']} tests passed")
            
            # Step 5: Gradual rollout
            print("\\n[5/6] Gradual rollout...")
            self._gradual_rollout(config)
            print("  ✓ Rollout complete (100% traffic)")
            
            # Step 6: Cleanup
            print("\\n[6/6] Cleanup old versions...")
            self._cleanup_old_versions(config)
            print("  ✓ Cleanup complete")
            
            print(f"\\n{'='*60}")
            print(f"✓ Deployment successful: {config['new_index_name']}")
            print(f"{'='*60}")
            
            return {'success': True, 'index': config['new_index_name']}
            
        except Exception as e:
            print(f"\\n✗ Deployment failed: {str(e)}")
            print(f"  Rolling back...")
            self._rollback(config)
            return {'success': False, 'error': str(e)}
    
    def _validate_config(self, config):
        """Validate deployment configuration."""
        required_fields = ['new_index_name', 'alias_name', 'schema', 'data_source']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config: {field}")
    
    def _create_green_index(self, config):
        """Create new index with updated schema."""
        self.index_client.create_or_update_index(config['schema'])
    
    def _load_data(self, config):
        """Load data into green index."""
        # Implementation would call data loading function
        pass
    
    def _run_validation(self, config):
        """Run automated validation suite."""
        # Implementation would run test queries and check metrics
        return {
            'passed': True,
            'test_count': 2847,
            'errors': []
        }
    
    def _gradual_rollout(self, config):
        """Gradual rollout with monitoring."""
        import time
        
        # 10% canary
        print("  • 10% canary deployment...")
        # Implementation would use application-level routing
        time.sleep(1)  # Simulate monitoring period
        print("    ✓ 10% canary successful (24hr monitoring)")
        
        # 50% rollout
        print("  • 50% rollout...")
        time.sleep(1)
        print("    ✓ 50% rollout successful (48hr monitoring)")
        
        # 100% cutover
        print("  • 100% cutover (alias update)...")
        self.alias_manager.update_alias(
            config['alias_name'],
            config['new_index_name']
        )
        print("    ✓ Alias updated (atomic cutover)")
    
    def _cleanup_old_versions(self, config):
        """Clean up old index versions."""
        # Implementation would enforce retention policy
        pass
    
    def _rollback(self, config):
        """Rollback to previous version."""
        if 'previous_index' in config:
            self.alias_manager.update_alias(
                config['alias_name'],
                config['previous_index']
            )
            print(f"  ✓ Rolled back to {config['previous_index']}")
        
        # Delete failed green index
        try:
            self.index_client.delete_index(config['new_index_name'])
            print(f"  ✓ Deleted failed index {config['new_index_name']}")
        except:
            pass

# Usage (integrate with CI/CD pipeline)
from azure.search.documents.indexes.models import SearchIndex, SearchField, SearchFieldDataType

deployment = AutomatedBlueGreenDeployment(index_client, alias_manager)

config = {
    'new_index_name': 'products-v3-20240620',
    'alias_name': 'products-prod',
    'previous_index': 'products-v2-20240513',
    'schema': SearchIndex(
        name='products-v3-20240620',
        fields=[
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="price", type=SearchFieldDataType.Double, filterable=True),
            SearchField(name="rating", type=SearchFieldDataType.Double, filterable=True)
        ]
    ),
    'data_source': 'cosmos-db-products',
    'document_count': 8200000
}

# Execute automated deployment
# result = deployment.execute_deployment(config)
```

**Automation Checklist**:
- ✓ Schema validation (syntax, field types)
- ✓ Index creation (green index)
- ✓ Data loading (with progress monitoring)
- ✓ Automated validation suite (1,000+ test queries)
- ✓ Gradual rollout (10% → 50% → 100%)
- ✓ Monitoring integration (alerts on anomalies)
- ✓ Automatic rollback on failure
- ✓ Version cleanup (retention policy)
- ✓ Documentation generation (version records)
- ✓ Notification (Slack/email on completion)

---

### 6. Monitor Index Statistics

**Comprehensive Index Monitoring**

```python
class IndexMonitoring:
    """Monitor index health and performance metrics."""
    
    def __init__(self, index_client):
        self.index_client = index_client
    
    def collect_index_metrics(self, index_name):
        """
        Collect comprehensive index metrics.
        
        Metrics:
        - Document count
        - Storage size
        - Query performance (via Azure Monitor)
        - Indexing lag (if using indexers)
        """
        from datetime import datetime
        
        # Get index statistics
        stats = self.index_client.get_index_statistics(index_name)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'index_name': index_name,
            'document_count': stats.document_count,
            'storage_size_bytes': stats.storage_size,
            'storage_size_mb': stats.storage_size / (1024 * 1024),
            'storage_size_gb': stats.storage_size / (1024 * 1024 * 1024)
        }
        
        # Get index definition for field count
        index = self.index_client.get_index(index_name)
        metrics['field_count'] = len(index.fields)
        metrics['scoring_profiles'] = len(index.scoring_profiles) if index.scoring_profiles else 0
        
        return metrics
    
    def monitor_growth_rate(self, index_name, baseline_metrics, current_metrics):
        """
        Calculate growth rate and predict capacity needs.
        """
        import datetime as dt
        
        baseline_date = dt.datetime.fromisoformat(baseline_metrics['timestamp'])
        current_date = dt.datetime.fromisoformat(current_metrics['timestamp'])
        
        days_elapsed = (current_date - baseline_date).days
        
        if days_elapsed == 0:
            return None
        
        # Calculate growth
        doc_growth = current_metrics['document_count'] - baseline_metrics['document_count']
        storage_growth_gb = current_metrics['storage_size_gb'] - baseline_metrics['storage_size_gb']
        
        # Daily growth rate
        docs_per_day = doc_growth / days_elapsed
        gb_per_day = storage_growth_gb / days_elapsed
        
        # Project 30/90 days
        projected_30d = {
            'document_count': current_metrics['document_count'] + (docs_per_day * 30),
            'storage_gb': current_metrics['storage_size_gb'] + (gb_per_day * 30)
        }
        
        projected_90d = {
            'document_count': current_metrics['document_count'] + (docs_per_day * 90),
            'storage_gb': current_metrics['storage_size_gb'] + (gb_per_day * 90)
        }
        
        return {
            'growth_rate': {
                'documents_per_day': docs_per_day,
                'gb_per_day': gb_per_day,
                'period_days': days_elapsed
            },
            'projections': {
                '30_days': projected_30d,
                '90_days': projected_90d
            },
            'alerts': self._check_capacity_alerts(projected_30d, projected_90d)
        }
    
    def _check_capacity_alerts(self, proj_30d, proj_90d):
        """Check if approaching capacity limits."""
        alerts = []
        
        # Example limits for S2 tier
        S2_MAX_DOCS = 60000000
        S2_MAX_STORAGE_GB = 100
        
        # Check 30-day projection
        if proj_30d['document_count'] > S2_MAX_DOCS * 0.8:
            alerts.append({
                'severity': 'warning',
                'message': f"30-day projection ({proj_30d['document_count']:,.0f} docs) approaching doc limit"
            })
        
        if proj_30d['storage_gb'] > S2_MAX_STORAGE_GB * 0.8:
            alerts.append({
                'severity': 'warning',
                'message': f"30-day projection ({proj_30d['storage_gb']:.1f} GB) approaching storage limit"
            })
        
        # Check 90-day projection
        if proj_90d['document_count'] > S2_MAX_DOCS:
            alerts.append({
                'severity': 'critical',
                'message': f"90-day projection ({proj_90d['document_count']:,.0f} docs) exceeds doc limit - upgrade needed"
            })
        
        if proj_90d['storage_gb'] > S2_MAX_STORAGE_GB:
            alerts.append({
                'severity': 'critical',
                'message': f"90-day projection ({proj_90d['storage_gb']:.1f} GB) exceeds storage limit - upgrade needed"
            })
        
        return alerts

# Usage
monitor = IndexMonitoring(index_client)

# Collect current metrics
current = monitor.collect_index_metrics('products-v2-20240513')
print(f"Current Index Metrics:")
print(f"  Documents: {current['document_count']:,}")
print(f"  Storage: {current['storage_size_gb']:.2f} GB")
print(f"  Fields: {current['field_count']}")

# Monitor growth (compare to baseline from 30 days ago)
baseline = {
    'timestamp': '2024-04-13T00:00:00',
    'document_count': 7800000,
    'storage_size_gb': 76.4
}

growth = monitor.monitor_growth_rate('products-v2-20240513', baseline, current)
if growth:
    print(f"\\nGrowth Analysis:")
    print(f"  Docs/day: {growth['growth_rate']['documents_per_day']:,.0f}")
    print(f"  GB/day: {growth['growth_rate']['gb_per_day']:.3f}")
    print(f"\\n30-day projection:")
    print(f"  Documents: {growth['projections']['30_days']['document_count']:,.0f}")
    print(f"  Storage: {growth['projections']['30_days']['storage_gb']:.1f} GB")
    
    if growth['alerts']:
        print(f"\\nAlerts:")
        for alert in growth['alerts']:
            print(f"  {alert['severity'].upper()}: {alert['message']}")
```

**Monitoring Recommendations**:
- Daily: Document count, storage size
- Weekly: Growth rate analysis, capacity projections
- Monthly: Performance trends, cost analysis
- On-demand: Query latency, indexing throughput

---

### 7. Export Schemas to Version Control

**Schema Version Control**

Store index schemas in Git alongside application code to:
- Track schema evolution over time
- Enable code review for schema changes
- Automate deployments from schema files
- Recover from accidental deletions

```python
class SchemaVersionControl:
    """Export and manage schemas in version control."""
    
    def __init__(self, index_client):
        self.index_client = index_client
    
    def export_schema_to_file(self, index_name, output_path):
        """
        Export index schema to JSON file.
        
        File can be committed to Git and used for:
        - Documentation
        - Automated deployments
        - Disaster recovery
        """
        import json
        from datetime import datetime
        
        # Get index
        index = self.index_client.get_index(index_name)
        
        # Convert to serializable format
        schema = {
            'metadata': {
                'exported_date': datetime.now().isoformat(),
                'index_name': index.name,
                'exporter': 'schema_version_control'
            },
            'fields': [
                {
                    'name': f.name,
                    'type': str(f.type),
                    'key': getattr(f, 'key', False),
                    'searchable': getattr(f, 'searchable', None),
                    'filterable': getattr(f, 'filterable', None),
                    'sortable': getattr(f, 'sortable', None),
                    'facetable': getattr(f, 'facetable', None),
                    'analyzer': getattr(f, 'analyzer_name', None)
                }
                for f in index.fields
            ],
            'scoring_profiles': [
                {
                    'name': p.name,
                    'text_weights': {
                        field: weight
                        for field, weight in (p.text_weights.weights.items() if p.text_weights else {}).items()
                    }
                }
                for p in (index.scoring_profiles or [])
            ],
            'cors_options': {
                'allowed_origins': index.cors_options.allowed_origins if index.cors_options else []
            }
        }
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)
        
        print(f"Exported schema to {output_path}")
        print(f"  Fields: {len(schema['fields'])}")
        print(f"  Scoring profiles: {len(schema['scoring_profiles'])}")
        
        return schema
    
    def import_schema_from_file(self, schema_file):
        """
        Import schema from JSON file to create index.
        
        Enables infrastructure-as-code for search indexes.
        """
        import json
        from azure.search.documents.indexes.models import (
            SearchIndex, SearchField, SearchFieldDataType,
            ScoringProfile, TextWeights, CorsOptions
        )
        
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        
        # Convert to SearchIndex
        fields = []
        for f_def in schema['fields']:
            field_type = getattr(SearchFieldDataType, f_def['type'].split('.')[-1].replace('Edm.', ''))
            
            field = SearchField(
                name=f_def['name'],
                type=field_type,
                key=f_def.get('key', False),
                searchable=f_def.get('searchable'),
                filterable=f_def.get('filterable'),
                sortable=f_def.get('sortable'),
                facetable=f_def.get('facetable')
            )
            
            fields.append(field)
        
        # Create scoring profiles
        scoring_profiles = []
        for sp_def in schema.get('scoring_profiles', []):
            if sp_def.get('text_weights'):
                text_weights = TextWeights(weights=sp_def['text_weights'])
                profile = ScoringProfile(
                    name=sp_def['name'],
                    text_weights=text_weights
                )
                scoring_profiles.append(profile)
        
        # Create index
        index = SearchIndex(
            name=schema['metadata']['index_name'],
            fields=fields,
            scoring_profiles=scoring_profiles if scoring_profiles else None
        )
        
        return index

# Usage
schema_vc = SchemaVersionControl(index_client)

# Export current production schema
# schema_vc.export_schema_to_file(
#     'products-v2-20240513',
#     'schemas/products-v2-schema.json'
# )

# Commit to Git
# git add schemas/products-v2-schema.json
# git commit -m "Add products index v2 schema"
# git push

# Later: Import schema from version control
# index_definition = schema_vc.import_schema_from_file('schemas/products-v2-schema.json')
# index_client.create_or_update_index(index_definition)
```

**Version Control Structure**:
```
repo/
├── schemas/
│   ├── products-v1-schema.json
│   ├── products-v2-schema.json
│   ├── products-v3-schema.json
│   └── README.md
├── deployment/
│   ├── deploy-index.py
│   └── rollback-index.py
└── version-history/
    └── products-index-versions.json
```

---

### 8. Plan Capacity Before Hitting Limits

**Proactive Capacity Planning**

Don't wait until hitting limits. Plan capacity upgrades when reaching **80% of tier limits**.

```python
class ProactiveCapacityPlanning:
    """Plan capacity upgrades before hitting limits."""
    
    def __init__(self):
        self.tier_limits = {
            'basic': {'docs': 1000000, 'storage_gb': 2},
            's1': {'docs': 15000000, 'storage_gb': 25},
            's2': {'docs': 60000000, 'storage_gb': 100},
            's3': {'docs': 60000000, 'storage_gb': 200}  # per partition
        }
    
    def assess_capacity_need(self, current_tier, current_metrics, growth_rate):
        """
        Assess if tier upgrade needed.
        
        Triggers:
        - Current usage > 80% of limit
        - 30-day projection > 100% of limit
        - 90-day projection > 120% of limit
        """
        limits = self.tier_limits[current_tier]
        
        # Current utilization
        doc_utilization = current_metrics['document_count'] / limits['docs']
        storage_utilization = current_metrics['storage_gb'] / limits['storage_gb']
        
        # Projections
        docs_30d = current_metrics['document_count'] + (growth_rate['docs_per_day'] * 30)
        storage_30d = current_metrics['storage_gb'] + (growth_rate['gb_per_day'] * 30)
        
        docs_90d = current_metrics['document_count'] + (growth_rate['docs_per_day'] * 90)
        storage_90d = current_metrics['storage_gb'] + (growth_rate['gb_per_day'] * 90)
        
        doc_util_30d = docs_30d / limits['docs']
        storage_util_30d = storage_30d / limits['storage_gb']
        
        doc_util_90d = docs_90d / limits['docs']
        storage_util_90d = storage_90d / limits['storage_gb']
        
        # Determine urgency
        if doc_utilization > 0.8 or storage_utilization > 0.8:
            urgency = 'immediate'
            message = f"Current usage > 80% of {current_tier.upper()} limits"
        elif doc_util_30d > 1.0 or storage_util_30d > 1.0:
            urgency = 'high'
            message = f"30-day projection exceeds {current_tier.upper()} limits"
        elif doc_util_90d > 1.0 or storage_util_90d > 1.0:
            urgency = 'medium'
            message = f"90-day projection exceeds {current_tier.upper()} limits"
        else:
            urgency = 'none'
            message = f"Capacity sufficient for 90+ days"
        
        # Recommend tier
        recommended_tier = self._recommend_tier_upgrade(
            docs_90d,
            storage_90d,
            current_tier
        )
        
        return {
            'current_tier': current_tier,
            'utilization': {
                'documents': f"{doc_utilization:.1%}",
                'storage': f"{storage_utilization:.1%}"
            },
            'projections': {
                '30_days': {
                    'documents': f"{doc_util_30d:.1%}",
                    'storage': f"{storage_util_30d:.1%}"
                },
                '90_days': {
                    'documents': f"{doc_util_90d:.1%}",
                    'storage': f"{storage_util_90d:.1%}"
                }
            },
            'urgency': urgency,
            'message': message,
            'recommended_tier': recommended_tier,
            'action_needed': urgency in ['immediate', 'high']
        }
    
    def _recommend_tier_upgrade(self, projected_docs, projected_storage_gb, current_tier):
        """Recommend tier based on projections."""
        # Add 20% buffer for safety
        required_docs = projected_docs * 1.2
        required_storage = projected_storage_gb * 1.2
        
        for tier in ['basic', 's1', 's2', 's3']:
            if tier == current_tier:
                continue  # Skip current tier
            
            limits = self.tier_limits[tier]
            if required_docs <= limits['docs'] and required_storage <= limits['storage_gb']:
                return tier
        
        return 's3'  # Maximum tier

# Usage
planner = ProactiveCapacityPlanning()

assessment = planner.assess_capacity_need(
    current_tier='s1',
    current_metrics={
        'document_count': 12000000,  # 80% of S1 limit (15M)
        'storage_gb': 20.5  # 82% of S1 limit (25GB)
    },
    growth_rate={
        'docs_per_day': 50000,
        'gb_per_day': 0.15
    }
)

print(f"Capacity Assessment:")
print(f"  Current tier: {assessment['current_tier'].upper()}")
print(f"  Utilization: {assessment['utilization']['documents']} docs, {assessment['utilization']['storage']} storage")
print(f"  90-day projection: {assessment['projections']['90_days']['documents']} docs, {assessment['projections']['90_days']['storage']} storage")
print(f"  Urgency: {assessment['urgency'].upper()}")
print(f"  {assessment['message']}")
if assessment['action_needed']:
    print(f"  ⚠️  ACTION: Upgrade to {assessment['recommended_tier'].upper()} tier")
```

**Capacity Planning Timeline**:
- **Immediate** (>80% current): Upgrade within 1 week
- **High** (30-day projection >100%): Plan upgrade within 2-4 weeks
- **Medium** (90-day projection >100%): Budget for upgrade in next quarter
- **None**: Monitor monthly

---

## Troubleshooting

### 1. Alias Not Updating

**Symptoms**: After updating alias, applications still query old index.

**Root Causes**:
- Application code directly references index name (not alias)
- Cached index client pointing to old index
- Typo in alias name (e.g., "products-prod" vs "product-prod")

**Diagnosis**:

```python
def diagnose_alias_issue(alias_name, index_client):
    """Diagnose alias update issues."""
    
    # Check if alias exists
    try:
        alias = index_client.get_alias(alias_name)
        print(f"✓ Alias '{alias_name}' exists")
        print(f"  Current target: {alias.indexes[0] if alias.indexes else 'None'}")
    except Exception as e:
        print(f"✗ Alias '{alias_name}' not found")
        print(f"  Error: {e}")
        
        # List all aliases
        all_aliases = list(index_client.list_aliases())
        print(f"\\nAvailable aliases:")
        for a in all_aliases:
            print(f"  • {a.name} → {a.indexes[0] if a.indexes else 'None'}")
        
        return
    
    # Verify application is using alias
    print(f"\\nVerify application code:")
    print(f"  ✓ CORRECT: SearchClient(index_name='{alias_name}')")
    print(f"  ✗ INCORRECT: SearchClient(index_name='products-v2-20240513')")

# Usage
# diagnose_alias_issue("products-prod", index_client)
```

**Solutions**:
1. **Fix application code**: Use alias name, not direct index reference
2. **Recreate search client**: Don't reuse cached clients after alias update
3. **Verify alias name**: Check for typos (use `list_aliases()` to confirm)

---

### 2. Schema Validation Failing

**Symptoms**: Index creation fails with schema validation errors.

**Common Errors**:
- Field name conflicts (reserved keywords)
- Invalid field types
- Missing key field
- Conflicting field properties (e.g., vector field marked as sortable)

**Diagnosis**:

```python
def validate_schema_before_creation(index_definition):
    """Pre-validate schema before attempting creation."""
    
    errors = []
    warnings = []
    
    # Check 1: Key field exists
    key_fields = [f for f in index_definition.fields if getattr(f, 'key', False)]
    if len(key_fields) == 0:
        errors.append("No key field defined (exactly one required)")
    elif len(key_fields) > 1:
        errors.append(f"Multiple key fields defined: {[f.name for f in key_fields]}")
    
    # Check 2: Reserved field names
    reserved_names = {'search.score', 'search.highlights', 'search.facets'}
    for field in index_definition.fields:
        if field.name.lower() in reserved_names:
            errors.append(f"Field '{field.name}' uses reserved name")
    
    # Check 3: Vector fields not sortable/filterable
    for field in index_definition.fields:
        if 'Vector' in str(field.type):
            if getattr(field, 'sortable', False):
                errors.append(f"Vector field '{field.name}' cannot be sortable")
            if getattr(field, 'filterable', False):
                warnings.append(f"Vector field '{field.name}' should not be filterable")
    
    # Check 4: Searchable fields with proper types
    for field in index_definition.fields:
        if getattr(field, 'searchable', False):
            if 'String' not in str(field.type):
                errors.append(f"Searchable field '{field.name}' must be String type (got {field.type})")
    
    # Report results
    print(f"Schema Validation:")
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for err in errors:
            print(f"    ✗ {err}")
    
    if warnings:
        print(f"  WARNINGS ({len(warnings)}):")
        for warn in warnings:
            print(f"    ⚠ {warn}")
    
    if not errors and not warnings:
        print(f"  ✓ Schema validation passed")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

# Usage
from azure.search.documents.indexes.models import SearchIndex, SearchField, SearchFieldDataType

test_index = SearchIndex(
    name="test-index",
    fields=[
        # Missing key=True on any field → ERROR
        SearchField(name="id", type=SearchFieldDataType.String),
        SearchField(name="title", type=SearchFieldDataType.String, searchable=True)
    ]
)

# validation_result = validate_schema_before_creation(test_index)
```

**Solutions**:
1. **Add key field**: Exactly one field must have `key=True`
2. **Avoid reserved names**: Don't use `search.*` field names
3. **Match types to properties**: Searchable → String, Sortable → String/Number/Date
4. **Vector field constraints**: Not sortable, not filterable

---

### 3. Data Not Appearing in New Index

**Symptoms**: New index created successfully but documents not appearing in search results.

**Root Causes**:
- Data loading failed silently
- Indexer not started (if using indexer)
- Documents uploaded but not searchable yet (indexing lag)
- Field mapping mismatch

**Diagnosis**:

```python
def diagnose_missing_data(index_name, index_client):
    """Diagnose why data not appearing in index."""
    
    # Check 1: Document count
    stats = index_client.get_index_statistics(index_name)
    doc_count = stats.document_count
    
    print(f"Index: {index_name}")
    print(f"  Document count: {doc_count:,}")
    
    if doc_count == 0:
        print(f"  ✗ No documents in index")
        print(f"  Possible causes:")
        print(f"    • Data loading not executed")
        print(f"    • Data loading failed (check error logs)")
        print(f"    • Indexer not started (if using indexer)")
        return
    
    # Check 2: Sample query
    from azure.search.documents import SearchClient
    
    search_client = SearchClient(
        endpoint=index_client._endpoint,
        index_name=index_name,
        credential=index_client._credential
    )
    
    try:
        results = list(search_client.search("*", top=5))
        print(f"  ✓ Sample query returned {len(results)} results")
        
        if results:
            sample_doc = results[0]
            print(f"  Sample document fields: {list(sample_doc.keys())}")
        
    except Exception as e:
        print(f"  ✗ Sample query failed: {e}")
    
    # Check 3: Indexer status (if applicable)
    try:
        indexers = list(index_client.get_indexers())
        relevant_indexers = [idx for idx in indexers if idx.target_index_name == index_name]
        
        if relevant_indexers:
            print(f"\\n  Indexers targeting this index:")
            for indexer in relevant_indexers:
                print(f"    • {indexer.name}")
                # Check status
                status = index_client.get_indexer_status(indexer.name)
                print(f"      Status: {status.status}")
                if status.last_result:
                    print(f"      Last run: {status.last_result.status}")
                    if status.last_result.error_message:
                        print(f"      Error: {status.last_result.error_message}")
        else:
            print(f"\\n  No indexers found for this index (manual upload expected)")
    
    except AttributeError:
        # get_indexers not available
        pass

# Usage
# diagnose_missing_data("products-v3-20240620", index_client)
```

**Solutions**:
1. **Verify data loading**: Check upload logs for errors
2. **Start indexer**: If using indexer, ensure it's running
3. **Wait for indexing**: Large uploads may take time (check document count increases)
4. **Check field mappings**: Ensure source data matches index schema

---

### 4. Gradual Rollout Not Working

**Symptoms**: Canary deployment not routing traffic correctly (0% or 100% instead of 10%).

**Root Causes**:
- Application doesn't support traffic splitting
- Routing logic error
- Load balancer configuration issue
- Alias doesn't support multiple indexes simultaneously (must use application-level routing)

**Note**: Azure AI Search aliases point to **single index only**. For gradual rollout, use application-level routing:

```python
class GradualRolloutRouter:
    """Application-level traffic routing for canary deployments."""
    
    def __init__(self, blue_index, green_index, rollout_percentage):
        """
        Args:
            blue_index: Current production index name
            green_index: New index name
            rollout_percentage: 0-100 (% of traffic to green)
        """
        self.blue_index = blue_index
        self.green_index = green_index
        self.rollout_percentage = rollout_percentage
    
    def route_search_request(self, user_id):
        """
        Route search request to blue or green index.
        
        Uses consistent hashing: Same user always gets same index.
        """
        import hashlib
        
        # Hash user ID to determine routing
        hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        bucket = hash_value % 100  # 0-99
        
        if bucket < self.rollout_percentage:
            index_name = self.green_index
            variant = 'green'
        else:
            index_name = self.blue_index
            variant = 'blue'
        
        return {
            'index_name': index_name,
            'variant': variant,
            'user_id': user_id
        }

# Usage in application
router = GradualRolloutRouter(
    blue_index="products-v2-20240513",
    green_index="products-v3-20240620",
    rollout_percentage=10  # 10% to green, 90% to blue
)

def search_products(query, user_id):
    """Search with gradual rollout routing."""
    routing = router.route_search_request(user_id)
    
    search_client = SearchClient(
        endpoint="https://mysearch.search.windows.net",
        index_name=routing['index_name'],  # Route to blue or green
        credential=credential
    )
    
    results = search_client.search(query)
    
    # Log variant for A/B testing analysis
    log_search_event(user_id, query, variant=routing['variant'])
    
    return results
```

**Solutions**:
1. **Implement application-level routing**: Use consistent hashing to split traffic
2. **Monitor both indexes**: Track metrics for blue and green separately
3. **Gradual increase**: 10% → 50% → 100% with monitoring at each stage
4. **Use alias for final cutover**: Once at 100%, update alias for all traffic

---

### 5. Old Index Deletion Failed

**Symptoms**: Cannot delete old index versions, deletion fails.

**Root Causes**:
- Index is currently aliased (in use by production)
- Indexer still pointing to index
- Application still has open connections
- Insufficient permissions

**Diagnosis**:

```python
def diagnose_deletion_failure(index_name, index_client):
    """Diagnose why index deletion failing."""
    
    print(f"Diagnosing deletion failure for: {index_name}")
    
    # Check 1: Is index aliased?
    all_aliases = list(index_client.list_aliases())
    aliased_by = [a.name for a in all_aliases if index_name in (a.indexes or [])]
    
    if aliased_by:
        print(f"  ✗ Index is aliased (in use)")
        print(f"    Aliases pointing to this index: {', '.join(aliased_by)}")
        print(f"    SOLUTION: Update aliases to point elsewhere before deletion")
        return
    else:
        print(f"  ✓ Index not aliased")
    
    # Check 2: Indexers pointing to this index
    try:
        indexers = list(index_client.get_indexers())
        using_indexers = [idx.name for idx in indexers if idx.target_index_name == index_name]
        
        if using_indexers:
            print(f"  ✗ Indexers pointing to this index")
            print(f"    Indexers: {', '.join(using_indexers)}")
            print(f"    SOLUTION: Delete or update indexers first")
            return
        else:
            print(f"  ✓ No indexers using this index")
    
    except AttributeError:
        pass
    
    # Check 3: Try deletion with error handling
    try:
        print(f"  Attempting deletion...")
        index_client.delete_index(index_name)
        print(f"  ✓ Deletion successful")
    
    except Exception as e:
        print(f"  ✗ Deletion failed: {str(e)}")
        
        if "not found" in str(e).lower():
            print(f"    Index already deleted or doesn't exist")
        elif "permission" in str(e).lower():
            print(f"    Insufficient permissions (need admin key)")
        else:
            print(f"    Unknown error - check Azure portal")

# Usage
# diagnose_deletion_failure("products-v1-20240415", index_client)
```

**Solutions**:
1. **Remove aliases**: Update all aliases to point away from index
2. **Delete indexers**: Remove or update indexers targeting this index
3. **Wait for connections**: Close all active search clients
4. **Check permissions**: Use admin key (not query key)

---

## Next Steps

- **[Dataset Preparation](./16-dataset-preparation.md)** - Preparing test data
- **[A/B Testing](./17-ab-testing-framework.md)** - Testing index changes
- **[CI/CD Pipelines](./22-cicd-pipelines.md)** - Automating deployments

---

*See also: [Scoring Profiles](./14-scoring-profiles.md) | [Query Optimization](./12-query-optimization.md)*
