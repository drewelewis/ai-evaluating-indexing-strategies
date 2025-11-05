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

### Index Management Fundamentals

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

### ✅ Do's
1. **Use aliases** for all production indexes
2. **Test schema changes** in non-production first
3. **Keep 2-3 recent versions** for quick rollback
4. **Document version changes** in metadata
5. **Automate deployments** with blue-green pattern
6. **Monitor index statistics** (doc count, storage)
7. **Export schemas** to version control
8. **Plan capacity** before hitting limits
9. **Use 2+ replicas** for production (99.9% SLA)
10. **Schedule cleanup** of old index versions

### ❌ Don'ts
1. **Don't** modify production indexes directly
2. **Don't** delete aliased indexes
3. **Don't** ignore breaking schema changes
4. **Don't** keep unlimited old versions (storage cost)
5. **Don't** skip validation before alias switch
6. **Don't** under-provision replicas (availability risk)
7. **Don't** over-partition small indexes (<1M docs)
8. **Don't** change analyzers without rebuilding
9. **Don't** forget rollback plan
10. **Don't** deploy Friday afternoon 😊

---

## Next Steps

- **[Dataset Preparation](./16-dataset-preparation.md)** - Preparing test data
- **[A/B Testing](./17-ab-testing-framework.md)** - Testing index changes
- **[CI/CD Pipelines](./22-cicd-pipelines.md)** - Automating deployments

---

*See also: [Scoring Profiles](./14-scoring-profiles.md) | [Query Optimization](./12-query-optimization.md)*
