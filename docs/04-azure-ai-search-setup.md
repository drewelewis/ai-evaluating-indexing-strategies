# Azure AI Search Setup

Complete guide to setting up and configuring Azure AI Search service for optimal search indexing evaluation and deployment.

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Service Creation](#service-creation)
- [Service Tier Selection](#service-tier-selection)
- [Index Configuration](#index-configuration)
- [Data Source Setup](#data-source-setup)
- [Security Configuration](#security-configuration)
- [Network Configuration](#network-configuration)
- [Cost Management](#cost-management)

---

## Prerequisites

### Azure Subscription
- Active Azure subscription
- Appropriate permissions (Contributor or Owner role)
- Budget allocation for search service

### Development Environment
```bash
# Install Azure CLI
# Download from: https://aka.ms/installazurecli

# Login to Azure
az login

# Set your subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Install Azure Search extension
az extension add --name search
```

### Required Python Packages
```bash
pip install azure-search-documents==11.4.0
pip install azure-identity==1.15.0
pip install azure-core==1.29.0
pip install openai==1.6.1
pip install python-dotenv==1.0.0
pip install pyyaml==6.0.1
```

---

## Service Creation

### Option 1: Azure Portal

1. **Navigate to Azure Portal**
   - Go to https://portal.azure.com
   - Click "+ Create a resource"
   - Search for "Azure AI Search"

2. **Configure Basic Settings**
   - **Subscription**: Select your subscription
   - **Resource Group**: Create new or use existing
   - **Service Name**: Unique name (e.g., `mysearch-prod-001`)
   - **Location**: Choose region closest to users
   - **Pricing Tier**: See [Service Tier Selection](#service-tier-selection)

3. **Review and Create**
   - Review configuration
   - Click "Create"
   - Wait 2-5 minutes for deployment

### Option 2: Azure CLI

```bash
# Set variables
RESOURCE_GROUP="rg-search-evaluation"
LOCATION="eastus"
SEARCH_SERVICE_NAME="search-eval-prod-001"
SKU="standard"

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Create search service
az search service create \
  --name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku $SKU \
  --partition-count 1 \
  --replica-count 1

# Get service details
az search service show \
  --name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP
```

### Option 3: ARM Template

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "searchServiceName": {
      "type": "string",
      "metadata": {
        "description": "Name of the Azure AI Search service"
      }
    },
    "location": {
      "type": "string",
      "defaultValue": "[resourceGroup().location]"
    },
    "sku": {
      "type": "string",
      "defaultValue": "standard",
      "allowedValues": ["free", "basic", "standard", "standard2", "standard3"]
    }
  },
  "resources": [
    {
      "type": "Microsoft.Search/searchServices",
      "apiVersion": "2023-11-01",
      "name": "[parameters('searchServiceName')]",
      "location": "[parameters('location')]",
      "sku": {
        "name": "[parameters('sku')]"
      },
      "properties": {
        "replicaCount": 1,
        "partitionCount": 1,
        "hostingMode": "default",
        "publicNetworkAccess": "enabled",
        "semanticSearch": "standard"
      }
    }
  ],
  "outputs": {
    "searchServiceEndpoint": {
      "type": "string",
      "value": "[concat('https://', parameters('searchServiceName'), '.search.windows.net')]"
    }
  }
}
```

Deploy ARM template:
```bash
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file search-service-template.json \
  --parameters searchServiceName=$SEARCH_SERVICE_NAME
```

### Option 4: Python SDK

```python
from azure.mgmt.search import SearchManagementClient
from azure.identity import DefaultAzureCredential
from azure.mgmt.search.models import SearchService, Sku

# Initialize clients
credential = DefaultAzureCredential()
subscription_id = "YOUR_SUBSCRIPTION_ID"
resource_group = "rg-search-evaluation"

search_mgmt_client = SearchManagementClient(credential, subscription_id)

# Create search service
search_service = SearchService(
    location="eastus",
    sku=Sku(name="standard"),
    replica_count=1,
    partition_count=1,
    hosting_mode="default"
)

# Deploy
search_mgmt_client.services.begin_create_or_update(
    resource_group_name=resource_group,
    search_service_name="search-eval-prod-001",
    service=search_service
).result()

print("Azure AI Search service created successfully")
```

---

## Service Tier Selection

### Tier Comparison

| Tier | Price/Month | Storage | Documents | QPS | Replicas | Partitions | Use Case |
|------|-------------|---------|-----------|-----|----------|------------|----------|
| **Free** | $0 | 50 MB | 10K | 3 | 1 | 1 | Development, POC |
| **Basic** | ~$75 | 2 GB | 1M | 15 | 3 | 1 | Small production |
| **Standard S1** | ~$250 | 25 GB | 15M | 50 | 12 | 12 | Medium production |
| **Standard S2** | ~$1,000 | 100 GB | 60M | 200 | 12 | 12 | Large production |
| **Standard S3** | ~$2,000 | 200 GB | 120M | 400 | 12 | 12 | Enterprise |

### Decision Matrix

```python
def recommend_service_tier(requirements):
    """Recommend Azure AI Search tier based on requirements."""
    
    # Extract requirements
    doc_count = requirements.get('document_count', 0)
    qps_needed = requirements.get('queries_per_second', 10)
    storage_gb = requirements.get('storage_gb', 1)
    high_availability = requirements.get('high_availability', False)
    
    # Decision logic
    if doc_count < 10000 and not requirements.get('production', False):
        return {
            'tier': 'Free',
            'monthly_cost': 0,
            'reason': 'Development/testing workload'
        }
    
    if doc_count < 1000000 and qps_needed < 15 and not high_availability:
        return {
            'tier': 'Basic',
            'monthly_cost': 75,
            'reason': 'Small production workload without HA requirements'
        }
    
    if doc_count < 15000000 and qps_needed < 50:
        return {
            'tier': 'Standard S1',
            'monthly_cost': 250,
            'reason': 'Standard production workload'
        }
    
    if doc_count < 60000000 and qps_needed < 200:
        return {
            'tier': 'Standard S2',
            'monthly_cost': 1000,
            'reason': 'Large-scale production workload'
        }
    
    return {
        'tier': 'Standard S3',
        'monthly_cost': 2000,
        'reason': 'Enterprise-scale workload'
    }

# Example usage
my_requirements = {
    'document_count': 5000000,
    'queries_per_second': 30,
    'storage_gb': 15,
    'high_availability': True,
    'production': True
}

recommendation = recommend_service_tier(my_requirements)
print(f"Recommended tier: {recommendation['tier']}")
print(f"Est. cost: ${recommendation['monthly_cost']}/month")
print(f"Reason: {recommendation['reason']}")
```

---

## Index Configuration

### Basic Index Schema

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
from azure.core.credentials import AzureKeyCredential

# Initialize index client
endpoint = "https://YOUR-SERVICE.search.windows.net"
api_key = "YOUR-API-KEY"

index_client = SearchIndexClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key)
)

# Define index schema
def create_basic_index(index_name):
    """Create a basic search index."""
    
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            sortable=True,
            analyzer_name="en.microsoft"
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name="en.microsoft"
        ),
        SimpleField(
            name="category",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        SimpleField(
            name="created_date",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True
        ),
        SimpleField(
            name="view_count",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True
        )
    ]
    
    index = SearchIndex(
        name=index_name,
        fields=fields
    )
    
    # Create index
    result = index_client.create_index(index)
    print(f"Index '{index_name}' created successfully")
    return result

# Create index
create_basic_index("products-index")
```

### Vector Search Index

```python
def create_vector_index(index_name):
    """Create index with vector search capabilities."""
    
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,  # OpenAI ada-002 dimension
            vector_search_profile_name="vector-profile"
        )
    ]
    
    # Configure vector search
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-algorithm",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-algorithm"
            )
        ]
    )
    
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )
    
    result = index_client.create_index(index)
    print(f"Vector index '{index_name}' created successfully")
    return result

# Create vector index
create_vector_index("products-vector-index")
```

### Hybrid Index (Text + Vector)

```python
def create_hybrid_index(index_name):
    """Create index supporting both text and vector search."""
    
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft"
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft"
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="hybrid-profile"
        ),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True)
    ]
    
    # HNSW configuration optimized for hybrid search
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hybrid-hnsw",
                parameters={
                    "m": 8,  # Higher for better recall in hybrid
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hybrid-profile",
                algorithm_configuration_name="hybrid-hnsw"
            )
        ]
    )
    
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )
    
    result = index_client.create_index(index)
    print(f"Hybrid index '{index_name}' created successfully")
    return result

# Create hybrid index
create_hybrid_index("products-hybrid-index")
```

---

## Data Source Setup

### Azure Blob Storage Integration

```python
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataSourceConnection,
    SearchIndexer,
    IndexingSchedule
)
from datetime import timedelta

# Initialize indexer client
indexer_client = SearchIndexerClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key)
)

def create_blob_data_source(data_source_name, storage_connection_string, container_name):
    """Create Azure Blob Storage data source."""
    
    data_source = SearchIndexerDataSourceConnection(
        name=data_source_name,
        type="azureblob",
        connection_string=storage_connection_string,
        container={"name": container_name}
    )
    
    result = indexer_client.create_data_source_connection(data_source)
    print(f"Data source '{data_source_name}' created")
    return result

def create_indexer(indexer_name, data_source_name, index_name):
    """Create indexer with schedule."""
    
    indexer = SearchIndexer(
        name=indexer_name,
        data_source_name=data_source_name,
        target_index_name=index_name,
        schedule=IndexingSchedule(interval=timedelta(hours=1))
    )
    
    result = indexer_client.create_indexer(indexer)
    print(f"Indexer '{indexer_name}' created with hourly schedule")
    return result

# Example usage
STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=..."
create_blob_data_source("blob-datasource", STORAGE_CONNECTION_STRING, "documents")
create_indexer("blob-indexer", "blob-datasource", "products-index")
```

### Azure Cosmos DB Integration

```python
def create_cosmos_data_source(data_source_name, cosmos_connection_string, database_name, collection_name):
    """Create Azure Cosmos DB data source."""
    
    data_source = SearchIndexerDataSourceConnection(
        name=data_source_name,
        type="cosmosdb",
        connection_string=cosmos_connection_string,
        container={
            "name": collection_name,
            "query": "SELECT * FROM c WHERE c._ts >= @HighWaterMark"
        },
        data_change_detection_policy={
            "@odata.type": "#Microsoft.Azure.Search.HighWaterMarkChangeDetectionPolicy",
            "highWaterMarkColumnName": "_ts"
        }
    )
    
    result = indexer_client.create_data_source_connection(data_source)
    print(f"Cosmos DB data source '{data_source_name}' created")
    return result

# Example usage
COSMOS_CONNECTION_STRING = "AccountEndpoint=https://...;AccountKey=...;Database=mydb"
create_cosmos_data_source("cosmos-datasource", COSMOS_CONNECTION_STRING, "mydb", "products")
```

---

## Security Configuration

### API Key Management

```python
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Store API keys in Azure Key Vault
def store_search_key_in_keyvault(vault_url, secret_name, api_key):
    """Store search API key securely in Key Vault."""
    
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    
    client.set_secret(secret_name, api_key)
    print(f"API key stored in Key Vault as '{secret_name}'")

# Retrieve API key from Key Vault
def get_search_key_from_keyvault(vault_url, secret_name):
    """Retrieve search API key from Key Vault."""
    
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    
    secret = client.get_secret(secret_name)
    return secret.value

# Usage
vault_url = "https://my-keyvault.vault.azure.net/"
api_key = get_search_key_from_keyvault(vault_url, "search-api-key")
```

### Role-Based Access Control (RBAC)

```bash
# Assign Search Service Contributor role
az role assignment create \
  --role "Search Service Contributor" \
  --assignee user@example.com \
  --scope /subscriptions/{subscription-id}/resourceGroups/{resource-group}/providers/Microsoft.Search/searchServices/{search-service-name}

# Assign Search Index Data Contributor role
az role assignment create \
  --role "Search Index Data Contributor" \
  --assignee user@example.com \
  --scope /subscriptions/{subscription-id}/resourceGroups/{resource-group}/providers/Microsoft.Search/searchServices/{search-service-name}
```

### Managed Identity Setup

```python
from azure.identity import ManagedIdentityCredential
from azure.search.documents import SearchClient

# Use managed identity for authentication
credential = ManagedIdentityCredential()

search_client = SearchClient(
    endpoint="https://YOUR-SERVICE.search.windows.net",
    index_name="products-index",
    credential=credential
)

# No API keys needed!
```

---

## Network Configuration

### Private Endpoint Setup

```bash
# Create private endpoint
az network private-endpoint create \
  --name search-private-endpoint \
  --resource-group $RESOURCE_GROUP \
  --vnet-name my-vnet \
  --subnet my-subnet \
  --private-connection-resource-id /subscriptions/{subscription-id}/resourceGroups/{resource-group}/providers/Microsoft.Search/searchServices/{search-service-name} \
  --group-id searchService \
  --connection-name search-private-connection

# Configure private DNS zone
az network private-dns zone create \
  --resource-group $RESOURCE_GROUP \
  --name privatelink.search.windows.net

az network private-dns link vnet create \
  --resource-group $RESOURCE_GROUP \
  --zone-name privatelink.search.windows.net \
  --name search-dns-link \
  --virtual-network my-vnet \
  --registration-enabled false
```

### IP Firewall Rules

```python
from azure.mgmt.search.models import IpRule

# Configure IP firewall
ip_rules = [
    IpRule(value="40.76.54.131"),  # Office IP
    IpRule(value="52.176.6.30/32")  # Application server
]

# Update search service
search_mgmt_client.services.update(
    resource_group_name=resource_group,
    search_service_name=search_service_name,
    service={
        "properties": {
            "networkRuleSet": {
                "ipRules": ip_rules
            }
        }
    }
)
```

---

## Cost Management

### Cost Monitoring

```python
def estimate_monthly_cost(tier, replicas, partitions, queries_per_month):
    """Estimate monthly Azure AI Search cost."""
    
    base_costs = {
        'free': 0,
        'basic': 75,
        'standard': 250,
        'standard2': 1000,
        'standard3': 2000
    }
    
    base_cost = base_costs.get(tier.lower(), 250)
    
    # Additional replica and partition costs
    if tier.lower() != 'free':
        # Replicas: ~20% of base cost per additional replica
        replica_cost = base_cost * 0.20 * (replicas - 1)
        
        # Partitions: ~50% of base cost per additional partition
        partition_cost = base_cost * 0.50 * (partitions - 1)
    else:
        replica_cost = 0
        partition_cost = 0
    
    # Query costs (minimal, included in tier)
    query_cost = 0
    
    total_cost = base_cost + replica_cost + partition_cost + query_cost
    
    return {
        'base_cost': base_cost,
        'replica_cost': replica_cost,
        'partition_cost': partition_cost,
        'total_monthly_cost': total_cost,
        'cost_per_1k_queries': total_cost / (queries_per_month / 1000) if queries_per_month > 0 else 0
    }

# Example
cost_breakdown = estimate_monthly_cost(
    tier='standard',
    replicas=2,
    partitions=1,
    queries_per_month=1000000
)

print(f"Total monthly cost: ${cost_breakdown['total_monthly_cost']:.2f}")
print(f"Cost per 1K queries: ${cost_breakdown['cost_per_1k_queries']:.4f}")
```

### Budget Alerts

```bash
# Create budget alert
az consumption budget create \
  --budget-name search-service-budget \
  --amount 500 \
  --time-grain Monthly \
  --start-date 2025-01-01 \
  --end-date 2026-01-01 \
  --notifications "actual_GreaterThan_80_Percent={\"enabled\":true,\"operator\":\"GreaterThan\",\"threshold\":80,\"contactEmails\":[\"admin@example.com\"]}"
```

---

## Next Steps

- **[Azure OpenAI Integration](./05-azure-openai-integration.md)** - Set up embedding generation
- **[Full-text Search Implementation](./08-fulltext-search-bm25.md)** - Implement your first search strategy
- **[Index Management](./15-index-management.md)** - Learn advanced index operations

---

*See also: [Azure Service Tiers](./28-azure-service-tiers.md) | [API Reference](./29-api-reference.md) | [Troubleshooting](./27-troubleshooting-guide.md)*