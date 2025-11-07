# Azure AI Search Setup

**Purpose**: Complete guide to setting up and configuring Azure AI Search service for optimal search indexing evaluation and deployment. This document covers everything from initial provisioning through production-ready configuration, security hardening, and cost optimization.

**Target Audience**: DevOps engineers, cloud architects, and developers responsible for deploying and managing Azure AI Search infrastructure.

**Reading Time**: 35-40 minutes (comprehensive setup), 15 minutes (quick start only)

**Related Documents**:
- `03-indexing-strategies.md`: Strategy selection and implementation patterns
- `05-azure-openai-integration.md`: Configuring Azure OpenAI for vector search
- `07-azure-monitor-logging.md`: Monitoring and observability setup
- `19-cost-analysis.md`: Cost optimization strategies

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

## Why Azure AI Search Setup Matters

**The cost of getting setup wrong**: Choosing the wrong service tier can cost 10x more than necessary, while inadequate replica configuration leads to throttling and poor user experience. Security misconfigurations expose sensitive data. This guide helps you avoid these pitfalls.

**Real-world impact example**:
```
Scenario 1: Under-provisioned setup
- Chose Basic tier ($75/month) for production workload
- 1 replica couldn't handle 50 QPS peak traffic
- Throttling (HTTP 429) on 20% of requests during peak hours
- User experience degraded → 35% drop in search usage
- Cost to upgrade to Standard (2 replicas): +$175/month
- Lesson: Tier selection based on actual QPS requirements critical

Scenario 2: Over-provisioned setup
- Chose Standard3 tier ($1,000/month) for dev workload
- Actually needed Basic tier ($75/month) for 5 QPS dev traffic
- Wasted $925/month ($11,100/year) due to over-provisioning
- Lesson: Right-size tiers for each environment (dev, staging, prod)

Scenario 3: Correct setup
- Evaluated actual requirements: 100 QPS peak, 50GB data
- Chose Standard tier ($250/month) with 2 replicas
- Handled traffic with <50ms P95 latency
- Added auto-scaling for Black Friday spike (3 replicas temporarily)
- Total cost: $250/month baseline + $125 for 3-day spike event
- Lesson: Right-sizing + elasticity = optimal cost/performance
```

**Three critical setup decisions**:
1. **Tier selection**: Determines capacity, cost, and features (semantic search only on Standard+)
2. **Replica/partition configuration**: Affects throughput, availability, and query latency
3. **Security posture**: Network isolation, authentication, encryption determine risk exposure

This guide provides decision frameworks for all three.

## Prerequisites

### Azure Subscription Requirements

**Subscription prerequisites**:
- Active Azure subscription with credit or billing configured
- Appropriate RBAC permissions:
  * **Owner** role: Full control (can create services, assign roles, configure networking)
  * **Contributor** role: Can create services but not manage access
  * **Minimum**: `Microsoft.Search/*` resource provider permissions
- Budget allocation:
  * Development: $100-300/month (Basic or Standard S1)
  * Production: $250-2,000/month (Standard S1-S3, depends on scale)
  * Enterprise: $2,000-10,000+/month (Standard S3 or Storage Optimized L1/L2)

**Quota validation**:
```bash
# Check Azure AI Search quota in your subscription
az search service quota show \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Typical quotas:
# Free tier: 1 per subscription
# Basic tier: 16 per subscription per region
# Standard tier: 12 per subscription per region
# Storage Optimized: 1 per subscription per region

# Request quota increase if needed:
# Azure Portal → Support → New support request → Quota increase
```

**Cost estimation before provisioning**:
- Use [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)
- Example calculation:
  ```
  Standard S1 tier:
  - Base: $250/month (1 partition, 1 replica)
  - Additional replica: +$250/month (for HA)
  - Additional partition: +$250/month (for >25GB data)
  - Semantic search: +$500/month (after 1K free queries)
  
  Total for prod setup (2 replicas, 1 partition, no semantic): $500/month
  ```

### Development Environment

**Required tools and packages**:

**Azure CLI setup** (required for service provisioning and management):
```bash
# Install Azure CLI (Windows)
# Download from: https://aka.ms/installazurecli
# Or use chocolatey:
choco install azure-cli

# Install Azure CLI (macOS)
brew update && brew install azure-cli

# Install Azure CLI (Linux - Ubuntu/Debian)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Verify installation
az --version  # Should show version 2.50+ for latest Azure AI Search features

# Login to Azure
az login  # Opens browser for authentication

# Set your subscription (if you have multiple)
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Verify current subscription
az account show

# Install Azure Search extension (for advanced CLI operations)
az extension add --name search
az extension update --name search  # Keep updated
```

**Why Azure CLI matters**: Enables infrastructure-as-code, automation, CI/CD integration. Portal is fine for one-off dev work, but production deployments need scripted, repeatable provisioning.

**Required Python Packages** (for programmatic index management and search):
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install Azure AI Search SDK (latest stable)
pip install azure-search-documents==11.4.0

# Azure identity for authentication (managed identity, service principal)
pip install azure-identity==1.15.0

# Core Azure SDK dependencies
pip install azure-core==1.29.0

# Azure OpenAI for embeddings (if using vector search)
pip install openai==1.6.1

# Configuration management
pip install python-dotenv==1.0.0  # Load .env files
pip install pyyaml==6.0.1  # Parse YAML configs

# Optional but recommended
pip install azure-mgmt-search==9.0.0  # Service management (create/delete services)
pip install requests==2.31.0  # HTTP client for custom integrations

# Save dependencies
pip freeze > requirements.txt
```

**Environment variables setup** (.env file):
```bash
# Create .env file for local development
cat <<EOF > .env
# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://YOUR-SERVICE.search.windows.net
AZURE_SEARCH_API_KEY=YOUR-ADMIN-API-KEY

# Azure OpenAI (if using vector search)
AZURE_OPENAI_ENDPOINT=https://YOUR-OPENAI.openai.azure.com
AZURE_OPENAI_API_KEY=YOUR-OPENAI-API-KEY
AZURE_OPENAI_DEPLOYMENT=text-embedding-ada-002

# Azure Cosmos DB (if using for analytics)
AZURE_COSMOS_ENDPOINT=https://YOUR-COSMOS.documents.azure.com:443
AZURE_COSMOS_KEY=YOUR-COSMOS-KEY

# Optional: Azure Monitor (for logging)
AZURE_LOG_ANALYTICS_WORKSPACE_ID=YOUR-WORKSPACE-ID
AZURE_LOG_ANALYTICS_KEY=YOUR-WORKSPACE-KEY
EOF

# Add .env to .gitignore (NEVER commit secrets)
echo ".env" >> .gitignore
```

**Using environment variables in Python**:
```python
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# Load environment variables from .env file
load_dotenv()

# Access credentials from environment
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
api_key = os.getenv("AZURE_SEARCH_API_KEY")

# Initialize client
search_client = SearchClient(
    endpoint=endpoint,
    index_name="my-index",
    credential=AzureKeyCredential(api_key)
)

# Now you can use search_client for queries
```

**Development environment validation**:
```python
# test_setup.py - Verify all dependencies and connectivity
import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

def test_azure_search_connection():
    """Test Azure AI Search connectivity."""
    load_dotenv()
    
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    
    if not endpoint or not api_key:
        print("❌ Azure Search credentials missing in .env file")
        return False
    
    try:
        index_client = SearchIndexClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
        
        # List indexes (confirms auth works)
        indexes = list(index_client.list_indexes())
        print(f"✅ Azure AI Search connection successful")
        print(f"   Found {len(indexes)} indexes")
        return True
        
    except Exception as e:
        print(f"❌ Azure AI Search connection failed: {e}")
        return False

def test_azure_openai_connection():
    """Test Azure OpenAI connectivity (if using vector search)."""
    from openai import AzureOpenAI
    
    load_dotenv()
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    if not endpoint or not api_key:
        print("⚠️  Azure OpenAI credentials missing (optional for BM25-only)")
        return True  # Not critical if only using BM25
    
    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-02-01"
        )
        
        # Test embedding generation
        response = client.embeddings.create(
            input="test connection",
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-ada-002")
        )
        
        print(f"✅ Azure OpenAI connection successful")
        print(f"   Embedding dimensions: {len(response.data[0].embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ Azure OpenAI connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Azure setup...\n")
    
    search_ok = test_azure_search_connection()
    openai_ok = test_azure_openai_connection()
    
    if search_ok:
        print("\n✅ Setup complete! Ready to start indexing.")
    else:
        print("\n❌ Setup incomplete. Fix errors above before continuing.")
```

**Common setup issues and fixes**:
```
Issue 1: "Az login failed - browser not opening"
Fix: Use device code flow: az login --use-device-code
     Then visit https://microsoft.com/devicelogin and enter code

Issue 2: "Insufficient permissions to create Azure AI Search service"
Fix: Request Owner or Contributor role on subscription/resource group
     Or request Microsoft.Search/* permissions from admin

Issue 3: "Quota exceeded for Standard tier in region"
Fix: Choose different region or request quota increase via Azure Support
     Check quota: az search service quota show

Issue 4: "Azure OpenAI application pending"
Fix: Azure OpenAI requires application approval (not instant)
     Apply at: https://aka.ms/oai/access
     Approval typically takes 1-2 business days
     Alternative: Use public OpenAI API temporarily for testing

Issue 5: "Python SDK version mismatch errors"
Fix: Ensure azure-search-documents >= 11.4.0
     Uninstall old versions: pip uninstall azure-search-documents
     Reinstall: pip install azure-search-documents==11.4.0
```

---

## Service Creation

**Four ways to create Azure AI Search**: Portal (quickest), CLI (scriptable), ARM template (infrastructure-as-code), Python SDK (programmatic). Choose based on your deployment workflow and automation needs.

### Option 1: Azure Portal (Recommended for First-Time Setup)

**When to use Portal**: Quick POC, learning the service, one-off dev instances, when you need visual confirmation of configuration.

**When NOT to use Portal**: Production deployments (use CLI/ARM for repeatability), CI/CD pipelines, multi-environment setups.

**Step-by-step portal creation**:

1. **Navigate to Azure Portal**
   - Go to https://portal.azure.com
   - Sign in with Azure credentials
   - Click "+ Create a resource" (top left)
   - Search for "Azure AI Search" (or "Azure Cognitive Search" in older portal)
   - Click "Create"

2. **Configure Basic Settings**
   - **Subscription**: Select your Azure subscription
   - **Resource Group**: 
     * Create new: Click "Create new" → Name it `rg-search-production` (or `rg-search-dev` for dev)
     * Or use existing: Select from dropdown
     * Why resource groups matter: Group related resources for cost tracking, access control, lifecycle management
   
   - **Service Name**: 
     * Must be globally unique (3-60 chars, lowercase, numbers, hyphens)
     * Naming convention: `search-{project}-{env}-{instance}`
     * Examples: `search-ecommerce-prod-001`, `search-knowledge-dev-01`
     * ⚠️ Cannot be changed after creation
   
   - **Location** (Region selection is critical):
     * **Latency optimization**: Choose region closest to your users
       - US East Coast users → East US or East US 2
       - US West Coast users → West US 2 or West US 3
       - European users → West Europe or North Europe
       - Asian users → Southeast Asia or East Asia
     
     * **Data residency**: Some orgs require data in specific geographies (GDPR, data sovereignty)
       - EU data must stay in EU → Choose West Europe, North Europe, France Central
       - US government → US Gov regions (requires special subscription)
     
     * **Feature availability**: Not all features available in all regions
       - Semantic search: Available in most regions (check docs)
       - Availability zones: Only in select regions
       - Storage Optimized tiers: Limited regions
     
     * **Cost variation**: Prices vary by region (typically <10% difference)
       - Check: https://azure.microsoft.com/pricing/details/search/
     
     * **Recommendation**: East US or West Europe (best feature availability, reliability)
   
   - **Pricing Tier** (see [Service Tier Selection](#service-tier-selection) for detailed guidance):
     * Free: POC only, limited to 3 QPS, 50 MB storage
     * Basic: Small production, <1M docs, <15 QPS
     * Standard S1: Most common, <15M docs, <50 QPS
     * Standard S2: Large scale, <60M docs, <200 QPS
     * Standard S3: Enterprise, <60M docs, <200+ QPS
     * Storage Optimized L1/L2: Massive document counts (>100M)

3. **Configure Scale Settings** (Replicas and Partitions)
   - **Replicas**: Number of search service instances for queries
     * 1 replica: No high availability (dev/test only)
     * 2 replicas: Minimum for production (99.9% SLA)
     * 3+ replicas: High traffic (each replica adds throughput)
     * Cost: Each replica costs same as base tier (2 replicas = 2x cost)
   
   - **Partitions**: Storage and indexing shards
     * 1 partition: Up to 25GB storage (S1), 50GB (S2), 100GB (S3)
     * 2+ partitions: Needed when data exceeds single partition capacity
     * Cost: Each partition costs same as base tier
   
   - **Example configurations**:
     ```
     Dev setup: 1 replica, 1 partition = $250/month (S1)
     Prod setup: 2 replicas, 1 partition = $500/month (S1)
     High-traffic prod: 3 replicas, 2 partitions = $1,500/month (S1)
     ```

4. **Configure Networking** (Optional, see [Network Configuration](#network-configuration))
   - **Public access**: Default, accessible from internet (secured by API keys)
   - **Private endpoint**: Accessible only from your VNet (recommended for production)
   - **Firewall rules**: Whitelist specific IP addresses
   
   - For first setup: Keep public access, add firewall rule for your dev machine IP
   - For production: Use private endpoints (configure later)

5. **Configure Tags** (Recommended for cost tracking)
   ```
   Environment: Production
   Project: SearchEvaluation
   Owner: platform-team@company.com
   CostCenter: Engineering
   ```
   - Why tags matter: Filter costs in Azure Cost Management, identify resources in large subscriptions

6. **Review and Create**
   - Review all settings carefully (most can be changed later except name and region)
   - Click "Create"
   - Deployment takes 2-5 minutes
   - Monitor deployment progress in Notifications (bell icon, top right)

7. **Post-Creation Validation**
   - Once deployed, click "Go to resource"
   - Note the **URL**: `https://YOUR-SERVICE-NAME.search.windows.net`
   - Note the **API keys**: Settings → Keys → Copy Primary Admin Key
   - Test connectivity using validation script above

**Portal creation success checklist**:
- ✅ Service shows "Running" status in portal
- ✅ URL accessible (https://YOUR-SERVICE.search.windows.net returns 403, not timeout)
- ✅ API keys visible in Settings → Keys
- ✅ Can create test index via portal or SDK
- ✅ Monitoring enabled (Metrics blade shows data within 5 minutes)

### Option 2: Azure CLI (Recommended for Production and CI/CD)

**When to use CLI**: Production deployments, CI/CD pipelines, infrastructure-as-code, repeatable multi-environment setups, automation scripts.

**Advantages over Portal**:
- Scriptable and repeatable (same command creates identical services)
- Version controlled (store scripts in Git)
- Automatable (integrate into CI/CD pipelines)
- Faster (no clicking through UI)
- Auditable (command history tracked)

**Prerequisites**: Azure CLI installed and logged in (see [Development Environment](#development-environment) section).

**Complete CLI provisioning script**:

```bash
# ============================================
# Azure AI Search Service Creation Script
# ============================================

# Configuration variables (customize these)
RESOURCE_GROUP="rg-search-evaluation"
LOCATION="eastus"  # or "westeurope", "westus2", etc.
SEARCH_SERVICE_NAME="search-eval-prod-001"  # Must be globally unique
SKU="standard"  # Options: free, basic, standard, standard2, standard3, storage_optimized_l1, storage_optimized_l2
REPLICA_COUNT=2  # High availability (2+) or dev (1)
PARTITION_COUNT=1  # Scale for storage (1 partition = 25GB for S1)

# Optional: Tags for resource organization
TAGS="Environment=Production Project=SearchEval Owner=platform-team CostCenter=Engineering"

# ============================================
# Step 1: Create Resource Group (if doesn't exist)
# ============================================
echo "Creating resource group: $RESOURCE_GROUP"

az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION \
  --tags $TAGS

# Verify creation
az group show --name $RESOURCE_GROUP --output table

# ============================================
# Step 2: Create Azure AI Search Service
# ============================================
echo "Creating Azure AI Search service: $SEARCH_SERVICE_NAME"
echo "This will take 2-5 minutes..."

az search service create \
  --name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku $SKU \
  --partition-count $PARTITION_COUNT \
  --replica-count $REPLICA_COUNT \
  --public-network-access Enabled \
  --semantic-search standard \
  --tags $TAGS

# Check creation status
if [ $? -eq 0 ]; then
    echo "✅ Search service created successfully"
else
    echo "❌ Search service creation failed"
    exit 1
fi

# ============================================
# Step 3: Get Service Details
# ============================================
echo "\n📋 Service Details:"

# Get endpoint URL
SEARCH_ENDPOINT=$(az search service show \
  --name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "properties.endpoint" \
  --output tsv)

echo "Endpoint: $SEARCH_ENDPOINT"

# Get admin API key (for indexing and management)
ADMIN_KEY=$(az search admin-key show \
  --service-name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "primaryKey" \
  --output tsv)

echo "Admin Key: $ADMIN_KEY"

# Get query API key (for search-only operations)
QUERY_KEY=$(az search query-key list \
  --service-name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "[0].key" \
  --output tsv)

echo "Query Key: $QUERY_KEY"

# ============================================
# Step 4: Save Credentials to .env File
# ============================================
echo "\n💾 Saving credentials to .env file..."

cat > .env <<EOF
# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT=$SEARCH_ENDPOINT
AZURE_SEARCH_ADMIN_KEY=$ADMIN_KEY
AZURE_SEARCH_QUERY_KEY=$QUERY_KEY
AZURE_SEARCH_SERVICE_NAME=$SEARCH_SERVICE_NAME

# Resource details
AZURE_RESOURCE_GROUP=$RESOURCE_GROUP
AZURE_LOCATION=$LOCATION
EOF

echo "✅ Credentials saved to .env file"
echo "⚠️  Remember to add .env to .gitignore"

# ============================================
# Step 5: Enable Diagnostic Logging (Monitoring)
# ============================================
echo "\n📊 Configuring diagnostic logging..."

# Get Log Analytics workspace ID (create if needed)
LOG_WORKSPACE="log-search-evaluation"

# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $LOG_WORKSPACE \
  --location $LOCATION \
  --tags $TAGS

WORKSPACE_ID=$(az monitor log-analytics workspace show \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $LOG_WORKSPACE \
  --query "id" \
  --output tsv)

# Enable diagnostic settings
az monitor diagnostic-settings create \
  --name "search-diagnostics" \
  --resource "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Search/searchServices/$SEARCH_SERVICE_NAME" \
  --workspace $WORKSPACE_ID \
  --logs '[
    {
      "category": "OperationLogs",
      "enabled": true,
      "retentionPolicy": {"enabled": true, "days": 30}
    }
  ]' \
  --metrics '[
    {
      "category": "AllMetrics",
      "enabled": true,
      "retentionPolicy": {"enabled": true, "days": 30}
    }
  ]'

echo "✅ Diagnostic logging configured"

# ============================================
# Step 6: Test Connection
# ============================================
echo "\n🧪 Testing connection..."

# Test using curl (requires jq for JSON parsing)
curl -s -H "api-key: $ADMIN_KEY" \
  "$SEARCH_ENDPOINT/indexes?api-version=2023-11-01" | jq '.value | length'

if [ $? -eq 0 ]; then
    echo "✅ Connection test successful"
else
    echo "❌ Connection test failed - check firewall rules"
fi

# ============================================
# Step 7: Summary
# ============================================
echo "\n✅ Setup Complete!"
echo "================================================"
echo "Service Name: $SEARCH_SERVICE_NAME"
echo "Endpoint: $SEARCH_ENDPOINT"
echo "SKU: $SKU ($REPLICA_COUNT replicas, $PARTITION_COUNT partitions)"
echo "Estimated Cost: \$$(calculate_monthly_cost $SKU $REPLICA_COUNT $PARTITION_COUNT)/month"
echo "================================================"
echo "\nNext steps:"
echo "1. Load .env file: source .env"
echo "2. Create search index (see Index Configuration section)"
echo "3. Upload documents"
echo "4. Configure monitoring (see Azure Monitor Logging guide)"
```

**Helper function for cost calculation**:
```bash
calculate_monthly_cost() {
    local sku=$1
    local replicas=$2
    local partitions=$3
    
    # Base costs per tier (as of 2024)
    case $sku in
        "free")
            echo 0
            ;;
        "basic")
            base=75
            ;;
        "standard")
            base=250
            ;;
        "standard2")
            base=1000
            ;;
        "standard3")
            base=2000
            ;;
        "storage_optimized_l1")
            base=2000
            ;;
        "storage_optimized_l2")
            base=4000
            ;;
        *)
            echo "Unknown SKU"
            return
            ;;
    esac
    
    # Calculate total (base * replicas * partitions)
    total=$((base * replicas * partitions))
    echo $total
}
```

**Running the script**:
```bash
# Save script as create-search-service.sh
chmod +x create-search-service.sh

# Run with custom parameters
SEARCH_SERVICE_NAME="my-search-prod" \
LOCATION="westeurope" \
REPLICA_COUNT=3 \
./create-search-service.sh

# Or edit variables in script and run
./create-search-service.sh
```

**CLI commands for common operations**:

```bash
# List all search services in subscription
az search service list --output table

# Get service status
az search service show \
  --name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "{Name:name, Status:provisioningState, SKU:sku.name, Replicas:replicaCount, Partitions:partitionCount}" \
  --output table

# Scale up replicas (increase throughput)
az search service update \
  --name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP \
  --replica-count 3

# Scale up partitions (increase storage)
az search service update \
  --name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP \
  --partition-count 2

# Regenerate admin key (if compromised)
az search admin-key renew \
  --service-name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP \
  --key-kind primary

# Create additional query key (for specific apps)
az search query-key create \
  --service-name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP \
  --name "app-query-key"

# Delete search service (careful - no undo!)
az search service delete \
  --name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP \
  --yes  # Skip confirmation (use cautiously)

# Check service quota/usage
az search service quota show \
  --service-name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP
```

**CLI troubleshooting**:
```
Issue: "SubscriptionNotRegistered for Microsoft.Search"
Fix: Register resource provider
     az provider register --namespace Microsoft.Search
     Wait 5 minutes, then retry service creation

Issue: "Service name already taken"
Fix: Search service names are globally unique
     Try different name: search-eval-prod-002, search-eval-westeu-001, etc.
     Check availability: az search service check-name --name YOUR-NAME

Issue: "InvalidParameter: semantic-search not available in Free tier"
Fix: Remove --semantic-search flag for Free tier
     Or upgrade to Standard tier

Issue: "QuotaExceeded: Standard tier limit reached"
Fix: Delete unused services or request quota increase
     Check quota: az search service quota show
     Request increase: Azure Portal → Support → Quota increase
```

**CLI Success Checklist**:
- ✅ Command exits with status code 0 (no errors)
- ✅ Service visible in `az search service list`
- ✅ Endpoint returns 403 when accessed (not 404 or timeout)
- ✅ Admin keys retrieved successfully
- ✅ .env file created with credentials
- ✅ Test connection succeeds (curl or Python script)

### Option 3: ARM Template (Infrastructure-as-Code)

**When to use ARM templates**: Enterprise deployments with strict compliance, multi-environment consistency (dev/staging/prod), GitOps workflows, Azure DevOps/GitHub Actions pipelines.

**Advantages over CLI**:
- Declarative (describe desired state, Azure handles how to get there)
- Idempotent (run multiple times safely, updates only what changed)
- Parameterized (single template for dev/staging/prod with different parameters)
- Dependency management (automatically handles resource creation order)
- Full audit trail (templates in Git = infrastructure change history)

**Production-ready ARM template** (search-service.json):

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "searchServiceName": {
      "type": "string",
      "metadata": {
        "description": "Name of the Azure AI Search service (globally unique)"
      },
      "minLength": 2,
      "maxLength": 60
    },
    "location": {
      "type": "string",
      "defaultValue": "[resourceGroup().location]",
      "metadata": {
        "description": "Azure region for deployment"
      }
    },
    "sku": {
      "type": "string",
      "defaultValue": "standard",
      "allowedValues": [
        "free",
        "basic",
        "standard",
        "standard2",
        "standard3",
        "storage_optimized_l1",
        "storage_optimized_l2"
      ],
      "metadata": {
        "description": "Pricing tier for search service"
      }
    },
    "replicaCount": {
      "type": "int",
      "defaultValue": 1,
      "minValue": 1,
      "maxValue": 12,
      "metadata": {
        "description": "Number of replicas (1=dev, 2+=prod for HA)"
      }
    },
    "partitionCount": {
      "type": "int",
      "defaultValue": 1,
      "minValue": 1,
      "maxValue": 12,
      "metadata": {
        "description": "Number of partitions (scale for storage)"
      }
    },
    "hostingMode": {
      "type": "string",
      "defaultValue": "default",
      "allowedValues": ["default", "highDensity"],
      "metadata": {
        "description": "Hosting mode (highDensity for many small indexes)"
      }
    },
    "semanticSearch": {
      "type": "string",
      "defaultValue": "disabled",
      "allowedValues": ["disabled", "free", "standard"],
      "metadata": {
        "description": "Semantic search tier (free=1K queries/mo, standard=unlimited at $500/mo)"
      }
    },
    "publicNetworkAccess": {
      "type": "string",
      "defaultValue": "enabled",
      "allowedValues": ["enabled", "disabled"],
      "metadata": {
        "description": "Public network access (disable for private endpoint only)"
      }
    },
    "tags": {
      "type": "object",
      "defaultValue": {
        "Environment": "Production",
        "ManagedBy": "ARM-Template"
      },
      "metadata": {
        "description": "Resource tags for organization and cost tracking"
      }
    }
  },
  "variables": {
    "searchServiceResourceId": "[resourceId('Microsoft.Search/searchServices', parameters('searchServiceName'))]"
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
      "tags": "[parameters('tags')]",
      "properties": {
        "replicaCount": "[parameters('replicaCount')]",
        "partitionCount": "[parameters('partitionCount')]",
        "hostingMode": "[parameters('hostingMode')]",
        "publicNetworkAccess": "[parameters('publicNetworkAccess')]",
        "semanticSearch": "[parameters('semanticSearch')]",
        "authOptions": {
          "aadOrApiKey": {
            "aadAuthFailureMode": "http401WithBearerChallenge"
          }
        },
        "disableLocalAuth": false,
        "encryptionWithCmk": {
          "enforcement": "Unspecified"
        }
      }
    }
  ],
  "outputs": {
    "searchServiceEndpoint": {
      "type": "string",
      "value": "[concat('https://', parameters('searchServiceName'), '.search.windows.net')]",
      "metadata": {
        "description": "Search service endpoint URL"
      }
    },
    "searchServiceId": {
      "type": "string",
      "value": "[variables('searchServiceResourceId')]",
      "metadata": {
        "description": "Resource ID for the search service"
      }
    },
    "searchServiceName": {
      "type": "string",
      "value": "[parameters('searchServiceName')]",
      "metadata": {
        "description": "Name of the deployed search service"
      }
    }
  }
}
```

**Parameter files** (for different environments):

**parameters-dev.json**:
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "searchServiceName": {
      "value": "search-eval-dev-001"
    },
    "location": {
      "value": "eastus"
    },
    "sku": {
      "value": "basic"
    },
    "replicaCount": {
      "value": 1
    },
    "partitionCount": {
      "value": 1
    },
    "semanticSearch": {
      "value": "disabled"
    },
    "tags": {
      "value": {
        "Environment": "Development",
        "Project": "SearchEvaluation",
        "ManagedBy": "ARM-Template",
        "CostCenter": "Engineering"
      }
    }
  }
}
```

**parameters-prod.json**:
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "searchServiceName": {
      "value": "search-eval-prod-001"
    },
    "location": {
      "value": "eastus"
    },
    "sku": {
      "value": "standard"
    },
    "replicaCount": {
      "value": 2
    },
    "partitionCount": {
      "value": 1
    },
    "semanticSearch": {
      "value": "standard"
    },
    "publicNetworkAccess": {
      "value": "disabled"
    },
    "tags": {
      "value": {
        "Environment": "Production",
        "Project": "SearchEvaluation",
        "ManagedBy": "ARM-Template",
        "CostCenter": "Engineering",
        "Tier": "Critical"
      }
    }
  }
}
```

**Deploying ARM template**:

```bash
# ============================================
# Deploy ARM Template - Complete Script
# ============================================

RESOURCE_GROUP="rg-search-evaluation"
LOCATION="eastus"
TEMPLATE_FILE="search-service.json"
ENVIRONMENT="dev"  # or "prod", "staging"

# Create resource group if doesn't exist
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Deploy with parameter file (recommended)
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file $TEMPLATE_FILE \
  --parameters "parameters-${ENVIRONMENT}.json" \
  --name "search-service-deployment-$(date +%Y%m%d-%H%M%S)" \
  --verbose

# Or deploy with inline parameters
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file $TEMPLATE_FILE \
  --parameters searchServiceName="search-eval-dev-002" \
               sku="basic" \
               replicaCount=1 \
               partitionCount=1 \
  --name "search-service-dev"

# Validate template before deploying (--what-if mode)
az deployment group what-if \
  --resource-group $RESOURCE_GROUP \
  --template-file $TEMPLATE_FILE \
  --parameters "parameters-prod.json"

# Get deployment outputs
az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name "search-service-deployment-latest" \
  --query "properties.outputs"

# Example: Extract endpoint from outputs
SEARCH_ENDPOINT=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name "search-service-deployment-latest" \
  --query "properties.outputs.searchServiceEndpoint.value" \
  --output tsv)

echo "Search Endpoint: $SEARCH_ENDPOINT"
```

**Advanced ARM template features**:

**1. Linked templates** (modular infrastructure):
```json
{
  "resources": [
    {
      "type": "Microsoft.Resources/deployments",
      "apiVersion": "2021-04-01",
      "name": "searchServiceDeployment",
      "properties": {
        "mode": "Incremental",
        "templateLink": {
          "uri": "https://raw.githubusercontent.com/your-org/templates/main/search-service.json"
        },
        "parameters": {
          "searchServiceName": {
            "value": "[parameters('searchServiceName')]"
          }
        }
      }
    }
  ]
}
```

**2. Conditional deployment** (dev vs. prod differences):
```json
{
  "resources": [
    {
      "condition": "[equals(parameters('environment'), 'production')]",
      "type": "Microsoft.Network/privateEndpoints",
      "apiVersion": "2021-05-01",
      "name": "[concat(parameters('searchServiceName'), '-private-endpoint')]",
      "properties": {
        "subnet": {
          "id": "[parameters('subnetId')]"
        },
        "privateLinkServiceConnections": [
          {
            "name": "search-private-link",
            "properties": {
              "privateLinkServiceId": "[variables('searchServiceResourceId')]",
              "groupIds": ["searchService"]
            }
          }
        ]
      }
    }
  ]
}
```

**3. Integration with Azure Key Vault** (secure parameter storage):
```bash
# Store admin key in Key Vault after deployment
ADMIN_KEY=$(az search admin-key show \
  --service-name $SEARCH_SERVICE_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "primaryKey" \
  --output tsv)

az keyvault secret set \
  --vault-name "kv-search-secrets" \
  --name "search-admin-key" \
  --value "$ADMIN_KEY"

# Reference from application
# No need to hardcode keys in .env or code
```

**ARM template troubleshooting**:
```
Issue: "ValidationError: Template parameter 'replicaCount' exceeds maximum"
Fix: Check sku limits (Free: 1 replica max, Basic: 3 max, Standard: 12 max)
     Adjust replicaCount in parameter file

Issue: "DeploymentFailed: Semantic search not available in Free tier"
Fix: Change semanticSearch parameter to "disabled" for Free tier
     Or upgrade sku to "standard"

Issue: "Conflict: Service already exists with different configuration"
Fix: ARM deployments are incremental by default (updates existing resources)
     To recreate: Delete service first, or use mode: "Complete" (dangerous - deletes other resources)

Issue: "What-if shows no changes, but deployment fails"
Fix: what-if doesn't validate quota limits or name availability
     Run actual deployment to see real errors
```

**ARM template best practices**:
1. **Use parameter files** (not inline parameters) for version control
2. **Validate with what-if** before production deployments
3. **Name deployments** with timestamps for audit trail
4. **Store templates in Git** (infrastructure-as-code)
5. **Use outputs** to pass values to other templates or scripts
6. **Tag resources** for cost tracking and resource management
7. **Test in dev** before deploying to production

**CI/CD integration** (Azure DevOps pipeline):
```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
      - main
  paths:
    include:
      - infra/search-service.json
      - infra/parameters-*.json

stages:
  - stage: DeployDev
    jobs:
      - job: DeploySearchService
        steps:
          - task: AzureResourceManagerTemplateDeployment@3
            inputs:
              azureSubscription: 'Azure-Service-Connection'
              resourceGroupName: 'rg-search-dev'
              location: 'East US'
              templateLocation: 'Linked artifact'
              csmFile: 'infra/search-service.json'
              csmParametersFile: 'infra/parameters-dev.json'
              deploymentMode: 'Incremental'
```

**ARM Success Checklist**:
- ✅ Template validates without errors (`az deployment group validate`)
- ✅ what-if shows expected changes
- ✅ Deployment succeeds with provisioningState = "Succeeded"
- ✅ Outputs contain expected values (endpoint, name, resourceId)
- ✅ Service accessible via outputted endpoint
- ✅ Template and parameters checked into Git

### Option 4: Python SDK (Programmatic Management)

**When to use Python SDK**: Building deployment automation tools, multi-tenant SaaS (programmatically create search services per customer), dynamic scaling based on application metrics, integration with existing Python infrastructure tooling.

**Advantages over CLI/Portal**:
- Full programmatic control (create, update, delete, scale services in code)
- Integration with existing Python applications (no shell scripting needed)
- Dynamic resource creation (e.g., create search service per new customer)
- Error handling and retries built into your application
- Supports complex workflows (check quota → create service → configure → validate)

**Prerequisites**:
```bash
pip install azure-mgmt-search azure-identity
```

**Complete Python service creation script**:

```python
"""
create_search_service.py
Complete script for creating and managing Azure AI Search services programmatically
"""

from azure.identity import DefaultAzureCredential
from azure.mgmt.search import SearchManagementClient
from azure.mgmt.search.models import SearchService, Sku, HostingMode, PublicNetworkAccess
import os
import time

# Configuration
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = "rg-search-evaluation"
LOCATION = "eastus"
SEARCH_SERVICE_NAME = "search-eval-dev-003"
SKU_NAME = "standard"  # free, basic, standard, standard2, standard3
REPLICA_COUNT = 1
PARTITION_COUNT = 1

def create_search_service():
    """
    Create Azure AI Search service with full configuration
    Returns: Search service endpoint and admin key
    """
    
    # Authenticate using DefaultAzureCredential
    # Tries multiple auth methods: environment vars, managed identity, Azure CLI
    credential = DefaultAzureCredential()
    
    # Initialize Search Management Client
    search_mgmt_client = SearchManagementClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID
    )
    
    print(f"Creating search service: {SEARCH_SERVICE_NAME}")
    print(f"  Location: {LOCATION}")
    print(f"  SKU: {SKU_NAME}")
    print(f"  Replicas: {REPLICA_COUNT}, Partitions: {PARTITION_COUNT}")
    
    # Check if service name is available
    name_availability = search_mgmt_client.services.check_name_availability(
        name=SEARCH_SERVICE_NAME
    )
    
    if not name_availability.is_name_available:
        print(f"❌ Service name '{SEARCH_SERVICE_NAME}' is not available")
        print(f"   Reason: {name_availability.reason}")
        print(f"   Message: {name_availability.message}")
        return None
    
    print("✅ Service name is available")
    
    # Define search service configuration
    search_service = SearchService(
        location=LOCATION,
        sku=Sku(name=SKU_NAME),
        replica_count=REPLICA_COUNT,
        partition_count=PARTITION_COUNT,
        hosting_mode=HostingMode.DEFAULT,  # or HostingMode.HIGH_DENSITY
        public_network_access=PublicNetworkAccess.ENABLED,
        # For production: consider PublicNetworkAccess.DISABLED with private endpoint
        tags={
            "Environment": "Development",
            "Project": "SearchEvaluation",
            "ManagedBy": "PythonSDK",
            "CostCenter": "Engineering"
        }
    )
    
    # Create the search service (async operation)
    try:
        poller = search_mgmt_client.services.begin_create_or_update(
            resource_group_name=RESOURCE_GROUP,
            search_service_name=SEARCH_SERVICE_NAME,
            service=search_service
        )
        
        print("⏳ Service creation in progress...")
        print("   This typically takes 2-5 minutes")
        
        # Wait for completion
        search_service_result = poller.result()
        
        print(f"✅ Search service created successfully")
        print(f"   Provisioning State: {search_service_result.provisioning_state}")
        print(f"   Status: {search_service_result.status}")
        
    except Exception as e:
        print(f"❌ Service creation failed: {str(e)}")
        return None
    
    # Get service endpoint
    endpoint = f"https://{SEARCH_SERVICE_NAME}.search.windows.net"
    
    # Retrieve admin keys
    admin_keys = search_mgmt_client.admin_keys.get(
        resource_group_name=RESOURCE_GROUP,
        search_service_name=SEARCH_SERVICE_NAME
    )
    
    primary_admin_key = admin_keys.primary_key
    
    # Retrieve query keys
    query_keys = list(search_mgmt_client.query_keys.list_by_search_service(
        resource_group_name=RESOURCE_GROUP,
        search_service_name=SEARCH_SERVICE_NAME
    ))
    
    query_key = query_keys[0].key if query_keys else None
    
    print("\n" + "="*60)
    print("DEPLOYMENT SUMMARY")
    print("="*60)
    print(f"Service Name: {SEARCH_SERVICE_NAME}")
    print(f"Endpoint: {endpoint}")
    print(f"Admin Key: {primary_admin_key}")
    print(f"Query Key: {query_key}")
    print(f"\nSave to .env file:")
    print(f'AZURE_SEARCH_ENDPOINT="{endpoint}"')
    print(f'AZURE_SEARCH_ADMIN_KEY="{primary_admin_key}"')
    print(f'AZURE_SEARCH_QUERY_KEY="{query_key}"')
    print("="*60)
    
    return {
        "endpoint": endpoint,
        "admin_key": primary_admin_key,
        "query_key": query_key
    }

def scale_search_service(replica_count=None, partition_count=None):
    """
    Scale existing search service (replicas and/or partitions)
    """
    credential = DefaultAzureCredential()
    search_mgmt_client = SearchManagementClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID
    )
    
    # Get current service configuration
    current_service = search_mgmt_client.services.get(
        resource_group_name=RESOURCE_GROUP,
        search_service_name=SEARCH_SERVICE_NAME
    )
    
    # Update replica/partition counts
    if replica_count is not None:
        current_service.replica_count = replica_count
    if partition_count is not None:
        current_service.partition_count = partition_count
    
    print(f"Scaling {SEARCH_SERVICE_NAME}...")
    print(f"  New Replicas: {current_service.replica_count}")
    print(f"  New Partitions: {current_service.partition_count}")
    
    # Apply changes
    poller = search_mgmt_client.services.begin_create_or_update(
        resource_group_name=RESOURCE_GROUP,
        search_service_name=SEARCH_SERVICE_NAME,
        service=current_service
    )
    
    result = poller.result()
    print(f"✅ Service scaled successfully: {result.provisioning_state}")

def delete_search_service():
    """
    Delete search service (cleanup)
    """
    credential = DefaultAzureCredential()
    search_mgmt_client = SearchManagementClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID
    )
    
    print(f"Deleting {SEARCH_SERVICE_NAME}...")
    search_mgmt_client.services.delete(
        resource_group_name=RESOURCE_GROUP,
        search_service_name=SEARCH_SERVICE_NAME
    )
    print("✅ Service deleted")

def list_search_services():
    """
    List all search services in subscription
    """
    credential = DefaultAzureCredential()
    search_mgmt_client = SearchManagementClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID
    )
    
    print("Search services in subscription:")
    services = search_mgmt_client.services.list_by_subscription()
    
    for service in services:
        print(f"  - {service.name} ({service.sku.name}) in {service.location}")

if __name__ == "__main__":
    # Example: Create service
    credentials = create_search_service()
    
    # Example: Scale service later
    # scale_search_service(replica_count=2, partition_count=1)
    
    # Example: List all services
    # list_search_services()
    
    # Example: Delete service (cleanup)
    # delete_search_service()
```

**Running the script**:
```bash
# Set subscription ID
export AZURE_SUBSCRIPTION_ID="your-subscription-id"

# Authenticate (uses az login credentials)
az login

# Run the script
python create_search_service.py
```

**Common Python SDK operations**:

```python
# Get service details
service = search_mgmt_client.services.get(
    resource_group_name=RESOURCE_GROUP,
    search_service_name=SEARCH_SERVICE_NAME
)
print(f"Status: {service.status}")
print(f"Provisioning State: {service.provisioning_state}")
print(f"Replicas: {service.replica_count}, Partitions: {service.partition_count}")

# Regenerate admin key
new_keys = search_mgmt_client.admin_keys.regenerate(
    resource_group_name=RESOURCE_GROUP,
    search_service_name=SEARCH_SERVICE_NAME,
    key_kind="primary"  # or "secondary"
)
print(f"New Primary Key: {new_keys.primary_key}")

# Create query key
query_key = search_mgmt_client.query_keys.create(
    resource_group_name=RESOURCE_GROUP,
    search_service_name=SEARCH_SERVICE_NAME,
    name="app-query-key"
)
print(f"New Query Key: {query_key.key}")

# Check quota usage (requires azure-mgmt-resource)
from azure.mgmt.resource import SubscriptionClient
sub_client = SubscriptionClient(credential)
usages = sub_client.subscriptions.list_locations(SUBSCRIPTION_ID)
# Note: Quota API varies by resource provider
```

**Python SDK troubleshooting**:
```
Issue: "DefaultAzureCredential failed to retrieve token"
Cause: Not logged into Azure CLI or missing environment variables
Fix: Run `az login` first, or set AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET

Issue: "AuthorizationFailed: The client does not have authorization to perform action"
Cause: Insufficient permissions (need Contributor or Search Service Contributor)
Fix: Request role assignment from subscription owner

Issue: "ServiceUnavailable: The subscription is disabled"
Cause: Azure subscription expired or suspended
Fix: Reactivate subscription in Azure Portal → Cost Management

Issue: "begin_create_or_update hangs indefinitely"
Cause: Network timeout or Azure outage
Fix: Set timeout: poller.result(timeout=600)  # 10 minutes
     Check Azure status: https://status.azure.com

Issue: "InvalidTemplate: SKU 'standard4' is not valid"
Cause: Invalid SKU name (typo)
Fix: Use correct names: free, basic, standard, standard2, standard3
```

**Python SDK best practices**:
1. **Use DefaultAzureCredential** (tries multiple auth methods automatically)
2. **Handle pollers properly** (wait for async operations to complete)
3. **Implement retry logic** (transient failures happen, especially during scaling)
4. **Store credentials securely** (use Azure Key Vault, never hardcode keys)
5. **Tag resources** (enables cost tracking and resource management)
6. **Validate inputs** (check name availability before attempting creation)
7. **Log operations** (audit trail for troubleshooting and compliance)

**Python SDK Success Checklist**:
- ✅ `DefaultAzureCredential()` authenticates successfully
- ✅ Service name availability check passes
- ✅ `begin_create_or_update` completes with `provisioning_state = "Succeeded"`
- ✅ Admin key and query key retrieved successfully
- ✅ Credentials saved to `.env` or Key Vault
- ✅ Test connectivity with `SearchClient` (see validation script in Prerequisites)

---

## Common Setup Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `az login` fails | Firewall blocking auth | Use `az login --use-device-code` |
| `ModuleNotFoundError: azure` | Wrong Python version or venv not activated | Verify Python 3.8+, activate venv, reinstall |
| `AuthenticationError` | Wrong API key or expired | Regenerate keys in Azure Portal |
| `ResourceNotFound` | Service name typo or wrong subscription | Verify `AZURE_SEARCH_ENDPOINT`, check subscription |
| SSL certificate errors | Corporate proxy/firewall | Configure proxy: `export HTTPS_PROXY=http://proxy:port` |

---

## Service Creation

**Four methods for creating Azure AI Search service**: Azure Portal (easiest for beginners), Azure CLI (best for automation), ARM templates (infrastructure-as-code), Python SDK (programmatic control). Choose based on your workflow and team's expertise.

### Option 1: Azure Portal

**When to use Portal**: First-time setup, ad-hoc dev environments, visual learners who prefer UI.
**Avoid Portal for**: Production deployments (not repeatable), multi-environment setup (tedious), CI/CD pipelines.

1. **Navigate to Azure Portal**
   - Go to https://portal.azure.com
   - Sign in with your Azure credentials
   - Click **"+ Create a resource"** (top-left)
   - Search for **"Azure AI Search"** (formerly "Azure Cognitive Search")
   - Click **"Create"**

2. **Configure Basic Settings**
   
   **Subscription**:
   - Select your Azure subscription
   - If multiple subscriptions available, choose the one with budget allocated
   - Verify quota availability (see Prerequisites section)
   
   **Resource Group**:
   - Click **"Create new"** for first deployment or select existing
   - Naming convention: `rg-<project>-<environment>-<region>`
   - Example: `rg-searcheval-prod-eastus`
   - Why resource groups matter: Logical container for related resources, enables bulk deletion, cost tracking by group
   
   **Service Name**:
   - Must be globally unique across all Azure customers
   - Naming convention: `<project>-<environment>-<instance>`
   - Example: `searcheval-prod-001` (instance number for scaling)
   - Constraints: 2-60 chars, lowercase letters, numbers, hyphens only
   - Tip: Check availability before continuing (Portal shows green checkmark if available)
   
   **Location (Region)**:
   - Critical decision affecting latency and cost
   - Choose region closest to majority of users for lowest query latency
   - Alternative: Choose region closest to data source for fastest indexing
   - Consider data residency requirements (GDPR, SOC compliance)
   - Region availability: Not all features available in all regions (check semantic search availability)
   - Recommended regions for production:
     * East US, East US 2: Largest capacity, most features, competitive pricing
     * West Europe: Good for EMEA users, GDPR-friendly
     * Southeast Asia: Good for APAC users
   
   **Pricing Tier**:
   - See detailed guidance in [Service Tier Selection](#service-tier-selection)
   - Quick guidance:
     * Development/testing: **Free** (1 service per subscription)
     * Small production (<1M docs, <15 QPS): **Basic** ($75/month)
     * Standard production (<15M docs, <50 QPS): **Standard S1** ($250/month)
     * Large-scale (15M+ docs, 50-200 QPS): **Standard S2** ($1,000/month)
     * Enterprise (60M+ docs, 200+ QPS): **Standard S3** ($2,000/month)
   - Note: Can change tier later, but requires service recreation (not in-place upgrade)

3. **Configure Scale Settings** (Standard+ tiers only)
   
   **Replicas** (for query performance and high availability):
   - Start with 1 replica for dev, 2+ for production
   - Each replica handles queries independently (adds to QPS capacity)
   - 2 replicas provide HA (99.9% SLA for queries)
   - 3 replicas provide HA for both queries and indexing (99.9% SLA)
   - Cost: Each additional replica costs 1x base tier price
   - Example: S1 with 2 replicas = $250 + $250 = $500/month
   
   **Partitions** (for storage capacity):
   - Start with 1 partition unless you have >25GB data (S1) or >100GB (S2)
   - Each partition adds storage capacity:
     * S1: 25GB per partition (max 12 partitions = 300GB)
     * S2: 100GB per partition (max 12 partitions = 1.2TB)
     * S3: 200GB per partition (max 12 partitions = 2.4TB)
   - Partitions also add to indexing throughput (parallel indexing)
   - Cost: Each additional partition costs 1x base tier price
   - Example: S1 with 2 partitions = $250 + $250 = $500/month
   
   **Semantic Search** (Standard+ tiers only):
   - Toggle "Semantic search" to **Standard** if needed
   - Adds $500/month after 1,000 free queries/month
   - See `03-indexing-strategies.md` Section 4 for when to enable
   - Recommendation: Start disabled, enable later if needed

4. **Review and Create**
   - Review all configuration (subscription, tier, replicas, partitions, cost estimate)
   - Azure shows estimated cost per month (verify against budget)
   - Click **"Create"**
   - Wait 2-5 minutes for deployment (can take up to 10 minutes for large tiers)
   - Click **"Go to resource"** when deployment completes

5. **Post-Creation Steps**
   - Copy **URL**: Found in Overview → Essentials → URL
     * Format: `https://YOUR-SERVICE.search.windows.net`
   - Copy **Admin Key**: Settings → Keys → Primary admin key
     * Use admin key for index creation, deletion (write operations)
   - Copy **Query Key**: Settings → Keys → Manage query keys → Default query key
     * Use query key for search queries (read-only, safer for client apps)
   - Store keys securely (Azure Key Vault recommended, .env file for dev only)

**Portal creation pros/cons**:
- ✅ Pros: Visual, easy for beginners, immediate feedback, no scripting needed
- ❌ Cons: Not repeatable, error-prone for multi-env, no version control, tedious for scale

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

---

## Service Tier Selection

**Why tier selection matters**: Right-sizing saves significant costs (over-provisioning wastes $10K+/year, under-provisioning causes throttling and user churn). This section helps you choose the optimal tier based on actual workload requirements, not guesswork.

### Comprehensive Tier Comparison

| Tier | Price/Month¹ | Storage/Partition | Max Docs² | QPS³ | Max Replicas | Max Partitions | SLA | Key Features | Best For |
|------|--------------|-------------------|-----------|------|--------------|----------------|-----|--------------|----------|
| **Free** | $0 | 50 MB | 10K | 3 | 1 | 1 | None | Basic search, no semantic | Learning, POCs, demos |
| **Basic** | ~$75 | 2 GB | 1M | 15 | 3 | 1 | 99.9% | HA with 2+ replicas | Small production apps |
| **Standard S1** | ~$250 | 25 GB | 15M | 50 | 12 | 12 | 99.9% | Full features, vector, semantic⁴ | Most production apps |
| **Standard S2** | ~$1,000 | 100 GB | 60M | 200 | 12 | 12 | 99.9% | Higher throughput, more storage | Large-scale apps |
| **Standard S3** | ~$2,000 | 200 GB | 120M+ | 400 | 12 | 12 | 99.9% | Enterprise scale | Multi-tenant SaaS, high-volume |
| **Standard S3 HD** | ~$2,000 | 200 GB | 1B+ small | 400 | 3 | 1 | 99.9% | High-density (many small indexes) | Multi-tenant with isolated indexes |
| **Storage L1** | ~$1,000 | 1 TB | 60M+ | 100 | 12 | 12 | 99.9% | Storage-optimized, lower QPS | Archive search, compliance |
| **Storage L2** | ~$2,000 | 2 TB | 120M+ | 200 | 12 | 12 | 99.9% | Massive storage | Large document repositories |

**Notes**:
1. Prices are base tier cost (1 replica, 1 partition). Each additional replica or partition adds 1x base cost.
2. Document count is approximate, actual depends on document size and complexity.
3. QPS (Queries Per Second) is per replica. 2 replicas double capacity.
4. Semantic search: Free tier includes 1,000 queries/month, then $500/month for Standard semantic.

### Tier Selection Decision Framework

**Use this framework to choose the right tier based on your specific requirements**:

```python
def recommend_service_tier(requirements):
    """
    Recommend Azure AI Search tier based on workload requirements
    Returns: Recommended tier, estimated cost, and reasoning
    """
    
    # Extract requirements
    doc_count = requirements.get('document_count', 0)
    doc_size_kb = requirements.get('avg_document_size_kb', 10)
    qps_needed = requirements.get('queries_per_second', 10)
    high_availability = requirements.get('high_availability', False)
    semantic_search = requirements.get('semantic_search', False)
    vector_search = requirements.get('vector_search', False)
    is_production = requirements.get('production', False)
    
    # Calculate storage needed
    storage_gb = (doc_count * doc_size_kb) / (1024 * 1024)
    storage_gb *= 1.2  # 20% overhead for indexes
    
    # Decision logic (from smallest to largest)
    
    # FREE TIER
    if doc_count < 10000 and not is_production and not semantic_search:
        return {
            'tier': 'Free',
            'replicas': 1,
            'partitions': 1,
            'monthly_cost': 0,
            'reasoning': (
                f"Free tier sufficient for {doc_count:,} documents. "
                f"Supports basic search and vector search. "
                "Limitations: 50MB storage, 3 QPS, no SLA, no semantic search."
            ),
            'warnings': [
                "No SLA - not suitable for production",
                "Limited to 3 QPS - will throttle under load",
                "Only 1 free service per subscription"
            ]
        }
    
    # BASIC TIER
    if (doc_count < 1_000_000 and 
        storage_gb < 2 and 
        qps_needed < 15 and
        not vector_search):  # Basic doesn't support vector well
        
        replicas = 2 if high_availability else 1
        cost = 75 * replicas  # Basic = $75/month
        
        return {
            'tier': 'Basic',
            'replicas': replicas,
            'partitions': 1,
            'monthly_cost': cost,
            'reasoning': (
                f"Basic tier handles {doc_count:,} documents ({storage_gb:.1f}GB). "
                f"Provides {qps_needed} QPS with {replicas} replicas. "
                f"HA: {'Yes (99.9% SLA)' if high_availability else 'No (single replica)'}"
            ),
            'warnings': [
                "Limited to 2GB storage (no partition scaling)",
                "Vector search possible but not optimized",
                "No semantic search support"
            ] if vector_search else []
        }
    
    # STANDARD S1 (most common production choice)
    if (doc_count < 15_000_000 and 
        storage_gb < 300 and  # S1 = 25GB per partition, max 12 partitions
        qps_needed < 600):     # S1 = 50 QPS per replica, max 12 replicas
        
        # Calculate replicas needed
        replicas_for_qps = max(2 if high_availability else 1, 
                               int(qps_needed / 50) + 1)
        replicas = min(replicas_for_qps, 12)
        
        # Calculate partitions needed
        partitions_for_storage = max(1, int(storage_gb / 25) + 1)
        partitions = min(partitions_for_storage, 12)
        
        # Cost calculation
        search_units = replicas * partitions
        base_cost = 250
        cost = base_cost * search_units
        
        # Add semantic search cost if needed
        if semantic_search:
            cost += 500  # Semantic search = $500/month
        
        return {
            'tier': 'Standard S1',
            'replicas': replicas,
            'partitions': partitions,
            'search_units': search_units,
            'monthly_cost': cost,
            'reasoning': (
                f"S1 handles {doc_count:,} documents ({storage_gb:.1f}GB). "
                f"Provides {replicas * 50} QPS capacity with {replicas} replicas. "
                f"Storage: {partitions * 25}GB ({partitions} partitions). "
                f"{'Semantic search enabled.' if semantic_search else ''}"
            ),
            'warnings': []
        }
    
    # STANDARD S2 (large-scale production)
    if (doc_count < 60_000_000 and
        storage_gb < 1200 and  # S2 = 100GB per partition, max 12
        qps_needed < 2400):     # S2 = 200 QPS per replica, max 12
        
        replicas_for_qps = max(2 if high_availability else 1,
                               int(qps_needed / 200) + 1)
        replicas = min(replicas_for_qps, 12)
        
        partitions_for_storage = max(1, int(storage_gb / 100) + 1)
        partitions = min(partitions_for_storage, 12)
        
        search_units = replicas * partitions
        base_cost = 1000
        cost = base_cost * search_units
        
        if semantic_search:
            cost += 500
        
        return {
            'tier': 'Standard S2',
            'replicas': replicas,
            'partitions': partitions,
            'search_units': search_units,
            'monthly_cost': cost,
            'reasoning': (
                f"S2 handles {doc_count:,} documents ({storage_gb:.1f}GB). "
                f"Provides {replicas * 200} QPS capacity with {replicas} replicas. "
                f"Storage: {partitions * 100}GB ({partitions} partitions)."
            ),
            'warnings': []
        }
    
    # STANDARD S3 (enterprise scale)
    replicas_for_qps = max(2 if high_availability else 1,
                           int(qps_needed / 400) + 1)
    replicas = min(replicas_for_qps, 12)
    
    partitions_for_storage = max(1, int(storage_gb / 200) + 1)
    partitions = min(partitions_for_storage, 12)
    
    search_units = replicas * partitions
    base_cost = 2000
    cost = base_cost * search_units
    
    if semantic_search:
        cost += 500
    
    return {
        'tier': 'Standard S3',
        'replicas': replicas,
        'partitions': partitions,
        'search_units': search_units,
        'monthly_cost': cost,
        'reasoning': (
            f"S3 handles {doc_count:,} documents ({storage_gb:.1f}GB). "
            f"Provides {replicas * 400} QPS capacity with {replicas} replicas. "
            f"Storage: {partitions * 200}GB ({partitions} partitions). "
            "Enterprise-scale for high-volume production workloads."
        ),
        'warnings': [
            f"High cost: ${cost:,}/month - verify budget approval",
            "Consider S3 HD if you need many small indexes (multi-tenant)"
        ]
    }

# Example usage
production_app = {
    'document_count': 2_500_000,
    'avg_document_size_kb': 15,
    'queries_per_second': 75,
    'high_availability': True,
    'semantic_search': True,
    'vector_search': True,
    'production': True
}

recommendation = recommend_service_tier(production_app)
print(f"Recommended Tier: {recommendation['tier']}")
print(f"Configuration: {recommendation['replicas']} replicas, {recommendation['partitions']} partitions")
print(f"Estimated Cost: ${recommendation['monthly_cost']}/month")
print(f"Reasoning: {recommendation['reasoning']}")
if recommendation['warnings']:
    print("Warnings:")
    for warning in recommendation['warnings']:
        print(f"  - {warning}")
```

**Output example**:
```
Recommended Tier: Standard S1
Configuration: 2 replicas, 2 partitions
Estimated Cost: $1,500/month
Reasoning: S1 handles 2,500,000 documents (36.0GB). Provides 100 QPS capacity with 2 replicas. Storage: 50GB (2 partitions). Semantic search enabled.
```

### Real-World Tier Selection Examples

**Example 1: Small e-commerce site**
```
Requirements:
- 50,000 products
- 10 queries/second average, 30 peak
- 5KB per document (250MB total)
- Need high availability (99.9% SLA)

Recommendation: Basic tier, 2 replicas
Cost: $150/month
Reasoning: Small doc count, low QPS, fits in 2GB. HA with 2 replicas gets SLA.
```

**Example 2: Medium knowledge base**
```
Requirements:
- 3 million documents
- 50 queries/second average, 120 peak
- 20KB per document (60GB total)
- Hybrid search (BM25 + vector)
- Semantic reranking

Recommendation: Standard S1, 3 replicas, 3 partitions
Cost: $2,750/month ($250 * 9 search units + $500 semantic)
Reasoning: Need 3 replicas for 150 QPS capacity, 3 partitions for 75GB storage, semantic adds $500.
```

**Example 3: Enterprise SaaS platform**
```
Requirements:
- 25 million documents
- 300 queries/second average, 600 peak
- 25KB per document (625GB total)
- Multi-region (2 regions)
- Semantic search

Recommendation: Standard S2, 4 replicas, 7 partitions per region (2 services)
Cost: $29,000/month per region × 2 = $58,000/month
Reasoning: Need S2 for capacity (800 QPS with 4 replicas, 700GB with 7 partitions). Multi-region requires 2 services.
Warning: Consider cost optimization (traffic manager failover, active-passive instead of active-active).
```

### Capacity Planning Formulas

```python
# Storage estimation
def estimate_storage_gb(doc_count, avg_doc_size_kb):
    """
    Estimate storage needed for Azure AI Search index
    Includes overhead for inverted index, vector index, and metadata
    """
    raw_storage_gb = (doc_count * avg_doc_size_kb) / (1024 * 1024)
    
    # Overhead factors
    inverted_index_overhead = 0.15  # 15% for BM25 inverted index
    vector_index_overhead = 0.25    # 25% for HNSW vector index (if enabled)
    metadata_overhead = 0.05        # 5% for metadata
    
    # Total with overhead
    total_overhead = 1 + inverted_index_overhead + vector_index_overhead + metadata_overhead
    estimated_storage = raw_storage_gb * total_overhead
    
    # Add 20% buffer for growth
    estimated_storage *= 1.2
    
    return estimated_storage

# QPS capacity calculation
def calculate_qps_capacity(tier, replicas):
    """Calculate total QPS capacity"""
    tier_base_qps = {
        'Free': 3,
        'Basic': 15,
        'Standard S1': 50,
        'Standard S2': 200,
        'Standard S3': 400
    }
    
    base = tier_base_qps.get(tier, 50)
    return base * replicas

# Cost estimation
def estimate_monthly_cost(tier, replicas, partitions, semantic_search=False):
    """Estimate total monthly cost"""
    tier_base_cost = {
        'Free': 0,
        'Basic': 75,
        'Standard S1': 250,
        'Standard S2': 1000,
        'Standard S3': 2000,
        'Storage L1': 1000,
        'Storage L2': 2000
    }
    
    base = tier_base_cost.get(tier, 0)
    search_units = replicas * partitions
    total = base * search_units
    
    if semantic_search and tier != 'Free':
        total += 500  # Semantic search = $500/month
    
    return total

# Example: Estimate for specific workload
doc_count = 5_000_000
avg_doc_size_kb = 12

storage_needed = estimate_storage_gb(doc_count, avg_doc_size_kb)
print(f"Storage needed: {storage_needed:.1f} GB")

# S1: 25GB per partition
partitions_needed = int(storage_needed / 25) + 1
print(f"Partitions needed (S1): {partitions_needed}")

# Need 100 QPS capacity
replicas_needed = int(100 / 50) + 1  # S1 = 50 QPS per replica
print(f"Replicas needed (S1): {replicas_needed}")

cost = estimate_monthly_cost('Standard S1', replicas_needed, partitions_needed, semantic_search=True)
print(f"Estimated cost: ${cost:,}/month")
```

### Tier Selection Best Practices

1. **Start smaller, scale up**: Begin with S1, monitor actual usage, scale up if needed
   - Easier to justify cost increase with data than to reduce over-provisioned service
   
2. **Separate dev/staging/prod tiers**: 
   - Dev: Free or Basic (save costs)
   - Staging: Same tier as prod (realistic testing)
   - Prod: Right-sized Standard tier

3. **Monitor actual QPS and storage**: 
   - Use Azure Monitor metrics (see `07-azure-monitor-logging.md`)
   - Review weekly for first month, monthly after stabilization
   - Scale proactively before hitting limits (not reactively after throttling)

4. **Plan for growth**:
   - If growing 20%+ per quarter, provision for 6-month growth
   - Use partitions for storage scaling (can add without downtime)
   - Use replicas for QPS scaling (can add without downtime)

5. **Cost optimization strategies**:
   - Use query keys (not admin keys) in applications (safer, rate-limited)
   - Implement caching layer (Redis, CDN) to reduce QPS hitting Search service
   - Use semantic search selectively (not on every query) to stay under 1K free queries/month
   - Consider reserved capacity for production (1-year or 3-year commitment for 20-30% discount)

6. **When to use S3 HD** (high-density):
   - Multi-tenant SaaS with isolated index per customer
   - Each customer has <200MB data
   - Need 100+ small indexes (S3 HD supports up to 1,000 indexes vs 200 on S3)
   - Tradeoff: Limited to 3 replicas and 1 partition

### Tier Migration (Changing Tiers)

**Important**: Azure AI Search does NOT support in-place tier changes. You must create a new service and migrate data.

**Zero-downtime migration process**:
```bash
# Step 1: Create new service at target tier
az search service create \
  --name search-eval-prod-v2 \
  --resource-group rg-search-evaluation \
  --sku standard2  # upgrading from S1 to S2

# Step 2: Recreate indexes on new service
# Use index definitions from old service
az search index create \
  --service-name search-eval-prod-v2 \
  --name products \
  --fields @index-schema.json

# Step 3: Reindex data (from original data source)
# Trigger indexers or bulk upload

# Step 4: Validate new service (run test queries)

# Step 5: Update application endpoint to new service
# Use feature flags or gradual rollout

# Step 6: Monitor for 24-48 hours

# Step 7: Delete old service (after validation)
az search service delete \
  --name search-eval-prod-v1 \
  --resource-group rg-search-evaluation
```

**Migration downtime alternatives**:
- **Parallel services**: Run both old and new services during migration, switch traffic via DNS/load balancer
- **Indexer-based**: If using Azure indexers, just point indexers at new service (automatic reindexing)
- **Blue-green deployment**: Keep old service as fallback, switch back if issues

### Tier Selection Checklist

Before choosing a tier, answer these questions:

- ✅ How many documents? (Current and 12-month projection)
- ✅ Average document size? (Check sample: `doc_size = len(json.dumps(doc))`)
- ✅ Expected QPS? (Use analytics: average, p95, p99, peak)
- ✅ High availability required? (Production = yes = 2+ replicas)
- ✅ Semantic search needed? (Adds $500/month, budget approved?)
- ✅ Vector search needed? (Requires Standard S1+)
- ✅ Storage growth rate? (Plan partitions for 6-12 month growth)
- ✅ Budget limit? (Get approval for estimated cost before provisioning)

---

## Index Configuration

**Why index configuration matters**: The index schema determines what you can search, how you can search it, and how much it costs. Poor schema design leads to incomplete search results, slow queries, or impossibility to implement needed features later. This section covers the critical decisions when designing your index schema.
            'monthly_cost': 1000,
            'reason': 'Large-scale production workload'
        }
    
    return {
        'tier': 'Standard S3',
        'monthly_cost': 2000,
        'reason': 'Enterprise-scale workload'
---

## Index Configuration

**Why index configuration matters**: The index schema determines what you can search, how you can search it, and how much it costs. Poor schema design leads to incomplete search results, slow queries, or impossibility to implement needed features later. This section covers the critical decisions when designing your index schema.

### Index Schema Fundamentals

**An Azure AI Search index consists of**:
1. **Fields**: The structure of your documents (like database columns)
2. **Analyzers**: How text is processed for search (tokenization, stemming, stop words)
3. **Scoring Profiles**: Custom relevance boosting (optional)
4. **Suggesters**: Autocomplete configuration (optional)
5. **CORS Options**: Cross-origin access for browser apps (optional)
6. **Vector Search Configuration**: HNSW algorithm settings for vector fields (optional)

### Field Types and Attributes

**Core field attributes** (choose carefully, **cannot change after index creation**):

| Attribute | Description | Use Case | Cost Impact |
|-----------|-------------|----------|-------------|
| **key** | Unique document identifier | Required for exactly 1 field (usually `id`) | None |
| **searchable** | Full-text search enabled | Fields users will search (title, content, description) | +storage (inverted index) |
| **filterable** | Supports filter queries (`$filter=category eq 'Electronics'`) | Facets, filters, sorting | +memory |
| **sortable** | Supports `$orderby` | Fields users will sort by (date, price, rating) | +memory |
| **facetable** | Supports faceted navigation | Category counts, price ranges | +memory |
| **retrievable** | Returned in search results | Most fields (default=true) | None |
| **analyzer** | Text processing rules | Language-specific (en.microsoft, fr.lucene, etc.) | +processing |

**Field data types**:

```python
from azure.search.documents.indexes.models import SearchFieldDataType

# String fields
SearchFieldDataType.String              # Text (max 32KB for searchable, 16MB for non-searchable)

# Numeric fields
SearchFieldDataType.Int32               # 32-bit integer
SearchFieldDataType.Int64               # 64-bit integer  
SearchFieldDataType.Double              # 64-bit float

# Date/time
SearchFieldDataType.DateTimeOffset      # ISO 8601 date-time

# Boolean
SearchFieldDataType.Boolean             # true/false

# Geospatial
SearchFieldDataType.GeographyPoint      # Lat/lon coordinates

# Vector (for vector search)
SearchFieldDataType.Collection(SearchFieldDataType.Single)  # Float array for embeddings

# Complex types
SearchFieldDataType.ComplexType         # Nested object (e.g., address with street, city, zip)
```

### Complete Index Schema Example

**Realistic e-commerce product index** (demonstrates all major features):

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
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticSearch,
    ScoringProfile,
    TextWeights
)
from azure.core.credentials import AzureKeyCredential

# Initialize index client
endpoint = "https://YOUR-SERVICE.search.windows.net"
api_key = "YOUR-ADMIN-KEY"

index_client = SearchIndexClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key)
)

def create_production_index(index_name):
    """
    Create production-ready product search index
    Supports: BM25 full-text, vector search, semantic search, faceting, filtering
    """
    
    # ========================================
    # FIELD DEFINITIONS
    # ========================================
    
    fields = [
        # KEY FIELD (required, exactly one)
        SimpleField(
            name="product_id",
            type=SearchFieldDataType.String,
            key=True,                    # Unique identifier
            filterable=True,
            sortable=True
        ),
        
        # SEARCHABLE TEXT FIELDS (BM25 full-text search)
        SearchableField(
            name="name",
            type=SearchFieldDataType.String,
            searchable=True,             # Enables full-text search
            filterable=True,
            sortable=True,
            facetable=False,
            analyzer_name="en.microsoft" # English language analyzer
        ),
        SearchableField(
            name="description",
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name="en.microsoft"
        ),
        SearchableField(
            name="brand",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True,              # Enable facets (brand counts)
            analyzer_name="keyword"      # Exact-match analyzer for brands
        ),
        
        # FILTERABLE/FACETABLE FIELDS (for faceted navigation)
        SimpleField(
            name="category",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,              # Category facets (Electronics: 120, Clothing: 85)
            sortable=True
        ),
        SimpleField(
            name="subcategory",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        
        # NUMERIC FIELDS (for filtering and sorting)
        SimpleField(
            name="price",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
            facetable=True               # Price range facets ($0-$50: 45, $50-$100: 30)
        ),
        SimpleField(
            name="rating",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True
        ),
        SimpleField(
            name="review_count",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True
        ),
        SimpleField(
            name="in_stock",
            type=SearchFieldDataType.Boolean,
            filterable=True,
            facetable=True
        ),
        
        # DATE FIELDS (for recency filtering/sorting)
        SimpleField(
            name="created_date",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True
        ),
        
        # VECTOR FIELD (for semantic similarity search)
        SearchField(
            name="description_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,  # OpenAI text-embedding-ada-002 dimension
            vector_search_profile_name="my-vector-profile"
        ),
        
        # METADATA FIELDS (retrievable, not searchable)
        SimpleField(
            name="image_url",
            type=SearchFieldDataType.String,
            retrievable=True,
            searchable=False
        ),
        SimpleField(
            name="sku",
            type=SearchFieldDataType.String,
            filterable=True
        )
    ]
    
    # ========================================
    # VECTOR SEARCH CONFIGURATION
    # ========================================
    
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="my-hnsw-algorithm",
                parameters={
                    "m": 4,                    # Connections per layer (higher=better recall, slower indexing)
                    "efConstruction": 400,     # Build quality (higher=better recall, slower indexing)
                    "efSearch": 500,           # Query quality (higher=better recall, slower queries)
                    "metric": "cosine"         # Distance metric (cosine, euclidean, dotProduct)
                }
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="my-vector-profile",
                algorithm_configuration_name="my-hnsw-algorithm"
            )
        ]
    )
    
    # ========================================
    # SEMANTIC SEARCH CONFIGURATION
    # ========================================
    
    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields={
            "title_field": SemanticField(field_name="name"),
            "content_fields": [SemanticField(field_name="description")],
            "keywords_fields": [SemanticField(field_name="category"), SemanticField(field_name="brand")]
        }
    )
    
    semantic_search = SemanticSearch(
        configurations=[semantic_config]
    )
    
    # ========================================
    # SCORING PROFILE (Custom relevance boosting)
    # ========================================
    
    scoring_profiles = [
        ScoringProfile(
            name="boost-popular-products",
            text_weights=TextWeights(
                weights={
                    "name": 3.0,        # Boost title matches 3x
                    "description": 1.0,  # Normal weight for description
                    "brand": 2.0        # Boost brand matches 2x
                }
            ),
            functions=[
                # Boost highly-rated products
                {
                    "type": "magnitude",
                    "field_name": "rating",
                    "boost": 2.0,
                    "interpolation": "linear",
                    "magnitude": {
                        "boostingRangeStart": 4.0,  # Start boost at 4-star ratings
                        "boostingRangeEnd": 5.0,
                        "constantBoostBeyondRange": True
                    }
                },
                # Boost products with many reviews (social proof)
                {
                    "type": "magnitude",
                    "field_name": "review_count",
                    "boost": 1.5,
                    "interpolation": "logarithmic",
                    "magnitude": {
                        "boostingRangeStart": 10,
                        "boostingRangeEnd": 1000
                    }
                },
                # Boost recent products (freshness)
                {
                    "type": "freshness",
                    "field_name": "created_date",
                    "boost": 1.2,
                    "interpolation": "linear",
                    "freshness": {
                        "boostingDuration": "P90D"  # Boost products from last 90 days
                    }
                }
            ]
        )
    ]
    
    # ========================================
    # CREATE INDEX
    # ========================================
    
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
        scoring_profiles=scoring_profiles
    )
    
    try:
        result = index_client.create_or_update_index(index)
        print(f"✅ Index '{index_name}' created successfully")
        print(f"   Fields: {len(fields)}")
        print(f"   Vector search: Enabled (1536 dimensions)")
        print(f"   Semantic search: Enabled")
        print(f"   Scoring profiles: {len(scoring_profiles)}")
        return result
    except Exception as e:
        print(f"❌ Index creation failed: {str(e)}")
        raise

# Create the index
create_production_index("products")
```

### Field Attribute Decision Guide

**"Should this field be searchable?"**
- ✅ Yes if: Users will type queries looking for content in this field (title, description, body text)
- ❌ No if: Metadata, IDs, URLs, image paths, exact-match only fields
- Cost: Searchable fields create inverted index (+20-30% storage)

**"Should this field be filterable?"**
- ✅ Yes if: Users will filter by this field (`price < 100`, `category eq 'Electronics'`)
- ✅ Yes if: Field is used in facets (facets require filterable)
- ❌ No if: Large text fields (inefficient, use searchable instead)
- Cost: Filterable fields stored in memory (+memory usage, faster queries)

**"Should this field be sortable?"**
- ✅ Yes if: Users will sort results by this field (price, date, rating, relevance)
- ❌ No if: Large collections or text fields (not useful for sorting)
- Cost: Sortable fields stored in memory (+memory usage)

**"Should this field be facetable?"**
- ✅ Yes if: You'll show counts per value (Categories: Electronics(120), Clothing(85))
- ✅ Yes if: Price ranges, date ranges (requires numeric or date fields)
- ❌ No if: High-cardinality fields (millions of unique values = slow, expensive)
- Cost: Facetable fields create aggregations (+memory, +query time for facet computation)

### Analyzer Selection

**Analyzers determine how text is processed for search**:

```python
# ========================================
# BUILT-IN ANALYZERS
# ========================================

# Language-specific analyzers (recommended for most use cases)
"en.microsoft"      # English: stemming, stop words, case-folding
"en.lucene"         # English (Lucene): alternative stemming approach
"fr.microsoft"      # French
"de.microsoft"      # German
"es.microsoft"      # Spanish
# See full list: https://learn.microsoft.com/azure/search/index-add-language-analyzers

# Special-purpose analyzers
"keyword"           # No tokenization (exact match only) - use for IDs, SKUs, brands
"whitespace"        # Split on whitespace only (preserves punctuation)
"standardasciifolding.lucene"  # Folds accents (café → cafe)
"pattern"           # Regex-based tokenization

# ========================================
# ANALYZER CHOICE EXAMPLES
# ========================================

# Product names (English, stem words)
SearchableField(
    name="name",
    type=SearchFieldDataType.String,
    analyzer_name="en.microsoft"
    # "running shoes" matches "run shoe", "runner shoes"
)

# SKU (exact match only)
SearchableField(
    name="sku",
    type=SearchFieldDataType.String,
    analyzer_name="keyword"
    # "ABC-123" matches only "ABC-123", not "ABC" or "123"
)

# Multilingual content (index-time vs query-time analyzers)
SearchableField(
    name="description",
    type=SearchFieldDataType.String,
    search_analyzer_name="en.microsoft",    # Query uses English
    index_analyzer_name="standardasciifolding.lucene"  # Index uses accent-folding
    # Allows "resume" query to match "résumé" in index
)
```

**When to use custom analyzers** (see `13-custom-analyzers.md`):
- Domain-specific tokenization (medical terms, chemical compounds, product codes)
- Synonym support (search "laptop" finds "notebook")
- Special character handling (remove or preserve @ # $ symbols)

### Index Design Best Practices

**1. Key field**:
```python
# ✅ Good: String, unique, stable, meaningful
SimpleField(name="product_id", type=SearchFieldDataType.String, key=True)

# ❌ Bad: Auto-incrementing integer (breaks when re-indexing)
# ❌ Bad: GUID (hard to debug, no business meaning)
```

**2. Minimize searchable fields**:
```python
# ✅ Good: Only fields users actually search
fields = [
    SearchableField(name="title", ...),       # Users search titles
    SearchableField(name="description", ...)  # Users search descriptions
]

# ❌ Bad: Making everything searchable (increases cost, decreases relevance)
fields = [
    SearchableField(name="id", ...),          # Never searched
    SearchableField(name="image_url", ...),   # Never searched
    SearchableField(name="internal_notes", ...)  # Shouldn't be searchable
]
```

**3. Use SimpleField for metadata**:
```python
# ✅ Good: SimpleField for non-searchable metadata
SimpleField(name="image_url", type=SearchFieldDataType.String)
SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset, filterable=True)

# ❌ Bad: SearchableField for URLs (waste of resources)
SearchableField(name="image_url", ...)  # Never useful to search URLs
```

**4. Plan for facets early**:
```python
# ✅ Good: Enable facetable on category, brand, price (low-cardinality)
SimpleField(name="category", type=SearchFieldDataType.String, facetable=True, filterable=True)

# ❌ Bad: Facetable on high-cardinality fields
SimpleField(name="description", type=SearchFieldDataType.String, facetable=True)
# Results in millions of facets (slow, not useful)
```

**5. Vector field sizing**:
```python
# ✅ Good: Match embedding model dimension exactly
SearchField(
    name="description_vector",
    vector_search_dimensions=1536  # Matches text-embedding-ada-002
)

# ❌ Bad: Dimension mismatch causes errors
SearchField(
    name="description_vector",
    vector_search_dimensions=512  # text-embedding-ada-002 produces 1536, not 512!
)
```

### Index Configuration Troubleshooting

```
Issue: "Field 'X' cannot be modified"
Cause: Cannot change field attributes after index creation
Fix: Create new index with correct schema, reindex data, update app to use new index
Prevention: Test schema thoroughly in dev before creating production index

Issue: "Query failed: Cannot filter on field 'description'"
Cause: Field not marked as filterable in schema
Fix: Recreate index with filterable=True for that field

Issue: "Vector search failed: dimension mismatch (expected 1536, got 768)"
Cause: Embedding model changed (e.g., switched from ada-002 to text-embedding-3-small)
Fix: Recreate index with vector_search_dimensions=768, re-embed and reindex all documents

Issue: "Indexing very slow (>10 minutes for 100K docs)"
Cause: Too many searchable fields, complex analyzers, vector indexing
Fix: Review schema, disable searchable on fields that don't need full-text search
     Consider batch upload optimization (see 16-dataset-preparation.md)

Issue: "Query results not relevant"
Cause: Wrong analyzer (e.g., keyword analyzer on natural language field)
Fix: Use language-specific analyzer (en.microsoft, fr.microsoft) for text fields
```

### Index Creation Checklist

Before creating production index:

- ✅ Key field defined (String, unique, stable)
- ✅ Searchable only on fields users will query (title, description, body text)
- ✅ Filterable on fields used in filters/facets (category, price, date, in_stock)
- ✅ Sortable on fields users will sort by (price, date, rating)
- ✅ Facetable on low-cardinality categorical fields (category, brand, <100 unique values)
- ✅ Analyzers match field language (en.microsoft for English, keyword for exact-match)
- ✅ Vector dimensions match embedding model (1536 for ada-002, 768 for 3-small/3-large)
- ✅ Semantic search configuration defines title/content/keyword fields correctly
- ✅ Scoring profile boosts important fields (if needed)
- ✅ Test schema in dev environment before production deployment

---

## Data Source Setup# Create index
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
---

## Data Source Setup

**Why data source setup matters**: Azure AI Search can automatically index data from Azure data sources using **indexers** (scheduled, incremental updates). Alternative: Manual push API (full control, more code). This section covers both approaches and when to use each.

### Data Ingestion Options

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **Indexers** (Pull) | Data in Azure Blob, Cosmos DB, SQL DB | Automatic, scheduled, incremental, change tracking | Requires Azure data source, less control |
| **Push API** (Push) | Custom data sources, on-prem data, complex ETL | Full control, any data source | Manual code, no automatic updates |

### Indexer-based Data Sources

**Supported data sources**:
- Azure Blob Storage (JSON, CSV, PDF, Office docs)
- Azure Cosmos DB  
- Azure SQL Database
- Azure Table Storage
- Azure Data Lake Storage Gen2

**Blob Storage indexer complete example**:

```python
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataSourceConnection,
    SearchIndexer,
    IndexingSchedule,
    FieldMapping
)
from datetime import timedelta
from azure.core.credentials import AzureKeyCredential

# Initialize
indexer_client = SearchIndexerClient(
    endpoint="https://YOUR-SERVICE.search.windows.net",
    credential=AzureKeyCredential("YOUR-ADMIN-KEY")
)

# 1. Create data source connection
data_source = SearchIndexerDataSourceConnection(
    name="blob-datasource",
    type="azureblob",
    connection_string="DefaultEndpointsProtocol=https;AccountName=YOUR-STORAGE-ACCOUNT;AccountKey=YOUR-KEY;EndpointSuffix=core.windows.net",
    container={"name": "documents", "query": "subfolder/"}  # Optional: index only subfolder
)

indexer_client.create_or_update_data_source_connection(data_source)

# 2. Create indexer (maps data source to index)
indexer = SearchIndexer(
    name="blob-indexer",
    data_source_name="blob-datasource",
    target_index_name="products",  # Must exist already
    schedule=IndexingSchedule(interval=timedelta(hours=1)),  # Run hourly
    field_mappings=[
        FieldMapping(source_field_name="metadata_storage_name", target_field_name="file_name"),
        FieldMapping(source_field_name="metadata_storage_path", target_field_name="file_url")
    ]
)

indexer_client.create_or_update_indexer(indexer)
print("✅ Indexer created, will run every hour")
```

### Push API (Manual Upload)

**When to use**: Custom data sources, complex transformations, immediate updates.

```python
from azure.search.documents import SearchClient

search_client = SearchClient(
    endpoint="https://YOUR-SERVICE.search.windows.net",
    index_name="products",
    credential=AzureKeyCredential("YOUR-ADMIN-KEY")
)

# Upload documents
documents = [
    {"product_id": "1", "name": "Laptop", "price": 999.99},
    {"product_id": "2", "name": "Mouse", "price": 29.99}
]

result = search_client.upload_documents(documents)
print(f"Uploaded {len(result)} documents")
```

---

## Security Configuration

**Key concepts**:
- **Admin keys**: Full access (create, delete indexes). Keep secret.
- **Query keys**: Read-only search access. Safe for client apps.
- **Managed Identity**: No keys needed (Azure AD authentication).

**Regenerate admin keys** (if compromised):
```bash
az search admin-key renew \
  --service-name YOUR-SERVICE \
  --resource-group YOUR-RG \
  --key-type primary
```

---

## Network Configuration

**Options**:
1. **Public access**: Default, accessible from anywhere
2. **IP firewall**: Allow only specific IPs
3. **Private endpoint**: Access only from VNet (most secure)

**Enable private endpoint** (recommended for production):
```bash
az network private-endpoint create \
  --name search-private-endpoint \
  --resource-group YOUR-RG \
  --vnet-name YOUR-VNET \
  --subnet YOUR-SUBNET \
  --private-connection-resource-id $(az search service show --name YOUR-SERVICE --resource-group YOUR-RG --query id -o tsv) \
  --group-id searchService \
  --connection-name search-connection
```

---

---

## Cost Management

**Monitor costs**:
```bash
# View current month costs
az consumption usage list \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --query "[?contains(instanceId, 'searchServices')].{Name:instanceName, Cost:pretaxCost}"
```

**Set budget alerts**:
- Azure Portal → Cost Management → Budgets
- Create alert at 80%, 100%, 120% of monthly budget
- Get email/SMS when threshold exceeded

**Cost optimization**:
- Start with S1, scale up only if needed
- Use query keys (not admin keys) in applications
- Disable semantic search if not using ($500/month savings)
- Consider reserved capacity (1-year commitment = 20% discount)

---

## Summary

This guide covered the complete Azure AI Search service setup process:

1. **Prerequisites**: Azure subscription, dev environment, cost estimation
2. **Service Creation**: Portal (easiest), CLI (automation), ARM (IaC), Python SDK (programmatic)
3. **Tier Selection**: Decision framework based on doc count, QPS, storage, HA requirements
4. **Index Configuration**: Field types, attributes, analyzers, vector search, semantic search
5. **Data Sources**: Indexers (automatic) vs Push API (manual)
6. **Security**: Admin keys, query keys, managed identity
7. **Networking**: Public access, IP firewall, private endpoints
8. **Cost Management**: Monitoring, budgets, optimization

**Next steps**:
- `05-azure-openai-integration.md`: Generate embeddings for vector search
- `09-vector-search.md`: Implement vector similarity search
- `10-hybrid-search.md`: Combine BM25 + vector for best results
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