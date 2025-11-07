# Azure OpenAI Integration

**Purpose**: Complete guide to integrating Azure OpenAI Service for generating vector embeddings that power semantic search, enabling Azure AI Search to understand query intent beyond exact keyword matches.

**Target Audience**: Python developers implementing vector search, data engineers preparing search data, search architects designing hybrid search systems.

**Reading Time**: 45-50 minutes (comprehensive), 20 minutes (quick implementation)

**Related Documents**:
- Prerequisites: `04-azure-ai-search-setup.md` (Azure AI Search service setup)
- Implementation: `09-vector-search.md` (Vector similarity search)
- Advanced: `10-hybrid-search.md` (Combining BM25 + vector)

---

## Why Azure OpenAI Integration Matters

**The problem with keyword-only search**: Traditional BM25 full-text search matches exact words and stems. User searches "best budget laptop" but your documents say "affordable computer for students" → **zero results** despite perfect semantic match.

**The vector search solution**: Azure OpenAI embeddings convert text to 1,536-dimensional vectors (ada-002 model) that capture semantic meaning. Queries and documents with similar meaning cluster together in vector space, even with zero word overlap.

**Real-world impact example**:

```
Scenario: Healthcare knowledge base (5M medical articles)

Before (BM25 only):
- Query: "heart attack symptoms"
- Matches: Articles containing exact words "heart attack symptoms"
- Miss: Articles about "myocardial infarction warning signs" (same condition, medical terminology)
- Recall: 42%
- User satisfaction: 3.1/5

After (Hybrid BM25 + Vector):
- Query: "heart attack symptoms" → embedded to vector
- Matches: Articles semantically similar (medical terms, lay terms, symptom descriptions)
- Finds: "myocardial infarction," "acute coronary syndrome," "chest pain indicators"
- Recall: 89% (+47 percentage points)
- User satisfaction: 4.6/5 (+1.5 points)
- Cost: +$120/month (embedding generation), net revenue +$45K/year (reduced support calls)
```

**Three critical integration decisions**:
1. **Embedding model choice**: ada-002 (1536-dim, $0.0001/1K tokens) vs. text-embedding-3-small (1536-dim, $0.00002/1K tokens, 5x cheaper) vs. text-embedding-3-large (3072-dim, $0.00013/1K tokens, highest quality)
2. **Chunking strategy**: Whole documents (simple, loses context for long docs) vs. smart chunking (preserves context, more embeddings to generate)
3. **Batch processing**: Sequential (simple, slow, expensive) vs. parallel batching (fast, complex error handling, cost-effective)

---

## 📋 Table of Contents
- [Azure OpenAI Service Setup](#azure-openai-service-setup)
- [Embedding Model Selection](#embedding-model-selection)
- [Document Chunking Strategy](#document-chunking-strategy)
- [Embedding Generation](#embedding-generation)
- [Batch Processing](#batch-processing)
- [Cost Optimization](#cost-optimization)
- [Performance Tuning](#performance-tuning)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## Azure OpenAI Service Setup

**Prerequisites before starting**:
- ✅ Azure subscription with Azure OpenAI access approved (apply at https://aka.ms/oai/access if pending)
- ✅ Azure AI Search service created (see `04-azure-ai-search-setup.md`)
- ✅ Azure CLI installed and authenticated (`az login`)
- ✅ Python 3.8+ with `openai` SDK installed (`pip install openai`)

### Step 1: Create Azure OpenAI Resource

**Region selection considerations**:
- **Quota availability**: Not all regions have capacity for all models
  - Check current availability: https://learn.microsoft.com/azure/ai-services/openai/concepts/models#model-summary-table-and-region-availability
  - Recommended regions (as of 2024): East US, Sweden Central, Switzerland North (high quota, latest models)
  - Avoid: Regions with frequent capacity constraints (check Azure status page)
- **Latency optimization**: Choose region closest to Azure AI Search service
  - Same region as Search service: <10ms latency (ideal for real-time embedding)
  - Cross-region (same continent): 20-50ms latency (acceptable for batch processing)
  - Cross-continent: 100-200ms latency (only for batch, not real-time)
- **Data residency**: GDPR, compliance requirements may mandate specific regions

**Create Azure OpenAI resource** (CLI method, recommended for automation):

```bash
# ============================================
# Azure OpenAI Resource Creation Script
# ============================================

# Configuration
RESOURCE_GROUP="rg-search-evaluation"
LOCATION="eastus"  # Or swedencentral, switzerlandnorth for better quota
OPENAI_NAME="openai-search-embeddings-prod"
SKU="S0"  # Standard tier (only option)

# Create resource group if not exists
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Create Azure OpenAI resource
echo "Creating Azure OpenAI resource: $OPENAI_NAME"
az cognitiveservices account create \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --kind OpenAI \
  --sku $SKU \
  --custom-domain $OPENAI_NAME \
  --tags Environment=Production Project=SearchEvaluation \
  --yes

# Get endpoint and key
OPENAI_ENDPOINT=$(az cognitiveservices account show \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.endpoint \
  --output tsv)

OPENAI_KEY=$(az cognitiveservices account keys list \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --query key1 \
  --output tsv)

# Save to .env file
cat >> .env << EOF
AZURE_OPENAI_ENDPOINT="$OPENAI_ENDPOINT"
AZURE_OPENAI_API_KEY="$OPENAI_KEY"
EOF

echo "✅ Azure OpenAI resource created successfully"
echo "Endpoint: $OPENAI_ENDPOINT"
echo "Key saved to .env file"
```

**Portal creation** (for first-time users preferring UI):
1. Azure Portal → Create a resource → Search "Azure OpenAI"
2. Basics tab:
   - Subscription: Choose your subscription
   - Resource group: `rg-search-evaluation` (same as Search service)
   - Region: `East US` (or region with quota availability)
   - Name: `openai-search-embeddings-prod` (globally unique)
   - Pricing tier: **Standard S0** (only option, pay-per-use)
3. Networking tab: Public endpoint (default) or private endpoint (for production security)
4. Tags tab: Add `Environment=Production`, `Project=SearchEvaluation` for cost tracking
5. Review + create → Wait 2-3 minutes for deployment
6. Go to resource → Keys and Endpoint → Copy Key 1 and Endpoint → Save to .env file

### Step 2: Deploy Embedding Model

**Why deployment is required**: Unlike other APIs, Azure OpenAI requires explicit model deployment before use. This allocates compute capacity (Tokens Per Minute quota) for your use case.

**Deployment creates**:
- **Model instance**: Dedicated capacity for your workload
- **TPM quota**: Tokens Per Minute throughput (120K TPM = ~120 API calls/minute for 1K token inputs)
- **Endpoint**: Deployment-specific name you'll reference in code

**Embedding model comparison** (choose based on requirements):

| Model | Dimensions | Cost per 1K tokens | Speed | Quality | Best For |
|-------|------------|-------------------|-------|---------|----------|
| **text-embedding-ada-002** | 1,536 | $0.0001 | Fast | Good | Legacy (still widely used) |
| **text-embedding-3-small** | 1,536 | $0.00002 | Fastest | Good | **Cost-sensitive production** (5x cheaper than ada-002) |
| **text-embedding-3-large** | 3,072 | $0.00013 | Slower | **Best** | Quality-critical applications, multilingual |

**Recommendation for most use cases**: **text-embedding-3-small**
- 5x cheaper than ada-002 ($0.10 per 1M tokens vs. $0.50)
- Comparable quality for English text
- Same 1,536 dimensions (compatible with existing indexes)
- Faster inference (lower latency)

**When to use text-embedding-3-large**:
- Multilingual search (superior non-English quality)
- Domain-specific content requiring nuanced understanding
- Budget allows for quality investment (30% more expensive than ada-002, but significantly better)

**Deploy embedding model via CLI**:

```bash
# ============================================
# Deploy Embedding Model - Complete Script
# ============================================

OPENAI_NAME="openai-search-embeddings-prod"
RESOURCE_GROUP="rg-search-evaluation"
DEPLOYMENT_NAME="text-embedding-3-small"  # Or ada-002, 3-large
MODEL_NAME="text-embedding-3-small"
MODEL_VERSION="1"  # Check latest version in Azure Portal
TPM_CAPACITY=120  # Tokens Per Minute (K) - Start with 120K, scale as needed

# Deploy model
echo "Deploying embedding model: $MODEL_NAME"
az cognitiveservices account deployment create \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --deployment-name $DEPLOYMENT_NAME \
  --model-name $MODEL_NAME \
  --model-version $MODEL_VERSION \
  --model-format OpenAI \
  --sku-capacity $TPM_CAPACITY \
  --sku-name "Standard"

# Verify deployment
echo "Verifying deployment..."
az cognitiveservices account deployment show \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --deployment-name $DEPLOYMENT_NAME \
  --output table

echo "✅ Model deployed successfully"
echo "Deployment name: $DEPLOYMENT_NAME"
echo "TPM Capacity: ${TPM_CAPACITY}K tokens/minute"
```

**Deploy via Azure Portal**:
1. Go to Azure OpenAI resource → Deployments (under Resource Management)
2. Click **"+ Create new deployment"**
3. Configure:
   - **Model**: `text-embedding-3-small` (recommended) or `text-embedding-ada-002` or `text-embedding-3-large`
   - **Deployment name**: `text-embedding-3-small` (use same name as model for simplicity)
   - **Model version**: `1` (or latest available)
   - **Deployment type**: Standard
   - **Tokens per Minute Rate Limit (thousands)**: `120` (start conservative, scale up)
     * 120K TPM ≈ 120 concurrent embedding calls/minute (assuming 1K tokens per document)
     * 240K TPM ≈ 240 concurrent calls (double capacity)
     * Monitor usage in Metrics, increase if hitting rate limits
4. Click **Create**
5. Wait ~30 seconds for deployment to complete
6. Test deployment with sample code (next section)

**TPM capacity planning**:

```python
def estimate_tpm_needed(requirements):
    """
    Estimate Tokens Per Minute (TPM) capacity needed for embedding workload
    """
    # Extract requirements
    docs_to_embed = requirements.get('document_count', 100000)
    avg_doc_tokens = requirements.get('avg_document_tokens', 500)
    embedding_window_hours = requirements.get('embedding_window_hours', 24)
    
    # Calculate total tokens
    total_tokens = docs_to_embed * avg_doc_tokens
    
    # Calculate tokens per minute (spread across embedding window)
    minutes_available = embedding_window_hours * 60
    tokens_per_minute = total_tokens / minutes_available
    
    # Add 50% buffer for retries, overhead, concurrent requests
    tokens_per_minute_buffered = tokens_per_minute * 1.5
    
    # Convert to thousands (Azure quota is in K)
    tpm_needed_k = int(tokens_per_minute_buffered / 1000) + 1
    
    # Round to common quota increments (120K, 240K, 360K, etc.)
    quota_increments = [120, 240, 360, 480, 600, 720, 1000, 1500, 2000]
    recommended_quota = next((q for q in quota_increments if q >= tpm_needed_k), tpm_needed_k)
    
    # Calculate embedding time
    embedding_time_hours = (total_tokens / (recommended_quota * 1000)) / 60
    
    # Cost estimation (text-embedding-3-small pricing: $0.00002 per 1K tokens)
    cost_per_1k_tokens = 0.00002
    total_cost = (total_tokens / 1000) * cost_per_1k_tokens
    
    return {
        'total_tokens': f"{total_tokens:,}",
        'tpm_needed': f"{tpm_needed_k}K",
        'recommended_quota': f"{recommended_quota}K",
        'estimated_time_hours': f"{embedding_time_hours:.1f}",
        'estimated_cost': f"${total_cost:.2f}",
        'model': 'text-embedding-3-small'
    }

# Example: 100K documents, 500 tokens each, embed in 24 hours
initial_index = {
    'document_count': 100000,
    'avg_document_tokens': 500,
    'embedding_window_hours': 24
}

estimate = estimate_tpm_needed(initial_index)
print("Embedding Capacity Estimate:")
print(f"  Total tokens: {estimate['total_tokens']}")
print(f"  TPM needed: {estimate['tpm_needed']}")
print(f"  Recommended quota: {estimate['recommended_quota']}")
print(f"  Embedding time: {estimate['estimated_time_hours']} hours")
print(f"  Estimated cost: {estimate['estimated_cost']}")
```

**Output example**:
```
Embedding Capacity Estimate:
  Total tokens: 50,000,000
  TPM needed: 52K
  Recommended quota: 120K
  Embedding time: 6.9 hours
  Estimated cost: $1.00
```

### Step 3: Test Connection and Embedding

**Complete test script** (validates deployment, generates sample embedding):

```python
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",  # Check latest API version
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def test_openai_connection():
    """
    Test Azure OpenAI connection and embedding generation
    Returns: True if successful, False otherwise
    """
    deployment_name = "text-embedding-3-small"  # Must match your deployment name
    
    try:
        print("Testing Azure OpenAI connection...")
        
        # Generate embedding for test text
        response = client.embeddings.create(
            input="Azure AI Search with vector embeddings enables semantic search",
            model=deployment_name
        )
        
        # Extract embedding
        embedding = response.data[0].embedding
        
        # Validate
        print("✅ Azure OpenAI connection successful")
        print(f"   Deployment: {deployment_name}")
        print(f"   Embedding dimensions: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Token usage: {response.usage.total_tokens} tokens")
        
        # Verify dimensions match expected model
        expected_dims = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072
        }
        
        expected = expected_dims.get(deployment_name, 1536)
        if len(embedding) != expected:
            print(f"⚠️  Warning: Expected {expected} dimensions, got {len(embedding)}")
            print("   Check deployment model matches deployment_name variable")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify AZURE_OPENAI_ENDPOINT in .env (format: https://YOUR-RESOURCE.openai.azure.com/)")
        print("2. Verify AZURE_OPENAI_API_KEY in .env")
        print("3. Verify deployment name matches exactly (case-sensitive)")
        print(f"4. Check deployment exists: az cognitiveservices account deployment list --name YOUR-RESOURCE --resource-group YOUR-RG")
        return False

if __name__ == "__main__":
    success = test_openai_connection()
    exit(0 if success else 1)
```

**Expected output**:
```
Testing Azure OpenAI connection...
✅ Azure OpenAI connection successful
   Deployment: text-embedding-3-small
   Embedding dimensions: 1536
   First 5 values: [0.0123, -0.0456, 0.0789, -0.0234, 0.0567]
   Token usage: 12 tokens
```

**Common connection errors**:
```
Error: "DeploymentNotFound: The API deployment for this resource does not exist"
Fix: Check deployment name spelling (case-sensitive)
     Run: az cognitiveservices account deployment list --name YOUR-RESOURCE --resource-group YOUR-RG

Error: "InvalidRequestError: The API deployment is not yet available"
Fix: Wait 30-60 seconds after creating deployment (deployment provisioning delay)

Error: "RateLimitError: Requests to the Embeddings API have exceeded call rate limit"
Fix: Reduce concurrent requests, or increase TPM quota in deployment settings

Error: "AuthenticationError: Incorrect API key provided"
Fix: Regenerate key in Azure Portal → Keys and Endpoint
     Update AZURE_OPENAI_API_KEY in .env file
```

---

## Embedding Model Selection

**Overview**: Azure OpenAI offers three embedding models with different quality/cost/dimension trade-offs. Your choice impacts search quality, latency, monthly costs, and index storage requirements.

**Why model selection matters**:
```
Scenario: Legal document search (2M paragraphs, 100K searches/month)

Model: text-embedding-ada-002
- Initial embedding cost: 2M paragraphs × 200 tokens avg × $0.0001/1K = $40
- Monthly search cost: 100K queries × 50 tokens avg × $0.0001/1K = $0.50
- Storage: 2M × 1,536 dims × 4 bytes = 12.3 GB vector storage
- Search latency: ~45ms (1,536-dim HNSW)
- Recall@10: 82%
- Total first month: $40.50

Model: text-embedding-3-small (5x cheaper)
- Initial embedding cost: 2M × 200 × $0.00002/1K = $8
- Monthly search cost: 100K × 50 × $0.00002/1K = $0.10
- Storage: 2M × 1,536 dims × 4 bytes = 12.3 GB (same)
- Search latency: ~42ms (slightly faster inference)
- Recall@10: 83% (+1 point improvement)
- Total first month: $8.10 (80% cost reduction!)

Model: text-embedding-3-large (highest quality)
- Initial embedding cost: 2M × 200 × $0.00013/1K = $52
- Monthly search cost: 100K × 50 × $0.00013/1K = $0.65
- Storage: 2M × 3,072 dims × 4 bytes = 24.6 GB (2x storage!)
- Search latency: ~75ms (larger vectors = slower search)
- Recall@10: 89% (+6 points over 3-small)
- Total first month: $52.65

Decision: text-embedding-3-small for production
- 80% cost savings vs. ada-002
- Better quality than ada-002
- Same dimensions = no index changes
- Acceptable latency for legal search
```

**Decision framework for choosing embedding model**:

### Model Comparison Table

| Model | Dimensions | Cost per 1M tokens | Release | Best For | When to Avoid |
|-------|------------|-------------------|---------|----------|---------------|
| **text-embedding-ada-002** | 1,536 | $0.10 | 2022 | Legacy compatibility | New projects (3-small is better + cheaper) |
| **text-embedding-3-small** | 1,536 | **$0.02** | 2024 | **Most production workloads** | Multilingual with non-English priority |
| **text-embedding-3-large** | 3,072 | $0.13 | 2024 | Quality-critical, multilingual | Cost-sensitive, high-volume |

### Decision Tree

```
Start: Do you need multilingual support (non-English priority)?
│
├─ YES → Do you have budget for 6.5x cost vs. 3-small?
│   │
│   ├─ YES → text-embedding-3-large
│   │        (Superior multilingual quality)
│   │
│   └─ NO → text-embedding-3-small
│           (Good multilingual, affordable)
│
└─ NO (English-only or English-dominant)
    │
    └─ Do you have existing ada-002 indexes?
        │
        ├─ YES → Are you satisfied with quality?
        │   │
        │   ├─ YES → Stay with ada-002 (no migration cost)
        │   │
        │   └─ NO → Migrate to text-embedding-3-small
        │           (Better quality, 5x cheaper, same dims)
        │
        └─ NO (new project) → text-embedding-3-small
                              (Best quality/cost ratio)
```

### Detailed Model Characteristics

#### text-embedding-ada-002 (Legacy, still widely used)

**Strengths**:
- ✅ Well-tested in production (2+ years)
- ✅ Extensive documentation and community examples
- ✅ Existing tooling compatibility
- ✅ Good English language performance

**Weaknesses**:
- ❌ 5x more expensive than 3-small ($0.10 vs. $0.02 per 1M tokens)
- ❌ Lower quality than 3-small/3-large
- ❌ Slower inference than 3-small

**When to use**:
- You have existing ada-002 indexes and quality is acceptable
- You're using third-party tools that specifically require ada-002
- You need stability over cost optimization (rare)

**Cost example** (100K documents, 500 tokens average):
```
100,000 docs × 500 tokens = 50M tokens
50M tokens × $0.0001/1K tokens = $5.00 embedding cost
```

#### text-embedding-3-small (⭐ **Recommended for most use cases**)

**Strengths**:
- ✅ **5x cheaper than ada-002** ($0.02 vs. $0.10 per 1M tokens)
- ✅ **Better quality than ada-002** (higher recall, precision)
- ✅ **Faster inference** (lower latency than ada-002 and 3-large)
- ✅ Same 1,536 dimensions (drop-in replacement for ada-002 indexes)
- ✅ Good multilingual support (60+ languages)
- ✅ Latest model architecture improvements

**Weaknesses**:
- ⚠️ Half the dimensions of 3-large (3,072 vs 1,536)
- ⚠️ Slightly lower quality than 3-large on complex/multilingual queries

**When to use**:
- ✅ **New projects** (default choice)
- ✅ **Cost-sensitive production** (startups, high-volume)
- ✅ **English-dominant content**
- ✅ **Migrating from ada-002** (better + cheaper)
- ✅ **Real-time search** (low latency critical)

**Cost example** (100K documents, 500 tokens average):
```
100,000 docs × 500 tokens = 50M tokens
50M tokens × $0.00002/1K tokens = $1.00 embedding cost
Savings vs. ada-002: $4.00 (80% reduction!)
```

**Performance comparison** (MTEB benchmark):
- English retrieval: **64.6** (vs. ada-002: 61.0, +3.6 points)
- Multilingual retrieval: **57.5** (vs. ada-002: 54.9, +2.6 points)
- Speed: **1.2x faster** than ada-002

#### text-embedding-3-large (Highest quality)

**Strengths**:
- ✅ **Best overall quality** across all languages
- ✅ **Superior multilingual performance** (especially Asian languages)
- ✅ **Better semantic understanding** of domain-specific terminology
- ✅ **Higher dimensional space** (3,072 dims) captures more nuance
- ✅ Best for complex queries with subtle semantic differences

**Weaknesses**:
- ❌ **6.5x more expensive than 3-small** ($0.13 vs. $0.02 per 1M tokens)
- ❌ **2x larger vectors** = 2x index storage costs
- ❌ **Slower search latency** (3,072-dim HNSW slower than 1,536-dim)
- ❌ **Higher compute costs** for embedding generation

**When to use**:
- ✅ **Quality-critical applications** (medical, legal where accuracy matters most)
- ✅ **Multilingual search** with non-English majority (Chinese, Japanese, Korean)
- ✅ **Domain-specific content** requiring deep semantic understanding
- ✅ **Low query volume** (cost impact manageable)
- ✅ **High-value use cases** (cost justified by business impact)

**When to avoid**:
- ❌ High-volume production (cost prohibitive)
- ❌ Real-time latency requirements (larger vectors = slower)
- ❌ Limited storage budget (2x storage vs. 3-small)
- ❌ English-only content (3-small sufficient)

**Cost example** (100K documents, 500 tokens average):
```
100,000 docs × 500 tokens = 50M tokens
50M tokens × $0.00013/1K tokens = $6.50 embedding cost
Storage: 100K × 3,072 dims × 4 bytes = 1.2 GB (vs. 600 MB for 3-small)
```

**Performance comparison** (MTEB benchmark):
- English retrieval: **69.2** (vs. 3-small: 64.6, +4.6 points)
- Multilingual retrieval: **63.8** (vs. 3-small: 57.5, +6.3 points)
- Chinese retrieval: **72.1** (vs. 3-small: 65.3, +6.8 points)

### Real-World Use Case Recommendations

#### E-commerce Product Search (100K products)
**Recommendation**: text-embedding-3-small
- Reason: High query volume (1M/month), cost-sensitive, English product names
- Cost: $1.00 initial + $2.00/month queries = **$3.00/month**
- Alternative (3-large): $6.50 initial + $13.00/month = $19.50/month (6.5x more!)

#### Medical Research Database (10M articles, multilingual)
**Recommendation**: text-embedding-3-large
- Reason: Quality critical for medical accuracy, multilingual sources, low query volume
- Cost: $650 initial + $13/month queries = **$663 first month**
- Justification: Avoiding misdiagnosis from poor search > cost savings

#### Customer Support Knowledge Base (50K articles)
**Recommendation**: text-embedding-3-small
- Reason: Moderate volume, English-dominant, cost-effective
- Cost: $0.50 initial + $1.00/month queries = **$1.50/month**

#### Legal Document Search (2M paragraphs, English)
**Recommendation**: text-embedding-3-small
- Reason: Large corpus, English legal terminology, high query volume
- Cost: $8 initial + $2/month queries = **$10/month**
- Alternative (ada-002): $40 initial + $10/month = $50/month (5x more)

#### Global E-learning Platform (multilingual courses)
**Recommendation**: text-embedding-3-large
- Reason: Critical for non-English learners, semantic understanding of educational content
- Cost: Higher initial investment justified by improved learning outcomes

### Migration Considerations (ada-002 → 3-small)

**Should you migrate?** Decision matrix:

| Current State | Recommendation | Rationale |
|---------------|----------------|-----------|
| Happy with ada-002 quality | **Stay** | No migration cost, quality acceptable |
| Quality issues with ada-002 | **Migrate to 3-small** | Better quality + 80% cost reduction |
| Non-English majority content | **Migrate to 3-large** | Significant multilingual improvement |
| High embedding costs | **Migrate to 3-small** | 80% cost savings, better quality |
| Real-time latency issues | **Migrate to 3-small** | Faster inference + smaller vectors |

**Migration process**:
1. **Test phase**: Create new index with 3-small, compare quality on sample queries
2. **Parallel run**: Run both indexes for 1-2 weeks, measure metrics
3. **Switch**: Route traffic to new index
4. **Cleanup**: Delete old ada-002 index (recover storage costs)

**Migration cost** (2M documents, 200 tokens average):
```
Re-embedding cost: 2M × 200 tokens × $0.00002/1K = $8
Original ada-002 cost: 2M × 200 tokens × $0.0001/1K = $40
Monthly savings: (100K queries × 50 tokens) × ($0.0001 - $0.00002)/1K = $0.40/month
Break-even: Never (migration cheaper + saves money ongoing!)
```

### Model Selection API Code

```python
from enum import Enum

class EmbeddingModel(Enum):
    """Azure OpenAI embedding models."""
    ADA_002 = "text-embedding-ada-002"
    SMALL_3 = "text-embedding-3-small"
    LARGE_3 = "text-embedding-3-large"
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions for this model."""
        dims = {
            self.ADA_002: 1536,
            self.SMALL_3: 1536,
            self.LARGE_3: 3072
        }
        return dims[self]
    
    @property
    def cost_per_1k_tokens(self) -> float:
        """Get cost per 1,000 tokens (USD)."""
        costs = {
            self.ADA_002: 0.0001,
            self.SMALL_3: 0.00002,
            self.LARGE_3: 0.00013
        }
        return costs[self]
    
    @property
    def release_year(self) -> int:
        """Get model release year."""
        years = {
            self.ADA_002: 2022,
            self.SMALL_3: 2024,
            self.LARGE_3: 2024
        }
        return years[self]

class ModelSelector:
    """Help select appropriate embedding model based on requirements."""
    
    @staticmethod
    def recommend_model(
        document_count: int,
        avg_tokens_per_doc: int,
        monthly_queries: int,
        avg_tokens_per_query: int,
        primary_language: str = "en",
        is_multilingual: bool = False,
        quality_critical: bool = False,
        budget_sensitive: bool = True
    ) -> dict:
        """
        Recommend embedding model based on requirements.
        
        Args:
            document_count: Number of documents to embed
            avg_tokens_per_doc: Average tokens per document
            monthly_queries: Expected monthly query volume
            avg_tokens_per_query: Average tokens per query
            primary_language: Primary language code (en, zh, ja, etc.)
            is_multilingual: Whether content spans multiple languages
            quality_critical: Whether quality is more important than cost
            budget_sensitive: Whether cost is a major factor
            
        Returns:
            Dictionary with recommendation and analysis
        """
        # Calculate costs for each model
        models = [EmbeddingModel.ADA_002, EmbeddingModel.SMALL_3, EmbeddingModel.LARGE_3]
        
        total_doc_tokens = document_count * avg_tokens_per_doc
        total_query_tokens = monthly_queries * avg_tokens_per_query
        
        results = []
        for model in models:
            # Embedding cost (one-time)
            embedding_cost = (total_doc_tokens / 1000) * model.cost_per_1k_tokens
            
            # Query cost (monthly)
            query_cost = (total_query_tokens / 1000) * model.cost_per_1k_tokens
            
            # Storage cost estimate (vector storage)
            storage_gb = (document_count * model.dimensions * 4) / (1024**3)
            storage_cost_monthly = storage_gb * 0.10  # ~$0.10/GB/month rough estimate
            
            # Total first month
            first_month_total = embedding_cost + query_cost + storage_cost_monthly
            
            # Ongoing monthly
            monthly_cost = query_cost + storage_cost_monthly
            
            results.append({
                'model': model.value,
                'dimensions': model.dimensions,
                'embedding_cost': embedding_cost,
                'monthly_query_cost': query_cost,
                'monthly_storage_cost': storage_cost_monthly,
                'first_month_total': first_month_total,
                'ongoing_monthly': monthly_cost,
                'storage_gb': storage_gb
            })
        
        # Decision logic
        if quality_critical and is_multilingual and primary_language != 'en':
            recommendation = EmbeddingModel.LARGE_3
            reason = "Quality-critical multilingual application requires highest quality model"
        
        elif budget_sensitive and not quality_critical:
            recommendation = EmbeddingModel.SMALL_3
            reason = "Budget-sensitive application, 3-small provides best cost/quality ratio"
        
        elif is_multilingual and primary_language in ['zh', 'ja', 'ko']:
            recommendation = EmbeddingModel.LARGE_3
            reason = "Asian language content benefits significantly from 3-large quality"
        
        elif document_count > 1000000:  # > 1M docs
            recommendation = EmbeddingModel.SMALL_3
            reason = "Large corpus requires cost-effective model (3-small)"
        
        else:
            recommendation = EmbeddingModel.SMALL_3
            reason = "Default recommendation: best quality/cost balance for most use cases"
        
        # Find recommended model results
        recommended_result = next(r for r in results if r['model'] == recommendation.value)
        
        return {
            'recommended_model': recommendation.value,
            'reason': reason,
            'cost_analysis': results,
            'recommended_cost': recommended_result,
            'savings_vs_ada002': results[0]['first_month_total'] - recommended_result['first_month_total']
        }

# Usage example
selector = ModelSelector()

recommendation = selector.recommend_model(
    document_count=100000,
    avg_tokens_per_doc=500,
    monthly_queries=50000,
    avg_tokens_per_query=50,
    primary_language="en",
    is_multilingual=False,
    quality_critical=False,
    budget_sensitive=True
)

print(f"Recommended model: {recommendation['recommended_model']}")
print(f"Reason: {recommendation['reason']}")
print(f"First month cost: ${recommendation['recommended_cost']['first_month_total']:.2f}")
print(f"Ongoing monthly cost: ${recommendation['recommended_cost']['ongoing_monthly']:.2f}")
print(f"Savings vs ada-002: ${recommendation['savings_vs_ada002']:.2f}/month")
```

**Output example**:
```
Recommended model: text-embedding-3-small
Reason: Budget-sensitive application, 3-small provides best cost/quality ratio
First month cost: $2.85
Ongoing monthly cost: $1.85
Savings vs ada-002: $7.15/month
```

### Key Takeaways

1. **For new projects**: Start with text-embedding-3-small (best cost/quality)
2. **For multilingual (non-English priority)**: Consider text-embedding-3-large
3. **For high-volume**: text-embedding-3-small is 5x cheaper than ada-002
4. **For quality-critical**: text-embedding-3-large worth the premium
5. **For ada-002 users**: Evaluate migration to 3-small (better + cheaper)

---

## Document Chunking Strategy

**Overview**: Raw documents are often too large for single embeddings (ada-002 limit: 8,191 tokens ≈ 6,000 words). Chunking splits documents into searchable units while preserving semantic coherence.

**Why chunking strategy matters**:

```
Scenario: Medical research papers (avg 10,000 words each)

Strategy 1: Whole Document Embedding (NO chunking)
- Approach: Embed entire 10,000-word paper as single vector
- Problem: Exceeds 8,191 token limit → truncation → lose 40% of content!
- Search impact: User searches "treatment side effects" but conclusion section (truncated) contains answer
- Result: ❌ Missed result, poor recall

Strategy 2: Fixed-Size Chunking (500 words, no overlap)
- Approach: Split every 500 words regardless of structure
- Problem: Splits mid-sentence, mid-paragraph, mid-table
- Example break: "Patients showed significant | improvement in symptoms..."
- Search impact: Lost context across chunk boundary
- Result: ⚠️ Works but degrades quality (70% recall)

Strategy 3: Structure-Aware Chunking (RECOMMENDED)
- Approach: Respect document structure (sections, paragraphs, tables)
- Rules: Never split tables, keep headers with content, overlap 50 tokens
- Example: Keep entire "Side Effects" section together (480 words)
- Search impact: Complete semantic units, context preserved
- Result: ✅ High recall (89%), coherent answers

Cost comparison (1,000 papers, 10K words each):
- Strategy 1: 1,000 embeddings (but 40% content lost!)
- Strategy 2: 20,000 chunks × $0.00002/1K tokens = $4.00
- Strategy 3: 18,500 chunks × $0.00002/1K tokens = $3.70 (fewer chunks due to smart boundaries)
```

**Three critical chunking decisions**:
1. **Chunk size**: 256 tokens (precise, more chunks) vs. 512 tokens (balanced) vs. 1024 tokens (more context, fewer chunks)
2. **Overlap**: 0% (no overlap, risk losing boundary context) vs. 10-20% (recommended, preserves context) vs. 50%+ (wasteful, duplicate content)
3. **Structure preservation**: Character-based (simplest, worst quality) vs. Sentence-based (better) vs. Structure-aware (best, most complex)

**When to use each chunking strategy**:

| Strategy | Best For | Avoid When | Complexity |
|----------|----------|------------|------------|
| **Whole Document** | Short docs (<1K tokens), metadata-only search | Long docs, need precise retrieval | Low |
| **Fixed Character Count** | Plain text, no structure | PDFs with tables/headers | Low |
| **Sentence Boundary** | Blog posts, articles | Technical docs with tables | Medium |
| **Structure-Aware** | **PDFs, reports, research papers** | Simple plain text | High |
| **Semantic Chunking** | Narrative documents, books | Structured documents | Very High |

### Chunk Size Selection Guide

**Trade-offs to consider**:

| Chunk Size | Pros | Cons | Best For |
|------------|------|------|----------|
| **Small (256 tokens)** | Precise retrieval, lower latency, less noise in results | More chunks = higher cost, less context | FAQ, product specs, definitions |
| **Medium (512 tokens)** | ⭐ Balanced context/precision | Medium cost/latency | **Most use cases** (articles, docs) |
| **Large (1024 tokens)** | Maximum context, fewer chunks | Slower search, more noise, risk missing precise matches | Long-form content, reports |

**Cost & latency impact** (100K documents example):

```python
# Chunk size comparison
chunk_scenarios = {
    "small_256": {
        "avg_tokens_per_chunk": 256,
        "chunks_per_doc": 8,  # 2K token doc / 256
        "total_chunks": 800000,
        "embedding_cost": 800000 * 256 / 1000 * 0.00002,  # $4.10
        "storage_gb": 800000 * 1536 * 4 / (1024**3),  # 4.8 GB
        "search_latency_ms": 35
    },
    "medium_512": {
        "avg_tokens_per_chunk": 512,
        "chunks_per_doc": 4,
        "total_chunks": 400000,
        "embedding_cost": 400000 * 512 / 1000 * 0.00002,  # $4.10
        "storage_gb": 400000 * 1536 * 4 / (1024**3),  # 2.4 GB
        "search_latency_ms": 45
    },
    "large_1024": {
        "avg_tokens_per_chunk": 1024,
        "chunks_per_doc": 2,
        "total_chunks": 200000,
        "embedding_cost": 200000 * 1024 / 1000 * 0.00002,  # $4.10
        "storage_gb": 200000 * 1536 * 4 / (1024**3),  # 1.2 GB
        "search_latency_ms": 50
    }
}

# Verdict: All same embedding cost (same total tokens)
# Difference: Storage (small needs 4x more than large)
#             Latency (small is faster, fewer vectors to scan)
#             Quality: Medium (512) offers best balance
```

**Recommendation**: Start with **512 tokens** for most use cases. Adjust based on:
- **Increase to 1024** if: Users need more context, long-form documents, cost-sensitive (less storage)
- **Decrease to 256** if: Short factual queries, latency critical, need very precise retrieval

### Overlap Strategy

**Why overlap matters**:
```
Without overlap:
Chunk 1: "...patients with diabetes mellitus."
Chunk 2: "Type 2 is more common..."

Query: "diabetes type 2 symptoms"
Problem: "diabetes" in Chunk 1, "type 2" in Chunk 2 → neither matches well!

With 50-token overlap:
Chunk 1: "...patients with diabetes mellitus. Type 2 is..."
Chunk 2: "...diabetes mellitus. Type 2 is more common..."

Query: "diabetes type 2 symptoms"
Result: ✅ Both chunks now contain "diabetes Type 2" → better matches!
```

**Overlap percentage guidelines**:

| Overlap | Token Count (512-token chunks) | Pros | Cons | When to Use |
|---------|-------------------------------|------|------|-------------|
| **0%** | 0 tokens | No duplicate embeddings, lowest cost | Risk losing context at boundaries | Short, independent paragraphs |
| **10%** | 51 tokens | Minimal cost increase, some protection | Limited boundary coverage | Blog posts, articles |
| **20%** (⭐ recommended) | 102 tokens | ✅ Good boundary coverage, manageable cost | 20% more embeddings | **Most documents** |
| **50%** | 256 tokens | Maximum boundary coverage | 2x embeddings cost! | Critical applications only |

**Cost impact of overlap** (100K documents, 4 chunks each):

```
No overlap:
- Chunks: 400,000
- Cost: $4.10

20% overlap:
- Chunks: 400,000 × 1.2 = 480,000 (20% more)
- Cost: $4.92 (+$0.82, or 20% more)

50% overlap:
- Chunks: 400,000 × 2 = 800,000 (100% more!)
- Cost: $8.20 (+$4.10, or 100% more)

Verdict: 20% overlap sweet spot (small cost, good quality)
```

**Recommended approach**: Use **10-20% overlap** for most use cases.

### Structure-Aware Chunking Implementation

**Why structure matters** (from earlier example):

```
PDF: Medical research paper with tables

Character-based chunking (BAD):
Chunk 1: "...showed improvement. | T"
Chunk 2: "able 1: Patient Outcom | es..."
Result: ❌ Table split across chunks, unreadable

Structure-aware chunking (GOOD):
Chunk 1: "...showed improvement."
Chunk 2: "[TABLE 1]\n| Patient | Outcome | Recovery |\n|---------|---------|----------|\n..."
Result: ✅ Table intact, searchable, meaningful
```

**The code below implements production-ready structure-aware chunking for PDFs using Azure AI Document Intelligence:**

> **What this code does**:
> 1. **PDFDocumentParser**: Parses PDF layout with Azure AI Document Intelligence
>    - Extracts tables, headers, paragraphs with bounding boxes
>    - Preserves document structure (sections, page numbers)
>    - Converts tables to markdown format for searchability
> 
> 2. **DocumentChunk dataclass**: Represents a single searchable chunk
>    - Tracks content, type (paragraph/table/header), page number
>    - Includes metadata for filtering and ranking
> 
> 3. **StructureAwareChunker**: Intelligent chunking that respects structure
>    - Never splits tables across chunks
>    - Keeps section headers with their content
>    - Implements configurable overlap
>    - Maintains parent section context
> 
> 4. **Usage**: Complete PDF → chunks pipeline
>    - Parse PDF → Extract structure → Chunk intelligently → Ready for embedding

```python
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

class ChunkType(Enum):
    """Types of document chunks."""
    PARAGRAPH = "paragraph"
    TABLE = "table"
    HEADER = "header"
    LIST = "list"

@dataclass
class DocumentChunk:
    """
    Represents a chunk of document content.
    
    Attributes:
        chunk_id: Unique identifier for this chunk
        content: Text content of the chunk
        chunk_type: Type of chunk (paragraph, table, header)
        page_number: Source page number
        bounding_box: Optional bounding box coordinates
        parent_section: Parent section title (for context)
        char_count: Character count
        token_estimate: Estimated token count
    """
    chunk_id: str
    content: str
    chunk_type: ChunkType
    page_number: int
    bounding_box: Optional[dict]
    parent_section: Optional[str]
    char_count: int
    token_estimate: int
    
    def to_search_document(self, doc_id: str, index: int) -> dict:
        """Convert chunk to Azure AI Search document format."""
        return {
            "id": f"{doc_id}_{index}",
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "page_number": self.page_number,
            "parent_section": self.parent_section,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
            "document_id": doc_id
        }

class PDFDocumentParser:
    """
    Parse PDF documents using Azure AI Document Intelligence.
    
    Extracts structured information including:
    - Text paragraphs with roles (heading, paragraph, etc.)
    - Tables with cell structure
    - Page layout and bounding boxes
    - Document sections based on headings
    """
    
    def __init__(self, endpoint: str, api_key: str):
        self.client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
    
    def parse_pdf(self, pdf_path: str) -> dict:
        """
        Parse PDF and extract structured content.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with structured document content
        """
        with open(pdf_path, "rb") as f:
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",  # Layout analysis model
                document=f
            )
        
        result = poller.result()
        
        return {
            "pages": self._extract_pages(result.pages),
            "paragraphs": self._extract_paragraphs(result.paragraphs),
            "tables": self._extract_tables(result.tables),
            "sections": self._extract_sections(result)
        }
    
    def _extract_pages(self, pages) -> List[dict]:
        """Extract page information."""
        return [
            {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "angle": page.angle,
                "unit": page.unit,
                "lines": [
                    {
                        "content": line.content,
                        "bounding_box": line.polygon
                    }
                    for line in page.lines
                ] if page.lines else []
            }
            for page in pages
        ]
    
    def _extract_paragraphs(self, paragraphs) -> List[dict]:
        """Extract paragraph information."""
        if not paragraphs:
            return []
            
        return [
            {
                "content": para.content,
                "bounding_regions": [
                    {
                        "page_number": region.page_number,
                        "polygon": region.polygon
                    }
                    for region in para.bounding_regions
                ] if para.bounding_regions else [],
                "role": para.role if hasattr(para, 'role') else None
            }
            for para in paragraphs
        ]
    
    def _extract_tables(self, tables) -> List[dict]:
        """Extract table information with structure."""
        if not tables:
            return []
            
        extracted_tables = []
        for table in tables:
            # Convert table to markdown format
            rows = {}
            for cell in table.cells:
                row_idx = cell.row_index
                if row_idx not in rows:
                    rows[row_idx] = {}
                rows[row_idx][cell.column_index] = cell.content
            
            # Build markdown table
            markdown_rows = []
            for row_idx in sorted(rows.keys()):
                row_cells = rows[row_idx]
                row_content = " | ".join(
                    row_cells.get(i, "") 
                    for i in range(table.column_count)
                )
                markdown_rows.append(f"| {row_content} |")
                
                # Add header separator after first row
                if row_idx == 0:
                    separator = " | ".join(["---"] * table.column_count)
                    markdown_rows.append(f"| {separator} |")
            
            extracted_tables.append({
                "row_count": table.row_count,
                "column_count": table.column_count,
                "markdown": "\n".join(markdown_rows),
                "bounding_regions": [
                    {
                        "page_number": region.page_number,
                        "polygon": region.polygon
                    }
                    for region in table.bounding_regions
                ] if table.bounding_regions else []
            })
        
        return extracted_tables
    
    def _extract_sections(self, result) -> List[dict]:
        """Extract document sections based on styles."""
        sections = []
        current_section = None
        
        if hasattr(result, 'paragraphs') and result.paragraphs:
            for para in result.paragraphs:
                # Check if paragraph is a heading
                role = getattr(para, 'role', None)
                
                if role and 'heading' in role.lower():
                    if current_section:
                        sections.append(current_section)
                    current_section = {
                        "title": para.content,
                        "level": self._extract_heading_level(role),
                        "content_paragraphs": []
                    }
                elif current_section:
                    current_section["content_paragraphs"].append(para.content)
            
            if current_section:
                sections.append(current_section)
        
        return sections
    
    def _extract_heading_level(self, role: str) -> int:
        """Extract heading level from role string."""
        import re
        match = re.search(r'(\d+)', role)
        return int(match.group(1)) if match else 1

class StructureAwareChunker:
    """
    Chunk documents while preserving structure.
    
    Key principles:
    1. Never split tables across chunks
    2. Keep headers with their content
    3. Respect paragraph boundaries
    4. Maintain parent section context
    """
    
    def __init__(
        self,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 50,
        preserve_tables: bool = True,
        preserve_sections: bool = True
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.preserve_tables = preserve_tables
        self.preserve_sections = preserve_sections
        
    def chunk_document(
        self,
        parsed_doc: dict,
        doc_id: str
    ) -> List[DocumentChunk]:
        """
        Chunk document respecting structure.
        
        Args:
            parsed_doc: Output from PDFDocumentParser
            doc_id: Document identifier
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        # Process sections if available
        if self.preserve_sections and parsed_doc.get("sections"):
            chunks.extend(
                self._chunk_sections(parsed_doc["sections"], doc_id)
            )
        
        # Process tables separately
        if self.preserve_tables and parsed_doc.get("tables"):
            chunks.extend(
                self._chunk_tables(parsed_doc["tables"], doc_id)
            )
        
        # Process remaining paragraphs
        if parsed_doc.get("paragraphs"):
            chunks.extend(
                self._chunk_paragraphs(
                    parsed_doc["paragraphs"],
                    doc_id,
                    exclude_roles=["heading"] if self.preserve_sections else []
                )
            )
        
        # Add chunk indices
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{doc_id}_chunk_{i:04d}"
        
        return chunks
    
    def _chunk_sections(
        self,
        sections: List[dict],
        doc_id: str
    ) -> List[DocumentChunk]:
        """Chunk document sections."""
        chunks = []
        
        for section in sections:
            section_title = section["title"]
            section_content = "\n\n".join(section["content_paragraphs"])
            
            # Combine title and content
            full_content = f"# {section_title}\n\n{section_content}"
            
            # Check if section fits in one chunk
            tokens = self._estimate_tokens(full_content)
            
            if tokens <= self.max_chunk_tokens:
                # Single chunk for entire section
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_section",
                    content=full_content,
                    chunk_type=ChunkType.HEADER,
                    page_number=1,  # TODO: Get from bounding regions
                    bounding_box=None,
                    parent_section=section_title,
                    char_count=len(full_content),
                    token_estimate=tokens
                ))
            else:
                # Split section content into smaller chunks
                section_chunks = self._split_text_with_overlap(
                    text=section_content,
                    prefix=f"# {section_title}\n\n",
                    parent_section=section_title
                )
                chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_tables(
        self,
        tables: List[dict],
        doc_id: str
    ) -> List[DocumentChunk]:
        """Chunk tables (keep tables whole)."""
        chunks = []
        
        for i, table in enumerate(tables):
            markdown = table["markdown"]
            
            # Get page number from bounding regions
            page_number = 1
            if table.get("bounding_regions"):
                page_number = table["bounding_regions"][0]["page_number"]
            
            # Create context-rich table chunk
            table_content = f"[TABLE {i+1}]\n{markdown}"
            
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_table_{i}",
                content=table_content,
                chunk_type=ChunkType.TABLE,
                page_number=page_number,
                bounding_box=table.get("bounding_regions", [{}])[0].get("polygon"),
                parent_section=None,
                char_count=len(table_content),
                token_estimate=self._estimate_tokens(table_content)
            ))
        
        return chunks
    
    def _chunk_paragraphs(
        self,
        paragraphs: List[dict],
        doc_id: str,
        exclude_roles: List[str] = None
    ) -> List[DocumentChunk]:
        """Chunk paragraphs with overlap."""
        exclude_roles = exclude_roles or []
        chunks = []
        
        # Filter paragraphs
        filtered_paras = [
            p for p in paragraphs
            if not (p.get("role") and p["role"].lower() in exclude_roles)
        ]
        
        # Combine paragraphs into chunks
        current_chunk = []
        current_tokens = 0
        
        for para in filtered_paras:
            para_text = para["content"]
            para_tokens = self._estimate_tokens(para_text)
            
            # Check if adding paragraph exceeds limit
            if current_tokens + para_tokens > self.max_chunk_tokens:
                if current_chunk:
                    # Save current chunk
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc_id}_para",
                        content=chunk_text,
                        chunk_type=ChunkType.PARAGRAPH,
                        page_number=self._get_page_number(para),
                        bounding_box=self._get_bounding_box(para),
                        parent_section=None,
                        char_count=len(chunk_text),
                        token_estimate=current_tokens
                    ))
                    
                    # Start new chunk with overlap
                    if self.overlap_tokens > 0 and len(current_chunk) > 1:
                        # Keep last paragraph for overlap
                        current_chunk = [current_chunk[-1], para_text]
                        current_tokens = (
                            self._estimate_tokens(current_chunk[-2]) +
                            para_tokens
                        )
                    else:
                        current_chunk = [para_text]
                        current_tokens = para_tokens
                else:
                    # Single paragraph too long - split it
                    split_chunks = self._split_long_paragraph(para_text, doc_id)
                    chunks.extend(split_chunks)
            else:
                current_chunk.append(para_text)
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_para",
                content=chunk_text,
                chunk_type=ChunkType.PARAGRAPH,
                page_number=1,
                bounding_box=None,
                parent_section=None,
                char_count=len(chunk_text),
                token_estimate=current_tokens
            ))
        
        return chunks
    
    def _split_text_with_overlap(
        self,
        text: str,
        prefix: str = "",
        parent_section: str = None
    ) -> List[DocumentChunk]:
        """Split long text into overlapping chunks."""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_tokens = self._estimate_tokens(prefix)
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.max_chunk_tokens:
                if current_chunk:
                    chunk_text = prefix + " ".join(current_chunk)
                    chunks.append(DocumentChunk(
                        chunk_id="temp",
                        content=chunk_text,
                        chunk_type=ChunkType.PARAGRAPH,
                        page_number=1,
                        bounding_box=None,
                        parent_section=parent_section,
                        char_count=len(chunk_text),
                        token_estimate=current_tokens
                    ))
                    
                    # Overlap: keep last few sentences
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk,
                        self.overlap_tokens
                    )
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = sum(
                        self._estimate_tokens(s) for s in current_chunk
                    ) + self._estimate_tokens(prefix)
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens + self._estimate_tokens(prefix)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunk_text = prefix + " ".join(current_chunk)
            chunks.append(DocumentChunk(
                chunk_id="temp",
                content=chunk_text,
                chunk_type=ChunkType.PARAGRAPH,
                page_number=1,
                bounding_box=None,
                parent_section=parent_section,
                char_count=len(chunk_text),
                token_estimate=current_tokens
            ))
        
        return chunks
    
    def _split_long_paragraph(
        self,
        text: str,
        doc_id: str
    ) -> List[DocumentChunk]:
        """Split a very long paragraph."""
        return self._split_text_with_overlap(text, "", None)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(
        self,
        sentences: List[str],
        target_tokens: int
    ) -> List[str]:
        """Get last N sentences for overlap."""
        overlap = []
        tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            if tokens + sentence_tokens > target_tokens:
                break
            overlap.insert(0, sentence)
            tokens += sentence_tokens
        
        return overlap
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def _get_page_number(self, para: dict) -> int:
        """Extract page number from paragraph."""
        if para.get("bounding_regions"):
            return para["bounding_regions"][0].get("page_number", 1)
        return 1
    
    def _get_bounding_box(self, para: dict) -> Optional[dict]:
        """Extract bounding box from paragraph."""
        if para.get("bounding_regions"):
            return para["bounding_regions"][0].get("polygon")
        return None

# Example: End-to-end PDF processing
def process_pdf_for_search(
    pdf_path: str,
    doc_id: str,
    form_recognizer_endpoint: str,
    form_recognizer_key: str
) -> List[DocumentChunk]:
    """
    Complete PDF processing pipeline.
    
    Args:
        pdf_path: Path to PDF file
        doc_id: Document identifier
        form_recognizer_endpoint: Azure AI Document Intelligence endpoint
        form_recognizer_key: API key
        
    Returns:
        List of searchable document chunks
    """
    # Step 1: Parse PDF
    parser = PDFDocumentParser(
        endpoint=form_recognizer_endpoint,
        api_key=form_recognizer_key
    )
    parsed_doc = parser.parse_pdf(pdf_path)
    
    # Step 2: Chunk with structure awareness
    chunker = StructureAwareChunker(
        max_chunk_tokens=512,
        overlap_tokens=50,
        preserve_tables=True,
        preserve_sections=True
    )
    chunks = chunker.chunk_document(parsed_doc, doc_id)
    
    # Step 3: Convert to search documents
    search_docs = [
        chunk.to_search_document(doc_id, i)
        for i, chunk in enumerate(chunks)
    ]
    
    return search_docs
```

### Chunking Strategy Comparison

**Evaluating different chunking approaches** for your specific use case:

```python
class ChunkingStrategyEvaluator:
    """Compare different chunking strategies."""
    
    def __init__(self, embedder):
        self.embedder = embedder
        
    def evaluate_strategies(
        self,
        pdf_path: str,
        ground_truth_queries: List[dict]
    ) -> dict:
        """
        Evaluate different chunking strategies.
        
        Args:
            pdf_path: PDF to chunk
            ground_truth_queries: List of {query, expected_chunk_ids}
            
        Returns:
            Comparison metrics
        """
        strategies = {
            "fixed_256": {"max_chunk_tokens": 256, "overlap_tokens": 25},
            "fixed_512": {"max_chunk_tokens": 512, "overlap_tokens": 50},
            "fixed_1024": {"max_chunk_tokens": 1024, "overlap_tokens": 100},
            "structure_aware": {
                "max_chunk_tokens": 512,
                "overlap_tokens": 50,
                "preserve_tables": True,
                "preserve_sections": True
            }
        }
        
        results = {}
        
        for name, config in strategies.items():
            chunker = StructureAwareChunker(**config)
            # Evaluate chunking quality
            metrics = self._evaluate_chunking(
                chunker,
                pdf_path,
                ground_truth_queries
            )
            results[name] = metrics
        
        return results
    
    def _evaluate_chunking(
        self,
        chunker: StructureAwareChunker,
        pdf_path: str,
        ground_truth: List[dict]
    ) -> dict:
        """Evaluate a chunking strategy."""
        # Simplified evaluation
        return {
            "avg_chunk_size": 450,
            "num_chunks": 25,
            "table_preservation": 0.95,
            "context_coherence": 0.88
        }
```

### Key Chunking Best Practices

1. **For PDFs with structure** → Use structure-aware chunking (Azure AI Document Intelligence)
2. **For plain text** → Use sentence-boundary chunking
3. **Chunk size** → Start with 512 tokens (balanced)
4. **Overlap** → Use 10-20% overlap (50-100 tokens for 512-token chunks)
5. **Tables** → Never split across chunks, convert to markdown
6. **Headers** → Keep with their content, use as parent_section metadata
7. **Testing** → Evaluate chunking quality with sample queries before full indexing

**Cost optimization tip**: Larger chunks = fewer embeddings = lower cost, but risks lower precision. Balance with quality requirements.

---

## Data Enrichment

### Overview: Enriching Chunks with Metadata

After chunking PDFs, enrich each chunk with extracted metadata to improve searchability and filtering.

```
Chunks → Entity Extraction → Key Phrases → Language Detection → Enriched Chunks → Embeddings
```

**Why enrichment matters**:

```
Scenario: Legal contract search

Without enrichment:
- Chunk: "The party agrees to terms..."
- Search: "Microsoft contracts"
- Result: ❌ No match (company name not explicitly in this chunk)

With enrichment (entity extraction):
- Chunk: "The party agrees to terms..."
- Entities extracted: ["Microsoft Corporation", "Q2 2024", "$50M"]
- Entity metadata: {"organizations": ["Microsoft Corporation"]}
- Search: "Microsoft contracts" + filter: organizations contains "Microsoft"
- Result: ✅ Match! (entity metadata enables discovery)

Value: 40% recall improvement for entity-based queries
Cost: +$0.01 per 1K chunks (Azure AI Language entity extraction)
```

**Three enrichment approaches**:

1. **Azure AI Language** (recommended): Entities, key phrases, language detection, sentiment
   - Cost: ~$0.01 per 1K text records (entity extraction)
   - Quality: High accuracy, pre-trained models
   - Use when: Need standard NLP features, multilingual
   
2. **Custom regex patterns**: Extract domain-specific metadata (emails, dates, IDs)
   - Cost: Free (runs locally)
   - Quality: Depends on pattern quality
   - Use when: Need specific formats, cost-sensitive

3. **Hybrid**: Azure AI Language + custom patterns
   - Cost: ~$0.01 per 1K + compute
   - Quality: Best of both
   - Use when: Complex requirements, budget allows

### Azure AI Language Integration

**Complete enrichment implementation** with entity extraction, key phrases, and language detection:

```python
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict

class ChunkEnricher:
    """Enrich document chunks with AI-extracted metadata."""
    
    def __init__(
        self,
        language_endpoint: str,
        language_key: str
    ):
        self.client = TextAnalyticsClient(
            endpoint=language_endpoint,
            credential=AzureKeyCredential(language_key)
        )
    
    def enrich_chunk(self, chunk: DocumentChunk) -> dict:
            List of document chunks
        """
        chunks = []
        
        # Process sections if available
        if self.preserve_sections and parsed_doc.get("sections"):
            chunks.extend(
                self._chunk_sections(parsed_doc["sections"], doc_id)
            )
        
        # Process tables separately
        if self.preserve_tables and parsed_doc.get("tables"):
            chunks.extend(
                self._chunk_tables(parsed_doc["tables"], doc_id)
            )
        
        # Process remaining paragraphs
        if parsed_doc.get("paragraphs"):
            chunks.extend(
                self._chunk_paragraphs(
                    parsed_doc["paragraphs"],
                    doc_id,
                    exclude_roles=["heading"] if self.preserve_sections else []
                )
            )
        
        # Add chunk indices
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{doc_id}_chunk_{i:04d}"
        
        return chunks
    
    def _chunk_sections(
        self,
        sections: List[dict],
        doc_id: str
    ) -> List[DocumentChunk]:
        """Chunk document sections."""
        chunks = []
        
        for section in sections:
            section_title = section["title"]
            section_content = "\n\n".join(section["content_paragraphs"])
            
            # Combine title and content
            full_content = f"# {section_title}\n\n{section_content}"
            
            # Check if section fits in one chunk
            tokens = self._estimate_tokens(full_content)
            
            if tokens <= self.max_chunk_tokens:
                # Single chunk for entire section
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_section",
                    content=full_content,
                    chunk_type=ChunkType.HEADER,
                    page_number=1,  # TODO: Get from bounding regions
                    bounding_box=None,
                    parent_section=section_title,
                    char_count=len(full_content),
                    token_estimate=tokens
                ))
            else:
                # Split section content into smaller chunks
                section_chunks = self._split_text_with_overlap(
                    text=section_content,
                    prefix=f"# {section_title}\n\n",
                    parent_section=section_title
                )
                chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_tables(
        self,
        tables: List[dict],
        doc_id: str
    ) -> List[DocumentChunk]:
        """Chunk tables (keep tables whole)."""
        chunks = []
        
        for i, table in enumerate(tables):
            markdown = table["markdown"]
            
            # Get page number from bounding regions
            page_number = 1
            if table.get("bounding_regions"):
                page_number = table["bounding_regions"][0]["page_number"]
            
            # Create context-rich table chunk
            table_content = f"[TABLE {i+1}]\n{markdown}"
            
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_table_{i}",
                content=table_content,
                chunk_type=ChunkType.TABLE,
                page_number=page_number,
                bounding_box=table.get("bounding_regions", [{}])[0].get("polygon"),
                parent_section=None,
                char_count=len(table_content),
                token_estimate=self._estimate_tokens(table_content)
            ))
        
        return chunks
    
    def _chunk_paragraphs(
        self,
        paragraphs: List[dict],
        doc_id: str,
        exclude_roles: List[str] = None
    ) -> List[DocumentChunk]:
        """Chunk paragraphs with overlap."""
        exclude_roles = exclude_roles or []
        chunks = []
        
        # Filter paragraphs
        filtered_paras = [
            p for p in paragraphs
            if not (p.get("role") and p["role"].lower() in exclude_roles)
        ]
        
        # Combine paragraphs into chunks
        current_chunk = []
        current_tokens = 0
        
        for para in filtered_paras:
            para_text = para["content"]
            para_tokens = self._estimate_tokens(para_text)
            
            # Check if adding paragraph exceeds limit
            if current_tokens + para_tokens > self.max_chunk_tokens:
                if current_chunk:
                    # Save current chunk
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc_id}_para",
                        content=chunk_text,
                        chunk_type=ChunkType.PARAGRAPH,
                        page_number=self._get_page_number(para),
                        bounding_box=self._get_bounding_box(para),
                        parent_section=None,
                        char_count=len(chunk_text),
                        token_estimate=current_tokens
                    ))
                    
                    # Start new chunk with overlap
                    if self.overlap_tokens > 0 and len(current_chunk) > 1:
                        # Keep last paragraph for overlap
                        current_chunk = [current_chunk[-1], para_text]
                        current_tokens = (
                            self._estimate_tokens(current_chunk[-2]) +
                            para_tokens
                        )
                    else:
                        current_chunk = [para_text]
                        current_tokens = para_tokens
                else:
                    # Single paragraph too long - split it
                    split_chunks = self._split_long_paragraph(para_text, doc_id)
                    chunks.extend(split_chunks)
            else:
                current_chunk.append(para_text)
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_para",
                content=chunk_text,
                chunk_type=ChunkType.PARAGRAPH,
                page_number=1,
                bounding_box=None,
                parent_section=None,
                char_count=len(chunk_text),
                token_estimate=current_tokens
            ))
        
        return chunks
    
    def _split_text_with_overlap(
        self,
        text: str,
        prefix: str = "",
        parent_section: str = None
    ) -> List[DocumentChunk]:
        """Split long text into overlapping chunks."""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_tokens = self._estimate_tokens(prefix)
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.max_chunk_tokens:
                if current_chunk:
                    chunk_text = prefix + " ".join(current_chunk)
                    chunks.append(DocumentChunk(
                        chunk_id="temp",
                        content=chunk_text,
                        chunk_type=ChunkType.PARAGRAPH,
                        page_number=1,
                        bounding_box=None,
                        parent_section=parent_section,
                        char_count=len(chunk_text),
                        token_estimate=current_tokens
                    ))
                    
                    # Overlap: keep last few sentences
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk,
                        self.overlap_tokens
                    )
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = sum(
                        self._estimate_tokens(s) for s in current_chunk
                    ) + self._estimate_tokens(prefix)
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens + self._estimate_tokens(prefix)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunk_text = prefix + " ".join(current_chunk)
            chunks.append(DocumentChunk(
                chunk_id="temp",
                content=chunk_text,
                chunk_type=ChunkType.PARAGRAPH,
                page_number=1,
                bounding_box=None,
                parent_section=parent_section,
                char_count=len(chunk_text),
                token_estimate=current_tokens
            ))
        
        return chunks
    
    def _split_long_paragraph(
        self,
        text: str,
        doc_id: str
    ) -> List[DocumentChunk]:
        """Split a very long paragraph."""
        return self._split_text_with_overlap(text, "", None)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(
        self,
        sentences: List[str],
        target_tokens: int
    ) -> List[str]:
        """Get last N sentences for overlap."""
        overlap = []
        tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            if tokens + sentence_tokens > target_tokens:
                break
            overlap.insert(0, sentence)
            tokens += sentence_tokens
        
        return overlap
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def _get_page_number(self, para: dict) -> int:
        """Extract page number from paragraph."""
        if para.get("bounding_regions"):
            return para["bounding_regions"][0].get("page_number", 1)
        return 1
    
    def _get_bounding_box(self, para: dict) -> Optional[dict]:
        """Extract bounding box from paragraph."""
        if para.get("bounding_regions"):
            return para["bounding_regions"][0].get("polygon")
        return None

# Example: End-to-end PDF processing
def process_pdf_for_search(
    pdf_path: str,
    doc_id: str,
    form_recognizer_endpoint: str,
    form_recognizer_key: str
) -> List[DocumentChunk]:
    """
    Complete PDF processing pipeline.
    
    Args:
        pdf_path: Path to PDF file
        doc_id: Document identifier
        form_recognizer_endpoint: Azure AI Document Intelligence endpoint
        form_recognizer_key: API key
        
    Returns:
        List of searchable document chunks
    """
    # Step 1: Parse PDF
    parser = PDFDocumentParser(
        endpoint=form_recognizer_endpoint,
        api_key=form_recognizer_key
    )
    parsed_doc = parser.parse_pdf(pdf_path)
    
    # Step 2: Chunk with structure awareness
    chunker = StructureAwareChunker(
        max_chunk_tokens=512,
        overlap_tokens=50,
        preserve_tables=True,
        preserve_sections=True
    )
    chunks = chunker.chunk_document(parsed_doc, doc_id)
    
    # Step 3: Convert to search documents
    search_docs = [
        chunk.to_search_document(doc_id, i)
        for i, chunk in enumerate(chunks)
    ]
    
    return search_docs
```

### Chunking Strategy Comparison

```python
class ChunkingStrategyEvaluator:
    """Compare different chunking strategies."""
    
    def __init__(self, embedder):
        self.embedder = embedder
        
    def evaluate_strategies(
        self,
        pdf_path: str,
        ground_truth_queries: List[dict]
    ) -> dict:
        """
        Evaluate different chunking strategies.
        
        Args:
            pdf_path: PDF to chunk
            ground_truth_queries: List of {query, expected_chunk_ids}
            
        Returns:
            Comparison metrics
        """
        strategies = {
            "fixed_256": {"max_chunk_tokens": 256, "overlap_tokens": 25},
            "fixed_512": {"max_chunk_tokens": 512, "overlap_tokens": 50},
            "fixed_1024": {"max_chunk_tokens": 1024, "overlap_tokens": 100},
            "structure_aware": {
                "max_chunk_tokens": 512,
                "overlap_tokens": 50,
                "preserve_tables": True,
                "preserve_sections": True
            }
        }
        
        results = {}
        
        for name, config in strategies.items():
            chunker = StructureAwareChunker(**config)
            # Evaluate chunking quality
            metrics = self._evaluate_chunking(
                chunker,
                pdf_path,
                ground_truth_queries
            )
            results[name] = metrics
        
        return results
    
    def _evaluate_chunking(
        self,
        chunker: StructureAwareChunker,
        pdf_path: str,
        ground_truth: List[dict]
    ) -> dict:
        """Evaluate a chunking strategy."""
        # Simplified evaluation
        return {
            "avg_chunk_size": 450,
            "num_chunks": 25,
            "table_preservation": 0.95,
            "context_coherence": 0.88
        }
```

---

## Data Enrichment

### Overview: Enriching Chunks with Metadata

After chunking PDFs, enrich each chunk with extracted metadata to improve searchability and filtering.

```
Chunks → Entity Extraction → Key Phrases → Language Detection → Enriched Chunks → Embeddings
```

### Azure AI Language Integration

```python
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict

class ChunkEnricher:
    """Enrich document chunks with AI-extracted metadata."""
    
    def __init__(
        self,
        language_endpoint: str,
        language_key: str
    ):
        self.client = TextAnalyticsClient(
            endpoint=language_endpoint,
            credential=AzureKeyCredential(language_key)
        )
    
    def enrich_chunk(self, chunk: DocumentChunk) -> dict:
        """
        Enrich a single chunk with metadata.
        
        Args:
            chunk: DocumentChunk to enrich
            
        Returns:
            Enriched metadata dictionary
        """
        content = chunk.content
        
        # Extract entities
        entities = self._extract_entities(content)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(content)
        
        # Detect language
        language = self._detect_language(content)
        
        # Build enriched metadata
        enriched = {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "chunk_type": chunk.chunk_type.value,
            "page_number": chunk.page_number,
            "parent_section": chunk.parent_section,
            "char_count": chunk.char_count,
            "token_estimate": chunk.token_estimate,
            # Enrichment fields
            "entities": entities,
            "key_phrases": key_phrases,
            "language": language,
            "entity_count": len(entities),
            "key_phrase_count": len(key_phrases)
        }
        
        return enriched
    
    def _extract_entities(self, text: str) -> List[dict]:
        """Extract named entities from text."""
        try:
            result = self.client.recognize_entities(
                documents=[text],
                language="en"
            )[0]
            
            if result.is_error:
                return []
            
            entities = []
            for entity in result.entities:
                entities.append({
                    "text": entity.text,
                    "category": entity.category,
                    "subcategory": entity.subcategory,
                    "confidence": entity.confidence_score
                })
            
            return entities
            
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        try:
            result = self.client.extract_key_phrases(
                documents=[text],
                language="en"
            )[0]
            
            if result.is_error:
                return []
            
            return list(result.key_phrases)
            
        except Exception as e:
            print(f"Key phrase extraction error: {e}")
            return []
    
    def _detect_language(self, text: str) -> str:
        """Detect primary language of text."""
        try:
            result = self.client.detect_language(
                documents=[text]
            )[0]
            
            if result.is_error:
                return "en"
            
            return result.primary_language.iso6391_name
            
        except Exception as e:
            print(f"Language detection error: {e}")
            return "en"
    
    def enrich_batch(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 10
    ) -> List[dict]:
        """
        Enrich multiple chunks in batches.
        
        Args:
            chunks: List of DocumentChunks
            batch_size: Batch size for API calls
            
        Returns:
            List of enriched metadata dictionaries
        """
        enriched_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            for chunk in batch:
                enriched = self.enrich_chunk(chunk)
                enriched_chunks.append(enriched)
        
        return enriched_chunks

class CustomMetadataExtractor:
    """Extract custom domain-specific metadata."""
    
    def __init__(self):
        self.patterns = {}
    
    def add_pattern(self, name: str, pattern: str):
        """
        Add custom extraction pattern.
        
        Args:
            name: Metadata field name
            pattern: Regex pattern to match
        """
        import re
        self.patterns[name] = re.compile(pattern)
    
    def extract_metadata(self, text: str) -> dict:
        """
        Extract custom metadata using patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        
        for name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            metadata[name] = matches if matches else []
        
        return metadata

# Example: Domain-specific patterns
def create_document_metadata_extractor() -> CustomMetadataExtractor:
    """Create extractor for common document patterns."""
    extractor = CustomMetadataExtractor()
    
    # Date patterns
    extractor.add_pattern(
        "dates",
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    )
    
    # Email addresses
    extractor.add_pattern(
        "emails",
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    
    # Phone numbers
    extractor.add_pattern(
        "phones",
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    )
    
    # Document IDs (example pattern)
    extractor.add_pattern(
        "document_ids",
        r'\b[A-Z]{2,4}-\d{4,8}\b'
    )
    
    return extractor

class EnrichedChunkIndexer:
    """Index enriched chunks to Azure AI Search."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def prepare_search_document(
        self,
        enriched_chunk: dict,
        embedding: List[float] = None
    ) -> dict:
        """
        Prepare enriched chunk for indexing.
        
        Args:
            enriched_chunk: Enriched chunk metadata
            embedding: Optional embedding vector
            
        Returns:
            Search document ready for indexing
        """
        # Flatten entities for search
        entity_names = [e["text"] for e in enriched_chunk.get("entities", [])]
        entity_categories = list(set(
            e["category"] for e in enriched_chunk.get("entities", [])
        ))
        
        search_doc = {
            "id": enriched_chunk["chunk_id"],
            "content": enriched_chunk["content"],
            "content_vector": embedding,
            "chunk_type": enriched_chunk["chunk_type"],
            "page_number": enriched_chunk["page_number"],
            "parent_section": enriched_chunk["parent_section"],
            "language": enriched_chunk.get("language", "en"),
            # Searchable enrichment fields
            "key_phrases": enriched_chunk.get("key_phrases", []),
            "entity_names": entity_names,
            "entity_categories": entity_categories,
            "entity_count": enriched_chunk.get("entity_count", 0),
            # Filterable metadata
            "has_entities": len(entity_names) > 0,
            "has_key_phrases": len(enriched_chunk.get("key_phrases", [])) > 0
        }
        
        return search_doc
    
    def index_enriched_chunks(
        self,
        enriched_chunks: List[dict],
        embeddings: List[List[float]] = None
    ):
        """
        Index enriched chunks to search service.
        
        Args:
            enriched_chunks: List of enriched chunks
            embeddings: Optional list of embedding vectors
        """
        search_docs = []
        
        for i, chunk in enumerate(enriched_chunks):
            embedding = embeddings[i] if embeddings else None
            doc = self.prepare_search_document(chunk, embedding)
            search_docs.append(doc)
        
        # Upload to search index
        result = self.search_client.upload_documents(
            documents=search_docs
        )
        
        return result

# Complete enrichment pipeline
def enrich_and_index_pdf(
    pdf_path: str,
    doc_id: str,
    form_recognizer_endpoint: str,
    form_recognizer_key: str,
    language_endpoint: str,
    language_key: str,
    openai_embedder,
    search_client
) -> dict:
    """
    Complete pipeline: Parse → Chunk → Enrich → Embed → Index.
    
    Args:
        pdf_path: Path to PDF file
        doc_id: Document identifier
        form_recognizer_endpoint: Azure AI Document Intelligence endpoint
        form_recognizer_key: Document Intelligence key
        language_endpoint: Azure AI Language endpoint
        language_key: Language service key
        openai_embedder: Embedder instance
        search_client: Search index client
        
    Returns:
        Pipeline execution summary
    """
    # Step 1: Parse PDF
    parser = PDFDocumentParser(
        endpoint=form_recognizer_endpoint,
        api_key=form_recognizer_key
    )
    parsed_doc = parser.parse_pdf(pdf_path)
    
    # Step 2: Chunk document
    chunker = StructureAwareChunker(
        max_chunk_tokens=512,
        overlap_tokens=50,
        preserve_tables=True,
        preserve_sections=True
    )
    chunks = chunker.chunk_document(parsed_doc, doc_id)
    
    # Step 3: Enrich chunks
    enricher = ChunkEnricher(
        language_endpoint=language_endpoint,
        language_key=language_key
    )
    enriched_chunks = enricher.enrich_batch(chunks)
    
    # Step 4: Generate embeddings
    embeddings = []
    for chunk in enriched_chunks:
        embedding = openai_embedder.embed_text(chunk["content"])
        embeddings.append(embedding)
    
    # Step 5: Index to search
    indexer = EnrichedChunkIndexer(search_client)
    result = indexer.index_enriched_chunks(enriched_chunks, embeddings)
    
    return {
        "document_id": doc_id,
        "chunks_created": len(chunks),
        "chunks_enriched": len(enriched_chunks),
        "chunks_indexed": len(result),
        "pipeline_status": "success"
    }
```

### Index Schema for Enriched Chunks

```python
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)

def create_enriched_chunk_index() -> SearchIndex:
    """Create search index for enriched document chunks."""
    
    fields = [
        SearchField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        SearchField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name="en.microsoft"
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="vector-profile"
        ),
        SearchField(
            name="chunk_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        SearchField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
            facetable=True
        ),
        SearchField(
            name="parent_section",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True
        ),
        SearchField(
            name="language",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        # Enrichment fields
        SearchField(
            name="key_phrases",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True
        ),
        SearchField(
            name="entity_names",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True
        ),
        SearchField(
            name="entity_categories",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
            facetable=True
        ),
        SearchField(
            name="entity_count",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True
        ),
        SearchField(
            name="has_entities",
            type=SearchFieldDataType.Boolean,
            filterable=True
        ),
        SearchField(
            name="has_key_phrases",
            type=SearchFieldDataType.Boolean,
            filterable=True
        )
    ]
    
    # Vector search configuration
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config"
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw-config")
        ]
    )
    
    index = SearchIndex(
        name="enriched-chunks",
        fields=fields,
        vector_search=vector_search
    )
    
    return index
```

### Querying Enriched Content

```python
class EnrichedSearchQuery:
    """Query enriched chunks with filters."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def search_by_entity(
        self,
        query: str,
        entity_category: str = None,
        top: int = 10
    ):
        """
        Search chunks containing specific entities.
        
        Args:
            query: Search query
            entity_category: Filter by entity category (Person, Organization, Location, etc.)
            top: Number of results
        """
        filter_expr = None
        if entity_category:
            filter_expr = f"entity_categories/any(c: c eq '{entity_category}')"
        
        results = self.search_client.search(
            search_text=query,
            filter=filter_expr,
            select=["content", "entity_names", "entity_categories", "page_number"],
            top=top
        )
        
        return list(results)
    
    def search_by_key_phrase(
        self,
        key_phrase: str,
        top: int = 10
    ):
        """Search chunks by key phrase."""
        filter_expr = f"key_phrases/any(k: k eq '{key_phrase}')"
        
        results = self.search_client.search(
            search_text="*",
            filter=filter_expr,
            select=["content", "key_phrases", "page_number"],
            top=top
        )
        
        return list(results)
    
    def search_with_enrichment_boost(
        self,
        query: str,
        boost_entities: bool = True,
        boost_key_phrases: bool = True
    ):
        """
        Search with enrichment-based scoring boost.
        
        Chunks with more entities/key phrases rank higher.
        """
        # Build scoring profile query
        scoring_params = []
        if boost_entities:
            scoring_params.append("entity_count-2.0")
        
        results = self.search_client.search(
            search_text=query,
            scoring_parameters=scoring_params,
            select=["content", "entity_count", "key_phrases", "page_number"],
            top=10
        )
        
        return list(results)
```

> **Note**: For comprehensive enrichment pipelines including skillsets, OCR, custom skills, 
> and knowledge store, see [Knowledge Mining (Page 26)](./26-knowledge-mining.md).

---

## Embedding Generation

### Single Text Embedding

```python
class AzureOpenAIEmbedder:
    """Azure OpenAI embedding generator."""
    
    def __init__(self, api_key, endpoint, api_version="2024-02-01"):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment_name = "text-embedding-ada-002"
        self.max_tokens = 8191  # ada-002 limit
    
    def embed_text(self, text):
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            List of 1536 floats
        """
        try:
            # Truncate if necessary
            text = self._truncate_text(text, self.max_tokens)
            
            response = self.client.embeddings.create(
                input=text,
                model=self.deployment_name
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def _truncate_text(self, text, max_tokens):
        """Truncate text to fit within token limit."""
        # Simple approximation: ~4 chars per token
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars]
        return text
    
    def get_embedding_stats(self, embedding):
        """Get statistics about an embedding."""
        import numpy as np
        
        if embedding is None:
            return None
        
        arr = np.array(embedding)
        return {
            'dimensions': len(embedding),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'l2_norm': float(np.linalg.norm(arr))
        }

# Usage
embedder = AzureOpenAIEmbedder(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

text = "Azure AI Search provides powerful search capabilities"
embedding = embedder.embed_text(text)
stats = embedder.get_embedding_stats(embedding)

print(f"Embedding dimensions: {stats['dimensions']}")
print(f"L2 norm: {stats['l2_norm']:.4f}")
```

### Document Embedding with Chunking

> **Note**: For PDF documents with structure (tables, headers), see the 
> [Document Parsing & Chunking](#document-parsing--chunking) section above for 
> production-ready structure-aware chunking using Azure AI Document Intelligence.
> 
> The basic chunking below is suitable for plain text only.

```python
class DocumentEmbedder:
    """Embed long documents with basic chunking strategy (plain text only)."""
    
    def __init__(self, embedder, chunk_size=1000, overlap=200):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text):
        """
        Split text into overlapping chunks (basic - use StructureAwareChunker for PDFs).
        
        Args:
            text: Plain text string
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.7:  # At least 70% of chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append({
                'text': chunk.strip(),
                'start': start,
                'end': end
            })
            
            start = end - self.overlap
        
        return chunks
    
    def embed_document(self, text, aggregate_method='mean'):
        """
        Embed a long document.
        
        Args:
            text: Document text
            aggregate_method: 'mean', 'max', or 'first'
            
        Returns:
            Aggregated embedding vector
        """
        import numpy as np
        
        # Split into chunks
        chunks = self.chunk_text(text)
        
        # Generate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            embedding = self.embedder.embed_text(chunk['text'])
            if embedding:
                embeddings.append(embedding)
        
        if not embeddings:
            return None
        
        # Aggregate embeddings
        embeddings_array = np.array(embeddings)
        
        if aggregate_method == 'mean':
            final_embedding = np.mean(embeddings_array, axis=0)
        elif aggregate_method == 'max':
            final_embedding = np.max(embeddings_array, axis=0)
        elif aggregate_method == 'first':
            final_embedding = embeddings_array[0]
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate_method}")
        
        return final_embedding.tolist()
    
    def embed_with_metadata(self, document):
        """
        Embed document with title and content separately.
        
        Args:
            document: Dict with 'title' and 'content'
            
        Returns:
            Dict with separate embeddings
        """
        result = {
            'title_embedding': None,
            'content_embedding': None,
            'combined_embedding': None
        }
        
        # Embed title
        if document.get('title'):
            result['title_embedding'] = self.embedder.embed_text(document['title'])
        
        # Embed content
        if document.get('content'):
            result['content_embedding'] = self.embed_document(document['content'])
        
        # Create combined embedding (weighted average)
        if result['title_embedding'] and result['content_embedding']:
            import numpy as np
            title_weight = 0.3
            content_weight = 0.7
            
            combined = (
                np.array(result['title_embedding']) * title_weight +
                np.array(result['content_embedding']) * content_weight
            )
            result['combined_embedding'] = combined.tolist()
        
        return result

# Usage
doc_embedder = DocumentEmbedder(embedder, chunk_size=1000, overlap=200)

long_document = """
[Your long document text here...]
This could be several pages of content that needs to be chunked.
"""

embedding = doc_embedder.embed_document(long_document, aggregate_method='mean')
print(f"Document embedded with {len(embedding)} dimensions")
```

---

## Batch Processing

### Efficient Batch Embedding

```python
import asyncio
from typing import List
import time

class BatchEmbedder:
    """Efficient batch embedding processor."""
    
    def __init__(self, embedder, batch_size=100, rate_limit_rpm=3000):
        self.embedder = embedder
        self.batch_size = batch_size
        self.rate_limit_rpm = rate_limit_rpm
        self.requests_per_second = rate_limit_rpm / 60
        self.min_request_interval = 1.0 / self.requests_per_second
    
    def embed_batch(self, texts):
        """
        Embed a batch of texts with rate limiting.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embeddings
        """
        embeddings = []
        last_request_time = 0
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            
            try:
                # Azure OpenAI supports batch embedding
                response = self.embedder.client.embeddings.create(
                    input=batch,
                    model=self.embedder.deployment_name
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                last_request_time = time.time()
                
                print(f"Processed {i + len(batch)}/{len(texts)} texts")
                
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Add None for failed embeddings
                embeddings.extend([None] * len(batch))
        
        return embeddings
    
    async def embed_batch_async(self, texts):
        """Asynchronous batch embedding."""
        tasks = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            task = asyncio.create_task(self._embed_batch_async(batch))
            tasks.append(task)
            
            # Add delay between batches for rate limiting
            await asyncio.sleep(self.min_request_interval)
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_embeddings = []
        for batch_result in results:
            all_embeddings.extend(batch_result)
        
        return all_embeddings
    
    async def _embed_batch_async(self, batch):
        """Embed a single batch asynchronously."""
        try:
            response = self.embedder.client.embeddings.create(
                input=batch,
                model=self.embedder.deployment_name
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error in async batch: {e}")
            return [None] * len(batch)

# Usage
batch_embedder = BatchEmbedder(embedder, batch_size=100)

# Process large dataset
texts = [
    "Document 1 content...",
    "Document 2 content...",
    # ... thousands of documents
]

# Synchronous processing
embeddings = batch_embedder.embed_batch(texts)

# Async processing (faster)
# embeddings = await batch_embedder.embed_batch_async(texts)
```

### Progress Tracking and Resume

```python
import json
import os

class ResumableEmbedder:
    """Batch embedder with progress tracking and resume capability."""
    
    def __init__(self, embedder, checkpoint_file="embedding_checkpoint.json"):
        self.embedder = embedder
        self.checkpoint_file = checkpoint_file
        self.checkpoint_interval = 100  # Save every 100 documents
    
    def embed_documents_with_resume(self, documents, id_field='id'):
        """
        Embed documents with checkpoint/resume support.
        
        Args:
            documents: List of document dicts
            id_field: Field name containing document ID
            
        Returns:
            Dict mapping document IDs to embeddings
        """
        # Load existing checkpoint if available
        embeddings_map = self._load_checkpoint()
        
        total = len(documents)
        processed = len(embeddings_map)
        
        print(f"Resuming from {processed}/{total} documents")
        
        for i, doc in enumerate(documents):
            doc_id = doc[id_field]
            
            # Skip if already processed
            if doc_id in embeddings_map:
                continue
            
            # Generate embedding
            text = self._extract_text(doc)
            embedding = self.embedder.embed_text(text)
            
            if embedding:
                embeddings_map[doc_id] = {
                    'embedding': embedding,
                    'text_length': len(text),
                    'timestamp': time.time()
                }
                
                processed += 1
                
                # Save checkpoint periodically
                if processed % self.checkpoint_interval == 0:
                    self._save_checkpoint(embeddings_map)
                    print(f"Checkpoint saved: {processed}/{total}")
        
        # Final save
        self._save_checkpoint(embeddings_map)
        print(f"✅ Complete: {processed}/{total} documents embedded")
        
        return embeddings_map
    
    def _extract_text(self, doc):
        """Extract text from document for embedding."""
        parts = []
        
        if doc.get('title'):
            parts.append(doc['title'])
        if doc.get('content'):
            parts.append(doc['content'])
        if doc.get('description'):
            parts.append(doc['description'])
        
        return " ".join(parts)
    
    def _load_checkpoint(self):
        """Load checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_checkpoint(self, embeddings_map):
        """Save checkpoint file."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(embeddings_map, f)

# Usage
resumable = ResumableEmbedder(embedder)

documents = [
    {'id': '1', 'title': 'Doc 1', 'content': '...'},
    {'id': '2', 'title': 'Doc 2', 'content': '...'},
    # ... many documents
]

embeddings = resumable.embed_documents_with_resume(documents)
```

---

## Cost Optimization

### Token Usage Tracking

```python
class CostAwareEmbedder:
    """Embedding generator with cost tracking."""
    
    def __init__(self, embedder, cost_per_1k_tokens=0.0001):
        self.embedder = embedder
        self.cost_per_1k_tokens = cost_per_1k_tokens  # ada-002 pricing
        self.total_tokens = 0
        self.total_cost = 0
    
    def estimate_tokens(self, text):
        """Estimate token count (rough approximation)."""
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def embed_text_with_cost(self, text):
        """Embed text and track cost."""
        embedding = self.embedder.embed_text(text)
        
        if embedding:
            tokens = self.estimate_tokens(text)
            cost = (tokens / 1000) * self.cost_per_1k_tokens
            
            self.total_tokens += tokens
            self.total_cost += cost
        
        return embedding
    
    def get_cost_summary(self):
        """Get cost summary."""
        return {
            'total_tokens': self.total_tokens,
            'total_cost_usd': self.total_cost,
            'average_tokens_per_request': self.total_tokens / max(1, self.request_count),
            'cost_per_embedding': self.total_cost / max(1, self.request_count)
        }
    
    def reset_tracking(self):
        """Reset cost tracking."""
        self.total_tokens = 0
        self.total_cost = 0
        self.request_count = 0

# Usage
cost_embedder = CostAwareEmbedder(embedder)

# Process documents
for doc in documents:
    embedding = cost_embedder.embed_text_with_cost(doc['content'])

# Get cost summary
summary = cost_embedder.get_cost_summary()
print(f"Total cost: ${summary['total_cost_usd']:.4f}")
print(f"Total tokens: {summary['total_tokens']:,}")
```

### Caching Strategy

```python
import hashlib
import pickle

class CachedEmbedder:
    """Embedder with local caching to reduce API calls."""
    
    def __init__(self, embedder, cache_dir="embedding_cache"):
        self.embedder = embedder
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, text):
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        """Get file path for cache entry."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def embed_text(self, text):
        """Embed text with caching."""
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                embedding = pickle.load(f)
            self.cache_hits += 1
            return embedding
        
        # Generate embedding
        embedding = self.embedder.embed_text(text)
        
        if embedding:
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            self.cache_misses += 1
        
        return embedding
    
    def get_cache_stats(self):
        """Get cache performance stats."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cost_saved': self.cache_hits * 0.0001  # Approximate savings
        }

# Usage
cached_embedder = CachedEmbedder(embedder)

# First run - cache misses
embedding1 = cached_embedder.embed_text("Sample text")

# Second run - cache hit (no API call, no cost!)
embedding2 = cached_embedder.embed_text("Sample text")

stats = cached_embedder.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
print(f"Cost saved: ${stats['cost_saved']:.4f}")
```

---

## Performance Tuning

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ParallelEmbedder:
    """Multi-threaded embedding processor."""
    
    def __init__(self, embedder, max_workers=5):
        self.embedder = embedder
        self.max_workers = max_workers
        self.lock = threading.Lock()
    
    def embed_batch_parallel(self, texts):
        """Process embeddings in parallel."""
        embeddings = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.embedder.embed_text, text): i
                for i, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    embedding = future.result()
                    embeddings[index] = embedding
                except Exception as e:
                    print(f"Error processing text {index}: {e}")
        
        return embeddings

# Usage
parallel_embedder = ParallelEmbedder(embedder, max_workers=5)
embeddings = parallel_embedder.embed_batch_parallel(texts)
```

---

## Error Handling

### Robust Error Handling

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustEmbedder:
    """Embedder with comprehensive error handling."""
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def embed_text_with_retry(self, text):
        """Embed text with automatic retry on failure."""
        try:
            return self.embedder.embed_text(text)
        except Exception as e:
            print(f"Embedding attempt failed: {e}")
            raise
    
    def embed_text_safe(self, text, fallback=None):
        """Embed text with fallback on error."""
        try:
            return self.embed_text_with_retry(text)
        except Exception as e:
            print(f"All retry attempts failed: {e}")
            return fallback

# Usage
robust_embedder = RobustEmbedder(embedder)
embedding = robust_embedder.embed_text_safe("Sample text", fallback=[0.0] * 1536)
```

---

## Best Practices

### Document Processing & Chunking Best Practices

1. **Use Structure-Aware Chunking for PDFs**
   - Parse PDFs with Azure AI Document Intelligence
   - Respect document structure (headers, tables, paragraphs)
   - Never split tables across chunks
   - Keep section headers with their content
   
2. **Choose Appropriate Chunk Sizes**
   - **Small chunks (256 tokens)**: Precise retrieval, more chunks to search
   - **Medium chunks (512 tokens)**: Good balance for most use cases
   - **Large chunks (1024 tokens)**: More context, fewer chunks
   
3. **Implement Overlap**
   - Use 10-20% overlap between chunks
   - Prevents information loss at boundaries
   - Helps with queries that span chunk boundaries
   
4. **Preserve Context**
   - Include parent section titles in chunks
   - Add metadata: page number, chunk type, bounding box
   - Maintain document hierarchy
   
5. **Handle Tables Appropriately**
   - Convert tables to markdown format
   - Keep entire table in single chunk when possible
   - Add table prefix: `[TABLE 1]` for clarity
   
6. **Optimize for Search**
   - Index chunks individually with metadata
   - Store document_id to reconstruct original
   - Enable filtering by page_number, chunk_type
   - Include parent_section for context

### Embedding Best Practices

1. **Cache embeddings** for frequently accessed content
2. **Batch process** when possible to reduce latency
3. **Monitor costs** with token tracking
4. **Use checkpoints** for long-running jobs
5. **Implement retry logic** for transient failures
6. **Normalize embeddings** for consistent similarity calculations

### General Don'ts

1. **Don't** embed the same text multiple times
2. **Don't** exceed rate limits (3,000 RPM for ada-002)
3. **Don't** ignore token limits (8,191 for ada-002)
4. **Don't** skip error handling
5. **Don't** forget to secure API keys
6. **Don't** use character-based chunking for PDFs with structure

### Complete PDF-to-Search Pipeline

```
1. PDF Document
   ↓
2. Azure AI Document Intelligence (parse layout)
   ↓
3. Structure-Aware Chunking (preserve tables/headers)
   ↓
4. Azure OpenAI (generate embeddings per chunk)
   ↓
5. Azure AI Search (index chunks with metadata)
   ↓
6. Vector/Hybrid Search (retrieve relevant chunks)
   ↓
7. Re-rank and present results
```

---

## Next Steps

- **[Vector Search Implementation](./09-vector-search.md)** - Use embeddings for search
- **[Hybrid Search](./10-hybrid-search.md)** - Combine text and vector search
- **[Cost Analysis](./19-cost-analysis.md)** - Optimize Azure OpenAI costs

---

*See also: [Azure AI Search Setup](./04-azure-ai-search-setup.md) | [Best Practices](./30-best-practices-checklist.md)*