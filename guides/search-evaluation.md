# Azure AI Search Evaluation Guide

<!-- Document Metadata -->
**Purpose**: Comprehensive practical guide for evaluating search indexing strategies using Azure AI Search, Azure Cognitive Services, and Azure monitoring tools  
**Target Audience**: Search engineers, data scientists, and developers implementing production search systems on Azure  
**Reading Time**: 60-75 minutes (comprehensive implementation guide)  
**Related Guides**: 
- [RAG Scenario Evaluation](./rag-scenario-evaluation.md) - RAG-specific evaluation patterns
- [docs/01-core-metrics.md](../docs/01-core-metrics.md) - Foundational metrics
- [docs/02-evaluation-frameworks.md](../docs/02-evaluation-frameworks.md) - Systematic testing approaches
- [docs/04-azure-ai-search-setup.md](../docs/04-azure-ai-search-setup.md) - Azure service configuration

---

## Why Azure-Specific Search Evaluation Matters

Generic search evaluation frameworks often miss critical Azure-specific capabilities that can dramatically improve both **search quality** and **cost efficiency**. Azure AI Search provides unique features—semantic ranking, integrated vector search, built-in AI enrichment, multi-region replication—that require specialized evaluation techniques to leverage effectively.

### The Azure Advantage: Real-World Impact

**Problem**: Many teams evaluate search systems using generic metrics and miss Azure-specific optimization opportunities that could save thousands of dollars monthly while improving search quality by 30-50%.

**Solution**: Azure-native evaluation harnesses platform capabilities—Azure Monitor for operational insights, Cognitive Services for query intent analysis, built-in semantic ranking for relevance improvement—creating a comprehensive evaluation framework unavailable in platform-agnostic approaches.

**Real-World Healthcare Search Example**:
- **Before Azure-Specific Evaluation**: 
  - Pure BM25 keyword search evaluated with generic metrics
  - 58% recall on medical terminology queries ("myocardial infarction" missed documents saying "heart attack")
  - $2,400/month Azure costs (S2 tier running 24/7 at 60% utilization)
  - Clinical staff satisfaction: 3.2/5
  - Average time to find patient records: 4.2 minutes

- **After Azure-Optimized Evaluation**:
  - Hybrid search (BM25 + vector) with semantic reranking, tuned through Azure-specific testing
  - 91% recall (+33 percentage points) using semantic matching for medical synonyms
  - $1,680/month (-30% cost) through auto-scaling during peak hours (8am-6pm) identified via Azure Monitor
  - Semantic ranking improved NDCG@10 from 0.67 → 0.84 (+25%)
  - Clinical staff satisfaction: 4.7/5 (+1.5 points)
  - Average time to find records: 1.8 minutes (-57%)
  - **Net annual impact**: $8,640 cost savings + estimated $85K productivity gains

### Three Critical Azure Evaluation Decisions

Every Azure AI Search evaluation must address:

1. **Search Mode Selection**: Which combination of keyword (BM25), vector (HNSW), hybrid (RRF fusion), and semantic ranking delivers the best quality/cost tradeoff for your query patterns?

2. **Performance vs. Cost Tradeoff**: How do you balance search latency, throughput requirements, and Azure service tier costs using real production usage patterns from Azure Monitor?

3. **Azure-Native Optimization**: Which Azure Cognitive Services integrations (Text Analytics for query intent, Form Recognizer for document extraction, Language Understanding for query expansion) improve search quality enough to justify their costs?

### What This Guide Covers

This comprehensive implementation guide provides:

- **Azure AI Search Engine Setup**: Complete implementation patterns for keyword, vector, hybrid, and semantic search using Azure SDK
- **Azure Cognitive Services Integration**: Query intent analysis, categorization, and language understanding for search improvement
- **Azure Monitor Integration**: Real-time performance metrics, latency percentiles, error analysis, and operational insights
- **Cost Optimization Strategies**: Auto-scaling recommendations, tier selection guidance, and usage pattern analysis
- **Load Testing Framework**: Concurrent user simulation, throughput analysis, and capacity planning
- **Index Optimization Techniques**: Data-driven field boosting, custom analyzer recommendations, and query pattern analysis
- **Complete Example Pipeline**: End-to-end evaluation workflow from setup through optimization and reporting

**After completing this guide, you will be able to**:
- Implement production-grade Azure AI Search evaluation pipelines with comprehensive metrics
- Leverage Azure Monitor and Cognitive Services for operational insights and query optimization
- Make data-driven decisions about search modes, service tiers, and feature adoption
- Reduce Azure search costs by 20-40% while maintaining or improving search quality
- Identify and fix performance bottlenecks using Azure-native monitoring and diagnostics

---

## Table of Contents

1. [Azure AI Search Setup](#-azure-ai-search-setup)
2. [Azure AI Search Engine Implementation](#-azure-ai-search-engine-implementation)
3. [Azure-Specific Evaluation Pipeline](#-azure-specific-evaluation-pipeline)
4. [Azure-Specific Optimization Strategies](#-azure-specific-optimization-strategies)
5. [Azure Monitoring and Alerting](#-azure-monitoring-and-alerting)
6. [Complete Azure Example](#-complete-azure-example)

---

This guide provides specific instructions for evaluating indexing strategies using Azure AI Search (formerly Azure Cognitive Search) and Azure-specific tools.

## 🔧 Azure AI Search Setup

### Understanding Azure SDK Dependencies

Azure AI Search evaluation requires several interconnected Azure SDKs. Understanding their roles prevents common setup issues and ensures you install only necessary dependencies.

**Core Dependencies Explained**:

1. **azure-search-documents (11.4.0+)**: Primary SDK for Azure AI Search operations
   - **Purpose**: Index creation, document indexing, search query execution
   - **Required for**: All search evaluation scenarios
   - **Key features**: Vector search support, semantic ranking, hybrid queries
   - **Version consideration**: 11.4.0+ required for latest vector search features (earlier versions lack VectorizedQuery support)

2. **azure-identity (1.15.0+)**: Unified authentication across Azure services
   - **Purpose**: Manages credentials for Azure SDK authentication (managed identity, Azure CLI, service principal)
   - **Required for**: Production deployments, CI/CD pipelines, multi-service integrations
   - **Alternative**: `AzureKeyCredential` for simple API key auth (development/testing only)
   - **Why use it**: Eliminates hardcoded credentials, enables zero-trust security patterns

3. **azure-ai-textanalytics (5.3.0+)**: Azure Cognitive Services for language understanding
   - **Purpose**: Query intent analysis, key phrase extraction, sentiment analysis
   - **Required for**: Advanced query categorization (navigational/informational/transactional)
   - **Optional if**: Your evaluation doesn't need query intent segmentation
   - **Cost consideration**: Charged per 1K text records (~$1-2/1K records depending on region)

4. **azure-storage-blob (12.19.0+)**: Azure Blob Storage SDK
   - **Purpose**: Store evaluation datasets, export results, manage large document corpuses
   - **Required for**: Large-scale evaluations (>10K documents), result archival, dataset versioning
   - **Optional if**: Working with small in-memory datasets (<1K documents)

### Prerequisites Installation

```bash
# Install Azure CLI and Azure AI Search SDK

# Core Azure AI Search dependencies (REQUIRED for all scenarios)
pip install azure-search-documents==11.4.0
pip install azure-identity==1.15.0

# Advanced analytics dependencies (OPTIONAL - install if using query intent analysis)
pip install azure-ai-textanalytics==5.3.0

# Storage dependencies (OPTIONAL - install if managing large datasets in Blob Storage)
pip install azure-storage-blob==12.19.0

# Additional recommended packages for production evaluation
pip install python-dotenv==1.0.0      # Environment variable management
pip install azure-monitor-query==1.3.0  # Azure Monitor integration for operational metrics
pip install openai==1.12.0             # Azure OpenAI for vector embeddings (if using vector search)
```

**Installation Verification**:

```python
# verify_installation.py
"""Verify all Azure SDK dependencies are correctly installed."""

def verify_azure_sdks():
    """Check Azure SDK installations and versions."""
    import sys
    
    required_packages = {
        'azure.search.documents': '11.4.0',
        'azure.identity': '1.15.0',
    }
    
    optional_packages = {
        'azure.ai.textanalytics': '5.3.0',
        'azure.storage.blob': '12.19.0',
        'openai': '1.12.0',
        'azure.monitor.query': '1.3.0'
    }
    
    results = []
    
    # Check required packages
    for package, min_version in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            status = '✓' if version >= min_version else f'⚠ Update needed (have {version}, need {min_version}+)'
            results.append(f"{status} {package}: {version}")
        except ImportError:
            results.append(f"✗ {package}: NOT INSTALLED (required)")
            
    # Check optional packages
    for package, min_version in optional_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            results.append(f"✓ {package}: {version} (optional)")
        except ImportError:
            results.append(f"- {package}: not installed (optional)")
    
    print("\n".join(results))
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 8):
        print(f"\n⚠ Python {python_version} detected. Python 3.8+ recommended for Azure SDKs.")
    else:
        print(f"\n✓ Python {python_version}")

if __name__ == "__main__":
    verify_azure_sdks()
```

**Common Installation Issues**:

1. **"No module named 'azure.search.documents'"**
   - **Cause**: Package not installed or wrong package name
   - **Fix**: `pip install azure-search-documents` (NOT azure-search)

2. **"VectorizedQuery not found"**
   - **Cause**: Old azure-search-documents version (<11.4.0)
   - **Fix**: `pip install --upgrade azure-search-documents>=11.4.0`

3. **SSL certificate errors on Windows**
   - **Cause**: Missing/outdated certifi package
   - **Fix**: `pip install --upgrade certifi`

4. **"DefaultAzureCredential failed to retrieve token"**
   - **Cause**: Not logged into Azure CLI or no managed identity configured
   - **Fix**: Run `az login` or use `AzureKeyCredential` for development

### Authentication Configuration

Azure AI Search supports multiple authentication methods, each suited for different deployment scenarios. Understanding when to use each method prevents security vulnerabilities and simplifies credential management.

**Authentication Method Decision Framework**:

| Method | Best For | Security Level | Complexity | Cost |
|--------|----------|----------------|------------|------|
| **AzureKeyCredential** (API keys) | Local development, quick prototypes | ⚠️ Medium | Low | Free |
| **DefaultAzureCredential** (managed identity) | Production deployments, Azure VMs/containers | ✓ High | Medium | Free |
| **AzureCliCredential** | CI/CD pipelines, developer machines | ✓ High | Low | Free |
| **Service Principal** | Cross-tenant access, automation | ✓ High | High | Free |

**Recommendation**: Use `AzureKeyCredential` for initial development, then migrate to `DefaultAzureCredential` for production (automatically uses managed identity in Azure environments, falls back to Azure CLI locally).

```python
# src/azure/auth.py
"""
Azure authentication manager supporting multiple credential types.

This module provides a unified interface for Azure AI Search authentication,
automatically selecting the best credential method for the current environment.
"""

import os
from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.core.credentials import AzureKeyCredential

class AzureAuthManager:
    """
    Manages Azure authentication across search and AI services.
    
    Supports multiple authentication methods:
    1. API Key (development): Fast, simple, less secure
    2. Managed Identity (production): Secure, automatic in Azure
    3. Azure CLI (local development): Secure, uses `az login` credentials
    
    Environment Variables Required:
    - AZURE_SEARCH_ENDPOINT: Search service URL (required)
    - AZURE_SEARCH_KEY: Admin API key (optional if using managed identity)
    - AZURE_OPENAI_ENDPOINT: OpenAI endpoint for embeddings (optional)
    - AZURE_OPENAI_KEY: OpenAI API key (optional if using managed identity)
    """
    
    def __init__(self):
        # Search service configuration
        self.search_key = os.getenv('AZURE_SEARCH_KEY')
        self.search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        
        # Validate required configuration
        if not self.search_endpoint:
            raise ValueError(
                "AZURE_SEARCH_ENDPOINT environment variable is required. "
                "Example: https://your-service.search.windows.net"
            )
        
        # OpenAI configuration (optional - only needed for vector search)
        self.openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.openai_key = os.getenv('AZURE_OPENAI_KEY')
        
    def get_search_credential(self):
        """
        Get credential for Azure AI Search.
        
        Priority order:
        1. If AZURE_SEARCH_KEY exists → Use AzureKeyCredential (simple API key auth)
        2. Otherwise → Use DefaultAzureCredential (managed identity or Azure CLI)
        
        Returns:
            AzureKeyCredential or DefaultAzureCredential: Authentication credential
            
        Example:
            >>> auth = AzureAuthManager()
            >>> credential = auth.get_search_credential()
            >>> client = SearchClient(endpoint, index_name, credential)
        """
        if self.search_key:
            # Development mode: Use API key for simplicity
            print("ℹ️ Using API key authentication (development mode)")
            return AzureKeyCredential(self.search_key)
        else:
            # Production mode: Use managed identity or Azure CLI
            # In Azure (VM/Container App/Function): Uses managed identity automatically
            # Locally: Falls back to `az login` credentials
            print("ℹ️ Using DefaultAzureCredential (managed identity or Azure CLI)")
            return DefaultAzureCredential()
    
    def get_openai_credential(self):
        """
        Get credential for Azure OpenAI (for embeddings).
        
        Only needed if evaluation includes vector search scenarios.
        
        Returns:
            AzureKeyCredential or DefaultAzureCredential: Authentication credential
            
        Raises:
            ValueError: If neither AZURE_OPENAI_KEY nor managed identity is available
        """
        if self.openai_key:
            return AzureKeyCredential(self.openai_key)
        else:
            return DefaultAzureCredential()
    
    def validate_connection(self):
        """
        Validate Azure AI Search connection and credentials.
        
        Performs a simple index list operation to verify:
        - Endpoint is reachable
        - Credentials are valid
        - Service is operational
        
        Returns:
            dict: Connection status and service information
            
        Example:
            >>> auth = AzureAuthManager()
            >>> status = auth.validate_connection()
            >>> print(status['status'])  # 'connected' or 'failed'
        """
        from azure.search.documents.indexes import SearchIndexClient
        
        try:
            credential = self.get_search_credential()
            index_client = SearchIndexClient(self.search_endpoint, credential)
            
            # List indexes to validate connection (doesn't modify anything)
            indexes = list(index_client.list_indexes())
            
            return {
                'status': 'connected',
                'endpoint': self.search_endpoint,
                'index_count': len(indexes),
                'indexes': [idx.name for idx in indexes]
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'endpoint': self.search_endpoint,
                'error': str(e),
                'troubleshooting': self._get_troubleshooting_tips(e)
            }
    
    def _get_troubleshooting_tips(self, error):
        """Generate troubleshooting tips based on error type."""
        error_msg = str(error).lower()
        
        if 'authentication' in error_msg or 'unauthorized' in error_msg:
            return [
                "Check AZURE_SEARCH_KEY is correct (must be admin key, not query key)",
                "Verify Azure CLI login: run `az login` and `az account show`",
                "For managed identity: ensure identity is assigned and has 'Search Service Contributor' role"
            ]
        elif 'not found' in error_msg or '404' in error_msg:
            return [
                "Verify AZURE_SEARCH_ENDPOINT format: https://SERVICE-NAME.search.windows.net",
                "Check service name spelling in endpoint URL",
                "Confirm search service exists: `az search service show --name SERVICE-NAME --resource-group RG-NAME`"
            ]
        elif 'timeout' in error_msg or 'connection' in error_msg:
            return [
                "Check network connectivity to Azure",
                "Verify firewall rules allow your IP address",
                "Confirm search service is not stopped or deallocated"
            ]
        else:
            return [
                "Check all environment variables are set correctly",
                "Verify Azure subscription is active",
                "Try using API key authentication first to isolate credential issues"
            ]
```

**Authentication Testing Script**:

```python
# test_auth.py
"""Test Azure AI Search authentication configuration."""

from src.azure.auth import AzureAuthManager

def test_authentication():
    """Test Azure authentication and connection."""
    print("Testing Azure AI Search Authentication\n" + "="*50)
    
    try:
        # Initialize authentication manager
        auth = AzureAuthManager()
        
        # Validate connection
        result = auth.validate_connection()
        
        if result['status'] == 'connected':
            print(f"✓ Successfully connected to {result['endpoint']}")
            print(f"✓ Found {result['index_count']} existing indexes")
            if result['indexes']:
                print(f"  Indexes: {', '.join(result['indexes'])}")
        else:
            print(f"✗ Connection failed: {result['error']}")
            print("\nTroubleshooting tips:")
            for tip in result['troubleshooting']:
                print(f"  • {tip}")
                
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("\nMake sure these environment variables are set:")
        print("  • AZURE_SEARCH_ENDPOINT (required)")
        print("  • AZURE_SEARCH_KEY (optional if using managed identity)")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

if __name__ == "__main__":
    test_authentication()
```

**Common Authentication Errors and Solutions**:

1. **"(Unauthorized) Access denied due to invalid credentials"**
   - **Cause**: Wrong API key or expired managed identity token
   - **Fix**: 
     - Verify key: Azure Portal → Search Service → Keys → Copy admin key
     - For managed identity: Ensure `Search Service Contributor` role assigned
     - Test with: `az rest --method get --url "https://YOUR-SERVICE.search.windows.net/indexes?api-version=2023-11-01" --resource https://search.azure.com`

2. **"DefaultAzureCredential failed to retrieve a token"**
   - **Cause**: No authentication method available (not logged into Azure CLI, no managed identity)
   - **Fix**:
     - Local development: Run `az login`
     - Azure VM/Container: Enable system-assigned managed identity
     - Fallback: Set `AZURE_SEARCH_KEY` environment variable

3. **"The specified resource name is invalid"**
   - **Cause**: Malformed search endpoint URL
   - **Fix**: Ensure format is exactly `https://SERVICE-NAME.search.windows.net` (no trailing slash, must use `.search.windows.net` domain)

4. **"Forbidden" (403 error)**
   - **Cause**: Using query key instead of admin key, or insufficient RBAC permissions
   - **Fix**: 
     - For API key auth: Must use admin key (query keys can't create indexes)
     - For RBAC: Assign `Search Service Contributor` (not just `Reader`)

### Environment Configuration

**Development Environment Setup** (.env file for local testing):

```bash
# .env file
# Azure AI Search Configuration (REQUIRED)
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your-search-admin-key-here  # Get from Azure Portal → Search Service → Keys

# Azure OpenAI Configuration (OPTIONAL - only if using vector search)
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com
AZURE_OPENAI_KEY=your-openai-key-here  # Get from Azure Portal → OpenAI Service → Keys
AZURE_OPENAI_API_VERSION=2024-02-01  # Use latest stable version
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002  # Your embedding model deployment name

# Azure Monitor Configuration (OPTIONAL - for operational metrics)
AZURE_LOG_ANALYTICS_WORKSPACE_ID=your-workspace-id  # Get from Portal → Log Analytics → Workspace ID
AZURE_SUBSCRIPTION_ID=your-subscription-id  # Run: az account show --query id -o tsv

# Azure Cognitive Services (OPTIONAL - for query intent analysis)
AZURE_TEXT_ANALYTICS_ENDPOINT=https://your-text-analytics.cognitiveservices.azure.com
AZURE_TEXT_ANALYTICS_KEY=your-text-analytics-key

# Evaluation Configuration
EVALUATION_RESULTS_CONTAINER=evaluation-results  # Blob container for storing results
AZURE_STORAGE_CONNECTION_STRING=your-storage-connection-string  # Optional: for result storage
```

**Production Environment Setup** (Azure Container Apps, Functions, VMs):

```bash
# Production .env (uses managed identity - no keys needed)
AZURE_SEARCH_ENDPOINT=https://prod-search-service.search.windows.net
# AZURE_SEARCH_KEY not set - will use managed identity

AZURE_OPENAI_ENDPOINT=https://prod-openai.openai.azure.com
# AZURE_OPENAI_KEY not set - will use managed identity

AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Monitor configuration
AZURE_LOG_ANALYTICS_WORKSPACE_ID=workspace-id-from-portal
AZURE_SUBSCRIPTION_ID=subscription-id-from-cli
```

**Environment Variable Management Best Practices**:

1. **Never commit credentials to Git**:
   ```bash
   # .gitignore
   .env
   .env.local
   *.key
   secrets/
   ```

2. **Use Azure Key Vault for production secrets**:
   ```python
   from azure.identity import DefaultAzureCredential
   from azure.keyvault.secrets import SecretClient
   
   def load_secrets_from_keyvault(vault_url):
       """Load secrets from Azure Key Vault (production)."""
       credential = DefaultAzureCredential()
       client = SecretClient(vault_url=vault_url, credential=credential)
       
       os.environ['AZURE_SEARCH_KEY'] = client.get_secret('search-admin-key').value
       os.environ['AZURE_OPENAI_KEY'] = client.get_secret('openai-key').value
   ```

3. **Validate all required variables on startup**:
   ```python
   def validate_environment():
       """Ensure all required environment variables are set."""
       required = ['AZURE_SEARCH_ENDPOINT']
       optional_vector = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_KEY']
       
       missing = [var for var in required if not os.getenv(var)]
       if missing:
           raise ValueError(f"Missing required environment variables: {missing}")
       
       # Warn if vector search variables missing
       if any(os.getenv(var) for var in optional_vector) and not all(os.getenv(var) for var in optional_vector):
           print("⚠️ Warning: Partial OpenAI configuration. Vector search may not work.")
   ```

## 🔍 Azure AI Search Engine Implementation

### Understanding Search Mode Selection

Azure AI Search offers four distinct search modes, each optimized for different query patterns and use cases. Choosing the wrong mode can result in 40-60% lower recall for certain query types.

**Search Mode Comparison**:

| Mode | Algorithm | Best For | Typical Recall | Latency | Cost |
|------|-----------|----------|----------------|---------|------|
| **Keyword** | BM25 | Exact term matching, IDs, SKUs | 45-65% | 20-50ms | Lowest |
| **Vector** | HNSW | Semantic similarity, synonyms | 70-85% | 80-150ms | Medium (+embedding costs) |
| **Hybrid** | BM25 + HNSW + RRF | General purpose, mixed queries | 80-92% | 100-180ms | Medium-High |
| **Semantic** | Hybrid + L2 reranker | Natural language questions | 85-95% | 150-250ms | Highest (+semantic SKU) |

**Decision Framework**:

1. **Use Keyword (BM25) if**:
   - Queries are mostly exact matches (product codes, IDs, technical terms)
   - Budget is constrained (no embedding costs, works on basic tier)
   - Latency must be <50ms
   - Example: "Find SKU ABC-123", "documents by author John Smith"

2. **Use Vector if**:
   - Queries use different vocabulary than documents (synonyms, paraphrasing)
   - Multilingual search across languages
   - Embedding infrastructure already exists
   - Example: "heart attack" should find "myocardial infarction"

3. **Use Hybrid if**:
   - Mixed query types (some exact, some semantic)
   - General-purpose search (80% solution for most queries)
   - Want best recall without semantic reranking costs
   - Example: Product search with both SKUs and descriptions

4. **Use Semantic if**:
   - Queries are natural language questions
   - Budget allows S1+ tier + semantic SKU ($250-500/month)
   - Need best possible relevance (90%+ recall required)
   - Example: "What's the best budget laptop for students under $500?"

### Basic Search Engine

```python
# src/engines/azure_search_engine.py
"""
Azure AI Search engine with support for keyword, vector, hybrid, and semantic search.

This module provides a production-ready search engine implementation that:
- Automatically handles index creation with proper field configurations
- Supports all Azure AI Search modes (keyword, vector, hybrid, semantic)
- Generates embeddings on-demand using Azure OpenAI
- Batch uploads documents efficiently
- Provides comprehensive error handling and validation
"""

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.search.documents.models import VectorizedQuery, QueryType, QueryCaptionType, QueryAnswerType
from ..azure.auth import AzureAuthManager
from .base_engine import BaseSearchEngine, SearchResult
import os

class AzureSearchEngine(BaseSearchEngine):
    """
    Azure AI Search engine supporting multiple search modes.
    
    Features:
    - Automatic index schema creation (text + vector fields)
    - HNSW vector search configuration (cosine similarity)
    - Semantic search configuration for L2 reranking
    - Batch document upload (100 docs per batch for efficiency)
    - Azure OpenAI integration for embedding generation
    
    Configuration Options:
    - index_name: Name of search index (default: 'search-index')
    - enable_vector_search: Enable vector field and HNSW index (default: False)
    - vector_dimensions: Embedding dimensions (default: 1536 for ada-002)
    - hnsw_m: HNSW M parameter (default: 4, higher = better recall but slower)
    - hnsw_ef_construction: Build-time accuracy (default: 400)
    - hnsw_ef_search: Query-time accuracy (default: 500)
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize authentication
        self.auth_manager = AzureAuthManager()
        self.endpoint = self.auth_manager.search_endpoint
        self.credential = self.auth_manager.get_search_credential()
        
        # Configure index clients
        self.index_name = config.get('index_name', 'search-index')
        self.index_client = SearchIndexClient(self.endpoint, self.credential)
        self.search_client = SearchClient(self.endpoint, self.index_name, self.credential)
        
        # Initialize embedding model if using vector search
        self.embedding_model = None
        if config.get('enable_vector_search', False):
            self.embedding_model = self._init_embedding_model(config)
            print(f"✓ Vector search enabled with {config.get('vector_dimensions', 1536)}-dimensional embeddings")
    
    def _init_embedding_model(self, config):
        """
        Initialize Azure OpenAI embedding model.
        
        Requires environment variables:
        - AZURE_OPENAI_ENDPOINT: OpenAI service endpoint
        - AZURE_OPENAI_KEY: API key (or use managed identity)
        - AZURE_OPENAI_API_VERSION: API version (e.g., '2024-02-01')
        
        Returns:
            AzureOpenAI: Configured OpenAI client for embeddings
            
        Raises:
            ValueError: If required environment variables are missing
        """
        from openai import AzureOpenAI
        
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        key = os.getenv('AZURE_OPENAI_KEY')
        api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
        
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT required for vector search")
        
        # Use API key if available, otherwise use managed identity via auth manager
        if key:
            return AzureOpenAI(
                api_key=key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
        else:
            # Use managed identity
            credential = self.auth_manager.get_openai_credential()
            return AzureOpenAI(
                azure_ad_token_provider=credential,
                api_version=api_version,
                azure_endpoint=endpoint
            )
    
    def create_index(self):
        """
        Create Azure AI Search index with optimized schema.
        
        Index Schema:
        - id (key): Unique document identifier
        - content (searchable): Main document text with English analyzer
        - title (searchable): Document title with English analyzer
        - category (filterable, facetable): Document category for filtering/faceting
        - metadata (simple): JSON metadata string
        - content_vector (vector, optional): 1536-dim embeddings for semantic search
        
        Vector Search Configuration (if enabled):
        - Algorithm: HNSW (Hierarchical Navigable Small World)
        - Metric: Cosine similarity
        - M=4, efConstruction=400, efSearch=500 (balanced quality/speed)
        
        Semantic Search Configuration:
        - Title field prioritized for semantic ranking
        - Content field used for semantic captions and answers
        
        Note: Deletes existing index if it exists (idempotent operation).
        """
        
        # Define base fields (always included)
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True  # Primary key field
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft"  # English language analyzer (stemming, stopwords)
            ),
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft"
            ),
            SimpleField(
                name="category",
                type=SearchFieldDataType.String,
                filterable=True,  # Enable $filter queries
                facetable=True    # Enable faceted navigation
            ),
            SimpleField(
                name="metadata",
                type=SearchFieldDataType.String
            ),
        ]
        
        # Add vector field if vector search is enabled
        vector_search_config = None
        if self.config.get('enable_vector_search', False):
            vector_dims = self.config.get('vector_dimensions', 1536)  # Default: ada-002 dimensions
            
            vector_field = VectorSearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_dims,
                vector_search_profile_name="my-vector-config"
            )
            fields.append(vector_field)
            
            # Configure HNSW algorithm for vector search
            # M: Number of bi-directional links (higher = better recall, more memory)
            # efConstruction: Build-time accuracy (higher = better index quality, slower builds)
            # efSearch: Query-time accuracy (higher = better recall, slower queries)
            vector_search_config = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="my-hnsw-config",
                        parameters=HnswParameters(
                            m=self.config.get('hnsw_m', 4),
                            ef_construction=self.config.get('hnsw_ef_construction', 400),
                            ef_search=self.config.get('hnsw_ef_search', 500),
                            metric=VectorSearchAlgorithmMetric.COSINE  # Cosine similarity for embeddings
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
            
            print(f"ℹ️ Vector search configured: {vector_dims}D HNSW (M={self.config.get('hnsw_m', 4)})")
        
        # Semantic search configuration (always included for potential use)
        # Defines which fields are used for L2 semantic reranking
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),      # Highest priority for relevance
                content_fields=[SemanticField(field_name="content")]  # Main content for semantic matching
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # Create index definition
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search_config,
            semantic_search=semantic_search
        )
        
        # Delete existing index if present (idempotent operation)
        try:
            self.index_client.delete_index(self.index_name)
            print(f"ℹ️ Deleted existing index: {self.index_name}")
        except Exception:
            pass  # Index doesn't exist, which is fine
        
        # Create new index
        self.index_client.create_index(index)
        print(f"✓ Created index: {self.index_name}")
        
        # Log index configuration summary
        field_count = len(fields)
        searchable_fields = [f.name for f in fields if isinstance(f, SearchableField)]
        print(f"  Fields: {field_count} ({len(searchable_fields)} searchable)")
        print(f"  Vector search: {'enabled' if vector_search_config else 'disabled'}")
        print(f"  Semantic search: enabled")
    
    def index_documents(self, documents):
        """
        Index documents in Azure AI Search with batch upload.
        
        Process:
        1. Create index (if not exists)
        2. Transform documents to search schema format
        3. Generate embeddings (if vector search enabled)
        4. Upload in batches of 100 documents
        
        Args:
            documents (list): List of dicts with keys:
                - id (str): Unique document identifier
                - content (str): Main document text
                - title (str, optional): Document title
                - metadata (dict, optional): Additional metadata
        
        Performance Notes:
        - Batch size of 100 balances throughput and error recovery
        - Embedding generation is the slowest step (80-150ms per document)
        - 10K documents ≈ 15-25 minutes with vector embeddings
        
        Example:
            >>> docs = [
            ...     {"id": "1", "content": "Azure AI Search guide", "title": "Search Guide"},
            ...     {"id": "2", "content": "Vector search tutorial", "title": "Vectors"}
            ... ]
            >>> engine.index_documents(docs)
            ✓ Created index: my-index
            ℹ️ Generating embeddings for 2 documents...
            ✓ Uploaded batch 1, 2 documents
        """
        
        # Create index first
        self.create_index()
        
        # Prepare documents for indexing
        search_documents = []
        
        print(f"ℹ️ Preparing {len(documents)} documents for indexing...")
        
        # Generate embeddings if vector search enabled (most expensive operation)
        if self.config.get('enable_vector_search', False):
            print(f"ℹ️ Generating embeddings for {len(documents)} documents...")
        
        for idx, doc in enumerate(documents):
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
                
                # Progress indicator for large datasets
                if (idx + 1) % 100 == 0:
                    print(f"  Generated embeddings: {idx + 1}/{len(documents)}")
            
            search_documents.append(search_doc)
        
        # Upload documents in batches of 100 (Azure AI Search best practice)
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(search_documents), batch_size):
            batch = search_documents[i:i + batch_size]
            
            try:
                result = self.search_client.upload_documents(documents=batch)
                total_uploaded += len(batch)
                print(f"✓ Uploaded batch {i//batch_size + 1}, {len(batch)} documents (total: {total_uploaded}/{len(search_documents)})")
                
                # Check for upload errors
                failed_docs = [r for r in result if not r.succeeded]
                if failed_docs:
                    print(f"  ⚠️ Warning: {len(failed_docs)} documents failed to index")
                    for failed in failed_docs[:5]:  # Show first 5 failures
                        print(f"    - Document {failed.key}: {failed.error_message}")
                        
            except Exception as e:
                print(f"✗ Batch {i//batch_size + 1} upload failed: {e}")
                # Continue with next batch instead of failing entirely
        
        self.is_indexed = True
        print(f"✓ Indexing complete: {total_uploaded}/{len(documents)} documents uploaded successfully")
    
    def _generate_embedding(self, text):
        """
        Generate embedding vector using Azure OpenAI.
        
        Args:
            text (str): Text to embed (max 8191 tokens for ada-002)
        
        Returns:
            list: 1536-dimensional embedding vector (for ada-002)
            
        Raises:
            Exception: If embedding generation fails (rate limits, authentication, etc.)
            
        Performance Notes:
        - Latency: 80-150ms per request
        - Rate limits: 
          - text-embedding-ada-002: ~1,440 RPM (requests per minute) on standard deployment
          - text-embedding-3-small: ~5,000 RPM
        - Cost:
          - ada-002: $0.0001/1K tokens (~$0.10 per 1M tokens)
          - 3-small: $0.00002/1K tokens (5x cheaper)
        
        Optimization Tips:
        - Batch multiple texts in single API call (up to 16 texts per request)
        - Cache embeddings to avoid regenerating for duplicate content
        - Use text-embedding-3-small for 5x cost savings with minimal quality loss
        """
        deployment_name = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')
        
        try:
            response = self.embedding_model.embeddings.create(
                input=text,
                model=deployment_name  # Actually deployment name, not model name
            )
            return response.data[0].embedding
            
        except Exception as e:
            # Handle common errors
            error_msg = str(e).lower()
            
            if 'rate limit' in error_msg:
                raise Exception(
                    f"Azure OpenAI rate limit exceeded. "
                    f"Consider: (1) Increase TPM quota, (2) Add retry with backoff, "
                    f"(3) Use batch embedding API"
                ) from e
            elif 'token limit' in error_msg:
                raise Exception(
                    f"Text exceeds token limit (8191 for ada-002). "
                    f"Truncate or chunk text before embedding."
                ) from e
            elif 'deployment' in error_msg or 'not found' in error_msg:
                raise Exception(
                    f"Deployment '{deployment_name}' not found. "
                    f"Check AZURE_OPENAI_EMBEDDING_DEPLOYMENT environment variable."
                ) from e
            else:
                raise  # Re-raise unexpected errors
    
    def search(self, query, top_k=10, search_mode="hybrid"):
        """
        Search documents using different Azure AI Search modes.
        
        Supports four search modes, each with different performance characteristics:
        
        1. **keyword**: Pure BM25 text matching (fastest, works on all tiers)
        2. **vector**: Pure semantic vector similarity (requires vector index)
        3. **hybrid**: Combined keyword + vector with RRF fusion (best recall)
        4. **semantic**: Hybrid + L2 semantic reranking (best relevance, S1+ only)
        
        Args:
            query (str): Search query text
            top_k (int): Number of results to return (default: 10)
            search_mode (str): Search mode - 'keyword', 'vector', 'hybrid', or 'semantic'
        
        Returns:
            list[SearchResult]: Ranked search results with scores and metadata
            
        Raises:
            ValueError: If documents not indexed or invalid search mode
            
        Performance Comparison (1M document index, typical query):
        - keyword: 20-50ms, 55-65% recall
        - vector: 80-150ms, 70-85% recall  
        - hybrid: 100-180ms, 80-92% recall
        - semantic: 150-250ms, 85-95% recall
        
        Cost Comparison (per 1K queries):
        - keyword: $0 (included in base tier)
        - vector: ~$0.10 (embedding costs only)
        - hybrid: ~$0.10 (embedding costs only)
        - semantic: ~$2.50 (1K semantic queries on S1 = 1,000 * $2.50/1K)
        
        Example:
            >>> results = engine.search("azure ai search tutorial", top_k=5, search_mode="hybrid")
            >>> print(f"Found {len(results)} results")
            >>> print(f"Top result: {results[0].content[:100]}")
        """
        
        if not self.is_indexed:
            raise ValueError(
                "Documents must be indexed before searching. "
                "Call index_documents() first."
            )
        
        # Validate search mode
        valid_modes = ["keyword", "vector", "hybrid", "semantic"]
        if search_mode not in valid_modes:
            raise ValueError(
                f"Unknown search mode: '{search_mode}'. "
                f"Must be one of: {', '.join(valid_modes)}"
            )
        
        # Route to appropriate search method
        search_results = []
        
        if search_mode == "keyword":
            search_results = self._keyword_search(query, top_k)
        elif search_mode == "vector":
            search_results = self._vector_search(query, top_k)
        elif search_mode == "hybrid":
            search_results = self._hybrid_search(query, top_k)
        elif search_mode == "semantic":
            search_results = self._semantic_search(query, top_k)
        
        return search_results
    
    def _keyword_search(self, query, top_k):
        """
        Pure keyword search using BM25 algorithm.
        
        Best for:
        - Exact term matching (product IDs, SKUs, names)
        - When latency must be <50ms
        - Basic tier deployments (no vector support)
        
        Limitations:
        - Misses semantic matches ("heart attack" won't find "myocardial infarction")
        - Sensitive to exact vocabulary matching
        - No understanding of synonyms or paraphrasing
        
        Performance:
        - Latency: 20-50ms (fastest option)
        - Recall: 45-65% (depends on query/document vocabulary overlap)
        
        Returns:
            list[SearchResult]: Results sorted by BM25 score (higher = better match)
        """
        results = self.search_client.search(
            search_text=query,
            top=top_k,
            include_total_count=True  # Get total matching documents (useful for pagination)
        )
        
        search_results = []
        for result in results:
            search_result = SearchResult(
                doc_id=result["id"],
                content=result["content"],
                score=result["@search.score"],  # BM25 score (typically 0-20 range)
                metadata={"title": result.get("title", "")}
            )
            search_results.append(search_result)
        
        return search_results
    
    def _vector_search(self, query, top_k):
        """
        Pure vector search using HNSW approximate nearest neighbor.
        
        Best for:
        - Semantic similarity ("cheap laptop" finds "affordable computer")
        - Cross-lingual search (query in English, find Spanish documents)
        - Paraphrasing and synonym matching
        
        Limitations:
        - Misses exact matches if embeddings don't capture them well
        - Requires embedding generation for every query (+80-150ms latency)
        - Costs $0.0001/1K tokens for embeddings (ada-002)
        
        Performance:
        - Latency: 80-150ms (slower due to embedding generation)
        - Recall: 70-85% (better for semantic queries)
        - Precision: Can retrieve false positives with similar embeddings but different meaning
        
        Returns:
            list[SearchResult]: Results sorted by cosine similarity (0-1, higher = more similar)
        """
        if not self.config.get('enable_vector_search', False):
            raise ValueError(
                "Vector search not enabled in configuration. "
                "Set 'enable_vector_search': True and ensure AZURE_OPENAI_* env vars are set."
            )
        
        # Generate query embedding (most expensive step: 80-150ms)
        query_vector = self._generate_embedding(query)
        
        # Create vectorized query for HNSW search
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,  # Number of nearest neighbors to retrieve
            fields="content_vector"  # Vector field to search
        )
        
        results = self.search_client.search(
            search_text=None,  # No text search, pure vector
            vector_queries=[vector_query],
            top=top_k
        )
        
        search_results = []
        for result in results:
            search_result = SearchResult(
                doc_id=result["id"],
                content=result["content"],
                score=result["@search.score"],  # Cosine similarity score (0-1 range)
                metadata={"title": result.get("title", "")}
            )
            search_results.append(search_result)
        
        return search_results
    
    def _hybrid_search(self, query, top_k):
        """
        Hybrid search combining BM25 keyword + HNSW vector with RRF fusion.
        
        Azure AI Search uses Reciprocal Rank Fusion (RRF) to combine:
        - BM25 keyword scores (exact term matching)
        - Vector cosine similarity (semantic matching)
        
        RRF Formula: score = Σ(1 / (k + rank_i)) for each ranking source
        - k = 60 (Azure's default RRF constant)
        - Results ranked independently by keyword and vector, then fused
        
        Best for:
        - General-purpose search (handles both exact and semantic queries well)
        - Mixed query types (some users search IDs, others use natural language)
        - Maximum recall without semantic ranking costs
        
        Performance:
        - Latency: 100-180ms (keyword + vector + fusion)
        - Recall: 80-92% (best of both worlds)
        - Cost: Same as vector-only (embedding generation is the only added cost)
        
        Returns:
            list[SearchResult]: Results sorted by RRF score (fused keyword + vector ranking)
        """
        if not self.config.get('enable_vector_search', False):
            # Graceful fallback: If vector search not configured, use keyword only
            print("⚠️ Vector search not enabled, falling back to keyword search")
            return self._keyword_search(query, top_k)
        
        # Generate query embedding for vector component
        query_vector = self._generate_embedding(query)
        
        # Create vectorized query
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )
        
        # Execute hybrid search (both keyword and vector simultaneously)
        results = self.search_client.search(
            search_text=query,  # Keyword component (BM25)
            vector_queries=[vector_query],  # Vector component (HNSW)
            top=top_k
        )
        
        search_results = []
        for result in results:
            search_result = SearchResult(
                doc_id=result["id"],
                content=result["content"],
                score=result["@search.score"],  # RRF fused score
                metadata={"title": result.get("title", "")}
            )
            search_results.append(search_result)
        
        return search_results
    
    def _semantic_search(self, query, top_k):
        """
        Semantic search using hybrid retrieval + Azure L2 semantic reranking.
        
        Process:
        1. Hybrid search retrieves top 50 candidates (keyword + vector)
        2. Azure's L2 semantic ranker reranks top 50 using deep learning model
        3. Returns top_k results with improved relevance
        
        L2 Semantic Ranker:
        - Microsoft's proprietary deep learning model (BERT-based)
        - Trained on Bing search query/document pairs
        - Excels at natural language question answering
        - Also generates captions (highlighted relevant excerpts)
        
        Best for:
        - Natural language questions ("what's the best budget laptop?")
        - Long-form queries (>5 words)
        - When maximum relevance is critical (customer-facing search)
        
        Limitations:
        - Requires S1 or higher tier (not available on Basic/S0)
        - Costs $2.50/1K queries (Standard pricing tier)
        - Slower than hybrid (adds 50-70ms for reranking)
        
        Performance:
        - Latency: 150-250ms (hybrid retrieval + reranking)
        - Recall: 85-95% (highest quality option)
        - NDCG improvement: +15-25% over hybrid on question-answering tasks
        
        Cost Example:
        - 100K queries/month with semantic search
        - Embedding costs: $10 (100K queries * $0.0001/1K tokens)
        - Semantic ranking: $250 (100K queries * $2.50/1K)
        - Total: $260/month (vs $10 for hybrid without semantic)
        
        Returns:
            list[SearchResult]: Results sorted by L2 semantic score with extracted captions
        """
        results = self.search_client.search(
            search_text=query,
            top=top_k,
            query_type=QueryType.SEMANTIC,  # Enable L2 semantic reranking
            semantic_configuration_name="my-semantic-config",  # Semantic field configuration
            query_caption=QueryCaptionType.EXTRACTIVE,  # Extract relevant excerpts
            query_answer=QueryAnswerType.EXTRACTIVE  # Extract direct answers (Q&A scenarios)
        )
        
        search_results = []
        for result in results:
            # Extract semantic captions (highlighted relevant text excerpts)
            captions = []
            if "@search.captions" in result:
                captions = [caption.text for caption in result["@search.captions"]]
            
            # Extract semantic answers (direct answers to questions)
            answers = []
            if "@search.answers" in result:
                answers = [answer.text for answer in result["@search.answers"]]
            
            search_result = SearchResult(
                doc_id=result["id"],
                content=result["content"],
                score=result["@search.score"],  # Hybrid RRF score (for initial retrieval)
                metadata={
                    "title": result.get("title", ""),
                    "captions": captions,  # Highlighted relevant excerpts
                    "answers": answers,  # Direct answer extractions
                    "reranker_score": result.get("@search.reranker_score")  # L2 semantic score (0-4 range)
                }
            )
            search_results.append(search_result)
        
        return search_results
    
    def get_stats(self):
        """
        Get index statistics from Azure AI Search.
        
        Returns:
            dict: Index statistics including:
                - document_count: Total documents in index
                - storage_size_mb: Total storage consumed (text + metadata)
                - vector_index_size_mb: Additional storage for vector index
                
        Example:
            >>> stats = engine.get_stats()
            >>> print(f"Documents: {stats['document_count']:,}")
            >>> print(f"Storage: {stats['storage_size_mb']:.2f} MB")
            >>> print(f"Vector index: {stats['vector_index_size_mb']:.2f} MB")
        """
        try:
            stats = self.index_client.get_index_statistics(self.index_name)
            
            return {
                "document_count": stats.document_count,
                "storage_size_mb": stats.storage_size / (1024 * 1024),
                "vector_index_size_mb": getattr(stats, 'vector_index_size', 0) / (1024 * 1024)
            }
        except Exception as e:
            return {
                "error": str(e),
                "document_count": 0,
                "storage_size_mb": 0,
                "vector_index_size_mb": 0
            }
```

---

## 📊 Azure-Specific Evaluation Pipeline

### Understanding Query Intent Analysis

Azure Cognitive Services Text Analytics provides powerful query categorization capabilities that enable **intent-based evaluation**—measuring search quality separately for navigational, informational, and transactional queries, which often have very different success criteria.

**Why Intent-Based Evaluation Matters**:

- **Navigational queries** ("Azure portal login"): Success = finding exact page (precision matters most)
- **Informational queries** ("how to configure Azure search"): Success = comprehensive results (recall matters most)
- **Transactional queries** ("buy Azure credits"): Success = conversion path (click-through and conversion matter)

**Real-World Impact**: E-commerce company found that treating all queries equally masked problems:
- Overall precision@5: 0.68 (acceptable)
- But transactional query precision@5: 0.42 (terrible—users couldn't find products to buy)
- After intent-based optimization: Transactional precision improved to 0.79, sales +18%

### Azure Cognitive Services Integration

```python
# src/azure/cognitive_services.py
"""
Azure Cognitive Services integration for advanced query analysis.

This module provides query intent categorization using Azure Text Analytics,
enabling evaluation segmentation by query type (navigational/informational/transactional).

Cost Considerations:
- Text Analytics pricing: ~$1-2 per 1K text records (depends on region/tier)
- Key phrase extraction: Included in text records pricing
- Sentiment analysis: Included in text records pricing
- For evaluation, cost is one-time per query set (not per search execution)

Example: 10K query evaluation dataset = $10-20 for intent analysis
"""

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os

class AzureCognitiveServices:
    """
    Azure Cognitive Services client for query intent analysis.
    
    Capabilities:
    1. Key phrase extraction: Identify main topics in queries
    2. Sentiment analysis: Detect query urgency/frustration
    3. Query categorization: Classify as navigational/informational/transactional
    
    Environment Variables:
    - AZURE_TEXT_ANALYTICS_ENDPOINT: Text Analytics endpoint URL
    - AZURE_TEXT_ANALYTICS_KEY: API key for authentication
    """
    
    def __init__(self):
        endpoint = os.getenv('AZURE_TEXT_ANALYTICS_ENDPOINT')
        key = os.getenv('AZURE_TEXT_ANALYTICS_KEY')
        
        if not endpoint or not key:
            raise ValueError(
                "Azure Text Analytics not configured. Set these environment variables:\n"
                "  AZURE_TEXT_ANALYTICS_ENDPOINT=https://YOUR-RESOURCE.cognitiveservices.azure.com\n"
                "  AZURE_TEXT_ANALYTICS_KEY=your-key-here"
            )
        
        self.text_client = TextAnalyticsClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key)
        )
    
    def analyze_query_intent(self, queries):
        """
        Analyze query intent using Azure Text Analytics.
        
        Extracts:
        - Key phrases: Main topics/entities in query
        - Sentiment: Positive/neutral/negative (helps identify frustrated users)
        - Confidence scores: How certain the analysis is
        
        Args:
            queries (list[str]): List of query strings to analyze
        
        Returns:
            list[dict]: Analysis results with keys:
                - query: Original query text
                - key_phrases: List of extracted key phrases
                - sentiment: 'positive', 'neutral', or 'negative'
                - confidence: 0-1 confidence score
        
        Example:
            >>> cs = AzureCognitiveServices()
            >>> queries = ["how to create azure search index", "buy azure subscription"]
            >>> analysis = cs.analyze_query_intent(queries)
            >>> print(analysis[0]['key_phrases'])
            ['azure search index', 'create']
        """
        
        # Batch extract key phrases (identifies query topics)
        key_phrases_result = self.text_client.extract_key_phrases(queries)
        
        # Batch analyze sentiment (identifies frustrated/urgent queries)
        sentiment_result = self.text_client.analyze_sentiment(queries)
        
        analysis = []
        for i, query in enumerate(queries):
            # Handle API errors gracefully
            key_phrases = []
            if not key_phrases_result[i].is_error:
                key_phrases = key_phrases_result[i].key_phrases
            
            sentiment = "neutral"
            confidence = 0.5
            if not sentiment_result[i].is_error:
                sentiment = sentiment_result[i].sentiment
                confidence = sentiment_result[i].confidence_scores.positive
            
            analysis.append({
                "query": query,
                "key_phrases": key_phrases,
                "sentiment": sentiment,
                "confidence": confidence
            })
        
        return analysis
    
    def categorize_queries(self, queries):
        """
        Categorize queries by intent type using heuristic rules.
        
        Intent Categories:
        1. **Navigational**: Finding specific page/item
           - Indicators: Brand names, product IDs, "login", "portal"
           - Example: "Azure portal login", "product SKU-12345"
        
        2. **Informational**: Learning/research  
           - Indicators: Question words (how/what/why), "tutorial", "guide"
           - Example: "how to configure azure search", "what is vector search"
        
        3. **Transactional**: Intent to act/purchase
           - Indicators: "buy", "purchase", "download", "subscribe"
           - Example: "buy azure subscription", "download azure cli"
        
        Args:
            queries (list[str]): List of query strings
        
        Returns:
            dict: Categorized queries with keys:
                - navigational: List of navigational query analysis results
                - informational: List of informational query analysis results
                - transactional: List of transactional query analysis results
        
        Accuracy Note:
        - Heuristic-based categorization is 70-80% accurate
        - For production, consider training a custom classifier on your query logs
        - Azure Custom Text Classification can reach 90%+ accuracy with labeled data
        
        Example:
            >>> categories = cs.categorize_queries([
            ...     "how to create index",  # informational
            ...     "buy azure subscription",  # transactional
            ...     "azure portal"  # navigational
            ... ])
            >>> len(categories['informational'])  # 1
            >>> len(categories['transactional'])  # 1
        """
        # Analyze all queries first
        analysis = self.analyze_query_intent(queries)
        
        categories = {
            "navigational": [],
            "informational": [],
            "transactional": []
        }
        
        for item in analysis:
            # Convert key phrases to lowercase for matching
            key_phrases = [phrase.lower() for phrase in item["key_phrases"]]
            query_lower = item["query"].lower()
            
            # Transactional indicators (high priority - check first)
            transactional_terms = ["buy", "purchase", "price", "cost", "order", 
                                   "subscribe", "download", "install", "get"]
            if any(term in query_lower for term in transactional_terms):
                categories["transactional"].append(item)
                continue
            
            # Informational indicators (question patterns)
            informational_terms = ["how", "what", "why", "when", "where", "who",
                                    "tutorial", "guide", "learn", "explain", "understand"]
            if any(term in query_lower for term in informational_terms):
                categories["informational"].append(item)
                continue
            
            # Navigational indicators (specific item/page seeking)
            navigational_terms = ["login", "portal", "dashboard", "home", "main",
                                   "official", "website"]
            # Also check if query looks like an ID/SKU (contains numbers and short)
            has_id_pattern = any(char.isdigit() for char in item["query"]) and len(item["query"].split()) <= 3
            
            if any(term in query_lower for term in navigational_terms) or has_id_pattern:
                categories["navigational"].append(item)
                continue
            
            # Default: Treat as navigational if no clear indicators
            # (Conservative approach - navigational queries are most common)
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