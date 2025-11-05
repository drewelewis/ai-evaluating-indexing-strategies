# Azure AI Search Indexing Strategy Evaluation Knowledge Base

This repository contains comprehensive documentation and frameworks for measuring search relevancy and accuracy when experimenting with different indexing strategies using **Azure AI Search** and the broader Azure ecosystem.

## 🔷 Azure-Focused Architecture

This knowledge base is specifically designed around Azure services:
- **Azure AI Search** - Primary search service with multiple indexing modes
- **Azure OpenAI** - Embeddings and semantic capabilities  
- **Azure Cosmos DB** - Storing evaluation results and metrics
- **Azure Monitor & Application Insights** - Performance monitoring
- **Azure Load Testing** - Scalability evaluation
- **Power BI** - Analytics and reporting dashboards

## 📚 Knowledge Base Structure

### Core Documentation
- **[Core Metrics](./docs/01-core-metrics.md)** - Essential metrics for measuring search relevance and accuracy
- **[Evaluation Frameworks](./docs/02-evaluation-frameworks.md)** - Azure-native approaches to testing search performance
- **[Azure AI Search Strategies](./docs/03-azure-search-strategies.md)** - Comprehensive guide to Azure AI Search indexing modes
- **[Advanced Azure Techniques](./docs/04-azure-advanced-techniques.md)** - Semantic ranking, Azure OpenAI integration, and optimization

### Azure Implementation Guides
- **[Azure Setup Guide](./guides/azure-setup-guide.md)** - Complete Azure environment configuration
- **[Azure AI Search Evaluation](./guides/azure-ai-search-evaluation.md)** - Azure-specific evaluation pipelines
- **[Azure Cost Optimization](./guides/azure-cost-optimization.md)** - Cost management strategies for search workloads

### Azure-Native Tools
- **[Azure Evaluation Pipeline](./tools/azure_evaluation_pipeline.py)** - Automated evaluation using Azure services
- **[Azure Configuration](./tools/config.yaml)** - Production-ready Azure configuration templates
- **[Power BI Templates](./tools/powerbi-templates/)** - Pre-built dashboards for search analytics

### Quick Start
1. Review the [Core Metrics](./docs/01-core-metrics.md) to understand key measurement approaches
2. Choose an [Evaluation Framework](./docs/02-evaluation-frameworks.md) that fits your use case
3. Select appropriate [Indexing Strategies](./docs/03-indexing-strategies.md) to test
4. Follow the [Implementation Guide](./guides/implementation-guide.md) to set up your evaluation pipeline

## 🔧 Azure Services Used

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **Azure AI Search** | Primary search engine | Standard tier, hybrid indexing |
| **Azure OpenAI** | Embeddings & semantic analysis | text-embedding-ada-002 model |
| **Azure Cosmos DB** | Results storage & analytics | NoSQL for flexible metrics storage |
| **Azure Monitor** | Performance monitoring | Custom metrics & alerts |
| **Azure Load Testing** | Scalability evaluation | Multi-region load simulation |
| **Power BI** | Analytics dashboards | Real-time search performance insights |
| **Azure Storage** | Datasets & model storage | Blob storage for large datasets |

## 🎯 Azure AI Search Evaluation Modes

This knowledge base covers evaluation of all Azure AI Search capabilities:

### 1. **Keyword Search (BM25)**
- Traditional full-text search with Azure's enhanced BM25
- Custom scoring profiles and analyzers
- **Target**: Precision@5 > 0.7, Latency < 50ms

### 2. **Vector Search** 
- Dense vector similarity using Azure OpenAI embeddings
- HNSW algorithm optimization
- **Target**: nDCG@10 > 0.8, Semantic accuracy > 85%

### 3. **Hybrid Search**
- Combines keyword and vector search with RRF
- Automatic query routing and result fusion
- **Target**: Best of both worlds, 20-30% improvement over single methods

### 4. **Semantic Search**
- Azure's built-in semantic ranking
- Deep learning models for result re-ranking
- **Target**: User satisfaction +40%, Click-through rate +25%

## 📈 Metrics Quick Reference
| Metric | Use Case | Target Value |
|--------|----------|--------------|
| Precision@K | Ranking quality | > 0.7 |
| Recall | Coverage | > 0.8 |
| MAP | Overall ranking | > 0.6 |
| MRR | First relevant result | > 0.6 |
| nDCG | Graded relevance | > 0.8 |
| CTR | User engagement | Baseline + 15% |

---
*Last updated: November 2025*