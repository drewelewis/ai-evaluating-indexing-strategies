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

### Foundation Documents (Start Here)
- **[📖 Table of Contents](./docs/00-table-of-contents.md)** - Complete navigation guide for all documentation
- **[📊 Core Metrics](./docs/01-core-metrics.md)** - Essential metrics for measuring search relevance and accuracy (Precision, Recall, MRR, NDCG)
- **[🔬 Evaluation Frameworks](./docs/02-evaluation-frameworks.md)** - Systematic approaches to testing search systems
- **[🗂️ Indexing Strategies](./docs/03-indexing-strategies.md)** - Comprehensive guide to Azure AI Search indexing strategy selection
- **[🚀 Advanced Techniques](./docs/04-advanced-techniques.md)** - Advanced search optimization techniques

### Azure Service Setup & Integration
- **[☁️ Azure AI Search Setup](./docs/04-azure-ai-search-setup.md)** - Complete Azure AI Search service configuration guide
- **[🤖 Azure OpenAI Integration](./docs/05-azure-openai-integration.md)** - Vector embeddings and semantic capabilities setup
- **[💾 Azure Cosmos DB Analytics](./docs/06-azure-cosmos-db-analytics.md)** - Storing evaluation results and query analytics
- **[📈 Azure Monitor & Logging](./docs/07-azure-monitor-logging.md)** - Performance monitoring and operational insights

### Search Implementation Types
- **[🔤 Full-Text Search (BM25)](./docs/08-fulltext-search-bm25.md)** - Traditional keyword search with BM25 ranking
- **[🧮 Vector Search](./docs/09-vector-search.md)** - Dense vector similarity using embeddings and HNSW
- **[🔀 Hybrid Search](./docs/10-hybrid-search.md)** - Combined keyword + vector search with RRF fusion
- **[🎯 Semantic Search](./docs/11-semantic-search.md)** - Azure's L2 semantic ranking for natural language queries
- **[⚡ Query Optimization](./docs/12-query-optimization.md)** - Performance tuning and query efficiency
- **[🔧 Custom Analyzers](./docs/13-custom-analyzers.md)** - Custom text analysis and tokenization
- **[⭐ Scoring Profiles](./docs/14-scoring-profiles.md)** - Custom relevance scoring and field boosting

### Operational Guides
- **[🗃️ Index Management](./docs/15-index-management.md)** - Index lifecycle, updates, and maintenance
- **[📦 Dataset Preparation](./docs/16-dataset-preparation.md)** - Preparing evaluation datasets and test queries
- **[🧪 A/B Testing Framework](./docs/17-ab-testing-framework.md)** - Testing search strategy variations

### Practical Implementation Guides
- **[🎓 Golden Dataset Guide](./guides/golden-dataset-guide.md)** - Creating high-quality evaluation datasets
- **[🔍 Azure AI Search Evaluation](./guides/search-evaluation.md)** - Complete Azure-specific evaluation pipeline implementation
- **[📋 Sample Search Strategies](./guides/sample-search-strategies.md)** - Real-world search strategy examples

### Quick Start
1. **Start with the basics**: Read [Table of Contents](./docs/00-table-of-contents.md) for a complete overview
2. **Understand metrics**: Review [Core Metrics](./docs/01-core-metrics.md) to learn measurement approaches
3. **Choose a framework**: Select an [Evaluation Framework](./docs/02-evaluation-frameworks.md) that fits your use case
4. **Pick your strategy**: Review [Indexing Strategies](./docs/03-indexing-strategies.md) to select search modes to test
5. **Set up Azure**: Follow [Azure AI Search Setup](./docs/04-azure-ai-search-setup.md) to configure your environment
6. **Build your pipeline**: Use [Azure AI Search Evaluation Guide](./guides/search-evaluation.md) for complete implementation
7. **Prepare test data**: Follow [Golden Dataset Guide](./guides/golden-dataset-guide.md) to create evaluation datasets

## 🔧 Azure Services Used

This knowledge base provides comprehensive guidance for using Azure services in search evaluation:

| Service | Purpose | Key Features | Documentation |
|---------|---------|--------------|---------------|
| **Azure AI Search** | Primary search engine | BM25, Vector (HNSW), Hybrid (RRF), Semantic ranking | [Setup Guide](./docs/04-azure-ai-search-setup.md) |
| **Azure OpenAI** | Embeddings & semantic analysis | text-embedding-ada-002, text-embedding-3-small/large | [Integration Guide](./docs/05-azure-openai-integration.md) |
| **Azure Cosmos DB** | Results storage & analytics | NoSQL with flexible schema for metrics | [Analytics Guide](./docs/06-azure-cosmos-db-analytics.md) |
| **Azure Monitor** | Performance monitoring | Custom metrics, Log Analytics, KQL queries | [Logging Guide](./docs/07-azure-monitor-logging.md) |
| **Azure Cognitive Services** | Query intent analysis | Text Analytics for query categorization | [Evaluation Guide](./guides/search-evaluation.md) |
| **Azure Storage** | Datasets & artifacts | Blob storage for evaluation datasets | [Dataset Preparation](./docs/16-dataset-preparation.md) |

### Service Tier Recommendations

**For Evaluation/Testing (Low Cost)**
- Azure AI Search: Basic tier ($75/month)
- Azure OpenAI: Pay-as-you-go (~$10-50/month for testing)
- Azure Cosmos DB: Serverless (pay per use, ~$5-20/month)
- Azure Monitor: Free tier (first 5GB/month free)

**For Production (Balanced)**
- Azure AI Search: Standard S1 ($250/month) + Semantic ($250/month if needed)
- Azure OpenAI: Standard deployment (predictable costs)
- Azure Cosmos DB: Provisioned throughput (400-4000 RU/s, $24-600/month)
- Azure Monitor: Standard tier (~$50-100/month)

*See individual service documentation for detailed cost optimization strategies*

## 💡 Key Concepts

### Search Modes Explained

**When to use each mode** (detailed in [Indexing Strategies](./docs/03-indexing-strategies.md)):

1. **Keyword Search (BM25)** - Use when:
   - Queries use exact terminology from documents
   - Searching for IDs, SKUs, or precise terms
   - Budget is constrained (no embedding costs)
   - Latency must be <50ms

2. **Vector Search** - Use when:
   - Users express concepts differently than documents
   - Need multilingual search capabilities
   - Semantic understanding is critical
   - Example: "heart attack" should find "myocardial infarction"

3. **Hybrid Search** - Use when:
   - Mix of exact and semantic queries
   - Want best overall recall (80-92%)
   - Budget allows embeddings but not semantic tier
   - General-purpose production search

4. **Semantic Search** - Use when:
   - Users ask natural language questions
   - Budget allows S1+ tier + semantic SKU
   - Need maximum relevance (90%+ recall)
   - Example: "What's the best budget laptop for students?"

### Evaluation Workflow

The typical evaluation process (detailed in [Evaluation Frameworks](./docs/02-evaluation-frameworks.md)):

1. **Baseline Establishment** - Test current search performance
2. **Strategy Selection** - Choose search modes to evaluate
3. **Dataset Preparation** - Create golden dataset with relevance judgments
4. **Implementation** - Deploy search index with selected modes
5. **Metrics Collection** - Run evaluation pipeline, collect Precision/Recall/NDCG
6. **Analysis** - Compare modes, identify best performers by query type
7. **Optimization** - Tune parameters (HNSW, analyzers, scoring profiles)
8. **Validation** - A/B test in production, measure user engagement
9. **Monitoring** - Track operational metrics (latency, cost, errors)
10. **Iteration** - Continuous improvement based on production data

## 📖 Documentation Roadmap

### Completed Documentation (✅)
- Core foundation documents (metrics, frameworks, strategies)
- Azure service setup guides (AI Search, OpenAI, Cosmos DB, Monitor)
- Search implementation types (BM25, Vector, Hybrid, Semantic)
- Query optimization and customization guides
- Operational guides (index management, datasets, A/B testing)
- Practical implementation guides (evaluation pipeline, golden dataset)

### Coming Soon (🚧)
Additional operational and advanced topics will be documented as the repository evolves:
- Load testing strategies and frameworks
- Cost analysis and optimization techniques
- Multi-region deployment patterns
- Security and compliance considerations
- CI/CD pipeline integration
- Domain-specific search patterns
- Multilingual search evaluation
- Knowledge mining and enrichment

## 🤝 Contributing

This knowledge base is designed to be comprehensive and accurate. Contributions are welcome:

1. **Documentation improvements**: Clarifications, examples, corrections
2. **Code samples**: Evaluation scripts, Azure integration examples
3. **Real-world case studies**: Production search evaluation experiences
4. **Additional metrics**: New evaluation approaches and formulas

## 📞 Support & Resources

- **Azure AI Search Documentation**: https://learn.microsoft.com/azure/search/
- **Azure OpenAI Documentation**: https://learn.microsoft.com/azure/ai-services/openai/
- **Search Evaluation Research**: Academic papers and industry best practices
- **Community Forums**: Azure Tech Community, Stack Overflow

## 📝 License & Attribution

This knowledge base is provided for educational and reference purposes. When implementing search systems:
- Follow Azure service terms and conditions
- Respect data privacy and compliance requirements
- Properly attribute external research and methodologies
- Test thoroughly before production deployment

---

**Repository Status**: Active development  
**Last Updated**: November 2025  
**Primary Focus**: Azure AI Search evaluation best practices  
**Maintained By**: Azure search evaluation community

For questions, issues, or suggestions, please use GitHub Issues or Discussions.

## 🎯 Azure AI Search Evaluation Modes

This knowledge base covers comprehensive evaluation of all Azure AI Search capabilities:

### 1. **Keyword Search (BM25)** - [Full Documentation](./docs/08-fulltext-search-bm25.md)
- Traditional full-text search with Azure's enhanced BM25 algorithm
- Custom scoring profiles and language analyzers
- Best for: Exact term matching, product IDs, technical documentation
- **Evaluation Targets**: Precision@5 > 0.7, Latency < 50ms, Recall > 0.6

### 2. **Vector Search** - [Full Documentation](./docs/09-vector-search.md)
- Dense vector similarity using Azure OpenAI embeddings
- HNSW (Hierarchical Navigable Small World) algorithm for approximate nearest neighbors
- Best for: Semantic similarity, synonym matching, cross-lingual search
- **Evaluation Targets**: nDCG@10 > 0.8, Semantic accuracy > 85%, Recall > 0.75

### 3. **Hybrid Search** - [Full Documentation](./docs/10-hybrid-search.md)
- Combines keyword (BM25) and vector search using Reciprocal Rank Fusion (RRF)
- Automatic query routing and intelligent result fusion
- Best for: General-purpose search, mixed query types
- **Evaluation Targets**: Best of both worlds, 20-30% improvement over single methods, Recall > 0.85

### 4. **Semantic Search** - [Full Documentation](./docs/11-semantic-search.md)
- Azure's built-in L2 semantic ranking with deep learning models
- Semantic captions and answers extraction
- Best for: Natural language questions, long-form queries
- **Evaluation Targets**: User satisfaction +40%, Click-through rate +25%, nDCG@10 > 0.9

## 📊 Search Strategy Comparison

| Search Mode | Latency | Recall | Best Use Case | Azure Tier |
|-------------|---------|--------|---------------|------------|
| **Keyword (BM25)** | 20-50ms | 45-65% | Exact matches, IDs | Basic+ |
| **Vector** | 80-150ms | 70-85% | Semantic similarity | Standard+ |
| **Hybrid** | 100-180ms | 80-92% | General purpose | Standard+ |
| **Semantic** | 150-250ms | 85-95% | Natural language | S1+ |

*See [Indexing Strategies](./docs/03-indexing-strategies.md) for detailed decision framework*

## 📈 Metrics Quick Reference

Comprehensive metrics documentation available in [Core Metrics Guide](./docs/01-core-metrics.md)

| Metric | Definition | Use Case | Target Value | Documentation |
|--------|------------|----------|--------------|---------------|
| **Precision@K** | Relevance of top K results | Ranking quality | > 0.7 | [Section 2.1](./docs/01-core-metrics.md#precision-at-k) |
| **Recall@K** | Coverage in top K results | Result completeness | > 0.8 | [Section 2.2](./docs/01-core-metrics.md#recall-at-k) |
| **MAP** | Mean Average Precision | Overall ranking quality | > 0.6 | [Section 2.3](./docs/01-core-metrics.md#map) |
| **MRR** | Mean Reciprocal Rank | First relevant result | > 0.6 | [Section 2.4](./docs/01-core-metrics.md#mrr) |
| **nDCG@K** | Normalized Discounted Cumulative Gain | Graded relevance | > 0.8 | [Section 2.5](./docs/01-core-metrics.md#ndcg) |
| **F1 Score** | Harmonic mean of precision/recall | Balanced performance | > 0.7 | [Section 2.6](./docs/01-core-metrics.md#f1-score) |
| **CTR** | Click-through rate | User engagement | Baseline + 15% | [Section 3.1](./docs/01-core-metrics.md#user-engagement) |
| **Latency P95** | 95th percentile response time | Performance SLA | < 200ms | [Section 4.1](./docs/01-core-metrics.md#latency) |

## 🛠️ Repository Structure

```
ai-evaluating-indexing-strategies/
├── docs/                           # Comprehensive documentation
│   ├── 00-table-of-contents.md    # Complete navigation guide
│   ├── 01-core-metrics.md         # Metrics fundamentals
│   ├── 02-evaluation-frameworks.md # Testing approaches
│   ├── 03-indexing-strategies.md  # Strategy selection guide
│   ├── 04-advanced-techniques.md  # Advanced optimization
│   ├── 04-azure-ai-search-setup.md # Azure setup guide
│   ├── 05-azure-openai-integration.md # Embeddings setup
│   ├── 06-azure-cosmos-db-analytics.md # Analytics storage
│   ├── 07-azure-monitor-logging.md # Monitoring setup
│   ├── 08-fulltext-search-bm25.md # BM25 implementation
│   ├── 09-vector-search.md        # Vector search guide
│   ├── 10-hybrid-search.md        # Hybrid search guide
│   ├── 11-semantic-search.md      # Semantic ranking
│   ├── 12-query-optimization.md   # Performance tuning
│   ├── 13-custom-analyzers.md     # Text analysis
│   ├── 14-scoring-profiles.md     # Custom scoring
│   ├── 15-index-management.md     # Index operations
│   ├── 16-dataset-preparation.md  # Test data creation
│   └── 17-ab-testing-framework.md # A/B testing
├── guides/                         # Practical implementation
│   ├── golden-dataset-guide.md    # Dataset creation
│   ├── search-evaluation.md       # Azure evaluation pipeline
│   └── sample-search-strategies.md # Real-world examples
├── src/                            # Source code (when implemented)
│   ├── engines/                    # Search engine implementations
│   ├── evaluation/                 # Evaluation framework
│   ├── azure/                      # Azure service integrations
│   └── metrics/                    # Metric calculations
├── datasets/                       # Sample evaluation datasets
├── examples/                       # Code examples
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites
- Azure subscription with access to:
  - Azure AI Search (Standard tier or higher recommended)
  - Azure OpenAI Service (for vector/semantic search)
  - Azure Cosmos DB (optional, for analytics storage)
  - Azure Monitor & Log Analytics (for operational metrics)
- Python 3.8+ (for evaluation scripts)
- Basic understanding of search concepts and Azure services

### Step-by-Step Guide

1. **Understand the Fundamentals** (30-45 min)
   - Read [Table of Contents](./docs/00-table-of-contents.md) for repository overview
   - Study [Core Metrics](./docs/01-core-metrics.md) to understand evaluation metrics
   - Review [Evaluation Frameworks](./docs/02-evaluation-frameworks.md) for testing methodologies

2. **Choose Your Search Strategy** (15-20 min)
   - Read [Indexing Strategies](./docs/03-indexing-strategies.md) for decision framework
   - Understand tradeoffs between keyword, vector, hybrid, and semantic search
   - Identify which mode(s) best fit your use case

3. **Set Up Azure Services** (45-60 min)
   - Follow [Azure AI Search Setup](./docs/04-azure-ai-search-setup.md) to create search service
   - If using vector/semantic: [Azure OpenAI Integration](./docs/05-azure-openai-integration.md)
   - Optional: [Azure Cosmos DB Analytics](./docs/06-azure-cosmos-db-analytics.md) for results storage
   - Set up monitoring: [Azure Monitor & Logging](./docs/07-azure-monitor-logging.md)

4. **Prepare Your Evaluation Dataset** (2-4 hours)
   - Follow [Golden Dataset Guide](./guides/golden-dataset-guide.md)
   - Use [Dataset Preparation](./docs/16-dataset-preparation.md) for best practices
   - Aim for 100+ test queries with relevance judgments

5. **Implement Search Modes** (varies by complexity)
   - Start with [Full-Text Search (BM25)](./docs/08-fulltext-search-bm25.md)
   - Add [Vector Search](./docs/09-vector-search.md) if needed
   - Implement [Hybrid Search](./docs/10-hybrid-search.md) for best results
   - Enable [Semantic Search](./docs/11-semantic-search.md) for natural language

6. **Run Evaluation Pipeline** (1-2 hours)
   - Follow [Azure AI Search Evaluation](./guides/search-evaluation.md) guide
   - Execute baseline tests for each search mode
   - Collect metrics: Precision, Recall, MRR, NDCG, latency

7. **Optimize and Iterate** (ongoing)
   - Use [Query Optimization](./docs/12-query-optimization.md) for performance
   - Tune with [Custom Analyzers](./docs/13-custom-analyzers.md)
   - Adjust [Scoring Profiles](./docs/14-scoring-profiles.md)
   - Run [A/B Testing](./docs/17-ab-testing-framework.md) for validation

### Example: Evaluating Hybrid Search

```bash
# 1. Review hybrid search documentation
Read: docs/10-hybrid-search.md

# 2. Set up Azure AI Search with vector support
Follow: docs/04-azure-ai-search-setup.md
Deploy: Azure AI Search Standard tier
Enable: Vector search in index schema

# 3. Configure Azure OpenAI for embeddings
Follow: docs/05-azure-openai-integration.md
Deploy: text-embedding-3-small model
Test: Embedding generation

# 4. Prepare test dataset
Follow: guides/golden-dataset-guide.md
Create: 100-500 test queries with relevance judgments

# 5. Implement evaluation pipeline
Follow: guides/search-evaluation.md
Run: Evaluation for keyword, vector, and hybrid modes
Compare: Precision@5, Recall@10, NDCG@10, latency

# 6. Analyze results and optimize
Review: Metrics by query intent category
Tune: HNSW parameters, RRF weights
Iterate: Re-evaluate after changes
```