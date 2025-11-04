# AI Search Indexing Strategy Evaluation Knowledge Base

This repository contains comprehensive documentation and frameworks for measuring search relevancy and accuracy when experimenting with different indexing strategies for AI-powered search systems.

## 📚 Knowledge Base Structure

### Core Documentation
- **[Core Metrics](./docs/01-core-metrics.md)** - Essential metrics for measuring search relevance and accuracy
- **[Evaluation Frameworks](./docs/02-evaluation-frameworks.md)** - Systematic approaches to testing search performance
- **[Indexing Strategies](./docs/03-indexing-strategies.md)** - Comparison of keyword, vector, and hybrid indexing approaches
- **[Advanced Techniques](./docs/04-advanced-techniques.md)** - Semantic re-ranking, query rewriting, and feedback loops

### Practical Guides
- **[Implementation Guide](./guides/implementation-guide.md)** - Step-by-step implementation instructions
- **[Azure AI Search Evaluation](./guides/azure-ai-search-evaluation.md)** - Azure-specific evaluation pipelines
- **[RAG Scenario Evaluation](./guides/rag-scenario-evaluation.md)** - Retrieval-Augmented Generation evaluation strategies

### Tools & Templates
- **[Evaluation Pipeline](./tools/evaluation-pipeline.py)** - Automated metric computation framework
- **[Metrics Comparison](./tools/metrics-comparison.xlsx)** - Spreadsheet for comparing different approaches
- **[Test Datasets](./datasets/)** - Sample datasets for testing different scenarios

### Quick Start
1. Review the [Core Metrics](./docs/01-core-metrics.md) to understand key measurement approaches
2. Choose an [Evaluation Framework](./docs/02-evaluation-frameworks.md) that fits your use case
3. Select appropriate [Indexing Strategies](./docs/03-indexing-strategies.md) to test
4. Follow the [Implementation Guide](./guides/implementation-guide.md) to set up your evaluation pipeline

## 🎯 Key Success Criteria
- **nDCG > 0.8** for top 10 results in navigational queries
- **Precision@5 > 0.7** for informational queries
- **MRR > 0.6** for Q&A scenarios
- **CTR improvement** of 15%+ in A/B testing

## 🔧 Technologies Covered
- Azure AI Search
- Vector databases (Pinecone, Weaviate, Chroma)
- Hybrid search implementations
- RAG (Retrieval-Augmented Generation) systems
- Semantic re-ranking models

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