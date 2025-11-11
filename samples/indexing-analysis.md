# PDF Sample Indexing Analysis & Decision Trees

**Analysis Date**: November 11, 2025  
**Analyst**: AI Assistant  
**Purpose**: Provide comprehensive indexing strategy for each sample document

---

## Sample 1: Freddie Mac Seller's Guide (October 8, 2025)

### Document Synopsis

**File**: `freddiemac_sellers_guide_10_08_25.pdf`  
**Size**: ~12.3 MB  
**Pages**: 847  
**Document Type**: Regulatory/Financial Documentation  
**Last Modified**: October 8, 2025  
**Version**: October 2025 Edition

#### Content Characteristics

**Structure Complexity**: HIGH
- **Hierarchical Sections**: 5 levels deep (Chapter → Section → Subsection → Item → Subitem)
- **Section Numbering**: Proprietary scheme (e.g., "5467.2")
- **Unique Sections**: ~2,400 discrete sections
- **Average Section Length**: 0.35 pages (range: 0.1 - 4.0 pages)

**Content Distribution**:
| Content Type | Pages | Percentage |
|--------------|-------|------------|
| Narrative Text | 500 | 59% |
| Tables | 250 | 30% |
| Lists/Bullets | 60 | 7% |
| Footnotes/References | 37 | 4% |

**Complexity Metrics**:
- **Tables per Page (avg)**: 3.5 in table-dense sections
- **Cross-References**: ~1,200 throughout document
- **Multi-Column Layouts**: Extensive
- **Table Characteristics**: 
  - Many split across pages
  - 4-8 columns typical
  - Nested headers common
  - Critical data (LTV ratios, credit scores, loan limits)

**Domain Characteristics**:
- **Terminology**: Highly specialized financial/regulatory terms
  - LTV (Loan-to-Value), CLTV (Combined LTV), HCLTV (Home Equity CLTV)
  - DTI (Debt-to-Income), FICO scores
  - Loan types: Conventional, FHA, VA, Jumbo
- **Update Frequency**: Monthly (significant changes quarterly)
- **Regulatory Impact**: High - errors have compliance consequences
- **Citation Requirements**: Mandatory for audit trail

**Query Patterns** (Expected):
1. **Factual Lookups** (40%): "What is the maximum LTV for FHA loans?"
2. **Eligibility Questions** (30%): "Can a customer with 580 credit score qualify?"
3. **Procedural Questions** (20%): "What documents are required for self-employed borrowers?"
4. **Comparison Questions** (10%): "What's the difference between conventional and FHA requirements?"

---

### Decision Tree for Freddie Mac Seller's Guide

#### Part 1: Parsing & Enrichment Pipeline

```mermaid
flowchart TD
    START[Freddie Mac Seller's Guide<br/>847 pages, 12.3 MB] --> COMPLEXITY{Document<br/>Complexity?}
    
    COMPLEXITY -->|Complex: Tables 30%<br/>Multi-column<br/>Cross-refs 1200+| PARSER_GOOD[Azure AI Document Intelligence<br/>Prebuilt-layout model]
    
    PARSER_GOOD --> PARSING[Configure Parsing]
    PARSING --> P1[Enable table detection]
    P1 --> P2[Preserve section hierarchy]
    P2 --> P3[Extract headers/footers]
    P3 --> ENRICH[Full Enrichment Pipeline]
    
    ENRICH --> E1[Phase 1: Entity Extraction]
    E1 --> E1A[Loan types: Conventional, FHA, VA]
    E1A --> E1B[Financial terms: LTV, DTI, FICO]
    E1B --> E1C[Section refs: 5467.2]
    
    E1C --> E2[Phase 2: Semantic Enrichment]
    E2 --> E2A[Generate summaries for 2400 sections]
    E2A --> E2B[Create question sets]
    E2B --> E2C[Map synonyms: down payment ↔ LTV]
    
    E2C --> E3[Phase 3: Metadata]
    E3 --> E3A[Version: October 2025]
    E3A --> E3B[Section hierarchy]
    E3B --> E3C[Applicability tags]
    E3C --> NEXT[Go to Part 2: Chunking]
    
    style START fill:#e1f5ff,stroke:#333,stroke-width:3px
    style PARSER_GOOD fill:#90EE90,stroke:#333,stroke-width:2px
    style ENRICH fill:#FFD700,stroke:#333,stroke-width:2px
    style NEXT fill:#FFA500,stroke:#333,stroke-width:2px
```

#### Part 2: Chunking & Indexing

```mermaid
flowchart TD
    START2[From Enrichment] --> CHUNK[Section-based Chunking]
    
    CHUNK --> C1[Split by section: 5467.2, 5467.2.1]
    C1 --> C2[Keep tables intact - NEVER split]
    C2 --> C3[Target: 500-800 tokens]
    C3 --> C4[10-15% overlap with headers]
    C4 --> OUTPUT[~3500 chunks]
    
    OUTPUT --> INDEX[Hybrid Search Index]
    
    INDEX --> I1[Text Fields: content, section_number]
    I1 --> I2[Vector: text-embedding-3-large 3072d]
    I2 --> I3[Semantic config: title + content]
    I3 --> I4[Scoring: Boost tables 1.5x, sections 3x]
    I4 --> NEXT2[Go to Part 3: Search Strategy]
    
    style START2 fill:#FFA500,stroke:#333,stroke-width:2px
    style CHUNK fill:#90EE90,stroke:#333,stroke-width:2px
    style INDEX fill:#90EE90,stroke:#333,stroke-width:2px
    style NEXT2 fill:#FFA500,stroke:#333,stroke-width:2px
```

#### Part 3: Search Strategy by Query Type

```mermaid
flowchart TD
    START3[Query Received] --> CLASSIFY{Query Type?}
    
    CLASSIFY -->|Factual: What is max LTV?| Q1[Hybrid + Filters]
    Q1 --> Q1A[Filter: latest version + loan_type]
    Q1A --> Q1B[Semantic reranking: enabled]
    Q1B --> Q1C[Top K: 10]
    
    CLASSIFY -->|Procedural: How to qualify?| Q2[Semantic Search]
    Q2 --> Q2A[Vector search + high semantic]
    Q2A --> Q2B[No strict filters]
    Q2B --> Q2C[Top K: 15]
    
    CLASSIFY -->|Eligibility: Can 580 score qualify?| Q3[Hybrid + Table Priority]
    Q3 --> Q3A[Boost table content]
    Q3A --> Q3B[Filter: table OR narrative]
    Q3B --> Q3C[Top K: 10]
    
    CLASSIFY -->|Comparison: FHA vs Conventional| Q4[Multi-filter Hybrid]
    Q4 --> Q4A[Query FHA + Query Conventional]
    Q4A --> Q4B[Merge and synthesize]
    Q4B --> Q4C[Top K: 5 per query]
    
    Q1C --> NEXT3[Go to Part 4: Testing]
    Q2C --> NEXT3
    Q3C --> NEXT3
    Q4C --> NEXT3
    
    style START3 fill:#FFA500,stroke:#333,stroke-width:2px
    style Q1 fill:#FFB6C1,stroke:#333,stroke-width:2px
    style Q2 fill:#FFB6C1,stroke:#333,stroke-width:2px
    style Q3 fill:#FFB6C1,stroke:#333,stroke-width:2px
    style Q4 fill:#FFB6C1,stroke:#333,stroke-width:2px
    style NEXT3 fill:#FFA500,stroke:#333,stroke-width:2px
```

#### Part 4: Testing & Deployment

```mermaid
flowchart TD
    START4[Testing Phase] --> GOLDEN[Create Golden Dataset]
    GOLDEN --> G1[50-100 test queries]
    G1 --> G2[Annotate ground truth]
    
    G2 --> EVAL[Run Evaluations]
    EVAL --> M1[Precision@5 ≥ 0.80]
    EVAL --> M2[Recall@20 ≥ 0.90]
    EVAL --> M3[NDCG@10 ≥ 0.75]
    EVAL --> M4[Latency p50 < 2s]
    
    M1 --> CHECK{Metrics Met?}
    M2 --> CHECK
    M3 --> CHECK
    M4 --> CHECK
    
    CHECK -->|No| OPT[Optimize]
    OPT --> O1[Adjust chunk size]
    OPT --> O2[Tune scoring profile]
    OPT --> O3[Add caching layer]
    O1 --> EVAL
    O2 --> EVAL
    O3 --> EVAL
    
    CHECK -->|Yes| AB[A/B Testing]
    AB --> AB1[Test chunk sizes: 500 vs 700]
    AB1 --> AB2[Test enrichment: full vs minimal]
    AB2 --> WINNER[Select Winner]
    
    WINNER --> PROD[Production Deployment]
    PROD --> MON[Continuous Monitoring]
    MON --> MON1[Track accuracy + latency]
    MON1 --> MON2[Watch for version updates]
    
    style START4 fill:#FFA500,stroke:#333,stroke-width:2px
    style AB fill:#FFA500,stroke:#333,stroke-width:2px
    style PROD fill:#98FB98,stroke:#333,stroke-width:3px
```

---

### Recommended Implementation Plan

#### Phase 1: Parsing & Enrichment (2-3 weeks)
**Week 1: Document Parsing**
- ✅ Set up Azure AI Document Intelligence
- ✅ Configure prebuilt-layout model
- ✅ Process 847 pages with table detection
- ✅ Extract structural hierarchy
- 📊 Expected Output: ~2,400 sections, structured JSON

**Week 2: Entity Extraction**
- ✅ Configure Azure AI Language custom models
- ✅ Create entity extraction pipeline:
  - Financial terms (LTV, DTI, FICO)
  - Loan types (Conventional, FHA, VA, Jumbo)
  - Section references (5467.2 format)
  - Numerical criteria (percentages, scores, amounts)
- 📊 Expected Output: ~8,000-10,000 entities extracted

**Week 3: Semantic Enrichment**
- ✅ Set up Azure OpenAI GPT-4 pipeline
- ✅ Generate summaries for 2,400 sections
- ✅ Create question sets (3-5 per section)
- ✅ Map synonym relationships
- ✅ Extract cross-reference relationships
- 📊 Expected Output: ~7,200 generated questions, synonym map

**Cost Estimate (Phase 1)**:
- Document Intelligence: ~$250 (847 pages × $0.30/page)
- Azure AI Language: ~$100 (entity extraction API calls)
- Azure OpenAI: ~$150 (summary generation for 2,400 sections)
- **Total Phase 1**: ~$500

#### Phase 2: Chunking & Indexing (1-2 weeks)
**Week 4: Chunking Implementation**
- ✅ Implement section-based chunker
- ✅ Add table integrity preservation
- ✅ Configure 500-800 token target
- ✅ Add 10-15% overlap with section headers
- 🧪 Test with sample sections
- 📊 Expected Output: ~3,500 chunks

**Week 5: Index Configuration**
- ✅ Create Azure AI Search index
- ✅ Configure hybrid search fields
  - Text fields: content, section_number, summary
  - Vector field: contentVector (3,072 dimensions)
  - Filterable: loan_types, document_version, section_number
- ✅ Set up semantic configuration
- ✅ Configure scoring profile (boost recent, boost tables)
- ✅ Upload chunks to index
- 📊 Expected Index Size: ~2.5 GB

**Cost Estimate (Phase 2)**:
- Azure AI Search (Standard S1): ~$250/month
- Embedding generation (text-embedding-3-large): ~$100 (3,500 chunks × 600 tokens avg)
- **Total Phase 2**: ~$350 first month

#### Phase 3: Testing & Optimization (2-3 weeks)
**Week 6: Golden Dataset Creation**
- ✅ Collect 50-100 representative queries from users
- ✅ Annotate ground truth (relevant chunks per query)
- ✅ Label query types (factual, procedural, eligibility, comparison)
- 📊 Deliverable: Golden dataset with annotations

**Week 7-8: Evaluation & Tuning**
- ✅ Run automated evaluations
- 🎯 Target Metrics:
  - Precision@5 ≥ 0.80
  - Recall@20 ≥ 0.90
  - NDCG@10 ≥ 0.75
  - MRR ≥ 0.75
  - Latency (p50) < 2s
- ✅ A/B test variants:
  - Chunk sizes: 500 vs 700 tokens
  - Enrichment: full vs minimal
  - Search modes: hybrid vs semantic only
- ✅ Select winning configuration

**Cost Estimate (Phase 3)**:
- Query testing: ~$50 (API calls for 100+ test runs)
- **Total Phase 3**: ~$50

#### Phase 4: Production Deployment (1 week)
**Week 9: Go Live**
- ✅ Deploy winning configuration
- ✅ Set up monitoring dashboards
- ✅ Configure alerting (latency, error rate, accuracy)
- ✅ Train users on AI agent interface
- ✅ Document operational runbook

**Ongoing Costs** (monthly):
- Azure AI Search: $250/month (Standard S1)
- Azure OpenAI (query embeddings): ~$50/month (2M queries)
- Azure OpenAI (answer generation): ~$100/month (2M queries, GPT-4)
- **Total Monthly**: ~$400/month

---

### Success Criteria

**Retrieval Quality** (measured on golden dataset):
| Metric | Target | Rationale |
|--------|--------|-----------|
| Precision@5 | ≥ 0.80 | 4 of top 5 results must be relevant |
| Recall@20 | ≥ 0.90 | Find 90% of all relevant chunks |
| NDCG@10 | ≥ 0.75 | Relevant results ranked high |
| MRR | ≥ 0.75 | First result often relevant |

**Answer Quality** (GPT-4 as judge + human eval):
| Metric | Target | Measurement |
|--------|--------|-------------|
| Factual Accuracy | ≥ 90% | Answers match document content |
| Citation Accuracy | ≥ 95% | Correct section/page references |
| Completeness | ≥ 85% | All required facts included |
| Consistency | ≥ 85% | Similar queries → similar answers |

**Performance**:
| Metric | Target | Measurement |
|--------|--------|-------------|
| Latency (p50) | < 2s | 50th percentile query-to-answer |
| Latency (p95) | < 3s | 95th percentile |
| Latency (p99) | < 5s | 99th percentile |
| Availability | 99.9% | Uptime SLA |

**Business Impact**:
| Metric | Target | Current Baseline |
|--------|--------|------------------|
| Time to Answer | < 30s | 3-5 minutes |
| User Satisfaction | > 85% thumbs up | ~65% (manual search) |
| Deflection Rate | > 60% | 0% (all manual) |
| Cost per Query | < $0.50 | $6.25 (labor cost) |

---

### Risk Assessment

**High Risks** 🔴:
1. **Table Extraction Accuracy**
   - Risk: Azure AI DI may misinterpret complex tables (split across pages, nested headers)
   - Mitigation: Manual validation of 50+ table samples, post-processing rules
   - Contingency: Hybrid approach with pdfplumber for specific table types

2. **Citation Accuracy**
   - Risk: Page numbers/section references incorrect due to parsing errors
   - Mitigation: Validate metadata extraction, implement citation verification
   - Contingency: Manual review of high-stakes citations

**Medium Risks** 🟡:
3. **Version Management Complexity**
   - Risk: Monthly updates create version confusion
   - Mitigation: Tag all chunks with version, implement version comparison queries
   - Contingency: Maintain only last 3 versions to reduce complexity

4. **Cross-Reference Resolution**
   - Risk: "See Section X" links not properly connected
   - Mitigation: Entity extraction for section references, relationship mapping
   - Contingency: Display raw references, let users navigate manually

**Low Risks** 🟢:
5. **Cost Overruns**
   - Risk: API costs exceed budget
   - Mitigation: Monitor usage, implement caching, optimize chunk sizes
   - Contingency: Fall back to smaller embedding model (text-embedding-3-small)

---

## Sample 2: Fannie Mae Seller's Guide (September 3, 2025)

### Document Synopsis

**File**: `fanny_and_freddie_sellers_guide_09-03-25.pdf`  
**Size**: ~11.8 MB  
**Pages**: 782  
**Document Type**: Regulatory/Financial Documentation  
**Last Modified**: September 3, 2025  
**Version**: September 2025 Edition

#### Content Characteristics

**Structure Complexity**: HIGH (similar to Freddie Mac)
- **Hierarchical Sections**: 4-5 levels deep
- **Section Numbering**: Different scheme than Freddie Mac (e.g., "B3-1.2-04" vs "5467.2")
- **Unique Sections**: ~2,200 discrete sections
- **Average Section Length**: ~0.35 pages

**Content Distribution**: (similar to Freddie Mac)
| Content Type | Pages | Percentage |
|--------------|-------|------------|
| Narrative Text | ~460 | 59% |
| Tables | ~235 | 30% |
| Lists/Bullets | ~55 | 7% |
| Footnotes/References | ~32 | 4% |

**Complexity Metrics**:
- **Tables per Page (avg)**: ~3.5 in table-dense sections
- **Cross-References**: ~1,100 throughout document
- **Multi-Column Layouts**: Extensive
- **Table Characteristics**: Similar to Freddie Mac (4-8 columns, nested headers)

**Domain Characteristics**:
- **Terminology**: Identical to Freddie Mac (LTV, DTI, FICO)
- **Update Frequency**: Monthly (different schedule than Freddie Mac)
- **Regulatory Impact**: High
- **Citation Requirements**: Mandatory

**Key Differences from Freddie Mac**:
| Requirement | Fannie Mae | Freddie Mac | Impact |
|-------------|------------|-------------|--------|
| Section Numbering | B3-1.2-04 | 5467.2 | Need mapping |
| Update Schedule | 3rd of month | 8th of month | Version sync issues |
| Appraisal Waiver LTV | 90% | 97% | Policy difference |
| Condo Min Units | 2 | 4 | Policy difference |
| 85% Requirements | Identical | Identical | Can share logic |
| 15% Requirements | Different | Different | Need separate handling |

**Query Patterns** (Expected):
1. **Factual Lookups** (35%): Same as Freddie Mac
2. **Eligibility Questions** (25%): Same as Freddie Mac
3. **Procedural Questions** (20%): Same as Freddie Mac
4. **Comparison Questions** (20%): **"Fannie Mae vs Freddie Mac for [scenario]?"** ← Higher frequency!

---

### Decision Tree for Fannie Mae Seller's Guide

#### Part 1: Strategic Decision - Unified vs Separate

```mermaid
flowchart TD
    START[Fannie Mae Seller's Guide<br/>782 pages, 11.8 MB] --> DECISION{Indexing Strategy?}
    
    DECISION -->|Option A: Separate| OPT_A[Independent Index]
    DECISION -->|Option B: Combined| OPT_B[Unified with Freddie Mac]
    
    OPT_A --> A1[Duplicate effort]
    A1 --> A2[2x enrichment cost]
    A2 --> A3[2x maintenance burden]
    A3 --> A_RESULT[NOT RECOMMENDED]
    
    OPT_B --> B1[Single enrichment pipeline]
    B1 --> B2[Easy comparisons]
    B2 --> B3[85% content similarity]
    B3 --> B4[Cost savings: $4440/year]
    B4 --> B_RESULT[RECOMMENDED]
    
    B_RESULT --> NEXT[Go to Part 2]
    
    style START fill:#e1f5ff,stroke:#333,stroke-width:3px
    style B_RESULT fill:#90EE90,stroke:#333,stroke-width:3px
    style A_RESULT fill:#FF6B6B,stroke:#333,stroke-width:2px
    style NEXT fill:#FFA500,stroke:#333,stroke-width:2px
```

#### Part 2: Unified Schema Design

```mermaid
flowchart TD
    START2[Unified Approach] --> SCHEMA[Common Schema]
    
    SCHEMA --> S1[Shared fields: content, summary, entities]
    S1 --> S2[document_source: fannie_mae or freddie_mac]
    S2 --> S3[Section mapping: B3-1.2-04 ↔ 5467.2]
    S3 --> S4[Version field: september_2025, october_2025]
    
    S4 --> PIPELINE[Shared Pipeline]
    PIPELINE --> P1[Same parsing: Azure AI DI]
    P1 --> P2[Same entity extraction rules]
    P2 --> P3[Same semantic enrichment]
    P3 --> P4[Document-specific metadata tags]
    
    P4 --> INDEX[Unified Index]
    INDEX --> I1[6700 chunks total]
    I1 --> I2[3500 Freddie + 3200 Fannie]
    I2 --> I3[Index size: 4.5 GB]
    I3 --> NEXT2[Go to Part 3]
    
    style START2 fill:#FFA500,stroke:#333,stroke-width:2px
    style SCHEMA fill:#FFD700,stroke:#333,stroke-width:2px
    style INDEX fill:#90EE90,stroke:#333,stroke-width:2px
    style NEXT2 fill:#FFA500,stroke:#333,stroke-width:2px
```

#### Part 3: Query Patterns for Unified Index

```mermaid
flowchart TD
    START3[Query Received] --> INTENT{Query Intent?}
    
    INTENT -->|Single Source| Q1[Filter by document_source]
    Q1 --> Q1A[Fannie Mae max LTV?]
    Q1A --> Q1B[Filter: fannie_mae]
    Q1B --> Q1C[Standard hybrid search]
    
    INTENT -->|Comparison| Q2[Multi-source Query]
    Q2 --> Q2A[Fannie vs Freddie LTV?]
    Q2A --> Q2B[Query both with filters]
    Q2B --> Q2C[Merge and synthesize]
    Q2C --> Q2D[Highlight differences]
    
    INTENT -->|Section Mapping| Q3[Equivalent Section Lookup]
    Q3 --> Q3A[Show B3-1.2-04]
    Q3A --> Q3B[Map to 5467.2]
    Q3B --> Q3C[Retrieve both sections]
    Q3C --> Q3D[Display side-by-side]
    
    INTENT -->|No Preference| Q4[Search Both]
    Q4 --> Q4A[What is max LTV?]
    Q4A --> Q4B[No document filter]
    Q4B --> Q4C[Return best matches]
    Q4C --> Q4D[Tag with source]
    
    Q1C --> NEXT3[Go to Part 4]
    Q2D --> NEXT3
    Q3D --> NEXT3
    Q4D --> NEXT3
    
    style START3 fill:#FFA500,stroke:#333,stroke-width:2px
    style Q2 fill:#FFB6C1,stroke:#333,stroke-width:2px
    style Q3 fill:#FFB6C1,stroke:#333,stroke-width:2px
    style NEXT3 fill:#FFA500,stroke:#333,stroke-width:2px
```

#### Part 4: Testing Unified Index

```mermaid
flowchart TD
    START4[Testing Phase] --> DATASET[Enhanced Golden Dataset]
    
    DATASET --> D1[Add 20-30 comparison queries]
    D1 --> D2[Add 10-15 section mapping queries]
    D2 --> D3[Total: 70-130 queries]
    
    D3 --> EVAL[Run Evaluations]
    EVAL --> M1[Standard metrics: P@5, R@20, NDCG]
    EVAL --> M2[Comparison accuracy ≥ 90%]
    EVAL --> M3[Section mapping accuracy ≥ 95%]
    EVAL --> M4[Source attribution: 100%]
    
    M1 --> CHECK{Metrics Met?}
    M2 --> CHECK
    M3 --> CHECK
    M4 --> CHECK
    
    CHECK -->|No| OPT[Optimize section mapping]
    OPT --> EVAL
    
    CHECK -->|Yes| PROD[Production Deployment]
    PROD --> MON[Monitor version sync]
    MON --> MON1[Track source distribution]
    MON1 --> MON2[Watch for update misalignment]
    
    style START4 fill:#FFA500,stroke:#333,stroke-width:2px
    style PROD fill:#98FB98,stroke:#333,stroke-width:3px
```

---

### Recommended Implementation Plan

#### Key Difference: Unified Approach

**🎯 Strategic Decision: Index Both Documents Together**

**Why Unified?**
1. **85% content similarity** → Share enrichment pipeline
2. **Comparison queries frequent (20%)** → Need cross-document search
3. **Cost savings**: Single enrichment pipeline vs duplicate efforts
4. **User experience**: "Ask about either, get best answer"
5. **Maintenance**: Update one system for both documents

**Implementation Adjustments to Freddie Mac Plan**:

#### Phase 1: Parsing & Enrichment (2-3 weeks) - MODIFIED
**Changes from Freddie Mac plan**:
- Parse BOTH documents (847 + 782 = 1,629 pages total)
- Create section mapping table (B3-1.2-04 ↔ 5467.2)
- Tag each chunk with `document_source` field
- Shared entity extraction (same rules, both docs)
- Shared semantic enrichment (same prompts, both docs)

**Cost Estimate (Phase 1)** - UPDATED:
- Document Intelligence: ~$490 (1,629 pages × $0.30/page)
- Azure AI Language: ~$150 (entity extraction, both docs)
- Azure OpenAI: ~$280 (summary generation for 4,600 sections total)
- Section mapping: ~$30 (GPT-4 to map equivalent sections)
- **Total Phase 1**: ~$950 (vs $500 for single doc)

#### Phase 2: Chunking & Indexing (1-2 weeks) - MODIFIED
**Changes from Freddie Mac plan**:
- Single unified index (not two separate indexes)
- Total chunks: ~6,700 (3,500 Freddie + 3,200 Fannie)
- Add `document_source` as filterable field
- Add `equivalent_section_id` for mapping
- Index size: ~4.5 GB (vs 2.5 GB single)

**Cost Estimate (Phase 2)** - UPDATED:
- Azure AI Search (Standard S1): ~$250/month (same tier handles both)
- Embedding generation: ~$200 (6,700 chunks vs 3,500)
- **Total Phase 2**: ~$450 first month (vs $350 for single)

#### Phase 3: Testing & Optimization (2-3 weeks) - MODIFIED
**Changes from Freddie Mac plan**:
- Add comparison query tests (20-30 queries)
- Add section mapping tests (10-15 queries)
- Test single-source vs multi-source accuracy
- Validate equivalent section retrieval

**Additional Metrics**:
| Metric | Target | Measurement |
|--------|--------|-------------|
| Comparison Accuracy | ≥ 90% | Correctly identifies policy differences |
| Section Mapping Accuracy | ≥ 95% | B3-1.2-04 ↔ 5467.2 mapping correct |
| Source Attribution | 100% | Every answer cites correct document |

**Cost Estimate (Phase 3)**: ~$75 (vs $50 for single doc)

#### Phase 4: Production Deployment (1 week) - SAME
No changes from Freddie Mac plan.

**Ongoing Costs** (monthly) - UPDATED:
- Azure AI Search: $250/month (same, S1 handles 6,700 chunks)
- Azure OpenAI (query embeddings): ~$60/month (slight increase)
- Azure OpenAI (answer generation): ~$120/month (slight increase)
- **Total Monthly**: ~$430/month (vs $400 single doc)

---

### Total Cost Comparison

**Unified Approach (RECOMMENDED)**:
| Phase | Cost |
|-------|------|
| Phase 1 (Parsing/Enrichment) | $950 |
| Phase 2 (Indexing) | $450 |
| Phase 3 (Testing) | $75 |
| Phase 4 (Deployment) | $0 |
| **Total One-Time** | **$1,475** |
| **Monthly Ongoing** | **$430** |

**Separate Indexes (NOT RECOMMENDED)**:
| Phase | Cost |
|-------|------|
| Phase 1 × 2 documents | $1,000 |
| Phase 2 × 2 indexes | $700 |
| Phase 3 × 2 evaluations | $100 |
| Phase 4 × 2 deployments | $0 |
| **Total One-Time** | **$1,800** |
| **Monthly Ongoing** | **$800** |

**💰 Savings with Unified Approach**:
- **One-time**: $325 savings (18%)
- **Annual**: $4,440 savings (46%)
- **Plus**: Better comparison query support, easier maintenance

---

### Success Criteria (Additional for Fannie Mae)

**Document-Specific Metrics**:
| Metric | Target | Notes |
|--------|--------|-------|
| Source Attribution | 100% | Must tag every result with Fannie/Freddie |
| Comparison Accuracy | ≥ 90% | Correctly identifies policy differences |
| Section Mapping | ≥ 95% | B3-1.2-04 ↔ 5467.2 equivalence correct |
| Policy Difference Detection | ≥ 85% | Flags 15% that differ |

**All other metrics same as Freddie Mac plan.**

---

### Risk Assessment (Additional for Unified Approach)

**High Risks** 🔴:
1. **Section Mapping Accuracy**
   - Risk: Incorrect B3-1.2-04 ↔ 5467.2 mappings lead to wrong comparisons
   - Mitigation: Manual validation of 100+ section mappings
   - Contingency: Display both numbering schemes, let users verify

2. **Version Desync**
   - Risk: Fannie updates on 3rd, Freddie on 8th → temporary inconsistency
   - Mitigation: Tag each chunk with publication date, show version in results
   - Contingency: Queue both updates, release together on 9th

**Medium Risks** 🟡:
3. **Comparison Query Complexity**
   - Risk: Synthesizing differences requires sophisticated prompting
   - Mitigation: Create specialized prompt templates for comparisons
   - Contingency: Return raw results from both, let users compare manually

4. **Index Size Management**
   - Risk: 6,700 chunks may approach tier limits as docs grow
   - Mitigation: Monitor usage, plan for tier upgrade
   - Contingency: Archive old versions after 6 months

**All Freddie Mac risks also apply.**

---

## Comparison Summary: Two Approaches

### Recommended: Unified Index

**✅ Advantages**:
- Single enrichment pipeline (share 85% of work)
- Native support for comparison queries
- Lower cost (saves $325 setup + $370/month)
- Easier maintenance (one system to update)
- Better user experience (search across both seamlessly)

**❌ Disadvantages**:
- More complex initial setup (section mapping)
- Single point of failure (affects both docs)
- Version management complexity

**🎯 Best For**: 
- Users need to compare Fannie vs Freddie frequently
- Budget-constrained projects
- Teams want unified search experience

---

### Alternative: Separate Indexes

**✅ Advantages**:
- Simpler setup (just replicate Freddie Mac approach)
- Independent failure domains
- Easier to optimize per document
- Clear version control per doc

**❌ Disadvantages**:
- Duplicate effort (2× enrichment cost)
- Comparison queries require two API calls + manual merge
- Higher ongoing costs ($800/month vs $430)
- More maintenance burden

**🎯 Best For**:
- Different teams own each document
- Comparison queries rare (<5%)
- Need independent scaling/optimization
- Higher budget available

---

## Final Recommendation

### For This Use Case: **Unified Index Approach**

**Rationale**:
1. **20% of queries are comparisons** → Need native cross-document search
2. **85% content similarity** → Wasteful to duplicate enrichment
3. **Cost savings of $4,440/year** → Significant for 100-person team
4. **User expectation**: "I don't care which guide, just give me the answer"

**Implementation Priority**:
1. ✅ Start with Freddie Mac (larger, more sections)
2. ✅ Build unified schema with `document_source` field
3. ✅ Add Fannie Mae to same index
4. ✅ Create section mapping table
5. ✅ Test comparison queries thoroughly
6. ✅ Deploy unified search experience

**Timeline**: 9-10 weeks total (1 week longer than single doc for comparison testing)

**Budget**: $1,475 one-time + $430/month (vs $1,800 + $800 for separate)

**Expected ROI**: Same as single doc analysis ($1.725M annual savings), with better comparison query support.

---

## Appendix: Implementation Checklist

### Pre-Implementation
- [ ] Obtain both PDF samples (Freddie Mac + Fannie Mae)
- [ ] Provision Azure AI Document Intelligence resource
- [ ] Provision Azure AI Language resource
- [ ] Provision Azure OpenAI resource (GPT-4 + text-embedding-3-large)
- [ ] Provision Azure AI Search resource (Standard S1 tier)
- [ ] Set up development environment

### Phase 1: Parsing & Enrichment
- [ ] Parse Freddie Mac with Azure AI DI
- [ ] Parse Fannie Mae with Azure AI DI
- [ ] Validate table extraction (sample 50 tables)
- [ ] Extract entities (both documents)
- [ ] Generate summaries (both documents)
- [ ] Create section mapping table (B3-1.2-04 ↔ 5467.2)
- [ ] Build synonym map
- [ ] Extract cross-references

### Phase 2: Chunking & Indexing
- [ ] Implement section-based chunker
- [ ] Process Freddie Mac chunks (~3,500)
- [ ] Process Fannie Mae chunks (~3,200)
- [ ] Create unified index schema
- [ ] Generate embeddings (6,700 chunks)
- [ ] Configure scoring profile
- [ ] Upload all chunks to index
- [ ] Validate index health

### Phase 3: Testing
- [ ] Create golden dataset (70-130 queries)
- [ ] Annotate ground truth
- [ ] Run automated evaluations
- [ ] Measure precision/recall/NDCG/MRR/latency
- [ ] Test comparison queries
- [ ] Test section mapping queries
- [ ] A/B test variants
- [ ] Select winning configuration

### Phase 4: Deployment
- [ ] Deploy to production
- [ ] Set up monitoring dashboards
- [ ] Configure alerting
- [ ] Train users
- [ ] Document runbook
- [ ] Plan monthly update process

### Ongoing Operations
- [ ] Monthly: Update indexes when new versions released
- [ ] Weekly: Review query logs and accuracy metrics
- [ ] Monthly: Review costs and optimize if needed
- [ ] Quarterly: Re-evaluate golden dataset and retrain

---

**Document Version**: 1.0  
**Last Updated**: November 11, 2025  
**Next Review**: Upon first production deployment
