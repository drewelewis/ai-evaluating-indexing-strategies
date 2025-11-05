# AI Agent PDF Search Strategy Guide - OUTLINE

## Executive Summary
- Use case: AI Agent for customer service and sales professionals
- Document types: Freddie Mac/Fannie Mae seller's guides (PDF format)
- Goal: Enable AI agents to retrieve accurate, contextual information from complex financial documents

---

## 1. Introduction

### 1.1 Use Case Overview
- **Target Users**: Customer service representatives, sales professionals
- **Document Corpus**: Regulatory and guideline PDFs (Freddie Mac, Fannie Mae seller's guides)
- **AI Agent Requirements**:
  - Quick, accurate information retrieval
  - Context-aware responses
  - Support for complex queries (regulatory compliance, eligibility criteria, etc.)
  - Low latency for real-time customer interactions

### 1.2 Sample Documents Analysis
- **fanny_and_freddie_sellers_guide_09-03-25.pdf**
  - Document characteristics (size, structure, complexity)
  - Content types (text, tables, lists, regulatory sections)
- **freddiemac_sellers_guide_10_08_25.pdf**
  - Document characteristics
  - Content types and organizational structure

### 1.3 Key Challenges
- Complex PDF structure (multi-column, tables, nested sections)
- Financial/regulatory terminology
- Need for precise citations and references
- Handling updates and version control
- Balancing chunk size with context preservation

---

## 2. Document Parsing Strategy

### 2.1 Parser Selection Criteria
- Accuracy requirements for financial documents
- Table extraction capabilities
- Layout preservation vs. content extraction
- Processing speed and cost considerations

### 2.2 Recommended Approach: Azure AI Document Intelligence
**Reference**: [docs/05-azure-openai-integration.md - PDF Document Parsing]

#### Why Azure AI Document Intelligence?
- Superior table detection and extraction
- Preserves document structure and hierarchy
- Handles multi-column layouts
- Extracts headers, footers, section markers
- OCR capabilities for scanned content

#### Implementation Strategy
```
- Use prebuilt-layout model for initial parsing
- Extract structural elements:
  * Sections and headings
  * Tables with cell relationships
  * Lists and bullet points
  * Page numbers and references
- Preserve formatting metadata
```

### 2.3 Alternative Parsers (Comparison)
- **PyPDF2/PyMuPDF**: Simple text extraction (not recommended for complex docs)
- **pdfplumber**: Good for tables but limited layout understanding
- **Unstructured.io**: Good middle ground, open-source option
- **Adobe PDF Services**: Enterprise option with high accuracy

### 2.4 Parsing Output Structure
```
{
  "document_id": "freddiemac_sellers_guide_10_08_25",
  "pages": [...],
  "sections": [
    {
      "title": "Section 1234.5 - Eligibility Requirements",
      "content": "...",
      "tables": [...],
      "page_range": [45, 47],
      "subsections": [...]
    }
  ],
  "metadata": {...}
}
```

---

## 3. Enrichment Strategy

### 3.1 Enrichment Goals
- Add semantic context for better retrieval
- Extract domain-specific entities
- Identify cross-references and related sections
- Generate summaries for quick scanning

### 3.2 Recommended Enrichment Pipeline
**Reference**: [docs/05-azure-openai-integration.md - Data Enrichment], [docs/26-knowledge-mining.md]

#### Phase 1: Entity Extraction
- **Azure AI Language** for:
  - Named Entity Recognition (financial terms, loan types, etc.)
  - Key phrase extraction
  - Custom entity models for domain-specific terms
- Extract:
  * Loan product names
  * Regulatory codes and references
  * Numerical criteria (LTV ratios, credit scores, etc.)
  * Document section references

#### Phase 2: Semantic Enrichment
- **Azure OpenAI** for:
  - Section summaries
  - Question generation (what might users ask about this section?)
  - Relationship extraction (prerequisite sections, related topics)

#### Phase 3: Metadata Augmentation
- Add searchable metadata:
  * Document version and date
  * Section hierarchy (breadcrumb trail)
  * Topic categorization
  * Applicability tags (conventional loans, FHA, VA, etc.)

### 3.3 Enrichment Output Example
```
{
  "original_content": "...",
  "entities": {
    "loan_types": ["Conventional", "FHA", "VA"],
    "criteria": ["LTV <= 80%", "Credit Score >= 620"],
    "references": ["Section 1234.5", "Appendix B"]
  },
  "summary": "This section defines eligibility requirements...",
  "questions": [
    "What is the maximum LTV for conventional loans?",
    "What credit score is required?"
  ],
  "metadata": {
    "category": "Eligibility Requirements",
    "applicability": ["conventional_loans"],
    "version": "2025-10-08"
  }
}
```

---

## 4. Chunking Strategy

### 4.1 Chunking Approach for Financial PDFs
**Reference**: [docs/05-azure-openai-integration.md - PDF Chunking Strategies]

### 4.2 Recommended: Hybrid Semantic + Structural Chunking

#### Strategy Overview
- **Primary**: Section-based chunking (preserve regulatory sections intact)
- **Secondary**: Semantic chunking for large sections (max 1000 tokens)
- **Preserve**: Table integrity (never split tables across chunks)

#### Chunking Rules
1. **Respect Document Structure**
   - Keep section headings with content
   - Preserve parent-child relationships
   - Maintain table atomicity

2. **Size Constraints**
   - Target: 500-800 tokens per chunk
   - Maximum: 1000 tokens
   - Minimum: 100 tokens (avoid tiny fragments)

3. **Overlap Strategy**
   - 10-15% overlap between adjacent chunks
   - Include section header in overlap
   - Preserve context at boundaries

#### Chunking Examples
```
Chunk 1: Section 1234.5 - Eligibility Requirements
  - Section header
  - Introductory paragraph
  - First table (Loan Type Requirements)
  - Overlap: Section header + first paragraph

Chunk 2: Section 1234.5 - Eligibility Requirements (continued)
  - Section header (from overlap)
  - Second table (Credit Requirements)
  - Subsection 1234.5.1
  - Overlap: Subsection header
```

### 4.3 Alternative Chunking Strategies
- **Fixed-size chunking**: Not recommended (breaks context)
- **Sentence-based chunking**: Good for general content, poor for structured docs
- **Sliding window**: Potential fallback for dense sections

### 4.4 Chunk Metadata
```
{
  "chunk_id": "freddiemac_10_08_25_chunk_045",
  "document_id": "freddiemac_sellers_guide_10_08_25",
  "section": "1234.5 - Eligibility Requirements",
  "breadcrumb": "Chapter 12 > Section 1234 > 1234.5",
  "page_range": [45, 46],
  "chunk_type": "section_with_table",
  "tokens": 642,
  "overlap_with": ["chunk_044", "chunk_046"]
}
```

---

## 5. Indexing Strategy

### 5.1 Index Schema Design
**Reference**: [docs/03-indexing-strategies.md], [docs/15-index-management.md]

### 5.2 Recommended: Hybrid Search Index

#### Index Fields
```
- id (key)
- content (searchable, full text)
- content_vector (vector field, 1536 dimensions)
- section_title (searchable, facetable)
- document_name (filterable, facetable)
- document_version (filterable, sortable)
- page_numbers (collection, filterable)
- breadcrumb (searchable, filterable)
- loan_types (collection, filterable, facetable)
- regulatory_codes (collection, filterable)
- chunk_type (filterable, facetable)
- summary (searchable)
- entities (collection, searchable)
- last_updated (filterable, sortable)
```

### 5.3 Index Configuration

#### Analyzers
- **Content field**: en.microsoft (standard English)
- **Regulatory codes**: keyword (exact match)
- **Section titles**: pattern analyzer (preserve numbers/codes)

#### Vector Configuration
- **Model**: text-embedding-ada-002 (or text-embedding-3-small)
- **Dimensions**: 1536
- **Algorithm**: HNSW
- **Distance metric**: Cosine similarity

#### Semantic Configuration
**Reference**: [docs/11-semantic-search.md]
- **Title field**: section_title
- **Content fields**: content, summary
- **Keyword fields**: regulatory_codes, loan_types

### 5.4 Scoring Profile
**Reference**: [docs/14-scoring-profiles.md]

```
Scoring Profile: "ai-agent-relevance"
- Text weights:
  * section_title: 3.0
  * summary: 2.0
  * content: 1.0
  * entities: 1.5

- Freshness function:
  * Field: last_updated
  * Boost: 1.2
  * Interpolation: linear

- Tag boost:
  * Field: loan_types
  * Boost for matching user context
```

---

## 6. Search Strategy

### 6.1 Recommended: Hybrid Search with Semantic Ranking
**Reference**: [docs/10-hybrid-search.md], [docs/11-semantic-search.md]

### 6.2 Search Flow for AI Agent

#### Step 1: Query Understanding
- Extract intent from user question
- Identify key entities (loan types, criteria, etc.)
- Determine query type (factual, procedural, eligibility check)

#### Step 2: Search Execution
```
1. Hybrid Search (BM25 + Vector)
   - Generate query embedding
   - Execute keyword + vector search
   - Retrieve top 50 candidates

2. Semantic Reranking
   - Apply Azure AI Search semantic ranker
   - Boost to top 20 most relevant

3. Filter Application
   - Apply document version filters
   - Filter by loan type if specified
   - Filter by regulatory code if referenced
```

#### Step 3: Result Processing
- Extract relevant passages (captions)
- Identify supporting tables
- Collect cross-references
- Rank by relevance + recency

### 6.3 Query Patterns and Strategies

| Query Type | Example | Search Strategy |
|------------|---------|-----------------|
| Factual | "What is the maximum LTV for FHA loans?" | Hybrid + filters (loan_type=FHA) |
| Procedural | "How do I verify employment history?" | Semantic search, emphasize process sections |
| Eligibility | "Is a 580 credit score acceptable?" | Hybrid + table search, filter criteria |
| Comparison | "Difference between Fannie Mae and Freddie Mac guidelines" | Multi-index search, side-by-side results |
| Update | "What changed in the October 2025 update?" | Filter by version, recency boost |

### 6.4 Search Parameters
```
{
  "search_text": "user query",
  "vector_query": [embedding],
  "top": 20,
  "search_mode": "all",
  "query_type": "semantic",
  "semantic_configuration": "financial-docs",
  "select": ["content", "section_title", "breadcrumb", "page_numbers"],
  "filter": "document_version eq '2025-10-08'",
  "order_by": "@search.reranker_score desc"
}
```

---

## 7. Testing & Evaluation Strategy

### 7.1 Golden Dataset Creation
**Reference**: [docs/16-dataset-preparation.md]

#### 7.1.1 Dataset Size and Composition
- **Minimum viable**: 50 queries (for initial testing)
- **Recommended**: 100-200 queries (for robust evaluation)
- **Production-ready**: 300+ queries (for ongoing monitoring)

#### 7.1.2 Query Source Strategy
1. **Historical Support Tickets** (40%)
   - Extract actual questions from customer service logs
   - Anonymize and categorize by topic
   - Include failed/difficult queries

2. **Expert-Generated Queries** (30%)
   - Sales professionals write common questions
   - Subject matter experts create edge cases
   - Include regulatory compliance questions

3. **Synthetic Queries** (20%)
   - Use GPT-4 to generate variations
   - Create queries from document sections
   - Paraphrase existing queries

4. **User Testing** (10%)
   - Conduct user interviews
   - Observe real usage patterns
   - Capture natural language variations

#### 7.1.3 Test Query Categories with Examples

**1. Factual Questions (30%)**
- "What is the minimum credit score for conventional loans?"
- "What documents are required for self-employed borrowers?"
- "What is the maximum debt-to-income ratio for FHA loans?"
- "How long must a borrower wait after bankruptcy?"
- **Difficulty**: Easy to Medium
- **Expected retrieval**: Single chunk with specific fact

**2. Eligibility Checks (25%)**
- "Can I get an FHA loan with a 580 credit score?"
- "Is a 95% LTV acceptable for first-time homebuyers?"
- "Can a borrower with 12 months of employment qualify?"
- "Is a property with foundation issues eligible for financing?"
- **Difficulty**: Medium
- **Expected retrieval**: Multiple chunks (criteria + exceptions)

**3. Procedural Questions (20%)**
- "How do I submit a loan application?"
- "What is the process for manual underwriting?"
- "What are the steps for appraisal review?"
- "How do I request an exception to policy?"
- **Difficulty**: Medium to Hard
- **Expected retrieval**: Sequential chunks describing process

**4. Comparative Questions (15%)**
- "What's the difference between Fannie Mae and Freddie Mac requirements?"
- "How do conventional loans differ from FHA loans?"
- "Compare 15-year vs 30-year mortgage requirements"
- "Difference between conforming and non-conforming loans"
- **Difficulty**: Hard
- **Expected retrieval**: Chunks from multiple documents/sections

**5. Update/Change Questions (10%)**
- "What changed in the October 2025 seller's guide?"
- "Were credit score requirements updated recently?"
- "What are the new appraisal waiver guidelines?"
- "Has the LTV limit changed for conventional loans?"
- **Difficulty**: Hard
- **Expected retrieval**: Version comparison, recent chunks

#### 7.1.4 Ground Truth Annotation Schema
```json
{
  "query_id": "Q001",
  "query": "What is the maximum LTV for FHA loans?",
  "query_type": "factual",
  "difficulty": "easy",
  "intent": "retrieve_specific_criterion",
  
  "relevant_chunks": [
    {
      "chunk_id": "chunk_123",
      "relevance_score": 5,
      "relevance_type": "primary_answer",
      "excerpt": "FHA loans allow a maximum LTV of 96.5%..."
    },
    {
      "chunk_id": "chunk_456",
      "relevance_score": 3,
      "relevance_type": "supporting_context",
      "excerpt": "For streamline refinances, LTV up to 97.75%..."
    }
  ],
  
  "expected_answer": "The maximum LTV for FHA loans is 96.5% for purchase transactions and 97.75% for FHA Streamline Refinances.",
  
  "section_references": ["Section 4567.8", "Section 4567.9"],
  "page_numbers": [234, 235],
  "document_sources": ["freddiemac_sellers_guide_10_08_25"],
  
  "required_entities": ["FHA", "LTV", "96.5%"],
  "required_context": ["purchase", "refinance"],
  
  "annotation_metadata": {
    "annotator": "expert_1",
    "annotation_date": "2025-11-01",
    "confidence": "high",
    "notes": "Answer varies by transaction type"
  }
}
```

#### 7.1.5 Annotation Guidelines
- **Relevance Scoring** (1-5 scale):
  - 5: Perfect answer, complete information
  - 4: Good answer, minor missing details
  - 3: Relevant but incomplete
  - 2: Tangentially relevant
  - 1: Not relevant

- **Multiple Annotators**: Have 2-3 people annotate each query
- **Inter-Annotator Agreement**: Measure with Cohen's Kappa (target >0.7)
- **Disagreement Resolution**: Expert review for conflicting annotations

### 7.2 Evaluation Metrics
**Reference**: [docs/01-core-metrics.md], [docs/02-evaluation-frameworks.md]

#### 7.2.1 Retrieval Quality Metrics

**Precision@K** (K = 1, 3, 5, 10)
```
Precision@5 = (Number of relevant chunks in top 5) / 5

Target: ≥ 0.80 (at least 4 of top 5 are relevant)

Why it matters: Users typically look at top 5 results
How to measure: Compare retrieved chunks to annotated relevant_chunks
```

**Recall@K** (K = 10, 20, 50)
```
Recall@20 = (Number of relevant chunks found in top 20) / (Total relevant chunks)

Target: ≥ 0.90 (find 90% of all relevant chunks)

Why it matters: AI needs sufficient context to generate accurate answers
How to measure: Check if all annotated chunks appear in top 20
```

**Mean Reciprocal Rank (MRR)**
```
MRR = Average of (1 / rank of first relevant result)

Target: ≥ 0.75

Example: If first relevant result at position 2, RR = 1/2 = 0.5

Why it matters: Measures how quickly users find relevant information
How to measure: Track position of first relevant chunk across all queries
```

**Normalized Discounted Cumulative Gain (NDCG@K)**
```
NDCG@10 = DCG@10 / IDCG@10

Target: ≥ 0.75

Why it matters: Rewards relevant results appearing higher
How to measure: Use relevance scores (1-5) and compute DCG
```

#### 7.2.2 AI Agent-Specific Metrics

**Answer Accuracy** (Binary: Correct/Incorrect)
```
Answer Accuracy = (Correct answers) / (Total queries)

Target: ≥ 0.90

Evaluation method:
1. Generate answer from retrieved chunks
2. Compare to expected_answer
3. Use GPT-4 as judge (scale 1-5)
4. Manual review for borderline cases

Categories:
- Fully Correct (5): All facts accurate, complete
- Mostly Correct (4): Minor omissions
- Partially Correct (3): Some accurate facts
- Incorrect (1-2): Wrong or misleading
```

**Citation Accuracy**
```
Citation Accuracy = (Correct citations) / (Total citations)

Target: ≥ 0.95

Checks:
- Section numbers match source
- Page numbers are accurate
- Document version is correct
- No hallucinated references
```

**Context Completeness** (Coverage Score)
```
Completeness = (Retrieved key facts) / (Required key facts)

Target: ≥ 0.85

Example:
Required facts: ["max LTV", "transaction type", "exceptions"]
Retrieved facts: ["max LTV", "transaction type"]
Completeness = 2/3 = 0.67 (insufficient)
```

**Latency Metrics**
```
End-to-End Latency Breakdown:
- Query parsing: <50ms
- Search execution: <500ms
- Result processing: <200ms
- AI generation: <1000ms
Total target: <2000ms (2 seconds)

P95 latency: <3 seconds
P99 latency: <5 seconds
```

**Consistency Metrics**
```
Consistency Score = Similar queries return similar answers

Method:
1. Create query variations (synonyms, paraphrases)
2. Generate answers for each
3. Measure semantic similarity (cosine >0.85)
4. Manual review for factual consistency
```

#### 7.2.3 Business Metrics

**User Satisfaction**
- Thumbs up/down on answers
- Target: >85% positive feedback
- Track improvement over time

**Deflection Rate**
- Queries resolved without human agent
- Target: >60% deflection for common questions
- Measure escalation to human support

**Time to Resolution**
- Average time to get satisfactory answer
- Target: <30 seconds for factual questions
- Compare to baseline (without AI agent)

### 7.3 A/B Testing Framework
**Reference**: [docs/17-ab-testing-framework.md]

#### 7.3.1 Experimental Design Principles

**Baseline Establishment**
```
1. Define control variant (baseline)
   - Current production system OR
   - Simplest reasonable approach

2. Measure baseline metrics
   - Run on full golden dataset
   - Record all metrics
   - Calculate confidence intervals

3. Document configuration
   - All parameters and settings
   - Index schema version
   - Model versions
```

**Statistical Significance**
```
Sample size calculation:
- Minimum 50 queries per variant
- Use power analysis (power = 0.8, alpha = 0.05)
- Larger samples for small effect sizes

Example:
- Detect 5% improvement in Precision@5
- Need ~200 queries per variant
- Total: 400 queries for A/B test
```

**Randomization Strategy**
- Randomize query assignment to variants
- Stratify by query type (maintain distribution)
- Use same queries across all variants for fair comparison
- Control for time-of-day effects

#### 7.3.2 Test Scenario 1: Parsing Strategy Comparison

**Hypothesis**: Azure AI Document Intelligence provides better retrieval accuracy than simpler parsers

**Variants**:
```
A (Control): PyMuPDF basic text extraction
B: pdfplumber with table extraction
C (Recommended): Azure AI Document Intelligence with layout

Configuration per variant:
- Same chunking strategy
- Same embedding model
- Same search configuration
```

**Metrics to Measure**:
- Primary: Precision@5 (target: C > A by ≥10%)
- Secondary: Table-related query accuracy
- Secondary: Multi-column layout handling

**Success Criteria**:
- Variant C shows statistically significant improvement (p < 0.05)
- Improvement worth the cost (Azure DI pricing vs. accuracy gain)

**Expected Results**:
- Simple parsers: Precision@5 ~0.50-0.60
- pdfplumber: Precision@5 ~0.65-0.70
- Azure DI: Precision@5 ~0.80-0.85

#### 7.3.3 Test Scenario 2: Chunking Strategy Comparison

**Hypothesis**: Section-based semantic chunking outperforms fixed-size chunking

**Variants**:
```
A (Baseline): Fixed 512 tokens, 10% overlap
B: Fixed 800 tokens, 15% overlap
C: Sentence-based, 500-700 tokens
D (Recommended): Section-based semantic, 500-1000 tokens, preserve tables

Parameters to control:
- All use same parsed documents
- All use same embeddings
- All use same search mode
```

**Metrics to Measure**:
- Primary: Recall@20 (does chunking preserve context?)
- Primary: Context Completeness (are all facts captured?)
- Secondary: Precision@5
- Secondary: Answer Accuracy

**Success Criteria**:
- Variant D shows Recall@20 ≥ 0.90
- Variant D has Context Completeness ≥ 0.85
- Statistically significant vs. baseline (p < 0.05)

**Analysis by Query Type**:
```
Query Type        | Expected Best Variant
------------------|----------------------
Factual           | A or D (both work)
Eligibility       | D (needs complete context)
Procedural        | D (sequential sections matter)
Comparative       | D (cross-section references)
```

**Expected Results**:
```
Variant | Recall@20 | Completeness | Precision@5
--------|-----------|--------------|-------------
A       | 0.75      | 0.70         | 0.78
B       | 0.78      | 0.73         | 0.75
C       | 0.82      | 0.78         | 0.77
D       | 0.92      | 0.88         | 0.82
```

#### 7.3.4 Test Scenario 3: Search Mode Comparison

**Hypothesis**: Hybrid search with semantic reranking provides best overall performance

**Variants**:
```
A: Pure keyword (BM25 only)
B: Pure vector (embeddings only)
C: Hybrid (BM25 + vector, RRF fusion)
D (Recommended): Hybrid + semantic reranking

Search parameters:
- Keyword: search_mode="all", top=20
- Vector: k_nearest_neighbors=20
- Hybrid: top=20 from each, RRF with k=60
- Semantic: query_type="semantic", apply L2 reranking
```

**Metrics to Measure**:
- Primary: NDCG@10 (ranking quality)
- Primary: MRR (first relevant result position)
- Secondary: Precision@5
- Tertiary: Latency

**Success Criteria**:
- Variant D achieves NDCG@10 ≥ 0.75
- Variant D has best MRR across query types
- Latency remains <500ms for search

**Analysis by Query Type**:
```
Query Type        | Expected Best Mode
------------------|--------------------
Factual           | Keyword (exact terms work)
Eligibility       | Hybrid (needs both)
Procedural        | Semantic (intent matters)
Comparative       | Semantic (conceptual understanding)
```

**Expected Results**:
```
Variant | NDCG@10 | MRR   | Precision@5 | Latency
--------|---------|-------|-------------|----------
A       | 0.65    | 0.68  | 0.75        | 250ms
B       | 0.70    | 0.72  | 0.78        | 300ms
C       | 0.76    | 0.78  | 0.82        | 450ms
D       | 0.82    | 0.83  | 0.85        | 650ms
```

#### 7.3.5 Test Scenario 4: Enrichment Impact Analysis

**Hypothesis**: Full enrichment pipeline improves retrieval and answer quality

**Variants**:
```
A (Baseline): No enrichment
   - Raw chunks only
   - No entity extraction
   - No summaries

B: Entity extraction only
   - Extract: loan types, criteria, codes
   - Add as searchable fields
   - No summaries

C: Summaries only
   - Generate chunk summaries
   - No entity extraction

D (Recommended): Full enrichment
   - Entity extraction
   - Chunk summaries
   - Generated questions
   - Metadata augmentation
```

**Cost Considerations**:
```
Variant | API Calls per Chunk | Est. Cost/1000 chunks
--------|--------------------|-----------------------
A       | 0                  | $0
B       | 1 (Azure AI Lang)  | $10
C       | 1 (GPT-4 mini)     | $3
D       | 3 (Lang + 2xGPT)   | $15

ROI Calculation: Does accuracy improvement justify cost?
```

**Metrics to Measure**:
- Primary: Recall@20 (does enrichment help find chunks?)
- Primary: Answer Accuracy (does AI generate better answers?)
- Secondary: Precision@5
- Business: Cost per accurate answer

**Success Criteria**:
- Variant D shows Answer Accuracy ≥ 0.90
- Improvement over baseline ≥ 15%
- Cost justified by deflection rate improvement

**Expected Results**:
```
Variant | Recall@20 | Answer Acc | Cost/Query
--------|-----------|------------|------------
A       | 0.82      | 0.75       | $0.001
B       | 0.88      | 0.82       | $0.012
C       | 0.85      | 0.85       | $0.005
D       | 0.92      | 0.91       | $0.018
```

#### 7.3.6 Test Scenario 5: Embedding Model Comparison

**Hypothesis**: Larger/newer embedding models improve vector search quality

**Variants**:
```
A: text-embedding-ada-002 (1536 dim, $0.0001/1K tokens)
B: text-embedding-3-small (1536 dim, $0.00002/1K tokens)
C: text-embedding-3-large (3072 dim, $0.00013/1K tokens)

Considerations:
- Index size (storage cost)
- Search latency (dimension impact)
- Embedding generation cost
```

**Metrics to Measure**:
- Primary: NDCG@10 for vector/hybrid search
- Secondary: Semantic query performance
- Tertiary: Latency and cost

**Expected Trade-offs**:
```
Model         | Quality | Latency | Cost  | Storage
--------------|---------|---------|-------|----------
ada-002       | Good    | Fast    | Med   | 6 GB
3-small       | Good    | Fast    | Low   | 6 GB
3-large       | Best    | Slower  | High  | 12 GB
```

#### 7.3.7 Running A/B Tests - Step-by-Step

**Phase 1: Setup (Day 1-2)**
1. Define hypothesis and variants
2. Determine sample size (power analysis)
3. Configure systems for each variant
4. Verify configurations are correct
5. Run smoke tests (5-10 queries)

**Phase 2: Execution (Day 3-5)**
1. Randomize query assignments
2. Execute searches across all variants
3. Log all results and metrics
4. Monitor for errors/anomalies
5. Ensure balanced load

**Phase 3: Analysis (Day 6-7)**
1. Calculate metrics for each variant
2. Run statistical significance tests
   - T-test for continuous metrics
   - Chi-square for categorical
   - Bonferroni correction for multiple comparisons
3. Visualize results (box plots, error bars)
4. Analyze by query type/difficulty
5. Check for interaction effects

**Phase 4: Decision (Day 8)**
1. Compare against success criteria
2. Consider practical significance (not just statistical)
3. Evaluate cost-benefit trade-offs
4. Review edge cases and failure modes
5. Make go/no-go decision

**Phase 5: Documentation (Day 9-10)**
1. Document findings
2. Record winning configuration
3. Note lessons learned
4. Update recommendations
5. Plan next iteration

#### 7.3.8 Multivariate Testing Strategy

**When to Use**:
- Testing multiple factors simultaneously
- Understanding interaction effects
- Accelerating optimization

**Example: Combined Chunking + Search Mode Test**
```
Factors:
- Factor A: Chunking (Fixed, Semantic)
- Factor B: Search Mode (Keyword, Hybrid, Semantic)

Variants (2x3 = 6 combinations):
1. Fixed + Keyword
2. Fixed + Hybrid
3. Fixed + Semantic
4. Semantic + Keyword
5. Semantic + Hybrid
6. Semantic + Semantic

Sample size: 100 queries per variant = 600 total

Analysis:
- Main effect of chunking
- Main effect of search mode
- Interaction effect
```

**Interaction Effects to Watch**:
- Semantic chunking may work better with semantic search
- Entity enrichment may boost keyword search more than vector
- Small chunks may need higher overlap with hybrid search

### 7.4 Iterative Optimization Process

#### 7.4.1 Baseline → Optimization Loop

**Iteration 1: Establish Baseline**
```
Week 1-2:
- Implement simplest reasonable approach
- Measure all metrics on golden dataset
- Identify biggest weaknesses
- Set improvement targets

Example Baseline Results:
- Precision@5: 0.65 (target: 0.80)
- Recall@20: 0.75 (target: 0.90)
- Answer Accuracy: 0.72 (target: 0.90)

Priority: Improve Recall first (missing too much context)
```

**Iteration 2: Address Biggest Gap**
```
Week 3-4:
- Hypothesis: Chunking loses context
- Test: Section-based vs. fixed chunking
- Result: Recall improves to 0.88
- Side effect: Precision drops slightly to 0.62

Decision: Accept chunking change, address precision next
```

**Iteration 3: Fix Regression**
```
Week 5-6:
- Hypothesis: Need better ranking
- Test: Add semantic reranking
- Result: Precision improves to 0.82, Recall stays 0.88

Decision: Both targets met, move to answer quality
```

**Iteration 4: Optimize End-to-End**
```
Week 7-8:
- Hypothesis: Enrichment helps AI generate better answers
- Test: Add entity extraction + summaries
- Result: Answer Accuracy improves to 0.91

Decision: All targets met, move to production pilot
```

#### 7.4.2 Failure Mode Analysis

**Common Failure Patterns**:

1. **Missing Key Information (Low Recall)**
   - Symptom: Relevant chunks not retrieved
   - Root causes:
     * Chunks too small (context split)
     * Embedding model doesn't capture domain
     * Keywords mismatch (user vs. document language)
   - Solutions:
     * Increase chunk size/overlap
     * Add synonyms/query expansion
     * Try different embedding model

2. **Irrelevant Results on Top (Low Precision)**
   - Symptom: Top results not relevant
   - Root causes:
     * Weak scoring profile
     * Generic content ranks high
     * Missing filters
   - Solutions:
     * Tune field weights
     * Add freshness/quality boosting
     * Apply document type filters

3. **Inconsistent Answers (Low Consistency)**
   - Symptom: Similar queries get different answers
   - Root causes:
     * Result ranking varies
     * AI hallucinating
     * Contradictory information in docs
   - Solutions:
     * Improve ranking stability
     * Use lower temperature (AI generation)
     * Consolidate contradictions

4. **Slow Performance (High Latency)**
   - Symptom: >2 second response time
   - Root causes:
     * Large result set processing
     * Expensive semantic reranking
     * AI generation timeout
   - Solutions:
     * Reduce top_k
     * Cache common queries
     * Use faster models

#### 7.4.3 Edge Case Testing

**Rare Query Types** (test with 10-20 examples each):
- Negation: "loans that do NOT require..."
- Numerical ranges: "credit score between 620 and 680"
- Multiple conditions: "FHA loans with LTV >95% AND credit score <600"
- Temporal: "requirements before 2024"
- Ambiguous: "what are the limits?" (which limits?)

**Document Edge Cases**:
- Very long sections (>2000 tokens)
- Tables spanning multiple pages
- Footnotes and appendices
- Cross-references between documents
- Contradictory information

**Stress Testing**:
- 100 queries in parallel
- Queries with 50+ word length
- Queries in broken English
- Queries with typos
- Empty/nonsense queries

---

## 8. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Parse sample PDFs with Azure AI Document Intelligence
- [ ] Implement basic chunking (section-based)
- [ ] Create initial index schema
- [ ] Index sample document
- [ ] Test basic keyword search

### Phase 2: Enrichment & Vector Search (Weeks 3-4)
- [ ] Implement entity extraction pipeline
- [ ] Add summary generation
- [ ] Generate embeddings for all chunks
- [ ] Update index with vector fields
- [ ] Test vector and hybrid search

### Phase 3: Optimization (Weeks 5-6)
- [ ] Create golden test dataset (50-100 queries)
- [ ] Implement evaluation framework
- [ ] Run A/B tests on chunking strategies
- [ ] Run A/B tests on search modes
- [ ] Optimize chunk size and overlap

### Phase 4: AI Agent Integration (Weeks 7-8)
- [ ] Integrate search with AI agent (GPT-4, Claude, etc.)
- [ ] Implement result post-processing
- [ ] Add citation generation
- [ ] Test end-to-end accuracy
- [ ] Measure latency and optimize

### Phase 5: Production Readiness (Weeks 9-10)
- [ ] Implement document versioning
- [ ] Set up monitoring and alerting
- [ ] Configure auto-scaling
- [ ] Security review (RBAC, encryption)
- [ ] Load testing
- [ ] Documentation and training

---

## 9. Decision Tree (Mermaid)

```mermaid
graph TD
    A[Start: PDF Documents for AI Agent] --> B{Document Complexity?}
    
    B -->|Simple Text Only| C[Use PyMuPDF/pdfplumber]
    B -->|Complex: Tables/Multi-column| D[Use Azure AI Document Intelligence]
    
    D --> E{Enrichment Needed?}
    
    E -->|Yes: Regulatory Docs| F[Full Enrichment Pipeline]
    E -->|No: Simple Q&A| G[Basic Metadata Only]
    
    F --> F1[Entity Extraction - Azure AI Language]
    F1 --> F2[Summary Generation - GPT-4]
    F2 --> F3[Question Generation]
    F3 --> H
    
    G --> H{Chunking Strategy?}
    
    H -->|Structured Docs| I[Section-based + Semantic Chunking]
    H -->|Unstructured| J[Fixed-size with Overlap]
    
    I --> I1[Preserve section boundaries]
    I1 --> I2[Keep tables intact]
    I2 --> I3[500-800 token chunks]
    I3 --> K
    
    J --> J1[512 token chunks]
    J1 --> J2[10% overlap]
    J2 --> K
    
    K{Index Strategy?} --> L[Hybrid Search Index]
    
    L --> L1[Text fields: BM25]
    L1 --> L2[Vector field: Embeddings]
    L2 --> L3[Semantic config: Reranking]
    L3 --> M
    
    M{Search Mode?} --> N{Query Type?}
    
    N -->|Factual| N1[Hybrid + Filters]
    N -->|Procedural| N2[Semantic Search]
    N -->|Eligibility| N3[Hybrid + Table Search]
    N -->|Comparison| N4[Multi-doc Hybrid]
    
    N1 --> O
    N2 --> O
    N3 --> O
    N4 --> O
    
    O[Execute Search] --> P{Results Quality?}
    
    P -->|Poor Precision| Q1[Adjust scoring profile]
    P -->|Poor Recall| Q2[Adjust chunk size]
    P -->|Good| R
    
    Q1 --> S[Re-evaluate]
    Q2 --> S
    
    S --> P
    
    R[Testing Phase] --> T{Create Golden Dataset}
    
    T --> T1[50-100 test queries]
    T1 --> T2[Annotate ground truth]
    T2 --> U
    
    U[Run Evaluations] --> V{Metrics Meet Targets?}
    
    V -->|Precision@5 < 0.8| W1[Optimize chunking]
    V -->|NDCG@10 < 0.7| W2[Tune scoring profile]
    V -->|Latency > 2s| W3[Optimize index/cache]
    V -->|All targets met| X
    
    W1 --> U
    W2 --> U
    W3 --> U
    
    X[A/B Testing] --> Y{Compare Variants}
    
    Y --> Y1[Chunking strategies]
    Y --> Y2[Search modes]
    Y --> Y3[Enrichment levels]
    
    Y1 --> Z
    Y2 --> Z
    Y3 --> Z
    
    Z[Select Winner] --> AA[Production Deployment]
    
    AA --> AB[Monitor Performance]
    AB --> AC{Issues Detected?}
    
    AC -->|Yes| AD[Investigate & Fix]
    AC -->|No| AE[Continuous Optimization]
    
    AD --> AB
    AE --> AB
    
    style A fill:#e1f5ff
    style D fill:#90EE90
    style F fill:#FFD700
    style I fill:#90EE90
    style L fill:#90EE90
    style N1 fill:#FFB6C1
    style N2 fill:#FFB6C1
    style N3 fill:#FFB6C1
    style N4 fill:#FFB6C1
    style R fill:#FFA500
    style X fill:#FFA500
    style AA fill:#98FB98
```

---

## 10. Key Recommendations Summary

### 10.1 Critical Success Factors
1. **Use Azure AI Document Intelligence** for parsing complex PDFs
2. **Implement section-based chunking** that preserves table integrity
3. **Apply full enrichment pipeline** (entities, summaries, questions)
4. **Use hybrid search with semantic reranking** for best results
5. **Create comprehensive test dataset** before production
6. **Monitor and iterate** based on real usage patterns

### 10.2 Common Pitfalls to Avoid
- ❌ Breaking tables across chunks
- ❌ Using fixed-size chunking for structured documents
- ❌ Skipping enrichment for regulatory documents
- ❌ Relying only on keyword search for semantic queries
- ❌ Not testing with real user queries
- ❌ Ignoring document versioning

### 10.3 Performance Targets
- **Precision@5**: ≥ 0.80 (80% of top 5 results relevant)
- **Recall@20**: ≥ 0.90 (90% of relevant chunks in top 20)
- **NDCG@10**: ≥ 0.75 (good ranking quality)
- **Latency**: ≤ 2 seconds (end-to-end)
- **Answer Accuracy**: ≥ 90% (AI generates correct answer)

---

## 11. Reference Documentation

### Core Strategy Documents
- **[docs/05-azure-openai-integration.md]**: PDF parsing, chunking, enrichment
- **[docs/03-indexing-strategies.md]**: Index design patterns
- **[docs/10-hybrid-search.md]**: Hybrid search implementation
- **[docs/11-semantic-search.md]**: Semantic ranking configuration
- **[docs/26-knowledge-mining.md]**: Skillsets and enrichment pipelines

### Testing & Evaluation
- **[docs/01-core-metrics.md]**: Evaluation metrics
- **[docs/02-evaluation-frameworks.md]**: Testing methodology
- **[docs/16-dataset-preparation.md]**: Golden dataset creation
- **[docs/17-ab-testing-framework.md]**: A/B testing setup

### Advanced Topics
- **[docs/14-scoring-profiles.md]**: Custom relevance tuning
- **[docs/15-index-management.md]**: Index lifecycle management
- **[docs/23-monitoring-alerting.md]**: Production monitoring
- **[docs/24-domain-specific-solutions.md]**: Financial services patterns

---

## 12. Next Steps

1. **Review sample PDFs** to understand structure and complexity
2. **Set up Azure resources** (AI Document Intelligence, AI Search, OpenAI)
3. **Parse and analyze first document** to validate approach
4. **Create initial golden dataset** (20-30 queries) for rapid testing
5. **Implement Phase 1** (parsing + basic indexing)
6. **Iterate based on results** from evaluation metrics

---

*Last Updated: November 5, 2025*
*Status: OUTLINE - Ready for implementation*
