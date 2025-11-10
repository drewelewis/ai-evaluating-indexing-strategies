# Hybrid Search - Part 1: Real-World Case Study

## Contoso Electronics: Transforming E-commerce Search with Hybrid BM25 + Vector

### Executive Summary

**Organization**: Contoso Electronics - Leading online electronics retailer  
**Scale**: 2.1 million monthly users, 500,000 SKUs across 50 categories, 8 million searches/month  
**Challenge**: Poor relevance for semantic queries (52% precision), 15% zero-result rate, diverse query types  
**Solution**: Implemented hybrid search (BM25 + vector) with intelligent query routing  
**Timeline**: 14 weeks (planning through deployment)  
**Investment**: $428,000 (development, embeddings, infrastructure, testing)  
**Results**: Precision +31% (68% → 89%), add-to-cart +50%, revenue +$2.4M/year  
**ROI**: 561% Year 1, 2.1-month payback period  

---

## Company Background

Contoso Electronics is a major North American online electronics retailer, serving 2.1 million monthly users across web, mobile, and voice platforms. The platform offers:

- **500,000 active SKUs** (laptops, smartphones, TVs, components, accessories, smart home)
- **Product catalog**: Manufacturer specifications, user reviews, expert ratings, compatibility data
- **Content volume**: 2.8 million product documents (including variants, reviews, guides)
- **Average transaction value**: $425
- **Search-driven revenue**: 68% of transactions start with search

The platform serves four primary customer segments:
1. **General consumers** (45%) - browsing for common electronics
2. **Tech enthusiasts** (28%) - researching specific features, comparing specs
3. **Business buyers** (18%) - purchasing in bulk, specific requirements
4. **Gift shoppers** (9%) - finding products by description, not model numbers

Average search behavior:
- **4.2 searches per session**
- **12.4 minutes dwell time**
- **1.8 product pages viewed** before cart addition
- **Search-to-purchase CTR**: 12.3% (industry benchmark: 15-20%)
- **Zero-result searches**: 15% (causing 31% abandonment)

---

## The Problem: Diverse Query Types, Single Search Approach

### Problem Statement

By Q1 2024, Contoso's search experience was failing diverse query types. The existing BM25-only full-text search worked well for exact SKU lookups but struggled with semantic, descriptive, and exploratory queries that represented 70% of traffic.

**Quantified Impact:**
- **Overall precision**: 68% (measured on test set of 5,000 queries with manual judgments)
- **Semantic query precision**: 52% ("laptop for video editing", "best budget smartphone")
- **Zero-result rate**: 15% of searches (1.2M searches/month)
- **Add-to-cart rate**: 12.3% (below 15-20% industry benchmark)
- **Customer satisfaction**: 3.2/5.0 for search experience (annual survey, n=18,200)
- **Abandonment after failed search**: 31% (390K monthly visitors lost)
- **Revenue impact**: Estimated $4.8M/year in lost sales from search failures

### Query Type Analysis

**Analysis of 8 million queries over 90 days:**

| Query Type | Percentage | Example | BM25 Precision | Customer Satisfaction |
|------------|------------|---------|----------------|----------------------|
| **Exact SKU** | 12% | "Dell XPS 9520" | 98% ✓ | 4.8/5.0 ✓ |
| **Model number** | 18% | "MacBook Pro M3" | 89% ✓ | 4.2/5.0 ✓ |
| **Semantic descriptive** | 35% | "laptop for video editing" | 52% ✗ | 2.8/5.0 ✗ |
| **Feature-based** | 20% | "4K TV with 120Hz" | 74% △ | 3.4/5.0 △ |
| **Exploratory** | 10% | "gaming setup" | 45% ✗ | 2.4/5.0 ✗ |
| **Question format** | 5% | "What's the best laptop for ML?" | 38% ✗ | 2.6/5.0 ✗ |

**Key Findings:**
1. **70% of queries** (semantic + exploratory + questions) had poor relevance
2. **BM25 excelled** at exact matching (SKU, model numbers) - 98% precision
3. **BM25 failed** at semantic understanding - couldn't map "video editing" → high RAM, powerful GPU, fast SSD
4. **Zero-result queries** disproportionately affected semantic searches (24% vs 2% for SKU)

### Root Cause Analysis

**Technical Investigation (6 weeks, data science + engineering team):**

1. **Keyword Mismatch for Semantic Intent**
   - User query: "laptop for video editing"
   - BM25 matched: "laptop" + "video" + "editing"
   - Top results: Laptops with "video editing software trial" in descriptions
   - Missed intent: High-performance specs (16GB+ RAM, discrete GPU, fast NVMe SSD)
   - Ideal results ranked #32-#47: High-end workstation laptops without "video editing" in text

2. **Synonym and Vocabulary Gap**
   - User query: "budget-friendly smartphone"
   - BM25 searched: "budget" + "friendly" + "smartphone"
   - Missed synonyms: "affordable", "cheap", "value", "under $300"
   - Custom synonym lists (8,400 entries) incomplete and caused precision issues
   - False positives: "budget-friendly cases for smartphones" ranked higher than actual phones

3. **Descriptive vs Specification Language**
   - User: "laptop with long battery life"
   - BM25: Matched product descriptions mentioning "battery life"
   - Missed: Products with 12+ hour battery specs but generic descriptions
   - Problem: Natural language ("long battery life") ≠ specifications ("14 hours battery")

4. **Cross-Category Conceptual Searches**
   - User: "home office setup"
   - Expected: Monitor, desk lamp, keyboard, mouse, webcam (bundle concept)
   - BM25 results: Individual products mentioning "home office" in descriptions
   - Missed: Conceptual understanding that query spans multiple categories

5. **Misspellings and Typos**
   - 8% of queries contained misspellings: "wirless headphones", "lpatop"
   - BM25 with fuzzy matching: Too many false positives
   - BM25 without fuzzy: Zero results for misspelled queries
   - Balance problem: Either too lenient or too strict

### Business Impact Analysis

**Conversion Funnel Breakdown:**
```
8,000,000 searches/month
    ↓
    ├─ 15% (1,200,000) → Zero results → 31% abandon (372K lost visitors)
    ↓
6,800,000 with results
    ↓
    ├─ 42% poor relevance → 18% abandon (514K lost visitors)
    ↓
3,946,000 acceptable results
    ↓
    ├─ 12.3% add to cart → 485,358 cart adds
    ↓
    ├─ 65% complete purchase → 315,483 transactions
    ↓
Revenue: 315,483 × $425 = $134M/month

LOST OPPORTUNITY:
- 886K abandoned visitors × 8% recovery = 70,880 potential transactions
- 70,880 × $425 = $30.1M/year lost revenue
- Conservative estimate (4% recovery): $15M/year lost
```

---

## Previous Attempts and Limitations

### Attempt 1: Expanded Synonyms and Fuzzy Matching (Month 1-4, 2023)

**Approach:**
- Expanded synonym dictionary from 2,100 to 8,400 entries
- Enabled fuzzy matching (Levenshtein distance ≤2)
- Added product category synonyms ("laptop" ↔ "notebook" ↔ "portable computer")
- Cost: $68,000 (linguistic consultants, development, testing)

**Results:**
- **Zero-result rate**: 15% → 12% (-3pp improvement ✓)
- **Semantic query precision**: 52% → 54% (+2pp marginal)
- **False positive rate**: 4% → 11% (+7pp degradation ✗)
- **Query latency**: 45ms → 72ms (+60% slower ✗)

**Problems Encountered:**
1. **Synonym conflicts**: "Pro" expanded to "Professional" matched random products with "Professional" in unrelated context
2. **Over-matching**: Fuzzy matching caused "Canon" → "Can on" matches
3. **Maintenance burden**: 8,400 entries required constant updates for new products/terminology
4. **Precision degradation**: False positives frustrated users more than zero results

**Decision**: Kept limited synonym list (3,200 high-confidence terms), disabled fuzzy matching for most fields

### Attempt 2: BM25 Field Boosting and Scoring Profiles (Month 5-8, 2023)

**Approach:**
- Created custom scoring profiles with field boosting:
  * Title: 5x weight
  * Brand: 3x weight
  * Category: 2x weight
  * Specifications: 1.5x weight
  * Descriptions: 1x weight
- Added freshness boosting (newer products +10% score)
- Popularity boosting (sales count +15% score)
- Cost: $42,000 (data analysis, development, A/B testing)

**Results:**
- **Model/brand queries**: 89% → 92% precision (+3pp ✓)
- **Semantic queries**: 52% → 56% precision (+4pp marginal)
- **Overall precision**: 68% → 71% (+3pp modest)
- **Add-to-cart rate**: 12.3% → 12.9% (+0.6pp)

**Problems Encountered:**
1. **Overfitting**: Heavily promoted popular products even when not relevant
2. **New product problem**: Excellent niche products ranked poorly due to low sales history
3. **Still keyword-bound**: Couldn't overcome fundamental keyword mismatch for semantic queries
4. **Tuning complexity**: Required constant re-tuning as catalog evolved

**Decision**: Kept moderate field boosting (title 2x, brand 1.5x), abandoned popularity boosting

### Attempt 3: Query Expansion with Product Taxonomy (Month 9-12, 2023)

**Approach:**
- Built product taxonomy: 12 main categories → 87 subcategories → 1,200 product types
- Query expansion: Map query to category → expand with category-specific terms
- Example: "video editing laptop" → expand with "RAM 16GB", "GPU", "SSD NVMe"
- Cost: $95,000 (taxonomy construction, ML model training, integration)

**Results:**
- **Category-specific queries**: 74% → 79% precision (+5pp ✓)
- **Semantic queries**: 52% → 61% precision (+9pp improvement ✓)
- **Ambiguous queries**: Degraded (expanded in wrong category)
- **Overall precision**: 68% → 73% (+5pp)

**Problems Encountered:**
1. **Category misclassification**: 18% of queries mapped to wrong category
2. **Over-expansion**: Generated 20-40 terms, diluted original query intent
3. **Brittle mappings**: Required manual rules for 1,200 product types
4. **Couldn't bridge concept gap**: Still keyword-based, just with more keywords

**Decision**: Recognized fundamental limitation - BM25 alone couldn't understand semantic intent, no amount of expansion would solve it

### Cumulative Investment in BM25 Optimization: $205,000

**Total improvement from baseline**: 68% → 73% precision (+5pp, 7% relative improvement)  
**ROI**: Modest - $205K for 7% improvement, but hit BM25 ceiling

---

## The Solution: Hybrid Search (BM25 + Vector Embeddings)

### Strategic Decision: Why Hybrid?

After recognizing BM25's fundamental limitations, the team evaluated three approaches:

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Vector-only** | Strong semantic understanding | Poor exact matching, high embedding costs | ✗ Rejected |
| **Hybrid (BM25 + Vector)** | Best of both worlds, proven in research | Complex implementation, moderate costs | ✓ **Selected** |
| **Semantic ranking** | Easy to add on existing BM25 | Requires Azure Standard tier ($500/month), limited control | △ Future consideration |

**Why Hybrid Won:**
1. **Complementary strengths**: BM25 (98% SKU precision) + Vector (semantic understanding)
2. **Research backing**: Microsoft Research showed 15-30% improvement in benchmarks
3. **Industry validation**: Amazon, Google, Bing all use hybrid approaches in production
4. **Risk mitigation**: If vector fails, BM25 provides fallback
5. **Query adaptability**: Can weight BM25 vs vector based on query type

### Implementation Overview

**Timeline**: 14 weeks (October 2023 - January 2024)

**Phase 1: Design and POC** (Week 1-2)
- Architecture design: Parallel BM25 + vector with RRF fusion
- Embedding model selection: text-embedding-3-large (3,072 dimensions)
- POC with 10,000 products, 500 test queries
- POC results: 73% → 84% precision (+11pp), validated approach

**Phase 2: Data Preparation and Embedding Generation** (Week 3-6)
- **Field strategy**: 
  * Title embeddings (concise, high-weight): For quick semantic matching
  * Content embeddings (detailed): Specifications + descriptions + reviews
- **Embedding generation**:
  * 500,000 products × 2 embeddings (title + content) = 1,000,000 embeddings
  * Batch processing: 10,000 products/day for 50 days
  * Cost: $62.50 (500M tokens × $0.13/1M for text-embedding-3-large)
- **Quality validation**: Manual review of 1,000 random embeddings, similarity spot-checks

**Phase 3: Index Design and Migration** (Week 7-9)
- **New index schema**:
  ```json
  {
    "fields": [
      {"name": "id", "type": "Edm.String", "key": true},
      {"name": "title", "type": "Edm.String", "searchable": true},
      {"name": "description", "type": "Edm.String", "searchable": true},
      {"name": "specifications", "type": "Edm.String", "searchable": true},
      {"name": "brand", "type": "Edm.String", "searchable": true, "filterable": true},
      {"name": "category", "type": "Edm.String", "filterable": true},
      {"name": "price", "type": "Edm.Double", "filterable": true, "sortable": true},
      {"name": "titleVector", "type": "Collection(Edm.Single)", "dimensions": 3072, "vectorSearchProfile": "title-profile"},
      {"name": "contentVector", "type": "Collection(Edm.Single)", "dimensions": 3072, "vectorSearchProfile": "content-profile"}
    ]
  }
  ```
- **Vector search profiles**:
  * Algorithm: HNSW (Hierarchical Navigable Small World)
  * M: 4 (connections per layer)
  * efConstruction: 400
  * efSearch: 500
- **Index size**: 87GB BM25 data + 46GB vectors = 133GB total
- **Azure tier upgrade**: S2 → S3 (need storage + compute for vectors)
- **Migration**: Blue-green deployment, zero downtime

**Phase 4: Query Routing Logic** (Week 10-11)
- **Query classifier**:
  ```python
  def classify_query(query):
      # Exact SKU pattern (ABC-1234 format)
      if re.match(r'^[A-Z]{2,4}-\d{4,6}$', query):
          return 'exact_match', {'alpha_text': 1.0, 'alpha_vector': 0.0}
      
      # Model/brand with numbers
      if re.match(r'[A-Za-z]+\s+[A-Z0-9]{3,}', query):
          return 'model_search', {'alpha_text': 0.7, 'alpha_vector': 0.3}
      
      # Question format
      if query.endswith('?') or query.lower().startswith(('what', 'which', 'how')):
          return 'semantic_question', {'alpha_text': 0.3, 'alpha_vector': 0.7}
      
      # Descriptive (3+ words, no numbers)
      if len(query.split()) >= 3 and not re.search(r'\d', query):
          return 'semantic_descriptive', {'alpha_text': 0.4, 'alpha_vector': 0.6}
      
      # Default balanced
      return 'general', {'alpha_text': 0.5, 'alpha_vector': 0.5}
  ```
- **Routing decision matrix**: 5 query types with optimal weight configs
- **Validation**: 87% classification accuracy on 2,000 manually labeled queries

**Phase 5: RRF Fusion Implementation** (Week 12)
- **RRF parameters**:
  * Constant k: 60 (Azure AI Search default)
  * Over-fetch: k_nearest_neighbors = 50 (for vector) vs top = 10 (final results)
- **Weighted RRF formula**:
  ```
  combined_score = α_text × RRF_text + α_vector × RRF_vector
  ```
- **Performance optimization**: Parallel execution of BM25 and vector searches

**Phase 6: A/B Testing and Tuning** (Week 13-14)
- **Test design**:
  * Duration: 14 days
  * Traffic split: 50% control (BM25-only) / 50% treatment (hybrid)
  * Sample size: 4 million searches (2M per variant)
  * Stratified by query type: Ensured representative distribution
- **Primary metrics**: Precision, add-to-cart rate, revenue per search
- **Secondary metrics**: Latency, zero-result rate, user satisfaction
- **Guardrails**: Latency p95 < 200ms, error rate < 0.1%

---

## Results and Business Impact

### A/B Test Results (14 days, 4M searches)

**Overall Metrics:**

| Metric | Control (BM25) | Treatment (Hybrid) | Improvement | p-value |
|--------|----------------|-------------------|-------------|---------|
| **Precision** | 73.2% | 89.1% | +15.9pp (+22%) | <0.001 ✓✓✓ |
| **Zero-result rate** | 12.4% | 4.8% | -7.6pp (-61%) | <0.001 ✓✓✓ |
| **Add-to-cart rate** | 12.9% | 18.7% | +5.8pp (+45%) | <0.001 ✓✓✓ |
| **CTR@10** | 48.2% | 64.5% | +16.3pp (+34%) | <0.001 ✓✓✓ |
| **Revenue/search** | $16.75 | $24.32 | +$7.57 (+45%) | <0.001 ✓✓✓ |
| **Dwell time** | 12.4 min | 15.8 min | +3.4 min (+27%) | <0.001 ✓✓✓ |
| **Latency p95** | 68ms | 118ms | +50ms | — |
| **User satisfaction** | 3.4/5.0 | 4.2/5.0 | +0.8 | <0.001 ✓✓✓ |

**Query Type Breakdown:**

| Query Type | Traffic % | Control Precision | Hybrid Precision | Improvement |
|------------|-----------|-------------------|------------------|-------------|
| Exact SKU | 12% | 98.2% | 97.8% | -0.4pp (stable ✓) |
| Model number | 18% | 91.5% | 93.2% | +1.7pp (slight ✓) |
| **Semantic descriptive** | 35% | 61.4% | **86.7%** | **+25.3pp (41% lift ✓✓✓)** |
| Feature-based | 20% | 78.9% | 89.4% | +10.5pp (13% lift ✓✓) |
| Exploratory | 10% | 58.2% | 84.1% | +25.9pp (45% lift ✓✓✓) |
| Question format | 5% | 52.8% | 82.3% | +29.5pp (56% lift ✓✓✓) |

**Key Findings:**
1. **Biggest wins**: Semantic queries (+25-30pp precision improvement)
2. **No degradation**: Exact SKU/model queries maintained 97-98% precision
3. **Latency acceptable**: +50ms (68ms → 118ms) within budget (<200ms)
4. **Universal improvement**: All query types benefited or stayed stable

### Post-Deployment Performance (3 months)

**Search Quality:**
- Overall precision: 73% → 89% (+16pp, 22% improvement)
- Zero-result rate: 12% → 5% (-7pp, 58% reduction)
- User satisfaction: 3.4/5.0 → 4.2/5.0 (+0.8 stars)
- Repeat search rate: 4.2 → 2.8 per session (-33%, users find results faster)

**Business Metrics:**
- Add-to-cart rate: 12.9% → 18.7% (+5.8pp, 45% improvement)
- Average order value: $425 → $448 (+$23, 5.4% increase - better product discovery)
- Monthly transactions: 315K → 472K (+157K, 50% increase)
- Revenue per search: $16.75 → $24.32 (+45%)

**Operational:**
- Query latency p95: 68ms → 118ms (+50ms, within SLA)
- Query latency p99: 95ms → 187ms (+92ms)
- Abandonment after failed search: 31% → 14% (-17pp, 55% reduction)
- Customer support tickets (search-related): 2,400/month → 980/month (-59%)

**Segment Analysis (by query type):**
- Semantic query satisfaction: 2.8/5.0 → 4.5/5.0 (+1.7 stars, biggest improvement)
- Exact SKU satisfaction: 4.8/5.0 → 4.7/5.0 (stable, no degradation)
- Feature-based satisfaction: 3.4/5.0 → 4.3/5.0 (+0.9 stars)

---

## ROI Calculation

### Investment Breakdown

**One-Time Costs:**
```
1. Development (14 weeks)
   - Backend engineering (2 FTEs × 14 weeks × $4,000/week): $112,000
   - Data science/ML (1 FTE × 14 weeks × $4,500/week): $63,000
   - QA/Testing (0.5 FTE × 6 weeks × $3,500/week): $10,500
   - Project management: $15,000
   Subtotal: $200,500

2. Embedding Generation (Initial)
   - text-embedding-3-large: 500M tokens × $0.13/1M: $65
   - Compute for processing: $2,500
   Subtotal: $2,565

3. Infrastructure Setup
   - Azure AI Search tier upgrade (S2 → S3): $0 (monthly, see below)
   - Index migration tooling: $8,000
   - Monitoring/observability setup: $5,000
   Subtotal: $13,000

4. Testing and Validation
   - A/B test infrastructure: $12,000
   - Manual relevance evaluation (contractors): $18,000
   Subtotal: $30,000

Total One-Time Investment: $246,065
```

**Recurring Costs (Monthly):**
```
1. Azure AI Search
   - Previous (S2): $1,000/month
   - New (S3): $2,000/month
   - Increase: $1,000/month

2. Embedding Costs
   - New products: 2,000/month × 1,000 tokens × $0.13/1M: $0.26
   - Query embeddings (cached 80%): 1.6M/month × 10 tokens × $0.13/1M: $2.08
   - Subtotal: $2.34/month (~$3 rounded)

3. Monitoring and Maintenance
   - Application Insights, dashboards: $150/month
   - Ongoing tuning (0.1 FTE): $1,600/month
   Subtotal: $1,750/month

Total Monthly Recurring: $2,753/month
```

**Total Year 1 Cost**: $246,065 (one-time) + $2,753 × 12 (recurring) = $279,101

### Revenue Impact

**Monthly Revenue Increase:**
```
Before Hybrid:
- 8M searches × 12.9% add-to-cart × 65% completion × $425 AOV = $28.3M/month

After Hybrid:
- 8M searches × 18.7% add-to-cart × 68% completion × $448 AOV = $45.4M/month

Monthly Revenue Increase: $45.4M - $28.3M = $17.1M/month
Annual Revenue Increase: $17.1M × 12 = $205.2M/year
```

**Conservative Attribution (30% to search improvements):**
- Attributed annual revenue increase: $205.2M × 30% = **$61.6M/year**

### ROI Calculation

```
Year 1 ROI:
- Net benefit: $61,600,000 - $279,101 = $61,320,899
- ROI: ($61,320,899 / $279,101) × 100 = 21,965%
- Payback period: $279,101 / ($61,600,000/12) = 0.065 months = 2.0 days

Conservative ROI (10% attribution):
- Attributed revenue: $205.2M × 10% = $20.52M/year
- ROI: ($20,520,000 - $279,101) / $279,101 × 100 = 7,251%
- Payback period: $279,101 / ($20,520,000/12) = 0.163 months = 4.9 days

Using 30% attribution (reasonable for major search overhaul):
- ROI: 21,965% Year 1
- Payback: 2.0 days
```

**Even More Conservative (Cost-Benefit excluding revenue):**
```
Measurable cost savings:
1. Customer support reduction: 1,420 tickets/month × $45/ticket = $63,900/month
2. Reduced abandoned cart recovery efforts: $12,000/month
3. Lower customer acquisition cost (higher satisfaction → retention): $8,000/month
Total monthly savings: $83,900
Annual savings: $1,006,800

Cost-only ROI: ($1,006,800 - $279,101) / $279,101 = 261%
Payback: 3.3 months (from cost savings alone)
```

---

## Key Success Factors

### What Worked

1. **Hybrid Approach (Not Vector-Only)**
   - Preserved BM25's 98% precision on exact matches
   - Added semantic understanding for 70% of queries
   - Risk mitigation: If one approach fails, other provides fallback

2. **Query Routing Intelligence**
   - Adaptive weighting based on query classification (87% accuracy)
   - Exact SKUs → 100% BM25 (no vector overhead)
   - Semantic queries → 60-70% vector weighting
   - Prevented vector from degrading exact match performance

3. **Two-Vector Strategy (Title + Content)**
   - Title vector (concise): Fast semantic matching on product names
   - Content vector (detailed): Deep semantic matching on specs/reviews
   - Title had 2x weight in fusion (more important signal)

4. **RRF Fusion (Not Score Normalization)**
   - Rank-based fusion avoided BM25 vs vector score scaling issues
   - k=60 (default) worked well without tuning
   - Over-fetching (k=50 for vectors) ensured good fusion quality

5. **Parallel Execution**
   - BM25 and vector searches ran concurrently
   - Latency = max(BM25, vector) + fusion_overhead (5-10ms)
   - Kept p95 latency under 120ms (acceptable)

6. **Comprehensive A/B Testing**
   - 14-day test with 4M searches ensured statistical significance
   - Stratified sampling validated performance across all query types
   - Guardrails prevented deployment if exact match degraded

7. **Incremental Rollout**
   - 10% → 25% → 50% → 100% over 4 weeks
   - Monitored metrics at each stage, ready to rollback
   - Caught edge cases (unusual product names) before full deployment

---

## Lessons Learned

### What Worked Well

1. **Start with POC (10K products, 500 queries)**
   - Validated 11pp improvement before full investment
   - Identified optimal embedding model (text-embedding-3-large > ada-002)
   - De-risked $280K investment with $8K POC

2. **Query classification 87% accuracy was sufficient**
   - Initially targeted 95% but found diminishing returns
   - 87% achieved 95% of maximum benefit at 40% of development cost
   - Fallback to balanced hybrid (0.5/0.5) worked well for misclassifications

3. **Two-vector strategy (title + content)**
   - Title vector alone: 83% precision (good but not great)
   - Content vector alone: 85% precision
   - **Both vectors**: 89% precision (synergy effect)
   - Worth the 2x storage cost

### What Didn't Work

1. **Initial RRF k tuning**
   - Spent 2 weeks testing k values (30, 45, 60, 90, 120)
   - Differences were minimal (87.2% vs 89.1% for k=45 vs k=60)
   - **Learning**: Default k=60 is good, not worth extensive tuning

2. **Per-category embedding models**
   - Tried training category-specific models for laptops, TVs, phones
   - Hoped for better precision within categories
   - Results: Marginal 1-2pp improvement, not worth complexity
   - **Learning**: General-purpose embedding models (text-embedding-3-large) work well

3. **Complex query routing with ML classifier**
   - Built ML classifier (logistic regression) for query type prediction
   - Accuracy: 91% vs 87% for rule-based
   - **Problem**: Required training data maintenance, model retraining
   - **Learning**: Simple rule-based routing (87% accuracy) better trade-off

### Recommendations for Others

1. **Do a POC first** - Don't commit $200K+ without validation
2. **Hybrid > pure vector** - Unless you have zero exact-match queries
3. **Query routing is valuable** - Even simple rules (87% accuracy) provide big lift
4. **Two-vector strategy** - If budget allows, use title + content embeddings
5. **Don't over-tune RRF** - Default k=60 works well
6. **Monitor by query type** - Aggregate metrics hide important segment performance
7. **Latency budget** - Plan for +50-100ms, communicate with stakeholders

---

## Conclusion

Contoso Electronics' hybrid search implementation demonstrates that combining BM25 and vector search delivers superior results compared to either approach alone:

- **22% precision improvement** (73% → 89%)
- **45% add-to-cart increase** (12.9% → 18.7%)
- **$61.6M annual revenue impact** (30% attribution)
- **21,965% ROI Year 1** with 2-day payback period

The key insight: **Different query types need different search approaches**. Hybrid search with intelligent query routing provides the best of both worlds - precision for exact matches and semantic understanding for descriptive queries.

**Total Word Count**: ~4,650 words
