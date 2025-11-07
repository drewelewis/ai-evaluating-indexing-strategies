# Core Metrics for Search Relevance and Accuracy

**Purpose**: This document provides a comprehensive guide to measuring search system performance through quantitative metrics. Understanding these metrics is essential for evaluating different indexing strategies, conducting A/B tests, and demonstrating the business value of search improvements.

**Who Should Read This**: Search engineers, data scientists, product managers, and anyone responsible for evaluating or improving search quality.

**Reading Time**: 15-20 minutes

---

## Why Metrics Matter

Before implementing any indexing strategy—whether full-text search with BM25, vector search with embeddings, or hybrid approaches—you need objective ways to measure performance. Without metrics, you're flying blind:

- **Without metrics**: "The new chunking strategy seems better"
- **With metrics**: "Chunking increased Precision@5 from 0.68 to 0.82 (+20.6%), with statistical significance p<0.01"

Metrics enable you to:
1. **Compare approaches objectively** - BM25 vs vector vs hybrid search
2. **Track improvements over time** - Is your system getting better?
3. **Detect regressions** - Did the latest deployment break something?
4. **Communicate with stakeholders** - Show ROI of search investments
5. **Prioritize optimizations** - Focus on areas with biggest impact

### The Precision-Recall Tradeoff

One of the fundamental challenges in search is balancing **precision** (accuracy) and **recall** (completeness). Understanding this tradeoff is critical:

**Real-world example** (E-commerce product search):
- **High precision, low recall**: Showing only exact matches for "red shoes" returns 5 perfect products but misses 20 other relevant ones (users complain "not enough options")
- **High recall, low precision**: Showing anything related to "shoes" returns 50 products including brown boots and shoe polish (users complain "too much irrelevant junk")
- **Balanced approach**: Showing 15-20 well-ranked red shoes with some related items (optimal user experience)

Different applications require different balances:
- **Medical diagnosis systems**: Favor recall (can't afford to miss relevant diagnoses)
- **Legal document review**: Favor recall (missing key evidence is costly)
- **E-commerce search**: Balance both (need relevant products without overwhelming users)
- **Q&A systems**: Favor precision (one perfect answer beats many mediocre ones)

---

## 📊 Primary Relevance Metrics

### 1. Precision & Recall
The foundational metrics for search quality assessment. These metrics answer two fundamental questions: "Are the results we return actually relevant?" (precision) and "Are we finding all the relevant results that exist?" (recall).

#### Precision
**Definition**: Of all the results your system returned, what percentage were actually relevant?

```
Precision = Relevant results retrieved ÷ Total results retrieved
```
- **Purpose**: Measures accuracy by focusing on reducing irrelevant results
- **Range**: 0.0 to 1.0 (higher is better)
- **When to use**: When false positives are costly (user annoyance, wasted time)
- **Target**: > 0.7 for most applications, > 0.9 for high-stakes scenarios

**Real-world interpretation**:
- **Precision = 0.90**: 90% of returned results are relevant (10% noise)
- **Precision = 0.50**: Half your results are junk (poor user experience)
- **Precision = 0.30**: Users spend more time filtering than finding (system failure)

**When precision matters most**:
- Customer-facing search (low tolerance for irrelevant results)
- Premium/paid search services (users expect quality)
- Mobile applications (limited screen space for results)
- Time-sensitive tasks (emergency medical information)

**Example calculation**:
```
Query: "mortgage insurance requirements"
Top 10 results: 7 relevant, 3 irrelevant
Precision = 7/10 = 0.70
```

#### Recall
**Definition**: Of all the relevant documents that exist in your corpus, what percentage did you actually find?

```
Recall = Relevant results retrieved ÷ Total relevant results in corpus
```
- **Purpose**: Measures completeness of retrieval—are you finding everything that matters?
- **Range**: 0.0 to 1.0 (higher is better)
- **When to use**: When missing relevant results is costly (legal discovery, medical diagnosis)
- **Target**: > 0.8 for comprehensive search scenarios, > 0.95 for critical applications

**Real-world interpretation**:
- **Recall = 0.95**: You found 95% of all relevant documents (acceptable miss rate)
- **Recall = 0.60**: You missed 40% of relevant documents (significant blind spots)
- **Recall = 0.30**: You found only 30% of what exists (major system failure)

**When recall matters most**:
- Legal e-discovery (missing evidence can lose cases)
- Medical literature search (missing studies can harm patients)
- Competitive intelligence (missing competitor info is strategic risk)
- Regulatory compliance (missing relevant regulations creates liability)

**Example calculation**:
```
Query: "mortgage insurance requirements"
Total relevant documents in corpus: 20
Documents retrieved: 15 (including some irrelevant ones)
Relevant documents retrieved: 15
Recall = 15/20 = 0.75 (missed 5 relevant documents)
```

**The Precision-Recall Tension**:
```
Strategy A: Return 5 results, all relevant
├── Precision: 5/5 = 1.00 (perfect!)
└── Recall: 5/20 = 0.25 (terrible - missed 75%)

Strategy B: Return 50 results, 18 relevant, 32 irrelevant
├── Precision: 18/50 = 0.36 (poor user experience)
└── Recall: 18/20 = 0.90 (good coverage)

Strategy C: Return 20 results, 16 relevant, 4 irrelevant
├── Precision: 16/20 = 0.80 (good)
└── Recall: 16/20 = 0.80 (good)
└── This is typically the best balanced approach
```

#### F1-Score
**Definition**: The harmonic mean of precision and recall, providing a single metric that balances both concerns.

```
F1 = 2 × (Precision × Recall) ÷ (Precision + Recall)
```
- **Purpose**: Harmonic mean balancing precision and recall into a single number
- **Best for**: Overall system performance assessment when both metrics matter equally
- **Why harmonic mean**: Penalizes extreme imbalances (unlike arithmetic mean)

**Why use F1 instead of average**?
```
Scenario 1: Precision=1.0, Recall=0.1
├── Arithmetic mean: (1.0 + 0.1)/2 = 0.55 (misleadingly decent)
└── F1 Score: 2*(1.0*0.1)/(1.0+0.1) = 0.18 (accurately reflects poor recall)

Scenario 2: Precision=0.8, Recall=0.8
├── Arithmetic mean: (0.8 + 0.8)/2 = 0.80
└── F1 Score: 2*(0.8*0.8)/(0.8+0.8) = 0.80 (same when balanced)

Scenario 3: Precision=0.9, Recall=0.6
├── Arithmetic mean: (0.9 + 0.6)/2 = 0.75
└── F1 Score: 2*(0.9*0.6)/(0.9+0.6) = 0.72 (penalizes imbalance)
```

**When to use F1**:
- Comparing systems with different precision-recall tradeoffs
- Single-number performance reports for stakeholders
- Optimization objectives when both metrics are equally important

**F1 Score interpretation**:
- **F1 > 0.80**: Excellent balanced performance
- **F1 = 0.60-0.80**: Good, room for improvement
- **F1 < 0.60**: System needs significant work

### 2. Precision@K
**Definition**: Evaluates ranking quality by measuring what percentage of the top K results are actually relevant. This is one of the most practical metrics because it reflects what users actually see.

```
Precision@K = Relevant results in top K ÷ K
```

**Why Precision@K matters more than overall precision**:
Most users only look at the first few results. If your system returns 100 results with 80 relevant (precision=0.80) but the first 10 are all irrelevant (Precision@10=0.0), users will think your search is broken. Precision@K captures what users actually experience.

**Common K values and use cases:**
- **Precision@1**: Critical for single-answer queries (Q&A, navigation, voice assistants)
  - Example: "What is the maximum LTV ratio?" — First result must be correct
  - Target: > 0.80 for Q&A systems, > 0.90 for voice assistants
  
- **Precision@5**: Standard for search result pages (desktop display)
  - Example: E-commerce product listings, document search
  - Reflects typical "above the fold" results
  - Target: > 0.70 for general search, > 0.80 for premium services
  
- **Precision@10**: Full first page evaluation (mobile scrolling)
  - Example: Blog search, article discovery
  - Captures what users see before pagination
  - Target: > 0.60 for exploratory search, > 0.70 for focused search
  
- **Precision@20**: Extended result evaluation (deep exploration)
  - Example: Research tasks, comprehensive discovery
  - Used for recall-heavy applications
  - Target: > 0.50 for research tools

**Real-world comparison**:
```
Query: "FHA loan credit score requirements"

System A (BM25 only):
├── Top 5 results: 3 relevant → Precision@5 = 0.60
└── Top 10 results: 5 relevant → Precision@10 = 0.50

System B (Hybrid search):
├── Top 5 results: 5 relevant → Precision@5 = 1.00
└── Top 10 results: 8 relevant → Precision@10 = 0.80

Decision: Hybrid search provides significantly better user experience
```

**Target values by application type:**

| Application Type | Precision@1 | Precision@5 | Precision@10 |
|-----------------|-------------|-------------|--------------|
| Voice assistants | > 0.90 | - | - |
| Q&A systems | > 0.80 | > 0.70 | - |
| E-commerce search | > 0.70 | > 0.70 | > 0.60 |
| Document retrieval | > 0.60 | > 0.60 | > 0.60 |
| Exploratory search | > 0.50 | > 0.60 | > 0.60 |
| Research tools | - | > 0.50 | > 0.60 |

### 3. Mean Average Precision (MAP)
**Definition**: Considers not just whether relevant results are in the top K, but WHERE they're ranked. MAP rewards systems that put the most relevant results first.

```
MAP = (1/Q) × Σ(Average Precision for each query)
```

Where Average Precision for a single query:
```
AP = (1/R) × Σ(Precision@k × rel(k))
```
- **R**: Total relevant documents for the query
- **rel(k)**: 1 if item at rank k is relevant, 0 otherwise

**Why MAP is better than Precision@K**:

Precision@K only looks at a fixed cutoff, but MAP considers the entire ranking:

**Why MAP is better than Precision@K**:

Precision@K only looks at a fixed cutoff, but MAP considers the entire ranking:

```
Query: "conventional loan down payment requirements"

Ranking A: Relevant at positions [1, 2, 3, 8, 9]
├── Precision@5 = 3/5 = 0.60
├── AP = (1/5) × [(1/1 × 1) + (2/2 × 1) + (3/3 × 1) + (4/8 × 1) + (5/9 × 1)]
├── AP = 0.2 × [1.0 + 1.0 + 1.0 + 0.50 + 0.56] = 0.812
└── MAP rewards early relevant results

Ranking B: Relevant at positions [3, 7, 8, 9, 10]
├── Precision@5 = 1/5 = 0.20 (much worse)
├── AP = (1/5) × [(1/3 × 1) + (2/7 × 1) + (3/8 × 1) + (4/9 × 1) + (5/10 × 1)]
├── AP = 0.2 × [0.33 + 0.29 + 0.38 + 0.44 + 0.50] = 0.388
└── MAP penalizes late relevant results

Insight: Ranking A is much better (0.812 vs 0.388) even though both have 5 relevant docs
```

**When to use MAP vs Precision@K**:
- **Use MAP when**: Ranking order matters significantly (news search, recommendation systems, document retrieval)
- **Use Precision@K when**: You only care about a fixed number of results (top 5 products, top 10 articles)
- **Use both when**: You need comprehensive evaluation (most production search systems)

**MAP interpretation by score**:
- **MAP > 0.80**: Excellent—relevant results consistently appear early
- **MAP = 0.60-0.80**: Good—most relevant results in top positions
- **MAP = 0.40-0.60**: Moderate—relevant results scattered throughout
- **MAP < 0.40**: Poor—relevant results often buried deep

**Characteristics:**
- **Range**: 0.0 to 1.0
- **Strengths**: Rewards systems that rank relevant results higher
- **Target**: > 0.6 for general search, > 0.7 for domain-specific, > 0.8 for Q&A systems
- **Best for**: Comprehensive ranking evaluation

**Common pitfall**: MAP can be inflated if you have many queries with few relevant documents. Always report MAP alongside Recall to get full picture.

### 4. Mean Reciprocal Rank (MRR)
**Definition**: Focuses exclusively on the rank of the FIRST relevant result. If the first relevant result is at position 1, you get full credit (1.0). If it's at position 2, you get half credit (0.5), and so on.

```
MRR = (1/Q) × Σ(1/rank of first relevant result)
```

**Why MRR matters for specific scenarios**:

MRR is laser-focused on a single question: "How quickly does the user find their first relevant result?" This matters when:
- Users typically only need one answer (Q&A systems)
- The task is navigational (finding a specific page/document)
- Users won't scroll if first results are irrelevant

**Real-world comparison**:
```
Q&A System Evaluation (20 queries)

System A (Keyword search):
├── Query 1: First relevant at position 1 → 1/1 = 1.00
├── Query 2: First relevant at position 3 → 1/3 = 0.33
├── Query 3: First relevant at position 2 → 1/2 = 0.50
├── ... (17 more queries)
└── MRR = average = 0.65

System B (Semantic search):
├── Query 1: First relevant at position 1 → 1/1 = 1.00
├── Query 2: First relevant at position 1 → 1/1 = 1.00
├── Query 3: First relevant at position 2 → 1/2 = 0.50
├── ... (17 more queries)
└── MRR = average = 0.85

Conclusion: System B gets users to relevant results faster (0.85 vs 0.65)
```

**MRR vs MAP comparison**:
```
Query: "What is PMI?"

Ranking A: Relevant at [1, 5, 7]
├── MRR = 1/1 = 1.00 (perfect! first result is relevant)
└── MAP = (1/3) × [1/1 + 2/5 + 3/7] = 0.61 (considers all relevant results)

Ranking B: Relevant at [3, 4, 5]
├── MRR = 1/3 = 0.33 (first relevant result at position 3)
└── MAP = (1/3) × [1/3 + 2/4 + 3/5] = 0.64 (MAP actually higher!)

Insight: MRR says Ranking A is better (1.00 vs 0.33) because first result matters most
         MAP says Ranking B is better (0.64 vs 0.61) because it clusters relevant results
```

**When to prefer MRR**:
- **Q&A systems**: "What is the maximum LTV?" — User needs one answer fast
- **Navigation queries**: "Find the underwriting guidelines PDF" — User wants specific document
- **Voice assistants**: Can only speak one answer
- **Auto-complete**: Users pick from top suggestions

**When to prefer MAP**:
- **Exploratory search**: Users want to see multiple perspectives
- **Research tasks**: Users will review many results
- **Product search**: Users compare multiple options

**Use cases and targets:**
| Use Case | Target MRR | Reasoning |
|----------|-----------|-----------|
| Q&A systems | > 0.80 | First result must be right most of the time |
| Navigation queries | > 0.85 | Users know exactly what they want |
| Auto-complete | > 0.75 | Top 2-3 suggestions should work |
| Voice assistants | > 0.90 | No second chances in voice interaction |
| Document search | > 0.60 | Users more tolerant, will scroll |

**Common MRR pitfall**:
MRR ignores everything after the first relevant result. A system with MRR=1.0 might have terrible recall:
```
Query: "FHA loan requirements"
Results: [relevant, irrelevant, irrelevant, irrelevant, ...]
├── MRR = 1.0 (perfect! first result is relevant)
└── Recall = 0.10 (terrible! missed 90% of relevant documents)

Always report MRR alongside Recall or MAP for complete picture.
```

### 5. Normalized Discounted Cumulative Gain (nDCG)
**Definition**: The most sophisticated ranking metric—accounts for both position AND graded relevance (not just binary relevant/irrelevant). Essential when relevance isn't black-and-white.

```
DCG@k = Σ(rel_i / log2(i + 1)) for i = 1 to k
nDCG@k = DCG@k / IDCG@k
```

Where:
- **rel_i**: Relevance score at position i (typically 0-3 scale)
- **IDCG@k**: Ideal DCG (perfect ranking of same items)
- **log2(i + 1)**: Position discount (reduces value of results further down)

**Why graded relevance matters**:

Most search scenarios aren't binary (relevant vs irrelevant). Real users see shades of gray:

```
Query: "mortgage insurance requirements for conventional loans"

Document A: Comprehensive guide to MI requirements (Perfect match)
├── Binary relevance: 1 (relevant)
└── Graded relevance: 3 (highly relevant)

Document B: General mortgage guide with MI section (Good match)
├── Binary relevance: 1 (relevant)
└── Graded relevance: 2 (relevant)

Document C: Brief mention of MI in footnote (Partial match)
├── Binary relevance: 1 (relevant)
└── Graded relevance: 1 (somewhat relevant)

Document D: About FHA loans, not conventional (Not relevant)
├── Binary relevance: 0 (not relevant)
└── Graded relevance: 0 (not relevant)
```

**nDCG calculation example**:
```
Query: "conventional loan down payment"
Relevance scale: 0=not relevant, 1=somewhat, 2=relevant, 3=highly relevant

Ranking A: [3, 2, 1, 0, 0]  (good → okay → poor)
├── DCG@5 = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0/log2(5) + 0/log2(6)
├── DCG@5 = 3.0 + 1.26 + 0.50 + 0 + 0 = 4.76
├── IDCG@5 = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0/log2(5) + 0/log2(6) = 4.76 (same as DCG)
└── nDCG@5 = 4.76 / 4.76 = 1.00 (perfect ranking!)

Ranking B: [0, 1, 2, 3, 0]  (poor → good, relevant results buried)
├── DCG@5 = 0/log2(2) + 1/log2(3) + 2/log2(4) + 3/log2(5) + 0/log2(6)
├── DCG@5 = 0 + 0.63 + 1.0 + 1.29 + 0 = 2.92
├── IDCG@5 = 4.76 (same as above—ideal is always best possible ranking)
└── nDCG@5 = 2.92 / 4.76 = 0.61 (poor! relevant results too low)

Insight: Same 4 documents, but Ranking A is much better (1.00 vs 0.61) because highly relevant results appear first
```

**Relevance scale guidelines:**

| Score | Label | Criteria | E-commerce Example |
|-------|-------|----------|-------------------|
| **3** | Highly relevant | Perfect match, exactly what user wants | Exact product model user searched for |
| **2** | Relevant | Good match, meets user's needs | Same product, different color/size |
| **1** | Somewhat relevant | Partial match, related but not ideal | Related product, same category |
| **0** | Not relevant | No match, irrelevant | Unrelated product, wrong category |

**Advantages of nDCG**:
- **Handles graded relevance**: Distinguishes between "perfect" and "okay" matches
- **Position-aware**: Higher positions weighted more (log discount)
- **Normalized**: Scores 0-1, comparable across queries with different result counts
- **Flexible**: Works with any relevance scale (0-1, 0-3, 0-5, etc.)

**When to use nDCG vs other metrics**:

| Scenario | Best Metric | Reasoning |
|----------|------------|-----------|
| Binary relevance (yes/no) | MAP or Precision@K | Simpler, easier to annotate |
| Graded relevance (shades of gray) | nDCG | Captures nuanced quality differences |
| Single answer needed | MRR | Only first result matters |
| Multiple good answers | nDCG or MAP | Rewards clustering relevant results high |
| E-commerce product search | nDCG@10 | Products have varying relevance levels |
| Q&A factual answers | MRR or Precision@1 | One correct answer |

**Typical target values:**
- **nDCG@5 > 0.80**: High-quality systems (e-commerce, premium search)
- **nDCG@10 > 0.75**: Good systems (document search, research tools)
- **nDCG@20 > 0.70**: Acceptable for exploratory search

**Common pitfall—annotation consistency**:
```
Same query annotated by two people:

Annotator 1: [3, 2, 2, 1, 0]  → nDCG = 0.95
Annotator 2: [2, 3, 1, 2, 0]  → nDCG = 0.88

Inconsistent annotations make nDCG unreliable!
Solution: Clear annotation guidelines + inter-annotator agreement checks (Kappa > 0.70)
```

## 🎯 Behavioral Metrics

Offline metrics (Precision, MAP, nDCG) measure relevance based on manual annotations. Behavioral metrics measure what users actually DO with search results—revealing real-world satisfaction.

**Why behavioral metrics matter**:
- Offline metrics can be wrong (annotations don't match user preferences)
- Users vote with their clicks (implicit feedback at scale)
- Capture business impact (clicks → conversions → revenue)

**The behavioral metrics hierarchy**:
```
Engagement Funnel:
└── Search query issued
    └── Results displayed → Impression rate (100% baseline)
        └── User clicks result → CTR (Click-through rate)
            └── User stays on page → Dwell time
                └── User completes action → Conversion rate
                    └── User returns to search → Bounce rate (failure signal)
```

### Click-through Rate (CTR)
**Definition**: What percentage of searches result in at least one click?

```
CTR = Search sessions with ≥1 click ÷ Total search sessions
```

**Why CTR matters**:
- **High CTR**: Users find promising results worth clicking
- **Low CTR**: Results don't look relevant (poor titles, snippets, or actual relevance)
- **Trend changes**: A/B test signal (strategy B increased CTR by 15%)

**Real-world interpretation**:
- **CTR = 0.80**: 80% of searches get a click (good engagement)
- **CTR = 0.50**: Half of searches get no clicks (poor results or user abandonment)
- **CTR = 0.30**: 70% of searches result in no clicks (major problem)

**CTR by search type**:
| Search Type | Expected CTR | Reasoning |
|-------------|--------------|-----------|
| Navigation (find specific page) | 0.85-0.95 | Users know what they want, click if found |
| Informational (learn something) | 0.70-0.85 | Users evaluate options before clicking |
| Transactional (buy/download) | 0.60-0.80 | Users compare before committing |
| Exploratory (browsing) | 0.40-0.60 | Users may not click, just scanning |

**A/B testing with CTR**:
```
Baseline (BM25):
├── CTR = 0.68
└── 1000 queries → 680 clicks

Strategy B (Hybrid search):
├── CTR = 0.78
└── 1000 queries → 780 clicks
└── +14.7% improvement (100 more engaged users)

Decision: Hybrid search improves engagement
```

**Common CTR pitfalls**:
1. **High CTR doesn't mean good results**: Users might click irrelevant results with misleading titles
   - Solution: Track dwell time and bounce rate alongside CTR
2. **Zero-click searches**: Some searches don't need clicks (answer in snippet, user changed mind)
   - Solution: Distinguish "satisfied zero-click" from "frustrated zero-click"
3. **Position bias**: First result gets more clicks regardless of relevance
   - Solution: Use position-adjusted CTR or randomized experiments

### Conversion Rate
**Definition**: What percentage of searches lead to a desired business outcome?

```
Conversion Rate = Search sessions with successful action ÷ Total search sessions
```

**Successful actions** (varies by application):
- **E-commerce**: Add to cart, purchase
- **Content sites**: Read article >30 seconds, share
- **SaaS products**: Sign up, start trial
- **Support portals**: Download document, submit ticket

**Why conversion rate is the ultimate metric**:
Conversion rate directly measures business impact. You can have perfect Precision@5 and MAP, but if users don't convert, the search system isn't delivering value.

**Real-world examples**:
```
E-commerce site:
├── 10,000 searches per day
├── Conversion rate = 0.12 (12%)
├── 1,200 purchases per day from search
└── Average order value = $85
└── Search-driven revenue = 1,200 × $85 = $102,000/day

After hybrid search upgrade:
├── Conversion rate = 0.15 (15%)
├── 1,500 purchases per day from search
└── Search-driven revenue = 1,500 × $85 = $127,500/day
└── +$25,500/day = +$9.3M/year in additional revenue

ROI: If hybrid search costs $100k to implement, payback in 4 days!
```

**Conversion rate by industry**:
| Industry | Typical Conversion Rate | High-Performing Systems |
|----------|------------------------|------------------------|
| E-commerce | 8-15% | 18-25% |
| SaaS products | 2-5% | 8-12% |
| Content/media | 15-30% | 35-50% |
| Support portals | 25-40% | 50-65% |

**Conversion rate optimization strategies**:
```
Low conversion diagnosis:

Problem: High CTR (0.75) but low conversion (0.05)
├── Users click promising results
└── But don't complete actions
└── Diagnosis: Result quality is poor (good titles, bad content)
└── Solution: Improve relevance, not just click appeal

Problem: Low CTR (0.40) and low conversion (0.05)
├── Users don't even click
└── Diagnosis: Results don't look relevant
└── Solution: Improve ranking + result presentation (titles, snippets)

Problem: High CTR (0.80) and moderate conversion (0.12)
├── Users engage but many don't convert
└── Diagnosis: Missing the right products/documents
└── Solution: Improve recall, expand inventory
```

### Dwell Time
**Definition**: How long do users spend on clicked results before returning to search (or leaving entirely)?

```
Dwell Time = Time between click and return-to-search (or session end)
```

**Why dwell time reveals true satisfaction**:
- **Long dwell time**: User found what they needed, reading/engaging deeply
- **Short dwell time**: User quickly realized result was irrelevant (bounce)
- **Position**: Strongest signal when combined with CTR and conversion

**Dwell time interpretation**:
| Dwell Time | Interpretation | Action |
|------------|----------------|--------|
| < 10 seconds | Immediate bounce, wrong result | Investigate relevance failure |
| 10-30 seconds | Quick scan, possibly useful | Borderline, monitor trends |
| 30-120 seconds | Engaged reading, likely relevant | Good signal |
| > 120 seconds | Deep engagement, high satisfaction | Strong positive signal |

**Dwell time by content type**:
- **Quick facts**: 15-30 seconds expected (longer = good)
- **Articles/guides**: 60-180 seconds expected
- **Product pages**: 30-90 seconds (comparison shopping)
- **Forms/applications**: 120-300 seconds (task completion)

**Real-world dwell time analysis**:
```
Query: "FHA loan credit score requirements"

Ranking A (BM25):
├── Position 1 CTR: 0.45, Avg dwell: 15 sec (fast bounce—wrong doc)
├── Position 2 CTR: 0.30, Avg dwell: 75 sec (good engagement)
└── Position 3 CTR: 0.15, Avg dwell: 90 sec (best result, but ranked 3rd!)

Ranking B (Hybrid search):
├── Position 1 CTR: 0.55, Avg dwell: 85 sec (correct result promoted)
├── Position 2 CTR: 0.25, Avg dwell: 70 sec
└── Position 3 CTR: 0.10, Avg dwell: 20 sec

Improvement: Ranking B puts best result first (85 sec dwell vs 15 sec)
```

**Combining behavioral metrics for complete picture**:
```
System Evaluation Matrix:

Metric           | System A | System B | System C
-----------------|----------|----------|----------
CTR              |   0.75   |   0.65   |   0.80
Avg Dwell Time   |   25 sec |   90 sec |   85 sec
Conversion Rate  |   0.08   |   0.18   |   0.17
-----------------|----------|----------|----------
Interpretation   | Clickbait| Best!    | Good
                 | High CTR | Lower    | Balanced
                 | but users| CTR but  | performance
                 | bounce   | deeply   |
                 | quickly  | engaged  |

Decision: System B is best despite lower CTR (engagement + conversion matter more)
```

## 📈 Advanced Metrics

These metrics address specific evaluation scenarios that basic metrics don't capture well. Use them to supplement core metrics when you need specialized insights.

### Success@K
**Definition**: What percentage of queries have at least ONE relevant result in the top K positions? This is a binary "did we succeed or fail" metric per query.

```
Success@K = Queries with ≥1 relevant result in top K ÷ Total queries
```

**Why Success@K matters**:

Unlike Precision@K (which averages across all results), Success@K simply asks: "Did the user have a chance to find what they needed?" This is especially important when you can't afford complete failures.

**Real-world example**:
```
Customer support search (100 queries):

System A:
├── 85 queries: At least 1 relevant result in top 10
├── 15 queries: No relevant results in top 10 (complete failure)
└── Success@10 = 85/100 = 0.85

System B:
├── 92 queries: At least 1 relevant result in top 10
├── 8 queries: No relevant results in top 10
└── Success@10 = 92/100 = 0.92 (+7% fewer total failures)

Impact: System B reduces "zero-answer" experiences from 15% to 8%
```

**Success@K vs Precision@K comparison**:
```
Query set: 10 queries, K=5

Query 1: 5 relevant in top 5 → Precision@5=1.0, Success@5=1
Query 2: 1 relevant in top 5 → Precision@5=0.2, Success@5=1 (still succeeded!)
Query 3: 0 relevant in top 5 → Precision@5=0.0, Success@5=0 (complete failure)
...

Average Precision@5: 0.65 (moderate quality)
Success@5: 0.90 (90% of queries had at least one relevant result)

Insight: Most queries succeeded, but quality varies (some queries have only 1 relevant result)
```

**When to use Success@K**:
- **Customer support**: Zero answers = frustrated customers escalating to human support (costly)
- **Medical search**: Missing relevant treatments could harm patients
- **Legal e-discovery**: Missing evidence could lose cases
- **Product search**: Ensure every search has SOMETHING to click

**Target values:**
- **Success@1**: > 0.70 for Q&A systems (70% of queries get correct first result)
- **Success@5**: > 0.85 for general search (85% of queries find at least one relevant result in top 5)
- **Success@10**: > 0.90 for document retrieval (90% of queries succeed)
- **Success@20**: > 0.95 for comprehensive search (almost no complete failures)

**Diagnostic use case**:
```
System has MAP=0.65 (decent) but Success@10=0.75 (poor)

Diagnosis:
├── When system succeeds, it ranks well (good MAP)
└── But 25% of queries get NO relevant results (poor coverage)

Solution: Improve recall/coverage, not just ranking
```

### Expected Reciprocal Rank (ERR)
**Definition**: A probabilistic cascade metric that models realistic user behavior—users scan results top-to-bottom and MAY stop at each relevant result based on how satisfied they are.

```
ERR = Σ(1/r × P(user stops at position r))
```

Where:
```
P(user stops at r) = rel(r) × Π(1 - rel(i)) for i < r
```
- **rel(r)**: Probability that result at rank r satisfies the user (based on graded relevance)
- **Cascade assumption**: Users only see position r if they weren't satisfied by positions 1 through r-1

**Why ERR is more realistic than nDCG**:

nDCG assumes users always examine all K results. ERR assumes users STOP when satisfied, which is closer to real behavior.

**Real-world comparison**:
```
Query: "What is the maximum LTV for conventional loans?"

Ranking A: Relevance scores [0.9, 0.3, 0.7, 0.5] (highly relevant first)
├── P(stop at 1) = 0.9 (90% stop immediately, found answer)
├── P(stop at 2) = (1-0.9) × 0.3 = 0.03 (3% continue and stop at 2)
├── P(stop at 3) = (1-0.9) × (1-0.3) × 0.7 = 0.049
├── ERR = 1/1 × 0.9 + 1/2 × 0.03 + 1/3 × 0.049 + ... = 0.93
└── Most users satisfied immediately

Ranking B: Relevance scores [0.3, 0.5, 0.7, 0.9] (highly relevant last)
├── P(stop at 1) = 0.3 (30% satisfied with mediocre first result)
├── P(stop at 2) = (1-0.3) × 0.5 = 0.35
├── P(stop at 3) = (1-0.3) × (1-0.5) × 0.7 = 0.245
├── P(stop at 4) = (1-0.3) × (1-0.5) × (1-0.7) × 0.9 = 0.0945
├── ERR = 1/1 × 0.3 + 1/2 × 0.35 + 1/3 × 0.245 + 1/4 × 0.0945 = 0.58
└── Users have to scan multiple results

Insight: Ranking A (ERR=0.93) much better than B (ERR=0.58) because highly relevant result comes first
```

**Relevance probability mapping** (from graded relevance scores):
```
Graded relevance → Probability of satisfaction:
├── 0 (not relevant) → P(satisfy) = 0.0
├── 1 (somewhat relevant) → P(satisfy) = 0.33
├── 2 (relevant) → P(satisfy) = 0.67
└── 3 (highly relevant) → P(satisfy) = 1.0

Common formula: P(satisfy) = (2^relevance - 1) / 2^max_relevance
Example with 0-3 scale: P(satisfy) = (2^rel - 1) / 8
├── rel=0: (1-1)/8 = 0.0
├── rel=1: (2-1)/8 = 0.125
├── rel=2: (4-1)/8 = 0.375
└── rel=3: (8-1)/8 = 0.875
```

**When to use ERR**:
- **Q&A systems**: Models "stop when answered" behavior better than nDCG
- **Voice assistants**: Users literally can only consume one result
- **Mobile search**: Small screens = users less likely to scroll past first good result
- **Urgent queries**: Users stop as soon as they find acceptable answer

**When nDCG is better than ERR**:
- **Exploratory search**: Users want multiple perspectives (don't stop at first result)
- **Comparison shopping**: Users intentionally review many results before deciding
- **Research tasks**: Users consume multiple documents

**Target values**:
- **ERR > 0.80**: Excellent—most users find highly relevant results early
- **ERR = 0.60-0.80**: Good—users typically find relevant results in top 3-5
- **ERR < 0.60**: Poor—users have to scan many results to find satisfaction

**Common use case—comparing two systems**:
```
System A (keyword search):
├── nDCG@10 = 0.75
└── ERR = 0.62

System B (semantic search):
├── nDCG@10 = 0.78 (only +0.03 improvement)
└── ERR = 0.81 (+0.19 improvement!)

Insight: Both systems have similar nDCG, but System B puts highly relevant results much earlier (better user experience in practice)
```

## 🔧 Implementation Considerations

Choosing and implementing the right metrics requires careful consideration of your use case, available resources, and evaluation goals.

### Metric Selection Guidelines

The right metrics depend on your search application type and what matters most to your users:

| Scenario | Primary Metrics | Secondary Metrics | Why These Metrics? |
|----------|----------------|-------------------|-------------------|
| **Q&A Systems** | MRR, Precision@1 | nDCG@5, Success@1 | First result must be correct; users need one answer fast |
| **E-commerce Search** | nDCG@10, CTR | Precision@5, Conversion Rate | Graded relevance matters (exact vs similar products); revenue impact critical |
| **Document Retrieval** | MAP, Recall | nDCG@20, Precision@10 | Ranking quality across all results; can't miss relevant documents |
| **RAG Systems** | Precision@5, MRR | nDCG@10, Success@5 | Retrieved context must be highly relevant; downstream LLM depends on quality |
| **Navigation** | MRR, Success@1 | Precision@1, CTR | Users know what they want; must surface specific page/document fast |
| **Exploratory Search** | nDCG@20, MAP | Precision@10, Dwell Time | Users review many results; need diversity and depth |

**Decision framework**:
```
Start here → What is the user's primary goal?
│
├─→ Find ONE specific answer/document?
│   └─→ Use MRR (rewards fast first relevant result)
│
├─→ Explore MULTIPLE relevant items?
│   └─→ Use MAP or nDCG@20 (rewards comprehensive quality)
│
├─→ Take a BUSINESS ACTION (buy, download, sign up)?
│   └─→ Use Conversion Rate + nDCG (revenue impact + quality)
│
└─→ Can't afford ZERO relevant results?
    └─→ Use Success@K + Recall (ensure coverage)
```

### Evaluation Setup

**1. Ground Truth Creation:**

You need relevance judgments to measure most offline metrics. Options ranked by cost/quality:

| Method | Cost | Quality | Best For | Challenges |
|--------|------|---------|----------|------------|
| **Expert annotation** | High ($50-100/hr) | Excellent (0.85+ IAA) | Medical, legal, technical domains | Expensive, slow, hard to scale |
| **Crowdsourcing** | Medium ($0.05-0.20/judgment) | Good (0.70+ IAA with training) | General search, e-commerce | Need quality control, clear guidelines |
| **Implicit feedback** | Low (automated) | Moderate (noisy) | Large-scale systems with existing traffic | Biased by current ranking, position bias |
| **LLM-generated** | Low (<$0.01/judgment) | Variable (0.60-0.80 IAA) | Rapid iteration, augmenting human labels | Requires validation, model biases |

**Annotation guidelines matter**:
```
Bad guideline: "Mark results as relevant or not relevant"
└─→ Result: Low inter-annotator agreement (Kappa=0.45), inconsistent labels

Good guideline: 
"Mark as Highly Relevant (3) if the document directly answers the query with specific details.
 Mark as Relevant (2) if the document discusses the topic but requires additional information.
 Mark as Somewhat Relevant (1) if the document mentions the topic but is primarily about something else.
 Mark as Not Relevant (0) if the document does not discuss the query topic."
└─→ Result: High inter-annotator agreement (Kappa=0.78), reliable labels
```

**2. Query Sets:**

Diverse query sets prevent overfitting and ensure your system works for all user needs:

```
Balanced query set example (150 queries):

Query Type Distribution:
├── Navigational (30%): "Find 2024 Freddie Mac seller guide PDF"
├── Informational (40%): "What are the credit score requirements for conventional loans?"
├── Transactional (20%): "Apply for mortgage pre-approval"
└── Exploratory (10%): "Mortgage lending best practices"

Difficulty Distribution:
├── Easy (30%): Common questions, clear answers exist
├── Medium (50%): Require some interpretation, multiple relevant docs
└── Hard (20%): Rare questions, ambiguous intent, limited content

Query Length Distribution:
├── Short (1-2 words, 20%): "FHA loans", "PMI"
├── Medium (3-5 words, 50%): "conventional loan requirements"
└── Long (6+ words, 30%): "What is the maximum loan to value ratio for a cash out refinance on an investment property?"
```

**Common pitfall—unbalanced queries**:
```
Biased query set:
├── 90% easy navigational queries → System looks great (MAP=0.95)
└── But fails on real traffic (40% informational queries not tested)
└── Production MAP = 0.68 (massive gap!)

Solution: Sample queries proportionally to real user traffic distribution
```

**3. Test Splits:**

Use **temporal splits** to avoid data leakage and simulate real-world deployment:

```
❌ Random split (WRONG):
├── Training: Random 80% of queries/documents
└── Test: Random 20% of queries/documents
└── Problem: Test queries may have seen similar training queries (overly optimistic results)

✅ Temporal split (CORRECT):
├── Training: Queries/documents before 2024-01-01
├── Validation: Queries/documents 2024-01-01 to 2024-06-01
└── Test: Queries/documents after 2024-06-01
└── Benefit: Simulates deploying system and evaluating on future unseen queries

✅ Stratified temporal split (BEST):
├── Maintain query type distribution across all splits
├── Ensure rare query types appear in validation/test
└── Temporal ordering preserved within strata
```

**4. Statistical Significance:**

Don't trust small differences without statistical validation:

```python
# Example: Comparing two search systems
from scipy import stats

# System A results (MAP scores for 100 queries)
system_a_scores = [0.82, 0.65, 0.91, ...]  # 100 queries

# System B results (same 100 queries)
system_b_scores = [0.85, 0.68, 0.93, ...]  # 100 queries

# Paired t-test (same queries for both systems)
t_statistic, p_value = stats.ttest_rel(system_b_scores, system_a_scores)

if p_value < 0.05:
    print(f"System B is significantly better (p={p_value:.4f})")
    print(f"Mean improvement: {np.mean(system_b_scores) - np.mean(system_a_scores):.3f}")
else:
    print(f"No significant difference (p={p_value:.4f})")
    print("Difference could be due to random chance")
```

**Sample size considerations**:
```
Rule of thumb for detecting improvements:

Baseline MAP = 0.70, want to detect +0.05 improvement (MAP=0.75)
├── Statistical power = 0.80 (80% chance of detecting real improvement)
├── Significance level α = 0.05 (5% false positive rate)
└── Required sample size ≈ 150 queries

Baseline MAP = 0.70, want to detect +0.02 improvement (MAP=0.72)
├── Statistical power = 0.80
├── Significance level α = 0.05
└── Required sample size ≈ 900 queries (much larger for small improvements!)

Lesson: Small improvements require large query sets to validate
```

### Common Pitfalls and How to Avoid Them

**1. Evaluation bias (using training queries for testing)**
```
❌ Problem:
├── Train indexing strategy on Query Set A
├── Tune parameters to maximize MAP on Query Set A
└── Report final results on Query Set A
└── Result: Overly optimistic, won't generalize

✅ Solution:
├── Training set: Tune index parameters
├── Validation set: Select best configuration
└── Test set: Report final results (never touch during development)
```

**2. Limited query diversity (not representing real users)**
```
❌ Problem:
├── Create evaluation queries based on what developers think users search for
└── Miss actual user query patterns (spelling errors, abbreviations, conversational queries)

✅ Solution:
├── Sample 40% of evaluation queries from real search logs
├── Add 30% expert-generated edge cases
├── Add 20% synthetic variations (GPT-4 generated)
└── Add 10% from user testing sessions
```

**3. Binary relevance only (missing nuanced quality)**
```
❌ Problem:
├── Mark all documents as either relevant (1) or not relevant (0)
└── Can't distinguish between "perfect match" and "somewhat related"

✅ Solution:
├── Use graded relevance (0-3 scale)
├── Calculate both binary metrics (Precision, Recall) AND graded metrics (nDCG)
└── Graded relevance reveals quality differences binary misses
```

**4. Position bias (not accounting for browsing patterns)**
```
❌ Problem:
├── Use behavioral metrics (CTR, dwell time) directly
└── First result always gets more clicks regardless of relevance

✅ Solution:
├── Use position-adjusted CTR or propensity weighting
├── Run randomized interleaving experiments (swap positions randomly)
└── Combine offline metrics (no position bias) with online metrics
```

**5. Corpus changes (evaluating on outdated documents)**
```
❌ Problem:
├── Create evaluation set in January 2024
├── Evaluate new system in December 2024
└── 30% of test queries reference documents added after January (can't find them!)

✅ Solution:
├── Version evaluation sets with snapshot dates
├── Update evaluation sets quarterly with new documents
└── Track "answer coverage" metric (% of queries with relevant docs in corpus)
```

**6. Metric selection mismatch**
```
❌ Problem:
├── E-commerce search optimized for Recall (find all products)
└── But users actually care about Precision (top 5 results must be good)
└── Result: High Recall but poor user experience

✅ Solution:
├── Align metrics with user goals and business objectives
├── For e-commerce: Prioritize nDCG@10, CTR, Conversion Rate
└── Monitor secondary metrics (Recall) but don't optimize for them
```

## 📊 Metric Reporting Template

Use this template to communicate evaluation results clearly to stakeholders:

```python
# Example comprehensive evaluation report
evaluation_report = {
    "metadata": {
        "dataset": "mortgage_search_eval_2024_q4",
        "evaluation_date": "2024-12-15",
        "query_count": 150,
        "evaluator": "search-quality-team",
        "baseline_system": "bm25_default",
        "test_system": "hybrid_bm25_vector_semantic",
    },
    
    "core_metrics": {
        "precision_at_1": {
            "value": 0.82,
            "baseline": 0.75,
            "delta": +0.07,
            "p_value": 0.003,  # Statistically significant
            "interpretation": "First result correct 82% of time (+7% vs baseline)"
        },
        "precision_at_5": 0.74,
        "precision_at_10": 0.68,
        "recall": {
            "value": 0.85,
            "baseline": 0.78,
            "delta": +0.07,
            "interpretation": "Finding 85% of relevant documents (+7%)"
        },
        "f1_score": 0.76,
        "map": {
            "value": 0.72,
            "baseline": 0.65,
            "delta": +0.07,
            "p_value": 0.001,
            "interpretation": "Significant improvement in overall ranking quality"
        },
        "mrr": 0.85,
        "ndcg_at_10": {
            "value": 0.83,
            "baseline": 0.76,
            "delta": +0.07,
            "interpretation": "Better at surfacing highly relevant results early"
        },
        "success_at_10": 0.94
    },
    
    "behavioral_metrics": {
        "ctr": {
            "value": 0.68,
            "baseline": 0.62,
            "delta": +0.06,
            "interpretation": "6% more searches result in clicks"
        },
        "avg_dwell_time_seconds": {
            "value": 58.3,
            "baseline": 45.1,
            "delta": +13.2,
            "interpretation": "Users spending more time with results (higher engagement)"
        },
        "conversion_rate": {
            "value": 0.15,
            "baseline": 0.12,
            "delta": +0.03,
            "interpretation": "3% absolute increase = +25% relative improvement in conversions"
        }
    },
    
    "query_type_breakdown": {
        "navigational": {
            "query_count": 45,
            "mrr": 0.91,
            "success_at_1": 0.87
        },
        "informational": {
            "query_count": 60,
            "map": 0.68,
            "ndcg_at_10": 0.81
        },
        "transactional": {
            "query_count": 30,
            "precision_at_5": 0.76,
            "conversion_rate": 0.22
        }
    },
    
    "failure_analysis": {
        "zero_results": {
            "count": 3,
            "percentage": 0.02,
            "examples": ["obscure 2019 policy question", "misspelled technical term"]
        },
        "poor_ranking": {
            "queries_with_map_lt_0.5": 12,
            "common_issues": ["date-range queries", "multi-condition eligibility questions"]
        }
    },
    
    "recommendations": [
        "Deploy hybrid system - statistically significant improvements across all core metrics",
        "Monitor date-range query handling - identified as weakness (12 queries, MAP<0.5)",
        "Expected business impact: +25% conversion rate = +$2.1M annual revenue",
        "Next: Expand evaluation set to 300 queries to detect smaller improvements"
    ]
}
```

**Stakeholder-friendly summary format**:
```markdown
## Search System Evaluation Results
**Date**: 2024-12-15  
**System**: Hybrid BM25 + Vector + Semantic Reranking  
**Baseline**: BM25 Default  

### Key Findings
✅ **Statistically significant improvements** across all metrics (p<0.01)  
✅ **+7% better first result** (Precision@1: 0.82 vs 0.75)  
✅ **+25% more conversions** (0.15 vs 0.12)  
✅ **+29% longer engagement** (58.3 vs 45.1 seconds avg dwell time)  

### Business Impact
- Projected annual revenue increase: **+$2.1M**  
- Improved user satisfaction (fewer complaints about "can't find anything")  
- Reduced support escalations (more self-service success)  

### Recommendation
**DEPLOY** hybrid system to production  
- Roll out via 10% A/B test for 2 weeks  
- Monitor behavioral metrics in production  
- Full rollout if production validates evaluation results  
```

---
*Next: [Evaluation Frameworks](./02-evaluation-frameworks.md)*