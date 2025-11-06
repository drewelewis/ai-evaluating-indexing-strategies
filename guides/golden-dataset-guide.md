# Golden Dataset Creation Guide

**Purpose**: A golden dataset is a curated collection of test queries with verified correct answers used to evaluate search and AI agent performance systematically.

**Target Audience**: Search engineers, QA teams, product managers, and data scientists building RAG applications.

**Reading Time**: 15-20 minutes

---

## Table of Contents

1. [What is a Golden Dataset?](#1-what-is-a-golden-dataset)
2. [Why You Need One](#2-why-you-need-one)
3. [Dataset Size Guidelines](#3-dataset-size-guidelines)
4. [Query Source Mix](#4-query-source-mix)
5. [Query Categories](#5-query-categories)
6. [Annotation Schema](#6-annotation-schema)
7. [Step-by-Step Creation Process](#7-step-by-step-creation-process)
8. [Quality Assurance](#8-quality-assurance)
9. [Maintenance & Updates](#9-maintenance--updates)
10. [Tools & Templates](#10-tools--templates)

---

## 1. What is a Golden Dataset?

A **golden dataset** (also called ground truth dataset, evaluation dataset, or test set) is a carefully curated collection of:

1. **Queries**: Real or realistic questions users ask your system
2. **Expected Results**: What the correct answer or retrieved documents should be
3. **Metadata**: Additional context for evaluation (difficulty, category, etc.)

**Example Entry**:
```json
{
  "query_id": "Q001",
  "query": "What is the maximum LTV ratio for conventional mortgages?",
  "category": "factual",
  "difficulty": "easy",
  "expected_answer": "The maximum LTV ratio is 97% for eligible primary residences with mortgage insurance according to Fannie Mae/Freddie Mac guidelines.",
  "relevant_sections": ["Section 4.2.1", "Section 5.1"],
  "document_source": "Fannie Mae Selling Guide 2025",
  "expected_entities": ["LTV", "97%", "mortgage insurance"],
  "created_by": "expert_1",
  "verified_by": "expert_2",
  "created_date": "2025-11-01"
}
```

**What Makes It "Golden"?**
- ✅ **Expert-verified**: Answers reviewed by domain experts
- ✅ **Representative**: Covers real user needs
- ✅ **Stable**: Doesn't change frequently (unlike live data)
- ✅ **Comprehensive**: Spans all important use cases
- ✅ **Measurable**: Enables quantitative evaluation

---

## 2. Why You Need One

### Without a Golden Dataset:
- ❌ No objective way to measure if changes improve or degrade performance
- ❌ Anecdotal testing ("this query works better now")
- ❌ Can't compare different approaches systematically
- ❌ Regression bugs go undetected
- ❌ Stakeholders can't see measurable progress

### With a Golden Dataset:
- ✅ **Objective metrics**: Precision@5 went from 0.65 → 0.82
- ✅ **A/B testing**: Compare chunking strategies with confidence
- ✅ **Regression detection**: Catch performance drops in CI/CD
- ✅ **Stakeholder communication**: "We improved answer accuracy by 23%"
- ✅ **Continuous improvement**: Track performance over time

### Real-World Impact Example:

**Before Golden Dataset**:
> "The new chunking strategy seems better. Deploy it?"

**After Golden Dataset**:
> "New chunking strategy tested on 150 queries:
> - Precision@5: 0.78 → 0.84 (+7.7%)
> - Answer Accuracy: 0.82 → 0.89 (+8.5%)
> - Latency: 1.2s → 1.4s (+16.7%)
> - Recommendation: Deploy (accuracy gains outweigh latency cost)"

---

## 3. Dataset Size Guidelines

### Minimum Viable Dataset (MVP)
**Size**: 50 queries  
**Timeline**: 1-2 weeks to create  
**Use Case**: Initial proof-of-concept, basic A/B tests  
**Limitation**: May not cover all edge cases  

### Recommended Baseline
**Size**: 100-200 queries  
**Timeline**: 3-4 weeks to create  
**Use Case**: Production readiness, comprehensive testing  
**Coverage**: All major query categories with multiple examples  

### Enterprise-Grade
**Size**: 300-500+ queries  
**Timeline**: 6-8 weeks to create  
**Use Case**: Mission-critical applications, regulatory compliance  
**Coverage**: Extensive edge cases, multiple difficulty levels  

### Progressive Growth Strategy

```
Phase 1 (Week 1-2): 50 queries
├── 40% from historical data (20 queries)
├── 30% expert-generated (15 queries)
├── 20% synthetic (10 queries)
└── 10% user testing (5 queries)

Phase 2 (Week 3-4): Expand to 150 queries (+100)
├── Add edge cases discovered in testing
├── Fill gaps in category distribution
└── Add multi-hop reasoning queries

Phase 3 (Week 5-8): Expand to 300 queries (+150)
├── Add adversarial queries
├── Add ambiguous/difficult queries
└── Add cross-document queries

Ongoing: Add 5-10 queries per month
├── New features or policy changes
├── User-reported failures
└── Seasonal/temporal queries
```

---

## 4. Query Source Mix

A robust golden dataset combines queries from multiple sources to ensure representativeness.

### 4.1 Historical Queries (40% of dataset)

**Source**: Actual queries from existing systems (logs, tickets, emails, chat transcripts)

**Pros**:
- ✅ Guaranteed real user needs
- ✅ Natural language variations
- ✅ Reveals actual pain points

**Cons**:
- ❌ May be biased toward current system capabilities
- ❌ May miss queries users gave up asking

**How to Collect**:

```python
# Example: Extract from support ticket system
import pandas as pd

tickets = pd.read_csv('support_tickets.csv')

# Filter for mortgage guideline questions
guideline_tickets = tickets[
    tickets['category'].isin(['Guidelines', 'Policy', 'Eligibility'])
]

# Extract question patterns
queries = guideline_tickets['description'].apply(extract_question)

# Deduplicate similar queries
unique_queries = deduplicate_queries(queries, similarity_threshold=0.85)

print(f"Found {len(unique_queries)} unique historical queries")
```

**Example Historical Queries**:
- "Customer has 630 credit score - what's the max LTV we can do?"
- "Borrower is self-employed, what income docs do we need?"
- "Can we use boarder income for qualifying?"

### 4.2 Expert-Generated Queries (30% of dataset)

**Source**: Domain experts who understand what users *should* be able to ask

**Pros**:
- ✅ Covers comprehensive domain knowledge
- ✅ Includes sophisticated queries
- ✅ Represents ideal use cases

**Cons**:
- ❌ May not match actual user language
- ❌ Can be too technical or formal

**How to Create**:

1. **Recruit 3-5 domain experts** (underwriters, loan officers, compliance specialists)
2. **Provide query templates** for each category (see Section 5)
3. **Review documentation** and create questions that *should* be answerable
4. **Brainstorm edge cases** and complex scenarios

**Example Expert-Generated Queries**:
- "What are the reserve requirements for a borrower with 6 financed investment properties?"
- "How is the waiting period calculated after Chapter 13 bankruptcy dismissal versus discharge?"
- "Can subordinate financing from an employer be combined with a grant from a nonprofit?"

### 4.3 Synthetic Queries (20% of dataset)

**Source**: AI-generated variations of existing queries or document-based generation

**Pros**:
- ✅ Rapid creation at scale
- ✅ Good for testing robustness to phrasing
- ✅ Can generate multilingual variants

**Cons**:
- ❌ May generate unrealistic queries
- ❌ Requires human verification
- ❌ Can miss nuanced real-world needs

**How to Generate**:

```python
from openai import AzureOpenAI

client = AzureOpenAI(...)

# Method 1: Paraphrase existing queries
def generate_paraphrases(original_query, n=5):
    prompt = f"""Generate {n} different ways to ask this question:
    
    Original: "{original_query}"
    
    Requirements:
    - Keep the same meaning
    - Vary formality (casual to professional)
    - Use different vocabulary
    - Include typos/informal language for some variants
    
    Output as JSON list."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    
    return parse_response(response)

# Method 2: Generate from document content
def generate_from_section(section_text):
    prompt = f"""Based on this policy section, generate 3-5 questions 
    that users might ask:
    
    Section: {section_text}
    
    Requirements:
    - Questions should be realistic (how users actually talk)
    - Mix factual, procedural, and eligibility questions
    - Include some that require synthesis across paragraphs
    
    Output as JSON list."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return parse_response(response)
```

**Example Synthetic Queries** (generated from "What is the maximum LTV?"):
- "whats the highest ltv we can go?" (casual, typo)
- "I need to know the loan to value ceiling for conforming loans" (formal)
- "maximum leverage allowed on conventional mortgage?" (different terminology)
- "how much can borrower finance - ltv wise?" (conversational)

### 4.4 User Testing Queries (10% of dataset)

**Source**: Live user testing sessions with target audience

**Pros**:
- ✅ Validates assumptions about real usage
- ✅ Discovers unexpected query patterns
- ✅ Includes conversational context

**Cons**:
- ❌ Time-intensive to organize
- ❌ Small sample size
- ❌ May require IRB/ethics approval

**How to Conduct**:

1. **Recruit 5-10 representative users** (customer service reps, loan officers)
2. **Give realistic scenarios**: "A borrower asks about FHA eligibility..."
3. **Ask them to formulate queries** they would use
4. **Record queries verbatim** (with permission)
5. **Note follow-up questions** in conversational sequences

**Example User Testing Queries**:
- "ok so if they have a 580 score can they still do 3.5% down or no?"
- "wait does the 2 year rule apply to commission or just bonus income?"
- "quick question - homeready vs home possible, same thing?"

---

## 5. Query Categories

A well-balanced golden dataset covers all types of questions users ask.

### 5.1 Category Distribution

Category | % of Dataset | Complexity | Avg Words | Example
---------|--------------|------------|-----------|----------
Factual | 30% | Low-Medium | 8-12 | "What is the maximum DTI ratio?"
Eligibility | 25% | Medium | 12-18 | "Can a borrower with 620 credit score qualify for HomeReady?"
Procedural | 20% | Medium-High | 15-25 | "How do you calculate self-employment income for qualification?"
Comparative | 15% | High | 12-20 | "What's the difference between Chapter 7 and Chapter 13 waiting periods?"
Update/Temporal | 10% | Low-Medium | 8-15 | "What are the 2025 conforming loan limits?"

### 5.2 Factual Queries (30%)

**Characteristics**:
- Single, definitive answer
- Usually answerable from one section
- Straightforward retrieval

**Templates**:
- "What is [concept]?"
- "What are the [limits/requirements/criteria] for [topic]?"
- "Define [term]"

**Examples**:
```json
[
  {"query": "What is the maximum LTV for conventional mortgages?"},
  {"query": "What does CLTV stand for?"},
  {"query": "What is the minimum credit score for a conventional loan?"},
  {"query": "How many months of reserves are required for investment properties?"},
  {"query": "What is the conforming loan limit in 2025?"}
]
```

### 5.3 Eligibility Queries (25%)

**Characteristics**:
- Scenario-based
- Requires evaluating criteria
- May involve multiple conditions

**Templates**:
- "Can a borrower with [condition] qualify for [loan type]?"
- "Is [scenario] eligible under [program]?"
- "Does [situation] meet the requirements for [product]?"

**Examples**:
```json
[
  {"query": "Can a borrower with 580 credit score get a conventional loan?"},
  {"query": "Is a borrower with 55% DTI eligible for HomeReady?"},
  {"query": "Can gift funds be used for the entire down payment on an investment property?"},
  {"query": "Does a non-occupant co-borrower qualify for 97% LTV?"},
  {"query": "Is rental income from Airbnb allowed for qualifying?"}
]
```

### 5.4 Procedural Queries (20%)

**Characteristics**:
- How-to questions
- Multi-step processes
- Often requires synthesizing information

**Templates**:
- "How do you [calculate/verify/document] [concept]?"
- "What is the process for [action]?"
- "What steps are required to [goal]?"

**Examples**:
```json
[
  {"query": "How do you calculate qualifying income for a self-employed borrower?"},
  {"query": "What is the process for verifying employment?"},
  {"query": "How do you determine the value for a cash-out refinance?"},
  {"query": "What steps are required to use retirement assets as reserves?"},
  {"query": "How is rental income calculated for a 2-unit property?"}
]
```

### 5.5 Comparative Queries (15%)

**Characteristics**:
- Requires comparing multiple concepts
- More complex reasoning
- Often involves tradeoffs

**Templates**:
- "What is the difference between [A] and [B]?"
- "How does [A] compare to [B]?"
- "[A] vs [B]"
- "When should I use [A] versus [B]?"

**Examples**:
```json
[
  {"query": "What's the difference between Fannie Mae and Freddie Mac LTV limits?"},
  {"query": "How does HomeReady compare to conventional financing?"},
  {"query": "Chapter 7 vs Chapter 13 bankruptcy - which has shorter waiting period?"},
  {"query": "What's the difference between rate/term and cash-out refinance?"},
  {"query": "Desktop Underwriter vs Loan Product Advisor - what's the difference?"}
]
```

### 5.6 Update/Temporal Queries (10%)

**Characteristics**:
- Time-sensitive information
- Requires current/recent data
- Often involves limits or thresholds that change

**Templates**:
- "What are the [current/2025] [limits/requirements]?"
- "Has [policy] changed recently?"
- "When did [change] take effect?"

**Examples**:
```json
[
  {"query": "What are the 2025 conforming loan limits?"},
  {"query": "What are the current mortgage insurance rates?"},
  {"query": "When did the new appraisal requirements take effect?"},
  {"query": "Has the maximum DTI ratio changed in 2025?"},
  {"query": "What's the current interest rate for HomeReady loans?"}
]
```

---

## 6. Annotation Schema

Each query in your golden dataset should include rich metadata for evaluation and analysis.

### 6.1 Core Fields (Required)

```json
{
  "query_id": "string (unique identifier)",
  "query": "string (the actual question)",
  "ground_truth": "string (verified correct answer)",
  "category": "enum (factual|eligibility|procedural|comparative|temporal)",
  "created_date": "ISO 8601 date"
}
```

### 6.2 Extended Fields (Recommended)

```json
{
  "difficulty": "enum (easy|medium|hard)",
  "expected_doc_ids": ["array of relevant document IDs"],
  "expected_sections": ["array of section references"],
  "expected_entities": ["array of key entities in answer"],
  "requires_multi_hop": "boolean",
  "requires_synthesis": "boolean",
  "expected_latency_ms": "integer (SLA for this query type)",
  "priority": "enum (p0|p1|p2|p3)",
  "source": "enum (historical|expert|synthetic|user_testing)",
  "created_by": "string (annotator ID)",
  "verified_by": "string (reviewer ID)",
  "notes": "string (any special considerations)"
}
```

### 6.3 Difficulty Levels

**Easy (33%)**:
- Single-hop retrieval
- Clear terminology
- Unambiguous answer
- Example: "What is the maximum LTV?"

**Medium (50%)**:
- May require 2-3 chunks
- Some domain knowledge needed
- Moderate reasoning
- Example: "How do you calculate commission income for qualifying?"

**Hard (17%)**:
- Multi-hop reasoning
- Requires synthesis across sections
- Ambiguous or complex scenario
- Example: "Can a self-employed borrower with fluctuating income use a non-occupant co-borrower to qualify for a 2-unit property with 95% LTV?"

### 6.4 Relevance Judgments

For search evaluation, you need to label which documents/chunks are relevant:

```json
{
  "query_id": "Q042",
  "query": "What documentation is required for self-employed borrowers?",
  "relevant_docs": [
    {
      "doc_id": "freddie_mac_5401",
      "section": "5401.2",
      "relevance": 3,  // 0=not relevant, 1=marginally, 2=relevant, 3=highly relevant
      "is_essential": true
    },
    {
      "doc_id": "freddie_mac_5402",
      "section": "5402.1",
      "relevance": 2,
      "is_essential": false
    }
  ],
  "irrelevant_docs": [
    {
      "doc_id": "freddie_mac_4203",
      "reason": "About salaried income, not self-employed"
    }
  ]
}
```

---

## 7. Step-by-Step Creation Process

### Phase 1: Planning (Week 1)

**Step 1.1: Define Scope**
- [ ] Determine target dataset size (50/150/300)
- [ ] Identify document sources covered
- [ ] Define timeline and resources
- [ ] Assign annotators and reviewers

**Step 1.2: Create Annotation Guidelines**

Document with examples of:
- What makes a good query
- How to write ground truth answers
- Difficulty level criteria
- Category definitions
- Edge case handling

**Step 1.3: Set Up Tools**

Choose format:
- Spreadsheet (Google Sheets, Excel) - easiest for small teams
- JSONL files - better for programmatic access
- Annotation platform (Label Studio, Prodigy) - best for scale

### Phase 2: Query Collection (Weeks 1-2)

**Step 2.1: Gather Historical Queries**
```python
# Extract from various sources
historical_queries = []

# Support tickets
historical_queries.extend(extract_from_tickets('support_tickets.csv'))

# Email logs (with permission)
historical_queries.extend(extract_from_emails('email_archive'))

# Chat transcripts
historical_queries.extend(extract_from_chats('chat_logs.json'))

# Deduplicate
unique_queries = deduplicate(historical_queries, threshold=0.85)
```

**Step 2.2: Expert Query Generation**

Hold 2-hour workshop with 3-5 experts:
1. Explain golden dataset purpose
2. Share query templates by category
3. Review sample documents together
4. Each expert generates 10-15 queries
5. Group discussion to refine

**Step 2.3: Synthetic Augmentation**

```python
# Generate paraphrases of existing queries
for query in seed_queries:
    paraphrases = generate_paraphrases(query, n=3)
    synthetic_queries.extend(paraphrases)

# Generate from document sections
for section in important_sections:
    new_queries = generate_from_section(section.text)
    synthetic_queries.extend(new_queries)
```

**Step 2.4: User Testing**

- Schedule 5-10 user sessions (30-45 min each)
- Present realistic scenarios
- Ask users to formulate queries naturally
- Record verbatim (with consent)
- Note follow-up questions and clarifications

### Phase 3: Annotation (Weeks 2-3)

**Step 3.1: Assign Queries to Annotators**

Split dataset among 2-3 annotators:
- Each query gets 1 primary annotator
- 20% of queries get dual annotation (inter-annotator agreement check)
- Senior expert reviews all annotations

**Step 3.2: Write Ground Truth Answers**

Guidelines for annotators:
```
✅ DO:
- Use exact language from source documents when possible
- Cite specific sections (e.g., "Section 5401.2")
- Include key numbers and dates
- Be comprehensive but concise (2-4 sentences ideal)
- Note if answer requires multiple sources

❌ DON'T:
- Paraphrase unnecessarily
- Add information not in documents
- Make assumptions about edge cases
- Use overly technical jargon if document doesn't
- Leave ambiguity in the answer
```

**Example Good Annotation**:
```json
{
  "query": "What is the maximum DTI for HomeReady loans?",
  "ground_truth": "HomeReady loans allow a maximum debt-to-income ratio of 50% with strong compensating factors. Standard maximum is 45%. Manual underwriting typically requires DTI not exceeding 36% housing and 43% total (Section 5301.2, Fannie Mae Selling Guide).",
  "expected_sections": ["Section 5301.2"],
  "difficulty": "medium",
  "notes": "Answer varies by underwriting method"
}
```

**Step 3.3: Label Relevant Documents**

For each query, identify:
1. **Must-have documents**: Essential for answering (relevance=3)
2. **Should-have documents**: Helpful context (relevance=2)
3. **Nice-to-have documents**: Related but not necessary (relevance=1)

### Phase 4: Quality Assurance (Week 3-4)

**Step 4.1: Calculate Inter-Annotator Agreement**

For queries with dual annotation:
```python
from sklearn.metrics import cohen_kappa_score

# Compare category labels
kappa = cohen_kappa_score(annotator1_categories, annotator2_categories)
print(f"Category agreement (Cohen's Kappa): {kappa:.2f}")

# Target: Kappa > 0.70 (substantial agreement)
```

**Step 4.2: Expert Review**

Senior expert reviews all annotations for:
- [ ] Factual accuracy
- [ ] Completeness
- [ ] Consistency with guidelines
- [ ] Source citations correct

**Step 4.3: Pilot Testing**

Before finalizing:
1. Run dataset against current system
2. Identify queries that fail completely
3. Check if failures are due to:
   - Poor query formulation (fix query)
   - Incorrect ground truth (fix annotation)
   - Legitimate system gap (keep as-is, track improvement)

### Phase 5: Finalization (Week 4)

**Step 5.1: Balance Dataset**

Check distribution:
```python
import pandas as pd

df = pd.read_json('golden_dataset.jsonl', lines=True)

# Category distribution
print(df['category'].value_counts(normalize=True))
# Target: factual 30%, eligibility 25%, procedural 20%, comparative 15%, temporal 10%

# Difficulty distribution
print(df['difficulty'].value_counts(normalize=True))
# Target: easy 33%, medium 50%, hard 17%

# Source distribution
print(df['source'].value_counts(normalize=True))
# Target: historical 40%, expert 30%, synthetic 20%, user_testing 10%
```

If imbalanced, add queries to underrepresented categories.

**Step 5.2: Create Dataset Documentation**

Include in README:
- Dataset version and creation date
- Number of queries by category
- Annotator list and qualifications
- Annotation guidelines used
- Known limitations
- Update schedule

**Step 5.3: Version Control**

```bash
# Store in git with semantic versioning
git add golden_dataset_v1.0.jsonl
git commit -m "Initial golden dataset release: 150 queries"
git tag v1.0

# Create changelog
echo "v1.0 - 2025-11-05: Initial release (150 queries)" >> CHANGELOG.md
```

---

## 8. Quality Assurance

### 8.1 Quality Metrics

Track these metrics during creation:

Metric | Target | How to Measure
-------|--------|---------------
Inter-annotator agreement | Kappa > 0.70 | Dual-annotate 20% sample
Query diversity (unique bigrams) | > 80% | `len(unique_bigrams) / len(total_bigrams)`
Average query length | 10-15 words | `mean(query.split())`
Ground truth completeness | 100% | All queries have verified answers
Category balance | ±5% from target | Check distribution
Difficulty balance | ±10% from target | Check distribution

### 8.2 Common Quality Issues

**Issue 1: Queries Too Similar**
```
❌ Bad:
- "What is the maximum LTV?"
- "What is the max LTV?"
- "What's the LTV limit?"

✅ Better:
- "What is the maximum LTV?"
- "What LTV is allowed for investment properties?"
- "Can I exceed 80% LTV with mortgage insurance?"
```

**Issue 2: Unanswerable Queries**
```
❌ Bad:
- "What will interest rates be next year?" (speculation)
- "Should I choose 15 or 30 year term?" (advice)
- "Why did my loan get denied?" (case-specific)

✅ Better:
- "What are the current interest rate ranges?"
- "What are the DTI requirements for 15-year vs 30-year mortgages?"
- "What are common reasons for loan denial?"
```

**Issue 3: Ambiguous Ground Truth**
```
❌ Bad:
Query: "What documentation is needed?"
Ground Truth: "Income and asset documentation"
(Too vague - what type of borrower?)

✅ Better:
Query: "What documentation is needed to verify income for a salaried W-2 employee?"
Ground Truth: "Pay stubs covering 30 days, W-2s for most recent 2 years, and written verification of employment (VOE) per Section 5401.1"
```

### 8.3 Review Checklist

Before finalizing dataset, verify:

- [ ] All queries are grammatically complete questions
- [ ] No duplicate queries (>85% similarity)
- [ ] All ground truth answers cite source sections
- [ ] Category labels are consistent
- [ ] Difficulty ratings match complexity
- [ ] No placeholder or "TODO" entries
- [ ] Special characters properly escaped (for JSON/JSONL)
- [ ] Query IDs are unique and sequential
- [ ] Dataset README is complete
- [ ] Version number assigned

---

## 9. Maintenance & Updates

A golden dataset is not static - it evolves with your system and domain.

### 9.1 Update Triggers

**Immediate Updates** (within 1 week):
- 📄 Document source updated (new policy effective)
- 🐛 Ground truth error discovered
- 🚨 Critical query missing (user escalation)

**Quarterly Updates** (every 3 months):
- ➕ Add 10-20 new queries from production failures
- 🔄 Refresh temporal queries (dates, limits)
- 📊 Rebalance categories if drift detected

**Annual Updates** (once per year):
- 🧹 Remove deprecated queries
- 📈 Expand dataset by 20-30%
- 🔍 Full quality audit and inter-annotator agreement check

### 9.2 Versioning Strategy

Use semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (schema change, category redefinition)
- **MINOR**: New queries added, significant updates
- **PATCH**: Bug fixes (ground truth corrections)

Example:
```
v1.0.0 - 2025-11-01: Initial release (150 queries)
v1.1.0 - 2025-12-15: Added 25 edge case queries
v1.1.1 - 2026-01-10: Corrected 3 ground truth errors
v1.2.0 - 2026-02-01: Quarterly update (20 new queries, updated 2025 limits)
v2.0.0 - 2026-05-01: Schema change (added multi-hop field)
```

### 9.3 Tracking Production Failures

Continuously improve by monitoring live system:

```python
# Log queries that fail evaluation criteria
def log_failure(query, reason, score):
    """Add failed production queries to review queue"""
    failure_log.append({
        'query': query,
        'reason': reason,
        'score': score,
        'timestamp': datetime.now(),
        'review_status': 'pending'
    })
    
    # If failure rate high, flag for dataset addition
    if score < 0.50:  # Threshold for critical failure
        candidate_for_golden_dataset.append(query)

# Monthly review
def review_candidates():
    """Expert reviews failed queries for dataset inclusion"""
    for query in candidate_for_golden_dataset:
        if is_valid_query(query) and not is_duplicate(query):
            add_to_golden_dataset(query)
```

### 9.4 Deprecation Policy

Remove queries when:
- Policy no longer exists (deprecated programs)
- Question no longer relevant (outdated limits)
- Duplicate of newer, better-formed query
- Consistently scores 100% (no longer challenging)

Before removing:
1. Mark as `deprecated` for 1 version cycle
2. Document reason in changelog
3. Archive in separate file (historical dataset)

---

## 10. Tools & Templates

### 10.1 Spreadsheet Template

[Download: golden_dataset_template.xlsx]

**Columns**:
- A: Query ID
- B: Query
- C: Category
- D: Difficulty
- E: Ground Truth
- F: Expected Sections
- G: Source
- H: Created By
- I: Verified By
- J: Notes

**Formulas**:
- Category distribution: `=COUNTIF(C:C,"factual")/COUNTA(C:C)`
- Average query length: `=AVERAGE(LEN(B:B)-LEN(SUBSTITUTE(B:B," ",""))+1)`

### 10.2 JSONL Template

```jsonl
{"query_id": "Q001", "query": "What is the maximum LTV ratio?", "category": "factual", "difficulty": "easy", "ground_truth": "The maximum LTV ratio is 97% for eligible primary residences with mortgage insurance.", "expected_sections": ["Section 4.2.1"], "source": "expert", "created_by": "expert_1", "created_date": "2025-11-01"}
{"query_id": "Q002", "query": "How do you calculate self-employment income?", "category": "procedural", "difficulty": "medium", "ground_truth": "Self-employment income is calculated by averaging the most recent two years of tax returns. Use Schedule C net profit, add back non-cash deductions like depreciation, and subtract any unusual one-time income.", "expected_sections": ["Section 5401.3", "Section 5401.4"], "source": "historical", "created_by": "expert_2", "created_date": "2025-11-01"}
```

### 10.3 Python Validation Script

```python
import json
import jsonschema

# Define schema
schema = {
    "type": "object",
    "required": ["query_id", "query", "category", "ground_truth"],
    "properties": {
        "query_id": {"type": "string", "pattern": "^Q[0-9]{3,}$"},
        "query": {"type": "string", "minLength": 10},
        "category": {"enum": ["factual", "eligibility", "procedural", "comparative", "temporal"]},
        "difficulty": {"enum": ["easy", "medium", "hard"]},
        "ground_truth": {"type": "string", "minLength": 20},
        "expected_sections": {"type": "array", "items": {"type": "string"}},
        "source": {"enum": ["historical", "expert", "synthetic", "user_testing"]}
    }
}

# Validate dataset
def validate_golden_dataset(filepath):
    errors = []
    query_ids = set()
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                
                # Schema validation
                jsonschema.validate(item, schema)
                
                # Uniqueness check
                if item['query_id'] in query_ids:
                    errors.append(f"Line {i}: Duplicate query_id {item['query_id']}")
                query_ids.add(item['query_id'])
                
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")
            except jsonschema.ValidationError as e:
                errors.append(f"Line {i}: Schema violation - {e.message}")
    
    if errors:
        print(f"❌ Validation failed with {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10
            print(f"  - {error}")
    else:
        print(f"✅ Dataset valid: {len(query_ids)} queries")
    
    return len(errors) == 0

# Usage
validate_golden_dataset('golden_dataset_v1.0.jsonl')
```

### 10.4 Category Balance Check

```python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_dataset_balance(filepath):
    df = pd.read_json(filepath, lines=True)
    
    # Target distribution
    targets = {
        'factual': 0.30,
        'eligibility': 0.25,
        'procedural': 0.20,
        'comparative': 0.15,
        'temporal': 0.10
    }
    
    # Actual distribution
    actual = df['category'].value_counts(normalize=True).to_dict()
    
    # Compare
    print("Category Balance:")
    print("-" * 50)
    for category in targets:
        target_pct = targets[category] * 100
        actual_pct = actual.get(category, 0) * 100
        diff = actual_pct - target_pct
        status = "✅" if abs(diff) < 5 else "⚠️"
        print(f"{status} {category:15} Target: {target_pct:5.1f}%  Actual: {actual_pct:5.1f}%  Diff: {diff:+5.1f}%")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Category distribution
    df['category'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title('Category Distribution')
    ax1.set_ylabel('Count')
    
    # Difficulty distribution
    df['difficulty'].value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax2.set_title('Difficulty Distribution')
    
    plt.tight_layout()
    plt.savefig('dataset_balance.png')
    print("\n📊 Visualization saved to dataset_balance.png")

# Usage
analyze_dataset_balance('golden_dataset_v1.0.jsonl')
```

### 10.5 Query Similarity Checker

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_queries(filepath, threshold=0.85):
    """Find queries that are too similar (potential duplicates)"""
    
    # Load queries
    with open(filepath, 'r') as f:
        items = [json.loads(line) for line in f]
    
    queries = [item['query'] for item in items]
    query_ids = [item['query_id'] for item in items]
    
    # Compute TF-IDF similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(queries)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find similar pairs
    duplicates = []
    n = len(queries)
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i][j] > threshold:
                duplicates.append({
                    'query1_id': query_ids[i],
                    'query1': queries[i],
                    'query2_id': query_ids[j],
                    'query2': queries[j],
                    'similarity': similarity_matrix[i][j]
                })
    
    # Report
    if duplicates:
        print(f"⚠️  Found {len(duplicates)} similar query pairs (>{threshold*100:.0f}% similarity):")
        for dup in duplicates[:10]:
            print(f"\n  {dup['query1_id']}: {dup['query1']}")
            print(f"  {dup['query2_id']}: {dup['query2']}")
            print(f"  Similarity: {dup['similarity']:.2%}")
    else:
        print(f"✅ No duplicate queries found (threshold: {threshold*100:.0f}%)")
    
    return duplicates

# Usage
find_similar_queries('golden_dataset_v1.0.jsonl', threshold=0.85)
```

---

## Summary

Creating a high-quality golden dataset requires:

1. **Planning** (1 week): Define scope, create guidelines, set up tools
2. **Collection** (1-2 weeks): Gather queries from 4 sources (40% historical, 30% expert, 20% synthetic, 10% user testing)
3. **Annotation** (1-2 weeks): Write ground truth answers, label relevant documents
4. **QA** (1 week): Inter-annotator agreement, expert review, pilot testing
5. **Maintenance** (ongoing): Update quarterly, add production failures, version control

**Key Success Factors**:
- ✅ Balanced category distribution (30/25/20/15/10)
- ✅ Multiple annotators with domain expertise
- ✅ Comprehensive documentation and versioning
- ✅ Regular updates as domain evolves
- ✅ Integration with CI/CD for regression testing

**Resources Required**:
- 2-3 annotators (domain experts)
- 1 senior reviewer
- 3-4 weeks initial creation
- 2-4 hours/month ongoing maintenance

**Expected Outcome**:
A production-ready dataset of 150-300 queries that enables objective evaluation, A/B testing, and continuous improvement of your AI Agent search system.

---

**Next Steps**:
1. Review the evaluation JSONL files already created (`freddie_mac_eval.jsonl`, `fannie_mae_eval.jsonl`)
2. Use this guide to expand from 30 to 150+ queries per index
3. Implement validation scripts to ensure quality
4. Integrate with Azure AI Foundry evaluation workflow (Section 7.5 of main guide)

**Related Documents**:
- `sample-search-strategies.md` - Section 7: Testing & Evaluation Strategy
- `freddie_mac_eval.jsonl` - 30 baseline queries for Freddie Mac index
- `fannie_mae_eval.jsonl` - 30 baseline queries for Fannie Mae index
