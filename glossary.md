# Glossary of Search and AI Terms

A comprehensive reference of technical terms, algorithms, and concepts used in Azure AI Search, vector databases, and information retrieval systems.

---

## A

### **ANN (Approximate Nearest Neighbor)**
A family of algorithms designed to find the nearest neighbors to a query vector in high-dimensional space with high recall but without the computational cost of exhaustive search. ANN algorithms trade perfect accuracy for speed, typically achieving 95-99% recall while being 10-1000× faster than exact search. Common ANN algorithms include HNSW, IVF, and LSH.

**Example**: Finding the 10 most similar products to a query embedding among 1 million products in <100ms using HNSW instead of ~10 seconds with exact search.

**Related**: HNSW, Vector Search, Recall

---

### **API Key**
A secret token used to authenticate requests to Azure AI Search. Azure provides two types: **admin keys** (full read-write access) and **query keys** (read-only access for searching). Best practice is to use query keys in client applications and rotate admin keys regularly.

**Security Note**: Never embed admin keys in client-side code or commit them to version control.

**Related**: Authentication, RBAC

---

### **Analyzer**
A text processing pipeline in Azure AI Search that converts input text into searchable tokens. An analyzer consists of:
1. **Character filters**: Remove or replace characters (e.g., HTML stripping)
2. **Tokenizer**: Split text into tokens (e.g., whitespace, ngram)
3. **Token filters**: Transform tokens (e.g., lowercase, stemming, stopword removal)

**Example**: The `en.microsoft` analyzer converts "Running quickly!" → ["run", "quick"]

**Related**: Tokenization, Stemming, Stopwords

---

### **Azure AI Search**
Microsoft's cloud-based search-as-a-service platform (formerly Azure Cognitive Search) that provides full-text search, vector search, hybrid search, and semantic ranking capabilities. Supports BM25 scoring, HNSW vector indexing, and integrated AI enrichment.

**Pricing Tiers**: Free, Basic, Standard (S1-S3), Storage Optimized (L1-L2)

**Related**: BM25, Vector Search, Semantic Search

---

## B

### **BM25 (Best Matching 25)**
A probabilistic ranking function used for full-text search that scores documents based on term frequency (TF) and inverse document frequency (IDF) with saturation. BM25 improves on TF-IDF by:
- **Term frequency saturation**: Diminishing returns for high term counts (prevents keyword stuffing)
- **Document length normalization**: Adjusts for document length variation
- **Tunable parameters**: k1 (term frequency saturation) and b (length normalization)

**Formula**: `BM25(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))`

**Default parameters**: k1=1.2, b=0.75

**Related**: TF-IDF, Full-Text Search, Scoring Profile

---

### **Batch Size**
The number of items processed together in a single operation. For embedding generation, larger batch sizes (100-1000) improve throughput but increase latency per batch. For indexing, Azure AI Search supports batches up to 1000 documents or 16MB per request.

**Optimization**: Use larger batches (500-1000) for bulk operations, smaller batches (10-50) for real-time updates.

**Related**: Throughput, Indexing

---

### **Boosting**
A technique to increase or decrease the importance of specific fields or terms in search results. In Azure AI Search:
- **Field boosting**: `search.in(title, 'laptop^3')` makes title matches 3× more important
- **Scoring profiles**: Complex boosting based on field values, freshness, or custom functions

**Example**: Boost recent documents: `freshness(publishDate, boostRecent=P7D, constant=2)` doubles scores for documents from the last 7 days.

**Related**: Scoring Profile, Relevance Tuning

---

## C

### **Caching**
Storing computed results for reuse to reduce latency and cost:
- **Query embedding caching**: Store embeddings of common queries (50-80% hit rate reduces API calls by half)
- **Result caching**: Cache search results for identical queries (CDN or Redis)
- **Analyzer caching**: Azure automatically caches analyzer results

**Cost Impact**: Caching 1000 common query embeddings saves ~$50/month for high-traffic applications.

**Related**: Performance Optimization, Embedding Generation

---

### **Cosine Similarity**
A measure of similarity between two vectors based on the cosine of the angle between them, ranging from -1 (opposite) to 1 (identical). For normalized vectors, cosine similarity equals dot product.

**Formula**: `cos(θ) = (A · B) / (||A|| × ||B||)`

**When to use**: Preferred for text embeddings because it's invariant to vector magnitude (focuses on direction/meaning, not length).

**Example**: Query embedding [0.5, 0.5, 0.5, 0.5] vs document [0.7, 0.7, 0.1, 0.1] = 0.8 similarity

**Related**: Dot Product, Euclidean Distance, Vector Search

---

### **Cross-Encoder**
A neural model that scores query-document pairs jointly by encoding both together, producing more accurate relevance scores than bi-encoders. Cross-encoders are used in **reranking** after initial retrieval because they're too slow for first-pass search (10-100× slower than bi-encoders).

**Architecture**: `[CLS] query [SEP] document [SEP]` → BERT/Transformer → relevance score

**Use case**: Rerank top 20-100 candidates retrieved by BM25 or vector search.

**Related**: Bi-Encoder, Reranking, Semantic Search

---

## D

### **Dense Vectors**
High-dimensional numerical representations (embeddings) where most values are non-zero, typically produced by neural networks. Contrasts with **sparse vectors** (mostly zeros) used in traditional TF-IDF.

**Characteristics**:
- Dimensions: 384-3072 (common models)
- Storage: 1.5-12 KB per vector (768 dims × 4 bytes/float = 3 KB)
- Semantic: Similar meanings → similar vectors

**Related**: Embeddings, Vector Search, Sparse Vectors

---

### **Dimensions (Vector Dimensionality)**
The number of components in a vector embedding. Higher dimensions capture more semantic nuance but increase:
- **Storage**: Linear growth (768d = 3KB, 3072d = 12KB per vector)
- **Compute**: Quadratic growth for similarity calculations
- **Latency**: 20-40% increase per 2× dimensions

**Common sizes**:
- 384: Small models (all-MiniLM-L6-v2) - fast, lower quality
- 768: Medium models (BERT, sentence-transformers) - balanced
- 1536: Large models (text-embedding-ada-002) - high quality
- 3072: Largest models (text-embedding-3-large) - best quality

**Related**: Embeddings, Vector Search

---

### **Dot Product**
A similarity metric computed as the sum of element-wise multiplication: `A · B = Σ(Ai × Bi)`. Faster than cosine similarity but sensitive to vector magnitude.

**When to use**: When vectors are normalized (unit length), dot product equals cosine similarity.

**Performance**: ~2× faster than cosine (no normalization needed).

**Related**: Cosine Similarity, Euclidean Distance

---

## E

### **Embeddings**
Dense vector representations of text, images, or other data produced by neural networks, capturing semantic meaning in high-dimensional space. Similar concepts have similar embeddings.

**Generation**:
- **Text**: OpenAI (text-embedding-3-large), Azure OpenAI, Sentence Transformers
- **Images**: CLIP, ResNet
- **Multimodal**: CLIP (text + image in same space)

**Properties**:
- **Semantic**: "king" - "man" + "woman" ≈ "queen"
- **Contextual**: Same word in different contexts → different embeddings

**Related**: Vector Search, Dense Vectors

---

### **Euclidean Distance (L2 Distance)**
The straight-line distance between two points in vector space: `d = √(Σ(Ai - Bi)²)`. Lower distance = higher similarity.

**When to use**: Geographical coordinates, when magnitude matters.

**When NOT to use**: Text embeddings (cosine similarity is better).

**Related**: Cosine Similarity, Dot Product

---

### **Exact Search (Brute Force)**
Exhaustive comparison of a query vector against all vectors in the index to find true nearest neighbors. Guarantees 100% recall but computationally expensive:
- **Time complexity**: O(N × D) for N vectors of D dimensions
- **Latency**: ~10ms per 1000 vectors (768d)

**Use case**: Small datasets (<10K vectors) or when perfect recall is required.

**Related**: ANN, HNSW

---

## F

### **Facets**
Aggregations of field values used for filtering and navigation in search UIs. Azure AI Search automatically computes facet counts.

**Example**:
```json
{
  "facets": ["category", "price,interval:100"],
  "results": {
    "category": [{"value": "Laptops", "count": 1523}],
    "price": [{"value": "0-100", "count": 234}]
  }
}
```

**Related**: Filtering, Search UI

---

### **Field**
A named property in a search index document. Field types include:
- **Edm.String**: Text (searchable, filterable, facetable)
- **Collection(Edm.Single)**: Vector (searchable with vector algorithms)
- **Edm.Int32/Double**: Numeric (filterable, sortable, facetable)
- **Edm.DateTimeOffset**: Dates (filterable, sortable)

**Related**: Index Schema, Data Type

---

### **Filtering**
Applying conditions to narrow search results before or after scoring. Pre-filtering (before search) is 2-5× faster than post-filtering.

**OData syntax**: `$filter=category eq 'Laptops' and price le 1000`

**Performance**: Filter on indexed fields for <10ms overhead.

**Related**: Facets, Query Optimization

---

### **Full-Text Search**
Search based on keyword matching and text analysis (tokenization, stemming). Uses inverted indexes for fast retrieval and BM25 for scoring.

**Capabilities**:
- Phrase queries: `"gaming laptop"`
- Wildcards: `lapt*`
- Fuzzy search: `laptop~1` (1 edit distance)
- Boolean: `gaming AND laptop`

**Related**: BM25, Analyzer, Inverted Index

---

## H

### **HNSW (Hierarchical Navigable Small World)**
A graph-based ANN algorithm that organizes vectors in a multi-layer graph structure for efficient nearest neighbor search. HNSW achieves:
- **Recall**: 95-99% at top-10
- **Speed**: 10-1000× faster than exact search
- **Scalability**: Sub-linear query time O(log N)

**Parameters**:
- **m**: Connections per node (4-64, default 4). Higher = better recall, more memory
- **efConstruction**: Build-time accuracy (100-1000, default 400). Higher = better index quality, slower build
- **efSearch**: Query-time accuracy (100-1000, default 500). Higher = better recall, slower queries

**Memory**: ~150-200 bytes per vector (768d, m=4)

**Related**: ANN, Vector Search, Graph Index

---

### **Hybrid Search**
Combining multiple search techniques (typically BM25 + vector search) and fusing results for superior relevance. Uses Reciprocal Rank Fusion (RRF) to merge rankings.

**Benefits**:
- **Complementary**: BM25 for exact matches, vectors for semantics
- **Robustness**: If one mode fails, the other provides fallback
- **Performance**: 15-30% improvement over single-mode search

**Formula**: `RRF_score = α × RRF_text + (1-α) × RRF_vector`

**Related**: RRF, BM25, Vector Search

---

## I

### **IDF (Inverse Document Frequency)**
A measure of how rare a term is across all documents in a corpus. Rare terms have higher IDF values and contribute more to relevance scores.

**Formula**: `IDF(term) = log((N - n + 0.5) / (n + 0.5))` where N = total docs, n = docs containing term

**Example**: 
- "the" appears in 95% of docs → IDF ≈ 0.1 (low weight)
- "kubernetes" appears in 2% of docs → IDF ≈ 3.9 (high weight)

**Related**: TF-IDF, BM25

---

### **Index**
A data structure optimizing search and retrieval. Azure AI Search uses:
- **Inverted index**: Maps terms → document IDs (for full-text search)
- **HNSW graph**: Maps vectors → nearest neighbors (for vector search)
- **Column store**: Stores field values for filtering/faceting

**Related**: Inverted Index, HNSW

---

### **Inverted Index**
A mapping from terms to the documents containing them, enabling fast full-text search. Structure:

```
Term        → Documents (with positions)
"laptop"    → [doc1: [0, 15], doc5: [3], doc12: [7]]
"gaming"    → [doc1: [5], doc3: [0, 22]]
```

**Query "gaming laptop"**: Intersect doc lists → [doc1] appears in both

**Related**: Full-Text Search, BM25

---

## K

### **k (Number of Results)**
The number of nearest neighbors to retrieve in vector search. Higher k:
- **Improves recall**: More candidates for filtering/reranking
- **Increases latency**: Linear growth (k=10: 20ms, k=100: 120ms)
- **Increases cost**: More data transfer

**Guidelines**:
- Simple search: k=10-20
- Search + filtering: k=50-100 (over-fetch before filter)
- Hybrid search: k=50-100 (better fusion)

**Related**: Vector Search, Over-Fetching

---

### **k1 Parameter (BM25)**
Controls term frequency saturation in BM25. Higher k1 = less saturation (term frequency has more impact).

**Range**: 1.0-2.0 (default: 1.2)
- k1=1.0: Earlier saturation (good for short documents)
- k1=2.0: Later saturation (good for long documents)

**Related**: BM25, b Parameter

---

## L

### **Latency**
Time between query submission and result delivery. Components:
- **Network**: 10-50ms (varies by region)
- **Query processing**: 5-20ms (embedding generation if vector search)
- **Search execution**: 20-150ms (depends on index size, k value)
- **Result processing**: 5-10ms

**Targets**:
- P50: <100ms
- P95: <200ms
- P99: <500ms

**Related**: Performance Optimization, Throughput

---

### **LSH (Locality-Sensitive Hashing)**
An ANN algorithm that hashes similar vectors to the same buckets using special hash functions. Faster than HNSW for very high-dimensional data (>2000d) but lower recall.

**Use case**: Extremely large datasets (>100M vectors) where HNSW memory is prohibitive.

**Related**: ANN, HNSW

---

## M

### **Mean Reciprocal Rank (MRR)**
An evaluation metric measuring how highly the first relevant result is ranked. MRR is the average of reciprocal ranks across queries.

**Formula**: `MRR = (1/|Q|) × Σ(1/rank_i)` where rank_i is the position of the first relevant result

**Example**: 
- Query 1: First relevant at position 1 → 1/1 = 1.0
- Query 2: First relevant at position 3 → 1/3 = 0.333
- MRR = (1.0 + 0.333) / 2 = 0.667

**When to use**: Single relevant result per query (e.g., question answering)

**Related**: NDCG, Precision, Recall

---

## N

### **NDCG (Normalized Discounted Cumulative Gain)**
The primary metric for evaluating ranked search results, accounting for both relevance and position. Higher positions contribute more to the score.

**Formula**: `NDCG@k = DCG@k / IDCG@k`
- **DCG**: `Σ(2^rel_i - 1) / log2(i + 1)` (actual ranking)
- **IDCG**: DCG of the ideal ranking

**Range**: 0.0-1.0 (1.0 = perfect ranking)

**Example**: Top-3 results [high, low, high] vs ideal [high, high, low] → NDCG@3 = 0.89

**Related**: DCG, MRR, Evaluation Metrics

---

### **Normalization (Vector)**
Scaling a vector to unit length (magnitude = 1), making cosine similarity equal to dot product.

**Formula**: `v_norm = v / ||v||` where `||v|| = √(Σvi²)`

**Benefit**: 2× faster similarity calculation (dot product instead of cosine)

**Related**: Cosine Similarity, Dot Product

---

## O

### **Over-Fetching**
Retrieving more results than needed to compensate for filtering or improve fusion quality. Common in:
- **Vector + filter**: Fetch k=100, filter → return 10
- **Hybrid search**: Fetch k=50 per mode for better RRF fusion

**Guidelines**: Over-fetch 5-10× when filtering is selective.

**Related**: k, Hybrid Search, Filtering

---

## P

### **Partition Key**
In Azure Cosmos DB, the property used to distribute data across physical partitions. Critical for performance and scalability.

**Best practices**:
- High cardinality (many unique values)
- Even distribution
- Supports common query patterns

**Related**: Azure Cosmos DB, Sharding

---

### **Precision**
The fraction of retrieved results that are relevant.

**Formula**: `Precision@k = (relevant results in top-k) / k`

**Example**: Top-10 results contain 7 relevant → Precision@10 = 0.7

**Trade-off**: Higher precision often means lower recall.

**Related**: Recall, F1 Score

---

## Q

### **Query Embedding**
The vector representation of a search query, generated using the same embedding model as document embeddings. Must use identical model/version for consistency.

**Caching**: Cache common queries (50-80% hit rate) to reduce API calls and latency.

**Related**: Embeddings, Vector Search

---

### **Query Routing**
Dynamically selecting or weighting search strategies based on query characteristics:
- SKU pattern → BM25 (α=0.8)
- Question format → Vector (α=0.2)
- Default → Hybrid (α=0.5)

**Implementation**: Rule-based or ML classifier

**Related**: Hybrid Search, Weighting

---

## R

### **Recall**
The fraction of all relevant results that were retrieved.

**Formula**: `Recall@k = (relevant results in top-k) / (total relevant results)`

**Example**: 7 retrieved out of 12 total relevant → Recall = 0.58

**ANN context**: Recall@10 measures how often true top-10 neighbors are in retrieved top-10 (target: >95%)

**Related**: Precision, NDCG

---

### **Reranking**
A two-stage retrieval process:
1. **First stage**: Fast retrieval (BM25/vector) gets top-100 candidates
2. **Second stage**: Expensive model (cross-encoder) reranks top-100 → top-10

**Benefit**: 10-20% improvement in NDCG while maintaining low latency (only rerank 100, not millions)

**Related**: Cross-Encoder, Semantic Ranking

---

### **RRF (Reciprocal Rank Fusion)**
An algorithm for merging multiple ranked lists by summing reciprocal ranks. Used in hybrid search to combine BM25 and vector results.

**Formula**: `RRF(doc) = Σ 1/(k + rank_mode(doc))` across all modes (k=60 default)

**Example**:
- Doc A: BM25 rank=1, vector rank=5 → RRF = 1/61 + 1/65 = 0.0318
- Doc B: BM25 rank=3, vector rank=2 → RRF = 1/63 + 1/62 = 0.0320
- Doc B ranks higher (0.0320 > 0.0318)

**Benefits**: Scale-invariant (works with different scoring ranges), proven effective

**Related**: Hybrid Search, Result Fusion

---

## S

### **Scoring Profile**
A custom configuration in Azure AI Search that modifies BM25 scores based on:
- **Field weights**: `title^3` (3× importance)
- **Freshness**: Boost recent documents
- **Magnitude**: Boost high-value fields (price, rating)
- **Distance**: Boost geographically close results

**Example**: E-commerce scoring prioritizing title matches, recent products, highly rated items.

**Related**: BM25, Boosting, Relevance Tuning

---

### **Semantic Search (Azure)**
Azure AI Search's built-in semantic ranking using Microsoft's deep learning models. Adds:
- **Semantic reranking**: Reorders top-50 BM25 results using BERT-style model
- **Captions**: Extracts relevant passages
- **Answers**: Generates direct answers when possible

**Cost**: +$500/month (Standard pricing)

**Improvement**: 10-25% NDCG boost over BM25 alone

**Related**: Reranking, Cross-Encoder

---

### **Sharding**
Distributing an index across multiple partitions for scalability. Azure AI Search automatically shards based on tier:
- S1: 1 partition
- S2: 2 partitions
- S3: 3 partitions

**Related**: Partition, Scalability

---

### **Similarity Metric**
The function used to compare vectors:
- **Cosine similarity**: Text embeddings (direction matters)
- **Dot product**: Normalized vectors (2× faster than cosine)
- **Euclidean distance**: Coordinates, when magnitude matters

**Related**: Cosine Similarity, Dot Product, Euclidean Distance

---

### **Sparse Vectors**
Vectors where most values are zero, used in traditional IR (TF-IDF, BM25). Efficient storage (only store non-zero values) but less semantic than dense vectors.

**Example**: TF-IDF vector for "laptop gaming" in 50K vocabulary → [0, 0, ..., 0.8, 0, ..., 0.6, ..., 0] (only 2 non-zero)

**Related**: Dense Vectors, TF-IDF

---

### **Stemming**
Reducing words to their root form for matching variants.

**Example**: "running", "runs", "ran" → "run"

**Algorithms**:
- **Porter**: Aggressive (faster, less accurate)
- **Snowball**: Balanced
- **Lemmatization**: Most accurate (uses dictionary)

**Related**: Analyzer, Tokenization

---

### **Stopwords**
Common words filtered out during indexing/search ("the", "and", "is") because they provide little relevance signal and inflate index size.

**Trade-off**: Filters reduce index size by 30-40% but can break phrase queries ("to be or not to be" → "not" only)

**Related**: Analyzer, Token Filters

---

## T

### **TF (Term Frequency)**
The number of times a term appears in a document. Higher TF → higher relevance (but with saturation in BM25).

**Raw TF issue**: "laptop" appears 100× → dominates score (keyword stuffing)

**BM25 solution**: Saturated TF prevents this gaming

**Related**: IDF, BM25

---

### **TF-IDF (Term Frequency-Inverse Document Frequency)**
A classical relevance scoring function combining term frequency (how often term appears in doc) and inverse document frequency (how rare the term is).

**Formula**: `TF-IDF(term, doc) = TF(term, doc) × IDF(term)`

**Limitation**: No term frequency saturation (BM25 improves on this)

**Related**: BM25, IDF

---

### **Throughput**
The number of requests processed per unit time (queries per second, QPS). Azure AI Search throughput depends on:
- **Tier**: S1 (15 QPS), S2 (60 QPS), S3 (180 QPS)
- **Replicas**: Linear scaling (3 replicas = 3× throughput)
- **Query complexity**: Simple BM25 (20ms) vs complex vector (150ms)

**Related**: Latency, Scalability

---

### **Tokenization**
Splitting text into individual tokens (words/subwords) for indexing and search.

**Tokenizers**:
- **Standard**: Unicode word boundaries
- **Whitespace**: Split on spaces only
- **NGram**: Character sequences (ab, abc, bcd for "abcd")

**Example**: "Hello, world!" → ["Hello", "world"]

**Related**: Analyzer, Stemming

---

## V

### **Vector Database**
A specialized database optimized for storing and searching high-dimensional vectors using ANN algorithms. Examples: Pinecone, Weaviate, Azure AI Search (hybrid).

**Capabilities**:
- Fast vector search (HNSW, IVF)
- Metadata filtering
- Hybrid search (vectors + metadata)

**Related**: Vector Search, ANN, HNSW

---

### **Vector Search**
Finding documents with vectors most similar to a query vector using ANN algorithms. Enables semantic search, recommendation, and similarity matching.

**Process**:
1. Generate query embedding
2. ANN search in HNSW index
3. Compute similarities
4. Return top-k results

**Use cases**: Semantic search, image similarity, recommendation

**Related**: Embeddings, HNSW, ANN

---

## W

### **Weighting (Hybrid Search)**
Controlling the relative influence of different search modes in hybrid search using weights (α):

**Formula**: `score = α_text × score_text + α_vector × score_vector`

**Strategies**:
- **Balanced**: α=0.5/0.5 (default)
- **Text-heavy**: α=0.7/0.3 (SKU searches)
- **Vector-heavy**: α=0.3/0.7 (semantic queries)

**Related**: Hybrid Search, Query Routing

---

## Acronyms Quick Reference

| Acronym | Full Name | Category |
|---------|-----------|----------|
| ANN | Approximate Nearest Neighbor | Algorithm |
| BM25 | Best Matching 25 | Scoring |
| DCG | Discounted Cumulative Gain | Metric |
| HNSW | Hierarchical Navigable Small World | Algorithm |
| IDF | Inverse Document Frequency | Scoring |
| IVF | Inverted File Index | Algorithm |
| KQL | Kusto Query Language | Query Language |
| LSH | Locality-Sensitive Hashing | Algorithm |
| MRR | Mean Reciprocal Rank | Metric |
| NDCG | Normalized Discounted Cumulative Gain | Metric |
| QPS | Queries Per Second | Performance |
| RBAC | Role-Based Access Control | Security |
| RRF | Reciprocal Rank Fusion | Algorithm |
| TF | Term Frequency | Scoring |
| TF-IDF | Term Frequency-Inverse Document Frequency | Scoring |

---

## Additional Resources

- **[Azure AI Search Documentation](https://learn.microsoft.com/azure/search/)**: Official Microsoft documentation
- **[Vector Search Guide](./docs/09-vector-search.md)**: Detailed vector search implementation
- **[BM25 Guide](./docs/08-fulltext-search-bm25.md)**: Full-text search best practices
- **[Hybrid Search Guide](./docs/10-hybrid-search.md)**: Combining search modes
- **[Evaluation Metrics](./docs/01-core-metrics.md)**: Understanding NDCG, MRR, precision, recall

---

*Last updated: November 2025*
