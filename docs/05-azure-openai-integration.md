# Azure OpenAI Integration

Complete guide to integrating Azure OpenAI Service for embedding generation, semantic capabilities, and AI-powered search enhancements.

## 📋 Table of Contents
- [Overview](#overview)
- [Service Setup](#service-setup)
- [Embedding Generation](#embedding-generation)
- [Batch Processing](#batch-processing)
- [Cost Optimization](#cost-optimization)
- [Performance Tuning](#performance-tuning)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## Overview

### Why Azure OpenAI for Search?

Azure OpenAI provides powerful embedding models that enable:
- **Semantic Search**: Understanding query intent beyond keywords
- **Multilingual Support**: Cross-language search capabilities
- **Contextual Understanding**: Capturing document meaning and relationships
- **Similarity Matching**: Finding conceptually related content

### Architecture Overview

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Azure OpenAI Service   │
│  (Embedding Generation) │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Azure AI Search        │
│  (Vector Search)        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│  Search Results │
└─────────────────┘
```

---

## Service Setup

### Step 1: Create Azure OpenAI Resource

#### Using Azure Portal

1. Navigate to Azure Portal
2. Create a resource → Search "Azure OpenAI"
3. Configure:
   - **Resource Group**: Same as your search service
   - **Region**: `East US`, `West Europe`, or `South Central US`
   - **Name**: `openai-search-embeddings-prod`
   - **Pricing Tier**: Standard S0

#### Using Azure CLI

```bash
# Variables
RESOURCE_GROUP="rg-search-evaluation"
LOCATION="eastus"
OPENAI_NAME="openai-search-prod"

# Create OpenAI resource
az cognitiveservices account create \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --kind OpenAI \
  --sku S0 \
  --yes

# Get endpoint and key
OPENAI_ENDPOINT=$(az cognitiveservices account show \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.endpoint \
  --output tsv)

OPENAI_KEY=$(az cognitiveservices account keys list \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --query key1 \
  --output tsv)

echo "Endpoint: $OPENAI_ENDPOINT"
echo "Key: $OPENAI_KEY"
```

### Step 2: Deploy Embedding Model

```bash
# Deploy text-embedding-ada-002
az cognitiveservices account deployment create \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --deployment-name text-embedding-ada-002 \
  --model-name text-embedding-ada-002 \
  --model-version "2" \
  --model-format OpenAI \
  --sku-capacity 120 \
  --sku-name "Standard"

# Verify deployment
az cognitiveservices account deployment list \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --output table
```

### Step 3: Configure Access

```python
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Test connection
def test_openai_connection():
    """Test Azure OpenAI connection."""
    try:
        response = client.embeddings.create(
            input="Test connection",
            model="text-embedding-ada-002"
        )
        print("✅ Azure OpenAI connection successful")
        print(f"Embedding dimensions: {len(response.data[0].embedding)}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

test_openai_connection()
```

---

## Embedding Generation

### Single Text Embedding

```python
class AzureOpenAIEmbedder:
    """Azure OpenAI embedding generator."""
    
    def __init__(self, api_key, endpoint, api_version="2024-02-01"):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment_name = "text-embedding-ada-002"
        self.max_tokens = 8191  # ada-002 limit
    
    def embed_text(self, text):
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            List of 1536 floats
        """
        try:
            # Truncate if necessary
            text = self._truncate_text(text, self.max_tokens)
            
            response = self.client.embeddings.create(
                input=text,
                model=self.deployment_name
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def _truncate_text(self, text, max_tokens):
        """Truncate text to fit within token limit."""
        # Simple approximation: ~4 chars per token
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars]
        return text
    
    def get_embedding_stats(self, embedding):
        """Get statistics about an embedding."""
        import numpy as np
        
        if embedding is None:
            return None
        
        arr = np.array(embedding)
        return {
            'dimensions': len(embedding),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'l2_norm': float(np.linalg.norm(arr))
        }

# Usage
embedder = AzureOpenAIEmbedder(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

text = "Azure AI Search provides powerful search capabilities"
embedding = embedder.embed_text(text)
stats = embedder.get_embedding_stats(embedding)

print(f"Embedding dimensions: {stats['dimensions']}")
print(f"L2 norm: {stats['l2_norm']:.4f}")
```

### Document Embedding with Chunking

```python
class DocumentEmbedder:
    """Embed long documents with chunking strategy."""
    
    def __init__(self, embedder, chunk_size=1000, overlap=200):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text):
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.7:  # At least 70% of chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append({
                'text': chunk.strip(),
                'start': start,
                'end': end
            })
            
            start = end - self.overlap
        
        return chunks
    
    def embed_document(self, text, aggregate_method='mean'):
        """
        Embed a long document.
        
        Args:
            text: Document text
            aggregate_method: 'mean', 'max', or 'first'
            
        Returns:
            Aggregated embedding vector
        """
        import numpy as np
        
        # Split into chunks
        chunks = self.chunk_text(text)
        
        # Generate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            embedding = self.embedder.embed_text(chunk['text'])
            if embedding:
                embeddings.append(embedding)
        
        if not embeddings:
            return None
        
        # Aggregate embeddings
        embeddings_array = np.array(embeddings)
        
        if aggregate_method == 'mean':
            final_embedding = np.mean(embeddings_array, axis=0)
        elif aggregate_method == 'max':
            final_embedding = np.max(embeddings_array, axis=0)
        elif aggregate_method == 'first':
            final_embedding = embeddings_array[0]
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate_method}")
        
        return final_embedding.tolist()
    
    def embed_with_metadata(self, document):
        """
        Embed document with title and content separately.
        
        Args:
            document: Dict with 'title' and 'content'
            
        Returns:
            Dict with separate embeddings
        """
        result = {
            'title_embedding': None,
            'content_embedding': None,
            'combined_embedding': None
        }
        
        # Embed title
        if document.get('title'):
            result['title_embedding'] = self.embedder.embed_text(document['title'])
        
        # Embed content
        if document.get('content'):
            result['content_embedding'] = self.embed_document(document['content'])
        
        # Create combined embedding (weighted average)
        if result['title_embedding'] and result['content_embedding']:
            import numpy as np
            title_weight = 0.3
            content_weight = 0.7
            
            combined = (
                np.array(result['title_embedding']) * title_weight +
                np.array(result['content_embedding']) * content_weight
            )
            result['combined_embedding'] = combined.tolist()
        
        return result

# Usage
doc_embedder = DocumentEmbedder(embedder, chunk_size=1000, overlap=200)

long_document = """
[Your long document text here...]
This could be several pages of content that needs to be chunked.
"""

embedding = doc_embedder.embed_document(long_document, aggregate_method='mean')
print(f"Document embedded with {len(embedding)} dimensions")
```

---

## Batch Processing

### Efficient Batch Embedding

```python
import asyncio
from typing import List
import time

class BatchEmbedder:
    """Efficient batch embedding processor."""
    
    def __init__(self, embedder, batch_size=100, rate_limit_rpm=3000):
        self.embedder = embedder
        self.batch_size = batch_size
        self.rate_limit_rpm = rate_limit_rpm
        self.requests_per_second = rate_limit_rpm / 60
        self.min_request_interval = 1.0 / self.requests_per_second
    
    def embed_batch(self, texts):
        """
        Embed a batch of texts with rate limiting.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embeddings
        """
        embeddings = []
        last_request_time = 0
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            
            try:
                # Azure OpenAI supports batch embedding
                response = self.embedder.client.embeddings.create(
                    input=batch,
                    model=self.embedder.deployment_name
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                last_request_time = time.time()
                
                print(f"Processed {i + len(batch)}/{len(texts)} texts")
                
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Add None for failed embeddings
                embeddings.extend([None] * len(batch))
        
        return embeddings
    
    async def embed_batch_async(self, texts):
        """Asynchronous batch embedding."""
        tasks = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            task = asyncio.create_task(self._embed_batch_async(batch))
            tasks.append(task)
            
            # Add delay between batches for rate limiting
            await asyncio.sleep(self.min_request_interval)
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_embeddings = []
        for batch_result in results:
            all_embeddings.extend(batch_result)
        
        return all_embeddings
    
    async def _embed_batch_async(self, batch):
        """Embed a single batch asynchronously."""
        try:
            response = self.embedder.client.embeddings.create(
                input=batch,
                model=self.embedder.deployment_name
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error in async batch: {e}")
            return [None] * len(batch)

# Usage
batch_embedder = BatchEmbedder(embedder, batch_size=100)

# Process large dataset
texts = [
    "Document 1 content...",
    "Document 2 content...",
    # ... thousands of documents
]

# Synchronous processing
embeddings = batch_embedder.embed_batch(texts)

# Async processing (faster)
# embeddings = await batch_embedder.embed_batch_async(texts)
```

### Progress Tracking and Resume

```python
import json
import os

class ResumableEmbedder:
    """Batch embedder with progress tracking and resume capability."""
    
    def __init__(self, embedder, checkpoint_file="embedding_checkpoint.json"):
        self.embedder = embedder
        self.checkpoint_file = checkpoint_file
        self.checkpoint_interval = 100  # Save every 100 documents
    
    def embed_documents_with_resume(self, documents, id_field='id'):
        """
        Embed documents with checkpoint/resume support.
        
        Args:
            documents: List of document dicts
            id_field: Field name containing document ID
            
        Returns:
            Dict mapping document IDs to embeddings
        """
        # Load existing checkpoint if available
        embeddings_map = self._load_checkpoint()
        
        total = len(documents)
        processed = len(embeddings_map)
        
        print(f"Resuming from {processed}/{total} documents")
        
        for i, doc in enumerate(documents):
            doc_id = doc[id_field]
            
            # Skip if already processed
            if doc_id in embeddings_map:
                continue
            
            # Generate embedding
            text = self._extract_text(doc)
            embedding = self.embedder.embed_text(text)
            
            if embedding:
                embeddings_map[doc_id] = {
                    'embedding': embedding,
                    'text_length': len(text),
                    'timestamp': time.time()
                }
                
                processed += 1
                
                # Save checkpoint periodically
                if processed % self.checkpoint_interval == 0:
                    self._save_checkpoint(embeddings_map)
                    print(f"Checkpoint saved: {processed}/{total}")
        
        # Final save
        self._save_checkpoint(embeddings_map)
        print(f"✅ Complete: {processed}/{total} documents embedded")
        
        return embeddings_map
    
    def _extract_text(self, doc):
        """Extract text from document for embedding."""
        parts = []
        
        if doc.get('title'):
            parts.append(doc['title'])
        if doc.get('content'):
            parts.append(doc['content'])
        if doc.get('description'):
            parts.append(doc['description'])
        
        return " ".join(parts)
    
    def _load_checkpoint(self):
        """Load checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_checkpoint(self, embeddings_map):
        """Save checkpoint file."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(embeddings_map, f)

# Usage
resumable = ResumableEmbedder(embedder)

documents = [
    {'id': '1', 'title': 'Doc 1', 'content': '...'},
    {'id': '2', 'title': 'Doc 2', 'content': '...'},
    # ... many documents
]

embeddings = resumable.embed_documents_with_resume(documents)
```

---

## Cost Optimization

### Token Usage Tracking

```python
class CostAwareEmbedder:
    """Embedding generator with cost tracking."""
    
    def __init__(self, embedder, cost_per_1k_tokens=0.0001):
        self.embedder = embedder
        self.cost_per_1k_tokens = cost_per_1k_tokens  # ada-002 pricing
        self.total_tokens = 0
        self.total_cost = 0
    
    def estimate_tokens(self, text):
        """Estimate token count (rough approximation)."""
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def embed_text_with_cost(self, text):
        """Embed text and track cost."""
        embedding = self.embedder.embed_text(text)
        
        if embedding:
            tokens = self.estimate_tokens(text)
            cost = (tokens / 1000) * self.cost_per_1k_tokens
            
            self.total_tokens += tokens
            self.total_cost += cost
        
        return embedding
    
    def get_cost_summary(self):
        """Get cost summary."""
        return {
            'total_tokens': self.total_tokens,
            'total_cost_usd': self.total_cost,
            'average_tokens_per_request': self.total_tokens / max(1, self.request_count),
            'cost_per_embedding': self.total_cost / max(1, self.request_count)
        }
    
    def reset_tracking(self):
        """Reset cost tracking."""
        self.total_tokens = 0
        self.total_cost = 0
        self.request_count = 0

# Usage
cost_embedder = CostAwareEmbedder(embedder)

# Process documents
for doc in documents:
    embedding = cost_embedder.embed_text_with_cost(doc['content'])

# Get cost summary
summary = cost_embedder.get_cost_summary()
print(f"Total cost: ${summary['total_cost_usd']:.4f}")
print(f"Total tokens: {summary['total_tokens']:,}")
```

### Caching Strategy

```python
import hashlib
import pickle

class CachedEmbedder:
    """Embedder with local caching to reduce API calls."""
    
    def __init__(self, embedder, cache_dir="embedding_cache"):
        self.embedder = embedder
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, text):
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        """Get file path for cache entry."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def embed_text(self, text):
        """Embed text with caching."""
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                embedding = pickle.load(f)
            self.cache_hits += 1
            return embedding
        
        # Generate embedding
        embedding = self.embedder.embed_text(text)
        
        if embedding:
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            self.cache_misses += 1
        
        return embedding
    
    def get_cache_stats(self):
        """Get cache performance stats."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cost_saved': self.cache_hits * 0.0001  # Approximate savings
        }

# Usage
cached_embedder = CachedEmbedder(embedder)

# First run - cache misses
embedding1 = cached_embedder.embed_text("Sample text")

# Second run - cache hit (no API call, no cost!)
embedding2 = cached_embedder.embed_text("Sample text")

stats = cached_embedder.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
print(f"Cost saved: ${stats['cost_saved']:.4f}")
```

---

## Performance Tuning

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ParallelEmbedder:
    """Multi-threaded embedding processor."""
    
    def __init__(self, embedder, max_workers=5):
        self.embedder = embedder
        self.max_workers = max_workers
        self.lock = threading.Lock()
    
    def embed_batch_parallel(self, texts):
        """Process embeddings in parallel."""
        embeddings = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.embedder.embed_text, text): i
                for i, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    embedding = future.result()
                    embeddings[index] = embedding
                except Exception as e:
                    print(f"Error processing text {index}: {e}")
        
        return embeddings

# Usage
parallel_embedder = ParallelEmbedder(embedder, max_workers=5)
embeddings = parallel_embedder.embed_batch_parallel(texts)
```

---

## Error Handling

### Robust Error Handling

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustEmbedder:
    """Embedder with comprehensive error handling."""
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def embed_text_with_retry(self, text):
        """Embed text with automatic retry on failure."""
        try:
            return self.embedder.embed_text(text)
        except Exception as e:
            print(f"Embedding attempt failed: {e}")
            raise
    
    def embed_text_safe(self, text, fallback=None):
        """Embed text with fallback on error."""
        try:
            return self.embed_text_with_retry(text)
        except Exception as e:
            print(f"All retry attempts failed: {e}")
            return fallback

# Usage
robust_embedder = RobustEmbedder(embedder)
embedding = robust_embedder.embed_text_safe("Sample text", fallback=[0.0] * 1536)
```

---

## Best Practices

### ✅ Do's
1. **Cache embeddings** for frequently accessed content
2. **Batch process** when possible to reduce latency
3. **Monitor costs** with token tracking
4. **Use checkpoints** for long-running jobs
5. **Implement retry logic** for transient failures
6. **Normalize embeddings** for consistent similarity calculations

### ❌ Don'ts
1. **Don't** embed the same text multiple times
2. **Don't** exceed rate limits (3,000 RPM for ada-002)
3. **Don't** ignore token limits (8,191 for ada-002)
4. **Don't** skip error handling
5. **Don't** forget to secure API keys

---

## Next Steps

- **[Vector Search Implementation](./09-vector-search.md)** - Use embeddings for search
- **[Hybrid Search](./10-hybrid-search.md)** - Combine text and vector search
- **[Cost Analysis](./19-cost-analysis.md)** - Optimize Azure OpenAI costs

---

*See also: [Azure AI Search Setup](./04-azure-ai-search-setup.md) | [Best Practices](./30-best-practices-checklist.md)*