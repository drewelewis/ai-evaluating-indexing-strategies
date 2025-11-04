# Advanced Techniques for Search Optimization

This document covers sophisticated methods to enhance search relevance beyond basic indexing strategies, including semantic re-ranking, query enhancement, and feedback mechanisms.

## 🧠 Semantic Re-ranking with Cross-Encoders

### Overview
Cross-encoder models provide more accurate relevance scoring by jointly encoding query and document pairs, enabling nuanced understanding of semantic relationships.

### Architecture
```python
class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        self.max_candidates = 100  # Limit for computational efficiency
    
    def rerank(self, query, candidates, top_k=10):
        if len(candidates) > self.max_candidates:
            candidates = candidates[:self.max_candidates]
        
        # Create query-document pairs
        pairs = [(query, doc.content) for doc in candidates]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Rerank and return top results
        ranked_results = sorted(
            zip(candidates, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return ranked_results[:top_k]
```

### Multi-Stage Ranking Pipeline
```python
class MultiStageRanker:
    def __init__(self, first_stage, reranker, final_ranker=None):
        self.first_stage = first_stage      # Fast retrieval (BM25/Vector)
        self.reranker = reranker           # Cross-encoder reranking
        self.final_ranker = final_ranker   # Optional: LLM-based final ranking
    
    def search(self, query, final_top_k=10):
        # Stage 1: Fast retrieval (1000+ candidates)
        candidates = self.first_stage.search(query, top_k=1000)
        
        # Stage 2: Cross-encoder reranking (100 candidates)
        reranked = self.reranker.rerank(query, candidates, top_k=100)
        
        # Stage 3: Optional LLM final ranking (10-20 candidates)
        if self.final_ranker:
            final_results = self.final_ranker.rank(query, reranked, top_k=final_top_k)
        else:
            final_results = reranked[:final_top_k]
        
        return final_results
```

### Model Selection Guide

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| **ms-marco-MiniLM-L-6-v2** | ⚡⚡⚡ | ⭐⭐ | High-throughput, general queries |
| **ms-marco-electra-base** | ⚡⚡ | ⭐⭐⭐ | Balanced performance |
| **monoT5-large** | ⚡ | ⭐⭐⭐⭐ | Best quality, research use |
| **bge-reranker-large** | ⚡⚡ | ⭐⭐⭐ | Multilingual support |

### Performance Optimization
```python
class OptimizedReranker:
    def __init__(self, model_name, batch_size=32, cache_size=1000):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size
        self.cache = LRUCache(cache_size)
        self.model.half()  # Use FP16 for faster inference
    
    def rerank_batch(self, query_doc_pairs):
        # Check cache first
        uncached_pairs = []
        cached_scores = {}
        
        for i, (query, doc) in enumerate(query_doc_pairs):
            cache_key = hash(f"{query}:{doc}")
            if cache_key in self.cache:
                cached_scores[i] = self.cache[cache_key]
            else:
                uncached_pairs.append((i, query, doc))
        
        # Process uncached pairs in batches
        if uncached_pairs:
            pairs_only = [(q, d) for _, q, d in uncached_pairs]
            batch_scores = self.model.predict(pairs_only, batch_size=self.batch_size)
            
            for (orig_idx, query, doc), score in zip(uncached_pairs, batch_scores):
                cache_key = hash(f"{query}:{doc}")
                self.cache[cache_key] = score
                cached_scores[orig_idx] = score
        
        # Reconstruct full scores array
        return [cached_scores[i] for i in range(len(query_doc_pairs))]
```

## 🔄 Query Enhancement and Rewriting

### Query Expansion with LLMs
```python
class LLMQueryExpander:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.expansion_prompt = """
        Given the search query: "{query}"
        
        Generate 3-5 related search terms or phrases that would help find relevant documents:
        1. Synonyms and alternative phrasings
        2. Related concepts and topics
        3. More specific or general variants
        
        Return only the expanded terms, one per line:
        """
    
    def expand_query(self, original_query):
        prompt = self.expansion_prompt.format(query=original_query)
        response = self.llm.generate(prompt, max_tokens=100)
        
        expanded_terms = [term.strip() for term in response.split('\n') if term.strip()]
        
        # Combine original with expanded terms
        return {
            "original": original_query,
            "expanded": expanded_terms,
            "combined": f"{original_query} {' '.join(expanded_terms)}"
        }
```

### Query Classification and Routing
```python
class QueryClassifier:
    def __init__(self):
        self.patterns = {
            "navigational": [
                r"homepage|home page|main page",
                r"login|sign in|log in",
                r"contact us|contact info"
            ],
            "transactional": [
                r"buy|purchase|order|shop",
                r"price|cost|cheap|expensive",
                r"download|install"
            ],
            "informational": [
                r"how to|what is|why does",
                r"tutorial|guide|help",
                r"definition|meaning"
            ],
            "exact_match": [
                r"^\w+\d+$",  # Product codes like ABC123
                r"^[\w\-]+\.(com|org|net)$",  # Domain names
                r'"[^"]+"'  # Quoted exact phrases
            ]
        }
    
    def classify(self, query):
        query_lower = query.lower()
        scores = {}
        
        for intent, patterns in self.patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            scores[intent] = score
        
        # Return intent with highest score, or "general" if no clear match
        max_intent = max(scores, key=scores.get)
        return max_intent if scores[max_intent] > 0 else "general"
```

### Spell Correction and Normalization
```python
class QueryNormalizer:
    def __init__(self):
        self.spell_checker = SpellChecker()
        self.synonyms = self.load_domain_synonyms()
    
    def normalize(self, query):
        # Step 1: Spell correction
        words = query.split()
        corrected_words = []
        
        for word in words:
            if word.lower() not in self.spell_checker:
                candidates = self.spell_checker.candidates(word)
                if candidates:
                    corrected_word = min(candidates, key=lambda x: editdistance.eval(word, x))
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        corrected_query = " ".join(corrected_words)
        
        # Step 2: Synonym expansion
        expanded_query = self.expand_synonyms(corrected_query)
        
        # Step 3: Normalization
        normalized_query = self.normalize_text(expanded_query)
        
        return {
            "original": query,
            "corrected": corrected_query,
            "expanded": expanded_query,
            "normalized": normalized_query
        }
    
    def expand_synonyms(self, query):
        words = query.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            if word.lower() in self.synonyms:
                expanded_words.extend(self.synonyms[word.lower()])
        
        return " ".join(expanded_words)
```

### Intent-Based Query Enhancement
```python
class IntentBasedQueryEnhancer:
    def __init__(self, intent_classifier, query_expanders):
        self.classifier = intent_classifier
        self.expanders = query_expanders
    
    def enhance_query(self, query):
        intent = self.classifier.classify(query)
        
        enhancement_strategies = {
            "navigational": self.enhance_navigational,
            "transactional": self.enhance_transactional,
            "informational": self.enhance_informational,
            "exact_match": self.enhance_exact_match
        }
        
        enhancer = enhancement_strategies.get(intent, self.enhance_general)
        return enhancer(query, intent)
    
    def enhance_navigational(self, query, intent):
        # For navigational queries, prioritize exact matches and boost title fields
        return {
            "enhanced_query": query,
            "search_config": {
                "boost_fields": {"title": 3.0, "url": 2.0},
                "exact_match_priority": True
            }
        }
    
    def enhance_transactional(self, query, intent):
        # For transactional queries, add commercial terms and boost product fields
        commercial_terms = ["buy", "price", "shop", "purchase", "order"]
        enhanced = f"{query} {' '.join(commercial_terms[:2])}"
        
        return {
            "enhanced_query": enhanced,
            "search_config": {
                "boost_fields": {"product_name": 2.0, "price": 1.5},
                "filter_available": True
            }
        }
```

## 🔄 Feedback Loops and Learning

### Click-Through Rate (CTR) Learning
```python
class CTRLearningModel:
    def __init__(self):
        self.click_model = LogisticRegression()
        self.features = FeatureExtractor()
        self.is_trained = False
    
    def extract_features(self, query, document, position):
        return {
            "query_doc_similarity": self.compute_similarity(query, document),
            "document_quality_score": document.quality_score,
            "position": position,
            "query_length": len(query.split()),
            "document_length": len(document.content.split()),
            "exact_match_count": self.count_exact_matches(query, document),
            "bm25_score": self.compute_bm25(query, document)
        }
    
    def train(self, interaction_data):
        features = []
        labels = []
        
        for interaction in interaction_data:
            for position, doc in enumerate(interaction.results):
                feature_vector = self.extract_features(
                    interaction.query, doc, position
                )
                features.append(list(feature_vector.values()))
                labels.append(1 if doc.id in interaction.clicked_docs else 0)
        
        self.click_model.fit(features, labels)
        self.is_trained = True
    
    def predict_ctr(self, query, documents):
        if not self.is_trained:
            return [0.5] * len(documents)  # Default probability
        
        features = []
        for position, doc in enumerate(documents):
            feature_vector = self.extract_features(query, doc, position)
            features.append(list(feature_vector.values()))
        
        return self.click_model.predict_proba(features)[:, 1]
```

### Reinforcement Learning for Ranking
```python
class RankingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(10000)
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def build_model(self):
        model = Sequential([
            Dense(256, activation='relu', input_shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def get_state(self, query, candidates):
        # Extract features for current ranking state
        features = []
        for doc in candidates:
            doc_features = self.extract_document_features(query, doc)
            features.extend(doc_features)
        return np.array(features)
    
    def choose_action(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        q_values = self.q_network.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])
    
    def learn(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > 32:
            batch = self.memory.sample(32)
            self.replay(batch)
```

### Real-Time Learning Pipeline
```python
class RealTimeLearningPipeline:
    def __init__(self, search_engine, feedback_processor):
        self.search_engine = search_engine
        self.feedback_processor = feedback_processor
        self.update_frequency = 3600  # Update every hour
        self.last_update = time.time()
    
    def process_search_interaction(self, query, results, user_feedback):
        # Log interaction for learning
        interaction = {
            "timestamp": time.time(),
            "query": query,
            "results": results,
            "clicks": user_feedback.get("clicks", []),
            "dwell_times": user_feedback.get("dwell_times", {}),
            "explicit_feedback": user_feedback.get("ratings", {})
        }
        
        self.feedback_processor.add_interaction(interaction)
        
        # Trigger model update if enough time has passed
        if time.time() - self.last_update > self.update_frequency:
            self.update_ranking_model()
    
    def update_ranking_model(self):
        # Collect recent feedback
        recent_interactions = self.feedback_processor.get_recent_interactions()
        
        if len(recent_interactions) < 100:  # Minimum data threshold
            return
        
        # Retrain or fine-tune ranking model
        self.search_engine.ranking_model.update(recent_interactions)
        
        # Validate performance on held-out set
        validation_score = self.validate_model_performance()
        
        if validation_score > self.current_baseline:
            self.search_engine.deploy_updated_model()
            self.current_baseline = validation_score
        
        self.last_update = time.time()
```

## 🎯 Advanced Retrieval Patterns

### Dense Passage Retrieval (DPR)
```python
class DensePassageRetriever:
    def __init__(self, question_encoder, passage_encoder, index):
        self.question_encoder = question_encoder
        self.passage_encoder = passage_encoder
        self.index = index
    
    def retrieve(self, question, top_k=100):
        # Encode question
        question_embedding = self.question_encoder.encode([question])
        
        # Search for similar passages
        scores, passage_ids = self.index.search(question_embedding, top_k)
        
        # Retrieve passage content
        passages = self.get_passages(passage_ids)
        
        return list(zip(passages, scores))
    
    def train_biencoder(self, training_data):
        # Training loop for question and passage encoders
        for batch in training_data:
            questions = batch["questions"]
            positive_passages = batch["positive_passages"]
            negative_passages = batch["negative_passages"]
            
            # Compute embeddings
            q_embeds = self.question_encoder.encode(questions)
            pos_embeds = self.passage_encoder.encode(positive_passages)
            neg_embeds = self.passage_encoder.encode(negative_passages)
            
            # Contrastive loss
            loss = self.compute_contrastive_loss(q_embeds, pos_embeds, neg_embeds)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
```

### ColBERT-style Late Interaction
```python
class ColBERTRetriever:
    def __init__(self, model_checkpoint):
        self.colbert = ColBERT.from_pretrained(model_checkpoint)
        self.index = None
    
    def index_collection(self, passages):
        # Generate token-level embeddings for all passages
        passage_embeddings = []
        
        for passage in passages:
            # Each passage becomes a matrix of token embeddings
            token_embeddings = self.colbert.encode_passage(passage)
            passage_embeddings.append(token_embeddings)
        
        # Build efficient index for MaxSim operations
        self.index = self.build_maxsim_index(passage_embeddings)
    
    def search(self, query, top_k=10):
        # Encode query to token embeddings
        query_embeddings = self.colbert.encode_query(query)
        
        # Compute MaxSim scores with all passages
        scores = self.compute_maxsim_scores(query_embeddings, self.index)
        
        # Return top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(idx, scores[idx]) for idx in top_indices]
    
    def compute_maxsim_scores(self, query_embeddings, passage_index):
        scores = []
        
        for passage_embeddings in passage_index:
            # For each query token, find max similarity with passage tokens
            token_scores = []
            for q_token in query_embeddings:
                max_sim = max(cosine_similarity(q_token, p_token) 
                            for p_token in passage_embeddings)
                token_scores.append(max_sim)
            
            # Sum over query tokens
            passage_score = sum(token_scores)
            scores.append(passage_score)
        
        return scores
```

### Multi-Vector Retrieval
```python
class MultiVectorRetriever:
    def __init__(self, chunk_encoder, summary_encoder):
        self.chunk_encoder = chunk_encoder
        self.summary_encoder = summary_encoder
        self.chunk_index = None
        self.summary_index = None
    
    def index_documents(self, documents):
        chunk_embeddings = []
        summary_embeddings = []
        
        for doc in documents:
            # Create multiple representations per document
            chunks = self.split_into_chunks(doc.content)
            summary = self.generate_summary(doc.content)
            
            # Encode each representation
            for chunk in chunks:
                chunk_emb = self.chunk_encoder.encode([chunk])
                chunk_embeddings.append((doc.id, chunk_emb))
            
            summary_emb = self.summary_encoder.encode([summary])
            summary_embeddings.append((doc.id, summary_emb))
        
        self.chunk_index = self.build_index(chunk_embeddings)
        self.summary_index = self.build_index(summary_embeddings)
    
    def retrieve(self, query, strategy="hybrid"):
        query_embedding = self.chunk_encoder.encode([query])
        
        if strategy == "chunk_only":
            return self.search_chunks(query_embedding)
        elif strategy == "summary_only":
            return self.search_summaries(query_embedding)
        else:  # hybrid
            chunk_results = self.search_chunks(query_embedding, top_k=50)
            summary_results = self.search_summaries(query_embedding, top_k=50)
            return self.merge_results(chunk_results, summary_results)
```

## 📊 Performance Monitoring and Optimization

### Real-Time Quality Monitoring
```python
class QualityMonitor:
    def __init__(self, baseline_metrics):
        self.baseline = baseline_metrics
        self.current_window = collections.deque(maxlen=1000)
        self.alert_thresholds = {
            "ctr_drop": 0.05,
            "latency_increase": 100,  # ms
            "error_rate": 0.01
        }
    
    def track_query(self, query_metadata):
        self.current_window.append(query_metadata)
        
        # Check for quality degradation every 100 queries
        if len(self.current_window) % 100 == 0:
            self.check_quality_alerts()
    
    def check_quality_alerts(self):
        current_metrics = self.compute_window_metrics()
        
        alerts = []
        
        # CTR degradation
        if current_metrics["ctr"] < self.baseline["ctr"] - self.alert_thresholds["ctr_drop"]:
            alerts.append(f"CTR dropped to {current_metrics['ctr']:.3f}")
        
        # Latency increase
        if current_metrics["p95_latency"] > self.baseline["p95_latency"] + self.alert_thresholds["latency_increase"]:
            alerts.append(f"P95 latency increased to {current_metrics['p95_latency']:.1f}ms")
        
        # Error rate increase
        if current_metrics["error_rate"] > self.alert_thresholds["error_rate"]:
            alerts.append(f"Error rate increased to {current_metrics['error_rate']:.3f}")
        
        if alerts:
            self.send_alerts(alerts)
```

---
*Next: [Implementation Guide](../guides/implementation-guide.md)*