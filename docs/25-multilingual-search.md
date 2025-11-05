# Multilingual Search

Comprehensive guide to implementing international search capabilities with language detection, multi-language analyzers, translation integration, and cross-language search strategies.

---

## 1. Language Detection

### Automatic Language Identification

```python
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchableField,
    SimpleField,
    SearchFieldDataType
)
from typing import List, Dict, Optional
from dataclasses import dataclass

class LanguageDetector:
    """Detect document language using Azure AI Language"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str
    ):
        credential = AzureKeyCredential(api_key)
        self.client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=credential
        )
    
    def detect_language(
        self,
        text: str,
        country_hint: str = "US"
    ) -> Dict:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            country_hint: ISO 3166-1 alpha-2 country code
            
        Returns:
            Language detection result with confidence
        """
        documents = [{"id": "1", "text": text, "countryHint": country_hint}]
        
        response = self.client.detect_language(
            documents=documents,
            show_stats=False
        )[0]
        
        return {
            'language': response.primary_language.iso6391_name,
            'language_name': response.primary_language.name,
            'confidence': response.primary_language.confidence_score,
            'is_reliable': response.primary_language.confidence_score > 0.8
        }
    
    def detect_batch_languages(
        self,
        texts: List[str]
    ) -> List[Dict]:
        """
        Detect languages for multiple documents
        
        Args:
            texts: List of text documents
            
        Returns:
            List of language detection results
        """
        documents = [
            {"id": str(i), "text": text}
            for i, text in enumerate(texts)
        ]
        
        response = self.client.detect_language(documents=documents)
        
        results = []
        for doc in response:
            if not doc.is_error:
                results.append({
                    'language': doc.primary_language.iso6391_name,
                    'language_name': doc.primary_language.name,
                    'confidence': doc.primary_language.confidence_score
                })
            else:
                results.append({
                    'language': 'unknown',
                    'error': doc.error.message
                })
        
        return results


class MultilingualDocumentProcessor:
    """Process documents with automatic language detection"""
    
    def __init__(
        self,
        language_detector: LanguageDetector
    ):
        self.language_detector = language_detector
    
    def process_document(
        self,
        document: Dict
    ) -> Dict:
        """
        Process document with language detection
        
        Args:
            document: Document with 'content' field
            
        Returns:
            Document enriched with language metadata
        """
        content = document.get('content', '')
        
        # Detect language
        lang_result = self.language_detector.detect_language(content)
        
        # Add language fields
        document['language'] = lang_result['language']
        document['language_name'] = lang_result['language_name']
        document['language_confidence'] = lang_result['confidence']
        
        # Select appropriate analyzer
        document['analyzer'] = self._get_analyzer_for_language(
            lang_result['language']
        )
        
        return document
    
    def _get_analyzer_for_language(self, language_code: str) -> str:
        """
        Get appropriate analyzer for language
        
        Args:
            language_code: ISO 639-1 language code
            
        Returns:
            Analyzer name
        """
        # Language-specific Microsoft analyzers
        language_analyzers = {
            'ar': 'ar.microsoft',      # Arabic
            'bn': 'bn.microsoft',      # Bengali
            'bg': 'bg.microsoft',      # Bulgarian
            'ca': 'ca.microsoft',      # Catalan
            'zh-Hans': 'zh-Hans.microsoft',  # Chinese Simplified
            'zh-Hant': 'zh-Hant.microsoft',  # Chinese Traditional
            'hr': 'hr.microsoft',      # Croatian
            'cs': 'cs.microsoft',      # Czech
            'da': 'da.microsoft',      # Danish
            'nl': 'nl.microsoft',      # Dutch
            'en': 'en.microsoft',      # English
            'et': 'et.microsoft',      # Estonian
            'fi': 'fi.microsoft',      # Finnish
            'fr': 'fr.microsoft',      # French
            'de': 'de.microsoft',      # German
            'el': 'el.microsoft',      # Greek
            'gu': 'gu.microsoft',      # Gujarati
            'he': 'he.microsoft',      # Hebrew
            'hi': 'hi.microsoft',      # Hindi
            'hu': 'hu.microsoft',      # Hungarian
            'is': 'is.microsoft',      # Icelandic
            'id': 'id.microsoft',      # Indonesian
            'ga': 'ga.microsoft',      # Irish
            'it': 'it.microsoft',      # Italian
            'ja': 'ja.microsoft',      # Japanese
            'kn': 'kn.microsoft',      # Kannada
            'ko': 'ko.microsoft',      # Korean
            'lv': 'lv.microsoft',      # Latvian
            'lt': 'lt.microsoft',      # Lithuanian
            'ms': 'ms.microsoft',      # Malay
            'ml': 'ml.microsoft',      # Malayalam
            'mr': 'mr.microsoft',      # Marathi
            'nb': 'nb.microsoft',      # Norwegian (Bokmål)
            'fa': 'fa.lucene',         # Persian (Lucene)
            'pl': 'pl.microsoft',      # Polish
            'pt-BR': 'pt-Br.microsoft', # Portuguese (Brazil)
            'pt-PT': 'pt-Pt.microsoft', # Portuguese (Portugal)
            'pa': 'pa.microsoft',      # Punjabi
            'ro': 'ro.microsoft',      # Romanian
            'ru': 'ru.microsoft',      # Russian
            'sr': 'sr-cyrillic.microsoft',  # Serbian (Cyrillic)
            'sk': 'sk.microsoft',      # Slovak
            'sl': 'sl.microsoft',      # Slovenian
            'es': 'es.microsoft',      # Spanish
            'sv': 'sv.microsoft',      # Swedish
            'ta': 'ta.microsoft',      # Tamil
            'te': 'te.microsoft',      # Telugu
            'th': 'th.microsoft',      # Thai
            'tr': 'tr.microsoft',      # Turkish
            'uk': 'uk.microsoft',      # Ukrainian
            'ur': 'ur.microsoft',      # Urdu
            'vi': 'vi.microsoft'       # Vietnamese
        }
        
        return language_analyzers.get(language_code, 'standard.lucene')
```

---

## 2. Multilingual Index Schema

### Index Design for Multiple Languages

```python
class MultilingualIndexBuilder:
    """Build search indexes supporting multiple languages"""
    
    @staticmethod
    def create_multilingual_index(
        index_name: str,
        supported_languages: Optional[List[str]] = None
    ) -> SearchIndex:
        """
        Create index with multilingual support
        
        Args:
            index_name: Index name
            supported_languages: List of ISO 639-1 language codes
            
        Returns:
            Multilingual search index
        """
        if supported_languages is None:
            # Default: English, Spanish, French, German, Chinese
            supported_languages = ['en', 'es', 'fr', 'de', 'zh-Hans']
        
        fields = [
            # Document identifier
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True
            ),
            
            # Language metadata
            SimpleField(
                name="language",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SimpleField(
                name="language_confidence",
                type=SearchFieldDataType.Double,
                filterable=True
            ),
            
            # Original content (language-neutral)
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                analyzer_name="standard.lucene"
            ),
            
            # Title fields per language
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                analyzer_name="standard.lucene"
            )
        ]
        
        # Add language-specific fields
        for lang in supported_languages:
            analyzer = MultilingualDocumentProcessor(None)._get_analyzer_for_language(lang)
            
            # Language-specific content field
            fields.append(
                SearchableField(
                    name=f"content_{lang}",
                    type=SearchFieldDataType.String,
                    analyzer_name=analyzer,
                    searchable=True
                )
            )
            
            # Language-specific title field
            fields.append(
                SearchableField(
                    name=f"title_{lang}",
                    type=SearchFieldDataType.String,
                    analyzer_name=analyzer,
                    searchable=True
                )
            )
        
        # Common metadata fields
        fields.extend([
            SimpleField(
                name="created_date",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True
            ),
            SimpleField(
                name="author",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SearchField(
                name="tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True,
                filterable=True,
                facetable=True
            ),
            SearchField(
                name="available_languages",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True
            )
        ])
        
        return SearchIndex(
            name=index_name,
            fields=fields
        )
    
    @staticmethod
    def create_single_field_multilingual_index(
        index_name: str
    ) -> SearchIndex:
        """
        Create index with single content field using language analyzer
        
        This approach uses the detected language analyzer on a single field
        rather than separate fields per language.
        
        Args:
            index_name: Index name
            
        Returns:
            Search index
        """
        return SearchIndex(
            name=index_name,
            fields=[
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True
                ),
                SimpleField(
                    name="language",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SearchableField(
                    name="title",
                    type=SearchFieldDataType.String,
                    # Analyzer will be set per-document using searchFields parameter
                    analyzer_name="standard.lucene"
                ),
                SearchableField(
                    name="content",
                    type=SearchFieldDataType.String,
                    analyzer_name="standard.lucene"
                ),
                SearchField(
                    name="available_languages",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True,
                    facetable=True
                )
            ]
        )
```

---

## 3. Translation Integration

### Azure Translator Integration

```python
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential as TranslatorCredential

class TranslationManager:
    """Manage document and query translation"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        region: str
    ):
        self.client = TextTranslationClient(
            endpoint=endpoint,
            credential=TranslatorCredential(api_key),
            region=region
        )
    
    def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> Dict:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language (auto-detect if None)
            
        Returns:
            Translation result
        """
        try:
            response = self.client.translate(
                body=[{"text": text}],
                to_language=[target_language],
                from_language=source_language
            )
            
            translation = response[0].translations[0]
            
            return {
                'translated_text': translation.text,
                'target_language': translation.to,
                'source_language': response[0].detected_language.language if source_language is None else source_language,
                'confidence': response[0].detected_language.score if source_language is None else 1.0
            }
        except Exception as e:
            return {
                'error': str(e),
                'translated_text': text  # Fallback to original
            }
    
    def translate_document(
        self,
        document: Dict,
        target_language: str,
        fields_to_translate: List[str]
    ) -> Dict:
        """
        Translate specific fields in a document
        
        Args:
            document: Source document
            target_language: Target language code
            fields_to_translate: List of field names to translate
            
        Returns:
            Document with translated fields
        """
        translated_doc = document.copy()
        source_language = document.get('language', None)
        
        for field in fields_to_translate:
            if field in document:
                result = self.translate_text(
                    text=document[field],
                    target_language=target_language,
                    source_language=source_language
                )
                
                # Add translated field
                translated_field_name = f"{field}_{target_language}"
                translated_doc[translated_field_name] = result['translated_text']
        
        # Track available languages
        if 'available_languages' not in translated_doc:
            translated_doc['available_languages'] = []
        
        if target_language not in translated_doc['available_languages']:
            translated_doc['available_languages'].append(target_language)
        
        return translated_doc
    
    def batch_translate(
        self,
        texts: List[str],
        target_language: str,
        source_language: Optional[str] = None
    ) -> List[Dict]:
        """
        Translate multiple texts
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language (auto-detect if None)
            
        Returns:
            List of translation results
        """
        body = [{"text": text} for text in texts]
        
        response = self.client.translate(
            body=body,
            to_language=[target_language],
            from_language=source_language
        )
        
        results = []
        for item in response:
            translation = item.translations[0]
            results.append({
                'translated_text': translation.text,
                'target_language': translation.to,
                'source_language': item.detected_language.language if source_language is None else source_language
            })
        
        return results


class CrossLanguageSearcher:
    """Enable cross-language search with query translation"""
    
    def __init__(
        self,
        search_client: SearchClient,
        translator: TranslationManager
    ):
        self.search_client = search_client
        self.translator = translator
    
    def search_cross_language(
        self,
        query: str,
        query_language: str,
        target_languages: List[str],
        search_mode: str = 'all'
    ) -> Dict:
        """
        Search across multiple languages
        
        Args:
            query: Search query
            query_language: Language of the query
            target_languages: Languages to search in
            search_mode: 'all' or 'any'
            
        Returns:
            Combined search results
        """
        all_results = []
        
        for target_lang in target_languages:
            # Translate query if different from target
            if query_language != target_lang:
                translation = self.translator.translate_text(
                    text=query,
                    target_language=target_lang,
                    source_language=query_language
                )
                search_query = translation['translated_text']
            else:
                search_query = query
            
            # Search in target language
            results = self.search_client.search(
                search_text=search_query,
                filter=f"language eq '{target_lang}'",
                select=["id", "title", "content", "language", "created_date"],
                top=20
            )
            
            # Add language context to results
            for result in results:
                result['search_language'] = target_lang
                result['original_query'] = query
                result['translated_query'] = search_query
                all_results.append(result)
        
        # Sort by relevance score (if available)
        all_results.sort(
            key=lambda x: x.get('@search.score', 0),
            reverse=True
        )
        
        return {
            'results': all_results[:50],  # Top 50 across all languages
            'total_count': len(all_results),
            'languages_searched': target_languages,
            'query': query,
            'query_language': query_language
        }
```

---

## 4. Language-Specific Analyzers

### Custom Analyzer Configuration

```python
from azure.search.documents.indexes.models import (
    LexicalAnalyzer,
    CustomAnalyzer,
    LexicalTokenizer,
    TokenFilter,
    CharFilter,
    PatternAnalyzer,
    StopwordsTokenFilter,
    LowercaseTokenFilter,
    AsciiFoldingTokenFilter
)

class LanguageAnalyzerConfigurator:
    """Configure custom analyzers for specific languages"""
    
    @staticmethod
    def create_custom_analyzer_for_chinese() -> CustomAnalyzer:
        """
        Create custom analyzer for Chinese text
        
        Uses standard tokenizer with CJK width normalization
        """
        return CustomAnalyzer(
            name="custom_chinese",
            tokenizer_name="standard",
            token_filters=[
                "cjk_width",      # Normalize CJK width differences
                "lowercase",
                "cjk_bigram"      # Create bigrams for CJK characters
            ],
            char_filters=[]
        )
    
    @staticmethod
    def create_custom_analyzer_for_german() -> CustomAnalyzer:
        """
        Create custom analyzer for German with compound word handling
        """
        return CustomAnalyzer(
            name="custom_german",
            tokenizer_name="standard",
            token_filters=[
                "lowercase",
                "german_normalization",  # Handle umlauts
                "german_stemmer"
            ],
            char_filters=[]
        )
    
    @staticmethod
    def create_custom_analyzer_for_french() -> CustomAnalyzer:
        """
        Create custom analyzer for French with elision handling
        """
        return CustomAnalyzer(
            name="custom_french",
            tokenizer_name="standard",
            token_filters=[
                "elision",           # Handle French elisions (l', d', etc.)
                "lowercase",
                "french_stemmer"
            ],
            char_filters=[]
        )
    
    @staticmethod
    def create_stopwords_filter(language: str) -> StopwordsTokenFilter:
        """
        Create stopwords filter for specific language
        
        Args:
            language: Language code
            
        Returns:
            Stopwords token filter
        """
        # Predefined stopwords lists
        stopwords_lists = {
            '_english_': ['a', 'an', 'the', 'is', 'are', 'was', 'were'],
            '_french_': ['le', 'la', 'les', 'un', 'une', 'de', 'du'],
            '_german_': ['der', 'die', 'das', 'ein', 'eine', 'und'],
            '_spanish_': ['el', 'la', 'los', 'las', 'un', 'una', 'de'],
        }
        
        stopwords_key = f'_{language}_'
        
        return StopwordsTokenFilter(
            name=f"stopwords_{language}",
            stopwords=stopwords_lists.get(stopwords_key, []),
            ignore_case=True
        )


class MultilingualSynonymManager:
    """Manage synonyms across languages"""
    
    def __init__(self):
        self.synonym_maps = {}
    
    def create_synonym_map(
        self,
        language: str
    ) -> List[str]:
        """
        Create synonym map for specific language
        
        Args:
            language: Language code
            
        Returns:
            List of synonym rules
        """
        # Example synonym maps
        synonym_maps = {
            'en': [
                'car, automobile, vehicle',
                'laptop, notebook, computer',
                'mobile, cell phone, smartphone'
            ],
            'es': [
                'coche, automóvil, vehículo',
                'portátil, ordenador portátil, computadora portátil',
                'móvil, teléfono móvil, smartphone'
            ],
            'fr': [
                'voiture, automobile, véhicule',
                'ordinateur portable, laptop',
                'mobile, téléphone portable, smartphone'
            ],
            'de': [
                'Auto, Automobil, Fahrzeug',
                'Laptop, Notebook, Computer',
                'Handy, Mobiltelefon, Smartphone'
            ]
        }
        
        return synonym_maps.get(language, [])
    
    def get_cross_language_synonyms(
        self,
        term: str,
        source_language: str,
        target_languages: List[str],
        translator: TranslationManager
    ) -> Dict[str, List[str]]:
        """
        Get synonyms across languages
        
        Args:
            term: Source term
            source_language: Source language code
            target_languages: Target language codes
            translator: Translation manager
            
        Returns:
            Synonyms by language
        """
        synonyms = {source_language: [term]}
        
        for target_lang in target_languages:
            translation = translator.translate_text(
                text=term,
                target_language=target_lang,
                source_language=source_language
            )
            synonyms[target_lang] = [translation['translated_text']]
        
        return synonyms
```

---

## 5. Cultural Relevance

### Region-Specific Ranking

```python
from datetime import datetime
from typing import Dict, List

class CulturalRelevanceOptimizer:
    """Optimize search relevance for cultural context"""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def apply_cultural_boosting(
        self,
        query: str,
        user_region: str,
        user_language: str
    ) -> List[Dict]:
        """
        Apply cultural and regional relevance boosting
        
        Args:
            query: Search query
            user_region: User's region (ISO 3166-1 alpha-2)
            user_language: User's language (ISO 639-1)
            
        Returns:
            Search results with cultural boosting
        """
        # Build scoring parameters
        scoring_params = [
            f"userLanguage-{user_language}",
            f"userRegion-{user_region}"
        ]
        
        # Search with cultural context
        results = self.search_client.search(
            search_text=query,
            scoring_parameters=scoring_params,
            filter=f"language eq '{user_language}' or 'available_languages'/any(lang: lang eq '{user_language}')",
            select=["id", "title", "content", "language", "created_date"],
            top=50
        )
        
        return list(results)
    
    def get_regional_date_format(self, region: str) -> str:
        """
        Get preferred date format for region
        
        Args:
            region: Region code
            
        Returns:
            Date format string
        """
        date_formats = {
            'US': '%m/%d/%Y',      # MM/DD/YYYY
            'GB': '%d/%m/%Y',      # DD/MM/YYYY
            'DE': '%d.%m.%Y',      # DD.MM.YYYY
            'FR': '%d/%m/%Y',      # DD/MM/YYYY
            'JP': '%Y/%m/%d',      # YYYY/MM/DD
            'CN': '%Y-%m-%d',      # YYYY-MM-DD
        }
        
        return date_formats.get(region, '%Y-%m-%d')
    
    def format_currency(
        self,
        amount: float,
        currency: str,
        locale: str
    ) -> str:
        """
        Format currency for locale
        
        Args:
            amount: Currency amount
            currency: Currency code (USD, EUR, GBP, etc.)
            locale: Locale code
            
        Returns:
            Formatted currency string
        """
        # Simplified currency formatting
        currency_symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'CNY': '¥'
        }
        
        symbol = currency_symbols.get(currency, currency)
        
        # Different decimal formats by locale
        if locale in ['en-US']:
            return f"{symbol}{amount:,.2f}"
        elif locale in ['de-DE', 'fr-FR']:
            # European format: use comma for decimal
            formatted = f"{amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            return f"{formatted} {symbol}"
        else:
            return f"{symbol}{amount:.2f}"


class MultilingualContentRouter:
    """Route users to appropriate language content"""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def get_available_languages(
        self,
        document_id: str
    ) -> List[str]:
        """
        Get available languages for a document
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of available language codes
        """
        doc = self.search_client.get_document(key=document_id)
        return doc.get('available_languages', [doc.get('language')])
    
    def get_preferred_language_version(
        self,
        document_id: str,
        user_languages: List[str]
    ) -> Dict:
        """
        Get document in user's preferred language
        
        Args:
            document_id: Document identifier
            user_languages: User's language preferences (ordered)
            
        Returns:
            Document in best available language
        """
        doc = self.search_client.get_document(key=document_id)
        available_languages = doc.get('available_languages', [])
        
        # Find first matching language
        selected_language = None
        for lang in user_languages:
            if lang in available_languages:
                selected_language = lang
                break
        
        # Fallback to document's primary language
        if selected_language is None:
            selected_language = doc.get('language', 'en')
        
        # Return appropriate content fields
        return {
            'id': doc['id'],
            'language': selected_language,
            'title': doc.get(f'title_{selected_language}', doc.get('title')),
            'content': doc.get(f'content_{selected_language}', doc.get('content')),
            'available_languages': available_languages
        }
    
    def suggest_language_alternatives(
        self,
        query: str,
        current_language: str
    ) -> List[Dict]:
        """
        Suggest search results in alternative languages
        
        Args:
            query: Search query
            current_language: Current language
            
        Returns:
            Results in other languages
        """
        # Search in all languages except current
        results = self.search_client.search(
            search_text=query,
            filter=f"language ne '{current_language}'",
            facets=["language"],
            select=["id", "title", "language"],
            top=10
        )
        
        # Group by language
        alternatives = {}
        for result in results:
            lang = result['language']
            if lang not in alternatives:
                alternatives[lang] = []
            alternatives[lang].append(result)
        
        return [
            {
                'language': lang,
                'count': len(docs),
                'sample_results': docs[:3]
            }
            for lang, docs in alternatives.items()
        ]
```

---

## 6. Search Quality for Multilingual Content

### Relevance Testing Across Languages

```python
class MultilingualSearchEvaluator:
    """Evaluate search quality across languages"""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def evaluate_cross_language_precision(
        self,
        test_queries: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Evaluate precision across languages
        
        Args:
            test_queries: Dictionary of language -> list of {query, expected_results}
            
        Returns:
            Precision metrics by language
        """
        results = {}
        
        for language, queries in test_queries.items():
            precision_scores = []
            
            for test_case in queries:
                query = test_case['query']
                expected = set(test_case['expected_results'])
                
                # Execute search
                search_results = self.search_client.search(
                    search_text=query,
                    filter=f"language eq '{language}'",
                    top=10
                )
                
                # Calculate precision
                retrieved = set(r['id'] for r in search_results)
                relevant_retrieved = expected.intersection(retrieved)
                
                precision = len(relevant_retrieved) / len(retrieved) if retrieved else 0
                precision_scores.append(precision)
            
            results[language] = {
                'avg_precision': sum(precision_scores) / len(precision_scores) if precision_scores else 0,
                'query_count': len(queries),
                'min_precision': min(precision_scores) if precision_scores else 0,
                'max_precision': max(precision_scores) if precision_scores else 0
            }
        
        return results
    
    def compare_analyzer_performance(
        self,
        test_query: str,
        language: str,
        analyzers: List[str]
    ) -> Dict:
        """
        Compare different analyzers for a language
        
        Args:
            test_query: Test search query
            language: Language code
            analyzers: List of analyzer names to compare
            
        Returns:
            Comparison results
        """
        results = {}
        
        for analyzer in analyzers:
            # Search with specific analyzer (requires searchFields parameter)
            search_results = self.search_client.search(
                search_text=test_query,
                filter=f"language eq '{language}'",
                top=20
            )
            
            results[analyzer] = {
                'result_count': len(list(search_results)),
                'top_results': [r['title'] for r in list(search_results)[:5]]
            }
        
        return results
```

---

## 7. Best Practices

### Multilingual Search Recommendations

1. **Always Include Language Field**
   - Store detected language in dedicated field
   - Enable filtering by language
   - Track language confidence scores

2. **Use Language-Specific Analyzers**
   - Apply appropriate analyzer for each language
   - Consider custom analyzers for specialized domains
   - Test analyzer performance with sample queries

3. **Implement Fallback Strategies**
   - Default to English if language detection fails
   - Provide alternative language suggestions
   - Support cross-language search when appropriate

4. **Handle Translation Carefully**
   - Cache translated content to reduce API calls
   - Track translation confidence
   - Allow users to view original language

5. **Consider Cultural Context**
   - Format dates/numbers per region
   - Use appropriate currency symbols
   - Respect cultural sensitivities

6. **Optimize Performance**
   - Use language-specific indexes for large datasets
   - Implement language-based routing
   - Cache frequently accessed translations

7. **Test Across Languages**
   - Create test queries for each supported language
   - Measure precision/recall per language
   - Compare analyzer performance

8. **Maintain Consistency**
   - Use standard language codes (ISO 639-1)
   - Document supported languages
   - Provide clear language selection UI

### Common Patterns

- **Separate Fields**: One content field per language (best for small language sets)
- **Single Field**: Dynamic analyzer selection (best for many languages)
- **Hybrid**: Base content + translated versions
- **Translation on Demand**: Translate queries, not documents

### Supported Languages

Azure AI Search supports 50+ languages with Microsoft/Lucene analyzers:

- **Western European**: English, Spanish, French, German, Italian, Portuguese, Dutch
- **Eastern European**: Polish, Russian, Czech, Romanian, Ukrainian
- **Asian**: Chinese (Simplified/Traditional), Japanese, Korean, Thai, Vietnamese
- **Middle Eastern**: Arabic, Hebrew, Persian
- **Indian**: Hindi, Bengali, Tamil, Telugu

---

*Last Updated: November 5, 2025*
