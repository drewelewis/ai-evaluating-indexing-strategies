# Knowledge Mining & AI Enrichment Pipelines

## Table of Contents

- [Overview](#overview)
- [Skillset Architecture](#skillset-architecture)
- [Built-in Cognitive Skills](#built-in-cognitive-skills)
- [Custom Skills](#custom-skills)
- [Knowledge Store](#knowledge-store)
- [Document Cracking](#document-cracking)
- [Enrichment Pipeline Orchestration](#enrichment-pipeline-orchestration)
- [Production Patterns](#production-patterns)
- [Cost Optimization](#cost-optimization)
- [Use Cases](#use-cases)

---

## Overview

### What is Knowledge Mining?

Knowledge mining uses AI to extract insights, structure, and metadata from unstructured content.

```
Raw Documents → AI Skills → Structured Knowledge → Search Index
```

**Key Components**:
- **Skillsets**: AI processing pipelines
- **Cognitive Skills**: Pre-built AI capabilities
- **Knowledge Store**: Persistent enrichment storage
- **Indexers**: Orchestration and scheduling

### When to Use Knowledge Mining

- **OCR**: Extract text from scanned documents or images
- **Entity Recognition**: Identify people, organizations, locations, dates
- **Key Phrase Extraction**: Surface important terms
- **Translation**: Multi-language document processing
- **Custom AI**: Domain-specific extraction (medical codes, legal citations)

---

## Skillset Architecture

### Skillset Definition

```python
from azure.search.documents.indexes.models import (
    SearchIndexerSkillset,
    EntityRecognitionSkill,
    KeyPhraseExtractionSkill,
    OcrSkill,
    MergeSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry
)
from typing import List

class SkillsetBuilder:
    """Build AI enrichment skillsets."""
    
    def __init__(self):
        self.skills = []
        self.name = None
        self.description = None
        self.cognitive_services_key = None
    
    def set_metadata(
        self,
        name: str,
        description: str,
        cognitive_services_key: str = None
    ):
        """
        Set skillset metadata.
        
        Args:
            name: Skillset name
            description: Skillset description
            cognitive_services_key: Optional Cognitive Services key for paid skills
        """
        self.name = name
        self.description = description
        self.cognitive_services_key = cognitive_services_key
        return self
    
    def add_ocr_skill(
        self,
        context: str = "/document/normalized_images/*",
        detect_orientation: bool = True,
        default_language: str = "en"
    ):
        """
        Add OCR skill to extract text from images.
        
        Args:
            context: Context where skill executes
            detect_orientation: Enable orientation detection
            default_language: Default OCR language
        """
        skill = OcrSkill(
            name="ocr-skill",
            description="Extract text from images",
            context=context,
            detect_orientation=detect_orientation,
            default_language_code=default_language,
            inputs=[
                InputFieldMappingEntry(
                    name="image",
                    source="/document/normalized_images/*"
                )
            ],
            outputs=[
                OutputFieldMappingEntry(
                    name="text",
                    target_name="text"
                )
            ]
        )
        self.skills.append(skill)
        return self
    
    def add_entity_recognition_skill(
        self,
        context: str = "/document",
        categories: List[str] = None,
        minimum_precision: float = 0.5
    ):
        """
        Add entity recognition skill.
        
        Args:
            context: Context where skill executes
            categories: Entity categories to extract (Person, Organization, Location, etc.)
            minimum_precision: Minimum confidence threshold
        """
        if categories is None:
            categories = [
                "Person",
                "Organization",
                "Location",
                "DateTime",
                "Quantity",
                "Email",
                "URL"
            ]
        
        skill = EntityRecognitionSkill(
            name="entity-recognition-skill",
            description="Extract named entities",
            context=context,
            categories=categories,
            minimum_precision=minimum_precision,
            inputs=[
                InputFieldMappingEntry(
                    name="text",
                    source="/document/content"
                ),
                InputFieldMappingEntry(
                    name="languageCode",
                    source="/document/language"
                )
            ],
            outputs=[
                OutputFieldMappingEntry(
                    name="persons",
                    target_name="persons"
                ),
                OutputFieldMappingEntry(
                    name="organizations",
                    target_name="organizations"
                ),
                OutputFieldMappingEntry(
                    name="locations",
                    target_name="locations"
                ),
                OutputFieldMappingEntry(
                    name="dateTimes",
                    target_name="dateTimes"
                ),
                OutputFieldMappingEntry(
                    name="entities",
                    target_name="entities"
                )
            ]
        )
        self.skills.append(skill)
        return self
    
    def add_key_phrase_skill(
        self,
        context: str = "/document",
        max_key_phrases: int = 10
    ):
        """
        Add key phrase extraction skill.
        
        Args:
            context: Context where skill executes
            max_key_phrases: Maximum phrases to extract
        """
        skill = KeyPhraseExtractionSkill(
            name="key-phrase-skill",
            description="Extract key phrases",
            context=context,
            max_key_phrase_count=max_key_phrases,
            inputs=[
                InputFieldMappingEntry(
                    name="text",
                    source="/document/content"
                ),
                InputFieldMappingEntry(
                    name="languageCode",
                    source="/document/language"
                )
            ],
            outputs=[
                OutputFieldMappingEntry(
                    name="keyPhrases",
                    target_name="keyPhrases"
                )
            ]
        )
        self.skills.append(skill)
        return self
    
    def add_merge_skill(
        self,
        context: str = "/document",
        insert_pre_tag: str = " ",
        insert_post_tag: str = " "
    ):
        """
        Add merge skill to combine text from multiple sources.
        
        Args:
            context: Context where skill executes
            insert_pre_tag: Text before merged content
            insert_post_tag: Text after merged content
        """
        skill = MergeSkill(
            name="merge-skill",
            description="Merge OCR text with original content",
            context=context,
            insert_pre_tag=insert_pre_tag,
            insert_post_tag=insert_post_tag,
            inputs=[
                InputFieldMappingEntry(
                    name="text",
                    source="/document/content"
                ),
                InputFieldMappingEntry(
                    name="itemsToInsert",
                    source="/document/normalized_images/*/text"
                ),
                InputFieldMappingEntry(
                    name="offsets",
                    source="/document/normalized_images/*/contentOffset"
                )
            ],
            outputs=[
                OutputFieldMappingEntry(
                    name="mergedText",
                    target_name="merged_content"
                )
            ]
        )
        self.skills.append(skill)
        return self
    
    def build(self) -> SearchIndexerSkillset:
        """
        Build the skillset.
        
        Returns:
            SearchIndexerSkillset ready for deployment
        """
        skillset = SearchIndexerSkillset(
            name=self.name,
            description=self.description,
            skills=self.skills
        )
        
        if self.cognitive_services_key:
            from azure.search.documents.indexes.models import CognitiveServicesAccountKey
            skillset.cognitive_services_account = CognitiveServicesAccountKey(
                key=self.cognitive_services_key
            )
        
        return skillset

# Example: Document processing skillset
def create_document_processing_skillset(
    cognitive_services_key: str = None
) -> SearchIndexerSkillset:
    """Create comprehensive document processing skillset."""
    
    builder = SkillsetBuilder()
    builder.set_metadata(
        name="document-processing-skillset",
        description="OCR, entity recognition, and key phrase extraction",
        cognitive_services_key=cognitive_services_key
    )
    
    # Add skills in order of execution
    builder.add_ocr_skill()  # Extract text from images
    builder.add_merge_skill()  # Merge OCR with original content
    builder.add_entity_recognition_skill()  # Extract entities
    builder.add_key_phrase_skill()  # Extract key phrases
    
    return builder.build()
```

### Deploying Skillsets

```python
from azure.search.documents.indexes import SearchIndexerClient
from azure.core.credentials import AzureKeyCredential

class SkillsetManager:
    """Manage skillsets in Azure AI Search."""
    
    def __init__(
        self,
        search_endpoint: str,
        admin_key: str
    ):
        self.client = SearchIndexerClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_or_update_skillset(
        self,
        skillset: SearchIndexerSkillset
    ):
        """
        Create or update a skillset.
        
        Args:
            skillset: Skillset to deploy
        """
        try:
            result = self.client.create_or_update_skillset(skillset)
            print(f"Skillset '{skillset.name}' deployed successfully")
            return result
        except Exception as e:
            print(f"Error deploying skillset: {e}")
            raise
    
    def get_skillset(self, name: str) -> SearchIndexerSkillset:
        """Get skillset by name."""
        return self.client.get_skillset(name)
    
    def delete_skillset(self, name: str):
        """Delete a skillset."""
        self.client.delete_skillset(name)
        print(f"Skillset '{name}' deleted")
    
    def list_skillsets(self) -> List[str]:
        """List all skillset names."""
        skillsets = self.client.get_skillsets()
        return [s.name for s in skillsets]
```

---

## Built-in Cognitive Skills

### OCR (Optical Character Recognition)

Extract text from images and scanned documents.

```python
from azure.search.documents.indexes.models import OcrSkill

def create_advanced_ocr_skill() -> OcrSkill:
    """Create OCR skill with advanced settings."""
    
    skill = OcrSkill(
        name="advanced-ocr",
        description="Extract text from images with orientation detection",
        context="/document/normalized_images/*",
        # Advanced settings
        text_extraction_algorithm="printed",  # or "handwritten"
        default_language_code="en",
        detect_orientation=True,
        line_ending="Space",  # or "CarriageReturn"
        inputs=[
            InputFieldMappingEntry(
                name="image",
                source="/document/normalized_images/*"
            )
        ],
        outputs=[
            OutputFieldMappingEntry(
                name="text",
                target_name="ocrText"
            ),
            OutputFieldMappingEntry(
                name="layoutText",
                target_name="ocrLayoutText"
            )
        ]
    )
    
    return skill
```

**Use Cases**:
- Scanned PDFs
- Invoices and receipts
- Handwritten documents
- Images with embedded text

### Entity Recognition

Identify and extract named entities.

```python
from azure.search.documents.indexes.models import EntityRecognitionSkill

def create_comprehensive_entity_skill() -> EntityRecognitionSkill:
    """Create entity recognition with all categories."""
    
    skill = EntityRecognitionSkill(
        name="comprehensive-entities",
        description="Extract all entity types",
        context="/document",
        categories=[
            "Person",
            "Organization",
            "Location",
            "DateTime",
            "Quantity",
            "Percentage",
            "Email",
            "URL",
            "IP Address",
            "Phone Number"
        ],
        minimum_precision=0.7,
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/content"),
            InputFieldMappingEntry(name="languageCode", source="/document/language")
        ],
        outputs=[
            OutputFieldMappingEntry(name="persons", target_name="persons"),
            OutputFieldMappingEntry(name="organizations", target_name="orgs"),
            OutputFieldMappingEntry(name="locations", target_name="locations"),
            OutputFieldMappingEntry(name="dateTimes", target_name="dates"),
            OutputFieldMappingEntry(name="quantities", target_name="quantities"),
            OutputFieldMappingEntry(name="emails", target_name="emails"),
            OutputFieldMappingEntry(name="urls", target_name="urls"),
            OutputFieldMappingEntry(name="entities", target_name="all_entities")
        ]
    )
    
    return skill
```

**Entity Categories**:
- **Person**: Names of individuals
- **Organization**: Company names, agencies
- **Location**: Cities, countries, landmarks
- **DateTime**: Dates, times, durations
- **Quantity**: Numbers, measurements
- **Email/URL**: Contact information
- **Custom**: Domain-specific entities via custom skills

### Key Phrase Extraction

Surface important terms and concepts.

```python
from azure.search.documents.indexes.models import KeyPhraseExtractionSkill

def create_key_phrase_skill(max_phrases: int = 20) -> KeyPhraseExtractionSkill:
    """Create key phrase extraction skill."""
    
    skill = KeyPhraseExtractionSkill(
        name="key-phrases",
        description="Extract important terms",
        context="/document",
        max_key_phrase_count=max_phrases,
        model_version="latest",
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/content"),
            InputFieldMappingEntry(name="languageCode", source="/document/language")
        ],
        outputs=[
            OutputFieldMappingEntry(name="keyPhrases", target_name="keyPhrases")
        ]
    )
    
    return skill
```

### Language Detection

Detect document language for multilingual processing.

```python
from azure.search.documents.indexes.models import LanguageDetectionSkill

def create_language_detection_skill() -> LanguageDetectionSkill:
    """Create language detection skill."""
    
    skill = LanguageDetectionSkill(
        name="language-detector",
        description="Detect document language",
        context="/document",
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/content")
        ],
        outputs=[
            OutputFieldMappingEntry(name="languageCode", target_name="language"),
            OutputFieldMappingEntry(name="languageName", target_name="languageName"),
            OutputFieldMappingEntry(name="score", target_name="languageScore")
        ]
    )
    
    return skill
```

### Sentiment Analysis

Analyze document sentiment.

```python
from azure.search.documents.indexes.models import SentimentSkill

def create_sentiment_skill() -> SentimentSkill:
    """Create sentiment analysis skill."""
    
    skill = SentimentSkill(
        name="sentiment-analyzer",
        description="Analyze document sentiment",
        context="/document",
        model_version="latest",
        include_opinion_mining=True,  # Detailed aspect-based sentiment
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/content"),
            InputFieldMappingEntry(name="languageCode", source="/document/language")
        ],
        outputs=[
            OutputFieldMappingEntry(name="sentiment", target_name="sentiment"),
            OutputFieldMappingEntry(name="confidenceScores", target_name="sentimentScores"),
            OutputFieldMappingEntry(name="sentences", target_name="sentenceSentiments")
        ]
    )
    
    return skill
```

**Sentiment Labels**:
- **Positive**: 0.6-1.0
- **Neutral**: 0.4-0.6
- **Negative**: 0.0-0.4

### PII Detection and Redaction

Detect and redact personally identifiable information.

```python
from azure.search.documents.indexes.models import PIIDetectionSkill

def create_pii_detection_skill(
    redact: bool = True,
    mask_character: str = "*"
) -> PIIDetectionSkill:
    """
    Create PII detection and redaction skill.
    
    Args:
        redact: Whether to redact detected PII
        mask_character: Character to use for masking
    """
    skill = PIIDetectionSkill(
        name="pii-detector",
        description="Detect and redact PII",
        context="/document",
        categories=[
            "Person",
            "Email",
            "Phone Number",
            "SSN",
            "Credit Card",
            "IP Address",
            "Address"
        ],
        default_language_code="en",
        minimum_precision=0.7,
        mask_character=mask_character if redact else None,
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/content"),
            InputFieldMappingEntry(name="languageCode", source="/document/language")
        ],
        outputs=[
            OutputFieldMappingEntry(name="piiEntities", target_name="piiEntities"),
            OutputFieldMappingEntry(name="maskedText", target_name="redactedContent") if redact else None
        ]
    )
    
    return skill
```

### Image Analysis

Analyze images for tags, descriptions, faces.

```python
from azure.search.documents.indexes.models import ImageAnalysisSkill

def create_image_analysis_skill() -> ImageAnalysisSkill:
    """Create image analysis skill."""
    
    skill = ImageAnalysisSkill(
        name="image-analyzer",
        description="Analyze images for content",
        context="/document/normalized_images/*",
        visual_features=[
            "Tags",
            "Description",
            "Faces",
            "Objects",
            "Brands",
            "Adult",
            "Color"
        ],
        details=["Landmarks", "Celebrities"],
        inputs=[
            InputFieldMappingEntry(name="image", source="/document/normalized_images/*")
        ],
        outputs=[
            OutputFieldMappingEntry(name="tags", target_name="imageTags"),
            OutputFieldMappingEntry(name="description", target_name="imageDescription"),
            OutputFieldMappingEntry(name="faces", target_name="imageFaces"),
            OutputFieldMappingEntry(name="objects", target_name="imageObjects")
        ]
    )
    
    return skill
```

### Translation

Translate documents to target languages.

```python
from azure.search.documents.indexes.models import TextTranslationSkill

def create_translation_skill(
    target_language: str = "en",
    suggested_from: str = None
) -> TextTranslationSkill:
    """
    Create text translation skill.
    
    Args:
        target_language: Target language code
        suggested_from: Optional source language hint
    """
    skill = TextTranslationSkill(
        name="translator",
        description=f"Translate to {target_language}",
        context="/document",
        default_to_language_code=target_language,
        default_from_language_code=suggested_from,
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/content")
        ],
        outputs=[
            OutputFieldMappingEntry(name="translatedText", target_name=f"translated_{target_language}"),
            OutputFieldMappingEntry(name="translatedFromLanguageCode", target_name="sourceLanguage"),
            OutputFieldMappingEntry(name="translatedToLanguageCode", target_name="targetLanguage")
        ]
    )
    
    return skill
```

---

## Custom Skills

### Custom Web API Skill

Call external APIs for domain-specific enrichment.

```python
from azure.search.documents.indexes.models import WebApiSkill

def create_custom_web_api_skill(
    api_endpoint: str,
    api_key: str = None
) -> WebApiSkill:
    """
    Create custom Web API skill.
    
    Args:
        api_endpoint: Custom skill endpoint URL
        api_key: Optional API key for authentication
    """
    headers = {}
    if api_key:
        headers["api-key"] = api_key
    
    skill = WebApiSkill(
        name="custom-extractor",
        description="Custom domain-specific extraction",
        context="/document",
        uri=api_endpoint,
        http_method="POST",
        http_headers=headers,
        timeout="00:03:00",  # 3 minutes
        batch_size=10,
        degree_of_parallelism=5,
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/content"),
            InputFieldMappingEntry(name="metadata", source="/document/metadata")
        ],
        outputs=[
            OutputFieldMappingEntry(name="customEntities", target_name="customEntities"),
            OutputFieldMappingEntry(name="extractedData", target_name="extractedData")
        ]
    )
    
    return skill
```

### Custom Skill Implementation (Azure Function)

```python
import json
import logging
import azure.functions as func
from typing import List, Dict

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function implementing custom skill.
    
    Expected input format:
    {
        "values": [
            {
                "recordId": "0",
                "data": {
                    "text": "Document content...",
                    "metadata": {...}
                }
            }
        ]
    }
    
    Expected output format:
    {
        "values": [
            {
                "recordId": "0",
                "data": {
                    "customEntities": [...],
                    "extractedData": {...}
                },
                "errors": [],
                "warnings": []
            }
        ]
    }
    """
    logging.info("Custom skill function triggered")
    
    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            "Invalid JSON in request body",
            status_code=400
        )
    
    if "values" not in req_body:
        return func.HttpResponse(
            "Request must contain 'values' array",
            status_code=400
        )
    
    results = []
    
    for value in req_body["values"]:
        record_id = value.get("recordId")
        data = value.get("data", {})
        text = data.get("text", "")
        
        try:
            # Custom extraction logic
            extracted = extract_custom_entities(text)
            
            results.append({
                "recordId": record_id,
                "data": {
                    "customEntities": extracted["entities"],
                    "extractedData": extracted["metadata"]
                },
                "errors": [],
                "warnings": []
            })
            
        except Exception as e:
            results.append({
                "recordId": record_id,
                "data": {},
                "errors": [{
                    "message": str(e)
                }],
                "warnings": []
            })
    
    response = {
        "values": results
    }
    
    return func.HttpResponse(
        json.dumps(response),
        mimetype="application/json",
        status_code=200
    )

def extract_custom_entities(text: str) -> Dict:
    """
    Custom entity extraction logic.
    
    Example: Extract medical codes, legal citations, etc.
    """
    import re
    
    # Example: Extract ICD-10 codes
    icd10_pattern = r'\b[A-Z]\d{2}(?:\.\d{1,4})?\b'
    icd10_codes = re.findall(icd10_pattern, text)
    
    # Example: Extract legal case citations
    case_pattern = r'\d+\s+[A-Z][a-z\.]+\s+\d+'
    case_citations = re.findall(case_pattern, text)
    
    entities = []
    
    for code in icd10_codes:
        entities.append({
            "text": code,
            "type": "ICD10_CODE",
            "confidence": 0.9
        })
    
    for citation in case_citations:
        entities.append({
            "text": citation,
            "type": "CASE_CITATION",
            "confidence": 0.85
        })
    
    return {
        "entities": entities,
        "metadata": {
            "icd10_count": len(icd10_codes),
            "citation_count": len(case_citations)
        }
    }
```

### Docker Container Custom Skill

```dockerfile
# Dockerfile for custom skill container
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy skill code
COPY skill.py .

# Expose port
EXPOSE 8080

# Run skill server
CMD ["python", "skill.py"]
```

```python
# skill.py - Flask server for custom skill
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/api/custom-skill', methods=['POST'])
def custom_skill():
    """Custom skill endpoint."""
    try:
        req_data = request.get_json()
        
        results = []
        for value in req_data.get("values", []):
            record_id = value.get("recordId")
            text = value.get("data", {}).get("text", "")
            
            # Custom processing
            extracted = process_document(text)
            
            results.append({
                "recordId": record_id,
                "data": extracted,
                "errors": [],
                "warnings": []
            })
        
        return jsonify({"values": results})
        
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

def process_document(text: str) -> dict:
    """Custom document processing logic."""
    # Implement domain-specific extraction
    return {
        "customField1": "extracted_value",
        "customField2": ["list", "of", "values"]
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

---

## Knowledge Store

### Overview

Knowledge store persists enriched data for analysis beyond search.

**Projection Types**:
- **Table Projections**: Normalized relational data
- **Object Projections**: JSON documents in blob storage
- **File Projections**: Images and binary files

### Defining Knowledge Store

```python
from azure.search.documents.indexes.models import (
    SearchIndexerKnowledgeStore,
    SearchIndexerKnowledgeStoreTableProjectionSelector,
    SearchIndexerKnowledgeStoreObjectProjectionSelector,
    SearchIndexerKnowledgeStoreFileProjectionSelector,
    SearchIndexerKnowledgeStoreProjection
)

def create_knowledge_store(
    storage_connection_string: str
) -> SearchIndexerKnowledgeStore:
    """
    Create knowledge store configuration.
    
    Args:
        storage_connection_string: Azure Storage connection string
    """
    # Table projections - normalized data
    table_projections = [
        SearchIndexerKnowledgeStoreTableProjectionSelector(
            table_name="Documents",
            generated_key_name="DocumentId",
            source="/document"
        ),
        SearchIndexerKnowledgeStoreTableProjectionSelector(
            table_name="Entities",
            generated_key_name="EntityId",
            source="/document/entities/*",
            source_context="/document/entities/*",
            inputs=[
                {"name": "entity", "source": "/document/entities/*/text"},
                {"name": "type", "source": "/document/entities/*/category"},
                {"name": "confidence", "source": "/document/entities/*/confidenceScore"}
            ]
        ),
        SearchIndexerKnowledgeStoreTableProjectionSelector(
            table_name="KeyPhrases",
            generated_key_name="PhraseId",
            source="/document/keyPhrases/*"
        )
    ]
    
    # Object projections - JSON blobs
    object_projections = [
        SearchIndexerKnowledgeStoreObjectProjectionSelector(
            storage_container="enriched-documents",
            generated_key_name="DocumentKey",
            source="/document",
            source_context="/document"
        )
    ]
    
    # File projections - images
    file_projections = [
        SearchIndexerKnowledgeStoreFileProjectionSelector(
            storage_container="images",
            generated_key_name="ImageId",
            source="/document/normalized_images/*"
        )
    ]
    
    # Combine projections
    projection = SearchIndexerKnowledgeStoreProjection(
        tables=table_projections,
        objects=object_projections,
        files=file_projections
    )
    
    knowledge_store = SearchIndexerKnowledgeStore(
        storage_connection_string=storage_connection_string,
        projections=[projection]
    )
    
    return knowledge_store
```

### Querying Knowledge Store

```python
from azure.data.tables import TableServiceClient
from azure.storage.blob import BlobServiceClient

class KnowledgeStoreQuery:
    """Query knowledge store projections."""
    
    def __init__(self, connection_string: str):
        self.table_service = TableServiceClient.from_connection_string(
            connection_string
        )
        self.blob_service = BlobServiceClient.from_connection_string(
            connection_string
        )
    
    def query_entities(
        self,
        entity_type: str = None,
        min_confidence: float = 0.7
    ) -> List[dict]:
        """
        Query entity table projection.
        
        Args:
            entity_type: Filter by entity type (Person, Organization, etc.)
            min_confidence: Minimum confidence score
        """
        table_client = self.table_service.get_table_client("Entities")
        
        # Build filter
        filters = []
        if entity_type:
            filters.append(f"type eq '{entity_type}'")
        if min_confidence:
            filters.append(f"confidence ge {min_confidence}")
        
        filter_str = " and ".join(filters) if filters else None
        
        # Query entities
        entities = table_client.query_entities(filter_str)
        
        return [dict(e) for e in entities]
    
    def get_enriched_document(
        self,
        document_key: str
    ) -> dict:
        """
        Get enriched document from object projection.
        
        Args:
            document_key: Document identifier
        """
        blob_client = self.blob_service.get_blob_client(
            container="enriched-documents",
            blob=f"{document_key}.json"
        )
        
        blob_data = blob_client.download_blob()
        document = json.loads(blob_data.readall())
        
        return document
    
    def analyze_entity_distribution(self) -> Dict[str, int]:
        """Analyze entity type distribution."""
        table_client = self.table_service.get_table_client("Entities")
        
        entities = table_client.query_entities()
        
        distribution = {}
        for entity in entities:
            entity_type = entity.get("type", "Unknown")
            distribution[entity_type] = distribution.get(entity_type, 0) + 1
        
        return distribution
```

---

## Document Cracking

### Configuration

Document cracking extracts content from various file formats.

```python
from azure.search.documents.indexes.models import (
    SearchIndexer,
    IndexingParameters,
    IndexingParametersConfiguration
)

def configure_document_cracking(
    parse_pdf: bool = True,
    extract_images: bool = True,
    ocr_images: bool = True
) -> IndexingParameters:
    """
    Configure document cracking parameters.
    
    Args:
        parse_pdf: Parse PDF structure
        extract_images: Extract embedded images
        ocr_images: Apply OCR to images
    """
    config = IndexingParametersConfiguration()
    
    if parse_pdf:
        # PDF parsing mode
        config.parsing_mode = "default"  # or "text" for text-only
        config.data_to_extract = "contentAndMetadata"  # or "allMetadata"
    
    if extract_images:
        # Image extraction
        config.image_action = "generateNormalizedImages"
        config.normalized_image_max_width = 2000
        config.normalized_image_max_height = 2000
    
    if ocr_images:
        # OCR configuration
        config.allow_skillset_to_read_file_data = True
    
    # Additional settings
    config.fail_on_unprocessable_document = False
    config.fail_on_unsupported_content_type = False
    config.index_storage_metadata_only_for_oversized_documents = True
    config.execution_environment = "standard"  # or "private" for private endpoints
    
    parameters = IndexingParameters(
        configuration=config
    )
    
    return parameters
```

### Supported File Formats

```python
SUPPORTED_FORMATS = {
    "Documents": [".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"],
    "Text": [".txt", ".csv", ".json", ".xml", ".html", ".md"],
    "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
    "Email": [".msg", ".eml"],
    "Compressed": [".zip"]
}

def get_parsing_mode(file_extension: str) -> str:
    """Get recommended parsing mode for file type."""
    
    mode_map = {
        ".pdf": "default",  # Full PDF parsing with structure
        ".json": "json",
        ".csv": "delimitedText",
        ".txt": "text"
    }
    
    return mode_map.get(file_extension.lower(), "default")
```

---

## Enrichment Pipeline Orchestration

### Complete Pipeline

```python
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexer,
    SearchIndexerDataSourceConnection,
    FieldMapping
)

class EnrichmentPipeline:
    """Orchestrate complete enrichment pipeline."""
    
    def __init__(
        self,
        search_endpoint: str,
        admin_key: str
    ):
        self.client = SearchIndexerClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(admin_key)
        )
    
    def create_pipeline(
        self,
        name: str,
        data_source_name: str,
        skillset_name: str,
        index_name: str,
        field_mappings: List[FieldMapping] = None,
        output_field_mappings: List[FieldMapping] = None,
        schedule_interval_minutes: int = None,
        knowledge_store: SearchIndexerKnowledgeStore = None
    ) -> SearchIndexer:
        """
        Create complete enrichment pipeline.
        
        Args:
            name: Indexer name
            data_source_name: Data source to process
            skillset_name: Skillset to apply
            index_name: Target index
            field_mappings: Source field mappings
            output_field_mappings: Enrichment output mappings
            schedule_interval_minutes: Indexing schedule
            knowledge_store: Optional knowledge store configuration
        """
        # Build indexer configuration
        indexer = SearchIndexer(
            name=name,
            description=f"Enrichment pipeline: {name}",
            data_source_name=data_source_name,
            skillset_name=skillset_name,
            target_index_name=index_name,
            field_mappings=field_mappings,
            output_field_mappings=output_field_mappings,
            parameters=configure_document_cracking()
        )
        
        # Add schedule if specified
        if schedule_interval_minutes:
            from azure.search.documents.indexes.models import IndexingSchedule
            from datetime import timedelta
            
            indexer.schedule = IndexingSchedule(
                interval=timedelta(minutes=schedule_interval_minutes)
            )
        
        # Add knowledge store if specified
        if knowledge_store:
            # Knowledge store attached to skillset, not indexer
            # This is a reference example
            pass
        
        # Create indexer
        result = self.client.create_or_update_indexer(indexer)
        print(f"Pipeline '{name}' created successfully")
        
        return result
    
    def run_pipeline(self, name: str):
        """Run indexer pipeline manually."""
        self.client.run_indexer(name)
        print(f"Pipeline '{name}' started")
    
    def get_pipeline_status(self, name: str) -> dict:
        """Get indexer execution status."""
        status = self.client.get_indexer_status(name)
        
        last_result = status.last_result
        if last_result:
            return {
                "status": last_result.status,
                "error_message": last_result.error_message,
                "items_processed": last_result.items_processed,
                "items_failed": last_result.items_failed,
                "start_time": last_result.start_time,
                "end_time": last_result.end_time
            }
        
        return {"status": "Not run yet"}
    
    def reset_pipeline(self, name: str):
        """Reset indexer to reprocess all documents."""
        self.client.reset_indexer(name)
        print(f"Pipeline '{name}' reset")
```

### Field Mappings

```python
from azure.search.documents.indexes.models import FieldMapping

def create_field_mappings() -> List[FieldMapping]:
    """Create source field mappings."""
    
    mappings = [
        FieldMapping(
            source_field_name="metadata_storage_path",
            target_field_name="id",
            mapping_function={"name": "base64Encode"}
        ),
        FieldMapping(
            source_field_name="metadata_storage_name",
            target_field_name="fileName"
        ),
        FieldMapping(
            source_field_name="metadata_content_type",
            target_field_name="contentType"
        )
    ]
    
    return mappings

def create_output_field_mappings() -> List[FieldMapping]:
    """Create enrichment output field mappings."""
    
    mappings = [
        FieldMapping(
            source_field_name="/document/merged_content",
            target_field_name="content"
        ),
        FieldMapping(
            source_field_name="/document/entities",
            target_field_name="entities"
        ),
        FieldMapping(
            source_field_name="/document/keyPhrases",
            target_field_name="keyPhrases"
        ),
        FieldMapping(
            source_field_name="/document/language",
            target_field_name="language"
        )
    ]
    
    return mappings
```

---

## Production Patterns

### Error Handling

```python
class RobustEnrichmentPipeline(EnrichmentPipeline):
    """Pipeline with comprehensive error handling."""
    
    def run_with_retry(
        self,
        name: str,
        max_retries: int = 3,
        retry_delay_seconds: int = 60
    ):
        """Run pipeline with retry logic."""
        import time
        
        for attempt in range(max_retries):
            try:
                self.run_pipeline(name)
                
                # Wait for completion
                time.sleep(30)
                
                # Check status
                status = self.get_pipeline_status(name)
                
                if status["status"] == "success":
                    print(f"Pipeline completed successfully")
                    return status
                elif status["status"] == "transientFailure":
                    print(f"Transient failure, retrying... (attempt {attempt + 1})")
                    time.sleep(retry_delay_seconds)
                else:
                    print(f"Pipeline failed: {status.get('error_message')}")
                    return status
                    
            except Exception as e:
                print(f"Error running pipeline: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_seconds)
                else:
                    raise
    
    def monitor_pipeline(
        self,
        name: str,
        check_interval_seconds: int = 60,
        max_checks: int = 60
    ):
        """Monitor pipeline execution."""
        import time
        
        for i in range(max_checks):
            status = self.get_pipeline_status(name)
            
            print(f"Check {i + 1}: Status={status['status']}, "
                  f"Processed={status.get('items_processed', 0)}, "
                  f"Failed={status.get('items_failed', 0)}")
            
            if status["status"] in ["success", "error"]:
                return status
            
            time.sleep(check_interval_seconds)
        
        print("Monitoring timeout reached")
        return status
```

### Incremental Enrichment

```python
def configure_incremental_processing() -> IndexingParameters:
    """Configure incremental change detection."""
    
    config = IndexingParametersConfiguration()
    
    # Enable change detection
    config.fail_on_unprocessable_document = False
    
    # Incremental processing
    # Indexer will only process new/modified documents
    # based on data source high water mark
    
    parameters = IndexingParameters(
        configuration=config
    )
    
    return parameters
```

### Debugging Pipeline

```python
class PipelineDebugger:
    """Debug enrichment pipeline issues."""
    
    def __init__(self, indexer_client: SearchIndexerClient):
        self.client = indexer_client
    
    def get_document_enrichment(
        self,
        indexer_name: str,
        document_id: str
    ) -> dict:
        """
        Get enrichment details for specific document.
        
        This requires debug session to be enabled.
        """
        # Enable debug sessions in Azure Portal
        # Navigate to: Indexer → Debug Session
        
        # Example debug output structure
        return {
            "document_id": document_id,
            "enrichment_steps": [
                {
                    "skill": "OCR",
                    "status": "success",
                    "output": "Extracted text..."
                },
                {
                    "skill": "EntityRecognition",
                    "status": "success",
                    "output": ["Entity1", "Entity2"]
                }
            ]
        }
    
    def analyze_failures(self, indexer_name: str) -> List[dict]:
        """Analyze failed documents."""
        status = self.client.get_indexer_status(indexer_name)
        
        failures = []
        for execution in status.execution_history:
            if execution.errors:
                for error in execution.errors:
                    failures.append({
                        "document_key": error.key,
                        "error_message": error.error_message,
                        "status_code": error.status_code,
                        "timestamp": execution.start_time
                    })
        
        return failures
```

---

## Cost Optimization

### Caching Enrichments

```python
from azure.search.documents.indexes.models import SearchIndexerCache

def enable_enrichment_caching(
    storage_connection_string: str
) -> SearchIndexerCache:
    """
    Enable enrichment caching to reduce AI API calls.
    
    Cached enrichments are reused for unchanged documents.
    """
    cache = SearchIndexerCache(
        storage_connection_string=storage_connection_string,
        enable_reprocessing=True
    )
    
    return cache

# Apply to indexer
def create_cached_indexer(
    name: str,
    data_source: str,
    skillset: str,
    index: str,
    cache_connection_string: str
) -> SearchIndexer:
    """Create indexer with enrichment caching."""
    
    indexer = SearchIndexer(
        name=name,
        data_source_name=data_source,
        skillset_name=skillset,
        target_index_name=index,
        cache=enable_enrichment_caching(cache_connection_string)
    )
    
    return indexer
```

### Skill Selection Strategy

```python
def optimize_skillset_for_cost(document_type: str) -> List:
    """
    Select minimal necessary skills based on document type.
    
    Fewer skills = lower AI API costs.
    """
    skill_recommendations = {
        "scanned_pdf": [
            "OCR",  # Required for text extraction
            "EntityRecognition",  # High value
            "KeyPhraseExtraction"  # High value
        ],
        "text_document": [
            "EntityRecognition",
            "KeyPhraseExtraction",
            "LanguageDetection"
        ],
        "image": [
            "OCR",
            "ImageAnalysis"
        ],
        "minimal": [
            "EntityRecognition"  # Single most valuable skill
        ]
    }
    
    return skill_recommendations.get(document_type, skill_recommendations["minimal"])
```

### Batch Processing

```python
def configure_batch_processing(
    batch_size: int = 100,
    max_parallelism: int = 10
) -> IndexingParameters:
    """Configure indexer for efficient batch processing."""
    
    config = IndexingParametersConfiguration()
    config.batch_size = batch_size
    config.max_failed_items = 10
    config.max_failed_items_per_batch = 5
    
    parameters = IndexingParameters(
        configuration=config,
        max_degree_of_parallelism=max_parallelism
    )
    
    return parameters
```

---

## Use Cases

### Contract Analysis

```python
def create_contract_analysis_skillset() -> SearchIndexerSkillset:
    """Skillset for legal contract processing."""
    
    builder = SkillsetBuilder()
    builder.set_metadata(
        name="contract-analyzer",
        description="Extract entities, dates, and parties from contracts"
    )
    
    # Extract key information
    builder.add_entity_recognition_skill(
        categories=["Person", "Organization", "DateTime", "Quantity"]
    )
    
    builder.add_key_phrase_skill(max_key_phrases=20)
    
    # Add custom skill for contract-specific fields
    # (parties, effective dates, termination clauses, etc.)
    
    return builder.build()
```

### Medical Records

```python
def create_medical_records_skillset() -> SearchIndexerSkillset:
    """Skillset for medical document processing."""
    
    builder = SkillsetBuilder()
    builder.set_metadata(
        name="medical-records-processor",
        description="Extract medical entities and redact PII"
    )
    
    # OCR for scanned records
    builder.add_ocr_skill()
    builder.add_merge_skill()
    
    # PII detection and redaction (HIPAA compliance)
    # This would use PIIDetectionSkill
    
    # Custom skill for ICD-10 codes, medications, diagnoses
    # WebApiSkill pointing to medical entity extraction API
    
    return builder.build()
```

### Financial Documents

```python
def create_financial_analysis_skillset() -> SearchIndexerSkillset:
    """Skillset for financial document processing."""
    
    builder = SkillsetBuilder()
    builder.set_metadata(
        name="financial-analyzer",
        description="Extract financial entities and amounts"
    )
    
    builder.add_entity_recognition_skill(
        categories=["DateTime", "Quantity", "Percentage", "Organization"]
    )
    
    # Custom skill for:
    # - Currency amounts
    # - Account numbers
    # - Transaction IDs
    # - Stock symbols
    
    return builder.build()
```

### Legal Discovery

```python
def create_legal_discovery_skillset() -> SearchIndexerSkillset:
    """Skillset for legal document discovery."""
    
    builder = SkillsetBuilder()
    builder.set_metadata(
        name="legal-discovery",
        description="Extract case citations, precedents, and legal entities"
    )
    
    builder.add_ocr_skill()  # Many legal docs are scanned
    builder.add_entity_recognition_skill()
    builder.add_key_phrase_skill()
    
    # Custom skill for:
    # - Case citations (e.g., "123 F.3d 456")
    # - Statute references
    # - Legal precedents
    # - Judge names
    
    return builder.build()
```

---

## Best Practices

### Skillset Design

1. **Order Skills Logically**: OCR → Merge → Language Detection → Entity Recognition
2. **Minimize Redundancy**: Don't extract same information with multiple skills
3. **Use Caching**: Enable enrichment cache for large document sets
4. **Handle Errors Gracefully**: Configure `failOnUnsupportedContentType: false`
5. **Monitor Costs**: Track Cognitive Services API usage

### Knowledge Store Design

1. **Normalize Tables**: Separate entities, key phrases into distinct tables
2. **Include Timestamps**: Track when enrichments were created
3. **Add Confidence Scores**: Store entity/skill confidence for filtering
4. **Link Projections**: Use generated keys to relate tables/objects

### Custom Skills

1. **Implement Timeouts**: Skills should respond within 3 minutes
2. **Support Batching**: Process multiple records per request
3. **Return Structured Errors**: Include error details in response
4. **Version APIs**: Support backward compatibility
5. **Monitor Performance**: Track latency and throughput

### Production Deployment

1. **Use Private Endpoints**: Secure Cognitive Services and Storage
2. **Enable Diagnostic Logging**: Monitor enrichment pipeline health
3. **Set Up Alerts**: Notify on indexer failures
4. **Test Incrementally**: Validate skills on sample documents first
5. **Document Skillsets**: Maintain clear documentation of extraction logic

---

For integration with Azure OpenAI embeddings, see [Azure OpenAI Integration (Page 05)](./05-azure-openai-integration.md).

For monitoring enrichment pipelines, see [Monitoring & Alerting (Page 23)](./23-monitoring-alerting.md).
