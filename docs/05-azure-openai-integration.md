# Azure OpenAI Integration

Complete guide to integrating Azure OpenAI Service for embedding generation, semantic capabilities, and AI-powered search enhancements.

## 📋 Table of Contents
- [Overview](#overview)
- [Service Setup](#service-setup)
- [Document Parsing & Chunking](#document-parsing--chunking)
- [Data Enrichment](#data-enrichment)
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

## Document Parsing & Chunking

### Overview: PDF Processing Pipeline

For PDF documents with complex structure (tables, headers, lists), chunking must respect document layout to maintain context and searchability.

```
PDF → Azure AI Document Intelligence → Layout Analysis → Smart Chunking → Embeddings → Index
```

### Azure AI Document Intelligence Integration

```python
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class ChunkType(Enum):
    """Types of document chunks."""
    PARAGRAPH = "paragraph"
    TABLE = "table"
    HEADER = "header"
    LIST = "list"
    FOOTER = "footer"

@dataclass
class DocumentChunk:
    """A chunk of document content with metadata."""
    chunk_id: str
    content: str
    chunk_type: ChunkType
    page_number: int
    bounding_box: Optional[dict]
    parent_section: Optional[str]
    char_count: int
    token_estimate: int
    
    def to_search_document(self, doc_id: str, chunk_index: int) -> dict:
        """Convert to Azure AI Search document."""
        return {
            "id": f"{doc_id}_{chunk_index}",
            "document_id": doc_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "page_number": self.page_number,
            "parent_section": self.parent_section,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate
        }

class PDFDocumentParser:
    """Parse PDF documents using Azure AI Document Intelligence."""
    
    def __init__(self, endpoint: str, api_key: str):
        self.client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
        
    def parse_pdf(self, pdf_path: str, model: str = "prebuilt-layout") -> dict:
        """
        Parse PDF with layout analysis.
        
        Args:
            pdf_path: Path to PDF file
            model: "prebuilt-layout", "prebuilt-read", or "prebuilt-document"
            
        Returns:
            Parsed document structure
        """
        with open(pdf_path, "rb") as f:
            poller = self.client.begin_analyze_document(
                model_id=model,
                document=f
            )
        
        result = poller.result()
        
        return {
            "content": result.content,
            "pages": self._extract_pages(result.pages),
            "paragraphs": self._extract_paragraphs(result.paragraphs),
            "tables": self._extract_tables(result.tables),
            "sections": self._extract_sections(result)
        }
    
    def _extract_pages(self, pages) -> List[dict]:
        """Extract page information."""
        return [
            {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "angle": page.angle,
                "unit": page.unit,
                "lines": [
                    {
                        "content": line.content,
                        "bounding_box": line.polygon
                    }
                    for line in page.lines
                ] if page.lines else []
            }
            for page in pages
        ]
    
    def _extract_paragraphs(self, paragraphs) -> List[dict]:
        """Extract paragraph information."""
        if not paragraphs:
            return []
            
        return [
            {
                "content": para.content,
                "bounding_regions": [
                    {
                        "page_number": region.page_number,
                        "polygon": region.polygon
                    }
                    for region in para.bounding_regions
                ] if para.bounding_regions else [],
                "role": para.role if hasattr(para, 'role') else None
            }
            for para in paragraphs
        ]
    
    def _extract_tables(self, tables) -> List[dict]:
        """Extract table information with structure."""
        if not tables:
            return []
            
        extracted_tables = []
        for table in tables:
            # Convert table to markdown format
            rows = {}
            for cell in table.cells:
                row_idx = cell.row_index
                if row_idx not in rows:
                    rows[row_idx] = {}
                rows[row_idx][cell.column_index] = cell.content
            
            # Build markdown table
            markdown_rows = []
            for row_idx in sorted(rows.keys()):
                row_cells = rows[row_idx]
                row_content = " | ".join(
                    row_cells.get(i, "") 
                    for i in range(table.column_count)
                )
                markdown_rows.append(f"| {row_content} |")
                
                # Add header separator after first row
                if row_idx == 0:
                    separator = " | ".join(["---"] * table.column_count)
                    markdown_rows.append(f"| {separator} |")
            
            extracted_tables.append({
                "row_count": table.row_count,
                "column_count": table.column_count,
                "markdown": "\n".join(markdown_rows),
                "bounding_regions": [
                    {
                        "page_number": region.page_number,
                        "polygon": region.polygon
                    }
                    for region in table.bounding_regions
                ] if table.bounding_regions else []
            })
        
        return extracted_tables
    
    def _extract_sections(self, result) -> List[dict]:
        """Extract document sections based on styles."""
        sections = []
        current_section = None
        
        if hasattr(result, 'paragraphs') and result.paragraphs:
            for para in result.paragraphs:
                # Check if paragraph is a heading
                role = getattr(para, 'role', None)
                
                if role and 'heading' in role.lower():
                    if current_section:
                        sections.append(current_section)
                    current_section = {
                        "title": para.content,
                        "level": self._extract_heading_level(role),
                        "content_paragraphs": []
                    }
                elif current_section:
                    current_section["content_paragraphs"].append(para.content)
            
            if current_section:
                sections.append(current_section)
        
        return sections
    
    def _extract_heading_level(self, role: str) -> int:
        """Extract heading level from role string."""
        import re
        match = re.search(r'(\d+)', role)
        return int(match.group(1)) if match else 1

class StructureAwareChunker:
    """
    Chunk documents while preserving structure.
    
    Key principles:
    1. Never split tables across chunks
    2. Keep headers with their content
    3. Respect paragraph boundaries
    4. Maintain parent section context
    """
    
    def __init__(
        self,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 50,
        preserve_tables: bool = True,
        preserve_sections: bool = True
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.preserve_tables = preserve_tables
        self.preserve_sections = preserve_sections
        
    def chunk_document(
        self,
        parsed_doc: dict,
        doc_id: str
    ) -> List[DocumentChunk]:
        """
        Chunk document respecting structure.
        
        Args:
            parsed_doc: Output from PDFDocumentParser
            doc_id: Document identifier
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        # Process sections if available
        if self.preserve_sections and parsed_doc.get("sections"):
            chunks.extend(
                self._chunk_sections(parsed_doc["sections"], doc_id)
            )
        
        # Process tables separately
        if self.preserve_tables and parsed_doc.get("tables"):
            chunks.extend(
                self._chunk_tables(parsed_doc["tables"], doc_id)
            )
        
        # Process remaining paragraphs
        if parsed_doc.get("paragraphs"):
            chunks.extend(
                self._chunk_paragraphs(
                    parsed_doc["paragraphs"],
                    doc_id,
                    exclude_roles=["heading"] if self.preserve_sections else []
                )
            )
        
        # Add chunk indices
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{doc_id}_chunk_{i:04d}"
        
        return chunks
    
    def _chunk_sections(
        self,
        sections: List[dict],
        doc_id: str
    ) -> List[DocumentChunk]:
        """Chunk document sections."""
        chunks = []
        
        for section in sections:
            section_title = section["title"]
            section_content = "\n\n".join(section["content_paragraphs"])
            
            # Combine title and content
            full_content = f"# {section_title}\n\n{section_content}"
            
            # Check if section fits in one chunk
            tokens = self._estimate_tokens(full_content)
            
            if tokens <= self.max_chunk_tokens:
                # Single chunk for entire section
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_section",
                    content=full_content,
                    chunk_type=ChunkType.HEADER,
                    page_number=1,  # TODO: Get from bounding regions
                    bounding_box=None,
                    parent_section=section_title,
                    char_count=len(full_content),
                    token_estimate=tokens
                ))
            else:
                # Split section content into smaller chunks
                section_chunks = self._split_text_with_overlap(
                    text=section_content,
                    prefix=f"# {section_title}\n\n",
                    parent_section=section_title
                )
                chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_tables(
        self,
        tables: List[dict],
        doc_id: str
    ) -> List[DocumentChunk]:
        """Chunk tables (keep tables whole)."""
        chunks = []
        
        for i, table in enumerate(tables):
            markdown = table["markdown"]
            
            # Get page number from bounding regions
            page_number = 1
            if table.get("bounding_regions"):
                page_number = table["bounding_regions"][0]["page_number"]
            
            # Create context-rich table chunk
            table_content = f"[TABLE {i+1}]\n{markdown}"
            
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_table_{i}",
                content=table_content,
                chunk_type=ChunkType.TABLE,
                page_number=page_number,
                bounding_box=table.get("bounding_regions", [{}])[0].get("polygon"),
                parent_section=None,
                char_count=len(table_content),
                token_estimate=self._estimate_tokens(table_content)
            ))
        
        return chunks
    
    def _chunk_paragraphs(
        self,
        paragraphs: List[dict],
        doc_id: str,
        exclude_roles: List[str] = None
    ) -> List[DocumentChunk]:
        """Chunk paragraphs with overlap."""
        exclude_roles = exclude_roles or []
        chunks = []
        
        # Filter paragraphs
        filtered_paras = [
            p for p in paragraphs
            if not (p.get("role") and p["role"].lower() in exclude_roles)
        ]
        
        # Combine paragraphs into chunks
        current_chunk = []
        current_tokens = 0
        
        for para in filtered_paras:
            para_text = para["content"]
            para_tokens = self._estimate_tokens(para_text)
            
            # Check if adding paragraph exceeds limit
            if current_tokens + para_tokens > self.max_chunk_tokens:
                if current_chunk:
                    # Save current chunk
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc_id}_para",
                        content=chunk_text,
                        chunk_type=ChunkType.PARAGRAPH,
                        page_number=self._get_page_number(para),
                        bounding_box=self._get_bounding_box(para),
                        parent_section=None,
                        char_count=len(chunk_text),
                        token_estimate=current_tokens
                    ))
                    
                    # Start new chunk with overlap
                    if self.overlap_tokens > 0 and len(current_chunk) > 1:
                        # Keep last paragraph for overlap
                        current_chunk = [current_chunk[-1], para_text]
                        current_tokens = (
                            self._estimate_tokens(current_chunk[-2]) +
                            para_tokens
                        )
                    else:
                        current_chunk = [para_text]
                        current_tokens = para_tokens
                else:
                    # Single paragraph too long - split it
                    split_chunks = self._split_long_paragraph(para_text, doc_id)
                    chunks.extend(split_chunks)
            else:
                current_chunk.append(para_text)
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_para",
                content=chunk_text,
                chunk_type=ChunkType.PARAGRAPH,
                page_number=1,
                bounding_box=None,
                parent_section=None,
                char_count=len(chunk_text),
                token_estimate=current_tokens
            ))
        
        return chunks
    
    def _split_text_with_overlap(
        self,
        text: str,
        prefix: str = "",
        parent_section: str = None
    ) -> List[DocumentChunk]:
        """Split long text into overlapping chunks."""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_tokens = self._estimate_tokens(prefix)
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.max_chunk_tokens:
                if current_chunk:
                    chunk_text = prefix + " ".join(current_chunk)
                    chunks.append(DocumentChunk(
                        chunk_id="temp",
                        content=chunk_text,
                        chunk_type=ChunkType.PARAGRAPH,
                        page_number=1,
                        bounding_box=None,
                        parent_section=parent_section,
                        char_count=len(chunk_text),
                        token_estimate=current_tokens
                    ))
                    
                    # Overlap: keep last few sentences
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk,
                        self.overlap_tokens
                    )
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = sum(
                        self._estimate_tokens(s) for s in current_chunk
                    ) + self._estimate_tokens(prefix)
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens + self._estimate_tokens(prefix)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunk_text = prefix + " ".join(current_chunk)
            chunks.append(DocumentChunk(
                chunk_id="temp",
                content=chunk_text,
                chunk_type=ChunkType.PARAGRAPH,
                page_number=1,
                bounding_box=None,
                parent_section=parent_section,
                char_count=len(chunk_text),
                token_estimate=current_tokens
            ))
        
        return chunks
    
    def _split_long_paragraph(
        self,
        text: str,
        doc_id: str
    ) -> List[DocumentChunk]:
        """Split a very long paragraph."""
        return self._split_text_with_overlap(text, "", None)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(
        self,
        sentences: List[str],
        target_tokens: int
    ) -> List[str]:
        """Get last N sentences for overlap."""
        overlap = []
        tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            if tokens + sentence_tokens > target_tokens:
                break
            overlap.insert(0, sentence)
            tokens += sentence_tokens
        
        return overlap
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def _get_page_number(self, para: dict) -> int:
        """Extract page number from paragraph."""
        if para.get("bounding_regions"):
            return para["bounding_regions"][0].get("page_number", 1)
        return 1
    
    def _get_bounding_box(self, para: dict) -> Optional[dict]:
        """Extract bounding box from paragraph."""
        if para.get("bounding_regions"):
            return para["bounding_regions"][0].get("polygon")
        return None

# Example: End-to-end PDF processing
def process_pdf_for_search(
    pdf_path: str,
    doc_id: str,
    form_recognizer_endpoint: str,
    form_recognizer_key: str
) -> List[DocumentChunk]:
    """
    Complete PDF processing pipeline.
    
    Args:
        pdf_path: Path to PDF file
        doc_id: Document identifier
        form_recognizer_endpoint: Azure AI Document Intelligence endpoint
        form_recognizer_key: API key
        
    Returns:
        List of searchable document chunks
    """
    # Step 1: Parse PDF
    parser = PDFDocumentParser(
        endpoint=form_recognizer_endpoint,
        api_key=form_recognizer_key
    )
    parsed_doc = parser.parse_pdf(pdf_path)
    
    # Step 2: Chunk with structure awareness
    chunker = StructureAwareChunker(
        max_chunk_tokens=512,
        overlap_tokens=50,
        preserve_tables=True,
        preserve_sections=True
    )
    chunks = chunker.chunk_document(parsed_doc, doc_id)
    
    # Step 3: Convert to search documents
    search_docs = [
        chunk.to_search_document(doc_id, i)
        for i, chunk in enumerate(chunks)
    ]
    
    return search_docs
```

### Chunking Strategy Comparison

```python
class ChunkingStrategyEvaluator:
    """Compare different chunking strategies."""
    
    def __init__(self, embedder):
        self.embedder = embedder
        
    def evaluate_strategies(
        self,
        pdf_path: str,
        ground_truth_queries: List[dict]
    ) -> dict:
        """
        Evaluate different chunking strategies.
        
        Args:
            pdf_path: PDF to chunk
            ground_truth_queries: List of {query, expected_chunk_ids}
            
        Returns:
            Comparison metrics
        """
        strategies = {
            "fixed_256": {"max_chunk_tokens": 256, "overlap_tokens": 25},
            "fixed_512": {"max_chunk_tokens": 512, "overlap_tokens": 50},
            "fixed_1024": {"max_chunk_tokens": 1024, "overlap_tokens": 100},
            "structure_aware": {
                "max_chunk_tokens": 512,
                "overlap_tokens": 50,
                "preserve_tables": True,
                "preserve_sections": True
            }
        }
        
        results = {}
        
        for name, config in strategies.items():
            chunker = StructureAwareChunker(**config)
            # Evaluate chunking quality
            metrics = self._evaluate_chunking(
                chunker,
                pdf_path,
                ground_truth_queries
            )
            results[name] = metrics
        
        return results
    
    def _evaluate_chunking(
        self,
        chunker: StructureAwareChunker,
        pdf_path: str,
        ground_truth: List[dict]
    ) -> dict:
        """Evaluate a chunking strategy."""
        # Simplified evaluation
        return {
            "avg_chunk_size": 450,
            "num_chunks": 25,
            "table_preservation": 0.95,
            "context_coherence": 0.88
        }
```

---

## Data Enrichment

### Overview: Enriching Chunks with Metadata

After chunking PDFs, enrich each chunk with extracted metadata to improve searchability and filtering.

```
Chunks → Entity Extraction → Key Phrases → Language Detection → Enriched Chunks → Embeddings
```

### Azure AI Language Integration

```python
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict

class ChunkEnricher:
    """Enrich document chunks with AI-extracted metadata."""
    
    def __init__(
        self,
        language_endpoint: str,
        language_key: str
    ):
        self.client = TextAnalyticsClient(
            endpoint=language_endpoint,
            credential=AzureKeyCredential(language_key)
        )
    
    def enrich_chunk(self, chunk: DocumentChunk) -> dict:
        """
        Enrich a single chunk with metadata.
        
        Args:
            chunk: DocumentChunk to enrich
            
        Returns:
            Enriched metadata dictionary
        """
        content = chunk.content
        
        # Extract entities
        entities = self._extract_entities(content)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(content)
        
        # Detect language
        language = self._detect_language(content)
        
        # Build enriched metadata
        enriched = {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "chunk_type": chunk.chunk_type.value,
            "page_number": chunk.page_number,
            "parent_section": chunk.parent_section,
            "char_count": chunk.char_count,
            "token_estimate": chunk.token_estimate,
            # Enrichment fields
            "entities": entities,
            "key_phrases": key_phrases,
            "language": language,
            "entity_count": len(entities),
            "key_phrase_count": len(key_phrases)
        }
        
        return enriched
    
    def _extract_entities(self, text: str) -> List[dict]:
        """Extract named entities from text."""
        try:
            result = self.client.recognize_entities(
                documents=[text],
                language="en"
            )[0]
            
            if result.is_error:
                return []
            
            entities = []
            for entity in result.entities:
                entities.append({
                    "text": entity.text,
                    "category": entity.category,
                    "subcategory": entity.subcategory,
                    "confidence": entity.confidence_score
                })
            
            return entities
            
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        try:
            result = self.client.extract_key_phrases(
                documents=[text],
                language="en"
            )[0]
            
            if result.is_error:
                return []
            
            return list(result.key_phrases)
            
        except Exception as e:
            print(f"Key phrase extraction error: {e}")
            return []
    
    def _detect_language(self, text: str) -> str:
        """Detect primary language of text."""
        try:
            result = self.client.detect_language(
                documents=[text]
            )[0]
            
            if result.is_error:
                return "en"
            
            return result.primary_language.iso6391_name
            
        except Exception as e:
            print(f"Language detection error: {e}")
            return "en"
    
    def enrich_batch(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 10
    ) -> List[dict]:
        """
        Enrich multiple chunks in batches.
        
        Args:
            chunks: List of DocumentChunks
            batch_size: Batch size for API calls
            
        Returns:
            List of enriched metadata dictionaries
        """
        enriched_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            for chunk in batch:
                enriched = self.enrich_chunk(chunk)
                enriched_chunks.append(enriched)
        
        return enriched_chunks

class CustomMetadataExtractor:
    """Extract custom domain-specific metadata."""
    
    def __init__(self):
        self.patterns = {}
    
    def add_pattern(self, name: str, pattern: str):
        """
        Add custom extraction pattern.
        
        Args:
            name: Metadata field name
            pattern: Regex pattern to match
        """
        import re
        self.patterns[name] = re.compile(pattern)
    
    def extract_metadata(self, text: str) -> dict:
        """
        Extract custom metadata using patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        
        for name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            metadata[name] = matches if matches else []
        
        return metadata

# Example: Domain-specific patterns
def create_document_metadata_extractor() -> CustomMetadataExtractor:
    """Create extractor for common document patterns."""
    extractor = CustomMetadataExtractor()
    
    # Date patterns
    extractor.add_pattern(
        "dates",
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    )
    
    # Email addresses
    extractor.add_pattern(
        "emails",
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    
    # Phone numbers
    extractor.add_pattern(
        "phones",
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    )
    
    # Document IDs (example pattern)
    extractor.add_pattern(
        "document_ids",
        r'\b[A-Z]{2,4}-\d{4,8}\b'
    )
    
    return extractor

class EnrichedChunkIndexer:
    """Index enriched chunks to Azure AI Search."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def prepare_search_document(
        self,
        enriched_chunk: dict,
        embedding: List[float] = None
    ) -> dict:
        """
        Prepare enriched chunk for indexing.
        
        Args:
            enriched_chunk: Enriched chunk metadata
            embedding: Optional embedding vector
            
        Returns:
            Search document ready for indexing
        """
        # Flatten entities for search
        entity_names = [e["text"] for e in enriched_chunk.get("entities", [])]
        entity_categories = list(set(
            e["category"] for e in enriched_chunk.get("entities", [])
        ))
        
        search_doc = {
            "id": enriched_chunk["chunk_id"],
            "content": enriched_chunk["content"],
            "content_vector": embedding,
            "chunk_type": enriched_chunk["chunk_type"],
            "page_number": enriched_chunk["page_number"],
            "parent_section": enriched_chunk["parent_section"],
            "language": enriched_chunk.get("language", "en"),
            # Searchable enrichment fields
            "key_phrases": enriched_chunk.get("key_phrases", []),
            "entity_names": entity_names,
            "entity_categories": entity_categories,
            "entity_count": enriched_chunk.get("entity_count", 0),
            # Filterable metadata
            "has_entities": len(entity_names) > 0,
            "has_key_phrases": len(enriched_chunk.get("key_phrases", [])) > 0
        }
        
        return search_doc
    
    def index_enriched_chunks(
        self,
        enriched_chunks: List[dict],
        embeddings: List[List[float]] = None
    ):
        """
        Index enriched chunks to search service.
        
        Args:
            enriched_chunks: List of enriched chunks
            embeddings: Optional list of embedding vectors
        """
        search_docs = []
        
        for i, chunk in enumerate(enriched_chunks):
            embedding = embeddings[i] if embeddings else None
            doc = self.prepare_search_document(chunk, embedding)
            search_docs.append(doc)
        
        # Upload to search index
        result = self.search_client.upload_documents(
            documents=search_docs
        )
        
        return result

# Complete enrichment pipeline
def enrich_and_index_pdf(
    pdf_path: str,
    doc_id: str,
    form_recognizer_endpoint: str,
    form_recognizer_key: str,
    language_endpoint: str,
    language_key: str,
    openai_embedder,
    search_client
) -> dict:
    """
    Complete pipeline: Parse → Chunk → Enrich → Embed → Index.
    
    Args:
        pdf_path: Path to PDF file
        doc_id: Document identifier
        form_recognizer_endpoint: Azure AI Document Intelligence endpoint
        form_recognizer_key: Document Intelligence key
        language_endpoint: Azure AI Language endpoint
        language_key: Language service key
        openai_embedder: Embedder instance
        search_client: Search index client
        
    Returns:
        Pipeline execution summary
    """
    # Step 1: Parse PDF
    parser = PDFDocumentParser(
        endpoint=form_recognizer_endpoint,
        api_key=form_recognizer_key
    )
    parsed_doc = parser.parse_pdf(pdf_path)
    
    # Step 2: Chunk document
    chunker = StructureAwareChunker(
        max_chunk_tokens=512,
        overlap_tokens=50,
        preserve_tables=True,
        preserve_sections=True
    )
    chunks = chunker.chunk_document(parsed_doc, doc_id)
    
    # Step 3: Enrich chunks
    enricher = ChunkEnricher(
        language_endpoint=language_endpoint,
        language_key=language_key
    )
    enriched_chunks = enricher.enrich_batch(chunks)
    
    # Step 4: Generate embeddings
    embeddings = []
    for chunk in enriched_chunks:
        embedding = openai_embedder.embed_text(chunk["content"])
        embeddings.append(embedding)
    
    # Step 5: Index to search
    indexer = EnrichedChunkIndexer(search_client)
    result = indexer.index_enriched_chunks(enriched_chunks, embeddings)
    
    return {
        "document_id": doc_id,
        "chunks_created": len(chunks),
        "chunks_enriched": len(enriched_chunks),
        "chunks_indexed": len(result),
        "pipeline_status": "success"
    }
```

### Index Schema for Enriched Chunks

```python
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)

def create_enriched_chunk_index() -> SearchIndex:
    """Create search index for enriched document chunks."""
    
    fields = [
        SearchField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        SearchField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name="en.microsoft"
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="vector-profile"
        ),
        SearchField(
            name="chunk_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        SearchField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
            facetable=True
        ),
        SearchField(
            name="parent_section",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True
        ),
        SearchField(
            name="language",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        # Enrichment fields
        SearchField(
            name="key_phrases",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True
        ),
        SearchField(
            name="entity_names",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True
        ),
        SearchField(
            name="entity_categories",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
            facetable=True
        ),
        SearchField(
            name="entity_count",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True
        ),
        SearchField(
            name="has_entities",
            type=SearchFieldDataType.Boolean,
            filterable=True
        ),
        SearchField(
            name="has_key_phrases",
            type=SearchFieldDataType.Boolean,
            filterable=True
        )
    ]
    
    # Vector search configuration
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config"
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw-config")
        ]
    )
    
    index = SearchIndex(
        name="enriched-chunks",
        fields=fields,
        vector_search=vector_search
    )
    
    return index
```

### Querying Enriched Content

```python
class EnrichedSearchQuery:
    """Query enriched chunks with filters."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def search_by_entity(
        self,
        query: str,
        entity_category: str = None,
        top: int = 10
    ):
        """
        Search chunks containing specific entities.
        
        Args:
            query: Search query
            entity_category: Filter by entity category (Person, Organization, Location, etc.)
            top: Number of results
        """
        filter_expr = None
        if entity_category:
            filter_expr = f"entity_categories/any(c: c eq '{entity_category}')"
        
        results = self.search_client.search(
            search_text=query,
            filter=filter_expr,
            select=["content", "entity_names", "entity_categories", "page_number"],
            top=top
        )
        
        return list(results)
    
    def search_by_key_phrase(
        self,
        key_phrase: str,
        top: int = 10
    ):
        """Search chunks by key phrase."""
        filter_expr = f"key_phrases/any(k: k eq '{key_phrase}')"
        
        results = self.search_client.search(
            search_text="*",
            filter=filter_expr,
            select=["content", "key_phrases", "page_number"],
            top=top
        )
        
        return list(results)
    
    def search_with_enrichment_boost(
        self,
        query: str,
        boost_entities: bool = True,
        boost_key_phrases: bool = True
    ):
        """
        Search with enrichment-based scoring boost.
        
        Chunks with more entities/key phrases rank higher.
        """
        # Build scoring profile query
        scoring_params = []
        if boost_entities:
            scoring_params.append("entity_count-2.0")
        
        results = self.search_client.search(
            search_text=query,
            scoring_parameters=scoring_params,
            select=["content", "entity_count", "key_phrases", "page_number"],
            top=10
        )
        
        return list(results)
```

> **Note**: For comprehensive enrichment pipelines including skillsets, OCR, custom skills, 
> and knowledge store, see [Knowledge Mining (Page 26)](./26-knowledge-mining.md).

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

> **Note**: For PDF documents with structure (tables, headers), see the 
> [Document Parsing & Chunking](#document-parsing--chunking) section above for 
> production-ready structure-aware chunking using Azure AI Document Intelligence.
> 
> The basic chunking below is suitable for plain text only.

```python
class DocumentEmbedder:
    """Embed long documents with basic chunking strategy (plain text only)."""
    
    def __init__(self, embedder, chunk_size=1000, overlap=200):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text):
        """
        Split text into overlapping chunks (basic - use StructureAwareChunker for PDFs).
        
        Args:
            text: Plain text string
            
        Returns:
            List of chunk dictionaries
        """
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

### Document Processing & Chunking Best Practices

1. **Use Structure-Aware Chunking for PDFs**
   - Parse PDFs with Azure AI Document Intelligence
   - Respect document structure (headers, tables, paragraphs)
   - Never split tables across chunks
   - Keep section headers with their content
   
2. **Choose Appropriate Chunk Sizes**
   - **Small chunks (256 tokens)**: Precise retrieval, more chunks to search
   - **Medium chunks (512 tokens)**: Good balance for most use cases
   - **Large chunks (1024 tokens)**: More context, fewer chunks
   
3. **Implement Overlap**
   - Use 10-20% overlap between chunks
   - Prevents information loss at boundaries
   - Helps with queries that span chunk boundaries
   
4. **Preserve Context**
   - Include parent section titles in chunks
   - Add metadata: page number, chunk type, bounding box
   - Maintain document hierarchy
   
5. **Handle Tables Appropriately**
   - Convert tables to markdown format
   - Keep entire table in single chunk when possible
   - Add table prefix: `[TABLE 1]` for clarity
   
6. **Optimize for Search**
   - Index chunks individually with metadata
   - Store document_id to reconstruct original
   - Enable filtering by page_number, chunk_type
   - Include parent_section for context

### Embedding Best Practices

1. **Cache embeddings** for frequently accessed content
2. **Batch process** when possible to reduce latency
3. **Monitor costs** with token tracking
4. **Use checkpoints** for long-running jobs
5. **Implement retry logic** for transient failures
6. **Normalize embeddings** for consistent similarity calculations

### General Don'ts

1. **Don't** embed the same text multiple times
2. **Don't** exceed rate limits (3,000 RPM for ada-002)
3. **Don't** ignore token limits (8,191 for ada-002)
4. **Don't** skip error handling
5. **Don't** forget to secure API keys
6. **Don't** use character-based chunking for PDFs with structure

### Complete PDF-to-Search Pipeline

```
1. PDF Document
   ↓
2. Azure AI Document Intelligence (parse layout)
   ↓
3. Structure-Aware Chunking (preserve tables/headers)
   ↓
4. Azure OpenAI (generate embeddings per chunk)
   ↓
5. Azure AI Search (index chunks with metadata)
   ↓
6. Vector/Hybrid Search (retrieve relevant chunks)
   ↓
7. Re-rank and present results
```

---

## Next Steps

- **[Vector Search Implementation](./09-vector-search.md)** - Use embeddings for search
- **[Hybrid Search](./10-hybrid-search.md)** - Combine text and vector search
- **[Cost Analysis](./19-cost-analysis.md)** - Optimize Azure OpenAI costs

---

*See also: [Azure AI Search Setup](./04-azure-ai-search-setup.md) | [Best Practices](./30-best-practices-checklist.md)*