# Domain-Specific Solutions

Industry-specific search implementations with specialized schemas, relevance tuning, and compliance requirements for healthcare, legal, financial services, e-commerce, media & entertainment, and government sectors.

---

## 1. E-Commerce Search

### Product Catalog Schema

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    ComplexField
)
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ProductSearchSchema:
    """E-commerce product search index schema"""
    
    @staticmethod
    def create_index(index_name: str) -> SearchIndex:
        """
        Create product catalog index with e-commerce-specific fields
        
        Features:
        - SKU and variant management
        - Price and inventory tracking
        - Category hierarchies
        - Ratings and reviews
        - Image and media URLs
        """
        return SearchIndex(
            name=index_name,
            fields=[
                # Primary identifiers
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True
                ),
                SimpleField(
                    name="sku",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    sortable=True
                ),
                
                # Product information
                SearchableField(
                    name="name",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.microsoft",
                    sortable=True
                ),
                SearchableField(
                    name="description",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.microsoft"
                ),
                SearchableField(
                    name="brand",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                
                # Categorization
                SearchableField(
                    name="category",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SimpleField(
                    name="subcategory",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SearchField(
                    name="tags",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True,
                    filterable=True,
                    facetable=True
                ),
                
                # Pricing and inventory
                SimpleField(
                    name="price",
                    type=SearchFieldDataType.Double,
                    filterable=True,
                    sortable=True,
                    facetable=True
                ),
                SimpleField(
                    name="original_price",
                    type=SearchFieldDataType.Double,
                    filterable=True
                ),
                SimpleField(
                    name="discount_percentage",
                    type=SearchFieldDataType.Int32,
                    filterable=True,
                    facetable=True
                ),
                SimpleField(
                    name="in_stock",
                    type=SearchFieldDataType.Boolean,
                    filterable=True,
                    facetable=True
                ),
                SimpleField(
                    name="inventory_count",
                    type=SearchFieldDataType.Int32,
                    filterable=True
                ),
                
                # Ratings and reviews
                SimpleField(
                    name="rating",
                    type=SearchFieldDataType.Double,
                    filterable=True,
                    sortable=True,
                    facetable=True
                ),
                SimpleField(
                    name="review_count",
                    type=SearchFieldDataType.Int32,
                    filterable=True,
                    sortable=True
                ),
                
                # Media
                SearchField(
                    name="image_urls",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=False
                ),
                SimpleField(
                    name="primary_image_url",
                    type=SearchFieldDataType.String
                ),
                
                # Product attributes (variants)
                ComplexField(
                    name="attributes",
                    fields=[
                        SimpleField(name="size", type=SearchFieldDataType.String, filterable=True, facetable=True),
                        SimpleField(name="color", type=SearchFieldDataType.String, filterable=True, facetable=True),
                        SimpleField(name="material", type=SearchFieldDataType.String, filterable=True, facetable=True)
                    ],
                    collection=True
                ),
                
                # Metadata
                SimpleField(
                    name="created_date",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True,
                    sortable=True
                ),
                SimpleField(
                    name="last_updated",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True,
                    sortable=True
                ),
                SimpleField(
                    name="popularity_score",
                    type=SearchFieldDataType.Double,
                    sortable=True
                )
            ]
        )


class FacetedNavigationBuilder:
    """Build faceted navigation for e-commerce search"""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def build_facets(
        self,
        query: str,
        facet_fields: Optional[List[str]] = None
    ) -> Dict:
        """
        Build faceted navigation structure
        
        Args:
            query: Search query
            facet_fields: Fields to facet on
            
        Returns:
            Facet results with counts
        """
        if facet_fields is None:
            facet_fields = [
                "category",
                "brand",
                "price,values:0|25|50|100|200|500",  # Price ranges
                "rating,values:1|2|3|4|5",  # Rating buckets
                "in_stock",
                "discount_percentage,values:0|10|25|50"
            ]
        
        results = self.search_client.search(
            search_text=query,
            facets=facet_fields,
            top=0  # Only get facets, no results
        )
        
        facets = {}
        for facet_name, facet_results in results.get_facets().items():
            facets[facet_name] = [
                {
                    'value': result['value'],
                    'count': result['count']
                }
                for result in facet_results
            ]
        
        return facets
    
    def apply_filters(
        self,
        query: str,
        selected_facets: Dict[str, List[str]]
    ) -> List[Dict]:
        """
        Apply facet filters to search
        
        Args:
            query: Search query
            selected_facets: Selected facet values by field
            
        Returns:
            Filtered search results
        """
        filter_expressions = []
        
        for field, values in selected_facets.items():
            if len(values) == 1:
                filter_expressions.append(f"{field} eq '{values[0]}'")
            else:
                # Multiple values - use 'or' logic
                or_clauses = [f"{field} eq '{v}'" for v in values]
                filter_expressions.append(f"({' or '.join(or_clauses)})")
        
        filter_str = " and ".join(filter_expressions) if filter_expressions else None
        
        results = self.search_client.search(
            search_text=query,
            filter=filter_str,
            select=["id", "name", "price", "brand", "rating", "primary_image_url"],
            top=50
        )
        
        return list(results)


class PersonalizationEngine:
    """Personalized product recommendations"""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def get_personalized_results(
        self,
        query: str,
        user_id: str,
        user_preferences: Dict,
        browsing_history: List[str]
    ) -> List[Dict]:
        """
        Get personalized search results
        
        Args:
            query: Search query
            user_id: User identifier
            user_preferences: User preference data (categories, brands, price range)
            browsing_history: Recently viewed product IDs
            
        Returns:
            Personalized search results
        """
        # Build scoring profile boost based on preferences
        scoring_params = []
        
        if 'preferred_brands' in user_preferences:
            brands = ','.join(user_preferences['preferred_brands'])
            scoring_params.append(f"preferredBrands-{brands}")
        
        if 'price_range' in user_preferences:
            max_price = user_preferences['price_range'].get('max', 1000)
            scoring_params.append(f"maxPrice-{max_price}")
        
        # Build filter for user preferences
        filters = []
        if 'price_range' in user_preferences:
            min_price = user_preferences['price_range'].get('min', 0)
            max_price = user_preferences['price_range'].get('max', 10000)
            filters.append(f"price ge {min_price} and price le {max_price}")
        
        filter_str = " and ".join(filters) if filters else None
        
        results = self.search_client.search(
            search_text=query,
            filter=filter_str,
            scoring_parameters=scoring_params,
            select=["id", "name", "price", "brand", "rating", "primary_image_url"],
            top=50
        )
        
        return list(results)
    
    def get_similar_products(
        self,
        product_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Find similar products based on attributes
        
        Args:
            product_id: Reference product ID
            limit: Maximum results
            
        Returns:
            Similar products
        """
        # Get reference product
        reference = self.search_client.get_document(key=product_id)
        
        # Build query from product attributes
        search_terms = [
            reference.get('brand', ''),
            reference.get('category', ''),
            reference.get('subcategory', '')
        ]
        query = ' '.join(filter(None, search_terms))
        
        # Search excluding the reference product
        results = self.search_client.search(
            search_text=query,
            filter=f"id ne '{product_id}'",
            select=["id", "name", "price", "brand", "rating", "primary_image_url"],
            top=limit
        )
        
        return list(results)
```

---

## 2. Healthcare Search

### Medical Records Indexing

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import hashlib
import re
from typing import Dict, List, Optional

class MedicalRecordIndexer:
    """
    HIPAA-compliant medical record indexing
    
    Features:
    - PHI anonymization
    - Field-level encryption
    - Audit logging
    - Access controls
    """
    
    def __init__(
        self,
        search_client,
        key_vault_url: str
    ):
        self.search_client = search_client
        self.secret_client = SecretClient(
            vault_url=key_vault_url,
            credential=DefaultAzureCredential()
        )
    
    @staticmethod
    def create_medical_index() -> SearchIndex:
        """Create HIPAA-compliant medical records index"""
        return SearchIndex(
            name="medical-records",
            fields=[
                # De-identified patient reference
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True
                ),
                SimpleField(
                    name="patient_hash",  # Hashed patient ID
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                
                # Clinical information
                SearchableField(
                    name="diagnosis",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.microsoft"
                ),
                SearchableField(
                    name="symptoms",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.microsoft"
                ),
                SearchableField(
                    name="treatment",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.microsoft"
                ),
                
                # Medical codes
                SearchField(
                    name="icd10_codes",  # Diagnosis codes
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True,
                    facetable=True
                ),
                SearchField(
                    name="cpt_codes",  # Procedure codes
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True,
                    facetable=True
                ),
                SearchField(
                    name="snomed_codes",  # Clinical terminology
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True
                ),
                
                # Clinical metadata
                SimpleField(
                    name="department",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SimpleField(
                    name="facility",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SimpleField(
                    name="encounter_date",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True,
                    sortable=True
                ),
                SimpleField(
                    name="encounter_type",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                
                # Encrypted PHI references (stored in Key Vault)
                SimpleField(
                    name="encrypted_patient_name_ref",
                    type=SearchFieldDataType.String
                ),
                SimpleField(
                    name="encrypted_mrn_ref",  # Medical Record Number
                    type=SearchFieldDataType.String
                ),
                
                # Audit fields
                SimpleField(
                    name="created_by",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                SimpleField(
                    name="last_accessed",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True
                )
            ]
        )
    
    def anonymize_phi(self, text: str) -> str:
        """
        Remove or mask Protected Health Information (PHI)
        
        Args:
            text: Clinical text
            
        Returns:
            Anonymized text
        """
        # Remove common PHI patterns
        patterns = {
            # Social Security Numbers
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',
            # Phone numbers
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b': '[PHONE]',
            # Email addresses
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
            # Dates (MM/DD/YYYY or MM-DD-YYYY)
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b': '[DATE]',
            # Medical Record Numbers (MRN-XXXXXX)
            r'\bMRN-\d+\b': '[MRN]',
            # ZIP codes (specific - keep first 3 digits only)
            r'\b(\d{3})\d{2}\b': r'\1**'
        }
        
        anonymized = text
        for pattern, replacement in patterns.items():
            anonymized = re.sub(pattern, replacement, anonymized)
        
        return anonymized
    
    def hash_patient_id(self, patient_id: str) -> str:
        """
        Create consistent hash of patient ID for de-identification
        
        Args:
            patient_id: Original patient identifier
            
        Returns:
            SHA-256 hash
        """
        salt = self.secret_client.get_secret("patient-id-salt").value
        return hashlib.sha256(f"{patient_id}{salt}".encode()).hexdigest()
    
    def index_medical_record(
        self,
        record: Dict,
        user_id: str
    ) -> Dict:
        """
        Index medical record with HIPAA compliance
        
        Args:
            record: Medical record data
            user_id: User performing indexing (for audit)
            
        Returns:
            Indexed document
        """
        # Anonymize patient identifiers
        patient_hash = self.hash_patient_id(record['patient_id'])
        
        # Anonymize free-text fields
        diagnosis = self.anonymize_phi(record.get('diagnosis', ''))
        symptoms = self.anonymize_phi(record.get('symptoms', ''))
        treatment = self.anonymize_phi(record.get('treatment', ''))
        
        # Create indexed document
        document = {
            'id': record['record_id'],
            'patient_hash': patient_hash,
            'diagnosis': diagnosis,
            'symptoms': symptoms,
            'treatment': treatment,
            'icd10_codes': record.get('icd10_codes', []),
            'cpt_codes': record.get('cpt_codes', []),
            'snomed_codes': record.get('snomed_codes', []),
            'department': record.get('department'),
            'facility': record.get('facility'),
            'encounter_date': record.get('encounter_date'),
            'encounter_type': record.get('encounter_type'),
            'created_by': user_id,
            'last_accessed': datetime.utcnow().isoformat()
        }
        
        # Upload to index
        result = self.search_client.upload_documents(documents=[document])
        
        # Audit log (would integrate with Azure Monitor)
        self._log_access(
            action='INDEX',
            record_id=record['record_id'],
            user_id=user_id
        )
        
        return document
    
    def _log_access(self, action: str, record_id: str, user_id: str):
        """Log HIPAA audit trail"""
        # In production, send to Azure Monitor / Log Analytics
        print(f"AUDIT: {action} - Record: {record_id} - User: {user_id} - Time: {datetime.utcnow()}")


class ClinicalTerminologyMapper:
    """Map clinical terms to standard codes"""
    
    def __init__(self):
        # In production, load from terminology services
        self.icd10_mapping = {
            'diabetes': ['E11.9', 'E10.9'],
            'hypertension': ['I10'],
            'pneumonia': ['J18.9']
        }
        
        self.snomed_mapping = {
            'diabetes': ['73211009'],
            'hypertension': ['38341003'],
            'pneumonia': ['233604007']
        }
    
    def extract_codes(self, clinical_text: str) -> Dict[str, List[str]]:
        """
        Extract medical codes from clinical text
        
        Args:
            clinical_text: Clinical notes
            
        Returns:
            Extracted ICD-10 and SNOMED codes
        """
        text_lower = clinical_text.lower()
        
        icd10_codes = []
        snomed_codes = []
        
        for term, codes in self.icd10_mapping.items():
            if term in text_lower:
                icd10_codes.extend(codes)
        
        for term, codes in self.snomed_mapping.items():
            if term in text_lower:
                snomed_codes.extend(codes)
        
        return {
            'icd10_codes': list(set(icd10_codes)),
            'snomed_codes': list(set(snomed_codes))
        }
```

---

## 3. Legal Search

### Case Law and Document Discovery

```python
class LegalDocumentProcessor:
    """Legal document indexing with citation extraction"""
    
    @staticmethod
    def create_legal_index() -> SearchIndex:
        """Create legal document search index"""
        return SearchIndex(
            name="legal-documents",
            fields=[
                # Document identifiers
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True
                ),
                SimpleField(
                    name="case_number",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    sortable=True
                ),
                
                # Case information
                SearchableField(
                    name="case_title",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.lucene"
                ),
                SearchableField(
                    name="court",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SimpleField(
                    name="decision_date",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True,
                    sortable=True
                ),
                SearchableField(
                    name="jurisdiction",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                
                # Document content
                SearchableField(
                    name="full_text",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.lucene"
                ),
                SearchableField(
                    name="summary",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.lucene"
                ),
                
                # Legal metadata
                SearchField(
                    name="legal_topics",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True,
                    facetable=True
                ),
                SearchField(
                    name="statutes_cited",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True,
                    facetable=True
                ),
                SearchField(
                    name="parties",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True,
                    filterable=True
                ),
                SearchField(
                    name="judges",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True,
                    facetable=True
                ),
                
                # Citations
                SearchField(
                    name="case_citations",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True
                ),
                SimpleField(
                    name="citation_count",  # How many times cited
                    type=SearchFieldDataType.Int32,
                    sortable=True
                ),
                
                # Document classification
                SimpleField(
                    name="document_type",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SimpleField(
                    name="precedent_value",  # High/Medium/Low
                    type=SearchFieldDataType.String,
                    filterable=True,
                    sortable=True
                )
            ]
        )
    
    def extract_citations(self, legal_text: str) -> List[str]:
        """
        Extract legal citations from text
        
        Patterns:
        - Federal: 123 U.S. 456
        - State: 123 Cal.App.4th 456
        - Federal Reporter: 123 F.3d 456
        
        Args:
            legal_text: Legal document text
            
        Returns:
            List of extracted citations
        """
        citation_patterns = [
            # U.S. Supreme Court
            r'\b\d+\s+U\.S\.\s+\d+\b',
            # Federal Reporter
            r'\b\d+\s+F\.\d?d?\s+\d+\b',
            # State reporters
            r'\b\d+\s+[A-Z][a-z]+\.\s*(?:App\.)?\s*\d?[a-z]{2}\s+\d+\b'
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, legal_text)
            citations.extend(matches)
        
        return list(set(citations))
    
    def extract_statute_citations(self, legal_text: str) -> List[str]:
        """
        Extract statute citations
        
        Patterns:
        - USC: 42 U.S.C. § 1983
        - State: Cal. Civ. Code § 1234
        
        Args:
            legal_text: Legal document text
            
        Returns:
            List of statute citations
        """
        statute_patterns = [
            # U.S. Code
            r'\b\d+\s+U\.S\.C\.\s+§\s+\d+[a-z]?\b',
            # State codes
            r'\b[A-Z][a-z]+\.\s+[A-Z][a-z]+\.\s+Code\s+§\s+\d+(?:\.\d+)?\b'
        ]
        
        statutes = []
        for pattern in statute_patterns:
            matches = re.findall(pattern, legal_text)
            statutes.extend(matches)
        
        return list(set(statutes))


class PrecedentLinker:
    """Build citation graph for precedent analysis"""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def build_citation_graph(self, case_id: str) -> Dict:
        """
        Build citation network for a case
        
        Args:
            case_id: Case document ID
            
        Returns:
            Citation graph with citing/cited cases
        """
        # Get case document
        case = self.search_client.get_document(key=case_id)
        
        # Find cases this one cites
        cited_cases = []
        for citation in case.get('case_citations', []):
            results = self.search_client.search(
                search_text=f'"{citation}"',
                select=["id", "case_title", "decision_date"],
                top=1
            )
            cited_cases.extend(list(results))
        
        # Find cases that cite this one
        citing_cases = self.search_client.search(
            search_text=case.get('case_number', ''),
            filter=f"id ne '{case_id}'",
            select=["id", "case_title", "decision_date"],
            top=50
        )
        
        return {
            'case_id': case_id,
            'case_title': case.get('case_title'),
            'cited_cases': cited_cases,
            'citing_cases': list(citing_cases),
            'citation_count': len(list(citing_cases))
        }
    
    def calculate_precedent_value(
        self,
        case_id: str
    ) -> str:
        """
        Calculate precedent value based on citation count and recency
        
        Args:
            case_id: Case document ID
            
        Returns:
            Precedent value: High/Medium/Low
        """
        case = self.search_client.get_document(key=case_id)
        citation_count = case.get('citation_count', 0)
        
        # Age factor
        decision_date = case.get('decision_date')
        if decision_date:
            age_years = (datetime.utcnow() - datetime.fromisoformat(decision_date)).days / 365
        else:
            age_years = 100
        
        # Scoring
        if citation_count > 50 and age_years < 10:
            return 'High'
        elif citation_count > 20 or age_years < 5:
            return 'Medium'
        else:
            return 'Low'
```

---

## 4. Financial Services

### Regulatory Compliance Search

```python
class FinancialDocumentIndexer:
    """Financial document indexing with regulatory compliance"""
    
    @staticmethod
    def create_financial_index() -> SearchIndex:
        """Create financial documents index"""
        return SearchIndex(
            name="financial-documents",
            fields=[
                # Document identifiers
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True
                ),
                SimpleField(
                    name="document_number",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                
                # Document classification
                SimpleField(
                    name="document_type",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),  # 10-K, 10-Q, 8-K, Prospectus, etc.
                SimpleField(
                    name="filing_type",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SimpleField(
                    name="filing_date",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True,
                    sortable=True
                ),
                
                # Entity information
                SearchableField(
                    name="company_name",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SimpleField(
                    name="ticker_symbol",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                SimpleField(
                    name="cik_number",  # SEC Central Index Key
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                SimpleField(
                    name="industry_sector",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                
                # Document content
                SearchableField(
                    name="content",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.lucene"
                ),
                SearchableField(
                    name="executive_summary",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.lucene"
                ),
                
                # Financial metrics
                SimpleField(
                    name="fiscal_year",
                    type=SearchFieldDataType.Int32,
                    filterable=True,
                    facetable=True
                ),
                SimpleField(
                    name="fiscal_quarter",
                    type=SearchFieldDataType.Int32,
                    filterable=True
                ),
                SimpleField(
                    name="revenue",
                    type=SearchFieldDataType.Double,
                    filterable=True,
                    sortable=True
                ),
                SimpleField(
                    name="net_income",
                    type=SearchFieldDataType.Double,
                    filterable=True,
                    sortable=True
                ),
                
                # Regulatory tags
                SearchField(
                    name="regulatory_frameworks",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True,
                    facetable=True
                ),  # SOX, FINRA, SEC, GDPR, etc.
                SearchField(
                    name="risk_factors",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True,
                    filterable=True
                ),
                
                # Redaction status
                SimpleField(
                    name="contains_redacted_info",
                    type=SearchFieldDataType.Boolean,
                    filterable=True
                ),
                SimpleField(
                    name="classification_level",
                    type=SearchFieldDataType.String,
                    filterable=True
                )  # Public, Internal, Confidential, Restricted
            ]
        )
    
    def redact_sensitive_data(self, text: str) -> str:
        """
        Redact sensitive financial information
        
        Args:
            text: Document text
            
        Returns:
            Redacted text
        """
        patterns = {
            # Account numbers
            r'\b\d{8,17}\b': '[ACCOUNT]',
            # Credit card numbers
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b': '[CC]',
            # Routing numbers
            r'\b\d{9}\b': '[ROUTING]',
            # SSN (already covered in medical, but included for completeness)
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]'
        }
        
        redacted = text
        for pattern, replacement in patterns.items():
            redacted = re.sub(pattern, replacement, redacted)
        
        return redacted


class DocumentClassifier:
    """Classify financial documents by type"""
    
    def __init__(self):
        # SEC filing type signatures
        self.filing_signatures = {
            '10-K': ['annual report', 'fiscal year ended', 'form 10-k'],
            '10-Q': ['quarterly report', 'quarter ended', 'form 10-q'],
            '8-K': ['current report', 'form 8-k', 'material events'],
            'S-1': ['registration statement', 'securities act', 'form s-1'],
            'DEF 14A': ['proxy statement', 'annual meeting', 'def 14a']
        }
    
    def classify_document(self, document_text: str) -> str:
        """
        Classify financial document type
        
        Args:
            document_text: Document content
            
        Returns:
            Document type classification
        """
        text_lower = document_text.lower()
        
        for doc_type, signatures in self.filing_signatures.items():
            if any(sig in text_lower for sig in signatures):
                return doc_type
        
        return 'Other'
    
    def extract_financial_metrics(self, document_text: str) -> Dict:
        """
        Extract key financial metrics from document
        
        Args:
            document_text: Document content
            
        Returns:
            Extracted metrics
        """
        metrics = {}
        
        # Revenue pattern: "Revenue: $X million" or "Total revenue of $X billion"
        revenue_pattern = r'(?:revenue|sales).*?\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion)'
        revenue_match = re.search(revenue_pattern, document_text, re.IGNORECASE)
        
        if revenue_match:
            amount = float(revenue_match.group(1).replace(',', ''))
            unit = revenue_match.group(2).lower()
            multiplier = 1_000_000 if unit == 'million' else 1_000_000_000
            metrics['revenue'] = amount * multiplier
        
        # Net income pattern
        income_pattern = r'(?:net income|earnings).*?\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion)'
        income_match = re.search(income_pattern, document_text, re.IGNORECASE)
        
        if income_match:
            amount = float(income_match.group(1).replace(',', ''))
            unit = income_match.group(2).lower()
            multiplier = 1_000_000 if unit == 'million' else 1_000_000_000
            metrics['net_income'] = amount * multiplier
        
        return metrics
```

---

## 5. Media & Entertainment

### Content Recommendation

```python
class ContentRecommendationEngine:
    """Media content search and recommendation"""
    
    @staticmethod
    def create_media_index() -> SearchIndex:
        """Create media content index"""
        return SearchIndex(
            name="media-content",
            fields=[
                # Content identifiers
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True
                ),
                SimpleField(
                    name="content_id",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                
                # Content information
                SearchableField(
                    name="title",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.microsoft"
                ),
                SearchableField(
                    name="description",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.microsoft"
                ),
                SearchField(
                    name="genres",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True,
                    facetable=True
                ),
                
                # Content type
                SimpleField(
                    name="content_type",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),  # Movie, TV Show, Documentary, etc.
                SimpleField(
                    name="duration_minutes",
                    type=SearchFieldDataType.Int32,
                    filterable=True,
                    sortable=True
                ),
                
                # Credits
                SearchField(
                    name="actors",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True,
                    filterable=True,
                    facetable=True
                ),
                SearchField(
                    name="directors",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True,
                    filterable=True,
                    facetable=True
                ),
                
                # Release information
                SimpleField(
                    name="release_date",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True,
                    sortable=True
                ),
                SimpleField(
                    name="release_year",
                    type=SearchFieldDataType.Int32,
                    filterable=True,
                    facetable=True
                ),
                
                # Ratings
                SimpleField(
                    name="content_rating",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),  # G, PG, PG-13, R, etc.
                SimpleField(
                    name="user_rating",
                    type=SearchFieldDataType.Double,
                    filterable=True,
                    sortable=True
                ),
                SimpleField(
                    name="critic_rating",
                    type=SearchFieldDataType.Double,
                    filterable=True,
                    sortable=True
                ),
                
                # Engagement metrics
                SimpleField(
                    name="view_count",
                    type=SearchFieldDataType.Int64,
                    sortable=True
                ),
                SimpleField(
                    name="popularity_score",
                    type=SearchFieldDataType.Double,
                    sortable=True
                ),
                
                # Metadata
                SearchField(
                    name="keywords",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True,
                    filterable=True
                ),
                SimpleField(
                    name="language",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SearchField(
                    name="subtitles_available",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True
                )
            ]
        )
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def get_recommendations(
        self,
        user_id: str,
        viewing_history: List[str],
        preferences: Dict,
        limit: int = 20
    ) -> List[Dict]:
        """
        Get personalized content recommendations
        
        Args:
            user_id: User identifier
            viewing_history: Recently watched content IDs
            preferences: User preferences (genres, actors, etc.)
            limit: Maximum recommendations
            
        Returns:
            Recommended content
        """
        # Build query from user preferences
        search_terms = []
        
        if 'favorite_genres' in preferences:
            search_terms.extend(preferences['favorite_genres'])
        
        if 'favorite_actors' in preferences:
            search_terms.extend(preferences['favorite_actors'])
        
        query = ' '.join(search_terms) if search_terms else '*'
        
        # Build filters
        filters = []
        
        # Exclude already watched
        if viewing_history:
            history_filter = " and ".join([f"id ne '{cid}'" for cid in viewing_history[:50]])
            filters.append(f"({history_filter})")
        
        # Filter by content rating if specified
        if 'max_rating' in preferences:
            filters.append(f"content_rating eq '{preferences['max_rating']}'")
        
        filter_str = " and ".join(filters) if filters else None
        
        # Search with scoring boost
        results = self.search_client.search(
            search_text=query,
            filter=filter_str,
            order_by=["popularity_score desc", "user_rating desc"],
            select=["id", "title", "description", "genres", "user_rating", "content_type"],
            top=limit
        )
        
        return list(results)
    
    def find_similar_content(
        self,
        content_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Find content similar to given item
        
        Args:
            content_id: Reference content ID
            limit: Maximum results
            
        Returns:
            Similar content
        """
        # Get reference content
        reference = self.search_client.get_document(key=content_id)
        
        # Build query from attributes
        search_terms = []
        search_terms.extend(reference.get('genres', []))
        search_terms.extend(reference.get('actors', [])[:3])  # Top 3 actors
        search_terms.extend(reference.get('directors', []))
        
        query = ' '.join(search_terms)
        
        # Search excluding reference
        results = self.search_client.search(
            search_text=query,
            filter=f"id ne '{content_id}'",
            select=["id", "title", "description", "genres", "user_rating"],
            top=limit
        )
        
        return list(results)
```

---

## 6. Government Search

### Public Records Access

```python
class PublicRecordsIndexer:
    """Government document indexing with accessibility compliance"""
    
    @staticmethod
    def create_public_records_index() -> SearchIndex:
        """Create public records index"""
        return SearchIndex(
            name="public-records",
            fields=[
                # Document identifiers
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True
                ),
                SimpleField(
                    name="record_number",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                
                # Record classification
                SearchableField(
                    name="title",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.microsoft"
                ),
                SearchableField(
                    name="description",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.microsoft"
                ),
                SimpleField(
                    name="record_type",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                
                # Agency information
                SimpleField(
                    name="agency",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SimpleField(
                    name="department",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                
                # Dates
                SimpleField(
                    name="created_date",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True,
                    sortable=True
                ),
                SimpleField(
                    name="published_date",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True,
                    sortable=True
                ),
                
                # Content
                SearchableField(
                    name="content",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.microsoft"
                ),
                
                # Access control
                SimpleField(
                    name="access_level",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),  # Public, Restricted, Confidential
                SimpleField(
                    name="foia_exempt",
                    type=SearchFieldDataType.Boolean,
                    filterable=True
                ),
                SearchField(
                    name="exemption_codes",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True
                ),
                
                # Accessibility
                SimpleField(
                    name="wcag_compliant",
                    type=SearchFieldDataType.Boolean,
                    filterable=True
                ),
                SimpleField(
                    name="has_alt_text",
                    type=SearchFieldDataType.Boolean,
                    filterable=True
                ),
                SearchField(
                    name="languages_available",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True,
                    facetable=True
                )
            ]
        )


class FOIARequestHandler:
    """Freedom of Information Act request processing"""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def process_foia_request(
        self,
        query: str,
        date_range: Optional[Dict] = None
    ) -> Dict:
        """
        Process FOIA request and identify responsive documents
        
        Args:
            query: FOIA request query
            date_range: Optional date range filter
            
        Returns:
            Responsive documents and exemption analysis
        """
        # Build filter
        filters = ["access_level eq 'Public' or foia_exempt eq false"]
        
        if date_range:
            if 'start' in date_range:
                filters.append(f"created_date ge {date_range['start']}")
            if 'end' in date_range:
                filters.append(f"created_date le {date_range['end']}")
        
        filter_str = " and ".join(filters)
        
        # Search for responsive documents
        results = self.search_client.search(
            search_text=query,
            filter=filter_str,
            select=["id", "record_number", "title", "created_date", "access_level"],
            top=1000
        )
        
        responsive_docs = list(results)
        
        # Check for potentially exempt documents
        exempt_results = self.search_client.search(
            search_text=query,
            filter="foia_exempt eq true",
            select=["id", "record_number", "title", "exemption_codes"],
            top=100
        )
        
        exempt_docs = list(exempt_results)
        
        return {
            'responsive_documents': responsive_docs,
            'responsive_count': len(responsive_docs),
            'exempt_documents': exempt_docs,
            'exempt_count': len(exempt_docs),
            'total_documents': len(responsive_docs) + len(exempt_docs)
        }


class AccessibilityValidator:
    """Validate WCAG 2.1 compliance for documents"""
    
    def validate_document(self, document: Dict) -> Dict:
        """
        Validate document accessibility
        
        Args:
            document: Document metadata
            
        Returns:
            Validation results
        """
        issues = []
        
        # Check for alternative text on images
        if not document.get('has_alt_text', False):
            issues.append({
                'level': 'A',
                'criterion': '1.1.1 Non-text Content',
                'issue': 'Missing alternative text for images'
            })
        
        # Check for multiple language support
        languages = document.get('languages_available', [])
        if len(languages) < 1:
            issues.append({
                'level': 'AA',
                'criterion': '3.1.2 Language of Parts',
                'issue': 'No language metadata available'
            })
        
        # Check WCAG compliance flag
        wcag_compliant = document.get('wcag_compliant', False)
        
        return {
            'document_id': document.get('id'),
            'wcag_compliant': wcag_compliant,
            'issues': issues,
            'compliance_level': 'AAA' if wcag_compliant and len(issues) == 0 else 'Partial'
        }
```

---

## 7. Cross-Domain Best Practices

### Domain-Agnostic Optimization

```python
class DomainSearchOptimizer:
    """Cross-domain search optimization techniques"""
    
    def __init__(self, search_client, domain: str):
        self.search_client = search_client
        self.domain = domain
    
    def optimize_for_domain(self) -> Dict:
        """
        Apply domain-specific optimizations
        
        Returns:
            Optimization recommendations
        """
        optimizations = {
            'ecommerce': {
                'scoring_profile': 'price_and_rating',
                'facets': ['category', 'brand', 'price', 'rating'],
                'boosted_fields': ['name^3', 'brand^2', 'description'],
                'suggest_mode': 'twoPhase'
            },
            'healthcare': {
                'scoring_profile': 'relevance_only',
                'facets': ['department', 'encounter_type'],
                'boosted_fields': ['diagnosis^3', 'symptoms^2'],
                'security': 'high',
                'audit_logging': True
            },
            'legal': {
                'scoring_profile': 'citation_weighted',
                'facets': ['court', 'jurisdiction', 'legal_topics'],
                'boosted_fields': ['case_title^2', 'summary^1.5'],
                'citation_analysis': True
            },
            'financial': {
                'scoring_profile': 'recency_weighted',
                'facets': ['document_type', 'fiscal_year', 'industry_sector'],
                'boosted_fields': ['company_name^2', 'content'],
                'compliance_required': True
            },
            'media': {
                'scoring_profile': 'popularity_weighted',
                'facets': ['genres', 'release_year', 'content_rating'],
                'boosted_fields': ['title^3', 'actors^2', 'description'],
                'personalization': True
            },
            'government': {
                'scoring_profile': 'date_weighted',
                'facets': ['agency', 'record_type', 'access_level'],
                'boosted_fields': ['title^2', 'description'],
                'accessibility_required': True
            }
        }
        
        return optimizations.get(self.domain, {
            'scoring_profile': 'default',
            'facets': [],
            'boosted_fields': [],
            'notes': 'Generic configuration'
        })
    
    def create_domain_scoring_profile(self) -> Dict:
        """
        Create scoring profile optimized for domain
        
        Returns:
            Scoring profile configuration
        """
        domain_profiles = {
            'ecommerce': {
                'name': 'ecommerce-relevance',
                'text_weights': {
                    'name': 3.0,
                    'brand': 2.0,
                    'description': 1.0
                },
                'functions': [
                    {
                        'type': 'freshness',
                        'fieldName': 'created_date',
                        'boost': 2.0,
                        'interpolation': 'linear'
                    },
                    {
                        'type': 'magnitude',
                        'fieldName': 'rating',
                        'boost': 1.5,
                        'interpolation': 'linear'
                    }
                ]
            },
            'legal': {
                'name': 'legal-relevance',
                'text_weights': {
                    'case_title': 2.0,
                    'summary': 1.5,
                    'full_text': 1.0
                },
                'functions': [
                    {
                        'type': 'magnitude',
                        'fieldName': 'citation_count',
                        'boost': 3.0,
                        'interpolation': 'logarithmic'
                    },
                    {
                        'type': 'freshness',
                        'fieldName': 'decision_date',
                        'boost': 1.5,
                        'interpolation': 'linear'
                    }
                ]
            }
        }
        
        return domain_profiles.get(self.domain, {})
```

---

## Best Practices

### Domain-Specific Recommendations

1. **E-Commerce**
   - Use faceted navigation for product discovery
   - Implement personalization based on browsing history
   - Optimize for mobile search
   - Include rich media (images, videos)
   - Track conversion metrics

2. **Healthcare**
   - Prioritize HIPAA compliance
   - Implement field-level encryption for PHI
   - Use clinical terminology standards (ICD-10, SNOMED)
   - Maintain comprehensive audit logs
   - Restrict access based on roles

3. **Legal**
   - Build citation networks for precedent analysis
   - Extract and index legal codes
   - Implement advanced Boolean search
   - Support proximity search for legal phrases
   - Calculate precedent value

4. **Financial Services**
   - Ensure regulatory compliance (SEC, FINRA)
   - Redact sensitive financial data
   - Classify documents by type
   - Extract financial metrics
   - Implement version control

5. **Media & Entertainment**
   - Focus on content discovery
   - Implement collaborative filtering
   - Support multilingual content
   - Optimize for engagement metrics
   - Enable content recommendations

6. **Government**
   - Ensure WCAG 2.1 accessibility
   - Support FOIA requests
   - Implement access controls
   - Provide multilingual support
   - Maintain transparency

### Common Patterns

- **Field Boosting**: Prioritize key fields for relevance
- **Faceted Navigation**: Enable filtering and refinement
- **Compliance**: Implement domain-specific regulations
- **Personalization**: Tailor results to user preferences
- **Security**: Apply appropriate access controls
- **Audit Logging**: Track all search activities

---

*Last Updated: November 5, 2025*
