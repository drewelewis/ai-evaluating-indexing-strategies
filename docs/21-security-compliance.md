# Security & Compliance

## Overview

This guide covers security and compliance best practices for Azure AI Search, including authentication, authorization, encryption, network security, and regulatory compliance frameworks.

## Table of Contents

1. [Authentication Methods](#authentication-methods)
2. [Authorization & Access Control](#authorization--access-control)
3. [Encryption](#encryption)
4. [Network Security](#network-security)
5. [Compliance Frameworks](#compliance-frameworks)
6. [Security Monitoring](#security-monitoring)
7. [Data Governance](#data-governance)
8. [Security Best Practices](#security-best-practices)

## Authentication Methods

### API Key Authentication

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
import secrets
import hashlib
import json

@dataclass
class ApiKey:
    """API key with metadata."""
    key_id: str
    key_hash: str
    key_type: str  # 'admin' or 'query'
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    description: str
    
class ApiKeyManager:
    """Manage API keys for Azure AI Search."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.keys: List[ApiKey] = []
        
    def generate_api_key(
        self,
        key_type: str = 'query',
        description: str = '',
        expires_in_days: Optional[int] = None
    ) -> str:
        """
        Generate a new API key.
        
        Args:
            key_type: 'admin' or 'query'
            description: Purpose of the key
            expires_in_days: Days until expiration (None for no expiration)
            
        Returns:
            The generated API key
        """
        # Generate secure random key
        key = secrets.token_urlsafe(32)
        key_id = secrets.token_hex(8)
        
        # Hash the key for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Store key metadata
        api_key = ApiKey(
            key_id=key_id,
            key_hash=key_hash,
            key_type=key_type,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            last_used=None,
            description=description
        )
        
        self.keys.append(api_key)
        
        return key
    
    def rotate_api_key(self, key_id: str) -> str:
        """
        Rotate an API key (create new, mark old for deletion).
        
        Args:
            key_id: ID of key to rotate
            
        Returns:
            New API key
        """
        # Find existing key
        old_key = next((k for k in self.keys if k.key_id == key_id), None)
        if not old_key:
            raise ValueError(f"Key {key_id} not found")
        
        # Generate new key with same properties
        new_key = self.generate_api_key(
            key_type=old_key.key_type,
            description=f"Rotated from {key_id}",
            expires_in_days=90 if old_key.expires_at else None
        )
        
        # Mark old key for deletion (grace period)
        old_key.expires_at = datetime.utcnow() + timedelta(days=7)
        
        return new_key
    
    def validate_api_key(self, key: str, key_type: str = None) -> bool:
        """
        Validate an API key.
        
        Args:
            key: API key to validate
            key_type: Required key type ('admin' or 'query')
            
        Returns:
            True if valid, False otherwise
        """
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        for api_key in self.keys:
            if api_key.key_hash == key_hash:
                # Check expiration
                if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                    return False
                
                # Check key type if specified
                if key_type and api_key.key_type != key_type:
                    return False
                
                # Update last used
                api_key.last_used = datetime.utcnow()
                return True
        
        return False
    
    def cleanup_expired_keys(self) -> int:
        """
        Remove expired API keys.
        
        Returns:
            Number of keys removed
        """
        now = datetime.utcnow()
        initial_count = len(self.keys)
        
        self.keys = [
            k for k in self.keys
            if not k.expires_at or k.expires_at > now
        ]
        
        return initial_count - len(self.keys)
```

### Azure AD Authentication

```python
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

class AzureADAuthManager:
    """Manage Azure AD authentication for Azure AI Search."""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        
    def get_credential_default(self):
        """
        Get default Azure credential.
        
        Tries in order:
        1. Environment variables
        2. Managed identity
        3. Visual Studio Code
        4. Azure CLI
        5. Azure PowerShell
        """
        return DefaultAzureCredential()
    
    def get_credential_managed_identity(
        self,
        client_id: Optional[str] = None
    ):
        """
        Get managed identity credential.
        
        Args:
            client_id: Client ID for user-assigned managed identity
        """
        if client_id:
            return ManagedIdentityCredential(client_id=client_id)
        return ManagedIdentityCredential()
    
    def create_search_client(
        self,
        index_name: str,
        use_managed_identity: bool = True
    ) -> SearchClient:
        """
        Create search client with Azure AD auth.
        
        Args:
            index_name: Name of the index
            use_managed_identity: Use managed identity if True
        """
        if use_managed_identity:
            credential = self.get_credential_managed_identity()
        else:
            credential = self.get_credential_default()
        
        return SearchClient(
            endpoint=self.endpoint,
            index_name=index_name,
            credential=credential
        )
    
    def create_index_client(
        self,
        use_managed_identity: bool = True
    ) -> SearchIndexClient:
        """
        Create index client with Azure AD auth.
        
        Args:
            use_managed_identity: Use managed identity if True
        """
        if use_managed_identity:
            credential = self.get_credential_managed_identity()
        else:
            credential = self.get_credential_default()
        
        return SearchIndexClient(
            endpoint=self.endpoint,
            credential=credential
        )
```

## Authorization & Access Control

### Role-Based Access Control (RBAC)

```python
from enum import Enum
from typing import List, Set

class SearchRole(Enum):
    """Azure AI Search built-in roles."""
    OWNER = "Owner"
    CONTRIBUTOR = "Contributor"
    READER = "Reader"
    SEARCH_SERVICE_CONTRIBUTOR = "Search Service Contributor"
    SEARCH_INDEX_DATA_CONTRIBUTOR = "Search Index Data Contributor"
    SEARCH_INDEX_DATA_READER = "Search Index Data Reader"

@dataclass
class RoleAssignment:
    """RBAC role assignment."""
    principal_id: str  # User, group, or service principal
    role: SearchRole
    scope: str  # Resource scope
    assigned_at: datetime
    assigned_by: str

class RBACManager:
    """Manage role-based access control."""
    
    def __init__(self, resource_id: str):
        self.resource_id = resource_id
        self.assignments: List[RoleAssignment] = []
        
    def assign_role(
        self,
        principal_id: str,
        role: SearchRole,
        scope: str = None,
        assigned_by: str = "system"
    ):
        """
        Assign a role to a principal.
        
        Args:
            principal_id: Azure AD user/group/service principal ID
            role: Role to assign
            scope: Resource scope (defaults to service level)
            assigned_by: Who assigned the role
        """
        if not scope:
            scope = self.resource_id
        
        assignment = RoleAssignment(
            principal_id=principal_id,
            role=role,
            scope=scope,
            assigned_at=datetime.utcnow(),
            assigned_by=assigned_by
        )
        
        self.assignments.append(assignment)
    
    def revoke_role(self, principal_id: str, role: SearchRole):
        """
        Revoke a role from a principal.
        
        Args:
            principal_id: Azure AD principal ID
            role: Role to revoke
        """
        self.assignments = [
            a for a in self.assignments
            if not (a.principal_id == principal_id and a.role == role)
        ]
    
    def get_principal_roles(self, principal_id: str) -> Set[SearchRole]:
        """
        Get all roles for a principal.
        
        Args:
            principal_id: Azure AD principal ID
            
        Returns:
            Set of roles
        """
        return {
            a.role for a in self.assignments
            if a.principal_id == principal_id
        }
    
    def has_permission(
        self,
        principal_id: str,
        required_role: SearchRole
    ) -> bool:
        """
        Check if principal has required permission.
        
        Args:
            principal_id: Azure AD principal ID
            required_role: Required role
            
        Returns:
            True if principal has permission
        """
        roles = self.get_principal_roles(principal_id)
        
        # Check for exact role match
        if required_role in roles:
            return True
        
        # Check for higher-level roles
        if SearchRole.OWNER in roles or SearchRole.CONTRIBUTOR in roles:
            return True
        
        return False

class CustomRole:
    """Define custom RBAC roles."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.permissions: List[str] = []
        
    def add_permission(self, action: str):
        """
        Add permission to role.
        
        Args:
            action: Action like 'Microsoft.Search/searchServices/read'
        """
        self.permissions.append(action)
    
    def to_json(self) -> dict:
        """Convert to Azure role definition JSON."""
        return {
            "Name": self.name,
            "Description": self.description,
            "Actions": self.permissions,
            "NotActions": [],
            "AssignableScopes": [
                "/subscriptions/{subscription-id}"
            ]
        }

# Example: Read-only indexer role
def create_indexer_reader_role() -> CustomRole:
    """Create custom role for read-only indexer access."""
    role = CustomRole(
        name="Search Indexer Reader",
        description="Can view indexers and their status"
    )
    
    role.add_permission("Microsoft.Search/searchServices/read")
    role.add_permission("Microsoft.Search/searchServices/indexers/read")
    role.add_permission("Microsoft.Search/searchServices/indexers/status/read")
    
    return role
```

## Encryption

### Encryption at Rest

```python
from azure.keyvault.keys import KeyClient
from azure.identity import DefaultAzureCredential

class EncryptionManager:
    """Manage encryption for Azure AI Search."""
    
    def __init__(
        self,
        search_service_name: str,
        key_vault_name: str
    ):
        self.search_service_name = search_service_name
        self.key_vault_name = key_vault_name
        self.key_vault_uri = f"https://{key_vault_name}.vault.azure.net"
        
    def create_encryption_key(self, key_name: str) -> dict:
        """
        Create encryption key in Azure Key Vault.
        
        Args:
            key_name: Name for the encryption key
            
        Returns:
            Key metadata
        """
        credential = DefaultAzureCredential()
        key_client = KeyClient(
            vault_url=self.key_vault_uri,
            credential=credential
        )
        
        # Create RSA key
        key = key_client.create_rsa_key(
            name=key_name,
            size=2048
        )
        
        return {
            "key_name": key.name,
            "key_id": key.id,
            "key_version": key.properties.version,
            "enabled": key.properties.enabled
        }
    
    def get_encryption_config(self, key_name: str) -> dict:
        """
        Get encryption configuration for search index.
        
        Args:
            key_name: Name of the encryption key
            
        Returns:
            Encryption configuration
        """
        return {
            "encryptionKey": {
                "keyVaultKeyName": key_name,
                "keyVaultUri": self.key_vault_uri,
                "accessCredentials": {
                    "applicationId": None,  # Use managed identity
                    "applicationSecret": None
                }
            }
        }
    
    def enable_customer_managed_keys(
        self,
        index_name: str,
        key_name: str
    ) -> dict:
        """
        Enable customer-managed keys for an index.
        
        Args:
            index_name: Name of the search index
            key_name: Name of the encryption key
            
        Returns:
            Updated index configuration
        """
        encryption_config = self.get_encryption_config(key_name)
        
        # Index definition with encryption
        index_config = {
            "name": index_name,
            "encryptionKey": encryption_config["encryptionKey"]
        }
        
        return index_config
    
    def rotate_encryption_key(
        self,
        old_key_name: str,
        new_key_name: str
    ):
        """
        Rotate encryption key.
        
        Args:
            old_key_name: Current key name
            new_key_name: New key name
        """
        # Create new key
        new_key = self.create_encryption_key(new_key_name)
        
        # Update all encrypted indexes
        # Note: This requires re-indexing
        
        return new_key

class DataEncryption:
    """Handle data encryption/decryption."""
    
    def __init__(self, key_vault_uri: str):
        self.key_vault_uri = key_vault_uri
        
    def encrypt_sensitive_field(
        self,
        value: str,
        key_name: str
    ) -> str:
        """
        Encrypt sensitive field value.
        
        Args:
            value: Plain text value
            key_name: Encryption key name
            
        Returns:
            Encrypted value (base64)
        """
        # This would use Azure Key Vault encryption
        # For demonstration, showing the pattern
        import base64
        
        # In production, use Key Vault encrypt operation
        encrypted = base64.b64encode(value.encode()).decode()
        return encrypted
    
    def decrypt_sensitive_field(
        self,
        encrypted_value: str,
        key_name: str
    ) -> str:
        """
        Decrypt sensitive field value.
        
        Args:
            encrypted_value: Encrypted value
            key_name: Encryption key name
            
        Returns:
            Plain text value
        """
        import base64
        
        # In production, use Key Vault decrypt operation
        decrypted = base64.b64decode(encrypted_value).decode()
        return decrypted
```

### Encryption in Transit

```python
class TransportSecurity:
    """Manage encryption in transit."""
    
    @staticmethod
    def get_tls_config() -> dict:
        """
        Get recommended TLS configuration.
        
        Returns:
            TLS settings
        """
        return {
            "minimum_tls_version": "1.2",
            "supported_protocols": ["TLSv1.2", "TLSv1.3"],
            "cipher_suites": [
                "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
                "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
                "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384",
                "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256"
            ],
            "enforce_https": True
        }
    
    @staticmethod
    def validate_endpoint(endpoint: str) -> bool:
        """
        Validate endpoint uses HTTPS.
        
        Args:
            endpoint: Endpoint URL
            
        Returns:
            True if HTTPS, False otherwise
        """
        return endpoint.lower().startswith("https://")
    
    @staticmethod
    def create_secure_client_config() -> dict:
        """
        Create secure HTTP client configuration.
        
        Returns:
            Client config with security settings
        """
        return {
            "verify_ssl": True,
            "ssl_version": "TLSv1.2",
            "timeout": 30,
            "max_retries": 3,
            "headers": {
                "User-Agent": "SecureSearchClient/1.0",
                "X-Content-Type-Options": "nosniff",
                "Strict-Transport-Security": "max-age=31536000"
            }
        }
```

## Network Security

### Private Endpoints

```python
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.search import SearchManagementClient

class NetworkSecurityManager:
    """Manage network security for Azure AI Search."""
    
    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        search_service_name: str
    ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.search_service_name = search_service_name
        
    def configure_private_endpoint(
        self,
        vnet_name: str,
        subnet_name: str,
        private_endpoint_name: str
    ) -> dict:
        """
        Configure private endpoint for search service.
        
        Args:
            vnet_name: Virtual network name
            subnet_name: Subnet name
            private_endpoint_name: Name for private endpoint
            
        Returns:
            Private endpoint configuration
        """
        credential = DefaultAzureCredential()
        
        network_client = NetworkManagementClient(
            credential=credential,
            subscription_id=self.subscription_id
        )
        
        # Get subnet
        subnet = network_client.subnets.get(
            resource_group_name=self.resource_group,
            virtual_network_name=vnet_name,
            subnet_name=subnet_name
        )
        
        # Create private endpoint
        private_endpoint_params = {
            "location": "eastus",
            "subnet": {"id": subnet.id},
            "private_link_service_connections": [{
                "name": f"{private_endpoint_name}-connection",
                "private_link_service_id": f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Search/searchServices/{self.search_service_name}",
                "group_ids": ["searchService"]
            }]
        }
        
        return private_endpoint_params
    
    def configure_private_dns_zone(
        self,
        private_dns_zone_name: str = "privatelink.search.windows.net"
    ) -> dict:
        """
        Configure private DNS zone.
        
        Args:
            private_dns_zone_name: DNS zone name
            
        Returns:
            DNS zone configuration
        """
        return {
            "zone_name": private_dns_zone_name,
            "resource_group": self.resource_group,
            "virtual_network_links": [{
                "registration_enabled": False,
                "virtual_network_id": f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Network/virtualNetworks/vnet"
            }]
        }
    
    def disable_public_access(self) -> dict:
        """
        Disable public network access.
        
        Returns:
            Network rule configuration
        """
        return {
            "publicNetworkAccess": "Disabled",
            "networkRuleSet": {
                "bypass": "None",
                "ipRules": []
            }
        }

class FirewallManager:
    """Manage IP firewall rules."""
    
    def __init__(self, search_service_name: str):
        self.search_service_name = search_service_name
        self.ip_rules: List[dict] = []
        
    def add_ip_rule(self, ip_address: str, description: str = ""):
        """
        Add IP firewall rule.
        
        Args:
            ip_address: IP address or CIDR range
            description: Rule description
        """
        rule = {
            "value": ip_address,
            "description": description
        }
        self.ip_rules.append(rule)
    
    def add_service_tag(self, service_tag: str):
        """
        Add Azure service tag.
        
        Args:
            service_tag: Service tag like 'AzureCloud', 'AzureFrontDoor'
        """
        rule = {
            "value": service_tag,
            "description": f"Service tag: {service_tag}"
        }
        self.ip_rules.append(rule)
    
    def configure_firewall_rules(self) -> dict:
        """
        Get firewall configuration.
        
        Returns:
            Firewall rules configuration
        """
        return {
            "networkRuleSet": {
                "ipRules": self.ip_rules,
                "bypass": "AzureServices"  # Allow Azure services
            }
        }
    
    def allow_azure_services(self) -> dict:
        """
        Configure to allow Azure services.
        
        Returns:
            Network configuration
        """
        return {
            "networkRuleSet": {
                "bypass": "AzureServices",
                "ipRules": self.ip_rules
            }
        }

# Example usage
def setup_network_security():
    """Setup comprehensive network security."""
    
    # Configure private endpoint
    net_security = NetworkSecurityManager(
        subscription_id="sub-123",
        resource_group="rg-search",
        search_service_name="my-search-service"
    )
    
    private_endpoint = net_security.configure_private_endpoint(
        vnet_name="search-vnet",
        subnet_name="search-subnet",
        private_endpoint_name="search-pe"
    )
    
    # Configure DNS
    dns_zone = net_security.configure_private_dns_zone()
    
    # Configure firewall
    firewall = FirewallManager("my-search-service")
    
    # Allow specific IPs
    firewall.add_ip_rule("203.0.113.0/24", "Corporate network")
    firewall.add_ip_rule("198.51.100.5", "Admin workstation")
    
    # Allow Azure services
    firewall.add_service_tag("AzureCloud")
    
    return {
        "private_endpoint": private_endpoint,
        "dns_zone": dns_zone,
        "firewall": firewall.configure_firewall_rules()
    }
```

## Compliance Frameworks

### GDPR Compliance

```python
class GDPRCompliance:
    """Ensure GDPR compliance for Azure AI Search."""
    
    def __init__(self, search_service_name: str):
        self.search_service_name = search_service_name
        
    def configure_data_residency(self, region: str) -> dict:
        """
        Configure data residency for GDPR.
        
        Args:
            region: Azure region (e.g., 'westeurope', 'northeurope')
            
        Returns:
            Service configuration
        """
        eu_regions = [
            "westeurope", "northeurope", "francecentral",
            "germanywestcentral", "norwayeast", "switzerlandnorth"
        ]
        
        if region not in eu_regions:
            raise ValueError(f"Region {region} not in EU for GDPR compliance")
        
        return {
            "location": region,
            "properties": {
                "dataResidency": "EU",
                "publicNetworkAccess": "Disabled"
            }
        }
    
    def implement_right_to_erasure(
        self,
        user_id: str,
        index_name: str
    ) -> dict:
        """
        Implement right to erasure (right to be forgotten).
        
        Args:
            user_id: User identifier
            index_name: Index name
            
        Returns:
            Deletion operation details
        """
        return {
            "operation": "delete",
            "index": index_name,
            "filter": f"userId eq '{user_id}'",
            "audit_log": {
                "timestamp": datetime.utcnow(),
                "user_id": user_id,
                "action": "data_erasure",
                "compliance": "GDPR Article 17"
            }
        }
    
    def implement_data_portability(
        self,
        user_id: str,
        index_name: str,
        format: str = "json"
    ) -> dict:
        """
        Implement right to data portability.
        
        Args:
            user_id: User identifier
            index_name: Index name
            format: Export format ('json', 'csv', 'xml')
            
        Returns:
            Export configuration
        """
        return {
            "operation": "export",
            "index": index_name,
            "filter": f"userId eq '{user_id}'",
            "format": format,
            "compliance": "GDPR Article 20"
        }
    
    def configure_consent_management(self) -> dict:
        """
        Configure consent management.
        
        Returns:
            Consent configuration
        """
        return {
            "consent_fields": [
                {
                    "name": "marketing_consent",
                    "type": "boolean",
                    "required": False
                },
                {
                    "name": "analytics_consent",
                    "type": "boolean",
                    "required": False
                },
                {
                    "name": "consent_timestamp",
                    "type": "datetime",
                    "required": True
                },
                {
                    "name": "consent_version",
                    "type": "string",
                    "required": True
                }
            ],
            "default_consent": False
        }

class HIPAACompliance:
    """Ensure HIPAA compliance for healthcare data."""
    
    def __init__(self, search_service_name: str):
        self.search_service_name = search_service_name
        
    def configure_hipaa_requirements(self) -> dict:
        """
        Configure HIPAA requirements.
        
        Returns:
            HIPAA configuration
        """
        return {
            "encryption": {
                "at_rest": {
                    "enabled": True,
                    "customer_managed_keys": True
                },
                "in_transit": {
                    "minimum_tls": "1.2",
                    "enforce_https": True
                }
            },
            "access_control": {
                "authentication": "AzureAD",
                "authorization": "RBAC",
                "mfa_required": True
            },
            "audit_logging": {
                "enabled": True,
                "retention_days": 2555,  # 7 years
                "log_access": True,
                "log_modifications": True
            },
            "network_security": {
                "private_endpoints": True,
                "public_access": False,
                "ip_whitelist": []
            }
        }
    
    def anonymize_phi(self, document: dict) -> dict:
        """
        Anonymize Protected Health Information.
        
        Args:
            document: Document with PHI
            
        Returns:
            Anonymized document
        """
        phi_fields = [
            "patient_name",
            "ssn",
            "medical_record_number",
            "date_of_birth",
            "address",
            "phone"
        ]
        
        anonymized = document.copy()
        
        for field in phi_fields:
            if field in anonymized:
                # Replace with anonymized value
                anonymized[field] = self._anonymize_value(
                    anonymized[field],
                    field
                )
        
        return anonymized
    
    def _anonymize_value(self, value: str, field: str) -> str:
        """Anonymize a specific field value."""
        import hashlib
        
        # Use consistent hashing for same values
        hashed = hashlib.sha256(value.encode()).hexdigest()
        return f"ANON_{field.upper()}_{hashed[:8]}"

class ComplianceAuditor:
    """Audit compliance across frameworks."""
    
    def __init__(self, search_service_name: str):
        self.search_service_name = search_service_name
        self.violations: List[dict] = []
        
    def scan_for_violations(
        self,
        framework: str = "all"
    ) -> List[dict]:
        """
        Scan for compliance violations.
        
        Args:
            framework: 'gdpr', 'hipaa', 'soc2', 'iso27001', or 'all'
            
        Returns:
            List of violations
        """
        violations = []
        
        # Check encryption
        if not self._check_encryption():
            violations.append({
                "severity": "critical",
                "framework": "all",
                "issue": "Encryption not enabled",
                "remediation": "Enable customer-managed keys"
            })
        
        # Check network security
        if not self._check_network_security():
            violations.append({
                "severity": "high",
                "framework": "all",
                "issue": "Public access enabled",
                "remediation": "Configure private endpoints"
            })
        
        # Check audit logging
        if not self._check_audit_logging():
            violations.append({
                "severity": "high",
                "framework": ["hipaa", "soc2", "iso27001"],
                "issue": "Audit logging not enabled",
                "remediation": "Enable diagnostic settings"
            })
        
        # Check data retention
        if not self._check_data_retention():
            violations.append({
                "severity": "medium",
                "framework": "gdpr",
                "issue": "No data retention policy",
                "remediation": "Configure retention periods"
            })
        
        self.violations = violations
        return violations
    
    def _check_encryption(self) -> bool:
        """Check if encryption is properly configured."""
        # Implementation would check actual service config
        return True
    
    def _check_network_security(self) -> bool:
        """Check network security configuration."""
        return True
    
    def _check_audit_logging(self) -> bool:
        """Check audit logging configuration."""
        return True
    
    def _check_data_retention(self) -> bool:
        """Check data retention policies."""
        return True
    
    def generate_compliance_report(self) -> dict:
        """
        Generate compliance audit report.
        
        Returns:
            Compliance report
        """
        return {
            "service_name": self.search_service_name,
            "audit_date": datetime.utcnow(),
            "violations": self.violations,
            "compliance_score": self._calculate_compliance_score(),
            "recommendations": self._get_recommendations()
        }
    
    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)."""
        if not self.violations:
            return 100.0
        
        severity_weights = {
            "critical": 25,
            "high": 10,
            "medium": 5,
            "low": 2
        }
        
        total_deductions = sum(
            severity_weights.get(v["severity"], 0)
            for v in self.violations
        )
        
        return max(0, 100 - total_deductions)
    
    def _get_recommendations(self) -> List[str]:
        """Get remediation recommendations."""
        return [
            v["remediation"]
            for v in sorted(
                self.violations,
                key=lambda x: ["critical", "high", "medium", "low"].index(
                    x["severity"]
                )
            )
        ]
```

## Security Monitoring

### Audit Logging

```python
from azure.monitor.query import LogsQueryClient
from azure.identity import DefaultAzureCredential

class AuditLogger:
    """Manage audit logging for Azure AI Search."""
    
    def __init__(
        self,
        search_service_name: str,
        log_analytics_workspace_id: str
    ):
        self.search_service_name = search_service_name
        self.workspace_id = log_analytics_workspace_id
        
    def configure_diagnostic_settings(self) -> dict:
        """
        Configure diagnostic settings for audit logging.
        
        Returns:
            Diagnostic settings configuration
        """
        return {
            "name": "audit-logs",
            "logs": [
                {
                    "category": "OperationLogs",
                    "enabled": True,
                    "retentionPolicy": {
                        "enabled": True,
                        "days": 90
                    }
                },
                {
                    "category": "SearchAuditLogs",
                    "enabled": True,
                    "retentionPolicy": {
                        "enabled": True,
                        "days": 90
                    }
                }
            ],
            "metrics": [
                {
                    "category": "AllMetrics",
                    "enabled": True,
                    "retentionPolicy": {
                        "enabled": True,
                        "days": 30
                    }
                }
            ],
            "workspaceId": self.workspace_id
        }
    
    def query_audit_logs(
        self,
        start_time: datetime,
        end_time: datetime,
        operation: str = None
    ) -> List[dict]:
        """
        Query audit logs.
        
        Args:
            start_time: Start time for query
            end_time: End time for query
            operation: Filter by operation type
            
        Returns:
            List of audit log entries
        """
        credential = DefaultAzureCredential()
        client = LogsQueryClient(credential)
        
        # Build KQL query
        query = f"""
        AzureDiagnostics
        | where ResourceProvider == "MICROSOFT.SEARCH"
        | where ResourceId contains "{self.search_service_name}"
        | where TimeGenerated between (datetime({start_time.isoformat()}) .. datetime({end_time.isoformat()}))
        """
        
        if operation:
            query += f"| where OperationName == '{operation}'"
        
        query += """
        | project TimeGenerated, OperationName, CallerIpAddress, 
                  identity_claim_upn_s, ResultType, ResultDescription
        | order by TimeGenerated desc
        """
        
        response = client.query_workspace(
            workspace_id=self.workspace_id,
            query=query,
            timespan=None
        )
        
        return [dict(row) for row in response.tables[0].rows]
    
    def detect_suspicious_activity(
        self,
        time_window_hours: int = 24
    ) -> List[dict]:
        """
        Detect suspicious activity patterns.
        
        Args:
            time_window_hours: Time window to analyze
            
        Returns:
            List of suspicious activities
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        suspicious = []
        
        # Query failed authentication attempts
        failed_auth = self._query_failed_auth(start_time, end_time)
        if len(failed_auth) > 10:
            suspicious.append({
                "type": "excessive_failed_auth",
                "count": len(failed_auth),
                "severity": "high"
            })
        
        # Query unusual access patterns
        unusual_ips = self._query_unusual_ips(start_time, end_time)
        if unusual_ips:
            suspicious.append({
                "type": "unusual_ip_access",
                "ips": unusual_ips,
                "severity": "medium"
            })
        
        # Query bulk operations
        bulk_ops = self._query_bulk_operations(start_time, end_time)
        if bulk_ops:
            suspicious.append({
                "type": "bulk_data_access",
                "operations": bulk_ops,
                "severity": "medium"
            })
        
        return suspicious
    
    def _query_failed_auth(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[dict]:
        """Query failed authentication attempts."""
        return self.query_audit_logs(
            start_time,
            end_time,
            operation="AuthenticationFailed"
        )
    
    def _query_unusual_ips(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[str]:
        """Detect unusual IP addresses."""
        logs = self.query_audit_logs(start_time, end_time)
        
        # Get IP addresses
        ips = {}
        for log in logs:
            ip = log.get("CallerIpAddress")
            if ip:
                ips[ip] = ips.get(ip, 0) + 1
        
        # Return IPs with unusual patterns
        return [
            ip for ip, count in ips.items()
            if count > 100  # Threshold
        ]
    
    def _query_bulk_operations(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[dict]:
        """Query bulk operations."""
        logs = self.query_audit_logs(start_time, end_time)
        
        # Group by user
        user_ops = {}
        for log in logs:
            user = log.get("identity_claim_upn_s", "unknown")
            user_ops[user] = user_ops.get(user, 0) + 1
        
        # Return users with bulk operations
        return [
            {"user": user, "count": count}
            for user, count in user_ops.items()
            if count > 1000  # Threshold
        ]

class SecurityMonitor:
    """Monitor security events and alerts."""
    
    def __init__(self, search_service_name: str):
        self.search_service_name = search_service_name
        self.alerts: List[dict] = []
        
    def create_security_alert(
        self,
        alert_type: str,
        severity: str,
        threshold: float
    ) -> dict:
        """
        Create security alert rule.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            threshold: Alert threshold
            
        Returns:
            Alert configuration
        """
        alert = {
            "name": f"{alert_type}-alert",
            "severity": severity,
            "threshold": threshold,
            "condition": self._get_alert_condition(alert_type, threshold),
            "actions": [
                {
                    "type": "email",
                    "recipients": ["security@example.com"]
                },
                {
                    "type": "webhook",
                    "url": "https://alerts.example.com/webhook"
                }
            ]
        }
        
        self.alerts.append(alert)
        return alert
    
    def _get_alert_condition(
        self,
        alert_type: str,
        threshold: float
    ) -> str:
        """Get KQL condition for alert."""
        conditions = {
            "failed_auth": f"""
                AzureDiagnostics
                | where OperationName == "AuthenticationFailed"
                | summarize count() by bin(TimeGenerated, 5m)
                | where count_ > {threshold}
            """,
            "unusual_qps": f"""
                AzureDiagnostics
                | where OperationName == "Query.Search"
                | summarize qps = count() by bin(TimeGenerated, 1m)
                | where qps > {threshold}
            """,
            "data_exfiltration": f"""
                AzureDiagnostics
                | where OperationName == "Documents.Search"
                | summarize docs = count() by CallerIpAddress, bin(TimeGenerated, 1m)
                | where docs > {threshold}
            """
        }
        
        return conditions.get(alert_type, "")
    
    def setup_security_monitoring(self) -> dict:
        """
        Setup comprehensive security monitoring.
        
        Returns:
            Monitoring configuration
        """
        # Failed authentication alert
        self.create_security_alert(
            alert_type="failed_auth",
            severity="high",
            threshold=10
        )
        
        # Unusual QPS alert
        self.create_security_alert(
            alert_type="unusual_qps",
            severity="medium",
            threshold=1000
        )
        
        # Data exfiltration alert
        self.create_security_alert(
            alert_type="data_exfiltration",
            severity="critical",
            threshold=10000
        )
        
        return {
            "alerts": self.alerts,
            "log_retention_days": 90,
            "enable_threat_detection": True
        }
```

## Data Governance

### Data Classification

```python
from enum import Enum

class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "Public"
    INTERNAL = "Internal"
    CONFIDENTIAL = "Confidential"
    RESTRICTED = "Restricted"

@dataclass
class ClassifiedField:
    """Field with data classification."""
    field_name: str
    classification: DataClassification
    encryption_required: bool
    access_control: List[str]
    retention_days: int

class DataGovernanceManager:
    """Manage data governance policies."""
    
    def __init__(self, search_service_name: str):
        self.search_service_name = search_service_name
        self.classified_fields: List[ClassifiedField] = []
        
    def classify_field(
        self,
        field_name: str,
        classification: DataClassification,
        encryption_required: bool = False,
        access_control: List[str] = None,
        retention_days: int = 365
    ):
        """
        Classify a field.
        
        Args:
            field_name: Field name
            classification: Classification level
            encryption_required: Whether encryption is required
            access_control: Roles with access
            retention_days: Data retention period
        """
        field = ClassifiedField(
            field_name=field_name,
            classification=classification,
            encryption_required=encryption_required,
            access_control=access_control or [],
            retention_days=retention_days
        )
        
        self.classified_fields.append(field)
    
    def get_field_classification(
        self,
        field_name: str
    ) -> Optional[ClassifiedField]:
        """
        Get classification for a field.
        
        Args:
            field_name: Field name
            
        Returns:
            Field classification or None
        """
        return next(
            (f for f in self.classified_fields if f.field_name == field_name),
            None
        )
    
    def enforce_retention_policy(
        self,
        index_name: str
    ) -> List[dict]:
        """
        Enforce data retention policies.
        
        Args:
            index_name: Index name
            
        Returns:
            List of deletion operations
        """
        deletions = []
        
        for field in self.classified_fields:
            cutoff_date = datetime.utcnow() - timedelta(
                days=field.retention_days
            )
            
            deletions.append({
                "index": index_name,
                "filter": f"{field.field_name}_timestamp lt {cutoff_date.isoformat()}",
                "reason": "retention_policy",
                "classification": field.classification.value
            })
        
        return deletions
    
    def generate_data_map(self) -> dict:
        """
        Generate data map for governance.
        
        Returns:
            Data map showing all classified fields
        """
        return {
            "service": self.search_service_name,
            "fields": [
                {
                    "name": f.field_name,
                    "classification": f.classification.value,
                    "encryption": f.encryption_required,
                    "access_control": f.access_control,
                    "retention_days": f.retention_days
                }
                for f in self.classified_fields
            ],
            "generated_at": datetime.utcnow()
        }

# Example: Setup data governance
def setup_data_governance():
    """Setup data governance for search service."""
    
    governance = DataGovernanceManager("my-search-service")
    
    # Classify fields
    governance.classify_field(
        field_name="customer_name",
        classification=DataClassification.CONFIDENTIAL,
        encryption_required=True,
        access_control=["CustomerServiceRole"],
        retention_days=2555  # 7 years
    )
    
    governance.classify_field(
        field_name="ssn",
        classification=DataClassification.RESTRICTED,
        encryption_required=True,
        access_control=["ComplianceOfficerRole"],
        retention_days=2555
    )
    
    governance.classify_field(
        field_name="product_description",
        classification=DataClassification.PUBLIC,
        encryption_required=False,
        access_control=["AllUsers"],
        retention_days=365
    )
    
    # Generate data map
    data_map = governance.generate_data_map()
    
    return data_map
```

## Security Best Practices

### Security Checklist

```python
class SecurityBestPractices:
    """Security best practices checklist."""
    
    @staticmethod
    def get_checklist() -> List[dict]:
        """
        Get security best practices checklist.
        
        Returns:
            List of best practices
        """
        return [
            {
                "category": "Authentication",
                "practices": [
                    "Use Azure AD authentication instead of API keys where possible",
                    "Enable multi-factor authentication (MFA) for admin accounts",
                    "Rotate API keys quarterly",
                    "Use managed identities for Azure service authentication",
                    "Implement API key expiration policies"
                ]
            },
            {
                "category": "Authorization",
                "practices": [
                    "Implement role-based access control (RBAC)",
                    "Follow principle of least privilege",
                    "Use separate query and admin keys",
                    "Create custom roles for specific job functions",
                    "Regularly audit role assignments"
                ]
            },
            {
                "category": "Encryption",
                "practices": [
                    "Enable customer-managed keys for encryption at rest",
                    "Enforce TLS 1.2 or higher for in-transit encryption",
                    "Rotate encryption keys annually",
                    "Use Azure Key Vault for key management",
                    "Encrypt sensitive fields at application level"
                ]
            },
            {
                "category": "Network Security",
                "practices": [
                    "Configure private endpoints for production",
                    "Disable public network access when possible",
                    "Implement IP firewall rules",
                    "Use Azure service tags for firewall rules",
                    "Configure private DNS zones"
                ]
            },
            {
                "category": "Monitoring",
                "practices": [
                    "Enable diagnostic logging",
                    "Configure audit logs with 90-day retention",
                    "Set up alerts for failed authentication",
                    "Monitor for unusual access patterns",
                    "Implement security incident response plan"
                ]
            },
            {
                "category": "Compliance",
                "practices": [
                    "Document data residency requirements",
                    "Implement data retention policies",
                    "Enable audit trails for compliance",
                    "Conduct regular compliance audits",
                    "Maintain data classification inventory"
                ]
            },
            {
                "category": "Data Protection",
                "practices": [
                    "Classify data by sensitivity level",
                    "Implement data masking for sensitive fields",
                    "Enable soft delete for indexes",
                    "Regular backup of index definitions",
                    "Test data recovery procedures"
                ]
            },
            {
                "category": "Operational Security",
                "practices": [
                    "Use Infrastructure as Code (IaC) for deployments",
                    "Implement CI/CD security gates",
                    "Conduct security code reviews",
                    "Perform regular penetration testing",
                    "Maintain incident response runbooks"
                ]
            }
        ]
    
    @staticmethod
    def assess_security_posture(
        current_config: dict
    ) -> dict:
        """
        Assess current security posture.
        
        Args:
            current_config: Current security configuration
            
        Returns:
            Security assessment
        """
        checklist = SecurityBestPractices.get_checklist()
        
        assessment = {
            "overall_score": 0,
            "category_scores": {},
            "recommendations": []
        }
        
        # Check authentication
        auth_score = 0
        if current_config.get("azure_ad_enabled"):
            auth_score += 30
        if current_config.get("mfa_enabled"):
            auth_score += 20
        if current_config.get("managed_identity"):
            auth_score += 25
        if current_config.get("key_rotation"):
            auth_score += 25
        
        assessment["category_scores"]["authentication"] = auth_score
        
        # Check encryption
        encryption_score = 0
        if current_config.get("cmk_enabled"):
            encryption_score += 40
        if current_config.get("tls_version") >= 1.2:
            encryption_score += 30
        if current_config.get("key_vault"):
            encryption_score += 30
        
        assessment["category_scores"]["encryption"] = encryption_score
        
        # Check network security
        network_score = 0
        if current_config.get("private_endpoints"):
            network_score += 40
        if current_config.get("public_access_disabled"):
            network_score += 30
        if current_config.get("firewall_rules"):
            network_score += 30
        
        assessment["category_scores"]["network"] = network_score
        
        # Calculate overall score
        assessment["overall_score"] = sum(
            assessment["category_scores"].values()
        ) / len(assessment["category_scores"])
        
        # Generate recommendations
        if auth_score < 80:
            assessment["recommendations"].append(
                "Enable Azure AD authentication and MFA"
            )
        if encryption_score < 80:
            assessment["recommendations"].append(
                "Enable customer-managed keys"
            )
        if network_score < 80:
            assessment["recommendations"].append(
                "Configure private endpoints"
            )
        
        return assessment

# Example: Security assessment
def perform_security_assessment():
    """Perform comprehensive security assessment."""
    
    current_config = {
        "azure_ad_enabled": True,
        "mfa_enabled": True,
        "managed_identity": True,
        "key_rotation": False,
        "cmk_enabled": True,
        "tls_version": 1.2,
        "key_vault": True,
        "private_endpoints": False,
        "public_access_disabled": False,
        "firewall_rules": True
    }
    
    assessment = SecurityBestPractices.assess_security_posture(
        current_config
    )
    
    print(f"Overall Security Score: {assessment['overall_score']:.1f}/100")
    print("\nCategory Scores:")
    for category, score in assessment["category_scores"].items():
        print(f"  {category}: {score}/100")
    
    print("\nRecommendations:")
    for rec in assessment["recommendations"]:
        print(f"  - {rec}")
    
    return assessment
```

## Summary

This guide covered comprehensive security and compliance practices for Azure AI Search:

### Key Takeaways

1. **Authentication**: Use Azure AD with managed identities instead of API keys for production
2. **Authorization**: Implement RBAC with least privilege principle
3. **Encryption**: Enable customer-managed keys and enforce TLS 1.2+
4. **Network Security**: Configure private endpoints and disable public access
5. **Compliance**: Implement framework-specific requirements (GDPR, HIPAA)
6. **Monitoring**: Enable audit logging with security alerts
7. **Data Governance**: Classify data and enforce retention policies

### Security Layers

- **Identity Layer**: Azure AD, MFA, managed identities
- **Network Layer**: Private endpoints, firewalls, service tags
- **Data Layer**: Encryption at rest and in transit, customer-managed keys
- **Application Layer**: RBAC, custom roles, API key management
- **Monitoring Layer**: Audit logs, security alerts, threat detection
- **Compliance Layer**: GDPR, HIPAA, SOC 2, ISO 27001

### Production Recommendations

1. Always use managed identities for Azure service authentication
2. Configure private endpoints for production workloads
3. Enable diagnostic logging with 90-day retention minimum
4. Implement automated security scanning in CI/CD pipelines
5. Conduct quarterly security assessments
6. Maintain incident response runbooks
7. Perform annual disaster recovery drills
8. Document all compliance requirements

### Next Steps

- **Page 22**: CI/CD Pipelines - Automate secure deployments
- **Page 23**: Monitoring & Alerting - Production observability
- **Page 24**: Domain-Specific Solutions - Industry implementations
