# CI/CD Pipelines for Azure AI Search

## Table of Contents

- [Overview](#overview)
- [Infrastructure as Code](#infrastructure-as-code)
- [Azure DevOps Pipelines](#azure-devops-pipelines)
- [GitHub Actions](#github-actions)
- [Deployment Strategies](#deployment-strategies)
- [Index Migration](#index-migration)
- [Secret Management](#secret-management)
- [Testing Gates](#testing-gates)
- [Rollback Procedures](#rollback-procedures)
- [Best Practices](#best-practices)

---

## Overview

### CI/CD for Search Services

Automate deployment and management of Azure AI Search infrastructure and configurations.

```
Code → Build → Test → Deploy → Validate → Monitor
```

**Key Components**:
- **Infrastructure as Code (IaC)**: Bicep/ARM templates
- **Pipeline Automation**: Azure DevOps or GitHub Actions
- **Index Versioning**: Blue-green deployments
- **Secret Management**: Azure Key Vault integration
- **Testing Gates**: Smoke tests and validation

### Benefits

- **Consistency**: Repeatable deployments across environments
- **Version Control**: Track all configuration changes
- **Automation**: Reduce manual errors
- **Rapid Rollback**: Quick recovery from issues
- **Compliance**: Audit trail of all changes

---

## Infrastructure as Code

### Bicep Templates

```bicep
// search-service.bicep
@description('Search service name')
param searchServiceName string

@description('Location for resources')
param location string = resourceGroup().location

@description('SKU for search service')
@allowed([
  'free'
  'basic'
  'standard'
  'standard2'
  'standard3'
  'storage_optimized_l1'
  'storage_optimized_l2'
])
param sku string = 'standard'

@description('Number of replicas')
@minValue(1)
@maxValue(12)
param replicaCount int = 1

@description('Number of partitions')
@allowed([1, 2, 3, 4, 6, 12])
param partitionCount int = 1

@description('Public network access')
@allowed(['enabled', 'disabled'])
param publicNetworkAccess string = 'enabled'

@description('Tags for resources')
param tags object = {}

// Search Service
resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: searchServiceName
  location: location
  sku: {
    name: sku
  }
  properties: {
    replicaCount: replicaCount
    partitionCount: partitionCount
    hostingMode: 'default'
    publicNetworkAccess: publicNetworkAccess
    networkRuleSet: {
      ipRules: []
    }
    encryptionWithCmk: {
      enforcement: 'Unspecified'
    }
    disableLocalAuth: false
    authOptions: {
      aadOrApiKey: {
        aadAuthFailureMode: 'http401WithBearerChallenge'
      }
    }
  }
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
}

// Diagnostic Settings
resource diagnosticSettings 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'search-diagnostics'
  scope: searchService
  properties: {
    workspaceId: '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.OperationalInsights/workspaces/search-workspace'
    logs: [
      {
        category: 'OperationLogs'
        enabled: true
        retentionPolicy: {
          days: 90
          enabled: true
        }
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
        retentionPolicy: {
          days: 90
          enabled: true
        }
      }
    ]
  }
}

// Outputs
output searchServiceId string = searchService.id
output searchServiceName string = searchService.name
output searchServiceEndpoint string = 'https://${searchService.name}.search.windows.net'
output principalId string = searchService.identity.principalId
```

### Complete Infrastructure Template

```bicep
// main.bicep
@description('Environment name')
@allowed(['dev', 'staging', 'prod'])
param environment string

@description('Location')
param location string = resourceGroup().location

// Variables
var searchServiceName = 'search-${environment}-${uniqueString(resourceGroup().id)}'
var storageAccountName = 'storage${environment}${uniqueString(resourceGroup().id)}'
var keyVaultName = 'kv-${environment}-${uniqueString(resourceGroup().id)}'
var appInsightsName = 'appins-${environment}-${uniqueString(resourceGroup().id)}'

var tags = {
  environment: environment
  managedBy: 'bicep'
  project: 'azure-ai-search'
}

// Search Service
module searchService 'search-service.bicep' = {
  name: 'searchServiceDeployment'
  params: {
    searchServiceName: searchServiceName
    location: location
    sku: environment == 'prod' ? 'standard' : 'basic'
    replicaCount: environment == 'prod' ? 3 : 1
    partitionCount: environment == 'prod' ? 2 : 1
    publicNetworkAccess: environment == 'prod' ? 'disabled' : 'enabled'
    tags: tags
  }
}

// Storage Account (for indexer data sources)
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
  tags: tags
}

// Blob Container
resource blobContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/documents'
  properties: {
    publicAccess: 'None'
  }
}

// Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
  tags: tags
}

// Store Search Admin Key in Key Vault
resource searchAdminKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'search-admin-key'
  properties: {
    value: searchService.outputs.searchServiceName // Placeholder - actual key retrieved post-deployment
  }
}

// Application Insights
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    RetentionInDays: 90
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
  tags: tags
}

// Outputs
output searchServiceEndpoint string = searchService.outputs.searchServiceEndpoint
output searchServiceName string = searchService.outputs.searchServiceName
output storageAccountName string = storageAccount.name
output keyVaultName string = keyVault.name
output appInsightsInstrumentationKey string = appInsights.properties.InstrumentationKey
```

### Python IaC Manager

```python
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import DeploymentMode
import json
from typing import Dict, Optional

class IaCManager:
    """Manage Infrastructure as Code deployments."""
    
    def __init__(
        self,
        subscription_id: str,
        resource_group_name: str
    ):
        self.credential = DefaultAzureCredential()
        self.subscription_id = subscription_id
        self.resource_group_name = resource_group_name
        
        self.resource_client = ResourceManagementClient(
            credential=self.credential,
            subscription_id=subscription_id
        )
    
    def deploy_bicep(
        self,
        deployment_name: str,
        bicep_file_path: str,
        parameters: Dict = None
    ) -> dict:
        """
        Deploy Bicep template.
        
        Args:
            deployment_name: Name for deployment
            bicep_file_path: Path to Bicep file
            parameters: Deployment parameters
            
        Returns:
            Deployment result
        """
        # Convert Bicep to ARM template using Azure CLI
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as arm_file:
            # bicep build command
            result = subprocess.run(
                ['az', 'bicep', 'build', '--file', bicep_file_path, '--outfile', arm_file.name],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Bicep build failed: {result.stderr}")
            
            # Read ARM template
            with open(arm_file.name, 'r') as f:
                template = json.load(f)
        
        # Prepare parameters
        deployment_parameters = {}
        if parameters:
            for key, value in parameters.items():
                deployment_parameters[key] = {'value': value}
        
        # Deploy
        deployment_properties = {
            'mode': DeploymentMode.incremental,
            'template': template,
            'parameters': deployment_parameters
        }
        
        deployment_async_operation = self.resource_client.deployments.begin_create_or_update(
            resource_group_name=self.resource_group_name,
            deployment_name=deployment_name,
            parameters={'properties': deployment_properties}
        )
        
        # Wait for completion
        deployment_result = deployment_async_operation.result()
        
        return {
            'deployment_name': deployment_name,
            'provisioning_state': deployment_result.properties.provisioning_state,
            'outputs': deployment_result.properties.outputs
        }
    
    def validate_deployment(
        self,
        bicep_file_path: str,
        parameters: Dict = None
    ) -> dict:
        """
        Validate Bicep template without deploying.
        
        Args:
            bicep_file_path: Path to Bicep file
            parameters: Deployment parameters
            
        Returns:
            Validation result
        """
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as arm_file:
            subprocess.run(
                ['az', 'bicep', 'build', '--file', bicep_file_path, '--outfile', arm_file.name],
                capture_output=True
            )
            
            with open(arm_file.name, 'r') as f:
                template = json.load(f)
        
        deployment_parameters = {}
        if parameters:
            for key, value in parameters.items():
                deployment_parameters[key] = {'value': value}
        
        deployment_properties = {
            'mode': DeploymentMode.incremental,
            'template': template,
            'parameters': deployment_parameters
        }
        
        validation_result = self.resource_client.deployments.validate(
            resource_group_name=self.resource_group_name,
            deployment_name='validation',
            parameters={'properties': deployment_properties}
        )
        
        return {
            'valid': validation_result.error is None,
            'error': validation_result.error.message if validation_result.error else None
        }
    
    def get_deployment_status(self, deployment_name: str) -> dict:
        """Get deployment status."""
        deployment = self.resource_client.deployments.get(
            resource_group_name=self.resource_group_name,
            deployment_name=deployment_name
        )
        
        return {
            'name': deployment.name,
            'provisioning_state': deployment.properties.provisioning_state,
            'timestamp': deployment.properties.timestamp,
            'outputs': deployment.properties.outputs
        }
```

---

## Azure DevOps Pipelines

### Pipeline YAML

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
      - main
      - develop
  paths:
    include:
      - infrastructure/*
      - src/*
      - pipelines/*

variables:
  - group: search-service-variables
  - name: azureSubscription
    value: 'Azure-Service-Connection'
  - name: resourceGroupName
    value: 'rg-search-$(environment)'
  - name: location
    value: 'eastus'

stages:
  - stage: Build
    displayName: 'Build and Validate'
    jobs:
      - job: ValidateInfrastructure
        displayName: 'Validate Bicep Templates'
        pool:
          vmImage: 'ubuntu-latest'
        steps:
          - task: AzureCLI@2
            displayName: 'Install Bicep'
            inputs:
              azureSubscription: $(azureSubscription)
              scriptType: 'bash'
              scriptLocation: 'inlineScript'
              inlineScript: |
                az bicep install
                az bicep version
          
          - task: AzureCLI@2
            displayName: 'Validate Bicep Template'
            inputs:
              azureSubscription: $(azureSubscription)
              scriptType: 'bash'
              scriptLocation: 'inlineScript'
              inlineScript: |
                az deployment group validate \
                  --resource-group $(resourceGroupName) \
                  --template-file infrastructure/main.bicep \
                  --parameters environment=$(environment) location=$(location)
          
          - task: UsePythonVersion@0
            displayName: 'Use Python 3.11'
            inputs:
              versionSpec: '3.11'
          
          - script: |
              python -m pip install --upgrade pip
              pip install -r requirements.txt
              pip install pytest pytest-cov
            displayName: 'Install Python Dependencies'
          
          - script: |
              pytest tests/ --cov=src --cov-report=xml --cov-report=html
            displayName: 'Run Unit Tests'
          
          - task: PublishTestResults@2
            displayName: 'Publish Test Results'
            inputs:
              testResultsFormat: 'JUnit'
              testResultsFiles: '**/test-results.xml'
              failTaskOnFailedTests: true
          
          - task: PublishCodeCoverageResults@1
            displayName: 'Publish Code Coverage'
            inputs:
              codeCoverageTool: 'Cobertura'
              summaryFileLocation: '$(System.DefaultWorkingDirectory)/coverage.xml'

  - stage: DeployDev
    displayName: 'Deploy to Dev'
    dependsOn: Build
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/develop'))
    variables:
      - name: environment
        value: 'dev'
    jobs:
      - deployment: DeployInfrastructure
        displayName: 'Deploy Infrastructure'
        pool:
          vmImage: 'ubuntu-latest'
        environment: 'dev'
        strategy:
          runOnce:
            deploy:
              steps:
                - checkout: self
                
                - task: AzureCLI@2
                  displayName: 'Deploy Bicep Template'
                  inputs:
                    azureSubscription: $(azureSubscription)
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      az deployment group create \
                        --resource-group $(resourceGroupName) \
                        --template-file infrastructure/main.bicep \
                        --parameters environment=$(environment) location=$(location) \
                        --mode Incremental
                
                - task: AzureCLI@2
                  displayName: 'Get Deployment Outputs'
                  inputs:
                    azureSubscription: $(azureSubscription)
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      outputs=$(az deployment group show \
                        --resource-group $(resourceGroupName) \
                        --name main \
                        --query properties.outputs)
                      
                      echo "##vso[task.setvariable variable=searchEndpoint]$(echo $outputs | jq -r '.searchServiceEndpoint.value')"
                      echo "##vso[task.setvariable variable=searchServiceName]$(echo $outputs | jq -r '.searchServiceName.value')"
                
                - task: UsePythonVersion@0
                  inputs:
                    versionSpec: '3.11'
                
                - script: |
                    pip install -r requirements.txt
                  displayName: 'Install Dependencies'
                
                - task: AzureCLI@2
                  displayName: 'Configure Search Service'
                  inputs:
                    azureSubscription: $(azureSubscription)
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      python scripts/configure_search.py \
                        --endpoint $(searchEndpoint) \
                        --environment $(environment)
                
                - task: AzureCLI@2
                  displayName: 'Run Smoke Tests'
                  inputs:
                    azureSubscription: $(azureSubscription)
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      python tests/smoke_tests.py \
                        --endpoint $(searchEndpoint)

  - stage: DeployStaging
    displayName: 'Deploy to Staging'
    dependsOn: DeployDev
    condition: succeeded()
    variables:
      - name: environment
        value: 'staging'
    jobs:
      - deployment: DeployInfrastructure
        displayName: 'Deploy Infrastructure'
        pool:
          vmImage: 'ubuntu-latest'
        environment: 'staging'
        strategy:
          runOnce:
            deploy:
              steps:
                - checkout: self
                
                - task: AzureCLI@2
                  displayName: 'Deploy to Staging'
                  inputs:
                    azureSubscription: $(azureSubscription)
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      az deployment group create \
                        --resource-group $(resourceGroupName) \
                        --template-file infrastructure/main.bicep \
                        --parameters environment=$(environment) location=$(location)

  - stage: DeployProduction
    displayName: 'Deploy to Production'
    dependsOn: DeployStaging
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
    variables:
      - name: environment
        value: 'prod'
    jobs:
      - deployment: DeployInfrastructure
        displayName: 'Deploy Infrastructure'
        pool:
          vmImage: 'ubuntu-latest'
        environment: 'production'
        strategy:
          runOnce:
            deploy:
              steps:
                - checkout: self
                
                - task: AzureCLI@2
                  displayName: 'Deploy to Production'
                  inputs:
                    azureSubscription: $(azureSubscription)
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      az deployment group create \
                        --resource-group $(resourceGroupName) \
                        --template-file infrastructure/main.bicep \
                        --parameters environment=$(environment) location=$(location)
                
                - task: AzureCLI@2
                  displayName: 'Blue-Green Index Deployment'
                  inputs:
                    azureSubscription: $(azureSubscription)
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      python scripts/blue_green_deployment.py \
                        --environment prod
```

### Python Configuration Script

```python
# scripts/configure_search.py
import argparse
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.identity import DefaultAzureCredential
import json

def configure_search_service(endpoint: str, environment: str):
    """Configure search service with indexes and configurations."""
    
    credential = DefaultAzureCredential()
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    
    # Load index definitions
    with open(f'config/indexes_{environment}.json', 'r') as f:
        index_configs = json.load(f)
    
    for index_config in index_configs:
        index_name = index_config['name']
        
        print(f"Creating/updating index: {index_name}")
        
        # Create index (implementation depends on index structure)
        # This is a simplified example
        index = SearchIndex(
            name=index_name,
            fields=index_config['fields']
        )
        
        index_client.create_or_update_index(index)
        print(f"Index {index_name} configured successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--environment', required=True)
    
    args = parser.parse_args()
    configure_search_service(args.endpoint, args.environment)
```

---

## GitHub Actions

### Workflow YAML

```yaml
# .github/workflows/deploy.yml
name: Deploy Azure AI Search

on:
  push:
    branches:
      - main
      - develop
    paths:
      - 'infrastructure/**'
      - 'src/**'
      - '.github/workflows/**'
  pull_request:
    branches:
      - main
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy'
        required: true
        type: choice
        options:
          - dev
          - staging
          - prod

env:
  AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
  RESOURCE_GROUP_PREFIX: 'rg-search'
  LOCATION: 'eastus'

jobs:
  validate:
    name: Validate Infrastructure
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Install Bicep
        run: |
          az bicep install
          az bicep version
      
      - name: Validate Bicep template
        run: |
          az deployment group validate \
            --resource-group ${{ env.RESOURCE_GROUP_PREFIX }}-dev \
            --template-file infrastructure/main.bicep \
            --parameters environment=dev location=${{ env.LOCATION }}
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: true

  deploy-dev:
    name: Deploy to Dev
    needs: validate
    if: github.ref == 'refs/heads/develop' || github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    environment:
      name: dev
      url: ${{ steps.deploy.outputs.searchEndpoint }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Deploy infrastructure
        id: deploy
        uses: azure/arm-deploy@v1
        with:
          subscriptionId: ${{ env.AZURE_SUBSCRIPTION_ID }}
          resourceGroupName: ${{ env.RESOURCE_GROUP_PREFIX }}-dev
          template: infrastructure/main.bicep
          parameters: environment=dev location=${{ env.LOCATION }}
          deploymentMode: Incremental
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Configure search service
        env:
          SEARCH_ENDPOINT: ${{ steps.deploy.outputs.searchServiceEndpoint }}
        run: |
          python scripts/configure_search.py \
            --endpoint $SEARCH_ENDPOINT \
            --environment dev
      
      - name: Run smoke tests
        env:
          SEARCH_ENDPOINT: ${{ steps.deploy.outputs.searchServiceEndpoint }}
        run: |
          python tests/smoke_tests.py --endpoint $SEARCH_ENDPOINT

  deploy-staging:
    name: Deploy to Staging
    needs: deploy-dev
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: ${{ steps.deploy.outputs.searchEndpoint }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Deploy infrastructure
        id: deploy
        uses: azure/arm-deploy@v1
        with:
          subscriptionId: ${{ env.AZURE_SUBSCRIPTION_ID }}
          resourceGroupName: ${{ env.RESOURCE_GROUP_PREFIX }}-staging
          template: infrastructure/main.bicep
          parameters: environment=staging location=${{ env.LOCATION }}
      
      - name: Configure and test
        run: |
          python scripts/configure_search.py --endpoint ${{ steps.deploy.outputs.searchServiceEndpoint }} --environment staging
          python tests/smoke_tests.py --endpoint ${{ steps.deploy.outputs.searchServiceEndpoint }}

  deploy-prod:
    name: Deploy to Production
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment:
      name: production
      url: ${{ steps.deploy.outputs.searchEndpoint }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Deploy infrastructure
        id: deploy
        uses: azure/arm-deploy@v1
        with:
          subscriptionId: ${{ env.AZURE_SUBSCRIPTION_ID }}
          resourceGroupName: ${{ env.RESOURCE_GROUP_PREFIX }}-prod
          template: infrastructure/main.bicep
          parameters: environment=prod location=${{ env.LOCATION }}
      
      - name: Blue-Green Deployment
        env:
          SEARCH_ENDPOINT: ${{ steps.deploy.outputs.searchServiceEndpoint }}
        run: |
          python scripts/blue_green_deployment.py --environment prod
      
      - name: Post-deployment validation
        env:
          SEARCH_ENDPOINT: ${{ steps.deploy.outputs.searchServiceEndpoint }}
        run: |
          python tests/smoke_tests.py --endpoint $SEARCH_ENDPOINT
          python tests/integration_tests.py --endpoint $SEARCH_ENDPOINT
```

### Reusable Workflow

```yaml
# .github/workflows/reusable-deploy.yml
name: Reusable Deploy Workflow

on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
      resource-group:
        required: true
        type: string
    secrets:
      AZURE_CREDENTIALS:
        required: true
    outputs:
      searchEndpoint:
        description: "Search service endpoint"
        value: ${{ jobs.deploy.outputs.endpoint }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    outputs:
      endpoint: ${{ steps.deploy.outputs.searchServiceEndpoint }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Deploy
        id: deploy
        uses: azure/arm-deploy@v1
        with:
          resourceGroupName: ${{ inputs.resource-group }}
          template: infrastructure/main.bicep
          parameters: environment=${{ inputs.environment }}
```

---

## Deployment Strategies

### Blue-Green Deployment

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SearchAlias
from azure.identity import DefaultAzureCredential
from typing import Optional
import time

class BlueGreenDeployment:
    """Implement blue-green deployment for search indexes."""
    
    def __init__(self, search_endpoint: str):
        self.credential = DefaultAzureCredential()
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=self.credential
        )
    
    def deploy_new_version(
        self,
        base_index_name: str,
        new_index_definition: SearchIndex,
        alias_name: Optional[str] = None
    ) -> dict:
        """
        Deploy new index version using blue-green strategy.
        
        Args:
            base_index_name: Base name for index
            new_index_definition: New index schema
            alias_name: Optional alias name (defaults to base_index_name)
            
        Returns:
            Deployment summary
        """
        if alias_name is None:
            alias_name = base_index_name
        
        # Determine version numbers
        current_version = self._get_current_version(base_index_name)
        new_version = current_version + 1
        
        new_index_name = f"{base_index_name}-v{new_version}"
        old_index_name = f"{base_index_name}-v{current_version}"
        
        print(f"Deploying new version: {new_index_name}")
        
        # Step 1: Create new index
        new_index_definition.name = new_index_name
        self.index_client.create_index(new_index_definition)
        print(f"Created index: {new_index_name}")
        
        # Step 2: Populate new index
        # (This would be done by indexer or data upload)
        print(f"Populating {new_index_name}...")
        
        # Step 3: Validate new index
        validation_result = self._validate_index(new_index_name)
        
        if not validation_result['valid']:
            print(f"Validation failed: {validation_result['error']}")
            # Rollback - delete new index
            self.index_client.delete_index(new_index_name)
            raise Exception("Index validation failed")
        
        print(f"Index validation passed")
        
        # Step 4: Update alias to point to new index
        self._update_alias(alias_name, new_index_name)
        print(f"Alias '{alias_name}' now points to {new_index_name}")
        
        # Step 5: Monitor new index
        print("Monitoring new index performance...")
        time.sleep(60)  # Wait period
        
        # Step 6: Delete old index (optional - keep for rollback)
        # self.index_client.delete_index(old_index_name)
        # print(f"Deleted old index: {old_index_name}")
        
        return {
            'new_index': new_index_name,
            'old_index': old_index_name,
            'alias': alias_name,
            'status': 'success'
        }
    
    def rollback(
        self,
        base_index_name: str,
        alias_name: Optional[str] = None
    ):
        """
        Rollback to previous index version.
        
        Args:
            base_index_name: Base index name
            alias_name: Alias name
        """
        if alias_name is None:
            alias_name = base_index_name
        
        current_version = self._get_current_version(base_index_name)
        previous_version = current_version - 1
        
        if previous_version < 1:
            raise Exception("No previous version to rollback to")
        
        previous_index_name = f"{base_index_name}-v{previous_version}"
        
        # Update alias to point to previous version
        self._update_alias(alias_name, previous_index_name)
        print(f"Rolled back to {previous_index_name}")
        
        # Delete current version
        current_index_name = f"{base_index_name}-v{current_version}"
        self.index_client.delete_index(current_index_name)
        print(f"Deleted failed index: {current_index_name}")
    
    def _get_current_version(self, base_index_name: str) -> int:
        """Get current version number from existing indexes."""
        indexes = self.index_client.list_indexes()
        
        versions = []
        for index in indexes:
            if index.name.startswith(f"{base_index_name}-v"):
                try:
                    version = int(index.name.split('-v')[1])
                    versions.append(version)
                except ValueError:
                    continue
        
        return max(versions) if versions else 0
    
    def _validate_index(self, index_name: str) -> dict:
        """Validate index is ready for traffic."""
        try:
            index = self.index_client.get_index(index_name)
            
            # Check index exists and has schema
            if not index.fields:
                return {'valid': False, 'error': 'Index has no fields'}
            
            # Additional validation checks
            # - Document count
            # - Sample queries
            # - Performance tests
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _update_alias(self, alias_name: str, target_index_name: str):
        """Update or create alias pointing to target index."""
        try:
            # Try to get existing alias
            alias = self.index_client.get_alias(alias_name)
            
            # Update to point to new index
            alias.indexes = [target_index_name]
            self.index_client.create_or_update_alias(alias)
            
        except:
            # Create new alias
            alias = SearchAlias(
                name=alias_name,
                indexes=[target_index_name]
            )
            self.index_client.create_alias(alias)
```

### Canary Deployment

```python
class CanaryDeployment:
    """Implement canary deployment with gradual traffic shift."""
    
    def __init__(self, search_endpoint: str):
        self.credential = DefaultAzureCredential()
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=self.credential
        )
    
    def deploy_canary(
        self,
        stable_index: str,
        canary_index: str,
        canary_percentage: int = 10
    ):
        """
        Deploy canary index with percentage of traffic.
        
        Note: Azure AI Search doesn't natively support traffic splitting.
        This requires application-level routing logic.
        
        Args:
            stable_index: Current production index
            canary_index: New canary index
            canary_percentage: Percentage of traffic to canary (0-100)
        """
        print(f"Canary deployment: {canary_percentage}% to {canary_index}")
        
        # Application must implement routing logic:
        # - Random selection based on percentage
        # - User-based routing (e.g., internal users to canary)
        # - Geographic routing
        
        return {
            'stable_index': stable_index,
            'canary_index': canary_index,
            'canary_percentage': canary_percentage,
            'routing': 'application_level'
        }
```

---

## Index Migration

### Migration Script

```python
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from typing import List, Dict
import time

class IndexMigration:
    """Migrate data between search indexes."""
    
    def __init__(self, search_endpoint: str):
        self.credential = DefaultAzureCredential()
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=self.credential
        )
        self.endpoint = search_endpoint
    
    def migrate_index(
        self,
        source_index_name: str,
        target_index_name: str,
        batch_size: int = 1000,
        transform_func = None
    ) -> dict:
        """
        Migrate documents from source to target index.
        
        Args:
            source_index_name: Source index
            target_index_name: Target index
            batch_size: Documents per batch
            transform_func: Optional function to transform documents
            
        Returns:
            Migration summary
        """
        source_client = SearchClient(
            endpoint=self.endpoint,
            index_name=source_index_name,
            credential=self.credential
        )
        
        target_client = SearchClient(
            endpoint=self.endpoint,
            index_name=target_index_name,
            credential=self.credential
        )
        
        # Get total document count
        count_result = source_client.search(
            search_text="*",
            include_total_count=True,
            top=0
        )
        total_docs = count_result.get_count()
        
        print(f"Migrating {total_docs} documents from {source_index_name} to {target_index_name}")
        
        migrated = 0
        errors = []
        
        # Fetch and migrate in batches
        offset = 0
        while offset < total_docs:
            # Fetch batch
            results = source_client.search(
                search_text="*",
                top=batch_size,
                skip=offset
            )
            
            batch = []
            for result in results:
                doc = dict(result)
                
                # Apply transformation if provided
                if transform_func:
                    doc = transform_func(doc)
                
                batch.append(doc)
            
            if not batch:
                break
            
            # Upload batch to target
            try:
                upload_result = target_client.upload_documents(documents=batch)
                migrated += len(batch)
                print(f"Migrated {migrated}/{total_docs} documents")
            except Exception as e:
                errors.append({
                    'batch_offset': offset,
                    'error': str(e)
                })
                print(f"Error migrating batch at offset {offset}: {e}")
            
            offset += batch_size
            time.sleep(0.1)  # Rate limiting
        
        return {
            'total_documents': total_docs,
            'migrated': migrated,
            'errors': len(errors),
            'error_details': errors
        }
    
    def reindex_with_schema_change(
        self,
        old_index_name: str,
        new_index_definition: SearchIndex,
        field_mapping: Dict[str, str] = None
    ):
        """
        Reindex with schema changes.
        
        Args:
            old_index_name: Existing index
            new_index_definition: New index schema
            field_mapping: Map old field names to new names
        """
        new_index_name = new_index_definition.name
        
        # Create new index
        self.index_client.create_index(new_index_definition)
        print(f"Created new index: {new_index_name}")
        
        # Define transformation function
        def transform_document(doc: dict) -> dict:
            if field_mapping:
                transformed = {}
                for old_field, new_field in field_mapping.items():
                    if old_field in doc:
                        transformed[new_field] = doc[old_field]
                # Copy non-mapped fields
                for key, value in doc.items():
                    if key not in field_mapping and key in [f.name for f in new_index_definition.fields]:
                        transformed[key] = value
                return transformed
            return doc
        
        # Migrate data
        result = self.migrate_index(
            source_index_name=old_index_name,
            target_index_name=new_index_name,
            transform_func=transform_document
        )
        
        print(f"Reindexing complete: {result['migrated']} documents migrated")
        
        return result
```

---

## Secret Management

### Azure Key Vault Integration

```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

class SecretManager:
    """Manage secrets with Azure Key Vault."""
    
    def __init__(self, vault_url: str):
        self.credential = DefaultAzureCredential()
        self.client = SecretClient(
            vault_url=vault_url,
            credential=self.credential
        )
    
    def get_search_admin_key(self, search_service_name: str) -> str:
        """Get search service admin key from Key Vault."""
        secret_name = f"{search_service_name}-admin-key"
        secret = self.client.get_secret(secret_name)
        return secret.value
    
    def rotate_search_key(
        self,
        search_service_name: str,
        new_key: str
    ):
        """Rotate search service admin key."""
        secret_name = f"{search_service_name}-admin-key"
        self.client.set_secret(secret_name, new_key)
        print(f"Rotated key for {search_service_name}")
    
    def get_connection_string(self, resource_name: str) -> str:
        """Get connection string from Key Vault."""
        secret_name = f"{resource_name}-connection-string"
        secret = self.client.get_secret(secret_name)
        return secret.value
```

### Environment-specific Configuration

```python
# config/config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    
    environment: str
    search_endpoint: str
    search_sku: str
    replica_count: int
    partition_count: int
    key_vault_url: str
    storage_account: str
    application_insights_key: str
    enable_private_endpoint: bool
    
    @classmethod
    def from_environment(cls, env: str) -> 'EnvironmentConfig':
        """Load configuration for environment."""
        
        configs = {
            'dev': cls(
                environment='dev',
                search_endpoint=os.getenv('SEARCH_ENDPOINT_DEV'),
                search_sku='basic',
                replica_count=1,
                partition_count=1,
                key_vault_url=os.getenv('KEYVAULT_URL_DEV'),
                storage_account=os.getenv('STORAGE_ACCOUNT_DEV'),
                application_insights_key=os.getenv('APPINSIGHTS_KEY_DEV'),
                enable_private_endpoint=False
            ),
            'staging': cls(
                environment='staging',
                search_endpoint=os.getenv('SEARCH_ENDPOINT_STAGING'),
                search_sku='standard',
                replica_count=2,
                partition_count=1,
                key_vault_url=os.getenv('KEYVAULT_URL_STAGING'),
                storage_account=os.getenv('STORAGE_ACCOUNT_STAGING'),
                application_insights_key=os.getenv('APPINSIGHTS_KEY_STAGING'),
                enable_private_endpoint=False
            ),
            'prod': cls(
                environment='prod',
                search_endpoint=os.getenv('SEARCH_ENDPOINT_PROD'),
                search_sku='standard',
                replica_count=3,
                partition_count=2,
                key_vault_url=os.getenv('KEYVAULT_URL_PROD'),
                storage_account=os.getenv('STORAGE_ACCOUNT_PROD'),
                application_insights_key=os.getenv('APPINSIGHTS_KEY_PROD'),
                enable_private_endpoint=True
            )
        }
        
        return configs.get(env)
```

---

## Testing Gates

### Smoke Tests

```python
# tests/smoke_tests.py
import argparse
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.identity import DefaultAzureCredential
import sys

class SmokeTests:
    """Essential smoke tests for search service."""
    
    def __init__(self, endpoint: str):
        self.credential = DefaultAzureCredential()
        self.endpoint = endpoint
        self.index_client = SearchIndexClient(
            endpoint=endpoint,
            credential=self.credential
        )
    
    def run_all_tests(self) -> bool:
        """Run all smoke tests."""
        tests = [
            ('Service Connectivity', self.test_service_connectivity),
            ('List Indexes', self.test_list_indexes),
            ('Search Query', self.test_search_query),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                test_func()
                print(f"✓ {test_name}: PASSED")
                passed += 1
            except Exception as e:
                print(f"✗ {test_name}: FAILED - {e}")
                failed += 1
        
        print(f"\nResults: {passed} passed, {failed} failed")
        return failed == 0
    
    def test_service_connectivity(self):
        """Test basic service connectivity."""
        service_stats = self.index_client.get_service_statistics()
        assert service_stats is not None
    
    def test_list_indexes(self):
        """Test ability to list indexes."""
        indexes = list(self.index_client.list_indexes())
        assert isinstance(indexes, list)
    
    def test_search_query(self):
        """Test basic search query."""
        # Get first index
        indexes = list(self.index_client.list_indexes())
        if not indexes:
            print("  ⚠ No indexes to test")
            return
        
        index_name = indexes[0].name
        search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=index_name,
            credential=self.credential
        )
        
        # Execute simple query
        results = search_client.search(search_text="*", top=1)
        result_list = list(results)
        assert isinstance(result_list, list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', required=True)
    args = parser.parse_args()
    
    tests = SmokeTests(args.endpoint)
    success = tests.run_all_tests()
    
    sys.exit(0 if success else 1)
```

---

## Rollback Procedures

### Automated Rollback

```python
class RollbackManager:
    """Manage deployment rollbacks."""
    
    def __init__(self, search_endpoint: str):
        self.endpoint = search_endpoint
        self.credential = DefaultAzureCredential()
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=self.credential
        )
    
    def create_snapshot(self, index_name: str) -> str:
        """
        Create snapshot of index configuration.
        
        Returns:
            Snapshot ID
        """
        import json
        from datetime import datetime
        
        # Get index definition
        index = self.index_client.get_index(index_name)
        
        snapshot_id = f"{index_name}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        # Save snapshot (to blob storage in production)
        snapshot = {
            'snapshot_id': snapshot_id,
            'index_name': index_name,
            'timestamp': datetime.utcnow().isoformat(),
            'index_definition': index.as_dict()
        }
        
        # Store snapshot
        print(f"Created snapshot: {snapshot_id}")
        
        return snapshot_id
    
    def rollback_to_snapshot(self, snapshot_id: str):
        """Rollback index to snapshot."""
        # Load snapshot
        # Recreate index from snapshot
        # Migrate data if needed
        
        print(f"Rolling back to snapshot: {snapshot_id}")
    
    def validate_rollback(self, index_name: str) -> bool:
        """Validate rollback was successful."""
        try:
            # Check index exists
            index = self.index_client.get_index(index_name)
            
            # Run validation queries
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=index_name,
                credential=self.credential
            )
            
            results = search_client.search(search_text="*", top=1)
            list(results)  # Force execution
            
            return True
        except Exception as e:
            print(f"Rollback validation failed: {e}")
            return False
```

---

## Best Practices

### 1. Version Control

- Store all infrastructure as code in Git
- Use feature branches for changes
- Require pull request reviews
- Tag releases with semantic versioning

### 2. Environment Strategy

- Maintain separate environments: Dev → Staging → Production
- Use environment-specific configuration files
- Never deploy directly to production
- Test in staging with production-like data

### 3. Deployment Safety

- Always validate templates before deployment
- Use blue-green or canary deployments for production
- Implement automated rollback triggers
- Maintain previous index versions for quick rollback

### 4. Secret Management

- Never commit secrets to source control
- Use Azure Key Vault for all secrets
- Rotate keys regularly
- Use managed identities when possible

### 5. Testing

- Run unit tests on every commit
- Execute smoke tests after deployment
- Perform integration tests in staging
- Load test before production deployment

### 6. Monitoring

- Set up deployment notifications
- Monitor deployment metrics
- Track deployment success/failure rates
- Alert on failed deployments

### 7. Documentation

- Document deployment procedures
- Maintain runbooks for common issues
- Track breaking changes
- Document rollback procedures

---

For monitoring deployed services, see [Monitoring & Alerting (Page 23)](./23-monitoring-alerting.md).

For security configurations in CI/CD, see [Security & Compliance (Page 21)](./21-security-compliance.md).
