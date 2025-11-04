#!/usr/bin/env python3
"""
Automated Evaluation Pipeline for Search Indexing Strategies

This script provides a complete automation framework for evaluating and comparing
different search indexing strategies with comprehensive metrics and reporting.

Usage:
    python evaluation_pipeline.py --config config.yaml --dataset dataset.json --output results/
"""

import argparse
import json
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """Complete evaluation pipeline for search indexing strategies."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.results = {}
        self.start_time = time.time()
        
        logger.info(f"Initialized evaluation pipeline with config: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load evaluation dataset."""
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded dataset with {len(dataset)} queries from {dataset_path}")
        return dataset
    
    def setup_engines(self) -> Dict[str, Any]:
        """Set up all search engines for evaluation."""
        engines = {}
        
        for engine_name, engine_config in self.config['engines'].items():
            logger.info(f"Setting up {engine_name} engine...")
            
            if engine_config['type'] == 'elasticsearch':
                from src.engines.keyword_engine import KeywordSearchEngine
                engines[engine_name] = KeywordSearchEngine(engine_config)
                
            elif engine_config['type'] == 'vector':
                from src.engines.vector_engine import VectorSearchEngine
                engines[engine_name] = VectorSearchEngine(engine_config)
                
            elif engine_config['type'] == 'hybrid':
                from src.engines.hybrid_engine import HybridSearchEngine
                engines[engine_name] = HybridSearchEngine(engine_config)
                
            elif engine_config['type'] == 'azure':
                from src.engines.azure_search_engine import AzureSearchEngine
                engines[engine_name] = AzureSearchEngine(engine_config)
            
            else:
                raise ValueError(f"Unknown engine type: {engine_config['type']}")
        
        return engines
    
    def index_documents(self, engines: Dict[str, Any], documents: List[Dict[str, Any]]):
        """Index documents in all engines."""
        logger.info(f"Indexing {len(documents)} documents across {len(engines)} engines...")
        
        for engine_name, engine in engines.items():
            start_time = time.time()
            logger.info(f"Indexing documents in {engine_name}...")
            
            try:
                engine.index_documents(documents)
                index_time = time.time() - start_time
                
                logger.info(f"Successfully indexed documents in {engine_name} ({index_time:.2f}s)")
                
                # Store indexing metrics
                if 'indexing_metrics' not in self.results:
                    self.results['indexing_metrics'] = {}
                
                self.results['indexing_metrics'][engine_name] = {
                    'index_time_seconds': index_time,
                    'documents_indexed': len(documents),
                    'indexing_rate_docs_per_second': len(documents) / index_time
                }
                
            except Exception as e:
                logger.error(f"Failed to index documents in {engine_name}: {e}")
                raise
    
    def run_evaluation(self, engines: Dict[str, Any], dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive evaluation across all engines."""
        logger.info(f"Starting evaluation with {len(dataset)} queries across {len(engines)} engines...")
        
        from src.evaluation.evaluator import SearchEvaluator
        evaluator = SearchEvaluator(self.config['evaluation']['metrics'])
        
        evaluation_results = {}
        
        for engine_name, engine in engines.items():
            logger.info(f"Evaluating {engine_name}...")
            
            try:
                start_time = time.time()
                result = evaluator.evaluate_engine(engine, dataset)
                evaluation_time = time.time() - start_time
                
                result['evaluation_time_seconds'] = evaluation_time
                evaluation_results[engine_name] = result
                
                # Log key metrics
                metrics = result['metrics']
                logger.info(f"{engine_name} results:")
                logger.info(f"  Precision@5: {metrics.get('precision_at_5', 0):.3f}")
                logger.info(f"  Recall@10: {metrics.get('recall_at_10', 0):.3f}")
                logger.info(f"  MAP: {metrics.get('map', 0):.3f}")
                logger.info(f"  MRR: {metrics.get('mrr', 0):.3f}")
                logger.info(f"  Avg Latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
                
            except Exception as e:
                logger.error(f"Evaluation failed for {engine_name}: {e}")
                evaluation_results[engine_name] = {'error': str(e)}
        
        return evaluation_results
    
    def run_load_testing(self, engines: Dict[str, Any], test_queries: List[str]) -> Dict[str, Any]:
        """Run load testing for performance evaluation."""
        if not self.config.get('load_testing', {}).get('enabled', False):
            logger.info("Load testing disabled in configuration")
            return {}
        
        load_config = self.config['load_testing']
        logger.info(f"Running load test with {load_config['concurrent_users']} concurrent users...")
        
        load_results = {}
        
        for engine_name, engine in engines.items():
            logger.info(f"Load testing {engine_name}...")
            
            try:
                load_result = self._run_engine_load_test(
                    engine, 
                    test_queries,
                    load_config['concurrent_users'],
                    load_config['duration_minutes']
                )
                load_results[engine_name] = load_result
                
                logger.info(f"{engine_name} load test results:")
                logger.info(f"  Requests/sec: {load_result.get('requests_per_second', 0):.1f}")
                logger.info(f"  Avg latency: {load_result.get('avg_response_time_ms', 0):.1f}ms")
                logger.info(f"  Error rate: {load_result.get('error_rate', 0):.3f}")
                
            except Exception as e:
                logger.error(f"Load testing failed for {engine_name}: {e}")
                load_results[engine_name] = {'error': str(e)}
        
        return load_results
    
    def _run_engine_load_test(self, engine, queries: List[str], concurrent_users: int, duration_minutes: int) -> Dict[str, Any]:
        """Run load test for a single engine."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def worker():
            """Worker thread for load testing."""
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            worker_stats = {
                'response_times': [],
                'errors': [],
                'requests': 0
            }
            
            while time.time() < end_time:
                query = queries[worker_stats['requests'] % len(queries)]
                
                try:
                    request_start = time.time()
                    engine.search(query, top_k=10)
                    response_time = (time.time() - request_start) * 1000
                    
                    worker_stats['response_times'].append(response_time)
                    worker_stats['requests'] += 1
                    
                except Exception as e:
                    worker_stats['errors'].append(str(e))
                
                time.sleep(0.1)  # Small delay
            
            results_queue.put(worker_stats)
        
        # Start worker threads
        threads = []
        for _ in range(concurrent_users):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Aggregate results
        all_response_times = []
        all_errors = []
        total_requests = 0
        
        while not results_queue.empty():
            worker_result = results_queue.get()
            all_response_times.extend(worker_result['response_times'])
            all_errors.extend(worker_result['errors'])
            total_requests += worker_result['requests']
        
        # Calculate statistics
        if all_response_times:
            avg_response_time = sum(all_response_times) / len(all_response_times)
            p95_response_time = sorted(all_response_times)[int(0.95 * len(all_response_times))]
            p99_response_time = sorted(all_response_times)[int(0.99 * len(all_response_times))]
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
        
        return {
            'total_requests': total_requests,
            'error_count': len(all_errors),
            'error_rate': len(all_errors) / total_requests if total_requests > 0 else 0,
            'avg_response_time_ms': avg_response_time,
            'p95_response_time_ms': p95_response_time,
            'p99_response_time_ms': p99_response_time,
            'requests_per_second': total_requests / (duration_minutes * 60),
            'concurrent_users': concurrent_users,
            'duration_minutes': duration_minutes
        }
    
    def generate_comparison_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        logger.info("Generating comparison report...")
        
        # Extract key metrics for comparison
        comparison_metrics = ['precision_at_5', 'recall_at_10', 'map', 'mrr', 'avg_latency_ms']
        
        comparison = {}
        for metric in comparison_metrics:
            metric_values = {}
            
            for engine_name, result in evaluation_results.items():
                if 'metrics' in result and metric in result['metrics']:
                    metric_values[engine_name] = result['metrics'][metric]
            
            if metric_values:
                # Find best performing engine for this metric
                if metric == 'avg_latency_ms':  # Lower is better for latency
                    best_engine = min(metric_values.items(), key=lambda x: x[1])
                else:  # Higher is better for quality metrics
                    best_engine = max(metric_values.items(), key=lambda x: x[1])
                
                comparison[metric] = {
                    'values': metric_values,
                    'best_engine': best_engine[0],
                    'best_value': best_engine[1],
                    'improvement_over_baseline': self._calculate_improvement(metric_values)
                }
        
        return comparison
    
    def _calculate_improvement(self, metric_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate improvement percentages over baseline."""
        if not metric_values:
            return {}
        
        # Use first engine as baseline, or find 'keyword'/'baseline' engine
        baseline_engine = None
        for engine_name in ['keyword', 'baseline', 'bm25']:
            if engine_name in metric_values:
                baseline_engine = engine_name
                break
        
        if not baseline_engine:
            baseline_engine = list(metric_values.keys())[0]
        
        baseline_value = metric_values[baseline_engine]
        
        improvements = {}
        for engine_name, value in metric_values.items():
            if baseline_value > 0:
                improvement = ((value - baseline_value) / baseline_value) * 100
                improvements[engine_name] = improvement
            else:
                improvements[engine_name] = 0.0
        
        return improvements
    
    def export_results(self, output_dir: str):
        """Export all results to structured files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main results file
        results_file = output_path / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results exported to {results_file}")
        
        # Generate summary report
        summary_file = output_path / f"summary_report_{timestamp}.md"
        self._generate_markdown_report(summary_file)
        
        logger.info(f"Summary report generated: {summary_file}")
        
        # Export metrics CSV for analysis
        if 'evaluation_results' in self.results:
            csv_file = output_path / f"metrics_comparison_{timestamp}.csv"
            self._export_metrics_csv(csv_file)
            logger.info(f"Metrics CSV exported: {csv_file}")
    
    def _generate_markdown_report(self, output_file: Path):
        """Generate markdown summary report."""
        report_content = f"""# Search Indexing Strategy Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Evaluation Time:** {(time.time() - self.start_time) / 60:.1f} minutes

## Executive Summary

This report compares different search indexing strategies across multiple dimensions including relevance, performance, and resource usage.

"""
        
        # Add evaluation results summary
        if 'evaluation_results' in self.results:
            report_content += "## Evaluation Results\n\n"
            
            for engine_name, result in self.results['evaluation_results'].items():
                if 'metrics' in result:
                    metrics = result['metrics']
                    report_content += f"### {engine_name.title()} Engine\n\n"
                    report_content += f"- **Precision@5:** {metrics.get('precision_at_5', 0):.3f}\n"
                    report_content += f"- **Recall@10:** {metrics.get('recall_at_10', 0):.3f}\n"
                    report_content += f"- **MAP:** {metrics.get('map', 0):.3f}\n"
                    report_content += f"- **MRR:** {metrics.get('mrr', 0):.3f}\n"
                    report_content += f"- **Avg Latency:** {metrics.get('avg_latency_ms', 0):.1f}ms\n"
                    
                    if 'engine_stats' in result:
                        stats = result['engine_stats']
                        if 'total_docs' in stats:
                            report_content += f"- **Documents Indexed:** {stats['total_docs']:,}\n"
                    
                    report_content += "\n"
        
        # Add comparison summary
        if 'comparison_report' in self.results:
            report_content += "## Performance Comparison\n\n"
            comparison = self.results['comparison_report']
            
            for metric, data in comparison.items():
                report_content += f"### {metric.replace('_', ' ').title()}\n\n"
                report_content += f"**Best Performance:** {data['best_engine']} ({data['best_value']:.3f})\n\n"
                
                report_content += "| Engine | Value | Improvement |\n"
                report_content += "|--------|-------|-------------|\n"
                
                for engine, value in data['values'].items():
                    improvement = data['improvement_over_baseline'].get(engine, 0)
                    report_content += f"| {engine} | {value:.3f} | {improvement:+.1f}% |\n"
                
                report_content += "\n"
        
        # Add load testing results
        if 'load_testing_results' in self.results:
            report_content += "## Load Testing Results\n\n"
            
            for engine_name, result in self.results['load_testing_results'].items():
                if 'error' not in result:
                    report_content += f"### {engine_name.title()}\n\n"
                    report_content += f"- **Requests/sec:** {result.get('requests_per_second', 0):.1f}\n"
                    report_content += f"- **Avg Response Time:** {result.get('avg_response_time_ms', 0):.1f}ms\n"
                    report_content += f"- **P95 Response Time:** {result.get('p95_response_time_ms', 0):.1f}ms\n"
                    report_content += f"- **Error Rate:** {result.get('error_rate', 0):.3f}\n\n"
        
        # Add recommendations
        report_content += self._generate_recommendations()
        
        with open(output_file, 'w') as f:
            f.write(report_content)
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on results."""
        recommendations = """## Recommendations

Based on the evaluation results, here are our recommendations:

"""
        
        if 'comparison_report' in self.results:
            comparison = self.results['comparison_report']
            
            # Find overall best engine
            score_weights = {
                'precision_at_5': 0.3,
                'recall_at_10': 0.2,
                'map': 0.2,
                'mrr': 0.2,
                'avg_latency_ms': -0.1  # Negative weight (lower is better)
            }
            
            engine_scores = {}
            for engine_name in self.results.get('evaluation_results', {}).keys():
                score = 0
                for metric, weight in score_weights.items():
                    if metric in comparison and engine_name in comparison[metric]['values']:
                        value = comparison[metric]['values'][engine_name]
                        # Normalize latency (invert since lower is better)
                        if metric == 'avg_latency_ms':
                            normalized_value = 1000 / (value + 1)  # Avoid division by zero
                        else:
                            normalized_value = value
                        score += weight * normalized_value
                
                engine_scores[engine_name] = score
            
            if engine_scores:
                best_overall = max(engine_scores.items(), key=lambda x: x[1])
                recommendations += f"### Primary Recommendation: {best_overall[0].title()}\n\n"
                recommendations += f"The {best_overall[0]} engine shows the best overall performance across all metrics.\n\n"
            
            # Specific use case recommendations
            recommendations += "### Use Case Specific Recommendations:\n\n"
            
            if 'precision_at_5' in comparison:
                best_precision = comparison['precision_at_5']['best_engine']
                recommendations += f"- **High Precision Requirements:** Use {best_precision} for applications requiring accurate top results\n"
            
            if 'recall_at_10' in comparison:
                best_recall = comparison['recall_at_10']['best_engine']
                recommendations += f"- **Comprehensive Coverage:** Use {best_recall} for applications requiring high recall\n"
            
            if 'avg_latency_ms' in comparison:
                fastest_engine = min(comparison['avg_latency_ms']['values'].items(), key=lambda x: x[1])[0]
                recommendations += f"- **Low Latency Requirements:** Use {fastest_engine} for real-time applications\n"
        
        recommendations += "\n### Next Steps:\n\n"
        recommendations += "1. Implement A/B testing with the recommended strategy\n"
        recommendations += "2. Monitor key metrics in production\n"
        recommendations += "3. Iterate based on user feedback and business metrics\n"
        recommendations += "4. Consider hybrid approaches for optimal performance\n"
        
        return recommendations
    
    def _export_metrics_csv(self, output_file: Path):
        """Export metrics to CSV for further analysis."""
        import csv
        
        if 'evaluation_results' not in self.results:
            return
        
        # Collect all metrics
        all_metrics = set()
        for result in self.results['evaluation_results'].values():
            if 'metrics' in result:
                all_metrics.update(result['metrics'].keys())
        
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['engine'] + sorted(all_metrics)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for engine_name, result in self.results['evaluation_results'].items():
                if 'metrics' in result:
                    row = {'engine': engine_name}
                    row.update(result['metrics'])
                    writer.writerow(row)
    
    def run_complete_evaluation(self, dataset_path: str, documents_path: str, output_dir: str):
        """Run complete evaluation pipeline."""
        try:
            # Load data
            dataset = self.load_dataset(dataset_path)
            
            with open(documents_path, 'r') as f:
                documents = json.load(f)
            
            logger.info(f"Loaded {len(documents)} documents for indexing")
            
            # Set up engines
            engines = self.setup_engines()
            
            # Index documents
            self.index_documents(engines, documents)
            
            # Run evaluation
            evaluation_results = self.run_evaluation(engines, dataset)
            self.results['evaluation_results'] = evaluation_results
            
            # Generate comparison report
            comparison_report = self.generate_comparison_report(evaluation_results)
            self.results['comparison_report'] = comparison_report
            
            # Run load testing if enabled
            test_queries = [item['query'] for item in dataset[:50]]  # Use subset for load testing
            load_results = self.run_load_testing(engines, test_queries)
            if load_results:
                self.results['load_testing_results'] = load_results
            
            # Add metadata
            self.results['metadata'] = {
                'config_file': self.config,
                'dataset_size': len(dataset),
                'documents_count': len(documents),
                'engines_evaluated': list(engines.keys()),
                'evaluation_duration_seconds': time.time() - self.start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Export results
            self.export_results(output_dir)
            
            logger.info("Evaluation pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {e}")
            raise

def main():
    """Main entry point for the evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Automated Search Indexing Strategy Evaluation')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--dataset', required=True, help='Evaluation dataset path')
    parser.add_argument('--documents', required=True, help='Documents to index path')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input files
    for file_path in [args.config, args.dataset, args.documents]:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return 1
    
    try:
        # Initialize and run pipeline
        pipeline = EvaluationPipeline(args.config)
        pipeline.run_complete_evaluation(args.dataset, args.documents, args.output)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())