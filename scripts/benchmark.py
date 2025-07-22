#!/usr/bin/env python3
"""
Performance Benchmark Script for Hebrew Content Intelligence Service.
Tests performance under various loads and scenarios.
"""

import sys
import asyncio
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import json

import httpx
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings


class HebrewIntelligenceBenchmark:
    """Performance benchmark suite for Hebrew Content Intelligence Service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client: httpx.AsyncClient = None
        self.results: Dict[str, Any] = {}
        
        # Test data
        self.test_texts = [
            "זהו טקסט בדיקה בעברית",
            "עיצוב מטבח מודרני עם פתרונות אחסון חכמים ופונקציונליים",
            "בית יפה עם חדרים גדולים ונוף מרהיב לים התיכון",
            "טכנולוגיה מתקדמת לעסקים קטנים ובינוניים בישראל",
            "שירותי ייעוץ עסקי מקצועיים עם התמחות בחדשנות דיגיטלית",
            "מסעדה איטלקית אותנטית במרכז תל אביב עם אווירה רומנטית",
            "קורס פיתוח תוכנה מתקדם עם דגש על בינה מלאכותית ולמידת מכונה",
            "חנות בגדים מעצבים עם קולקציות בלעדיות ומחירים אטרקטיביים"
        ]
        
        self.test_keywords = [
            "עיצוב מטבח", "בית יפה", "טכנולוגיה מתקדמת", "ייעוץ עסקי",
            "מסעדה איטלקית", "פיתוח תוכנה", "חנות בגדים", "בינה מלאכותית",
            "פתרונות אחסון", "חדשנות דיגיטלית", "קולקציות בלעדיות", "אווירה רומנטית"
        ]
    
    async def setup(self):
        """Initialize benchmark environment."""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20)
        )
        
        # Wait for service to be ready
        await self.wait_for_service()
        logger.info("Benchmark environment initialized")
    
    async def cleanup(self):
        """Clean up benchmark environment."""
        if self.client:
            await self.client.aclose()
    
    async def wait_for_service(self, max_retries: int = 30, delay: float = 2.0):
        """Wait for the service to be ready."""
        for attempt in range(max_retries):
            try:
                response = await self.client.get("/health")
                if response.status_code == 200:
                    logger.info("Service is ready")
                    return
            except Exception as e:
                logger.debug(f"Service not ready (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
        
        raise RuntimeError("Service failed to become ready within timeout")
    
    async def benchmark_single_analysis(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark single content analysis requests."""
        logger.info(f"Benchmarking single analysis ({iterations} iterations)")
        
        times = []
        errors = 0
        
        for i in range(iterations):
            text = self.test_texts[i % len(self.test_texts)]
            
            start_time = time.time()
            try:
                response = await self.client.post("/analysis/analyze", json={
                    "text": text,
                    "options": {
                        "include_roots": True,
                        "include_morphology": True,
                        "include_keywords": True,
                        "include_themes": True
                    }
                })
                
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                else:
                    errors += 1
                    logger.warning(f"Request failed with status {response.status_code}")
                    
            except Exception as e:
                errors += 1
                logger.error(f"Request error: {e}")
        
        if times:
            return {
                "test_name": "single_analysis",
                "iterations": iterations,
                "successful_requests": len(times),
                "failed_requests": errors,
                "avg_response_time": statistics.mean(times),
                "median_response_time": statistics.median(times),
                "min_response_time": min(times),
                "max_response_time": max(times),
                "p95_response_time": self._percentile(times, 95),
                "p99_response_time": self._percentile(times, 99),
                "requests_per_second": len(times) / sum(times) if sum(times) > 0 else 0
            }
        else:
            return {
                "test_name": "single_analysis",
                "iterations": iterations,
                "successful_requests": 0,
                "failed_requests": errors,
                "error": "All requests failed"
            }
    
    async def benchmark_batch_analysis(self, batch_sizes: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """Benchmark batch content analysis requests."""
        logger.info(f"Benchmarking batch analysis (batch sizes: {batch_sizes})")
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            texts = (self.test_texts * ((batch_size // len(self.test_texts)) + 1))[:batch_size]
            
            times = []
            errors = 0
            
            # Run 10 iterations for each batch size
            for _ in range(10):
                start_time = time.time()
                try:
                    response = await self.client.post("/analysis/batch", json={
                        "texts": texts,
                        "options": {
                            "include_roots": True,
                            "include_keywords": True
                        }
                    })
                    
                    if response.status_code == 200:
                        elapsed = time.time() - start_time
                        times.append(elapsed)
                    else:
                        errors += 1
                        
                except Exception as e:
                    errors += 1
                    logger.error(f"Batch request error: {e}")
            
            if times:
                results[f"batch_size_{batch_size}"] = {
                    "batch_size": batch_size,
                    "iterations": 10,
                    "successful_requests": len(times),
                    "failed_requests": errors,
                    "avg_response_time": statistics.mean(times),
                    "median_response_time": statistics.median(times),
                    "avg_time_per_text": statistics.mean(times) / batch_size,
                    "texts_per_second": (batch_size * len(times)) / sum(times) if sum(times) > 0 else 0
                }
        
        return {
            "test_name": "batch_analysis",
            "results": results
        }
    
    async def benchmark_concurrent_requests(self, concurrency_levels: List[int] = [5, 10, 20, 50]) -> Dict[str, Any]:
        """Benchmark concurrent request handling."""
        logger.info(f"Benchmarking concurrent requests (levels: {concurrency_levels})")
        
        results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")
            
            async def make_request(text: str) -> float:
                start_time = time.time()
                try:
                    response = await self.client.post("/analysis/analyze", json={
                        "text": text,
                        "options": {"include_keywords": True}
                    })
                    
                    if response.status_code == 200:
                        return time.time() - start_time
                    else:
                        return -1  # Error indicator
                except Exception:
                    return -1  # Error indicator
            
            # Create concurrent tasks
            texts = (self.test_texts * ((concurrency // len(self.test_texts)) + 1))[:concurrency]
            tasks = [make_request(text) for text in texts]
            
            start_time = time.time()
            response_times = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Filter successful requests
            successful_times = [t for t in response_times if isinstance(t, float) and t > 0]
            errors = len(response_times) - len(successful_times)
            
            if successful_times:
                results[f"concurrency_{concurrency}"] = {
                    "concurrency_level": concurrency,
                    "total_requests": concurrency,
                    "successful_requests": len(successful_times),
                    "failed_requests": errors,
                    "total_time": total_time,
                    "avg_response_time": statistics.mean(successful_times),
                    "median_response_time": statistics.median(successful_times),
                    "max_response_time": max(successful_times),
                    "requests_per_second": len(successful_times) / total_time,
                    "success_rate": len(successful_times) / concurrency
                }
        
        return {
            "test_name": "concurrent_requests",
            "results": results
        }
    
    async def benchmark_keyword_operations(self, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark keyword expansion and search volume operations."""
        logger.info(f"Benchmarking keyword operations ({iterations} iterations)")
        
        expansion_times = []
        volume_times = []
        expansion_errors = 0
        volume_errors = 0
        
        for i in range(iterations):
            keywords = self.test_keywords[i % len(self.test_keywords):i % len(self.test_keywords) + 3]
            
            # Test keyword expansion
            start_time = time.time()
            try:
                response = await self.client.post("/keywords/expand", json={
                    "keywords": keywords,
                    "options": {
                        "include_morphological": True,
                        "include_semantic": True,
                        "max_variations_per_type": 5
                    }
                })
                
                if response.status_code == 200:
                    expansion_times.append(time.time() - start_time)
                else:
                    expansion_errors += 1
                    
            except Exception as e:
                expansion_errors += 1
                logger.error(f"Keyword expansion error: {e}")
            
            # Test search volume
            start_time = time.time()
            try:
                response = await self.client.post("/search-volume", json={
                    "keywords": keywords,
                    "location": "Israel",
                    "language": "he"
                })
                
                if response.status_code == 200:
                    volume_times.append(time.time() - start_time)
                else:
                    volume_errors += 1
                    
            except Exception as e:
                volume_errors += 1
                logger.error(f"Search volume error: {e}")
        
        return {
            "test_name": "keyword_operations",
            "keyword_expansion": {
                "iterations": iterations,
                "successful_requests": len(expansion_times),
                "failed_requests": expansion_errors,
                "avg_response_time": statistics.mean(expansion_times) if expansion_times else 0,
                "median_response_time": statistics.median(expansion_times) if expansion_times else 0
            },
            "search_volume": {
                "iterations": iterations,
                "successful_requests": len(volume_times),
                "failed_requests": volume_errors,
                "avg_response_time": statistics.mean(volume_times) if volume_times else 0,
                "median_response_time": statistics.median(volume_times) if volume_times else 0
            }
        }
    
    async def benchmark_caching_performance(self, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark caching performance."""
        logger.info(f"Benchmarking caching performance ({iterations} iterations)")
        
        test_text = self.test_texts[0]
        
        # First request (cache miss)
        start_time = time.time()
        response = await self.client.post("/analysis/analyze", json={
            "text": test_text,
            "options": {"include_keywords": True}
        })
        cache_miss_time = time.time() - start_time
        
        if response.status_code != 200:
            return {"test_name": "caching_performance", "error": "Initial request failed"}
        
        # Subsequent requests (cache hits)
        cache_hit_times = []
        errors = 0
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                response = await self.client.post("/analysis/analyze", json={
                    "text": test_text,
                    "options": {"include_keywords": True}
                })
                
                if response.status_code == 200:
                    cache_hit_times.append(time.time() - start_time)
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
                logger.error(f"Cache hit request error: {e}")
        
        if cache_hit_times:
            avg_cache_hit_time = statistics.mean(cache_hit_times)
            speedup = cache_miss_time / avg_cache_hit_time if avg_cache_hit_time > 0 else 0
            
            return {
                "test_name": "caching_performance",
                "cache_miss_time": cache_miss_time,
                "cache_hit_avg_time": avg_cache_hit_time,
                "cache_hit_median_time": statistics.median(cache_hit_times),
                "speedup_factor": speedup,
                "successful_cache_hits": len(cache_hit_times),
                "failed_requests": errors
            }
        else:
            return {
                "test_name": "caching_performance",
                "error": "All cache hit requests failed"
            }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        logger.info("Starting comprehensive benchmark suite")
        
        start_time = time.time()
        
        # Run all benchmarks
        benchmarks = [
            self.benchmark_single_analysis(100),
            self.benchmark_batch_analysis([5, 10, 20]),
            self.benchmark_concurrent_requests([5, 10, 20]),
            self.benchmark_keyword_operations(30),
            self.benchmark_caching_performance(30)
        ]
        
        results = await asyncio.gather(*benchmarks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Process results
        benchmark_results = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Benchmark {i} failed: {result}")
                benchmark_results[f"benchmark_{i}"] = {"error": str(result)}
            else:
                benchmark_results[result["test_name"]] = result
        
        return {
            "benchmark_suite": "Hebrew Content Intelligence Service",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_time,
            "service_url": self.base_url,
            "results": benchmark_results
        }
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable benchmark report."""
        report = []
        report.append("=" * 80)
        report.append("HEBREW CONTENT INTELLIGENCE SERVICE - BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Service URL: {results['service_url']}")
        report.append(f"Total Duration: {results['total_duration']:.2f} seconds")
        report.append("")
        
        for test_name, test_results in results["results"].items():
            if "error" in test_results:
                report.append(f"❌ {test_name.upper()}: FAILED")
                report.append(f"   Error: {test_results['error']}")
                report.append("")
                continue
            
            report.append(f"✅ {test_name.upper()}")
            report.append("-" * 40)
            
            if test_name == "single_analysis":
                report.append(f"   Iterations: {test_results['iterations']}")
                report.append(f"   Success Rate: {test_results['successful_requests']}/{test_results['iterations']}")
                report.append(f"   Avg Response Time: {test_results['avg_response_time']:.3f}s")
                report.append(f"   Median Response Time: {test_results['median_response_time']:.3f}s")
                report.append(f"   P95 Response Time: {test_results['p95_response_time']:.3f}s")
                report.append(f"   Requests/Second: {test_results['requests_per_second']:.2f}")
                
            elif test_name == "caching_performance":
                report.append(f"   Cache Miss Time: {test_results['cache_miss_time']:.3f}s")
                report.append(f"   Cache Hit Avg Time: {test_results['cache_hit_avg_time']:.3f}s")
                report.append(f"   Speedup Factor: {test_results['speedup_factor']:.1f}x")
                
            elif test_name == "concurrent_requests":
                for level, data in test_results["results"].items():
                    report.append(f"   {level}:")
                    report.append(f"     Success Rate: {data['success_rate']:.2%}")
                    report.append(f"     Avg Response Time: {data['avg_response_time']:.3f}s")
                    report.append(f"     Requests/Second: {data['requests_per_second']:.2f}")
            
            report.append("")
        
        return "\n".join(report)


async def main():
    """Main benchmark function."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )
    
    logger.info("Hebrew Content Intelligence Service - Performance Benchmark")
    logger.info("=" * 70)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Hebrew Content Intelligence Service")
    parser.add_argument("--url", default="http://localhost:8000", help="Service URL")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--report", help="Output file for human-readable report")
    args = parser.parse_args()
    
    benchmark = HebrewIntelligenceBenchmark(args.url)
    
    try:
        await benchmark.setup()
        results = await benchmark.run_all_benchmarks()
        
        # Save JSON results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {args.output}")
        
        # Generate and save report
        report = benchmark.generate_report(results)
        
        if args.report:
            with open(args.report, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {args.report}")
        else:
            print("\n" + report)
        
        logger.success("Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)
    
    finally:
        await benchmark.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
