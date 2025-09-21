import time
import statistics
import platform
import os
import psutil
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from app import generate_ai_test_cases, generate_mock_test_cases

# Try to import GPU monitoring libraries
try:
    import pynvml
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("GPU monitoring not available. Install pynvml with: pip install pynvml")

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

class PerformanceTester:
    def __init__(self):
        self.results = []
        self.cpu_usage = []
        self.gpu_usage = []
        self._init_gpu()
    
    def _init_gpu(self):
        """Initialize GPU monitoring if available."""
        self.gpu_handles = []
        if HAS_GPU:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
            except Exception as e:
                print(f"Warning: Could not initialize GPU monitoring: {e}")
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    def _get_gpu_usage(self) -> List[Dict[str, Any]]:
        """Get current GPU usage for all GPUs."""
        if not self.gpu_handles:
            return []
            
        gpu_stats = []
        for i, handle in enumerate(self.gpu_handles):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_stats.append({
                    'gpu_id': i,
                    'gpu_util': util.gpu,
                    'memory_util': (mem.used / mem.total) * 100 if mem.total > 0 else 0,
                    'memory_used_mb': mem.used / (1024 * 1024),
                    'memory_total_mb': mem.total / (1024 * 1024)
                })
            except Exception as e:
                print(f"Error getting GPU {i} stats: {e}")
        return gpu_stats
    
    def _monitor_resources(self, duration: float, interval: float = 0.5):
        """Monitor CPU and GPU usage during a test."""
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            # Record CPU usage
            cpu_usage = self._get_cpu_usage()
            self.cpu_usage.append({
                'timestamp': time.time(),
                'cpu_usage': cpu_usage
            })
            
            # Record GPU usage if available
            if HAS_GPU and self.gpu_handles:
                gpu_stats = self._get_gpu_usage()
                self.gpu_usage.extend([{
                    'timestamp': time.time(),
                    'gpu_id': gpu['gpu_id'],
                    'gpu_util': gpu['gpu_util'],
                    'memory_util': gpu['memory_util'],
                    'memory_used_mb': gpu['memory_used_mb']
                } for gpu in gpu_stats])
            
            time.sleep(interval)
    
    def run_test(self, test_cases: List[Dict[str, Any]], use_ai: bool = True, test_name: str = "Test"):
        """Run performance test on a set of test cases with detailed resource monitoring."""
        print(f"\n{'='*50}")
        print(f"Starting {test_name}")
        print(f"{'='*50}")
        
        test_results = {
            'test_name': test_name,
            'total_cases': len(test_cases),
            'response_times': [],
            'successful_runs': 0,
            'failed_runs': 0,
            'tokens_generated': 0,
            'test_cases': []
        }
        
        for idx, test_case in enumerate(test_cases, 1):
            test_name = test_case.get('name', f"Test {idx}")
            user_story = test_case.get('user_story', '')
            domain = test_case.get('domain', 'general')
            count = test_case.get('count', 2)  # Default to 2 test cases if not specified
            
            print(f"\n[{idx}/{len(test_cases)}] {test_name}")
            print(f"   Domain: {domain}")
            print(f"   Test Cases: {count}")
            print(f"   Description: {user_story[:80]}{'...' if len(user_story) > 80 else ''}")
            
            try:
                # Start resource monitoring in a separate thread
                import threading
                monitor_thread = threading.Thread(
                    target=self._monitor_resources,
                    args=(10, 0.5)  # Monitor for up to 10 seconds with 0.5s intervals
                )
                monitor_thread.daemon = True
                monitor_thread.start()
                
                start_time = time.time()
                
                try:
                    if use_ai:
                        result = generate_ai_test_cases(
                            user_story=user_story,
                            domain=domain,
                            count=count,
                            model="llama-3.1-8b-instant",
                            temperature=0.2,
                            max_tokens=4000,
                            timeout=30,
                            extra_instructions=""
                        )
                    else:
                        result = generate_mock_test_cases(
                            user_story=user_story,
                            domain=domain,
                            count=count
                        )
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    # Give the monitor thread a moment to finish
                    time.sleep(0.5)
                
                # Calculate tokens (approximate)
                tokens = len(user_story.split()) + sum(
                    len(str(tc).split()) 
                    for tc in result
                )
                
                test_results['tokens_generated'] += tokens
                test_results['response_times'].append(duration)
                test_results['successful_runs'] += 1
                
                test_results['test_cases'].append({
                    'user_story': user_story,
                    'duration': duration,
                    'tokens': tokens,
                    'success': True,
                    'cpu_avg': statistics.mean([x['cpu_usage'] for x in self.cpu_usage]) if self.cpu_usage else 0,
                    'gpu_avg': statistics.mean([x['gpu_util'] for x in self.gpu_usage]) if self.gpu_usage else 0,
                    'gpu_mem_avg': statistics.mean([x['memory_util'] for x in self.gpu_usage]) if self.gpu_usage else 0
                })
                
                print(f"✅ Success - {duration:.2f}s - {tokens} tokens")
                if self.cpu_usage:
                    print(f"   CPU Usage: {self.cpu_usage[-1]['cpu_usage']:.1f}%")
                if self.gpu_usage:
                    print(f"   GPU Usage: {self.gpu_usage[-1]['gpu_util']:.1f}%")
                    print(f"   GPU Memory: {self.gpu_usage[-1]['memory_util']:.1f}%")
                
                # Reset monitoring for next test
                self.cpu_usage = []
                self.gpu_usage = []
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Failed - {error_msg}")
                test_results['failed_runs'] += 1
                test_results['test_cases'].append({
                    'user_story': user_story,
                    'error': error_msg,
                    'success': False
                })
        
        # Calculate statistics
        if test_results['response_times']:
            times = test_results['response_times']
            test_results['avg_response_time'] = statistics.mean(times)
            test_results['min_response_time'] = min(times)
            test_results['max_response_time'] = max(times)
            total_time = sum(times)
            test_results['throughput'] = len(times) / total_time if total_time > 0 else float('inf')
            
            if len(times) > 1:
                test_results['std_dev'] = statistics.stdev(times)
                test_results['p95_response_time'] = statistics.quantiles(times, n=20)[-1]
            else:
                test_results['std_dev'] = 0
                test_results['p95_response_time'] = test_results['avg_response_time']
        
        self.results.append(test_results)
        return test_results
    
    def generate_report(self, output_file: str = "performance_report.md"):
        """Generate a markdown report of the performance test results."""
        if not self.results:
            return "No test results to report."
        
        # Create timestamp and environment info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        env_info = f"Python {platform.python_version()}; {platform.system()} {platform.release()}"
        
        # Generate charts
        self._generate_charts()
        
        # Generate report sections
        summary = self._generate_summary()
        ai_details = self._generate_test_details("AI Generation Test")
        mock_details = self._generate_test_details("Mock Generation Test")
        
        # Combine all sections
        report = f"""# 🚀 Test Case Generator - Performance Test Report

**Report Generated:** {timestamp}  
**Environment:** {env_info}

## 📊 Test Summary

{summary}

## 📈 Performance Analysis

### Response Time Comparison

![Response Time Comparison](response_time_comparison.png)

### Throughput Comparison

![Throughput Comparison](throughput_comparison.png)

### Resource Usage

#### CPU Usage Comparison

![CPU Usage Comparison](cpu_usage_comparison.png)

#### GPU Usage Comparison

![GPU Usage Comparison](gpu_usage_comparison.png)

## 📝 Detailed Results

### AI Generation Test

{ai_details}

### Mock Generation Test

{mock_details}

---

*Report generated by AI Test Case Generator Performance Test Suite*
""".format(
            timestamp=timestamp,
            env_info=env_info,
            summary=summary,
            ai_details=ai_details,
            mock_details=mock_details
        )
        
        # Save the report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def _generate_summary(self):
        """Generate the summary section of the report."""
        lines = [
            "| Test Name | Total | Success | Failed | Avg Time (s) | Throughput (req/s) |",
            "|-----------|-------|---------|--------|--------------|-------------------|"
        ]
        
        for result in self.results:
            line = "| {} | {} | {} | {} | {:.3f} | {:.2f} |".format(
                result['test_name'],
                result['total_cases'],
                result['successful_runs'],
                result['failed_runs'],
                result.get('avg_response_time', 0),
                result.get('throughput', 0)
            )
            lines.append(line)
        
        return "\n".join(lines)
    
    def _generate_test_details(self, test_name: str) -> str:
        """Generate details for a specific test."""
        result = next((r for r in self.results if r['test_name'] == test_name), None)
        if not result:
            return f"No data available for {test_name}"
        
        details = [
            f"- **Total Test Cases:** {result['total_cases']}",
            f"- **Successful Runs:** {result['successful_runs']}",
            f"- **Failed Runs:** {result['failed_runs']}",
            f"- **Average Response Time:** {result.get('avg_response_time', 0):.3f}s",
            f"- **Min/Max Response Time:** {result.get('min_response_time', 0):.3f}s / {result.get('max_response_time', 0):.3f}s",
            f"- **Throughput:** {result.get('throughput', 0):.2f} requests/second",
            f"- **Tokens Generated:** {result.get('tokens_generated', 0):,}",
            f"- **Avg CPU Usage:** {statistics.mean([x.get('cpu_avg', 0) for x in result.get('test_cases', []) if x.get('success', False)]):.1f}%" if any(x.get('success', False) for x in result.get('test_cases', [])) else "- **Avg CPU Usage:** N/A",
            f"- **Avg GPU Usage:** {statistics.mean([x.get('gpu_avg', 0) for x in result.get('test_cases', []) if x.get('success', False) and x.get('gpu_avg', 0) > 0]):.1f}%" if any(x.get('gpu_avg', 0) > 0 for x in result.get('test_cases', []) if x.get('success', False)) else "- **GPU Usage:** Not available"
            "",
            "#### Test Cases:"
        ]
        
        for i, case in enumerate(result['test_cases'], 1):
            status = "✅" if case.get('success', False) else "❌"
            user_story = case.get('user_story', 'No description')[:80]
            duration = f"{case.get('duration', 0):.3f}" if 'duration' in case else "N/A"
            tokens = case.get('tokens', 0)
            
            details.append(f"{i}. **{user_story}...**")
            details.append(f"   {status} {duration}s | {tokens} tokens")
            if 'cpu_avg' in case:
                details.append("   💻 CPU: {:.1f}%".format(case['cpu_avg']))
            if 'gpu_avg' in case and case['gpu_avg'] > 0:
                details.append("   🎮 GPU: {:.1f}% | VRAM: {:.1f}%".format(
                    case['gpu_avg'], case.get('gpu_mem_avg', 0)
                ))
            
            if 'error' in case:
                details.append(f"   > Error: {case['error']}")
            
            details.append("")  # Add empty line between test cases
        
        return "\n".join(details)
    
    def _generate_charts(self):
        """Generate comparison charts for the report."""
        ai_result = next((r for r in self.results if 'AI' in r['test_name']), None)
        mock_result = next((r for r in self.results if 'Mock' in r['test_name']), None)
        
        if not (ai_result or mock_result):
            return
        
        # Generate CPU/GPU usage charts if we have the data
        if any('cpu_avg' in case for case in (ai_result or {}).get('test_cases', [])) or \
           any('cpu_avg' in case for case in (mock_result or {}).get('test_cases', [])):
            self._generate_resource_charts(ai_result, mock_result)
        
        # Response Time Comparison
        self._generate_bar_chart(
            data=[
                ai_result.get('avg_response_time', 0) if ai_result else 0,
                mock_result.get('avg_response_time', 0) if mock_result else 0
            ],
            labels=['AI Generation', 'Mock Generation'],
            title='Average Response Time Comparison',
            ylabel='Time (seconds)',
            filename='response_time_comparison.png',
            value_format='.3f'
        )
        
        # Throughput Comparison
        self._generate_bar_chart(
            data=[
                ai_result.get('throughput', 0) if ai_result else 0,
                mock_result.get('throughput', 0) if mock_result else 0
            ],
            labels=['AI Generation', 'Mock Generation'],
            title='Throughput Comparison',
            ylabel='Requests per second',
            filename='throughput_comparison.png',
            value_format='.1f'
        )
    
    def _generate_resource_charts(self, ai_result, mock_result):
        """Generate CPU/GPU usage comparison charts."""
        import numpy as np
        
        # CPU Usage Comparison
        ai_cpu = [case.get('cpu_avg', 0) for case in ai_result.get('test_cases', []) if case.get('success', False)]
        mock_cpu = [case.get('cpu_avg', 0) for case in mock_result.get('test_cases', []) if case.get('success', False)]
        
        if ai_cpu or mock_cpu:
            plt.figure(figsize=(10, 6))
            
            x = np.arange(2)
            width = 0.35
            
            if ai_cpu:
                plt.bar(x - width/2, [sum(ai_cpu)/len(ai_cpu) if ai_cpu else 0, 0], 
                       width, label='AI Generation', color='#4c72b0')
            if mock_cpu:
                plt.bar(x + width/2, [0, sum(mock_cpu)/len(mock_cpu) if mock_cpu else 0], 
                       width, label='Mock Generation', color='#55a868')
            
            plt.title('Average CPU Usage Comparison')
            plt.ylabel('CPU Usage (%)')
            plt.xticks(x, ['AI Generation', 'Mock Generation'])
            plt.legend()
            plt.tight_layout()
            plt.savefig('cpu_usage_comparison.png')
            plt.close()
        
        # GPU Usage Comparison (if available)
        ai_gpu = [case.get('gpu_avg', 0) for case in ai_result.get('test_cases', []) 
                 if case.get('success', False) and case.get('gpu_avg', 0) > 0]
        mock_gpu = [case.get('gpu_avg', 0) for case in mock_result.get('test_cases', []) 
                   if case.get('success', False) and case.get('gpu_avg', 0) > 0]
        
        if ai_gpu or mock_gpu:
            plt.figure(figsize=(10, 6))
            
            x = np.arange(2)
            width = 0.35
            
            if ai_gpu:
                plt.bar(x - width/2, [sum(ai_gpu)/len(ai_gpu) if ai_gpu else 0, 0], 
                       width, label='AI Generation', color='#4c72b0')
            if mock_gpu:
                plt.bar(x + width/2, [0, sum(mock_gpu)/len(mock_gpu) if mock_gpu else 0], 
                       width, label='Mock Generation', color='#55a868')
            
            plt.title('Average GPU Usage Comparison')
            plt.ylabel('GPU Utilization (%)')
            plt.xticks(x, ['AI Generation', 'Mock Generation'])
            plt.legend()
            plt.tight_layout()
            plt.savefig('gpu_usage_comparison.png')
            plt.close()
    
    def _generate_bar_chart(self, data, labels, title, ylabel, filename, value_format):
        """Helper method to generate a bar chart."""
        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, data, color=['#4c72b0', '#55a868'])
        plt.title(title)
        plt.ylabel(ylabel)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, data):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f"{value:{value_format}}",
                ha='center',
                va='bottom'
            )
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

# Comprehensive test scenarios
TEST_CASES = [
    # 1. Basic Functionality
    {
        'name': 'User Authentication',
        'user_story': 'Test login/logout functionality with valid and invalid credentials',
        'domain': 'authentication',
        'count': 5,
        'priority': 'high'
    },
    # 2. Data Operations
    {
        'name': 'Database CRUD',
        'user_story': 'Test Create, Read, Update, and Delete operations on the database',
        'domain': 'database',
        'count': 4,
        'priority': 'high'
    },
    # 3. API Testing
    {
        'name': 'REST API Endpoints',
        'user_story': 'Test all REST API endpoints with valid and invalid payloads',
        'domain': 'api',
        'count': 8,
        'priority': 'critical'
    },
    # 4. Performance Testing
    {
        'name': 'Load Testing',
        'user_story': 'Simulate 1000 concurrent users accessing the system',
        'domain': 'performance',
        'count': 1,
        'priority': 'medium'
    },
    # 5. Security Testing
    {
        'name': 'Security Scans',
        'user_story': 'Perform security vulnerability scanning and penetration testing',
        'domain': 'security',
        'count': 3,
        'priority': 'critical'
    },
    # 6. Integration Testing
    {
        'name': 'Third-party Integrations',
        'user_story': 'Test integration with external services and APIs',
        'domain': 'integration',
        'count': 4,
        'priority': 'high'
    },
    # 7. UI/UX Testing
    {
        'name': 'User Interface',
        'user_story': 'Test user interface elements and user experience flows',
        'domain': 'ui',
        'count': 6,
        'priority': 'medium'
    },
    # 8. Edge Cases
    {
        'name': 'Boundary Testing',
        'user_story': 'Test system behavior with edge cases and boundary values',
        'domain': 'edge_cases',
        'count': 5,
        'priority': 'high'
    }
]

def run_tests(test_cases, test_name_suffix=""):
    tester = PerformanceTester()
    
    # Test with AI
    print(f"\n🚀 Starting AI performance test {test_name_suffix}...")
    ai_result = tester.run_test(
        test_cases=test_cases,
        use_ai=True,
        test_name=f"AI Generation {test_name_suffix}".strip()
    )
    
    # Test with mock data
    print("\n🔧 Starting mock performance test...")
    mock_result = tester.run_test(
        test_cases=test_cases,
        use_ai=False,
        test_name=f"Mock Generation {test_name_suffix}".strip()
    )
    
    return tester

def main():
    try:
        # Run all test cases
        print("\n" + "="*60)
        print("🚀 RUNNING COMPREHENSIVE TEST SUITE")
        print("="*60)
        
        # Run critical tests first
        critical_tests = [tc for tc in TEST_CASES if tc.get('priority') == 'critical']
        if critical_tests:
            print("\n🔴 RUNNING CRITICAL TESTS")
            run_tests(critical_tests, "(Critical)")
        
        # Then run high priority tests
        high_priority = [tc for tc in TEST_CASES if tc.get('priority') == 'high']
        if high_priority:
            print("\n🟠 RUNNING HIGH PRIORITY TESTS")
            run_tests(high_priority, "(High Priority)")
        
        # Then medium priority
        medium_priority = [tc for tc in TEST_CASES if tc.get('priority') == 'medium']
        if medium_priority:
            print("\n🟡 RUNNING MEDIUM PRIORITY TESTS")
            run_tests(medium_priority, "(Medium Priority)")
        
        # Finally, run all tests together
        print("\n🟢 RUNNING FULL TEST SUITE")
        tester = run_tests(TEST_CASES, "(Full Suite)")
        
        # Generate report
        print("\n📊 Generating performance report...")
        report_path = "performance_report.md"
        tester.generate_report(report_path)
        print(f"✅ Performance report generated: {report_path}")
        
        # Clean up old files
        for file in ['performance_test_simple.py', 'performance_test_fixed.py']:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"🗑️ Deleted old file: {file}")
                except Exception as e:
                    print(f"⚠️ Could not delete {file}: {str(e)}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
