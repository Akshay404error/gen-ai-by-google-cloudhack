import time
import json
import statistics
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from app import generate_ai_test_cases, generate_mock_test_cases

class PerformanceTester:
    def __init__(self):
        self.results = []
        
    def run_test(self, 
                test_cases: List[Dict[str, Any]], 
                model: str = "llama-3.1-8b-instant",
                temperature: float = 0.2,
                max_tokens: int = 2000,
                use_ai: bool = True,
                test_name: str = "default"):
        """Run performance test on a set of test cases."""
        test_results = {
            'test_name': test_name,
            'model': model,
            'total_cases': len(test_cases),
            'response_times': [],
            'successful_runs': 0,
            'failed_runs': 0,
            'avg_response_time': 0,
            'min_response_time': float('inf'),
            'max_response_time': 0,
            'tokens_generated': 0,
            'test_cases': []
        }
        
        for idx, test_case in enumerate(test_cases, 1):
            user_story = test_case.get('user_story', '')
            domain = test_case.get('domain', 'general')
            count = test_case.get('count', 5)
            
            print(f"\nRunning test case {idx}/{len(test_cases)}: {user_story[:50]}...")
            
            try:
                start_time = time.time()
                
                if use_ai:
                    result = generate_ai_test_cases(
                        user_story=user_story,
                        domain=domain,
                        count=count,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                else:
                    result = generate_mock_test_cases(
                        user_story=user_story,
                        domain=domain,
                        count=count
                    )
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Calculate tokens (approximate)
                tokens = len(user_story.split()) + sum(
                    len(str(tc).split()) 
                    for tc in result
                )
                
                test_results['tokens_generated'] += tokens
                test_results['response_times'].append(duration)
                test_results['successful_runs'] += 1
                
                if duration < test_results['min_response_time']:
                    test_results['min_response_time'] = duration
                if duration > test_results['max_response_time']:
                    test_results['max_response_time'] = duration
                
                test_results['test_cases'].append({
                    'user_story': user_story,
                    'duration': duration,
                    'tokens': tokens,
                    'success': True
                })
                
                print(f"✅ Success - {duration:.2f}s - {tokens} tokens")
                
            except Exception as e:
                test_results['failed_runs'] += 1
                test_results['test_cases'].append({
                    'user_story': user_story,
                    'error': str(e),
                    'success': False
                })
                print(f"❌ Failed - {str(e)}")
        
        # Calculate statistics
        if test_results['response_times']:
            test_results['avg_response_time'] = statistics.mean(test_results['response_times'])
            test_results['p95_response_time'] = statistics.quantiles(test_results['response_times'], n=20)[-1] if len(test_results['response_times']) > 1 else test_results['avg_response_time']
            total_time = sum(test_results['response_times'])
            test_results['throughput'] = len(test_results['response_times']) / total_time if total_time > 0 else float('inf')
        
        self.results.append(test_results)
        return test_results
    
    def generate_report(self, output_file: str = "performance_report.md"):
        """Generate a markdown report of the performance test results."""
        if not self.results:
            return "No test results to report."
        
        report = "# AI Test Case Generator Performance Report\n\n"
        
        # Summary Table
        report += "## 📊 Test Summary\n\n"
        report += "| Test Name | Model | Total Cases | Success | Failed | Avg Time (s) | Min Time (s) | Max Time (s) | Throughput (req/s) |\n"
        report += "|-----------|-------|-------------|---------|--------|--------------|--------------|--------------|-------------------|\n"
        
        for result in self.results:
            report += f"| {result['test_name']} | {result['model']} | {result['total_cases']} | {result['successful_runs']} | {result['failed_runs']} | {result['avg_response_time']:.2f} | {result['min_response_time']:.2f} | {result['max_response_time']:.2f} | {result.get('throughput', 0):.2f} |\n"
        
        # Detailed Results
        report += "\n## 📝 Detailed Results\n\n"
        
        for result in self.results:
            report += f"### 🚀 {result['test_name']} ({result['model']})\n"
            report += f"- **Total Test Cases:** {result['total_cases']}\n"
            report += f"- **Successful Runs:** {result['successful_runs']}\n"
            report += f"- **Failed Runs:** {result['failed_runs']}\n"
            report += f"- **Average Response Time:** {result['avg_response_time']:.2f}s\n"
            report += f"- **95th Percentile Response Time:** {result.get('p95_response_time', 0):.2f}s\n"
            report += f"- **Throughput:** {result.get('throughput', 0):.2f} requests/second\n"
            report += f"- **Total Tokens Generated:** {result['tokens_generated']:,}\n\n"
            
            # Response Time Distribution
            if len(result['response_times']) > 1:
                plt.figure(figsize=(10, 4))
                plt.hist(result['response_times'], bins=10, alpha=0.7, color='skyblue')
                plt.title(f"Response Time Distribution - {result['test_name']}")
                plt.xlabel('Response Time (seconds)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                plot_filename = f"response_time_{result['test_name'].lower().replace(' ', '_')}.png"
                plt.savefig(plot_filename, bbox_inches='tight')
                plt.close()
                
                report += f"![Response Time Distribution]({plot_filename})\n\n"
            
            # Failed Cases
            if result['failed_runs'] > 0:
                report += "#### ❌ Failed Test Cases\n"
                for case in result['test_cases']:
                    if not case.get('success', False):
                        report += f"- **User Story:** {case['user_story']}\n"
                        if 'error' in case:
                            report += f"  - **Error:** {case['error']}\n"
                        report += "\n"
        
        # Save the report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report

# Example test cases
TEST_CASES = [
    {
        'user_story': 'As a user, I want to log in to the system with my email and password',
        'domain': 'general',
        'count': 5
    },
    {
        'user_story': 'The system should monitor CPU temperature and trigger cooling when it exceeds 80°C',
        'domain': 'thermal',
        'count': 5
    },
    {
        'user_story': 'The device should safely shut down if the temperature reaches critical levels',
        'domain': 'safety',
        'count': 3
    },
    {
        'user_story': 'The system should read temperature from I2C sensor and log it every second',
        'domain': 'embedded',
        'count': 4
    }
]

def main():
    try:
        tester = PerformanceTester()
        
        # Test with AI
        print("\n🚀 Starting AI performance test...")
        ai_result = tester.run_test(
            test_cases=TEST_CASES,
            model="llama-3.1-8b-instant",
            use_ai=True,
            test_name="AI Generation Test"
        )
        
        # Test with mock data
        print("\n🔧 Starting mock performance test...")
        mock_result = tester.run_test(
            test_cases=TEST_CASES,
            use_ai=False,
            test_name="Mock Generation Test"
        )
        
        # Generate report
        print("\n📊 Generating performance report...")
        report = tester.generate_report()
        print(f"✅ Performance report generated: performance_report.md")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    main()
