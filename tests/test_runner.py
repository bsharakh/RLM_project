#!/usr/bin/env python3
"""Test runner for RLM system."""
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rlm import RLM
from src.config import Config
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


def load_test_questions(test_file="tests/test_questions.json"):
    """Load test questions from JSON file."""
    with open(test_file, 'r') as f:
        return json.load(f)


def run_test_set(rlm, test_set):
    """Run a single test set."""
    print(f"\n{Fore.BLUE}{'='*80}")
    print(f"{Fore.BLUE}Test Set: {test_set['name']}")
    print(f"{Fore.BLUE}Description: {test_set['description']}")
    print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}\n")
    
    results = []
    
    for q in test_set['questions']:
        print(f"\n{Fore.CYAN}Question ID: {q['id']}")
        print(f"Question: {q['question']}{Style.RESET_ALL}")
        
        if 'expected_answer' in q:
            print(f"{Fore.YELLOW}Expected: {q['expected_answer']}{Style.RESET_ALL}")
        
        print(f"\n{Fore.WHITE}Processing...{Style.RESET_ALL}\n")
        
        # Ask the question
        answer = rlm.answer(q['question'], q['context'])
        
        # Store result
        results.append({
            'id': q['id'],
            'question': q['question'],
            'answer': answer,
            'expected': q.get('expected_answer', 'N/A')
        })
        
        print(f"\n{Fore.GREEN}Actual Answer:{Style.RESET_ALL}")
        print(answer)
        print(f"\n{Fore.BLUE}{'-'*80}{Style.RESET_ALL}")
    
    return results


def run_all_tests(test_file="tests/test_questions.json"):
    """Run all test sets."""
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print(f"{Fore.MAGENTA}RLM Test Runner")
    print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}\n")
    
    # Initialize RLM
    try:
        rlm = RLM()
        config = rlm.get_config_info()
        print(f"{Fore.GREEN}✓ RLM initialized")
        print(f"  Boss Model: {config['rlm_model']}")
        print(f"  Intern Model: {config['sublm_model']}")
        print(f"  Max Iterations: {config['max_iterations']}")
        print(f"  Max Depth: {config['max_depth']}\n")
    except Exception as e:
        print(f"{Fore.RED}✗ Error initializing RLM: {e}")
        print(f"{Fore.YELLOW}Make sure you've set up your .env file with OPENAI_API_KEY")
        return
    
    # Load test questions
    try:
        test_data = load_test_questions(test_file)
    except Exception as e:
        print(f"{Fore.RED}✗ Error loading test questions: {e}")
        return
    
    # Run each test set
    all_results = []
    for test_set in test_data['test_sets']:
        results = run_test_set(rlm, test_set)
        all_results.extend(results)
    
    # Print summary
    print(f"\n\n{Fore.MAGENTA}{'='*80}")
    print(f"{Fore.MAGENTA}Test Summary")
    print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}\n")
    print(f"Total questions tested: {len(all_results)}")
    print(f"\nLogs saved to: {Fore.CYAN}logs/{Style.RESET_ALL}")
    print(f"\n{Fore.GREEN}All tests completed!{Style.RESET_ALL}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RLM tests')
    parser.add_argument(
        '--file',
        default='tests/test_questions.json',
        help='Path to test questions JSON file'
    )
    
    args = parser.parse_args()
    run_all_tests(args.file)


if __name__ == "__main__":
    main()
