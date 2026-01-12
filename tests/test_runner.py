#!/usr/bin/env python3
"""Test runner for RLM system with Easy/Hard mode selection."""
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


def select_difficulty():
    """Prompt user to select difficulty level."""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}RLM Test Runner - Select Difficulty Mode")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
    
    print(f"{Fore.GREEN}[1] Easy Mode{Style.RESET_ALL} - Straightforward questions with clear answers")
    print(f"    • Simple factual lookups")
    print(f"    • Context: ~500-1000 words")
    print(f"    • Expected iterations: 2-3")
    print(f"    • Test file: test_questions.json\n")
    
    print(f"{Fore.RED}[2] Hard Mode{Style.RESET_ALL} - Complex questions with scattered information")
    print(f"    • Multi-step reasoning required")
    print(f"    • Context: 2,000-12,000 words")
    print(f"    • Expected iterations: 5-7")
    print(f"    • Test file: test_questions_challenging.json\n")
    
    while True:
        try:
            choice = input(f"{Fore.YELLOW}Select mode [1 or 2]: {Style.RESET_ALL}").strip()
            if choice in ['1', '2']:
                return choice
            else:
                print(f"{Fore.RED}Invalid choice. Please enter 1 or 2.{Style.RESET_ALL}")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Test cancelled.{Style.RESET_ALL}")
            sys.exit(0)


def load_test_questions(test_file="tests/test_questions.json"):
    """Load test questions from JSON file."""
    with open(test_file, 'r') as f:
        return json.load(f)


def run_test_set(rlm, test_set, difficulty):
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
        
        # Show context length for hard mode
        if difficulty == 'hard':
            context_words = len(q['context'].split())
            print(f"{Fore.MAGENTA}Context size: {context_words:,} words{Style.RESET_ALL}")
        
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
        
        # Show comparison
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}RESULTS COMPARISON")
        print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}\n")
        
        print(f"{Fore.YELLOW}Expected Answer:{Style.RESET_ALL}")
        print(f"{q.get('expected_answer', 'N/A')}\n")
        
        print(f"{Fore.GREEN}RLM's Answer:{Style.RESET_ALL}")
        print(f"{answer}\n")
        
        # Show if they match
        if answer.strip() == q.get('expected_answer', '').strip():
            print(f"{Fore.GREEN}✓ EXACT MATCH!{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠ Different (not necessarily wrong - check manually){Style.RESET_ALL}")
        
        print(f"\n{Fore.BLUE}{'-'*80}{Style.RESET_ALL}")
    
    return results


def run_tests_with_mode(mode):
    """Run tests based on selected mode."""
    
    # Determine test file and difficulty label
    if mode == '1':
        test_file = "tests/test_questions.json"
        difficulty_label = "EASY MODE"
        difficulty_color = Fore.GREEN
        difficulty = "easy"
    else:
        test_file = "tests/test_questions_challenging.json"
        difficulty_label = "HARD MODE"
        difficulty_color = Fore.RED
        difficulty = "hard"
    
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print(f"{Fore.MAGENTA}RLM Test Runner - {difficulty_color}{difficulty_label}{Fore.MAGENTA}")
    print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}\n")
    
    # Initialize RLM
    try:
        rlm = RLM()
        config = rlm.get_config_info()
        print(f"{Fore.GREEN}✓ RLM initialized")
        print(f"  Boss Model: {config['rlm_model']}")
        print(f"  Intern Model: {config['sublm_model']}")
        print(f"  Max Iterations: {config['max_iterations']}")
        print(f"  Max Depth: {config['max_depth']}")
        print(f"  Metrics Enabled: {config.get('metrics_enabled', 'N/A')}\n")
    except Exception as e:
        print(f"{Fore.RED}✗ Error initializing RLM: {e}")
        print(f"{Fore.YELLOW}Make sure you've set up your .env file with OPENAI_API_KEY")
        return
    
    # Load test questions
    try:
        test_data = load_test_questions(test_file)
        print(f"{Fore.GREEN}✓ Loaded test file: {test_file}{Style.RESET_ALL}\n")
    except FileNotFoundError:
        print(f"{Fore.RED}✗ Error: Test file not found: {test_file}")
        print(f"{Fore.YELLOW}Make sure the file exists in the tests/ directory")
        return
    except Exception as e:
        print(f"{Fore.RED}✗ Error loading test questions: {e}")
        return
    
    # Run each test set
    all_results = []
    for test_set in test_data['test_sets']:
        results = run_test_set(rlm, test_set, difficulty)
        all_results.extend(results)
    
    # Print summary
    print(f"\n\n{Fore.MAGENTA}{'='*80}")
    print(f"{Fore.MAGENTA}Test Summary - {difficulty_color}{difficulty_label}{Fore.MAGENTA}")
    print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}\n")
    print(f"Total questions tested: {len(all_results)}")
    print(f"\nLogs saved to: {Fore.CYAN}logs/{Style.RESET_ALL}")
    
    if difficulty == "hard":
        print(f"\n{Fore.YELLOW}💡 Tip for Hard Mode:{Style.RESET_ALL}")
        print(f"  Check the logs to see how the boss breaks down complex questions")
        print(f"  and builds answers iteratively across multiple rounds.")
    
    print(f"\n{Fore.GREEN}All tests completed!{Style.RESET_ALL}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run RLM tests with difficulty selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py              # Interactive mode selection
  python test_runner.py --mode easy  # Run easy mode directly
  python test_runner.py --mode hard  # Run hard mode directly
  python test_runner.py --file tests/custom_tests.json  # Custom file
        """
    )
    parser.add_argument(
        '--mode',
        choices=['easy', 'hard'],
        help='Difficulty mode (skip interactive selection)'
    )
    parser.add_argument(
        '--file',
        help='Path to custom test questions JSON file (overrides mode selection)'
    )
    
    args = parser.parse_args()
    
    # If custom file specified, run it directly
    if args.file:
        print(f"\n{Fore.CYAN}Running custom test file: {args.file}{Style.RESET_ALL}")
        test_file = args.file
        difficulty = "custom"
        
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}RLM Test Runner - CUSTOM MODE")
        print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}\n")
        
        try:
            rlm = RLM()
            config = rlm.get_config_info()
            print(f"{Fore.GREEN}✓ RLM initialized")
            print(f"  Boss Model: {config['rlm_model']}")
            print(f"  Intern Model: {config['sublm_model']}")
            print(f"  Max Iterations: {config['max_iterations']}\n")
        except Exception as e:
            print(f"{Fore.RED}✗ Error initializing RLM: {e}")
            return
        
        try:
            test_data = load_test_questions(test_file)
            print(f"{Fore.GREEN}✓ Loaded test file: {test_file}{Style.RESET_ALL}\n")
        except Exception as e:
            print(f"{Fore.RED}✗ Error loading test questions: {e}")
            return
        
        all_results = []
        for test_set in test_data['test_sets']:
            results = run_test_set(rlm, test_set, difficulty)
            all_results.extend(results)
        
        print(f"\n\n{Fore.MAGENTA}Test Summary{Style.RESET_ALL}")
        print(f"Total questions: {len(all_results)}")
        print(f"Logs: {Fore.CYAN}logs/{Style.RESET_ALL}\n")
        return
    
    # If mode specified via command line, use it
    if args.mode:
        mode = '1' if args.mode == 'easy' else '2'
    else:
        # Interactive mode selection
        mode = select_difficulty()
    
    # Run tests with selected mode
    run_tests_with_mode(mode)


if __name__ == "__main__":
    main()