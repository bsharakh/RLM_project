#!/usr/bin/env python3
"""Simple demo script for RLM system."""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rlm import RLM
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


def demo():
    """Run a simple demonstration."""
    print(f"\n{Fore.BLUE}{'='*80}")
    print(f"{Fore.BLUE}RLM Demo - Boss and Intern Model")
    print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}\n")
    
    # Sample context about a fictional company
    context = """
    DataFlow Inc. is a technology company founded in 2018 by Dr. Sarah Chen and 
    Michael Rodriguez in Seattle, Washington. The company specializes in real-time 
    data analytics platforms for enterprise customers.
    
    In 2020, DataFlow received Series A funding of $15 million led by Benchmark Capital.
    The funding was used to expand their engineering team from 25 to 75 employees.
    
    Their flagship product, DataFlow Analytics Suite, was launched in 2021. It helps
    companies process and analyze streaming data with latency under 100 milliseconds.
    Major clients include Fortune 500 companies in finance, healthcare, and retail.
    
    In 2023, DataFlow opened offices in New York and London. The company now has
    over 200 employees and serves more than 150 enterprise customers worldwide.
    
    The CEO, Dr. Sarah Chen, has a PhD in Computer Science from MIT and previously
    worked at Google on distributed systems. CTO Michael Rodriguez was formerly
    a senior engineer at Amazon Web Services.
    """
    
    # Initialize RLM
    print(f"{Fore.CYAN}Initializing RLM system...{Style.RESET_ALL}")
    try:
        rlm = RLM(max_iterations=5)
        config = rlm.get_config_info()
        print(f"{Fore.GREEN}✓ RLM ready")
        print(f"  Boss: {config['rlm_model']}")
        print(f"  Intern: {config['sublm_model']}{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"{Fore.RED}✗ Error: {e}")
        print(f"{Fore.YELLOW}Make sure your .env file is configured with OPENAI_API_KEY")
        return
    
    # Demo questions
    questions = [
        "Who founded DataFlow Inc.?",
        "How much funding did they receive and when?",
        "What is their main product and what does it do?",
    ]
    
    print(f"{Fore.CYAN}Demo Context:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'='*80}")
    print(context[:200] + "...")
    print(f"{'='*80}{Style.RESET_ALL}\n")
    
    # Ask each question
    for i, question in enumerate(questions, 1):
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}Demo Question {i}/{len(questions)}")
        print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}\n")
        print(f"{Fore.CYAN}Q: {question}{Style.RESET_ALL}\n")
        
        answer = rlm.answer(question, context)
        
        print(f"\n{Fore.GREEN}{'─'*80}")
        print(f"Final Answer: {answer}")
        print(f"{'─'*80}{Style.RESET_ALL}\n")
        
        input(f"{Fore.YELLOW}Press Enter for next question...{Style.RESET_ALL}")
    
    print(f"\n{Fore.BLUE}{'='*80}")
    print(f"{Fore.BLUE}Demo Complete!")
    print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}\n")
    print(f"Check the {Fore.CYAN}logs/{Style.RESET_ALL} directory for detailed iteration logs.\n")


if __name__ == "__main__":
    demo()
