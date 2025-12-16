#!/usr/bin/env python3
"""
Demo showing METRICS feature.

This shows how each intermediate and final answer is evaluated on:
- Completeness (does it answer all parts?)
- Accuracy (is it factually correct?)
- Confidence (how certain is it?)
- Relevance (is it on-topic?)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rlm import RLM
from colorama import Fore, Style, init

init(autoreset=True)


def demo_metrics():
    """Demonstrate metrics evaluation."""
    print(f"\n{Fore.BLUE}{'='*80}")
    print(f"{Fore.BLUE}RLM Demo - Answer Quality Metrics")
    print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}\n")
    
    print("This demo shows how the system evaluates answer quality:")
    print(f"{Fore.CYAN}📊 Metrics Tracked:{Style.RESET_ALL}")
    print("  • Completeness: Does it answer all parts of the question?")
    print("  • Accuracy: Is it factually correct based on context?")
    print("  • Confidence: How certain/definitive is the answer?")
    print("  • Relevance: Is it on-topic and addressing the question?")
    print("  • Overall: Weighted combination of all metrics\n")
    
    context = """
    QuantumLeap Technologies was founded in January 2020 by Dr. Lisa Martinez 
    and Alex Kim in Boston, Massachusetts. Dr. Martinez has a PhD in Quantum 
    Computing from MIT and previously led Google's quantum research division. 
    Alex Kim was CTO at IBM's quantum computing unit for 5 years.
    
    The company focuses on developing practical quantum computing solutions 
    for drug discovery and materials science. Their main product, QuantumSim, 
    is a quantum simulation platform that helps pharmaceutical companies model 
    molecular interactions 1000x faster than classical computers.
    
    Funding rounds:
    - Seed round (June 2020): $3.5 million from Y Combinator
    - Series A (March 2021): $18 million led by Sequoia Capital
    - Series B (November 2022): $75 million led by Andreessen Horowitz
    
    The company currently employs 85 people across offices in Boston, San Francisco, 
    and London. Major clients include Pfizer, Moderna, and Johnson & Johnson.
    
    In 2023, they announced a partnership with MIT to establish a joint quantum 
    computing research lab. The company's technology has been cited in over 50 
    peer-reviewed papers and has filed 12 patents in quantum simulation.
    """
    
    # Initialize RLM with metrics enabled
    try:
        print(f"{Fore.CYAN}Initializing RLM with metrics enabled...{Style.RESET_ALL}")
        rlm = RLM(max_iterations=3, enable_metrics=True)
        print(f"{Fore.GREEN}✓ Ready\n{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error: {e}")
        print(f"{Fore.YELLOW}Make sure .env is configured with OPENAI_API_KEY")
        return
    
    # Ask a complex question
    question = "What are QuantumLeap's funding rounds and who were the lead investors?"
    
    print(f"{Fore.CYAN}Question:{Style.RESET_ALL} {question}\n")
    print(f"{Fore.YELLOW}Watch how metrics improve with each iteration!{Style.RESET_ALL}\n")
    print(f"{Fore.WHITE}{'='*80}{Style.RESET_ALL}\n")
    
    # This will now show metrics at each iteration
    answer = rlm.answer(question, context)
    
    print(f"\n{Fore.BLUE}{'='*80}")
    print(f"What You Just Saw:{Style.RESET_ALL}")
    print("1. Boss asks intern a question")
    print("2. Intern answers from context")
    print("3. Intermediate answer generated")
    print("4. 📊 Metrics evaluated for intermediate answer")
    print("5. Repeat until complete")
    print("6. 📊 Final metrics show overall answer quality")
    print(f"\n{Fore.CYAN}Metrics help you understand:{Style.RESET_ALL}")
    print("  • How well each iteration improves the answer")
    print("  • Whether the final answer is complete and accurate")
    print("  • If you need more iterations for better quality")
    print(f"\n{Fore.GREEN}Check logs/ for full metrics in JSON format!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    demo_metrics()
