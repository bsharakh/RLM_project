#!/usr/bin/env python3
"""
Demo showing INTERMEDIATE ANSWERS feature.

This shows how the RLM progressively builds knowledge with each iteration,
giving you an answer at each step that gets better and better.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rlm import RLM
from colorama import Fore, Style, init

init(autoreset=True)


def demo_intermediate_answers():
    """Demonstrate intermediate answer generation."""
    print(f"\n{Fore.BLUE}{'='*80}")
    print(f"{Fore.BLUE}RLM Demo - Intermediate Answers Feature")
    print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}\n")
    
    print("This demo shows how the RLM gives you progressive answers:")
    print("- After iteration 1: Initial answer based on first findings")
    print("- After iteration 2: Updated answer with more details")
    print("- After iteration 3: Refined answer")
    print("- Final: Complete synthesized answer\n")
    
    context = """
    TechVision Corp was founded in March 2019 by Emma Thompson and David Park.
    Emma Thompson previously worked as VP of Engineering at Meta for 8 years.
    David Park was a Product Manager at Google for 6 years before founding TechVision.
    
    The company is headquartered in Austin, Texas, with additional offices in 
    San Francisco and Boston. They currently employ 145 people across all locations.
    
    TechVision's main product is CloudSync Pro, a cloud-based collaboration platform
    that integrates with Slack, Microsoft Teams, and Google Workspace. The platform
    specializes in real-time document collaboration with AI-powered suggestions.
    
    In Series A funding (July 2020), they raised $12 million led by Andreessen Horowitz.
    In Series B funding (March 2022), they raised $35 million led by Sequoia Capital.
    The company is currently valued at approximately $200 million.
    
    Their main competitors are Notion, Coda, and Airtable. TechVision differentiates
    itself through superior AI integration and enterprise-grade security features.
    """
    
    # Initialize RLM
    try:
        print(f"{Fore.CYAN}Initializing RLM...{Style.RESET_ALL}")
        rlm = RLM(max_iterations=4)  # Use 4 iterations to show progression
        print(f"{Fore.GREEN}✓ Ready\n{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error: {e}")
        print(f"{Fore.YELLOW}Make sure .env is configured with OPENAI_API_KEY")
        return
    
    # Ask a multi-faceted question
    question = "Tell me about TechVision's funding history and who invested in them"
    
    print(f"{Fore.CYAN}Question:{Style.RESET_ALL} {question}\n")
    print(f"{Fore.YELLOW}Watch how the answer improves with each iteration!{Style.RESET_ALL}\n")
    print(f"{Fore.WHITE}{'='*80}{Style.RESET_ALL}\n")
    
    # This will now show intermediate answers at each iteration
    answer = rlm.answer(question, context)
    
    print(f"\n{Fore.BLUE}{'='*80}")
    print(f"Summary:{Style.RESET_ALL}")
    print("You saw the RLM:")
    print("1. Ask the intern a question")
    print("2. Get an answer")
    print("3. Generate intermediate answer ('what we know so far')")
    print("4. Ask a follow-up question for more details")
    print("5. Repeat until complete")
    print(f"\n{Fore.GREEN}Check the logs/ folder for full JSON output!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    demo_intermediate_answers()