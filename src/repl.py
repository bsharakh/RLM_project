#!/usr/bin/env python3
"""Interactive REPL for RLM system."""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rlm import RLM
from src.config import Config
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class RLMRepl:
    """Interactive REPL for RLM system."""
    
    def __init__(self):
        """Initialize REPL."""
        self.rlm = None
        self.context = ""
        self.running = True
        
    def start(self):
        """Start the REPL."""
        self.print_welcome()
        
        # Initialize RLM
        try:
            self.rlm = RLM()
            config = self.rlm.get_config_info()
            print(f"{Fore.GREEN}✓ RLM initialized")
            print(f"  Boss Model: {config['rlm_model']}")
            print(f"  Intern Model: {config['sublm_model']}")
            print(f"  Max Iterations: {config['max_iterations']}")
            print(f"  Max Depth: {config['max_depth']}\n")
        except Exception as e:
            print(f"{Fore.RED}✗ Error initializing RLM: {e}")
            print(f"{Fore.YELLOW}Make sure you've set up your .env file with OPENAI_API_KEY")
            return
        
        # Main REPL loop
        while self.running:
            try:
                user_input = input(f"{Fore.CYAN}> {Style.RESET_ALL}").strip()
                
                if not user_input:
                    continue
                
                self.handle_command(user_input)
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use 'exit' or 'quit' to leave the REPL")
            except EOFError:
                break
        
        print(f"\n{Fore.BLUE}Goodbye!")
    
    def handle_command(self, user_input):
        """Handle user commands."""
        # Check for special commands
        if user_input.lower() in ['exit', 'quit', 'q']:
            self.running = False
            return
        
        if user_input.lower() in ['help', 'h', '?']:
            self.print_help()
            return
        
        if user_input.lower() == 'clear':
            self.context = ""
            print(f"{Fore.GREEN}✓ Context cleared")
            return
        
        if user_input.lower() == 'show':
            self.show_context()
            return
        
        if user_input.lower() == 'config':
            self.show_config()
            return
        
        # Check for context assignment
        if user_input.startswith('context =') or user_input.startswith('context='):
            self.set_context(user_input)
            return
        
        # Check for question
        if user_input.startswith('ask(') or user_input.startswith('ask ('):
            self.ask_question(user_input)
            return
        
        # Default: treat as setting context
        print(f"{Fore.YELLOW}Tip: Use 'context = \"your text\"' to set context or 'ask(\"question\")' to ask")
    
    def set_context(self, user_input):
        """Set the context variable."""
        try:
            # Extract the value after '='
            value = user_input.split('=', 1)[1].strip()
            
            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            
            self.context = value
            print(f"{Fore.GREEN}✓ Context set ({len(self.context)} characters)")
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error setting context: {e}")
    
    def ask_question(self, user_input):
        """Ask a question using the RLM."""
        if not self.context:
            print(f"{Fore.RED}✗ No context set. Use 'context = \"your text\"' first")
            return
        
        try:
            # Extract question from ask("question")
            question = user_input.split('ask', 1)[1].strip()
            question = question.strip('()')
            
            # Remove quotes if present
            if (question.startswith('"') and question.endswith('"')) or \
               (question.startswith("'") and question.endswith("'")):
                question = question[1:-1]
            
            if not question:
                print(f"{Fore.RED}✗ Please provide a question")
                return
            
            # Ask the RLM
            print(f"\n{Fore.BLUE}Processing question...{Style.RESET_ALL}\n")
            answer = self.rlm.answer(question, self.context)
            
            print(f"\n{Fore.GREEN}{'='*80}")
            print(f"FINAL ANSWER")
            print(f"{'='*80}{Style.RESET_ALL}")
            print(answer)
            print()
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error: {e}")
    
    def show_context(self):
        """Show current context."""
        if not self.context:
            print(f"{Fore.YELLOW}No context set")
            return
        
        print(f"\n{Fore.CYAN}Current Context:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'='*80}")
        preview = self.context[:500] + "..." if len(self.context) > 500 else self.context
        print(preview)
        print(f"{'='*80}")
        print(f"Total length: {len(self.context)} characters\n")
    
    def show_config(self):
        """Show current configuration."""
        config = self.rlm.get_config_info()
        print(f"\n{Fore.CYAN}Current Configuration:{Style.RESET_ALL}")
        print(f"  Boss Model: {config['rlm_model']}")
        print(f"  Intern Model: {config['sublm_model']}")
        print(f"  Max Iterations: {config['max_iterations']}")
        print(f"  Max Depth: {config['max_depth']}")
        print(f"  Logging: {'Enabled' if Config.ENABLE_LOGGING else 'Disabled'}")
        print()
    
    def print_welcome(self):
        """Print welcome message."""
        print(f"\n{Fore.BLUE}{'='*80}")
        print(f"{Fore.BLUE}RLM (Recursive Language Model) REPL")
        print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}\n")
        print("Welcome to the RLM interactive environment!")
        print(f"Type {Fore.CYAN}'help'{Style.RESET_ALL} for available commands\n")
    
    def print_help(self):
        """Print help message."""
        print(f"\n{Fore.CYAN}Available Commands:{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}context = \"text\"{Style.RESET_ALL}  - Set the context/knowledge base")
        print(f"  {Fore.YELLOW}ask(\"question\"){Style.RESET_ALL}   - Ask a question about the context")
        print(f"  {Fore.YELLOW}show{Style.RESET_ALL}              - Show current context")
        print(f"  {Fore.YELLOW}clear{Style.RESET_ALL}             - Clear the context")
        print(f"  {Fore.YELLOW}config{Style.RESET_ALL}            - Show current configuration")
        print(f"  {Fore.YELLOW}help{Style.RESET_ALL}              - Show this help message")
        print(f"  {Fore.YELLOW}exit{Style.RESET_ALL}              - Exit the REPL\n")
        
        print(f"{Fore.CYAN}Example Usage:{Style.RESET_ALL}")
        print(f'  > context = "Python is a programming language created by Guido van Rossum."')
        print(f'  > ask("Who created Python?")')
        print()


def main():
    """Main entry point."""
    repl = RLMRepl()
    repl.start()


if __name__ == "__main__":
    main()
