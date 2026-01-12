"""Logging system for RLM project."""
import os
import json
from datetime import datetime
from pathlib import Path
from colorama import Fore, Style, init

from src.config import Config

# Initialize colorama
init(autoreset=True)


class RLMLogger:
    """Logger for tracking RLM iterations and conversations."""
    
    def __init__(self, session_id=None):
        """Initialize logger with optional session ID."""
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs = []
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"rlm_session_{self.session_id}.json"
        
    def log_iteration(self, iteration_num, question_type, question, answer, metadata=None):
        """Log a single iteration."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration_num,
            "type": question_type,  # "RLM_TO_SUBLM" or "RLM_TO_USER"
            "question": question,
            "answer": answer,
            "metadata": metadata or {}
        }
        
        self.logs.append(log_entry)
        
        # Console output if enabled
        if Config.ENABLE_LOGGING and Config.LOG_TO_CONSOLE:
            self._print_iteration(log_entry)
        
        # File output if enabled
        if Config.ENABLE_LOGGING and Config.LOG_TO_FILE:
            self._save_to_file()
    
    def _print_iteration(self, log_entry):
        """Print iteration to console with colors."""
        iteration = log_entry["iteration"]
        q_type = log_entry["type"]
        
        if q_type == "RLM_TO_SUBLM":
            print(f"\n{Fore.YELLOW}{'='*80}")
            print(f"{Fore.YELLOW}[ITERATION {iteration}] BOSS → INTERN{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{'='*80}")
            print(f"{Fore.CYAN}Question:{Style.RESET_ALL} {log_entry['question']}")
            print(f"{Fore.GREEN}Answer:{Style.RESET_ALL} {log_entry['answer']}")
        elif q_type == "RLM_TO_REPL":
            print(f"\n{Fore.YELLOW}{'='*80}")
            print(f"{Fore.YELLOW}[ITERATION {iteration}] BOSS → REPL{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{'='*80}")
            print(f"{Fore.CYAN}Code:{Style.RESET_ALL}")
            print(f"{log_entry['question']}")
            if log_entry.get('answer'):
                print(f"\n{Fore.GREEN}Output:{Style.RESET_ALL}")
                print(f"{log_entry['answer']}")
            if log_entry.get('metadata', {}).get('error'):
                print(f"\n{Fore.RED}Error: {log_entry['metadata']['error']}{Style.RESET_ALL}")
        elif q_type == "INTERMEDIATE_ANSWER":
            print(f"\n{Fore.BLUE}{'─'*80}")
            print(f"{Fore.BLUE}[INTERMEDIATE] {log_entry.get('question', 'What we know so far')}:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{log_entry['answer']}{Style.RESET_ALL}")
            
            # Display metrics if available
            if log_entry.get('metadata', {}).get('metrics'):
                metrics = log_entry['metadata']['metrics']
                print(f"\n{Fore.CYAN}📊 Answer Quality Metrics:{Style.RESET_ALL}")
                print(f"  Completeness: {Fore.GREEN if metrics['completeness'] >= 0.7 else Fore.YELLOW}{metrics['completeness']:.2f}{Style.RESET_ALL}")
                print(f"  Accuracy:     {Fore.GREEN if metrics['accuracy'] >= 0.7 else Fore.YELLOW}{metrics['accuracy']:.2f}{Style.RESET_ALL}")
                print(f"  Confidence:   {Fore.GREEN if metrics['confidence'] >= 0.7 else Fore.YELLOW}{metrics['confidence']:.2f}{Style.RESET_ALL}")
                print(f"  Relevance:    {Fore.GREEN if metrics['relevance'] >= 0.7 else Fore.YELLOW}{metrics['relevance']:.2f}{Style.RESET_ALL}")
                
                # Overall score with interpretation
                overall = metrics['overall_score']
                if overall >= 0.9:
                    color = Fore.GREEN
                    status = "Excellent"
                elif overall >= 0.75:
                    color = Fore.GREEN
                    status = "Good"
                elif overall >= 0.6:
                    color = Fore.YELLOW
                    status = "Fair"
                else:
                    color = Fore.RED
                    status = "Needs Improvement"
                
                print(f"  {'─'*30}")
                print(f"  Overall:      {color}{overall:.2f}{Style.RESET_ALL} ({status})")
            
            print(f"{Fore.BLUE}{'─'*80}{Style.RESET_ALL}")
        elif q_type == "FINAL_ANSWER":
            print(f"\n{Fore.MAGENTA}{'='*80}")
            print(f"{Fore.MAGENTA}[FINAL ANSWER]{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{'='*80}")
            print(f"{Fore.GREEN}{log_entry['answer']}{Style.RESET_ALL}")
            
            # Display final metrics if available
            if log_entry.get('metadata', {}).get('metrics'):
                metrics = log_entry['metadata']['metrics']
                print(f"\n{Fore.CYAN}📊 Final Answer Quality:{Style.RESET_ALL}")
                print(f"  Completeness: {Fore.GREEN if metrics['completeness'] >= 0.7 else Fore.YELLOW}{metrics['completeness']:.2f}{Style.RESET_ALL}")
                print(f"  Accuracy:     {Fore.GREEN if metrics['accuracy'] >= 0.7 else Fore.YELLOW}{metrics['accuracy']:.2f}{Style.RESET_ALL}")
                print(f"  Confidence:   {Fore.GREEN if metrics['confidence'] >= 0.7 else Fore.YELLOW}{metrics['confidence']:.2f}{Style.RESET_ALL}")
                print(f"  Relevance:    {Fore.GREEN if metrics['relevance'] >= 0.7 else Fore.YELLOW}{metrics['relevance']:.2f}{Style.RESET_ALL}")
                
                overall = metrics['overall_score']
                if overall >= 0.9:
                    color = Fore.GREEN
                    status = "⭐ Excellent"
                elif overall >= 0.75:
                    color = Fore.GREEN
                    status = "✓ Good"
                elif overall >= 0.6:
                    color = Fore.YELLOW
                    status = "~ Fair"
                else:
                    color = Fore.RED
                    status = "✗ Poor"
                
                print(f"  {'─'*30}")
                print(f"  Overall:      {color}{overall:.2f}{Style.RESET_ALL} {status}")
            
            print(f"{Fore.MAGENTA}{'='*80}")
        else:
            print(f"\n{Fore.WHITE}[{q_type}] {log_entry['question']}")
            print(f"{Fore.WHITE}Answer: {log_entry['answer']}")
    
    def _save_to_file(self):
        """Save logs to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "logs": self.logs
            }, f, indent=2)
    
    def log_start(self, user_question, context_preview):
        """Log the start of a question."""
        if Config.ENABLE_LOGGING and Config.LOG_TO_CONSOLE:
            print(f"\n{Fore.BLUE}{'='*80}")
            print(f"{Fore.BLUE}NEW QUESTION{Style.RESET_ALL}")
            print(f"{Fore.BLUE}{'='*80}")
            print(f"{Fore.CYAN}User Question:{Style.RESET_ALL} {user_question}")
            print(f"{Fore.CYAN}Context Length:{Style.RESET_ALL} {len(context_preview)} characters")
            print(f"{Fore.BLUE}{'='*80}\n")
    
    def log_error(self, error_msg):
        """Log an error."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "ERROR",
            "message": error_msg
        }
        self.logs.append(log_entry)
        
        if Config.ENABLE_LOGGING and Config.LOG_TO_CONSOLE:
            print(f"{Fore.RED}[ERROR] {error_msg}{Style.RESET_ALL}")
        
        if Config.ENABLE_LOGGING and Config.LOG_TO_FILE:
            self._save_to_file()
    
    def log_info(self, info_msg):
        """Log an info message."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "INFO",
            "message": info_msg
        }
        self.logs.append(log_entry)
        
        if Config.ENABLE_LOGGING and Config.LOG_TO_CONSOLE:
            print(f"{Fore.CYAN}[INFO] {info_msg}{Style.RESET_ALL}")
        
        if Config.ENABLE_LOGGING and Config.LOG_TO_FILE:
            self._save_to_file()
    
    def log_final_answer(self, answer, metadata=None):
        """Log the final answer."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "FINAL_ANSWER",
            "answer": answer,
            "metadata": metadata or {}
        }
        self.logs.append(log_entry)
        
        if Config.ENABLE_LOGGING and Config.LOG_TO_CONSOLE:
            print(f"\n{Fore.GREEN}{'='*80}")
            print(f"{Fore.GREEN}FINAL ANSWER:")
            print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
            print(f"{answer}")
            if metadata:
                print(f"\n{Fore.YELLOW}Metadata: {metadata}{Style.RESET_ALL}")
        
        if Config.ENABLE_LOGGING and Config.LOG_TO_FILE:
            self._save_to_file()
    
    def get_summary(self):
        """Get a summary of the session."""
        return {
            "session_id": self.session_id,
            "total_iterations": len([l for l in self.logs if l.get("type") == "RLM_TO_SUBLM"]),
            "log_file": str(self.log_file) if self.log_file.exists() else None
        }