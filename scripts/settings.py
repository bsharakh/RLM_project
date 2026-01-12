#!/usr/bin/env python3
"""
Settings Manager for RLM Project

This script allows you to view and modify RLM settings without editing files directly.
Settings are stored in .env file and can be easily changed here.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from colorama import Fore, Style, init
init(autoreset=True)


class SettingsManager:
    """Manage RLM settings."""
    
    def __init__(self, env_file=".env"):
        self.env_file = env_file
        self.settings = self.load_settings()
    
    def load_settings(self):
        """Load settings from .env file."""
        settings = {}
        
        if not os.path.exists(self.env_file):
            print(f"{Fore.YELLOW}Warning: {self.env_file} not found{Style.RESET_ALL}")
            return settings
        
        with open(self.env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    settings[key.strip()] = value.strip()
        
        return settings
    
    def save_settings(self):
        """Save settings back to .env file."""
        lines = []
        
        # Read existing file to preserve comments and structure
        if os.path.exists(self.env_file):
            with open(self.env_file, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#') and '=' in stripped:
                        key = stripped.split('=', 1)[0].strip()
                        if key in self.settings:
                            lines.append(f"{key}={self.settings[key]}\n")
                        else:
                            lines.append(line)
                    else:
                        lines.append(line)
        
        # Add any new settings
        existing_keys = set()
        with open(self.env_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key = line.split('=', 1)[0].strip()
                    existing_keys.add(key)
        
        for key, value in self.settings.items():
            if key not in existing_keys:
                lines.append(f"\n{key}={value}\n")
        
        # Write back
        with open(self.env_file, 'w') as f:
            f.writelines(lines)
    
    def display_settings(self):
        """Display current settings in a nice format."""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}RLM System Settings")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        
        # API Key
        api_key = self.settings.get('OPENAI_API_KEY', 'NOT SET')
        if api_key != 'NOT SET':
            masked_key = api_key[:10] + '...' + api_key[-4:]
        else:
            masked_key = api_key
        print(f"{Fore.YELLOW}🔑 API Configuration:{Style.RESET_ALL}")
        print(f"   OPENAI_API_KEY: {masked_key}\n")
        
        # Models
        print(f"{Fore.YELLOW}🤖 Model Configuration:{Style.RESET_ALL}")
        print(f"   [1] RLM_MODEL (Boss):   {Fore.GREEN}{self.settings.get('RLM_MODEL', 'gpt-4')}{Style.RESET_ALL}")
        print(f"   [2] SUBLM_MODEL (Intern): {Fore.GREEN}{self.settings.get('SUBLM_MODEL', 'gpt-4o-mini')}{Style.RESET_ALL}\n")
        
        # Iterations
        print(f"{Fore.YELLOW}🔄 Iteration Settings:{Style.RESET_ALL}")
        print(f"   [3] MAX_ITERATIONS: {Fore.GREEN}{self.settings.get('MAX_ITERATIONS', '5')}{Style.RESET_ALL}")
        print(f"   [4] MAX_DEPTH: {Fore.GREEN}{self.settings.get('MAX_DEPTH', '1')}{Style.RESET_ALL}\n")
        
        # Metrics
        print(f"{Fore.YELLOW}📊 Metrics Configuration:{Style.RESET_ALL}")
        print(f"   [5] ENABLE_METRICS: {Fore.GREEN}{self.settings.get('ENABLE_METRICS', 'true')}{Style.RESET_ALL}")
        print(f"   [6] METRICS_MODEL: {Fore.GREEN}{self.settings.get('METRICS_MODEL', 'gpt-4o-mini')}{Style.RESET_ALL}\n")
        
        # Logging
        print(f"{Fore.YELLOW}📝 Logging Configuration:{Style.RESET_ALL}")
        print(f"   [7] ENABLE_LOGGING: {Fore.GREEN}{self.settings.get('ENABLE_LOGGING', 'true')}{Style.RESET_ALL}")
        print(f"   [8] LOG_LEVEL: {Fore.GREEN}{self.settings.get('LOG_LEVEL', 'INFO')}{Style.RESET_ALL}")
        print(f"   [9] LOG_TO_FILE: {Fore.GREEN}{self.settings.get('LOG_TO_FILE', 'true')}{Style.RESET_ALL}")
        print(f"   [10] LOG_TO_CONSOLE: {Fore.GREEN}{self.settings.get('LOG_TO_CONSOLE', 'true')}{Style.RESET_ALL}\n")
        
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    def interactive_menu(self):
        """Interactive menu for changing settings."""
        while True:
            self.display_settings()
            
            print(f"{Fore.MAGENTA}Options:{Style.RESET_ALL}")
            print("  [1-10] Edit a setting")
            print("  [p] Apply preset configuration")
            print("  [s] Save and exit")
            print("  [q] Quit without saving")
            print()
            
            choice = input(f"{Fore.YELLOW}Select option: {Style.RESET_ALL}").strip().lower()
            
            if choice == 'q':
                print(f"{Fore.YELLOW}Exiting without saving...{Style.RESET_ALL}")
                break
            elif choice == 's':
                self.save_settings()
                print(f"{Fore.GREEN}✓ Settings saved to {self.env_file}{Style.RESET_ALL}")
                break
            elif choice == 'p':
                self.show_presets()
            elif choice in [str(i) for i in range(1, 11)]:
                self.edit_setting(int(choice))
            else:
                print(f"{Fore.RED}Invalid option{Style.RESET_ALL}")
                input("Press Enter to continue...")
    
    def edit_setting(self, num):
        """Edit a specific setting."""
        setting_map = {
            1: ('RLM_MODEL', 'Boss model (e.g., gpt-4, gpt-4o, gpt-4o-mini)'),
            2: ('SUBLM_MODEL', 'Intern model (e.g., gpt-4o-mini, gpt-4)'),
            3: ('MAX_ITERATIONS', 'Maximum iterations (e.g., 5, 10, 15)'),
            4: ('MAX_DEPTH', 'Maximum depth (usually 1)'),
            5: ('ENABLE_METRICS', 'Enable metrics (true/false)'),
            6: ('METRICS_MODEL', 'Metrics model (e.g., gpt-4o-mini)'),
            7: ('ENABLE_LOGGING', 'Enable logging (true/false)'),
            8: ('LOG_LEVEL', 'Log level (DEBUG, INFO, WARNING, ERROR)'),
            9: ('LOG_TO_FILE', 'Log to file (true/false)'),
            10: ('LOG_TO_CONSOLE', 'Log to console (true/false)')
        }
        
        key, description = setting_map[num]
        current = self.settings.get(key, 'NOT SET')
        
        print(f"\n{Fore.CYAN}Editing: {key}{Style.RESET_ALL}")
        print(f"Description: {description}")
        print(f"Current value: {Fore.YELLOW}{current}{Style.RESET_ALL}")
        
        new_value = input(f"New value (Enter to keep current): ").strip()
        
        if new_value:
            self.settings[key] = new_value
            print(f"{Fore.GREEN}✓ Updated {key} to {new_value}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Kept current value{Style.RESET_ALL}")
        
        input("\nPress Enter to continue...")
    
    def show_presets(self):
        """Show and apply preset configurations."""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}Preset Configurations")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        
        presets = {
            '1': {
                'name': 'Fast & Cheap',
                'desc': 'Fastest and cheapest, good for simple queries',
                'settings': {
                    'RLM_MODEL': 'gpt-4o-mini',
                    'SUBLM_MODEL': 'gpt-4o-mini',
                    'MAX_ITERATIONS': '3',
                    'ENABLE_METRICS': 'false'
                }
            },
            '2': {
                'name': 'Balanced (Default)',
                'desc': 'Good balance of quality and cost',
                'settings': {
                    'RLM_MODEL': 'gpt-4',
                    'SUBLM_MODEL': 'gpt-4o-mini',
                    'MAX_ITERATIONS': '5',
                    'ENABLE_METRICS': 'true'
                }
            },
            '3': {
                'name': 'High Quality',
                'desc': 'Best quality, slower and more expensive',
                'settings': {
                    'RLM_MODEL': 'gpt-4',
                    'SUBLM_MODEL': 'gpt-4',
                    'MAX_ITERATIONS': '10',
                    'ENABLE_METRICS': 'true'
                }
            },
            '4': {
                'name': 'Complex Research',
                'desc': 'For very complex multi-step questions',
                'settings': {
                    'RLM_MODEL': 'gpt-4',
                    'SUBLM_MODEL': 'gpt-4o-mini',
                    'MAX_ITERATIONS': '15',
                    'ENABLE_METRICS': 'true'
                }
            }
        }
        
        for key, preset in presets.items():
            print(f"{Fore.GREEN}[{key}] {preset['name']}{Style.RESET_ALL}")
            print(f"    {preset['desc']}")
            print(f"    Settings: {preset['settings']}\n")
        
        print(f"{Fore.YELLOW}[b] Back to main menu{Style.RESET_ALL}\n")
        
        choice = input("Select preset: ").strip()
        
        if choice in presets:
            for key, value in presets[choice]['settings'].items():
                self.settings[key] = value
            print(f"\n{Fore.GREEN}✓ Applied '{presets[choice]['name']}' preset{Style.RESET_ALL}")
            input("Press Enter to continue...")
        elif choice == 'b':
            return
        else:
            print(f"{Fore.RED}Invalid preset{Style.RESET_ALL}")
            input("Press Enter to continue...")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RLM Settings Manager - View and modify settings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/settings.py              # Interactive mode
  python scripts/settings.py --view       # Just view current settings
  python scripts/settings.py --preset 2   # Apply preset #2 (Balanced)
        """
    )
    parser.add_argument('--view', action='store_true', help='View settings and exit')
    parser.add_argument('--preset', choices=['1', '2', '3', '4'], help='Apply preset directly')
    parser.add_argument('--env-file', default='.env', help='Path to .env file')
    
    args = parser.parse_args()
    
    manager = SettingsManager(args.env_file)
    
    if args.view:
        manager.display_settings()
    elif args.preset:
        manager.show_presets()
        # Auto-apply preset logic would go here
    else:
        manager.interactive_menu()


if __name__ == "__main__":
    main()