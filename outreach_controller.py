#!/usr/bin/env python3
"""
Grant Outreach Manager – All‑in‑One Setup and Management Script
Handles setup, virtual environment, and daemon control
Cross-platform: Windows, macOS, Linux
"""

import sys
import os
import subprocess
import argparse
import time
import urllib.request
import platform
from pathlib import Path
from typing import Optional, Tuple

def _running_inside_venv() -> bool:
    """Check if we're running inside a virtual environment"""
    return (
        hasattr(sys, "real_prefix") or 
        getattr(sys, "base_prefix", sys.prefix) != sys.prefix
    )

# Auto-activate virtual environment if not already inside one
if not _running_inside_venv():
    root = Path(__file__).resolve().parent
    
    # Cross-platform virtual environment paths
    if platform.system() == "Windows":
        venv_python = root / ".venv" / "Scripts" / "python.exe"
    else:  # macOS, Linux
        venv_python = root / ".venv" / "bin" / "python"
    
    if venv_python.exists():
        # Re-execute this script with the venv Python
        print(f"🔄 Activating virtual environment...")
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)
    else:
        print("❌ Virtual environment not found. Run setup first:")
        print("   python outreach_controller.py --setup")
        sys.exit(1)

# Now import everything else after venv activation
import logging
import traceback
import json
from dotenv import load_dotenv

# ===== CONFIGURATION =====
VENV_DIR = ".venv"
DATASETS_DIR = "./datasets"
OUTPUTS_DIR = "./outputs"
LOGS_DIR = "./logs"
STATE_FILE = "./logs/processing_state.json"
LOG_FILE = "./logs/outreach.log"
PID_FILE = "./logs/daemon.pid"
SETUP_MARKER = "./.setup_complete"
DEPENDENCIES = ["openai", "pandas", "google-genai", "google-api-core", "python-dotenv", "groq"]
GENERATOR_DAEMON_FPATH = "outreach_daemon.py"

class OutreachManager:
    def __init__(self):
        self.venv_path = Path(VENV_DIR)
        
        # Cross-platform Python path
        if platform.system() == "Windows":
            self.python_path = self.venv_path / "Scripts" / "python.exe"
        else:  # macOS, Linux
            self.python_path = self.venv_path / "bin" / "python"

    def check_python_version(self):
        """Check Python version"""
        print(f"🐍 Python version: {sys.version}")
        print(f"💻 Platform: {platform.system()} {platform.release()}")
        
        if sys.version_info < (3, 6):
            print("❌ Python 3.6+ is required")
            sys.exit(1)
        
        print("✅ Python version compatible")

    def create_venv(self):
        """Create virtual environment - cross-platform"""
        print("🐍 Creating virtual environment...")
        try:
            # First try normal venv creation
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError:
            # If that fails, try without pip and install it manually
            print("⚠️  Standard venv creation failed, trying alternative method...")
            try:
                # Create venv without pip
                subprocess.run([sys.executable, "-m", "venv", "--without-pip", VENV_DIR], check=True)

                # Download get-pip.py
                print("📥 Downloading pip installer...")
                urllib.request.urlretrieve(
                    "https://bootstrap.pypa.io/get-pip.py",
                    os.path.join(VENV_DIR, "get-pip.py")
                )

                # Install pip in the venv
                print("📦 Installing pip in virtual environment...")
                subprocess.run([str(self.python_path), os.path.join(VENV_DIR, "get-pip.py")],
                             check=True, capture_output=True)

                # Clean up
                os.remove(os.path.join(VENV_DIR, "get-pip.py"))
                print("✅ Virtual environment created with manual pip installation")

            except Exception as e:
                print(f"❌ Failed to create virtual environment: {e}")
                if platform.system() == "Darwin":  # macOS
                    print("\n📋 On macOS, try installing Python via Homebrew:")
                    print("   brew install python")
                elif platform.system() == "Linux":
                    print("\n📋 On Linux, install python3-venv:")
                    print(f"   sudo apt install python3.{sys.version_info.minor}-venv")
                    print("Or try installing python3-pip:")
                    print("   sudo apt install python3-pip")
                else:  # Windows
                    print("\n📋 On Windows, ensure Python was installed with pip")
                    print("   Download from: https://python.org/downloads/")
                sys.exit(1)

    def install_dependencies(self):
        """Install dependencies in virtual environment"""
        print("📦 Installing dependencies...")
        for package in DEPENDENCIES:
            try:
                subprocess.run([str(self.python_path), "-m", "pip", "install", package],
                             check=True, capture_output=True, text=True)
                print(f"   ✅ {package} installed")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {package}: {e.stderr}")
                sys.exit(1)

    def setup_api_keys(self):
        """Setup all API keys - cross-platform"""
        print("\n🔑 API Keys Setup")
        print("----------------")
        
        env_file = Path('.env')
        env_vars = {}
        
        # Load existing .env if it exists
        if env_file.exists():
            load_dotenv()
            env_vars = dict(os.environ)
        
        # API keys to collect
        api_keys = {
            'OPENAI_API_KEY': 'OpenAI API key (https://platform.openai.com/api-keys)',
            'GEMINI_API_KEY': 'Google Gemini API key (https://makersuite.google.com/app/apikey)', 
            'GROQ_API_KEY': 'Groq API key (https://console.groq.com/)',
            'OPENROUTER_API_KEY': 'OpenRouter API key (https://openrouter.ai/keys)'
        }
        
        keys_found = 0
        for key, description in api_keys.items():
            current_key = env_vars.get(key)
            if current_key and len(current_key) > 10:
                print(f"✅ {key} found")
                keys_found += 1
        
        if keys_found >= 2:  # At least 2 API keys found
            print(f"✅ Found {keys_found} API keys in .env file")
            response = input("Update API keys? (y/N): ")
            if response.lower() != 'y':
                return env_vars
        
        # Collect missing or update existing keys
        for key, description in api_keys.items():
            current_key = env_vars.get(key)
            if current_key and len(current_key) > 10:
                response = input(f"Update {key}? (y/N): ")
                if response.lower() != 'y':
                    continue
            
            print(f"\n{description}")
            while True:
                new_key = input(f"🔐 Paste your {key} (or press Enter to skip): ").strip()
                if not new_key:
                    break
                env_vars[key] = new_key
                break
        
        # Optional site info for OpenRouter
        if 'OPENROUTER_API_KEY' in env_vars:
            if not env_vars.get('SITE_URL'):
                env_vars['SITE_URL'] = input("📝 Your site URL (optional): ").strip() or "https://localhost"
            if not env_vars.get('SITE_NAME'):
                env_vars['SITE_NAME'] = input("📝 Your site name (optional): ").strip() or "Outreach Generator"
        
        # Save to .env file
        with open(env_file, 'w') as f:
            f.write("# API Keys for LLM Rotation\n")
            for key in ['OPENAI_API_KEY', 'GEMINI_API_KEY', 'GROQ_API_KEY', 'OPENROUTER_API_KEY', 'SITE_URL', 'SITE_NAME']:
                if key in env_vars and env_vars[key]:
                    f.write(f"{key}={env_vars[key]}\n")
        
        print("✅ API keys saved to .env file")
        return env_vars

    def setup(self):
        """Run one-time setup - cross-platform"""
        print("🚀 Grant Outreach Generator Setup")
        print("=================================\n")

        self.check_python_version()

        # Create virtual environment if needed
        if not self.venv_path.exists():
            self.create_venv()
        else:
            print("✅ Virtual environment already exists")
        
        self.install_dependencies()
        
        # Setup ALL API keys
        api_keys = self.setup_api_keys()

        # Create directories
        print("\n📁 Creating directories...")
        for dir_path in [DATASETS_DIR, OUTPUTS_DIR, LOGS_DIR]:
            Path(dir_path).mkdir(exist_ok=True)
        print("✅ Directories created")

        # Mark setup as complete
        Path(SETUP_MARKER).touch()

        print("\n🎉 Setup complete!")
        print(f"\n📋 Quick Start ({platform.system()}):")
        print("   1. Drop LinkedIn CSV files in ./datasets/")
        print("   2. Run: python outreach_controller.py start")
        if platform.system() != "Windows":
            print("   3. Monitor: tail -f logs/outreach.log")
        else:
            print("   3. Monitor: Get-Content logs/outreach.log -Wait")
        print("   4. Stop: python outreach_controller.py stop")

        return api_keys

    def start(self, resume=False, test_mode=False, test_run_count=3):
        """Start the daemon - cross-platform"""
        # Check if already running
        pid = self.is_running()
        if pid:
            print(f"⚠️  Outreach generator already running (PID: {pid})")
            if platform.system() != "Windows":
                print("   📋 Monitor: tail -f logs/outreach.log")
            else:
                print("   📋 Monitor: Get-Content logs/outreach.log -Wait")
            print("   🛑 Stop: python outreach_controller.py stop")
            return

        # Run setup if needed
        if not Path(SETUP_MARKER).exists():
            api_keys = self.setup()
        else:
            # Load API keys from .env file
            load_dotenv()
            
            # Check for at least one API key
            api_key = (os.getenv('OPENAI_API_KEY') or 
                      os.getenv('GEMINI_API_KEY') or 
                      os.getenv('GROQ_API_KEY') or 
                      os.getenv('OPENROUTER_API_KEY'))
            
            if not api_key:
                print("❌ No API keys found in .env file")
                print("   Run setup again: python outreach_controller.py --setup")
                sys.exit(1)

        # Clean if not resuming
        if not resume:
            self.clean()

        print("🚀 Starting Grant Outreach Generator...")

        # Load environment variables from .env
        load_dotenv()
        env = os.environ.copy()

        try:
            if test_mode:
                # Import and run the daemon directly in test mode
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import outreach_daemon

                controller = outreach_daemon.DaemonController()
                controller.initialize()
                controller.test_run(run_count=test_run_count)
                return
            else:
                if platform.system() == "Windows":
                    # On Windows, run in foreground with proper signal handling
                    print("✅ Starting in foreground mode (Windows)")
                    
                    # Run daemon directly (not subprocess)
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    import outreach_daemon
                    
                    controller = outreach_daemon.DaemonController()
                    controller.initialize()
                    
                    # Write PID file
                    with open(PID_FILE, 'w') as f:
                        f.write(str(os.getpid()))
                    
                    print(f"✅ Started daemon (PID: {os.getpid()})")
                    print("📋 Press Ctrl+C to stop")
                    
                    # Run the daemon loop
                    controller.run_daemon_loop()
                    
                else:
                    # On Unix/macOS, use proper daemon forking
                    result = subprocess.run([str(self.python_path), GENERATOR_DAEMON_FPATH],
                                          env=env, check=True)

        except KeyboardInterrupt:
            print("\n🛑 Stopping daemon...")
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
            print("✅ Daemon stopped")
        except Exception as e:
            print(f"❌ Error starting generator: {e}")
            traceback.print_exc()
            sys.exit(1)

    def stop(self):
        """Stop the daemon - cross-platform"""
        pid = self.is_running()
        if not pid:
            print("⚠️  Outreach generator is not running")
            return

        try:
            if platform.system() == "Windows":
                # Windows process termination
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True)
            else:
                # Unix/macOS process termination
                os.kill(pid, 15)  # SIGTERM
                time.sleep(2)
                
                # Check if still running and force kill if needed
                try:
                    os.kill(pid, 0)  # Check if process exists
                    print("⚠️  Process still running, force killing...")
                    os.kill(pid, 9)  # SIGKILL
                except ProcessLookupError:
                    pass  # Process already dead

            # Remove PID file
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)

            print("✅ Outreach generator stopped")

        except (ProcessLookupError, subprocess.CalledProcessError):
            print("⚠️  Process not found, cleaning up PID file")
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
        except Exception as e:
            print(f"❌ Error stopping daemon: {e}")

    def is_running(self) -> Optional[int]:
        """Check if daemon is running - cross-platform"""
        if not os.path.exists(PID_FILE):
            return None

        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())

            if platform.system() == "Windows":
                # Windows process check
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    capture_output=True, text=True
                )
                if str(pid) in result.stdout:
                    return pid
            else:
                # Unix/macOS process check
                os.kill(pid, 0)  # This will raise OSError if process doesn't exist
                return pid

        except (ValueError, OSError, subprocess.CalledProcessError):
            # PID file exists but process is dead
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)

        return None

    def status(self):
        """Show daemon status - cross-platform"""
        pid = self.is_running()
        if pid:
            print(f"✅ Outreach generator is running (PID: {pid})")
            print(f"📁 Monitoring: {DATASETS_DIR}")
            print(f"📊 Outputs: {OUTPUTS_DIR}")
            print(f"📋 Logs: {LOG_FILE}")
            
            # Show platform-specific monitoring command
            if platform.system() != "Windows":
                print(f"🔍 Monitor: tail -f {LOG_FILE}")
            else:
                print(f"🔍 Monitor: Get-Content {LOG_FILE} -Wait")
        else:
            print("❌ Outreach generator is not running")

    def clean(self):
        """Clean logs and outputs"""
        print("🧹 Cleaning logs and outputs...")
        
        # Clean logs
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        print("   ✅ Cleaned logs")
        
        # Clean outputs
        if os.path.exists(OUTPUTS_DIR):
            for file in os.listdir(OUTPUTS_DIR):
                file_path = os.path.join(OUTPUTS_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        print("   ✅ Cleaned outputs")

def main():
    parser = argparse.ArgumentParser(description="Grant Outreach Generator - Cross-Platform")
    parser.add_argument('command', choices=['start', 'stop', 'status', 'restart', 'test'],
                       help='Command to execute')
    parser.add_argument('--setup', action='store_true', help='Run setup')
    parser.add_argument('--resume', action='store_true', help='Resume processing existing data')
    parser.add_argument('--run_count', type=int, default=3, help='Number of people to process in test mode')

    args = parser.parse_args()

    manager = OutreachManager()

    if args.setup:
        manager.setup()
        return

    if args.command == 'start':
        manager.start(resume=args.resume)
    elif args.command == 'stop':
        manager.stop()
    elif args.command == 'status':
        manager.status()
    elif args.command == 'restart':
        manager.stop()
        time.sleep(2)
        manager.start(resume=args.resume)
    elif args.command == 'test':
        manager.start(resume=args.resume, test_mode=True, test_run_count=args.run_count)

if __name__ == "__main__":
    main()
