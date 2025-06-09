#!/usr/bin/env python3
"""
SAGED Bias Analysis Platform - Full Application Launcher
Starts both frontend and backend servers simultaneously
"""

import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path

# Add the project root to Python path to ensure local saged package is used
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color=Colors.ENDC):
    print(f"{color}{message}{Colors.ENDC}")

def print_banner():
    print_colored("""
╔══════════════════════════════════════════════════════════════╗
║                    SAGED Bias Analysis Platform             ║
║                      Full Stack Launcher                    ║
╚══════════════════════════════════════════════════════════════╝
    """, Colors.HEADER)

class AppLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.project_root = Path(__file__).parent.absolute()
        self.backend_dir = self.project_root / "app" / "backend"
        self.frontend_dir = self.project_root / "app" / "frontend"
        
    def check_requirements(self):
        """Check if all required directories and files exist"""
        print_colored("🔍 Checking requirements...", Colors.OKBLUE)
        
        # Check backend
        if not self.backend_dir.exists():
            print_colored(f"❌ Backend directory not found: {self.backend_dir}", Colors.FAIL)
            return False
            
        if not (self.backend_dir / "main.py").exists():
            print_colored(f"❌ Backend main.py not found", Colors.FAIL)
            return False
            
        # Check frontend
        if not self.frontend_dir.exists():
            print_colored(f"❌ Frontend directory not found: {self.frontend_dir}", Colors.FAIL)
            return False
            
        if not (self.frontend_dir / "package.json").exists():
            print_colored(f"❌ Frontend package.json not found", Colors.FAIL)
            return False
            
        print_colored("✅ All requirements satisfied", Colors.OKGREEN)
        return True
        
    def install_frontend_deps(self):
        """Install frontend dependencies if node_modules doesn't exist"""
        if not (self.frontend_dir / "node_modules").exists():
            print_colored("📦 Installing frontend dependencies...", Colors.WARNING)
            try:
                result = subprocess.run(
                    ["npm", "install"],
                    cwd=self.frontend_dir,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print_colored("✅ Frontend dependencies installed", Colors.OKGREEN)
            except subprocess.CalledProcessError as e:
                print_colored(f"❌ Failed to install frontend dependencies: {e}", Colors.FAIL)
                return False
        return True
        
    def start_backend(self):
        """Start the FastAPI backend server"""
        print_colored("🚀 Starting backend server...", Colors.OKBLUE)
        
        try:
            # Change to backend directory and start server
            self.backend_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
                cwd=self.backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env={**os.environ, "PYTHONPATH": f"{self.project_root}{os.pathsep}{self.backend_dir}"}
            )
            
            # Start a thread to monitor backend output
            backend_monitor = threading.Thread(target=self._monitor_backend_output)
            backend_monitor.daemon = True
            backend_monitor.start()
            
            # Wait a moment for the backend to start
            time.sleep(3)
            
            if self.backend_process.poll() is None:
                print_colored("✅ Backend server started successfully", Colors.OKGREEN)
                print_colored("📊 Backend API: http://localhost:8000", Colors.OKCYAN)
                print_colored("📋 API Docs: http://localhost:8000/docs", Colors.OKCYAN)
                return True
            else:
                print_colored("❌ Backend server failed to start", Colors.FAIL)
                return False
                
        except Exception as e:
            print_colored(f"❌ Error starting backend: {e}", Colors.FAIL)
            return False
            
    def start_frontend(self):
        """Start the React frontend development server"""
        print_colored("🎨 Starting frontend server...", Colors.OKBLUE)
        
        try:
            # Print current working directory and command
            print_colored(f"Current working directory: {self.frontend_dir}", Colors.OKCYAN)
            
            # Try to find npm in common locations
            npm_paths = [
                "npm",  # Try direct npm first
                r"C:\Program Files\nodejs\npm.cmd",  # Common Windows installation path
                r"C:\Program Files (x86)\nodejs\npm.cmd",
                os.path.expanduser("~\\AppData\\Roaming\\npm\\npm.cmd"),  # User installation
            ]
            
            npm_cmd = None
            for path in npm_paths:
                try:
                    result = subprocess.run(
                        [path, "--version"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    npm_cmd = path
                    print_colored(f"Found npm at: {path}", Colors.OKGREEN)
                    print_colored(f"NPM version: {result.stdout.strip()}", Colors.OKCYAN)
                    break
                except:
                    continue
            
            if not npm_cmd:
                print_colored("❌ Could not find npm executable. Please ensure Node.js is installed and in PATH", Colors.FAIL)
                return False
            
            print_colored("Running command: npm run dev", Colors.OKCYAN)
            
            self.frontend_process = subprocess.Popen(
                [npm_cmd, "run", "dev"],
                cwd=self.frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start a thread to monitor frontend output
            frontend_monitor = threading.Thread(target=self._monitor_frontend_output)
            frontend_monitor.daemon = True
            frontend_monitor.start()
            
            # Wait for frontend to start
            time.sleep(5)
            
            if self.frontend_process.poll() is None:
                print_colored("✅ Frontend server started successfully", Colors.OKGREEN)
                return True
            else:
                # Get the error output
                _, stderr = self.frontend_process.communicate()
                print_colored(f"❌ Frontend server failed to start. Error: {stderr}", Colors.FAIL)
                return False
                
        except Exception as e:
            print_colored(f"❌ Error starting frontend: {str(e)}", Colors.FAIL)
            return False
            
    def _monitor_backend_output(self):
        """Monitor backend process output"""
        if not self.backend_process:
            return
            
        for line in iter(self.backend_process.stdout.readline, ''):
            if line.strip():
                if "Uvicorn running on" in line:
                    print_colored(f"[BACKEND] {line.strip()}", Colors.OKGREEN)
                elif "ERROR" in line.upper():
                    print_colored(f"[BACKEND] {line.strip()}", Colors.FAIL)
                elif "WARNING" in line.upper():
                    print_colored(f"[BACKEND] {line.strip()}", Colors.WARNING)
                    
    def _monitor_frontend_output(self):
        """Monitor frontend process output"""
        if not self.frontend_process:
            return
            
        frontend_url = None
        try:
            for line in iter(self.frontend_process.stdout.readline, ''):
                if line.strip():
                    if "Local:" in line and "http://localhost:" in line:
                        # Extract the URL
                        parts = line.split()
                        for part in parts:
                            if part.startswith("http://localhost:"):
                                frontend_url = part.rstrip('/')
                                print_colored(f"🌐 Frontend: {frontend_url}", Colors.OKCYAN)
                                break
                    elif "ready in" in line:
                        print_colored(f"[FRONTEND] {line.strip()}", Colors.OKGREEN)
                    elif "error" in line.lower():
                        print_colored(f"[FRONTEND] {line.strip()}", Colors.FAIL)
        except UnicodeDecodeError:
            # If we hit an encoding error, try to read with UTF-8
            self.frontend_process.stdout.reconfigure(encoding='utf-8')
            for line in iter(self.frontend_process.stdout.readline, ''):
                if line.strip():
                    if "Local:" in line and "http://localhost:" in line:
                        parts = line.split()
                        for part in parts:
                            if part.startswith("http://localhost:"):
                                frontend_url = part.rstrip('/')
                                print_colored(f"🌐 Frontend: {frontend_url}", Colors.OKCYAN)
                                break
                    elif "ready in" in line:
                        print_colored(f"[FRONTEND] {line.strip()}", Colors.OKGREEN)
                    elif "error" in line.lower():
                        print_colored(f"[FRONTEND] {line.strip()}", Colors.FAIL)
                    
    def cleanup(self):
        """Clean up processes on exit"""
        print_colored("\n🛑 Shutting down servers...", Colors.WARNING)
        
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                print_colored("✅ Frontend server stopped", Colors.OKGREEN)
            except:
                self.frontend_process.kill()
                print_colored("🔨 Frontend server force killed", Colors.WARNING)
                
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                print_colored("✅ Backend server stopped", Colors.OKGREEN)
            except:
                self.backend_process.kill()
                print_colored("🔨 Backend server force killed", Colors.WARNING)
                
        print_colored("👋 Goodbye!", Colors.HEADER)
        
    def run(self):
        """Main execution function"""
        print_banner()
        
        # Check requirements
        if not self.check_requirements():
            sys.exit(1)
            
        # Install frontend dependencies if needed
        if not self.install_frontend_deps():
            sys.exit(1)
            
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, lambda s, f: self.cleanup() or sys.exit(0))
        signal.signal(signal.SIGTERM, lambda s, f: self.cleanup() or sys.exit(0))
        
        try:
            # Start backend
            if not self.start_backend():
                sys.exit(1)
                
            # Start frontend
            if not self.start_frontend():
                self.cleanup()
                sys.exit(1)
                
            print_colored("\n" + "="*60, Colors.OKGREEN)
            print_colored("🎉 SAGED Platform is running!", Colors.OKGREEN)
            print_colored("="*60, Colors.OKGREEN)
            print_colored("📊 Backend API: http://localhost:8000", Colors.OKCYAN)
            print_colored("📋 API Documentation: http://localhost:8000/docs", Colors.OKCYAN)
            print_colored("🌐 Frontend App: Check terminal output above for the URL", Colors.OKCYAN)
            print_colored("="*60, Colors.OKGREEN)
            print_colored("💡 Press Ctrl+C to stop all servers", Colors.WARNING)
            print_colored("="*60 + "\n", Colors.OKGREEN)
            
            # Keep the main process alive
            try:
                while True:
                    time.sleep(1)
                    # Check if processes are still running
                    if self.backend_process and self.backend_process.poll() is not None:
                        print_colored("❌ Backend process died unexpectedly", Colors.FAIL)
                        break
                    if self.frontend_process and self.frontend_process.poll() is not None:
                        print_colored("❌ Frontend process died unexpectedly", Colors.FAIL)
                        break
            except KeyboardInterrupt:
                pass
                
        except Exception as e:
            print_colored(f"❌ Unexpected error: {e}", Colors.FAIL)
        finally:
            self.cleanup()

if __name__ == "__main__":
    launcher = AppLauncher()
    launcher.run() 