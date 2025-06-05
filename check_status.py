#!/usr/bin/env python3
"""
Quick status check for SAGED Bias Analysis Platform
"""

import requests
import sys
from urllib.parse import urlparse

def check_url(url, name):
    """Check if a URL is accessible"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}: {url} - OK")
            return True
        else:
            print(f"❌ {name}: {url} - HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: {url} - Connection refused")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ {name}: {url} - Timeout")
        return False
    except Exception as e:
        print(f"❌ {name}: {url} - Error: {e}")
        return False

def main():
    print("🔍 Checking SAGED Platform Status...\n")
    
    # Check backend
    backend_ok = check_url("http://localhost:8000/health", "Backend API")
    
    # Check frontend (multiple possible ports)
    frontend_ports = [3000, 3001, 3002, 3003]
    frontend_ok = False
    
    for port in frontend_ports:
        if check_url(f"http://localhost:{port}", f"Frontend (port {port})"):
            frontend_ok = True
            break
    
    if not frontend_ok:
        print("❌ Frontend: Not accessible on any common port")
    
    print("\n" + "="*50)
    if backend_ok and frontend_ok:
        print("🎉 All services are running!")
        print("📊 Backend API: http://localhost:8000")
        print("📋 API Docs: http://localhost:8000/docs")
        print("🌐 Frontend: Check above for the correct port")
    else:
        print("⚠️  Some services are not running")
        if not backend_ok:
            print("   - Start backend: cd app/backend && python run_server.py")
        if not frontend_ok:
            print("   - Start frontend: cd app/frontend && npm run dev")
        print("   - Or use: python start_full_app.py")
    
    print("="*50)
    
    return 0 if (backend_ok and frontend_ok) else 1

if __name__ == "__main__":
    sys.exit(main()) 