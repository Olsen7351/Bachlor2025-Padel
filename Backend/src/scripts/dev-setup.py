#!/usr/bin/env python3
"""
Development setup script for Padel Analyzer
Usage: python backend/src/scripts/dev-setup.py [start|stop|reset|status]
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_command(cmd, check=True, capture_output=False):
    """Run a shell command"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=check, 
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {e}")
        return None

def find_project_root():
    """Find the project root directory (where docker-compose.yml is)"""
    script_dir = Path(__file__).parent  # backend/src/scripts/
    
    # Go up directories until we find docker-compose.yml
    current = script_dir
    for _ in range(5):  # Limit search to avoid infinite loop
        current = current.parent
        docker_compose_path = current / "docker-compose.yml"
        if docker_compose_path.exists():
            return current
    
    # Fallback: assume it's 3 levels up
    return script_dir.parent.parent.parent

def start_services():
    """Start database services only"""
    print("🚀 Starting database services...")
    
    # Start database services only
    result = run_command("docker-compose up -d postgres redis")
    if not result:
        return False
    
    print("⏳ Waiting for services to be ready...")
    
    # Wait for PostgreSQL
    for i in range(30):
        result = run_command(
            "docker-compose exec -T postgres pg_isready -U padel_user -d padel_dev",
            check=False,
            capture_output=True
        )
        if result and result.returncode == 0:
            print("✅ PostgreSQL is ready")
            break
        time.sleep(1)
        print(f"⏳ Waiting for PostgreSQL... ({i+1}/30)")
    else:
        print("❌ PostgreSQL failed to start")
        return False
    
    # Check Redis
    result = run_command(
        "docker-compose exec -T redis redis-cli ping",
        check=False,
        capture_output=True
    )
    if result and result.returncode == 0:
        print("✅ Redis is ready")
    
    print("\n🎉 Database services are ready!")
    print("📊 Services running:")
    print("   - PostgreSQL: localhost:5432")
    print("   - Redis: localhost:6379")
    print("\n💡 Next steps:")
    print("   1. cd backend/src")
    print("   2. uv run python main.py")
    print("   3. Visit: http://localhost:8000/docs")
    
    return True

def stop_services():
    """Stop all services"""
    print("🛑 Stopping services...")
    run_command("docker-compose down")
    print("✅ Services stopped")

def reset_services():
    """Reset all services (removes data!)"""
    print("🔄 Resetting database...")
    print("⚠️  This will delete all data!")
    
    response = input("Are you sure? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled")
        return
    
    run_command("docker-compose down -v")
    print("✅ Database reset")
    
    # Restart services
    start_services()

def show_status():
    """Show status of services"""
    print("📊 Service Status:")
    run_command("docker-compose ps")

def main():
    # Find and change to project root
    project_root = find_project_root()
    print(f"📁 Project root: {project_root}")
    os.chdir(project_root)
    
    # Verify docker-compose.yml exists
    if not (project_root / "docker-compose.yml").exists():
        print("❌ docker-compose.yml not found!")
        sys.exit(1)
    
    command = sys.argv[1] if len(sys.argv) > 1 else "start"
    
    if command == "start":
        success = start_services()
        sys.exit(0 if success else 1)
    elif command == "stop":
        stop_services()
    elif command == "reset":
        reset_services()
    elif command == "status":
        show_status()
    else:
        print("Usage: python backend/src/scripts/dev-setup.py [start|stop|reset|status]")
        sys.exit(1)

if __name__ == "__main__":
    main()