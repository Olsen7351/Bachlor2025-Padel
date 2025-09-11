#!/usr/bin/env python3
"""
Development setup script for Padel Analyzer
Usage: python scripts/dev-setup.py [start|stop|reset|status]
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
            cmd, shell=True, check=check, capture_output=capture_output, text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {e}")
        return None


def check_docker():
    """Check if Docker is running"""
    result = run_command("docker info", check=False, capture_output=True)
    if result and result.returncode == 0:
        print("✅ Docker is running")
        return True
    else:
        print("❌ Docker is not running. Please start Docker Desktop.")
        return False


def check_dependencies():
    """Check if required tools are installed"""
    tools = ["docker", "docker-compose"]
    for tool in tools:
        result = run_command(f"which {tool}", check=False, capture_output=True)
        if result and result.returncode == 0:
            print(f"✅ {tool} is available")
        else:
            print(f"❌ {tool} not found. Please install Docker Desktop.")
            return False
    return True


def start_services():
    """Start all development services"""
    print("🚀 Starting development services...")

    if not check_docker():
        return False

    # Start services
    result = run_command("docker-compose up -d postgres redis")
    if not result:
        return False

    print("⏳ Waiting for services to be ready...")

    # Wait for PostgreSQL to be ready
    for i in range(30):
        result = run_command(
            "docker-compose exec -T postgres pg_isready -U padel_user -d padel_dev_db",
            check=False,
            capture_output=True,
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
        "docker-compose exec -T redis redis-cli ping", check=False, capture_output=True
    )
    if result and result.returncode == 0:
        print("✅ Redis is ready")

    print("\n🎉 Development environment is ready!")
    print("📊 Services running:")
    print("   - PostgreSQL: localhost:5432")
    print("   - Redis: localhost:6379")
    print("\n💡 Next steps:")
    print("   1. uv run python main.py")
    print("   2. Visit: http://localhost:8000/docs")

    return True


def stop_services():
    """Stop all services"""
    print("🛑 Stopping development services...")
    run_command("docker-compose down")
    print("✅ Services stopped")


def reset_services():
    """Reset all services (removes data!)"""
    print("🔄 Resetting development environment...")
    print("⚠️  This will delete all data!")

    response = input("Are you sure? (y/N): ")
    if response.lower() != "y":
        print("Cancelled")
        return

    run_command("docker-compose down -v")  # -v removes volumes (data)
    print("✅ Environment reset")

    # Restart services
    start_services()


def show_status():
    """Show status of services"""
    print("📊 Service Status:")
    run_command("docker-compose ps")

    print("\n🔧 Service Health:")

    # Check PostgreSQL
    result = run_command(
        "docker-compose exec -T postgres pg_isready -U padel_user -d padel_dev_db",
        check=False,
        capture_output=True,
    )
    if result and result.returncode == 0:
        print("✅ PostgreSQL: Healthy")
    else:
        print("❌ PostgreSQL: Not responding")

    # Check Redis
    result = run_command(
        "docker-compose exec -T redis redis-cli ping", check=False, capture_output=True
    )
    if result and result.returncode == 0:
        print("✅ Redis: Healthy")
    else:
        print("❌ Redis: Not responding")


def main():
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    if not check_dependencies():
        sys.exit(1)

    if len(sys.argv) < 2:
        command = "start"
    else:
        command = sys.argv[1]

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
        print("Usage: python scripts/dev-setup.py [start|stop|reset|status]")
        sys.exit(1)


if __name__ == "__main__":
    main()
