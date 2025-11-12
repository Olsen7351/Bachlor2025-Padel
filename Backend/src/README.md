# Padel Analyzer

A Python-based system for analyzing padel match videos using machine learning.

## Quick Start with UV

This project uses **UV** as our Python package manager. UV is fast, reliable, and handles virtual environments automatically.

### First Time Setup

1. **Install UV** (if you don't have it):
```bash
   # macOS/Linux
   brew install uv
   
   # Windows
   winget install --id=astral-sh.uv -e
```

2. **Install Docker Desktop**:
   - Download and install from: https://www.docker.com/products/docker-desktop/

3. **Install ffmpeg** (required for video processing):
```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian/WSL
   sudo apt-get update && sudo apt-get install ffmpeg
   
   # Windows
   winget install ffmpeg
```
   
   Verify installation:
```bash
   ffprobe -version
```

4. **Clone and setup the project**:
```bash
   git clone https://github.com/Olsen7351/Bachlor2025-Padel.git
   cd Backend/src
   uv sync  # Installs all dependencies and creates virtual environment
```

5. **Run the application**:
```bash
   # Start development environment (databases only)
   python scripts/dev-setup.py start  # Other commands: [stop|reset|status]

   # Run the FastAPI application
   uv run python main.py
```

The API will be available at:
- **API Docs**: http://localhost:8000/docs
- **API**: http://localhost:8000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## Essential UV Commands

### Running the Application
First you'll need to setup the development environment with a PostgreSQL database and Redis. This is done easily by having Docker Desktop and running the dev-setup script inside the **Backend/src/scripts** folder. 
```bash
# Run dev-setup script (from root)
python backend/src/scripts/dev-setup.py start

# Navigate to folder path
cd Backend/src/

# Start the FastAPI server
uv run python main.py

# Run with uvicorn directly
uv run uvicorn main:app --reload
```

### Managing Dependencies
```bash
# Add a new package
uv add fastapi

# Add development dependency (testing, linting, etc.)
uv add --dev pytest

# Remove a package
uv remove package-name

# Update all packages
uv sync --upgrade
```

### Development Commands
```bash
# You might need to activate virtual environment to make pytest work
source .venv/bin/activate

# Run tests
uv run pytest

# Format all code
uv run black .

# Check what's installed
uv pip list

# dev-script commands (from root)
python backend/src/scripts/dev-setup.py [start|stop|reset|status]
```

### Working with the Environment
```bash
# Stopping Docker and the containers (from root)
python backend/src/scripts/dev-setup.py stop

# Sync dependencies (after git pull)
uv sync

# Add new dependency and sync
uv add sqlalchemy
```

## Team Workflow

### When someone adds a new dependency:
1. They run: `uv add package-name`
2. They commit: `pyproject.toml` and `uv.lock`
3. You run: `git pull && uv sync`

### When you add a new dependency:
1. Run: `uv add package-name`
2. Commit: `pyproject.toml` and `uv.lock`
3. Push your changes

## What NOT to Commit

- `.venv/` folder (virtual environment - auto-generated)
- `__pycache__/` folders
- `.env` files (secrets and local config)

## Why UV?

- **Fast**: Much faster than pip
- **Reliable**: Lock file ensures everyone has identical dependencies
- **Simple**: Handles virtual environments automatically
- **Modern**: Built-in support for modern Python packaging

## Common Issues

**Environment issues?**
```bash
# Delete .venv and recreate
rm -rf .venv
uv sync
```

**Missing dependencies after git pull?**
```bash
uv sync
```

**Want to see what changed?**
```bash
git diff HEAD~1 uv.lock  # See what packages were added/updated
```

## Project Structure
***To be updated... (estimate for final structure thus far)***
```
Backend/src/
├── app/
│   ├── auth/                      # 🔐 Authentication Layer (Firebase)
│   │   ├── __init__.py
│   │   ├── firebase_service.py    # Firebase Admin SDK integration
│   │   └── dependencies.py        # Auth dependencies (get_current_user, etc.)
│   │
│   ├── domain/                    # 🏛️ Domain Models (Business Entities)
│   │   ├── __init__.py
│   │   ├── player.py              # Player domain entity (@dataclass)
│   │   ├── video.py               # Video domain entity
│   │   ├── match.py               # Match and related entities
│   │   └── analysis.py            # Analysis domain entity
│   │
│   ├── data/                      # 💾 Data Access Layer
│   │   ├── models/                # SQLAlchemy ORM Models
│   │   │   ├── __init__.py        # Imports all models for relationship resolution
│   │   │   ├── base.py            # Base SQLAlchemy declarative class
│   │   │   ├── player_model.py    # Player database model
│   │   │   ├── video_model.py     # Video database model
│   │   │   ├── match_model.py     # Match and related database models
│   │   │   └── analysis_model.py  # Analysis database model
│   │   ├── repositories/          # Repository Pattern Implementation
│   │   │   ├── __init__.py
│   │   │   ├── interfaces.py      # Repository interfaces (IPlayerRepository, etc.)
│   │   │   ├── base_repository.py # Generic base repository with CRUD
│   │   │   ├── player_repository.py
│   │   │   ├── video_repository.py
│   │   │   ├── match_repository.py
│   │   │   └── analysis_repository.py
│   │   └── connection.py          # Database session management
│   │
│   ├── business/                  # ⚙️ Business Logic Layer
│   │   ├── services/              # Business logic implementation
│   │   │   ├── __init__.py
│   │   │   ├── interfaces.py      # Service interfaces (IPlayerService, etc.)
│   │   │   ├── player_service.py  # Player business logic
│   │   │   ├── video_service.py   # Video processing logic
│   │   │   ├── match_service.py   # Match management logic
│   │   │   └── analysis_service.py # Analysis business logic
│   │   └── exceptions.py          # Business-specific exceptions
│   │
│   ├── presentation/              # 🌐 Presentation Layer (API)
│   │   ├── dtos/                  # Data Transfer Objects (Pydantic)
│   │   │   ├── __init__.py
│   │   │   ├── auth_dto.py        # Auth API contracts (Register, Login, etc.)
│   │   │   ├── player_dto.py      # Player API contracts
│   │   │   ├── video_dto.py       # Video API contracts
│   │   │   ├── match_dto.py       # Match API contracts
│   │   │   └── analysis_dto.py    # Analysis API contracts
│   │   └── controllers/           # FastAPI Controllers
│   │       ├── __init__.py
│   │       ├── auth_controller.py      # Auth endpoints (register, login, /me)
│   │       ├── player_controller.py    # Player endpoints
│   │       ├── video_controller.py     # Video endpoints
│   │       ├── match_controller.py     # Match endpoints
│   │       └── analysis_controller.py  # Analysis endpoints
│   │
│   ├── main.py                    # FastAPI app initialization
│   └── config.py                  # Application configuration (Pydantic Settings)
│
├── tests/                         # 🧪 Test Suite
│   ├── __init__.py
│   ├── conftest.py                # Shared pytest fixtures
│   ├── unit/                      # Unit tests (mocked dependencies)
│   │   ├── __init__.py
│   │   ├── business/              # Service layer tests
│   │   │   ├── __init__.py
│   │   │   ├── test_player_service.py  # UC-09 tests
│   │   │   ├── test_match_service.py   # UC-04 tests
│   │   │   └── test_video_service.py   # Video service tests
│   │   ├── data/                  # Repository tests (future)
│   │   │   └── __init__.py
│   │   └── presentation/          # Controller tests
│   │       ├── __init__.py
│   │       ├── test_auth_controller.py    # UC-00, UC-09 controller tests
│   │       ├── test_match_controller.py   # Match controller tests
│   │       └── test_video_controller.py   # Video controller tests
│   └── integration/               # Integration tests (future)
│       └── __init__.py
│
├── scripts/
│   └── dev-setup.py               # Development environment setup
│
├── main.py                        # Application entry point
├── pytest.ini                     # Pytest configuration
├── pyproject.toml                 # Project dependencies and configuration
├── .env                          # Environment variables (not tracked)
├── .gitignore                     # Git ignore rules
└── uv.lock                       # Exact dependency versions (tracked)
```

---

Need help? Check the [UV documentation](https://docs.astral.sh/uv/)