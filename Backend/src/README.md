# Padel Analyzer

A Python-based system for analyzing padel match videos using machine learning.

## 🚀 Quick Start with UV

This project uses **UV** as our Python package manager. UV is fast, reliable, and handles virtual environments automatically.

### First Time Setup

1. **Install UV** (if you don't have it):
   ```bash
   # macOS/Linux
   brew install uv
   
   # Windows
   winget install --id=astral-sh.uv -e

   # Docker Desktop
   Install Docker Desktop: https://www.docker.com/products/docker-desktop/
   ```

2. **Clone and setup the project**:
   ```bash
   git clone https://github.com/Olsen7351/Bachlor2025-Padel.git
   cd Backend/src
   uv sync  # Installs all dependencies and creates virtual environment
   ```

3. **Run the application**:
   ```bash
   # Start development environment
   python scripts/dev-setup.py start # Other commands include [stop|reset|status]

   uv run python main.py
   ```

That's it! 🎉

## 📦 Essential UV Commands

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

## 🔄 Team Workflow

### When someone adds a new dependency:
1. They run: `uv add package-name`
2. They commit: `pyproject.toml` and `uv.lock`
3. You run: `git pull && uv sync`

### When you add a new dependency:
1. Run: `uv add package-name`
2. Commit: `pyproject.toml` and `uv.lock`
3. Push your changes

## 🚫 What NOT to Commit

- `.venv/` folder (virtual environment - auto-generated)
- `__pycache__/` folders
- `.env` files (secrets and local config)

## 💡 Why UV?

- **Fast**: Much faster than pip
- **Reliable**: Lock file ensures everyone has identical dependencies
- **Simple**: Handles virtual environments automatically
- **Modern**: Built-in support for modern Python packaging

## 🆘 Common Issues

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

## 📚 Project Structure
***To be updated...***
```
Backend/src/
├── app/
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
│   │   │   ├── player_service.py  # Player business logic
│   │   │   ├── video_service.py   # Video processing logic
│   │   │   ├── match_service.py   # Match management logic
│   │   │   └── analysis_service.py # Analysis business logic
│   │   └── exceptions.py          # Business-specific exceptions
│   │
│   ├── presentation/              # 🌐 Presentation Layer (API)
│   │   ├── dtos/                  # Data Transfer Objects (Pydantic)
│   │   │   ├── __init__.py
│   │   │   ├── player_dto.py      # Player API contracts
│   │   │   ├── video_dto.py       # Video API contracts
│   │   │   ├── match_dto.py       # Match API contracts
│   │   │   └── analysis_dto.py    # Analysis API contracts
│   │   └── controllers/           # FastAPI Controllers
│   │       ├── __init__.py
│   │       ├── player_controller.py    # Player endpoints
│   │       ├── video_controller.py     # Video endpoints
│   │       ├── match_controller.py     # Match endpoints
│   │       └── analysis_controller.py  # Analysis endpoints
│   │
│   ├── main.py                    # FastAPI app initialization
│   └── config.py                  # Application configuration
│
├── scripts/
│   └── dev-setup.py               # Development environment setup
│
├── main.py                        # Application entry point
├── pyproject.toml                 # Project dependencies and configuration
├── .env                          # Environment variables
└── uv.lock                       # Exact dependency versions
```

---

Need help? Check the [UV documentation](https://docs.astral.sh/uv/)