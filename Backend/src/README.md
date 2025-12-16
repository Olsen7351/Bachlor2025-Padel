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
│   ├── auth/                           # Authentication Layer
│   │   ├── __init__.py
│   │   ├── firebase_service.py         # Firebase Admin SDK integration
│   │   └── dependencies.py             # Auth dependencies (get_current_user)
│   │
│   ├── domain/                         # Domain Models (Business Entities)
│   │   ├── __init__.py
│   │   ├── player.py                   # Player domain entity (@dataclass)
│   │   ├── video.py                    # Video domain entity
│   │   ├── match.py                    # Match domain entity
│   │   └── analysis.py                 # Analysis domain entity
│   │
│   ├── data/                           # Data Access Layer
│   │   ├── models/                     # SQLAlchemy ORM Models
│   │   │   ├── __init__.py             # Imports all models for relationship resolution
│   │   │   ├── base.py                 # Base SQLAlchemy declarative class
│   │   │   ├── player_model.py
│   │   │   ├── video_model.py
│   │   │   ├── match_model.py
│   │   │   └── analysis_model.py
│   │   ├── repositories/               # Repository Pattern Implementation
│   │   │   ├── __init__.py
│   │   │   ├── interfaces.py           # Repository interfaces (ABC)
│   │   │   ├── base_repository.py      # Generic CRUD base
│   │   │   ├── player_repository.py
│   │   │   ├── video_repository.py
│   │   │   ├── match_repository.py
│   │   │   ├── heatmap_repository.py
│   │   │   ├── rally_repository.py
│   │   │   ├── hits_repository.py
│   │   │   ├── summary_metrics_repository.py
│   │   │   └── analysis_repository.py
│   │   └── connection.py               # Database session management
│   │
│   ├── business/                       # Business Logic Layer
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── interfaces.py           # Service interfaces (ABC)
│   │   │   ├── player_service.py
│   │   │   ├── video_service.py
│   │   │   ├── match_service.py
│   │   │   ├── heatmap_service.py
│   │   │   ├── rally_service.py
│   │   │   ├── analysis_service.py
│   │   │   ├── ml_service.py           # ML orchestration
│   │   │   ├── file_storage.py         # File system operations
│   │   │   ├── video_converter.py      # Video processing utilities
│   │   │   │
│   │   │   └── deep_learning/             # ML Components
│   │   │      ├── architectures/          # Neural network definitions
│   │   │      │   └── shot_classifier.py
│   │   │      ├── court_info/             # Court calibration data
│   │   │      │   └── court_information.json
│   │   │      ├── models/                 # Trained model weights
│   │   │      │   ├── TrackNet_best.pt
│   │   │      │   ├── yolov8n-pose.pt
│   │   │      │   └── yolov8s.pt
│   │   │      ├── processors/             # Video processing
│   │   │      │   ├── export.py
│   │   │      │   └── shot_processor.py
│   │   │      ├── trackers/               # Object tracking
│   │   │      │   ├── ball_tracker.py
│   │   │      │   ├── player_tracker.py
│   │   │      │   ├── rally_tracker.py
│   │   │      │   └── simple_tracker.py
│   │   │      ├── utils/                  # ML utilities
│   │   │      │   ├── calibration_utils.py
│   │   │      │   ├── heatmap_utils.py
│   │   │      │   ├── inference_utils.py
│   │   │      │   └── video_utils.py
│   │   │      ├── results/                # Results processing
│   │   │      │   └── heatmap_results.py
│   │   │      ├── config.py               # ML configuration
│   │   │      ├── inference.py            # Inference engine
│   │   │      └── pipeline.py             # ML pipeline orchestration
│   │   │
│   │   └── exceptions.py               # Business-specific exceptions
│   │
│   ├── presentation/                   # Presentation Layer (API)
│   │   ├── dtos/                       # Data Transfer Objects (Pydantic)
│   │   │   ├── __init__.py
│   │   │   ├── auth_dto.py
│   │   │   ├── player_dto.py
│   │   │   ├── video_dto.py
│   │   │   ├── match_dto.py
│   │   │   ├── heatmap_dto.py
│   │   │   └── rally_dto.py
│   │   └── controllers/                # FastAPI Routers
│   │       ├── __init__.py
│   │       ├── auth_controller.py
│   │       ├── player_controller.py
│   │       ├── video_controller.py
│   │       ├── match_controller.py
│   │       ├── heatmap_controller.py
│   │       └── rally_controller.py
│   │
│   ├── main.py                         # FastAPI app initialization
│   ├── config.py                       # Application configuration
│   └── dependencies.py                 # Global dependency factories
│
├── tests/                              # Test Suite
│   ├── conftest.py                     # Shared pytest fixtures
│   ├── unit/
│   │   ├── business/                   # Service layer tests
│   │   │   ├── test_match_service.py
│   │   │   ├── test_player_service.py
│   │   │   └── test_video_service.py
│   │   ├── data/                       # Repository tests
│   │   │   └── __init__.py
│   │   └── presentation/               # Controller tests
│   │       ├── test_auth_controller.py
│   │       ├── test_match_controller.py
│   │       └── test_video_controller.py
│   └── integration/                    # Integration tests
│       └── __init__.py
│
├── scripts/
│   └── dev-setup.py                    # Development setup script
│
├── uploads/                            # Video upload directory
│   └── videos/
│
├── main.py                             # Application entry point
├── pytest.ini                          # Pytest configuration
├── pyproject.toml                      # Project dependencies
├── .env                                # Environment variables (not tracked)
├── .env.example                        # Environment template
└── uv.lock                             # Dependency lock file
```

---

Need help? Check the [UV documentation](https://docs.astral.sh/uv/)