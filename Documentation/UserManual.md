# ViborAI - Padel Videoanalysesystem

## Brugermanual

_Version 1.0 | December 2024_

---

## Indholdsfortegnelse

1. [Introduktion](#introduktion)
2. [Brugervejledning](#brugervejledning)
3. [Analysefunktioner](#analysefunktioner)
4. [Administratorguide](#administratorguide)

---

## Introduktion

ViborAI er et padel videoanalysesystem, der bruger kunstig intelligens til at analysere kampvideoer. Systemet leverer:

- **Slagoptælling** - Overhead-slag, lob, serv og grundslag
- **Heatmaps** - Spillerpositionering på banen
- **Duelanalyse** - Varighed og fordeling af dueller

Systemet sporer **Spiller 1 & 2** (nærmeste side) fuldt ud med AI. Spiller 3 & 4 (fjerneste side) har begrænset sporing.

---

# Del 1: Brugervejledning

## Opret konto og log ind

### Registrering

1. Klik på **"Opret bruger"** på login-siden
2. Udfyld: Navn, Email, Adgangskode (min. 6 tegn)
3. Klik **"Registrer"** → omdirigeres til login

### Login

1. Indtast Email og Adgangskode
2. Klik **"Log ind"** → omdirigeres til Dashboard

---

## Dashboard

Dashboard viser alle dine uploadede videoer med status:

| Status       | Betydning                       |
| ------------ | ------------------------------- |
| `uploaded`   | Venter på analyse               |
| `processing` | AI-analyse i gang               |
| `analyzed`   | Klik for at se resultater       |
| `failed`     | Fejl under analyse              |

---

## Upload video

1. Klik **"Upload kamp"** på Dashboard
2. Vælg videofil (MP4, AVI, MOV, MKV, WEBM)
3. Vælg **Banenummer**
4. Klik **"Vamos!"**

Analysen starter automatisk i baggrunden.

---

## Analysefunktioner

### Dueller

Viser duelstatistik: totale dueller, gennemsnitlig varighed, korteste/længste duel.

Dueller grupperes: Kort (<5s), Medium (5-15s), Lang (15-30s), Meget lang (>30s).

### Slag

Viser slagstatistik per spiller: Total slag, Overhead, Lob, Serve, Grundslag.

### Heatmap

Viser hvor spillerne opholder sig på banen med farvekodning:
- **Rød/Orange**: Høj aktivitet
- **Blå/Grøn**: Lav aktivitet

---

# Del 2: Administratorguide

## Firebase opsætning

1. Opret projekt på [Firebase Console](https://console.firebase.google.com)
2. Aktiver **Authentication** → **Email/Password**
3. Opret servicekonto: **Project Settings** → **Service accounts** → **Generate new private key**
4. Gem værdier fra JSON til `.env`

---

## Backend opsætning

### Installer afhængigheder

```bash
# Windows
winget install Python.Python.3.12 ffmpeg Docker.DockerDesktop astral-sh.uv
```

### Konfigurer miljø

Opret `Backend/src/.env`:

```env
DATABASE_URL=postgresql+asyncpg://padel_user:padel_password@localhost:5432/padel_dev
ENVIRONMENT=development
FIREBASE_PROJECT_ID=<dit-projekt-id>
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=<din-servicekonto-email>
FIREBASE_WEB_API_KEY=<din-web-api-key>
REDIS_URL=redis://localhost:6379/0
```

### Start server

```bash
cd Backend/src
python scripts/dev-setup.py start  # Start PostgreSQL og Redis
uv sync                            # Installer afhængigheder
uv run python main.py              # Start server
```

Server: `http://localhost:8000` | API docs: `http://localhost:8000/docs`

---

## Frontend opsætning

### Konfigurer miljø

Opret `frontend/bachelor2025-padel/.env`:

```env
VITE_FIREBASE_API_KEY=<din-api-key>
VITE_FIREBASE_AUTH_DOMAIN=<projekt>.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=<dit-projekt-id>
VITE_API_BASE_URL=http://localhost:8000/api
```

### Start server

```bash
cd frontend/bachelor2025-padel
npm install
npm run dev
```

Frontend: `http://localhost:5173`

---

## ML-modeller

Placer i `Backend/src/app/business/services/deep_learning/models/`:

| Modelfil           | Formål             |
| ------------------ | ------------------ |
| `TrackNet_best.pt` | Boldsporing        |
| `yolov8s.pt`       | Spillerdetektion   |
| `yolov8n-pose.pt`  | Pose-estimering    |
| `best_model.pth`   | Slagklassifikation |

---

## Porte

| Service    | Port |
| ---------- | ---- |
| Frontend   | 5173 |
| Backend    | 8000 |
| PostgreSQL | 5432 |
| Redis      | 6379 |

---

_Slut på brugermanual_
