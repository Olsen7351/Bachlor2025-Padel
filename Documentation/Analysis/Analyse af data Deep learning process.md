# Dataanalyse for Padel Video Detection System

## 1. Systemarkitektur og Pipeline

Systemet implementerer en computer vision pipeline bestående af video input fra stationært centerkamera (5-7m højde, 30-45° vinkel), frame extraction via OpenCV (1 FPS sampling), manuel annotation i Roboflow, og YOLOv8 model træning. Videodata indsamles i 720p-1080p opløsning med 25-30 FPS under konsistent LED belysning. Produktionsvideoer varierer mellem 5-60 minutter, hvorfra strategiske 2-3 minutters segmenter ekstraheres for maksimal variation i spillesituationer.

## 2. Annotation Protokol og Objektklasser

Systemet definerer seks objektklasser: player_1-4 (individuelle spillere med konsistent ID-tracking), racket (ketsjere), og ball. Player_1-2 repræsenterer hold A, player_3-4 hold B. Bounding boxes annoteres med maksimalt 10% padding, hvor spillere trackes gennem hele videosekvensen baseret på position og visuelle features. Bolden annoteres med minimum 15x15 pixels for at kompensere for lille størrelse i top-view (typisk 5-10 pixels).

## 3. Dataset Struktur og Skalering

**PoC Dataset**: 200-300 frames fra 5-10 videoer opdelt i 70% træning (140-210), 15% validering (30-45), 15% test (30-45). Opdeling sker på video-niveau for at undgå data leakage.
**Skalering**: MVP kræver 2,000-3,000 frames (30-50 videoer), Production 10,000+ frames (200+ videoer).
Data lagres i YOLO format med normaliserede koordinater og class mappings: 0-3 (spillere), 4 (racket), 5 (bold).

## 4. Identificerede Udfordringer og Mitigation

**Bold detection**: Lille størrelse (5-10 pixels) håndteres gennem minimum bounding box og strategisk frame selection.
**Spiller tracking**: ID-konsistens gennem videoer sikres via spatial position og holdtilhørsforhold verifikation.
**Skygger**: Eksplicit ekskluderet i annotation, træningsdata inkluderer varierede skyggeforhold.
**Begrænset data**: Transfer learning med pre-trained YOLO weights og konservativ augmentation (horizontal flip 50%, brightness ±20%, scale ±10%).

## 5. Validering og Test Protokol

**Validering**: Anvendes til hyperparameter tuning (learning rate 1e-3 til 1e-4, batch size 8-16) og early stopping (20 epochs uden forbedring).
**Test metrics**: mAP@0.5 target 80-90% (spillere), 50-70% (bold), <10% false positives, >15 FPS inference.
**Cross-validation**: K-fold (k=3) på video-niveau for robust performance estimering.
Kvalitativ analyse inkluderer failure case kategorisering og temporal consistency verifikation.

## 6. Implementeringsplan

**Uge 1**: Dataindsamling - 5-10 videoer, frame extraction (30-50 per video)
**Uge 2**: Annotation - 200-300 frames med kvalitetskontrol
**Uge 3**: Model træning - YOLOv8 med monitoring og checkpointing
**Uge 4**: Evaluering - Test set analyse, cross-validation, dokumentation

## 7. Forventede Begrænsninger og Success Kriterier

**Begrænsninger**: Inkonsistent bold detection, reduceret performance ved spilleroverlap, begrænset til center-view perspektiv.
**Success kriterier**: Minimum 3+ spillere detekteret konsistent, funktionel på standard GPU, skalerbar med yderligere data.
Systemet demonstrerer teknisk feasibility for padel analyse med mulighed for udvidelse til multi-camera setup og real-time analytics.

## 8. Task List for Implementation

### Dataindsamling

- Opsæt Roboflow konto og projekt struktur
- Indsaml 5-10 padel videoer (5-60 min) fra center kamera
- Verificer video kvalitet (min. 720p, stabil position)
- Ekstraher 30-50 frames per video via OpenCV
- Organisér frames i mappestruktur

### Annotation

- Definér annotation guidelines dokument
- Annotér player_1-4, racket, ball i 200-300 frames
- Verificer ID-konsistens gennem video sekvenser
- Kvalitetskontrol på 20% tilfældige frames
- Eksportér annotations i YOLO format

### Model Træning

- Split dataset (70/15/15)
- Konfigurér YOLOv8 med pre-trained weights
- Implementér data augmentation pipeline
- Træn model med early stopping
- Monitor validation metrics og gem best checkpoint

### Evaluering

- Kør inference på test set
- Beregn mAP, precision, recall per klasse
- Udført k-fold cross-validation
- Analysér failure cases og kategorisér fejltyper
- Dokumentér resultater og anbefalinger
