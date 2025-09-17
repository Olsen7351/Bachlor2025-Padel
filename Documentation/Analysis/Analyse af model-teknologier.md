# Samlet Analyse: Dataforberedelse, Modelvalg og Teknologier til Padel-Videoanalyse

## 1. Formål og Overordnet Arkitektur

Projektet har til formål at udvikle en maskinlæringsbaseret pipeline, der kan analysere padelkampe. Systemet skal kunne identificere bolden og de fire spillere, og på baggrund af dette udlede statistik såsom spillerpositioner, zonedækning, antal boldberøringer pr. spiller, antal fejl og duel-længder.

Systemet opdeles i følgende komponenter:

1. **Dataindsamling og Annotation**
2. **Objektdetektion**
3. **Tracking**
4. **Banekalibrering**
5. **Event-detektion og statistik**
6. **Evaluering og Test**

---

## 2. Dataindsamling, Annotation og Forberedelse

- Video input fra centerkamera (5-7m højde, 30-45° vinkel), 720p-1080p opløsning, 25-30 FPS.
- Frame extraction via OpenCV (1 FPS sampling), strategisk udvalgte 2-3 minutters segmenter for variation.
- Manuel annotation i Roboflow med seks objektklasser: player_1-4 (individuelle spillere med ID-tracking), racket og ball.
- Annotation guidelines: Bounding boxes med max 10% padding, bolden annoteres med min. 15x15 pixels.
- Data lagres i YOLO-format med normaliserede koordinater og class mappings: 0-3 (spillere), 4 (racket), 5 (bold).
- Dataset split: 70% træning, 15% validering, 15% test (opdelt på video-niveau for at undgå data leakage).
- Skalering: PoC kræver 200-300 frames, MVP 2.000-3.000 frames, produktion 10.000+ frames.

---

## 3. Modelvalg og Teknologier

### Objektdetektion

- **YOLOv8s/n** vælges som primær detektor for både spillere og bold.
- Fordele: Hurtig inferens, veldokumenteret, nem at fintune, understøtter edge/cloud.
- Ulemper: Små objekter (bolden) kræver ekstra tilpasning.
- Tiltag for bold-detektion: Øget input-opløsning, SAHI (Slicing Aided Hyper Inference), augmenteringer (motion blur, exposure jitter, syntetisk copy-paste).
- Alternativer: RT-DETR, Detectron2/MMDetection (mere komplekse, tungere opsætning).

### Tracking

- **ByteTrack** vælges til spillertracking (hurtig, robust, enkel integration).
- Opgradering til **StrongSORT** muligt ved ID-switch problemer.
- Boldtracking: YOLO-detektion + Kalman filter, suppleret med simple fysikregler (maksimal hastighed, banebegrænsninger, bounce).

### Banekalibrering

- PoC: Manuel markering af banens fire hjørner i hver video (homografi).
- Fremtid: Automatisering via klassisk computer vision eller keypoint-netværk.

### Event-detektion og Statistik

- Regelbaseret event-detektion: Rally-segmentering, slag-estimat, fejl, zonetid/heatmaps, duel-længde.
- Statistikker: Zonetid, antal boldberøringer pr. spiller, duel-længde.

---

## 4. Udfordringer og Mitigation

- **Bold detection**: Lille størrelse håndteres via minimum bounding box og strategisk frame selection.
- **Spiller tracking**: ID-konsistens sikres via spatial position og holdtilhørsforhold.
- **Skygger**: Ekskluderet i annotation, træningsdata inkluderer varierede skyggeforhold.
- **Begrænset data**: Transfer learning med pre-trained weights, konservativ augmentation (horizontal flip, brightness, scale).

---

## 5. Træning, Validering og Test

- Data augmentation pipeline implementeres.
- Træning med early stopping og monitoring af validation metrics.
- Hyperparameter tuning (learning rate, batch size).
- Test metrics: mAP@0.5 (spillere: 80-90%, bold: 50-70%), <10% false positives, >15 FPS inference.
- Cross-validation: K-fold (k=3) på video-niveau.
- Kvalitativ analyse: Failure case kategorisering, temporal consistency verifikation.

---

## 6. Implementeringsplan

- **Uge 1**: Dataindsamling (5-10 videoer, frame extraction)
- **Uge 2**: Annotation (200-300 frames, kvalitetskontrol)
- **Uge 3**: Model træning (YOLOv8, monitoring, checkpointing)
- **Uge 4**: Evaluering (test set analyse, cross-validation, dokumentation)

---

## 7. Begrænsninger og Success Kriterier

- **Begrænsninger**: Inkonsistent bold detection, reduceret performance ved spilleroverlap, begrænset til center-view.
- **Success kriterier**: Minimum 3+ spillere detekteret konsistent, funktionel på standard GPU, skalerbar med mere data.

---

## 8. Proof of Concept Leverancer

- Script, der kan:
  1. Køre en video gennem modellen
  2. Vise detekterede spillere og bold
  3. Spore dem over tid
  4. Udregne statistik (zonetid, touches, fejl, dueller)

---

## 9. Task List for Implementation

### Dataindsamling

- Opsæt Roboflow konto og projektstruktur
- Indsaml 5-10 padelvideoer fra centerkamera
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
- Udfør k-fold cross-validation
- Analysér failure cases og kategorisér fejltyper
- Dokumentér resultater og anbefalinger
