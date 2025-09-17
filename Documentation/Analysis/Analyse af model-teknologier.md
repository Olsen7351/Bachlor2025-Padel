# Analyse af modelvalg og teknologier til padel-videoanalyse

## Formål
Projektet har til formål at udvikle en maskinlæringsbaseret pipeline, der kan analysere padel kampe.  
Systemet skal kunne identificere **bolden** og de **fire spillere**, og på baggrund af dette udlede statistik såsom:

- Spillerpositioner og zonedækning  
- Antal boldberøringer pr. spiller  
- Antal fejl  
- Duel-længder  

Denne analyse fokuserer på model- og teknologivalg.

---

## Overordnet arkitektur
Systemet opdeles i følgende komponenter:

1. **Objektdetektion** – Identifikation af bold og spillere i hvert frame.  
2. **Tracking** – Konsistent opretholdelse af identiteter for både spillere og bold gennem hele kampen.  
3. **Banekalibrering** – Projektion af spiller- og boldpositioner ind i banens koordinatsystem.  
4. **Event-detektion og statistik** – Afledning af hændelser (slag, bounce, fejl) og beregning af kampstatistikker.  

Denne modulære opbygning giver fleksibilitet: hver del kan forbedres eller udskiftes, uden at det vælter hele systemet.

---

## Objektdetektion
Til objektdetektion anvender vi **YOLO-familien** (Indtil videre, *YOLOv8*). Disse modeller er hurtige, veldokumenterede og nemme at fintune.  

### Fordele
- Hurtig inferens (real-time muligt på GPU)  
- Mange ressourcer og eksempler tilgængelige  
- Understøtter både små og store modeller (fra edge-devices til cloud)  

### Ulemper
- Små objekter som padelbolden kan være vanskelige at fange, hvilket kræver ekstra tilpasning  

### Tiltag for bold-detektion
- Øget input-opløsning (960–1280 pixels i stedet for standard 640)  
- **SAHI (Slicing Aided Hyper Inference)**, hvor billeder deles i crops under inferens  
- Augmenteringer: motion blur, exposure jitter, syntetisk copy-paste af bolde  

### Alternativer
- **RT-DETR**: Moderne transformer-baseret detektor, høj præcision, men mere kompleks at træne  
- **Detectron2/MMDetection**: Fleksible frameworks med mange modeltyper, men tungere opsætning  

**Konklusion:** Til PoC anvendes **YOLOv8s/n** som primær detektor til både spillere og bold.

---

## Tracking
Når objekterne er detekteret, skal de spores over tid.

### Spillere
- **ByteTrack** vælges til PoC. Den er hurtig, robust og enkel at integrere.  
- Hvis der opstår problemer med ID-switches, kan vi opgradere til **StrongSORT**, som benytter re-identifikation til at holde styr på spillerne, selv når de overlapper eller krydser hinanden.

### Bold
Bolden spores med en kombination af detektioner og en simpel tracker:  
- **YOLO-detektion + Kalman filter**  
- Suppleret med simple fysikregler (maksimal hastighed, banebegrænsninger, bounce)  

Dette giver en stabil og letvægtsløsning.

---

## Banekalibrering
For at oversætte spiller- og boldpositioner til et fælles banekoordinatsystem, kræves homografi-estimering.

- **PoC-løsning:** Manuel markering af banens fire hjørner i hver video.  
- **Fordel:** Hurtigt og sikkert at implementere.  
- **Ulempe:** Skal udføres for hver ny video.  

**Fremtidigt:** Automatisering via klassisk computer vision (linjedetektion) eller et lille keypoint-neuralnet.

---

## Event-detektion og statistik
Når boldens trajectory er identificeret, kan vi begynde at udlede statistik og hændelser:

- **Rally-segmentering:** Perioder hvor bolden er i aktiv bevægelse.  
- **Slag-estimat:** Når boldens retning ændres tæt på en spiller.  
- **Fejl:** Når bolden går ud af banen eller direkte i væg.  
- **Zonetid og heatmaps:** Ved at projektere spillernes positioner på banen kan vi beregne hvor meget tid de bruger i forskellige områder.  
- **Duel-længde:** Antal boldberøringer mellem serves.  

Disse målinger giver direkte indsigt i spilleradfærd og kampforløb.

---

## Evaluering og PoC
Vi evaluerer modellen ud fra både klassiske ML-metrikker og de sportslige KPI’er.

### Detektion
- mAP (mean Average Precision)  
- Recall (særligt vigtigt for bolden)  

### Tracking
- IDF1 og MOTA for spillere  
- Stabilitet af boldsporing  

### Afledte KPI’er
- Zonetid (procentdel)  
- Antal boldberøringer pr. spiller  
- Duel-længde (sekunder eller antal slag)  

### Proof of Concept leverancer
- Et **Script**, der kan:
  1. Køre en video gennem modellen  
  2. Vise detekterede spillere og bold  
  3. Spore dem over tid  
  4. Udregne statistik (zonetid, touches, fejl, dueller)  

---

## Konklusion
Til projektets Proof of Concept vælger vi følgende teknologier:

- **YOLOv8s/n** som detektor (med teknikker til små objekter)  
- **ByteTrack** til spillertracking (opgraderbart til StrongSORT)  
- **YOLO + Kalman filter** til boldtracking  
- **Manuel homografi** til banekalibrering  
- Regelbaseret event-detektion til statistik  

Denne kombination giver en effektiv og realistisk løsning til PoC, som samtidig kan udvides til mere avancerede modeller i fremtidige iterationer.
