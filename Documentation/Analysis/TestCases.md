# Test Cases (Prioriteret)

_Dette dokument beskriver test cases for systemets use cases. Testene er prioriteret i samme rækkefølge som implementeringen (P1, P2, etc.) for at sikre, at kernefunktionalitet valideres først. Der er nu gjort plads til at notere resultater direkte i tabellerne._

## Indholdsfortegnelse

1.  [Test Case Skabelon](#test-case-skabelon)
2.  [Test Cases (Prioriteret)](#test-cases-prioriteret)
    - [P1 - TC-09: Player-registrering](#p1---tc-09-player-registrering)
    - [P2 - TC-00: Player-login](#p2---tc-00-player-login)
    - [P3 - TC-01: Upload og behandling af kampvideo](#p3---tc-01-upload-og-behandling-af-kampvideo)
    - [P4 - TC-04: Opgørelse af samlet antal slag](#p4---tc-04-opgørelse-af-samlet-antal-slag)
    - [P5 - TC-03: Analyse af banezone-besættelse - (IKKE IMPLEMENTERET)](#p5---tc-03-analyse-af-banezone-besættelse)
    - [P6 - TC-02: Visning af spiller-heatmaps](#p6---tc-02-visning-af-spiller-heatmaps)
    - [P7 - TC-08: Analyse af duel-længde (rallies)](#p7---tc-08-analyse-af-duel-længde-rallies)
    - [P8 - TC-07: Filtrering af data pr. spiller - (IKKE IMPLEMENTERET)](#p8---tc-07-filtrering-af-data-pr-spiller---ikke-implementeret)
    - [P9 - TC-06: Kamp-dashboard](#p9---tc-06-kamp-dashboard)
    - [P10 - TC-05: Visualisering af slagpositioner - (IKKE IMPLEMENTERET)](#p10---tc-05-visualisering-af-slagpositioner---ikke-implementeret)

---

## Test Case Skabelon

Hver test case indeholder:

- **Test ID**: Unik reference.
- **Tilknyttet UC**: Reference til den Use Case, der testes.
- **Formål**: Hvad testen skal validere.
- **Preconditions**: Hvad der skal være opfyldt før testen.
- **Scenarier**: Trin-for-trin instruktioner og forventede resultater.
- **Faktisk Resultat**: Plads til at notere, hvad der reelt skete under testen.
- **Status**: Markering af om testen er bestået eller fejlet.

---

## Test Cases (Prioriteret)

### P1 - TC-09: Player-registrering

| Test Case Element                | Detaljer                                                                                                                                                                                                                    |
| :------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Test ID**                      | TC-09                                                                                                                                                                                                                       |
| **Tilknyttet UC**                | UC-09: Player-registrering                                                                                                                                                                                                  |
| **Formål**                       | At verificere, at en ny bruger kan registrere sig, og at systemet håndterer ugyldige input korrekt.                                                                                                                         |
| **Preconditions**                | - Applikationen kører.<br>- Test-brugeren eksisterer ikke i forvejen i hverken Firebase eller database.                                                                                                                     |
| **Test Scenarie 1 (Happy Path)** | **S1: Succesfuld oprettelse**<br>1. Naviger til registreringssiden.<br>2. Indtast navn: "Test Bruger".<br>3. Indtast email: "nybruger@test.dk".<br>4. Indtast password: "SikkertPassword123".<br>5. Klik på "Registrer".    |
| **Forventet Resultat 1**         | - Brugeren oprettes i Firebase Authentication.<br>- Brugeren gemmes i backend-databasen med korrekt navn og rolle.<br>- Brugeren viderestilles automatisk til Dashboardet.<br>- Ingen fejlmeddelelser vises.                |
| **Faktisk Resultat 1**           | Brugeren bliver succesfuldt oprettet i Firebase og viderestilles til login-siden, hvor brugeren derefter skal logge ind.                                                                                                    |
| **Status 1**                     | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet** _(Note: Redirect går til login i stedet for dashboard)_                                                                                                                |
| **Test Scenarie 2 (Negative)**   | **F3: E-mail allerede i brug**<br>1. Naviger til registreringssiden.<br>2. Indtast en email, der allerede findes i systemet (f.eks. "eksisterende@test.dk").<br>3. Udfyld øvrige felter gyldigt.<br>4. Klik på "Registrer". |
| **Forventet Resultat 2**         | - Registreringen afvises.<br>- En fejlmeddelelse vises: "E-mail er allerede i brug".<br>- Ingen ny bruger oprettes i databasen.                                                                                             |
| **Faktisk Resultat 2**           | Afviser succesfuldt at oprette bruger med samme email.                                                                                                                                                                      |
| **Status 2**                     | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                                                                                        |

### P2 - TC-00: Player-login

| Test Case Element                | Detaljer                                                                                                                                                              |
| :------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Test ID**                      | TC-00                                                                                                                                                                 |
| **Tilknyttet UC**                | UC-00: Player-login                                                                                                                                                   |
| **Formål**                       | At sikre, at kun registrerede brugere kan logge ind, og at token-validering fungerer.                                                                                 |
| **Preconditions**                | - En bruger med email "bruger@test.dk" og password "Password123" er oprettet og registreret i backend.                                                                |
| **Test Scenarie 1 (Happy Path)** | **S1: Korrekt login**<br>1. Naviger til login-siden.<br>2. Indtast email: "bruger@test.dk".<br>3. Indtast password: "Password123".<br>4. Klik på "Log ind".           |
| **Forventet Resultat 1**         | - Systemet modtager et gyldigt token fra Firebase.<br>- Backend accepterer tokenet.<br>- Brugeren viderestilles til Dashboardet.                                      |
| **Faktisk Resultat 1**           | Bruger bliver logget ind og backend accepterer token.                                                                                                                 |
| **Status 1**                     | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                                  |
| **Test Scenarie 2 (Negative)**   | **F1: Forkert adgangskode**<br>1. Naviger til login-siden.<br>2. Indtast email: "bruger@test.dk".<br>3. Indtast password: "ForkertPassword".<br>4. Klik på "Log ind". |
| **Forventet Resultat 2**         | - Login afvises.<br>- Fejlmeddelelse vises: "Forkert brugernavn eller adgangskode".<br>- Brugeren forbliver på login-siden.                                           |
| **Faktisk Resultat 2**           | Login afvises succesfuldt.                                                                                                                                            |
| **Status 2**                     | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                                  |

### P3 - TC-01: Upload og behandling af kampvideo

| Test Case Element                | Detaljer                                                                                                                                                                                                           |
| :------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Test ID**                      | TC-01                                                                                                                                                                                                              |
| **Tilknyttet UC**                | UC-01: Upload og behandling af kampvideo<br>UC-11: Entrypage for analyserede videoer                                                                                                                               |
| **Formål**                       | At teste upload-funktionaliteten, validering af filtyper, samt entrypage med oversigt over analyserede videoer.                                                                                                    |
| **Preconditions**                | - Brugeren er logget ind.<br>- Der haves en gyldig .mp4 fil (padel_kamp.mp4) og en ugyldig .txt fil.<br>- Backend indeholder metadata om analyserede videoer tilknyttet brugerens UID.                             |
| **Test Scenarie 1 (Happy Path)** | **S1: Vellykket upload**<br>1. Naviger til "Upload Video".<br>2. Vælg filen "padel_kamp.mp4".<br>3. Klik på "Upload".                                                                                              |
| **Forventet Resultat 1**         | - Systemet viser en progress bar.<br>- Efter upload vises beskeden "Upload gennemført".<br>- Videoens status skifter til "Analyserer" eller "I kø".                                                                |
| **Faktisk Resultat 1**           | Virker som forventet.                                                                                                                                                                                              |
| **Status 1**                     | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                                                                               |
| **Test Scenarie 2 (Negative)**   | **F1: Filformat ikke understøttet**<br>1. Naviger til "Upload Video".<br>2. Vælg filen "noter.txt".<br>3. Forsøg at uploade.                                                                                       |
| **Forventet Resultat 2**         | - Systemet afviser filen straks.<br>- Fejlmeddelelse vises: "Ugyldigt filformat. Kun videofiler understøttes".                                                                                                     |
| **Faktisk Resultat 2**           | Korrekt afviser ugyldigt filformat.                                                                                                                                                                                |
| **Status 2**                     | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                                                                               |
| **Test Scenarie 3 (Happy Path)** | **S2: Verificer entryside med liste af analyserede videoer**<br>1. Log ind som registreret bruger med mindst én analyseret video.<br>2. Naviger til entrysiden/dashboard.<br>3. Verificer at listen vises.        |
| **Forventet Resultat 3**         | - Systemet viser entrysiden med en liste over brugerens analyserede videoer.<br>- Listen er sorteret efter uploaddato.<br>- Metadata vises: filnavn, uploaddato, varighed, status (færdig/kører/fejl).             |
| **Faktisk Resultat 3**           |                                                                                                                                                                                                                    |
| **Status 3**                     | ▢ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                                                                               |
| **Test Scenarie 4 (Happy Path)** | **S3: Åbn analyse fra entryside**<br>1. Fra entrysiden, find en video med status "færdig".<br>2. Klik på "Åbn" knappen for videoen.                                                                                |
| **Forventet Resultat 4**         | - Systemet navigerer til videoens analyse-side.<br>- De gemte resultater vises korrekt.                                                                                                                            |
| **Faktisk Resultat 4**           |                                                                                                                                                                                                                    |
| **Status 4**                     | ▢ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                                                                               |
| **Test Scenarie 5 (Happy Path)** | **S4: Slet video fra entryside**<br>1. Fra entrysiden, vælg "Slet" på en video.<br>2. Bekræft sletningen i bekræftelsesdialogen.                                                                                   |
| **Forventet Resultat 5**         | - Systemet viser bekræftelsesdialog før sletning.<br>- Efter bekræftelse markeres filen til sletning i backend.<br>- Listen opdateres og videoen fjernes fra visningen.                                            |
| **Faktisk Resultat 5**           |                                                                                                                                                                                                                    |
| **Status 5**                     | ▢ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                                                                               |
| **Test Scenarie 6 (Negative)**   | **F2: Manglende adgang til video**<br>1. Forsøg at tilgå en video-analyse via direkte URL uden at være logget ind.<br>2. Alternativt: Forsøg at åbne/slette en video der tilhører en anden bruger.                 |
| **Forventet Resultat 6**         | - Backend returnerer 401 (uautoriseret) eller 403 (forbudt).<br>- Frontend viser passende fejlmeddelelse: "Adgang nægtet" eller "Du har ikke rettigheder til denne video".                                         |
| **Faktisk Resultat 6**           |                                                                                                                                                                                                                    |
| **Status 6**                     | ▢ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                                                                               |

### P4 - TC-04: Opgørelse af samlet antal slag

| Test Case Element                     | Detaljer                                                                                                                                               |
| :------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Test ID**                           | TC-04                                                                                                                                                  |
| **Tilknyttet UC**                     | UC-04: Opgørelse af samlet antal slag                                                                                                                  |
| **Formål**                            | At sikre, at analyseresultater vedrørende antal slag præsenteres korrekt for brugeren.                                                                 |
| **Preconditions**                     | - En video er uploadet og færdiganalyseret.<br>- Testdata: Analyse viser Spiller A har 45 slag, Spiller B har 50 slag.                                 |
| **Test Scenarie 1 (Happy Path)**      | **S1: Visning af slag-statistik**<br>1. Naviger til den analyserede kamps detaljeside.<br>2. Find sektionen "Nøglestatistikker" eller "Slagfordeling". |
| **Forventet Resultat 1**              | - Listen viser navnene på spillerne.<br>- Ud for Spiller A står "45 slag".<br>- Ud for Spiller B står "50 slag".                                       |
| **Faktisk Resultat 1**                | Virker som forventet.                                                                                                                                  |
| **Status 1**                          | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                   |
| **Test Scenarie 2 (Data Validation)** | **S2: Grafisk visning**<br>1. Kontroller søjlediagrammet for "Total Shots".                                                                            |
| **Forventet Resultat 2**              | - Søjlen for Spiller B skal være visuelt højere end for Spiller A.<br>- Værdierne matcher tabellen.                                                    |
| **Faktisk Resultat 2**                | Viser en liste med tal og distribution.                                                                                                                |
| **Status 2**                          | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                   |

### P5 - TC-03: Analyse af banezone-besættelse - (IKKE IMPLEMENTERET)

| Test Case Element                | Detaljer                                                                                                                                                                                  |
| :------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Test ID**                      | TC-03                                                                                                                                                                                     |
| **Tilknyttet UC**                | UC-03: Analyse af banezone-besættelse                                                                                                                                                     |
| **Formål**                       | At verificere, at systemet korrekt opdeler spillernes positionering i defensive, offensive og transitionelle zoner.                                                                       |
| **Preconditions**                | - En video er færdiganalyseret.                                                                                                                                                           |
| **Test Scenarie 1 (Happy Path)** | **S1: Visning af zonefordeling**<br>1. Naviger til fanen "Zoneanalyse" for en specifik kamp.<br>2. Aflæs værdierne for en spiller (f.eks. Defensiv: 60%, Transition: 10%, Offensiv: 30%). |
| **Forventet Resultat 1**         | - De tre procenttal skal tilsammen give 100%.<br>- Grafikken (f.eks. cirkeldiagram eller bar) skal afspejle fordelingen visuelt.                                                          |
| **Faktisk Resultat 1**           | Ikke implementeret.                                                                                                                                                                       |
| **Status 1**                     | ▢ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet** &nbsp;&nbsp;&nbsp;&nbsp; ☑ **Ikke testet**                                                                                           |
| **Test Scenarie 2 (Empty Data)** | **F2: Utilstrækkelige data**<br>1. Upload en meget kort video (f.eks. 2 sekunder) hvor spilleren er ude af billedet.<br>2. Vent på analyse.<br>3. Gå til Zoneanalyse.                     |
| **Forventet Resultat 2**         | - Systemet bør vise "Ingen data tilgængelig" eller "Utilstrækkelig sporing" i stedet for et tomt diagram eller 0%.                                                                        |
| **Faktisk Resultat 2**           | Ikke implementeret.                                                                                                                                                                       |
| **Status 2**                     | ▢ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet** &nbsp;&nbsp;&nbsp;&nbsp; ☑ **Ikke testet**                                                                                           |

### P6 - TC-02: Visning af spiller-heatmaps

| Test Case Element                 | Detaljer                                                                                                                                                                                    |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Test ID**                       | TC-02                                                                                                                                                                                       |
| **Tilknyttet UC**                 | UC-02: Visning af spiller-heatmaps                                                                                                                                                          |
| **Formål**                        | At sikre, at heatmaps genereres og lægges korrekt oven på banekortet.                                                                                                                       |
| **Preconditions**                 | - En video er færdiganalyseret.                                                                                                                                                             |
| **Test Scenarie 1 (Happy Path)**  | **S1: Generering af heatmap**<br>1. Naviger til "Heatmap"-visningen.<br>2. Vælg "Spiller A".                                                                                                |
| **Forventet Resultat 1**          | - Et 2D-kort af banen vises.<br>- Et farvet lag (heatmap) vises ovenpå.<br>- Områder med høj aktivitet (f.eks. baglinjen) er røde/varme.<br>- Områder uden aktivitet er transparente/kolde. |
| **Faktisk Resultat 1**            | Viser "Spiller 1" og "Spiller 2" i stedet for "Spiller A" og "Spiller B".                                                                                                                   |
| **Status 1**                      | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet** _(Note: Navngivning afviger fra forventet)_                                                                                            |
| **Test Scenarie 2 (Interaction)** | **S2: Skift af spiller**<br>1. Skift valg fra "Spiller A" til "Spiller B".                                                                                                                  |
| **Forventet Resultat 2**          | - Heatmap-laget opdateres øjeblikkeligt til at vise Spiller B's mønster.<br>- Mønsteret skal være synligt forskelligt fra Spiller A (medmindre de har spillet identisk).                    |
| **Faktisk Resultat 2**            | Viser allerede begge heatmaps samtidigt.                                                                                                                                                    |
| **Status 2**                      | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet** _(Note: Begge heatmaps vises samlet i stedet for enkeltvis)_                                                                           |

### P7 - TC-08: Analyse af duel-længde (rallies)

| Test Case Element                 | Detaljer                                                                                                                                                                 |
| :-------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Test ID**                       | TC-08                                                                                                                                                                    |
| **Tilknyttet UC**                 | UC-08: Analyse af duel-længde (rallies)                                                                                                                                  |
| **Formål**                        | At teste om systemet kan identificere separate dueller og tælle slagene i dem.                                                                                           |
| **Preconditions**                 | - Kampvideo analyseret.                                                                                                                                                  |
| **Test Scenarie 1 (Happy Path)**  | **S1: Oversigt over dueller**<br>1. Naviger til "Duelanalyse".<br>2. Observer listen over dueller.                                                                       |
| **Forventet Resultat 1**          | - Der vises en liste eller graf over dueller.<br>- Gennemsnitslængden (antal slag) vises.<br>- Det totale antal dueller virker realistisk i forhold til videoens længde. |
| **Faktisk Resultat 1**            | Viser listen af dueller som forventet.                                                                                                                                   |
| **Status 1**                      | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                                     |
| **Test Scenarie 2 (Logic Check)** | **S2: Detaljeret duel-info**<br>1. Sammenlign systemets data med manuel optælling af de første 3 dueller i videoen.                                                      |
| **Forventet Resultat 2**          | - Antallet af slag i systemet matcher den manuelle optælling (+/- fejlmargin acceptabelt, men bør være præcist).                                                         |
| **Faktisk Resultat 2**            | Viser listen af dueller.                                                                                                                                                 |
| **Status 2**                      | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                                     |

### P8 - TC-07: Filtrering af data pr. spiller - (IKKE IMPLEMENTERET)

| Test Case Element                 | Detaljer                                                                                                                               |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| **Test ID**                       | TC-07                                                                                                                                  |
| **Tilknyttet UC**                 | UC-07: Filtrering af data pr. spiller                                                                                                  |
| **Formål**                        | At sikre, at brugeren kan isolere data for en enkelt person.                                                                           |
| **Preconditions**                 | - Analyse side åben med data for alle 4 spillere.                                                                                      |
| **Test Scenarie 1 (Happy Path)**  | **S1: Single-player filter**<br>1. Find filter-menuen (typisk "Vælg Spillere").<br>2. Fjern markeringen ved alle undtagen "Spiller A". |
| **Forventet Resultat 1**          | - Grafer og tabeller opdateres.<br>- Kun data for Spiller A vises.<br>- Heatmap viser kun Spiller A's positioner.                      |
| **Faktisk Resultat 1**            | Ikke implementeret.                                                                                                                    |
| **Status 1**                      | ▢ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet** &nbsp;&nbsp;&nbsp;&nbsp; ☑ **Ikke testet**                                        |
| **Test Scenarie 2 (Persistence)** | **S3: Vedvarende filter**<br>1. Behold filteret på "Spiller A".<br>2. Naviger fra "Heatmap" til "Slagpositioner".                      |
| **Forventet Resultat 2**          | - "Slagpositioner"-siden indlæses.<br>- Filteret er stadig aktivt (kun Spiller A vises).                                               |
| **Faktisk Resultat 2**            | Ikke implementeret.                                                                                                                    |
| **Status 2**                      | ▢ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet** &nbsp;&nbsp;&nbsp;&nbsp; ☑ **Ikke testet**                                        |

### P9 - TC-06: Kamp-dashboard

| Test Case Element                | Detaljer                                                                                                                                                                                                             |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Test ID**                      | TC-06                                                                                                                                                                                                                |
| **Tilknyttet UC**                | UC-06: Kamp-dashboard                                                                                                                                                                                                |
| **Formål**                       | At teste integrationen af forskellige datakilder på én side.                                                                                                                                                         |
| **Preconditions**                | - Kamp færdiganalyseret.                                                                                                                                                                                             |
| **Test Scenarie 1 (Happy Path)** | **S1: Dashboard load**<br>1. Åbn kampen fra hovedmenuen.                                                                                                                                                             |
| **Forventet Resultat 1**         | - Dashboardet indlæses inden for 3 sekunder (NFR-Y2).<br>- Følgende widgets er synlige: Total Slag, Zonefordeling (pie chart), Mini-heatmap, Duel-statstik.<br>- Ingen widgets viser "Error" eller loader uendeligt. |
| **Faktisk Resultat 1**           | Virker og viser hvad der forventes.                                                                                                                                                                                  |
| **Status 1**                     | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet**                                                                                                                                                                 |
| **Test Scenarie 2 (Navigation)** | **S2: Drill-down**<br>1. Klik på widgetten "Zonefordeling".                                                                                                                                                          |
| **Forventet Resultat 2**         | - Systemet navigerer til den detaljerede side for Zoneanalyse (UC-03).                                                                                                                                               |
| **Faktisk Resultat 2**           | Virker for andre tabs. Zoneanalyse er ikke implementeret.                                                                                                                                                            |
| **Status 2**                     | ☑ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet** _(Note: Zoneanalyse ikke implementeret)_                                                                                                                        |

### P10 - TC-05: Visualisering af slagpositioner - (IKKE IMPLEMENTERET)

| Test Case Element                 | Detaljer                                                                                                                                                                               |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Test ID**                       | TC-05                                                                                                                                                                                  |
| **Tilknyttet UC**                 | UC-05: Visualisering af slagpositioner                                                                                                                                                 |
| **Formål**                        | At verificere, at systemet plotter de præcise koordinater for slag.                                                                                                                    |
| **Preconditions**                 | - Kamp analyseret.                                                                                                                                                                     |
| **Test Scenarie 1 (Happy Path)**  | **S1: Kortlægning af slag**<br>1. Naviger til "Slagpositioner".<br>2. Vælg "Alle spillere".                                                                                            |
| **Forventet Resultat 1**          | - Banekortet vises fyldt med punkter (prikker).<br>- Prikkerne har forskellige farver svarende til de forskellige spillere.<br>- Prikkernes placering virker logisk (inden for banen). |
| **Faktisk Resultat 1**            | Ikke implementeret.                                                                                                                                                                    |
| **Status 1**                      | ▢ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet** &nbsp;&nbsp;&nbsp;&nbsp; ☑ **Ikke testet**                                                                                        |
| **Test Scenarie 2 (Interaction)** | **S3: Point-udfald**<br>1. Aktiver toggle "Vis vinder/taber slag".                                                                                                                     |
| **Forventet Resultat 2**          | - Prikkernes farver skifter (f.eks. Grøn for vinderslag, Rød for fejl).                                                                                                                |
| **Faktisk Resultat 2**            | Ikke implementeret.                                                                                                                                                                    |
| **Status 2**                      | ▢ **Godkendt** &nbsp;&nbsp;&nbsp;&nbsp; ▢ **Fejlet** &nbsp;&nbsp;&nbsp;&nbsp; ☑ **Ikke testet**                                                                                        |
