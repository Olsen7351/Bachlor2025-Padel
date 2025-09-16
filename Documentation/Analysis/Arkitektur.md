# Arkitekturdokumentation

## Oversigt

Padel Analyzer-systemet anvender et **3-lags arkitekturmønster** til backend API'et, hvilket giver en afbalanceret tilgang mellem adskillelse af bekymringer og implementeringssimplicitet. Denne arkitekturmæssige beslutning blev truffet efter nøje overvejelse af forskellige mønstre, herunder Clean Architecture, for bedst at passe til projektets omfang og krav. Python som programmeringssprog og de "normer" som medfølger at skrive i Python, blev også taget i betragtning.

## Begrundelse for Arkitekturvalg

### Hvorfor 3-lags Arkitektur?

Vores team valgte 3-lags arkitekturen af flere centrale årsager:

#### **Tilpasning til Projektomfang**
- **Passende Kompleksitet**: 3-lags tilgangen giver tilstrækkelig struktur til vores padel-analyse domæne uden at introducere unødvendig kompleksitet
- **Team Kendskab**: Mønsteret er velforstået og dokumenteret, hvilket reducerer indlæringskurven for teammedlemmer
- **Hurtig Udvikling**: Tillader hurtigere initial udvikling sammenlignet med mere komplekse arkitekturmønstre

#### **Clean Architecture Overvejelser**
Selvom vi oprindeligt overvejede **Clean Architecture** (Hexagonal Architecture), vurderede vi, at den var for omfattende til vores projekt:

- **Over-engineering Risiko**: Clean Architecture's flere abstraktionslag (value objects, entities, aggregates, use cases, interface adapters, frameworks) ville have introduceret betydelig kompleksitet for et system med relativt ligetil forretningslogik
- **Udviklingsomkostninger**: Den omfattende brug af interfaces, dependency inversion og multiple mapping-lag ville have sænket udviklingshastigheden uden proportionale fordele
- **Vedligeholdelseskompleksitet**: For et studenterprojekt med begrænsede langsigtede vedligeholdelseskrav ville de ekstra abstraktioner skabe unødvendig kognitiv overhead

#### **Overholdelse af SOLID Principper**
Vores 3-lags implementering opretholder stadig overholdelse af SOLID principper:

- **Single Responsibility**: Hvert lag har distinkte ansvarsområder
- **Open/Closed**: Let at udvide services og repositories
- **Liskov Substitution**: Repository mønster muliggør udskiftelige implementeringer
- **Interface Segregation**: Fokuserede interfaces per domæne
- **Dependency Inversion**: Services afhænger af repository abstraktioner

## Arkitekturlag

### 1. Præsentationslag (`presentation/`)
**Ansvar**: API kontrakter og HTTP request/response håndtering

```
presentation/
├── controllers/        # API endpoints og routing
└── dtos/              # Data Transfer Objects (API kontrakter)
```

- **Controllers**: Håndterer HTTP requests, input validering og response formatering
- **DTOs**: Definerer API kontrakter ved hjælp af Pydantic modeller til request/response serialisering
- **Fejlhåndtering**: Konverterer forretningsexceptions til passende HTTP statuskoder

### 2. Forretningslogiklag (`business/`)
**Ansvar**: Centrale forretningsregler, validering og orkestrering

```
business/
├── services/          # Forretningslogik implementering
└── exceptions.py      # Forretningsspecifikke exceptions
```

- **Services**: Implementerer forretningsregler, datavalidering og workflow orkestrering
- **Domæne Validering**: Håndhæver forretningsbegrænsninger og dataintegritet regler
- **Exception Håndtering**: Definerer og håndterer forretningsspecifikke fejltilstande

### 3. Dataadgangslag (`data/`)
**Ansvar**: Data persistering og database interaktioner

```
data/
├── models/            # SQLAlchemy ORM modeller
├── repositories/      # Repository mønster implementering
└── connection.py      # Database konfiguration
```

- **Modeller**: SQLAlchemy ORM modeller der mapper til database tabeller
- **Repositories**: Implementerer dataadgangsmønstre med domæne entitet konvertering
- **Forbindelseshåndtering**: Håndterer database sessioner og connection pooling

## Domænemodeller

Systemet bruger rene Python dataklasser som domænemodeller, der repræsenterer forretningsentiteter uafhængigt af persistering bekymringer:

```
domain/
├── player.py          # Player entitet
├── video.py           # Video entitet
├── match.py           # Match og relaterede entiteter
└── analysis.py        # Analysis entitet
```

## System Kommunikationsflow

### Frontend-Backend Kommunikation

```
┌─────────────────────┐
│ Frontend Applikation│
└──────────┬──────────┘
           │ HTTP Requests
           ▼
┌─────────────────────┐
│   FastAPI Router    │
└──────────┬──────────┘
           │ Route til Controller
           ▼
┌─────────────────────┐
│  Controller Layer   │ ◄── DTO Validering
└──────────┬──────────┘
           │ Forretningslogik
           ▼
┌─────────────────────┐
│ Business Service    │ ◄── Domæne Logik
└──────────┬──────────┘
           │ Data Adgang
           ▼
┌─────────────────────┐
│  Repository Layer   │ ◄── ORM Queries
└──────────┬──────────┘
           │ Database Kald
           ▼
┌─────────────────────┐
│ PostgreSQL Database │
└─────────────────────┘

    ┌─────────────┐
    │ Redis Cache │ ◄──── Session/Cache Data
    └─────────────┘
```

### Request Flow Eksempel

1. **Frontend** sender HTTP request til API endpoint
2. **FastAPI Router** router request til passende controller
3. **Controller** validerer request ved hjælp af Pydantic DTOs
4. **Service Layer** anvender forretningslogik og validering
5. **Repository Layer** konverterer domæneobjekter til/fra ORM modeller
6. **Database** persisterer eller henter data
7. **Response** flyder tilbage gennem lagene med passende transformationer

## Infrastruktur Arkitektur

### Teknologi Stack

- **Backend Framework**: FastAPI (Python)
- **Database**: PostgreSQL med SQLAlchemy ORM
- **Caching**: Redis
- **Validering**: Pydantic
- **Containerisering**: Docker & Docker Compose

## Fordele ved Denne Arkitektur

### **Vedligeholdelse**
- Klar adskillelse af bekymringer på tværs af lag
- Forudsigelig kodeorganisation og placering
- Let at lokalisere og modificere specifik funktionalitet

### **Testbarhed**
- Repository mønster muliggør let mocking til unit tests
- Forretningslogik isoleret fra infrastruktur bekymringer
- Klare grænser for integration testing

### **Skalerbarhed**
- Service lag kan let udvides med ny forretningslogik
- Repository mønster understøtter forskellige datakilder
- Stateless design understøtter horisontal skalering

### **Udviklingshastighed**
- Velkendte mønstre reducerer onboarding tid
- Ligetil debugging og problemløsning
- Hurtig feature udvikling uden arkitekturomkostninger

## Kompromiser og Begrænsninger

### **Kobling**
- Lag har noget kobling gennem delte DTOs og domænemodeller
- Mindre fleksibilitet sammenlignet med Clean Architecture's dependency inversion

### **Kompleksitetsvækst**
- Kan kræve refaktorering til Clean Architecture hvis forretningskompleksitet stiger betydeligt
- Nuværende struktur passer til nuværende omfang men kan kræve udvikling

### **Data Mapping**
- Manuel mapping mellem lag (DTOs ↔ Domæne ↔ ORM) kræver vedligeholdelse
- Multiple repræsentationer af lignende datastrukturer

## Konklusion

3-lags arkitekturen giver en optimal balance mellem struktur og simplicitet for projektet. Den muliggør ren adskillelse af bekymringer, opretholder SOLID principper og understøtter udviklingsteamets produktivitetsmål, samtidig med at den undgår kompleksitetsomkostningerne, der ville komme med mere sofistikerede arkitekturmønstre som Clean Architecture.

Denne tilgang tillader teamet at fokusere på at levere kernefunktionalitet effektivt, samtidig med at kodekvalitet og udvidelsesmuligheder for fremtidige forbedringer bibeholdes.