param(
    [Parameter(Mandatory = $true)]
    [string]$InputFolder,
    [string]$OutputFolder = "",
    [int]$FPS = 1,
    [string]$Quality = "high"
)

if (-not (Test-Path $InputFolder)) {
    Write-Error "Input folder not found: $InputFolder"
    exit 1
}

if ([string]::IsNullOrWhiteSpace($OutputFolder)) {
    $OutputFolder = Join-Path $InputFolder "extracted_frames"
}
New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null

# Check if ffmpeg is installed
$ffmpegPath = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpegPath) {
    Write-Error "ffmpeg not found. Please install ffmpeg first."
    Write-Host ""
    Write-Host "To install ffmpeg on Windows:" -ForegroundColor Yellow
    Write-Host "1. Using Chocolatey: choco install ffmpeg" -ForegroundColor Cyan
    Write-Host "2. Using Scoop: scoop install ffmpeg" -ForegroundColor Cyan
    Write-Host "3. Manual: https://ffmpeg.org/download.html" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

Write-Host "Using ffmpeg: $($ffmpegPath.Source)" -ForegroundColor Green
Write-Host ""

# Set quality parameter for ffmpeg
$qualityParam = switch ($Quality) {
    "high" { "-q:v 2" }
    "medium" { "-q:v 5" }
    "low" { "-q:v 10" }
    default { "-q:v 2" }
}

$extensions = @("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm", "*.flv", "*.m4v", "*.wmv")
$files = foreach ($ext in $extensions) { Get-ChildItem -Path $InputFolder -Filter $ext -File }

if ($files.Count -eq 0) {
    Write-Warning "No video files found in: $InputFolder"
    exit 0
}

Write-Host "Found $($files.Count) video(s) to process" -ForegroundColor Cyan
Write-Host "Frame rate: $FPS FPS"
Write-Host "Quality: $Quality"
Write-Host ""

$totalFrames = 0
$frameCounter = 1

foreach ($f in $files) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    $framePattern = Join-Path $OutputFolder "${name}_frame_%04d.jpg"

    Write-Host "Processing: $($f.Name)" -ForegroundColor Yellow
    Write-Host "  Output: $OutputFolder"

    # Extract frames using ffmpeg
    $ffmpegArgs = @(
        "-hide_banner", "-loglevel", "warning", "-y",
        "-i", $f.FullName,
        "-vf", "fps=$FPS",
        "-q:v", ($qualityParam -replace "-q:v ", ""),
        $framePattern
    )
  
    & ffmpeg @ffmpegArgs

    if ($LASTEXITCODE -eq 0) {
        $frameCount = (Get-ChildItem -Path $OutputFolder -Filter "${name}_frame_*.jpg").Count
        $totalFrames += $frameCount
        Write-Host "  Extracted: $frameCount frames" -ForegroundColor Green
    }
    else {
        Write-Host "  Error extracting frames" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Extraction Complete!" -ForegroundColor Green
Write-Host "Total frames extracted: $totalFrames"
Write-Host "Output location: $OutputFolder"
Write-Host ""
Write-Host "Next steps for YOLO training:" -ForegroundColor Yellow
Write-Host "1. Upload frames to Roboflow or use LabelImg"
Write-Host "2. Label: player_1, player_2, player_3, player_4, racket, ball"
Write-Host "3. Export in YOLO format"
Write-Host "4. Split dataset: 70% train, 15% val, 15% test (by video)"
