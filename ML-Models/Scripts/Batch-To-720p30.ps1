param(
  [Parameter(Mandatory=$true)]
  [string]$InputFolder,
  [string]$OutputFolder = ""
)

if (-not (Test-Path $InputFolder)) {
  Write-Error "Input folder not found: $InputFolder"
  exit 1
}

if ([string]::IsNullOrWhiteSpace($OutputFolder)) {
  $OutputFolder = Join-Path $InputFolder "720p30"
}
New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null

$extensions = @("*.mp4","*.mov","*.mkv","*.avi","*.webm","*.flv","*.m4v","*.wmv")
$files = foreach ($ext in $extensions) { Get-ChildItem -Path $InputFolder -Filter $ext -File }

foreach ($f in $files) {
  $name = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
  $out = Join-Path $OutputFolder ($name + "-720p30.mp4")

  if (Test-Path $out) {
    Write-Host "Skipping (exists): $out"
    continue
  }

  Write-Host "Processing: $($f.Name) -> $(Split-Path $out -Leaf)"
  & ffmpeg -hide_banner -loglevel error -y `
    -i $f.FullName `
    -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,fps=30,format=yuv420p" `
    -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p `
    -c:a aac -b:a 192k -ac 2 `
    -movflags +faststart `
    $out

  Write-Host "Done: $out"
}

Write-Host "All set. Output in: $OutputFolder"
