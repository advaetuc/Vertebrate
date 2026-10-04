param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $root 'app.py') -PathType Leaf)) {
    throw 'Run this script from the repository containing app.py.'
}
$prefix = $root.TrimEnd('\') + '\'
$names = @(
    'src', 'config', 'tests', 'tools', 'docs', 'reports', 'data/manifests',
    'pyproject.toml', 'check_times.py', 'print_times.py', 'yolo_fall_detector.py'
)
$targets = @()
foreach ($name in $names) {
    $path = [IO.Path]::GetFullPath((Join-Path $root $name))
    if (-not $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside the repository: $path"
    }
    if (Test-Path -LiteralPath $path) {
        $item = Get-Item -LiteralPath $path -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing linked target: $path"
        }
        if ($item.PSIsContainer -and (Get-ChildItem -LiteralPath $path -Recurse -Force -Attributes ReparsePoint)) {
            throw "Refusing directory containing links: $path"
        }
        $targets += $path
    }
}
if (-not $Apply) {
    $targets
    Write-Output 'Preview only. Use -Apply to back up and delete these targets.'
    exit 0
}
if ($targets.Count -eq 0) {
    Write-Output 'No legacy targets remain.'
    exit 0
}
$archiveDir = Join-Path $root '.archive'
New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
$archive = Join-Path $archiveDir ('legacy-' + [Guid]::NewGuid().ToString('N') + '.zip')
Compress-Archive -LiteralPath $targets -DestinationPath $archive -CompressionLevel Optimal
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    if ($zip.Entries.Count -eq 0) { throw 'Backup archive is empty; deletion aborted.' }
} finally {
    $zip.Dispose()
}
foreach ($path in $targets) {
    Remove-Item -LiteralPath $path -Recurse -Force
}
Write-Output "Legacy files removed. Backup: $archive"
Write-Output 'Preserved: app.py, weights, .venv, requirements files, models/, data/videos/, runtime/.'
