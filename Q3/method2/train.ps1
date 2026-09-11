param(
    [long]$Steps = 1000000,
    [string]$Config = 'configs/full.json',
    [string]$Device = 'cuda',
    [string]$Resume = '',
    [int]$Seed = 1234
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '..\..\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Python environment missing. Follow README installation instructions.'
}
$runName = 'runs/train_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$trainArgs = @('-m', 'q3ppo.train', '--config', $Config, '--device', $Device,
    '--steps', "$Steps", '--out', $runName, '--seed', "$Seed")
if ($Resume) { $trainArgs += @('--resume', $Resume) }
Push-Location $PSScriptRoot
try {
    & $pythonPath @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
} finally { Pop-Location }
