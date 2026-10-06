param(
    [ValidateSet('smoke', 'prepare', 'research')]
    [string]$Mode = 'smoke',
    [string]$ModelDir,
    [string]$IntentModel,
    [int]$Limit = 0
)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Project environment missing. Follow RUNNING.md to install it.'
}
$pipelineArgs = @((Join-Path $PSScriptRoot 'run_pipeline.py'), '--mode', $Mode)
if ($ModelDir) { $pipelineArgs += @('--model-dir', $ModelDir) }
if ($IntentModel) { $pipelineArgs += @('--intent-model', $IntentModel) }
if ($Limit -gt 0) { $pipelineArgs += @('--limit', $Limit) }
& $projectPython @pipelineArgs
exit $LASTEXITCODE
