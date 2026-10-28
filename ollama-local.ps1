[CmdletBinding()]
param(
    [ValidateSet('models', 'infra', 'migrate', 'api', 'worker', 'dispatcher', 'evaluate', 'status', 'stop')]
    [string]$Mode = 'models',
    [string]$Model,
    [string]$ReportDirectory,
    [int]$Seed = 20261026,
    [ValidateSet('true', 'false')]
    [string]$Think = 'false',
    [ValidateRange(4096, 32768)]
    [int]$NumCtx = 16384,
    [ValidateRange(128, 4096)]
    [int]$NumPredict = 2048,
    [ValidateRange(0.0, 2.0)]
    [double]$Temperature = 1.0,
    [switch]$ReuseExistingProject
)

$ErrorActionPreference = 'Stop'
if ($NumPredict -gt [Math]::Min(4096, $NumCtx / 2)) {
    throw 'NumPredict must not exceed half of NumCtx or 4096 tokens.'
}
$root = $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $root 'src') -PathType Container)) {
    $root = Split-Path -Parent $PSScriptRoot
}
$envPath = Join-Path $root '.env.ollama.local'
$statePath = Join-Path $root '.ollama-demo-state.json'
$pythonVenv = Join-Path $root '.venv\Scripts\python.exe'
$python = if (Test-Path -LiteralPath $pythonVenv) {
    $pythonVenv
} else {
    $resolved = Get-Command python -ErrorAction SilentlyContinue
    if (-not $resolved) { throw 'Python 3.11 or 3.12 is required.' }
    $resolved.Source
}

function Get-InstalledOllamaModels {
    try {
        $result = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 5
    } catch {
        throw 'Ollama is not responding at http://127.0.0.1:11434.'
    }
    if ($null -eq $result.models) { throw 'Ollama returned an invalid model list.' }
    return @($result.models)
}

function Get-ModelProfile([string]$Name) {
    if ([string]::IsNullOrWhiteSpace($Name)) { throw 'Pass an installed model name with -Model.' }
    $modelMatches = @(Get-InstalledOllamaModels | Where-Object { $_.name -ceq $Name })
    if ($modelMatches.Count -ne 1 -or $modelMatches[0].digest -notmatch '^[0-9a-f]{64}$') {
        throw 'The exact model name and full local digest could not be confirmed.'
    }
    if ($modelMatches[0].remote_host -or $modelMatches[0].remote_model) {
        throw 'Remote Ollama models are not supported by this local profile.'
    }
    return [pscustomobject]@{ name = $Name; digest = $modelMatches[0].digest }
}

function Read-PrivateConfig {
    if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
        throw 'Private configuration is missing. Run the infra mode first.'
    }
    $values = @{}
    foreach ($line in [System.IO.File]::ReadAllLines($envPath)) {
        if ($line -match '^\s*([^#\s=]+)=(.*)$') {
            $values[$Matches[1]] = $Matches[2].Trim()
        }
    }
    foreach ($name in @('AGENTOPS_POSTGRES_PASSWORD', 'AGENTOPS_API_KEYS', 'AGENTOPS_SESSION_SECRET')) {
        if (-not $values[$name]) { throw 'Private configuration is incomplete; values were not displayed.' }
    }
    return $values
}

function Read-ProjectState {
    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        throw 'No isolated Ollama project state exists. Run infra mode first.'
    }
    try {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    } catch {
        throw 'The local Ollama project state is invalid.'
    }
    if (
        $state.schema -cne 'aiops-ollama-local-v1' -or
        $state.project -notmatch '^agentops-ollama-[a-f0-9]{8}$' -or
        $state.model -notmatch '^[A-Za-z0-9_.:/-]+$' -or
        $state.model_digest -notmatch '^[0-9a-f]{64}$'
    ) {
        throw 'The local Ollama project state failed validation.'
    }
    foreach ($property in @('postgres_port', 'redis_port', 'api_port')) {
        if ([int]$state.$property -lt 1024 -or [int]$state.$property -gt 65535) {
            throw 'A local project port is invalid.'
        }
    }
    return $state
}

function Test-LocalPortFree([int]$Port) {
    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    if (@($listeners | Where-Object { $_.Port -eq $Port }).Count -gt 0) {
        throw "Required localhost port $Port is already in use. No service was started."
    }
}

function Assert-DockerReady {
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $docker) { throw 'Docker CLI is unavailable. Install and start Docker Engine before using infra mode.' }
    & $docker.Source info *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Docker Engine is not running. No local configuration or container was created.' }
    & $docker.Source image inspect postgres:16-alpine redis:7-alpine *> $null
    if ($LASTEXITCODE -ne 0) {
        throw 'Required postgres:16-alpine and redis:7-alpine images must already be present; no image was pulled.'
    }
}

function Invoke-Compose([object]$State, [string[]]$ComposeArguments) {
    $argsForDocker = @(
        'compose', '--env-file', $envPath,
        '--project-name', [string]$State.project,
        '--file', 'compose.yaml',
        '--file', 'compose.ollama.local.yaml'
    ) + $ComposeArguments
    Push-Location $root
    try {
        & docker @argsForDocker
        if ($LASTEXITCODE -ne 0) { throw 'Docker Compose failed. No secret values were displayed by the launcher.' }
    } finally {
        Pop-Location
    }
}

function Set-LocalRuntimeEnvironment([object]$State, [hashtable]$Config) {
    $env:PYTHONPATH = Join-Path $root 'src'
    $env:AGENTOPS_DATABASE_URL = ''
    $env:AGENTOPS_DATABASE_HOST = '127.0.0.1'
    $env:AGENTOPS_DATABASE_PORT = [string]$State.postgres_port
    $env:AGENTOPS_DATABASE_NAME = 'agentops'
    $env:AGENTOPS_DATABASE_USER = 'agentops'
    $env:AGENTOPS_DATABASE_PASSWORD = $Config['AGENTOPS_POSTGRES_PASSWORD']
    $env:AGENTOPS_API_KEYS = $Config['AGENTOPS_API_KEYS']
    $env:AGENTOPS_SESSION_SECRET = $Config['AGENTOPS_SESSION_SECRET']
    $env:AGENTOPS_BROWSER_ORIGINS = "http://127.0.0.1:$($State.api_port),http://localhost:$($State.api_port)"
    $env:AGENTOPS_SESSION_COOKIE_SECURE = 'false'
    $env:AGENTOPS_REDIS_URL = "redis://127.0.0.1:$($State.redis_port)/0"
    $env:AGENTOPS_REDIS_QUEUE = 'agentops.commands'
    $env:AGENTOPS_CELERY_TASK_NAME = 'agentops.execute_command'
    $env:AGENTOPS_LLM_PROVIDER = 'ollama'
    $env:AGENTOPS_LLM_PROTOCOL = 'ollama_chat'
    $env:AGENTOPS_LLM_BASE_URL = 'http://127.0.0.1:11434'
    $env:AGENTOPS_LLM_MODEL = [string]$State.model
    $env:AGENTOPS_LLM_MODEL_DIGEST = [string]$State.model_digest
    $env:AGENTOPS_LLM_TIMEOUT_SECONDS = '90'
    $env:AGENTOPS_LLM_MAX_RETRIES = '0'
    $env:AGENTOPS_LLM_NUM_CTX = [string]$NumCtx
    $env:AGENTOPS_LLM_NUM_PREDICT = [string]$NumPredict
    $env:AGENTOPS_LLM_THINK = $Think
    $env:AGENTOPS_LLM_TEMPERATURE = $Temperature.ToString([Globalization.CultureInfo]::InvariantCulture)
    $env:AGENTOPS_RUN_MAX_ACTIVE_SECONDS = '600'
    $env:AGENTOPS_CHECKPOINT_BACKEND = 'postgres'
    $env:AGENTOPS_MCP_READ_TOOL = 'get_service_status'
}

if ($Mode -eq 'models') {
    Get-InstalledOllamaModels |
        Sort-Object name |
        Select-Object name, digest |
        Format-Table -AutoSize
    exit 0
}

if ($Mode -eq 'infra') {
    $existingState = Test-Path -LiteralPath $statePath -PathType Leaf
    if ($existingState -and -not $ReuseExistingProject) {
        throw 'An isolated project state already exists. Pass -ReuseExistingProject to reuse that exact project.'
    }
    if (-not $existingState -and $ReuseExistingProject) {
        throw 'There is no saved project to reuse.'
    }
    Assert-DockerReady
    if ($existingState) {
        $state = Read-ProjectState
        if ($Model -and $Model -cne $state.model) { throw 'The saved project is pinned to a different model.' }
        $profile = Get-ModelProfile ([string]$state.model)
        if ($profile.digest -cne $state.model_digest) { throw 'The saved project model digest has changed.' }
    } else {
        $profile = Get-ModelProfile $Model
        foreach ($port in @(55432, 56379, 8001)) { Test-LocalPortFree $port }
        $state = [pscustomobject]@{
            schema = 'aiops-ollama-local-v1'
            project = 'agentops-ollama-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
            model = $profile.name
            model_digest = $profile.digest
            postgres_port = 55432
            redis_port = 56379
            api_port = 8001
            created_utc = [DateTime]::UtcNow.ToString('o')
        }
        if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
            $env:PYTHONPATH = Join-Path $root 'src'
            Push-Location $root
            try {
                & $python -B -m agentops.local_config --path $envPath *> $null
                if ($LASTEXITCODE -ne 0) { throw 'Private local configuration could not be generated.' }
            } finally {
                Pop-Location
            }
        }
        $state | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $statePath -Encoding utf8
    }
    $null = Read-PrivateConfig
    Invoke-Compose $state @('up', '--detach', '--wait', '--pull', 'never', 'postgres', 'redis')
    Write-Output "Isolated PostgreSQL and Redis are ready for project $($state.project). Model: $($state.model)."
    exit 0
}

$state = Read-ProjectState
if ($Mode -eq 'status') {
    Invoke-Compose $state @('ps')
    exit 0
}
if ($Mode -eq 'stop') {
    Invoke-Compose $state @('down')
    Write-Output 'Isolated services stopped; their project volume was preserved.'
    exit 0
}

$config = Read-PrivateConfig
$profile = Get-ModelProfile ([string]$state.model)
if ($profile.digest -cne $state.model_digest) { throw 'The installed Ollama model digest does not match this project.' }
Set-LocalRuntimeEnvironment $state $config
Push-Location $root
try {
    switch ($Mode) {
        'migrate' {
            & $python -B -m alembic upgrade head
            if ($LASTEXITCODE -ne 0) { throw 'Application database migrations failed.' }
            & $python -B -m agentops.checkpoint_postgres migrate
            if ($LASTEXITCODE -ne 0) { throw 'PostgreSQL checkpoint migrations failed.' }
        }
        'api' {
            & $python -B -m uvicorn agentops.api.main:app --host 127.0.0.1 --port ([int]$state.api_port)
        }
        'worker' {
            & $python -B -m celery -A agentops.worker.service:celery_app worker --loglevel=INFO --pool=solo --concurrency=1
        }
        'dispatcher' {
            & $python -B -m agentops.worker.dispatcher_service
        }
        'evaluate' {
            $reportPath = if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
                Join-Path $root ('output\ollama-' + [guid]::NewGuid().ToString('N'))
            } elseif ([IO.Path]::IsPathRooted($ReportDirectory)) {
                $ReportDirectory
            } else {
                Join-Path $root $ReportDirectory
            }
            $rootFull = [IO.Path]::GetFullPath($root).TrimEnd([IO.Path]::DirectorySeparatorChar)
            $reportFull = [IO.Path]::GetFullPath($reportPath)
            if (-not $reportFull.StartsWith(
                $rootFull + [IO.Path]::DirectorySeparatorChar,
                [StringComparison]::OrdinalIgnoreCase
            )) { throw 'Evaluation reports must stay inside this local project directory.' }
            if (Test-Path -LiteralPath $reportFull) {
                throw 'Evaluation report directory already exists; choose a new path.'
            }
            & $python -B -m agentops.cli evaluate --mode ollama --seed $Seed --output $reportFull
            if ($LASTEXITCODE -ne 0) { throw 'Local Ollama evaluation failed.' }
            Write-Output "Local Ollama evaluation reports were written to $reportFull"
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "The $Mode process exited with an error." }
} finally {
    Pop-Location
}
