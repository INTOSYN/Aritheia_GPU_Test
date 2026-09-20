# Install the open routine client without Cython or a private core package.
# Reuses an existing CUDA PyTorch installation; never installs a driver or Torch.
# Local source is the default. A future release wheel requires both URL and hash.
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$SourcePath = $PSScriptRoot,
    [string]$WheelUrl = $env:COMPUTEPROOF_WHEEL_URL,
    [string]$WheelSha256 = $env:COMPUTEPROOF_WHEEL_SHA256,
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA "ComputeProof"),
    [string]$RunRoot = (Join-Path $HOME "computeproof-runs"),
    [switch]$Run
)
$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") { throw "This installer is for native Windows. Use install-linux.sh in WSL2." }

function Invoke-CheckedPython {
    param([string]$Interpreter, [string[]]$Arguments)
    & $Interpreter @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }
}

$Python = (Get-Command $Python -ErrorAction Stop).Source
Invoke-CheckedPython $Python @("-c", "import sys,struct; assert sys.version_info >= (3,10) and struct.calcsize('P') == 8, '64-bit Python 3.10+ required'")
Invoke-CheckedPython $Python @("-c", "import torch; assert torch.cuda.is_available(), 'Activate an existing CUDA PyTorch environment first; this script will not replace Torch'")

if ($WheelUrl) {
    if ($WheelUrl -notmatch '^https://') { throw "WheelUrl must use HTTPS." }
    if ($WheelSha256 -notmatch '^[0-9a-fA-F]{64}$') { throw "WheelSha256 must contain the published SHA-256." }
} else {
    $SourcePath = (Resolve-Path $SourcePath).Path
    if (!(Test-Path (Join-Path $SourcePath "pyproject.toml")) -or
        !(Test-Path (Join-Path $SourcePath "src/agrel_public/exact.py"))) {
        throw "Select the complete rc5 source directory or provide a release wheel URL and SHA-256."
    }
}

New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$Venv = Join-Path $InstallRoot "venv-rc5"
Invoke-CheckedPython $Python @("-m", "venv", "--system-site-packages", $Venv)
$ClientPython = Join-Path $Venv "Scripts/python.exe"
$TempDirectory = $null
try {
    if ($WheelUrl) {
        $TempDirectory = Join-Path $InstallRoot ("download-" + [Guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Path $TempDirectory | Out-Null
        $Wheel = Join-Path $TempDirectory "computeproof-0.5.0rc5-py3-none-any.whl"
        Invoke-WebRequest -Uri $WheelUrl -OutFile $Wheel -UseBasicParsing
        $ActualHash = (Get-FileHash -LiteralPath $Wheel -Algorithm SHA256).Hash
        if ($ActualHash -ne $WheelSha256) { throw "Wheel SHA-256 mismatch; refusing installation." }
        $InstallTarget = "${Wheel}[llm]"
    } else {
        $InstallTarget = "${SourcePath}[llm]"
    }
    Invoke-CheckedPython $ClientPython @("-m", "pip", "install", "--disable-pip-version-check", $InstallTarget)
    Invoke-CheckedPython $ClientPython @("-c", "import agrel_public,torch; assert torch.cuda.is_available(); print('ComputeProof', agrel_public.__version__)")
    & $ClientPython -m computeproof doctor
    if ($LASTEXITCODE -ne 0) { throw "Preflight is incomplete; inspect the doctor output before testing." }
    if ($Run) {
        New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
        $Output = Join-Path $RunRoot ("run-" + [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ"))
        # No automatic download/execution/upload consent for optional deep cases.
        Invoke-CheckedPython $ClientPython @("-m", "computeproof", "guided", "--all-devices", "--out", $Output)
    } else {
        Write-Host "Installed. Start testing with: & '$ClientPython' -m computeproof guided"
    }
} finally {
    if ($TempDirectory -and (Test-Path -LiteralPath $TempDirectory)) {
        Remove-Item -LiteralPath $TempDirectory -Recurse -Force
    }
}
