# OpenCrab v2 on Windows:
#   powershell -ExecutionPolicy Bypass -File scripts\windows.ps1 <command> [run_campaign options]
#
#   setup   CPython 3.12 (python on PATH or the py launcher) into .venv, requirements.txt, the pinned GPT-2 snapshot
#   build   the C++ mirror with the clang of the ziglang package (cpp\build.py); executables in cpp\build\bin
#   test    C++ tests (hardware and portable AES), then the Python suite
#   quick   pipeline check of every stage with the tiny model, into runs\quick
#   full    the paper campaign (every stage but the calibration), into runs\full
#   tables  the tables of results\ regenerated into runs\tables_check and compared with results\tables
#
# Windows PowerShell 5.1. The script finds the artifact root from its own path, so it runs from
# any directory. OPENCRAB2_PYTHON=<interpreter> replaces .venv\Scripts\python.exe (it must be
# CPython 3.12 with requirements.txt).

$Root = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $Root '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$CheckRoot = Join-Path $Root 'runs\tables_check'
$script:ExitCode = 0

function Assert-Exit([string]$What) {
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

function Test-Python312([string]$Exe) {
    & $Exe -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Get-Python {
    if ($env:OPENCRAB2_PYTHON) {
        $py = $env:OPENCRAB2_PYTHON
    } elseif (Test-Path -LiteralPath $VenvPython) {
        $py = $VenvPython
    } else {
        throw "no interpreter: run 'scripts\windows.ps1 setup' first or set OPENCRAB2_PYTHON"
    }
    if (-not (Test-Python312 $py)) { throw "$py is not CPython 3.12" }
    return $py
}

function Find-BasePython {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        if (Test-Python312 $python.Source) { return $python.Source }
    }
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $exe = & $launcher.Source -3.12 -c 'import sys; print(sys.executable)'
        if ($LASTEXITCODE -eq 0 -and $exe) { return ($exe | Select-Object -Last 1).Trim() }
    }
    throw 'CPython 3.12 not found: install it from python.org (with the py launcher) or put it first on PATH'
}

function Set-LoopbackNoProxy {
    # loopback HTTP between the local processes must never go through a proxy
    if ($env:NO_PROXY) { $env:NO_PROXY = "127.0.0.1,localhost,$env:NO_PROXY" } else { $env:NO_PROXY = '127.0.0.1,localhost' }
}

function Invoke-Setup {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        $base = Find-BasePython
        Write-Host "[setup] .venv from $base"
        & $base -m venv $VenvDir
        Assert-Exit 'python -m venv'
    }
    if (-not (Test-Python312 $VenvPython)) { throw "$VenvPython is not CPython 3.12; remove .venv and run setup again" }
    & $VenvPython -m pip install -r requirements.txt
    Assert-Exit 'pip install'
    Write-Host '[setup] GPT-2 snapshot (548 MB) into the Hugging Face cache'
    $code = @'
import os
import sys
os.environ['HF_HUB_OFFLINE'] = '0'
sys.path.insert(0, 'python')
from huggingface_hub import snapshot_download
from opencrab2.highlevel import transformer as t
files = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']
snapshot_download(t.GPT2_REPO, revision=t.GPT2_REVISION, allow_patterns=files)
snap, rev = t.locate_snapshot(t.GPT2_REPO)
tok = t.load_gpt2_tokenizer()
print('[setup] GPT-2', t.GPT2_REPO, 'at', rev, 'in', snap, 'vocabulary', len(tok))
'@
    $code | & $VenvPython -
    Assert-Exit 'GPT-2 download'
}

function Invoke-Build {
    $py = Get-Python
    & $py cpp/build.py
    Assert-Exit 'cpp/build.py'
}

function Invoke-Test {
    $py = Get-Python
    Set-LoopbackNoProxy
    & $py cpp/build.py tests
    if ($LASTEXITCODE -ne 0) { $script:ExitCode = 1 }
    $env:OPENCRAB2_AES = 'portable'
    & $py cpp/build.py tests
    if ($LASTEXITCODE -ne 0) { $script:ExitCode = 1 }
    Remove-Item Env:OPENCRAB2_AES
    & $py -m pytest python/tests
    if ($LASTEXITCODE -ne 0) { $script:ExitCode = 1 }
}

function Invoke-Campaign([string[]]$CampaignArgs) {
    $py = Get-Python
    Set-LoopbackNoProxy
    & $py scripts/run_campaign.py @CampaignArgs
    if ($LASTEXITCODE -ne 0) { $script:ExitCode = $LASTEXITCODE }
}

function Invoke-Tables {
    $py = Get-Python
    Set-LoopbackNoProxy
    if (Test-Path -LiteralPath $CheckRoot) { Remove-Item -LiteralPath $CheckRoot -Recurse -Force }
    $rawDir = Join-Path $CheckRoot 'raw'
    $calDir = Join-Path $CheckRoot 'calibration'
    New-Item -ItemType Directory -Force -Path $rawDir | Out-Null
    New-Item -ItemType Directory -Force -Path $calDir | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $Root 'results\calibration') -File |
        Where-Object { $_.Name -like '*.json' } | Copy-Item -Destination $calDir
    Get-ChildItem -LiteralPath (Join-Path $Root 'results\raw') -File |
        Where-Object { $_.Name -like '*.jsonl' -or $_.Name -like '*.jsonl.gz' } | Copy-Item -Destination $rawDir
    & $py scripts/run_campaign.py --stage tables --out-root $CheckRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[tables] make_tables reported errors (see $CheckRoot\tables\manifest.json)"
        $script:ExitCode = 1
    }
    $shippedDir = Join-Path $Root 'results\tables'
    $newDir = Join-Path $CheckRoot 'tables'
    Write-Host "[tables] results\tables against $newDir"
    $shipped = @(Get-ChildItem -LiteralPath $shippedDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like '*.tex' })
    if ($shipped.Count -eq 0) {
        Write-Host '  results\tables holds no tables: nothing to compare'
        $script:ExitCode = 1
        return
    }
    $same = $true
    foreach ($f in $shipped) {
        $other = Join-Path $newDir $f.Name
        if (-not (Test-Path -LiteralPath $other)) {
            Write-Host "  missing  $($f.Name)"
            $same = $false
        } elseif ((Get-FileHash -LiteralPath $f.FullName).Hash -eq (Get-FileHash -LiteralPath $other).Hash) {
            Write-Host "  equal    $($f.Name)"
        } else {
            Write-Host "  differs  $($f.Name)"
            $same = $false
        }
    }
    foreach ($f in @(Get-ChildItem -LiteralPath $newDir -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like '*.tex' })) {
        if (-not (Test-Path -LiteralPath (Join-Path $shippedDir $f.Name))) {
            Write-Host "  new      $($f.Name)"
            $same = $false
        }
    }
    if ($same) {
        Write-Host '[tables] every table equals results\tables'
    } else {
        Write-Host '[tables] the regenerated tables differ from results\tables'
        $script:ExitCode = 1
    }
}

$command = ''
if ($args.Count -ge 1) { $command = [string]$args[0] }
$rest = @()
if ($args.Count -gt 1) { $rest = @($args[1..($args.Count - 1)]) }
$savedNoProxy = $env:NO_PROXY
$savedAes = $env:OPENCRAB2_AES
Push-Location -LiteralPath $Root
try {
    switch ($command) {
        'setup' { Invoke-Setup }
        'build' { Invoke-Build }
        'test' { Invoke-Test }
        'quick' { Invoke-Campaign (@('--quick') + $rest) }
        # The declared final campaign: every stage but the calibration (the shipped fitted model
        # prices it), both session matrices with 3 workers; '--stage all' adds a fresh calibration.
        # Wall clock of the first campaign on the authors' host (2026-10-11, diag4 at 6 workers):
        # primitives 50 s, cpp_bench 128 s, simulation 32 min (16 workers), diag4 2.8 h, full12 1.2 h
        # (3 workers), web 44 min, tables 2 s; calibration 4.6 h.
        'full' { Invoke-Campaign $rest }
        'tables' { Invoke-Tables }
        default {
            Write-Host 'usage: powershell -ExecutionPolicy Bypass -File scripts\windows.ps1 <setup|build|test|quick|full|tables> [run_campaign options]'
            $script:ExitCode = 2
        }
    }
} catch {
    Write-Host "error: $($_.Exception.Message)"
    $script:ExitCode = 1
} finally {
    Pop-Location
    $env:NO_PROXY = $savedNoProxy
    $env:OPENCRAB2_AES = $savedAes
}
exit $script:ExitCode
