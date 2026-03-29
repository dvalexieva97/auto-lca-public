# AUTO-LCA Windows Setup Script
# This script automates the installation of all prerequisites and sets up the project on Windows

param(
    [switch]$SkipPrerequisites,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "$Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "$Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "$Message" -ForegroundColor Red
}

function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

function Install-Chocolatey {
    Write-Step "Checking for Chocolatey package manager"
    if (Test-Command choco) {
        Write-Success "Chocolatey is already installed"
        return $true
    }
    
    Write-Warning "Chocolatey not found. Installing Chocolatey..."
    Write-Host "This requires Administrator privileges." -ForegroundColor Yellow
    
    try {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        Write-Success "Chocolatey installed successfully"
        return $true
    }
    catch {
        Write-Error "Failed to install Chocolatey: $_"
        Write-Host "Please install Chocolatey manually from https://chocolatey.org/install" -ForegroundColor Yellow
        return $false
    }
}

function Install-Git {
    Write-Step "Checking for Git"
    if (Test-Command git) {
        $version = git --version
        Write-Success "Git is already installed: $version"
        return $true
    }
    
    Write-Warning "Git not found. Installing Git..."
    if (Test-Command choco) {
        try {
            choco install git -y
            Write-Success "Git installed successfully"
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            return $true
        }
        catch {
            Write-Error "Failed to install Git via Chocolatey: $_"
        }
    }
    
    Write-Host "Please install Git manually from https://git-scm.com/download/win" -ForegroundColor Yellow
    return $false
}

function Install-Make {
    Write-Step "Checking for Make"
    if (Test-Command make) {
        Write-Success "Make is already installed"
        return $true
    }
    
    Write-Warning "Make not found. Installing Make..."
    
    # Option 1: Install via Chocolatey (GnuWin32 make)
    if (Test-Command choco) {
        try {
            choco install make -y
            Write-Success "Make installed successfully"
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            return $true
        }
        catch {
            Write-Warning "Failed to install Make via Chocolatey, trying alternative..."
        }
    }
    
    # Option 2: Install via Git for Windows (includes make in MSYS)
    $gitPath = (Get-Command git -ErrorAction SilentlyContinue).Source
    if ($gitPath) {
        $gitDir = Split-Path (Split-Path $gitPath)
        $mingwMake = Join-Path $gitDir "usr\bin\make.exe"
        if (Test-Path $mingwMake) {
            Write-Success "Make found in Git installation: $mingwMake"
            # Add to PATH for current session
            $mingwBin = Join-Path $gitDir "usr\bin"
            if ($env:Path -notlike "*$mingwBin*") {
                $env:Path = "$mingwBin;$env:Path"
                Write-Host "Added Git's MSYS bin to PATH for this session" -ForegroundColor Yellow
            }
            return $true
        }
    }
    
    Write-Host "Please install Make manually:" -ForegroundColor Yellow
    Write-Host "  1. Install Git for Windows (includes MSYS with make)" -ForegroundColor Yellow
    Write-Host "  2. Or install via Chocolatey: choco install make" -ForegroundColor Yellow
    return $false
}

function Install-Pyenv {
    Write-Step "Checking for pyenv-win"
    if (Test-Command pyenv) {
        Write-Success "pyenv is already installed"
        return $true
    }
    
    Write-Warning "pyenv-win not found. Installing pyenv-win..."
    
    $pyenvRoot = "$env:USERPROFILE\.pyenv"
    
    if (Test-Path $pyenvRoot) {
        Write-Warning "pyenv directory exists but pyenv command not found"
        Write-Host "Adding pyenv to PATH..." -ForegroundColor Yellow
    }
    else {
        # Install pyenv-win
        if (Test-Command git) {
            try {
                Write-Host "Cloning pyenv-win repository..." -ForegroundColor Yellow
                git clone https://github.com/pyenv-win/pyenv-win.git $pyenvRoot
                Write-Success "pyenv-win cloned successfully"
            }
            catch {
                Write-Error "Failed to clone pyenv-win: $_"
                return $false
            }
        }
        else {
            Write-Error "Git is required to install pyenv-win"
            return $false
        }
    }
    
    # Add pyenv to PATH
    $pyenvBin = "$pyenvRoot\pyenv-win\bin"
    $pyenvShims = "$pyenvRoot\pyenv-win\shims"
    
    if ($env:Path -notlike "*$pyenvBin*") {
        Write-Host "Adding pyenv to PATH..." -ForegroundColor Yellow
        [Environment]::SetEnvironmentVariable("PYENV_ROOT", $pyenvRoot, "User")
        [Environment]::SetEnvironmentVariable("PYENV", "$pyenvRoot\pyenv-win", "User")
        
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($userPath -notlike "*$pyenvBin*") {
            [Environment]::SetEnvironmentVariable("Path", "$userPath;$pyenvBin;$pyenvShims", "User")
        }
        
        # Add to current session
        $env:PYENV_ROOT = $pyenvRoot
        $env:PYENV = "$pyenvRoot\pyenv-win"
        $env:Path = "$pyenvBin;$pyenvShims;$env:Path"
        
        Write-Success "pyenv added to PATH"
    }
    
    # Verify installation
    if (Test-Command pyenv) {
        Write-Success "pyenv is now available"
        return $true
    }
    else {
        Write-Warning "pyenv installed but not in PATH. Please restart your terminal."
        return $false
    }
}

function Install-Python {
    param([string]$Version = "3.13.7")
    
    Write-Step "Installing Python $Version using pyenv"
    
    if (-not (Test-Command pyenv)) {
        Write-Error "pyenv is not available. Please restart your terminal and run this script again."
        return $false
    }
    
    try {
        # Check if Python version is already installed
        $installedVersions = pyenv versions --bare 2>$null
        if ($installedVersions -contains $Version) {
            Write-Success "Python $Version is already installed"
        }
        else {
            Write-Host "Installing Python $Version (this may take several minutes)..." -ForegroundColor Yellow
            pyenv install $Version
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to install Python $Version"
                return $false
            }
            Write-Success "Python $Version installed successfully"
        }
        
        # Set local version
        pyenv local $Version
        Write-Success "Python $Version set as local version"
        return $true
    }
    catch {
        Write-Error "Error installing Python: $_"
        return $false
    }
}

function Show-Help {
    Write-Host @"
AUTO-LCA Windows Setup Script

This script automates the installation of all prerequisites and sets up the project on Windows.

Usage:
    .\setup-windows.ps1 [options]

Options:
    -SkipPrerequisites    Skip installing prerequisites (Git, Make, pyenv)
    -Help                 Show this help message

What this script does:
    1. Installs Chocolatey (if not present)
    2. Installs Git (if not present)
    3. Installs Make (if not present)
    4. Installs pyenv-win (if not present)
    5. Installs Python 3.13.7 via pyenv
    6. Creates a virtual environment
    7. Installs project dependencies

Note: Some steps may require Administrator privileges.

After running this script:
    1. Restart your terminal/PowerShell
    2. Activate the virtual environment: .\auto-lca-env\Scripts\Activate.ps1
    3. Add your Mistral API key to .env file
"@
}

# Main execution
if ($Help) {
    Show-Help
    exit 0
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  AUTO-LCA Windows Setup Script" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if running as Administrator (optional, but helpful)
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "Not running as Administrator. Some installations may require elevation."
}

$success = $true

if (-not $SkipPrerequisites) {
    # Install prerequisites
    if (-not (Install-Chocolatey)) {
        Write-Warning "Chocolatey installation failed or skipped. Some automated installations may not work."
    }
    
    if (-not (Install-Git)) {
        Write-Error "Git installation failed. Please install Git manually."
        $success = $false
    }
    
    if (-not (Install-Make)) {
        Write-Warning "Make installation had issues. You may need to install it manually or use Git Bash."
    }
    
    if (-not (Install-Pyenv)) {
        Write-Error "pyenv installation failed. Please install pyenv-win manually."
        $success = $false
    }
    
    if (-not $success) {
        Write-Error "`nPrerequisites installation had errors. Please fix the issues above and try again."
        Write-Host "You can also run with -SkipPrerequisites to skip prerequisite installation." -ForegroundColor Yellow
        exit 1
    }
    
    # Install Python
    if (-not (Install-Python -Version "3.13.7")) {
        Write-Error "Python installation failed."
        exit 1
    }
}


# Final instructions
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Restart your terminal/PowerShell to ensure PATH updates take effect" -ForegroundColor Yellow
Write-Host "2. Activate the virtual environment:" -ForegroundColor Yellow
Write-Host "   .\auto-lca-env\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "3. Add your Mistral API key to .env file:" -ForegroundColor Yellow
Write-Host "   echo 'MISTRAL_API_KEY=`"your-api-key-here`"' > .env" -ForegroundColor White
Write-Host "4. You're ready to use AUTO-LCA!" -ForegroundColor Yellow
Write-Host "`nNote: If you see 'pyenv: command not found' after restarting, you may need to manually add pyenv to your PATH." -ForegroundColor Yellow
