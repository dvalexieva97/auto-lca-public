# Windows Quick Start Guide

## One-Command Setup (Recommended)

```powershell
.\setup-windows.ps1
```
After this, restart your terminal. Once you have restarted it, clone the repo if you haven't yet and initialize it with the makefile.
```
git clone https://github.com/dvalexieva97/auto-lca-public.git
cd auto-lca-public
make init
```

That's it!

## What You Need

- Windows 10 or 11
- PowerShell (pre-installed)
- Administrator privileges (for some installations)

## After Setup

1. **Restart your terminal** (to refresh PATH)
2. **Activate virtual environment**:
   ```powershell
   .\auto-lca-env\Scripts\Activate.ps1
   ```
3. **Add API key**:
   ```powershell
   echo 'MISTRAL_API_KEY="your-key-here"' > .env
   ```

## Troubleshooting

**Script won't run?**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```


If you're setting up on Windows, you have several options:

### Quick Setup (Recommended)

Use the automated PowerShell script:

```powershell
# Open PowerShell in the project directory
.\setup-windows.ps1
```
This script handles all prerequisites automatically. If you encounter issues, see the manual setup below.

### Manual Windows Setup

If the automated script doesn't work, follow these steps:

1. **Install Git for Windows**:
   - Download from [https://git-scm.com/download/win](https://git-scm.com/download/win)
   - During installation, select "Git from the command line and also from 3rd-party software"
   - This includes MSYS tools (including `make`) in the installation

2. **Install Chocolatey** (optional, but recommended):
   ```powershell
   # Run PowerShell as Administrator
   Set-ExecutionPolicy Bypass -Scope Process -Force
   [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
   iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```

3. **Install Make** (if not included with Git):
   ```powershell
   choco install make -y
   ```
   Or use the `make` that comes with Git for Windows (located in `C:\Program Files\Git\usr\bin\make.exe`)

4. **Install pyenv-win**:
   ```powershell
   # Clone pyenv-win
   git clone https://github.com/pyenv-win/pyenv-win.git $HOME\.pyenv\pyenv-win
   
   # Add to PATH (run in PowerShell)
   [Environment]::SetEnvironmentVariable("PYENV_ROOT", "$HOME\.pyenv", "User")
   [Environment]::SetEnvironmentVariable("PYENV", "$HOME\.pyenv\pyenv-win", "User")
   $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
   [Environment]::SetEnvironmentVariable("Path", "$userPath;$HOME\.pyenv\pyenv-win\bin;$HOME\.pyenv\pyenv-win\shims", "User")
   ```
   
   **Important**: Restart your terminal after adding pyenv to PATH.

5. **Install Python**:
   ```powershell
   # Restart terminal first, then:
   pyenv install 3.13.7
   pyenv local 3.13.7
   ```

6. **Run setup**:
   ```powershell
   make init
   ```
   
   Or use the Python setup script:
   ```powershell
   python setup.py
   ```

### Windows Terminal Configuration

If you're using Cursor or VS Code on Windows:

1. **Set default terminal to PowerShell**:
   - Open Settings (Ctrl+,)
   - Search for "terminal integrated default profile"
   - Select "PowerShell"

2. **Or use Git Bash**:
   - If you prefer Git Bash, you can use it instead
   - Make sure `make` is available in your PATH

### Common Windows Issues

**"make: command not found"**:
- Make sure Git for Windows is installed with MSYS tools
- Add Git's MSYS bin to PATH: `C:\Program Files\Git\usr\bin`
- Or install make via Chocolatey: `choco install make`

**"pyenv: command not found"**:
- Restart your terminal after installing pyenv-win
- Verify PATH includes: `%USERPROFILE%\.pyenv\pyenv-win\bin`
- Check environment variables in System Properties

**PowerShell execution policy error**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Virtual environment activation issues**:
```powershell
# If Activate.ps1 is blocked, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
