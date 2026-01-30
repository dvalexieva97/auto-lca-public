# Define variables
VENV_NAME := auto-lca-env
PYTHON_VERSION := 3.13.7

# Detect OS
UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")
ifeq ($(OS),Windows_NT)
	DETECTED_OS := Windows
else ifeq ($(UNAME_S),Linux)
	DETECTED_OS := Linux
else ifeq ($(UNAME_S),Darwin)
	DETECTED_OS := macOS
else
	DETECTED_OS := Unknown
endif

# Set OS-specific paths
ifeq ($(DETECTED_OS),Windows)
	VENV_BIN := Scripts
	VENV_PYTHON := $(VENV_NAME)/$(VENV_BIN)/python.exe
	VENV_PIP := $(VENV_NAME)/$(VENV_BIN)/pip.exe
	PYTHON_CMD := python
	RM_CMD := rmdir /s /q
	MKDIR_CMD := if not exist
	VENV_PY := $(VENV_NAME)/Scripts/python.exe
else
	VENV_BIN := bin
	VENV_PYTHON := $(VENV_NAME)/$(VENV_BIN)/python
	VENV_PIP := $(VENV_NAME)/$(VENV_BIN)/pip
	PYTHON_CMD := python3
	RM_CMD := rm -rf
	MKDIR_CMD := mkdir -p
	VENV_PY := $(VENV_NAME)/bin/python
endif


# Default target
.PHONY: all
all: init

# Initialize the project
.PHONY: init
init: check-os install-pyenv install-python venv install-deps verify-venv
	@echo "Project initialized successfully!"
	@echo "To activate the virtual environment, run:"
ifeq ($(DETECTED_OS),Windows)
	@echo "  $(VENV_NAME)\\$(VENV_BIN)\\activate"
else
	@echo "  source $(VENV_NAME)/$(VENV_BIN)/activate"
endif

# Check OS
.PHONY: check-os
check-os:
	@echo "Detected OS: $(DETECTED_OS)"

# Install pyenv based on OS
.PHONY: install-pyenv
install-pyenv:
	@echo "Checking for pyenv..."
ifeq ($(DETECTED_OS),macOS)
	@if ! command -v pyenv >/dev/null 2>&1; then \
		echo "Installing pyenv via Homebrew..."; \
		brew install pyenv; \
		echo "Adding pyenv to shell configuration..."; \
		if [ -f ~/.zshrc ]; then \
			echo 'export PYENV_ROOT="$$HOME/.pyenv"' >> ~/.zshrc; \
			echo 'command -v pyenv >/dev/null || export PATH="$$PYENV_ROOT/bin:$$PATH"' >> ~/.zshrc; \
			echo 'eval "$$(pyenv init -)"' >> ~/.zshrc; \
		fi; \
		if [ -f ~/.bash_profile ]; then \
			echo 'export PYENV_ROOT="$$HOME/.pyenv"' >> ~/.bash_profile; \
			echo 'command -v pyenv >/dev/null || export PATH="$$PYENV_ROOT/bin:$$PATH"' >> ~/.bash_profile; \
			echo 'eval "$$(pyenv init -)"' >> ~/.bash_profile; \
		fi; \
		export PYENV_ROOT="$$HOME/.pyenv"; \
		export PATH="$$PYENV_ROOT/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
	else \
		echo "pyenv is already installed."; \
	fi
else ifeq ($(DETECTED_OS),Linux)
	@if ! command -v pyenv >/dev/null 2>&1; then \
		echo "Installing pyenv dependencies..."; \
		if command -v apt-get >/dev/null 2>&1; then \
			sudo apt-get update && sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git; \
		elif command -v yum >/dev/null 2>&1; then \
			sudo yum install -y gcc openssl-devel bzip2-devel libffi-devel readline-devel sqlite-devel xz-devel tk-devel git; \
		fi; \
		echo "Installing pyenv..."; \
		curl https://pyenv.run | bash; \
		echo "Adding pyenv to shell configuration..."; \
		if [ -f ~/.bashrc ]; then \
			echo 'export PYENV_ROOT="$$HOME/.pyenv"' >> ~/.bashrc; \
			echo 'command -v pyenv >/dev/null || export PATH="$$PYENV_ROOT/bin:$$PATH"' >> ~/.bashrc; \
			echo 'eval "$$(pyenv init -)"' >> ~/.bashrc; \
		fi; \
		export PYENV_ROOT="$$HOME/.pyenv"; \
		export PATH="$$PYENV_ROOT/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
	else \
		echo "pyenv is already installed."; \
	fi
else ifeq ($(DETECTED_OS),Windows)
	@echo "For Windows, please install pyenv-win manually:"; \
	echo "  git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%\\.pyenv"; \
	echo "  setx PYENV_ROOT \"%USERPROFILE%\\.pyenv\""; \
	echo "  setx PATH \"%PYENV_ROOT%\\bin;%PYENV_ROOT%\\shims;%PATH%\""; \
	echo "Or use Chocolatey: choco install pyenv-win"; \
	if ! command -v pyenv >/dev/null 2>&1; then \
		echo "Warning: pyenv not found. Please install it manually."; \
	fi
else
	@echo "Unknown OS. Please install pyenv manually."
endif

# Helper to get pyenv path
PYENV_ROOT := $(shell echo $$HOME)/.pyenv
PYENV_BIN := $(PYENV_ROOT)/bin/pyenv

# Install Python version using pyenv
.PHONY: install-python
install-python:
	@echo "Installing Python $(PYTHON_VERSION) using pyenv..."
ifeq ($(DETECTED_OS),Windows)
	@if command -v pyenv >/dev/null 2>&1; then \
		if pyenv versions --bare | grep -q "^$(PYTHON_VERSION)$$"; then \
			echo "Python $(PYTHON_VERSION) is already installed."; \
		else \
			echo "Installing Python $(PYTHON_VERSION)..."; \
			pyenv install $(PYTHON_VERSION) || (echo "Failed to install Python $(PYTHON_VERSION)." && exit 1); \
		fi; \
		pyenv local $(PYTHON_VERSION); \
		echo "Python $(PYTHON_VERSION) set as local version."; \
	else \
		echo "Warning: pyenv not found. Using system Python."; \
	fi
else
	@if [ -f "$(PYENV_BIN)" ] || command -v pyenv >/dev/null 2>&1; then \
		export PYENV_ROOT="$(PYENV_ROOT)"; \
		export PATH="$$PYENV_ROOT/bin:$$PATH"; \
		eval "$$($(PYENV_BIN) init - 2>/dev/null || pyenv init -)"; \
		if $(PYENV_BIN) versions --bare 2>/dev/null | grep -q "^$(PYTHON_VERSION)$$" || pyenv versions --bare 2>/dev/null | grep -q "^$(PYTHON_VERSION)$$"; then \
			echo "Python $(PYTHON_VERSION) is already installed."; \
		else \
			echo "Installing Python $(PYTHON_VERSION)..."; \
			$(PYENV_BIN) install $(PYTHON_VERSION) 2>/dev/null || pyenv install $(PYTHON_VERSION) || (echo "Failed to install Python $(PYTHON_VERSION)." && exit 1); \
		fi; \
		$(PYENV_BIN) local $(PYTHON_VERSION) 2>/dev/null || pyenv local $(PYTHON_VERSION); \
		echo "Python $(PYTHON_VERSION) set as local version."; \
	else \
		echo "Warning: pyenv not found. Using system Python."; \
	fi
endif

# Create virtual environment
.PHONY: venv
venv:
	@echo "Creating virtual environment..."
ifeq ($(DETECTED_OS),Windows)
	@if command -v pyenv >/dev/null 2>&1 && pyenv versions --bare | grep -q "^$(PYTHON_VERSION)$$"; then \
		echo "Using pyenv Python $(PYTHON_VERSION)..."; \
		pyenv local $(PYTHON_VERSION); \
		PYENV_PYTHON=$$(pyenv which python); \
		$$PYENV_PYTHON -m venv $(VENV_NAME); \
	else \
		echo "Using system Python..."; \
		$(PYTHON_CMD) -m venv $(VENV_NAME); \
	fi
else
	@if [ -f "$(PYENV_BIN)" ] || command -v pyenv >/dev/null 2>&1; then \
		export PYENV_ROOT="$(PYENV_ROOT)"; \
		export PATH="$$PYENV_ROOT/bin:$$PATH"; \
		eval "$$($(PYENV_BIN) init - 2>/dev/null || pyenv init -)"; \
		if $(PYENV_BIN) versions --bare 2>/dev/null | grep -q "^$(PYTHON_VERSION)$$" || pyenv versions --bare 2>/dev/null | grep -q "^$(PYTHON_VERSION)$$"; then \
			echo "Using pyenv Python $(PYTHON_VERSION)..."; \
			$(PYENV_BIN) local $(PYTHON_VERSION) 2>/dev/null || pyenv local $(PYTHON_VERSION); \
			PYENV_PYTHON=$$($(PYENV_BIN) which python 2>/dev/null || pyenv which python); \
			echo "Creating venv with: $$PYENV_PYTHON"; \
			$$PYENV_PYTHON -m venv $(VENV_NAME); \
		else \
			echo "Python $(PYTHON_VERSION) not found in pyenv. Using system Python..."; \
			$(PYTHON_CMD) -m venv $(VENV_NAME); \
		fi; \
	else \
		echo "Using system Python..."; \
		$(PYTHON_CMD) -m venv $(VENV_NAME); \
	fi
endif
	@if [ -f "$(VENV_PYTHON)" ] || [ -f "$(VENV_NAME)/$(VENV_BIN)/python.exe" ]; then \
		echo "Virtual environment $(VENV_NAME) created successfully."; \
		$(VENV_PYTHON) --version 2>/dev/null || $(VENV_NAME)/$(VENV_BIN)/python.exe --version 2>/dev/null || echo "Warning: Could not verify Python in venv."; \
	else \
		echo "Error: Virtual environment $(VENV_NAME) was not created properly."; \
		exit 1; \
	fi

# Install dependencies from requirements.txt
.PHONY: install-deps
install-deps:
	@echo "Installing dependencies from requirements.txt..."
	@set -x
	$(VENV_PY) -m pip install -r requirements.txt
	@echo "Dependencies installed."

# Add or update VSCode settings to use the venv python
.PHONY: set-vscode-python
set-vscode-python:
	@echo "Setting VSCode Python interpreter to venv..."
ifeq ($(DETECTED_OS),Windows)
	@python -c "import os, json; path = os.path.abspath('$(VENV_PYTHON)').replace('\\', '\\\\'); settings_file = '.vscode/settings.json'; os.makedirs('.vscode', exist_ok=True); settings = json.load(open(settings_file)) if os.path.exists(settings_file) else {}; settings['python.pythonPath'] = path; json.dump(settings, open(settings_file, 'w'), indent=2)"
else
	@python_path="\"$$(pwd)/$(VENV_PYTHON)\""; \
	settings_file=".vscode/settings.json"; \
	if [ -f $$settings_file ]; then \
		if command -v jq >/dev/null 2>&1; then \
			jq '. + {"python.pythonPath": '$$python_path'}' $$settings_file > $$settings_file.tmp && mv $$settings_file.tmp $$settings_file; \
		else \
			python -c "import json, os; f='$$settings_file'; d=json.load(open(f)) if os.path.exists(f) else {}; d['python.pythonPath']=$$(pwd)/$(VENV_PYTHON); json.dump(d, open(f, 'w'), indent=2)"; \
		fi; \
	else \
		mkdir -p .vscode; \
		if command -v jq >/dev/null 2>&1; then \
			echo '{"python.pythonPath": ' $$python_path '}' > $$settings_file; \
		else \
			python -c "import json, os; os.makedirs('.vscode', exist_ok=True); json.dump({'python.pythonPath': '$$(pwd)/$(VENV_PYTHON)'}, open('$$settings_file', 'w'), indent=2)"; \
		fi; \
	fi
endif
	@echo "VSCode Python interpreter set."

# Verify venv is working
.PHONY: verify-venv
verify-venv:
	@echo "Verifying virtual environment..."
	@if [ -f "$(VENV_PYTHON)" ] || [ -f "$(VENV_NAME)/$(VENV_BIN)/python.exe" ]; then \
		echo "✓ Virtual environment exists"; \
		VENV_PYTHON_VERSION=$$($(VENV_PYTHON) --version 2>/dev/null || $(VENV_NAME)/$(VENV_BIN)/python.exe --version 2>/dev/null | cut -d' ' -f2); \
		echo "✓ Python version in venv: $$VENV_PYTHON_VERSION"; \
		if echo "$$VENV_PYTHON_VERSION" | grep -q "^$(PYTHON_VERSION)"; then \
			echo "✓ Python version matches required version $(PYTHON_VERSION)"; \
		else \
			echo "✗ WARNING: Python version mismatch! Expected $(PYTHON_VERSION), got $$VENV_PYTHON_VERSION"; \
			echo "  Run 'make clean && make init' to recreate the venv with the correct Python version."; \
		fi; \
		$(VENV_PIP) --version 2>/dev/null || $(VENV_NAME)/$(VENV_BIN)/pip.exe --version 2>/dev/null; \
		echo "✓ pip is accessible"; \
		echo "Virtual environment is properly initialized."; \
	else \
		echo "✗ Virtual environment not found. Run 'make init' first."; \
		exit 1; \
	fi

# Clean up
.PHONY: clean
clean:
	@echo "Cleaning up..."
ifeq ($(DETECTED_OS),Windows)
	@if exist $(VENV_NAME) rmdir /s /q $(VENV_NAME)
else
	@$(RM_CMD) $(VENV_NAME)
endif
	@rm -f .python-version
	@echo "Cleaned."