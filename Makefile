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

VENV_PY := $(shell pyenv which python)
# Set OS-specific paths
ifeq ($(DETECTED_OS),Windows)
	VENV_BIN := Scripts
	VENV_PYTHON := $(VENV_NAME)/$(VENV_BIN)/python.exe
	VENV_PIP := $(VENV_NAME)/$(VENV_BIN)/pip.exe
	PYTHON_CMD := python
	RM_CMD := rmdir /s /q
	MKDIR_CMD := if not exist
	VENV_PY := $(VENV_NAME)/Scripts/python.exe
	PATH := $(USERPROFILE)\.pyenv\pyenv-win\shims;$(PATH)
	PYENV_PY := $(shell pyenv which python)
else
	VENV_BIN := bin
	VENV_PYTHON := $(VENV_NAME)/$(VENV_BIN)/python
	VENV_PIP := $(VENV_NAME)/$(VENV_BIN)/pip
	PYTHON_CMD := python3
	RM_CMD := rm -rf
	MKDIR_CMD := mkdir -p
	VENV_PY := $(VENV_NAME)/bin/python
	PYENV_PY := $(shell pyenv which python)
endif


# Default target
.PHONY: all
all: init

# Initialize the project
.PHONY: init
init: check-os install-python venv install-deps
	@echo "Project initialized successfully!"
ifeq ($(DETECTED_OS),Windows)
	@echo "To activate the virtual environment, run:"
	@echo "  PowerShell: .\\$(VENV_NAME)\\$(VENV_BIN)\\Activate.ps1"
	@echo "  CMD:     $(VENV_NAME)\\$(VENV_BIN)\\activate.bat"
else
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV_NAME)/$(VENV_BIN)/activate"
endif
# .PHONY: init
# init: check-os install-python venv install-deps # verify-venv # install-pyenv
# 	@echo "Project initialized successfully!"
# 	@echo "To activate the virtual environment, run:"
# 	@if [ "$(DETECTED_OS)" = "Windows" ]; then \
# 		echo "  PowerShell: .\\$(VENV_NAME)\\$(VENV_BIN)\\Activate.ps1"; \
# 		echo "  CMD:     $(VENV_NAME)\\$(VENV_BIN)\\activate.bat"; \
# 	else \
# 		echo "  source $(VENV_NAME)/$(VENV_BIN)/activate"; \
# 	fi
# Check OS
.PHONY: check-os
check-os:
	@echo "Detected OS: $(DETECTED_OS)"

# Helper to get pyenv path
PYENV_ROOT := $(shell echo $$HOME)/.pyenv
PYENV_BIN := $(PYENV_ROOT)/bin/pyenv

# Install Python version using pyenv
.PHONY: install-python
install-python:
	@echo "Ensuring Python $(PYTHON_VERSION)..."
ifeq ($(DETECTED_OS),Windows)
	@pyenv install -s $(PYTHON_VERSION)
	@pyenv local $(PYTHON_VERSION)
else
	@pyenv install -s $(PYTHON_VERSION)
	@pyenv local $(PYTHON_VERSION)
endif
# Create virtual environment
.PHONY: venv
venv:
	@echo "Creating virtual environment..."
	@"$(PYENV_PY)" -m venv $(VENV_NAME)



# .PHONY: venv
# venv:
# 	@echo "Creating virtual environment..."
# 	@python -m venv $(VENV_NAME)


# Install dependencies from requirements.txt
.PHONY: install-deps
install-deps:
	@echo "Installing dependencies from requirements.txt..."
	$(VENV_PY) -m pip install -r requirements.txt
	@echo "Dependencies installed."

# Add or update VSCode/Cursor settings to use the venv python
# This is optional and only needed if using VSCode or Cursor IDE
.PHONY: set-vscode-python
set-vscode-python:
	@echo "Setting VSCode/Cursor Python interpreter to venv..."
	@if [ ! -f "$(VENV_PYTHON)" ] && [ ! -f "$(VENV_NAME)/$(VENV_BIN)/python.exe" ]; then \
		echo "Error: Virtual environment not found. Run 'make init' first."; \
		exit 1; \
	fi
	@mkdir -p .vscode
ifeq ($(DETECTED_OS),Windows)
	@python -c "import os, json; path = os.path.abspath('$(VENV_PYTHON)').replace(chr(92), '/'); settings_file = '.vscode/settings.json'; settings = json.load(open(settings_file)) if os.path.exists(settings_file) else {}; settings['python.defaultInterpreterPath'] = path; settings['python.terminal.activateEnvironment'] = True; json.dump(settings, open(settings_file, 'w'), indent=2)"
	@python -c "import os, json; launch_file = '.vscode/launch.json'; venv_python = os.path.abspath('$(VENV_PYTHON)').replace(chr(92), '/'); launch = {'version': '0.2.0', 'configurations': [{'name': 'Python: Current File', 'type': 'debugpy', 'request': 'launch', 'program': '$${file}', 'console': 'integratedTerminal', 'justMyCode': True, 'python': venv_python}]}; os.makedirs('.vscode', exist_ok=True); json.dump(launch, open(launch_file, 'w'), indent=2)"
else
	@python_path="$$(pwd)/$(VENV_PYTHON)"; \
	settings_file=".vscode/settings.json"; \
	python -c "import json, os; f='$$settings_file'; d=json.load(open(f)) if os.path.exists(f) else {}; d['python.defaultInterpreterPath']='$$python_path'; d['python.terminal.activateEnvironment']=True; json.dump(d, open(f, 'w'), indent=2)"
	@python_path="$$(pwd)/$(VENV_PYTHON)"; \
	launch_file=".vscode/launch.json"; \
	python -c "import json, os; launch={'version': '0.2.0', 'configurations': [{'name': 'Python: Current File', 'type': 'debugpy', 'request': 'launch', 'program': '\$${file}', 'console': 'integratedTerminal', 'justMyCode': True, 'python': '$$python_path'}]}; json.dump(launch, open('$$launch_file', 'w'), indent=2)"
endif
	@echo "✓ VSCode/Cursor Python interpreter configured."
	@echo "  - Settings: .vscode/settings.json"
	@echo "  - Debug config: .vscode/launch.json"
	@echo ""
	@echo "Note: This is optional. If not using VSCode/Cursor, you can ignore this."

# Clean up
.PHONY: clean
clean:
	@echo "Cleaning up..."
ifeq ($(DETECTED_OS),Windows)
	@if exist "$(VENV_NAME)" rmdir /s /q "$(VENV_NAME)"
	@if exist ".python-version" del ".python-version"
else
	@$(RM_CMD) $(VENV_NAME)
	@rm -f .python-version
endif
	@echo "Cleaned."
