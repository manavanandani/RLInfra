VENV=venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

.PHONY: help venv install train test clean

help:
	@echo "RLInfra Subcommands:"
	@echo "  venv       Create virtual environment"
	@echo "  install    Install dependencies & register local package"
	@echo "  train      Run local distributed simulation training run"
	@echo "  test       Run unit tests using pytest"
	@echo "  clean      Clean up build and profiling artifacts"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

train:
	$(PYTHON) src/main.py run-training --steps 10

test:
	$(PYTHON) -m pytest tests/ --cov=src

clean:
	rm -rf build/ dist/ *.egg-info/ mlruns/ profiler_traces/ $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
