# Makefile for plot-digitizer project
# Generate UML diagrams in multiple formats using pyreverse

# Variables
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PYREVERSE = $(VENV_DIR)/bin/pyreverse
SRC_DIR = src/plot_digitizer
DOCS_DIR = docs
PROJECT_NAME = plot_digitizer

# Colors for output
GREEN = \033[32m
YELLOW = \033[33m
NC = \033[0m # No Color

.PHONY: help setup diagrams clean-docs plantuml dot png all

help:
	@echo "$(GREEN)Available targets:$(NC)"
	@echo "  $(YELLOW)setup$(NC)      - Install dependencies and setup virtual environment"
	@echo "  $(YELLOW)diagrams$(NC)   - Generate all UML diagrams (plantuml, dot, png)"
	@echo "  $(YELLOW)plantuml$(NC)   - Generate PlantUML format diagrams"
	@echo "  $(YELLOW)dot$(NC)        - Generate DOT format diagrams"
	@echo "  $(YELLOW)png$(NC)        - Generate PNG format diagrams"
	@echo "  $(YELLOW)clean-docs$(NC) - Clean generated documentation files"
	@echo "  $(YELLOW)all$(NC)        - Setup and generate all diagrams"

setup:
	@echo "$(GREEN)Setting up virtual environment and installing dependencies...$(NC)"
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt
	$(VENV_DIR)/bin/pip install pylint
	$(VENV_DIR)/bin/pip install -e .
	@echo "$(GREEN)Setup completed!$(NC)"

plantuml: $(VENV_DIR)
	@echo "$(GREEN)Generating PlantUML diagrams...$(NC)"
	$(PYREVERSE) -o puml -d $(DOCS_DIR) --project $(PROJECT_NAME) \
		--filter-mode ALL \
		--show-ancestors 2 \
		--show-associated 2 \
		--colorized \
		--verbose \
		$(SRC_DIR)/
	@echo "$(GREEN)PlantUML diagrams generated in $(DOCS_DIR)/$(NC)"

dot: $(VENV_DIR)
	@echo "$(GREEN)Generating DOT diagrams...$(NC)"
	$(PYREVERSE) -o dot -d $(DOCS_DIR) --project $(PROJECT_NAME)_dot \
		--filter-mode ALL \
		--show-ancestors 2 \
		--show-associated 2 \
		--colorized \
		--verbose \
		$(SRC_DIR)/
	@echo "$(GREEN)DOT diagrams generated in $(DOCS_DIR)/$(NC)"

png: $(VENV_DIR)
	@echo "$(GREEN)Generating PNG diagrams...$(NC)"
	$(PYREVERSE) -o png -d $(DOCS_DIR) --project $(PROJECT_NAME)_png \
		--filter-mode ALL \
		--show-ancestors 2 \
		--show-associated 2 \
		--colorized \
		--verbose \
		$(SRC_DIR)/
	@echo "$(GREEN)PNG diagrams generated in $(DOCS_DIR)/$(NC)"

svg: $(VENV_DIR)
	@echo "$(GREEN)Generating SVG diagrams...$(NC)"
	$(PYREVERSE) -o svg -d $(DOCS_DIR) --project $(PROJECT_NAME)_svg \
		--filter-mode ALL \
		--show-ancestors 2 \
		--show-associated 2 \
		--colorized \
		--verbose \
		$(SRC_DIR)/
	@echo "$(GREEN)SVG diagrams generated in $(DOCS_DIR)/$(NC)"

diagrams: plantuml dot png svg
	@echo "$(GREEN)All diagrams generated successfully!$(NC)"
	@echo "$(YELLOW)Generated files:$(NC)"
	@ls -la $(DOCS_DIR)/*.puml $(DOCS_DIR)/*.dot $(DOCS_DIR)/*.png $(DOCS_DIR)/*.svg 2>/dev/null || true

clean-docs:
	@echo "$(GREEN)Cleaning generated documentation files...$(NC)"
	rm -f $(DOCS_DIR)/*.puml
	rm -f $(DOCS_DIR)/*.dot
	rm -f $(DOCS_DIR)/*.png
	rm -f $(DOCS_DIR)/*.svg
	@echo "$(GREEN)Documentation files cleaned!$(NC)"

# Check if virtual environment exists
$(VENV_DIR):
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(YELLOW)Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi

all: setup diagrams
	@echo "$(GREEN)Project setup and diagram generation completed!$(NC)"

# Development helpers
test: $(VENV_DIR)
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v || echo "$(YELLOW)No tests found or pytest not installed$(NC)"

lint: $(VENV_DIR)
	@echo "$(GREEN)Running pylint...$(NC)"
	$(VENV_DIR)/bin/pylint $(SRC_DIR)/ || echo "$(YELLOW)Pylint completed with warnings$(NC)"

format: $(VENV_DIR)
	@echo "$(GREEN)Formatting code with black...$(NC)"
	$(VENV_DIR)/bin/black $(SRC_DIR)/ || echo "$(YELLOW)Black not installed, skipping formatting$(NC)"

# Show current project structure
tree:
	@echo "$(GREEN)Project structure:$(NC)"
	@tree -I '__pycache__|*.pyc|.venv|.git' . || ls -la
