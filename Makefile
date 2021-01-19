# Signifies our desired python version
# Makefile macros (or variables) are defined a little bit differently than traditional bash,
# keep in mind that in the Makefile there's top-level Makefile-only syntax,
# and everything else is bash script syntax.
PYTHON = python3

# Oneshell means I can run multiple lines in a recipe in the same shell, so I don't have to
# chain commands together with semicolon
.ONESHELL:
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash

# .PHONY defines parts of the makefile that are not dependant on any specific file
# This is most often used to store functions
.PHONY = help env setup setup install-dependencies install-package clean

# Defines the default target that `make` will to try to make,
# or in the case of a phony target, execute the specified commands
# This target is executed whenever we just type `make`
.DEFAULT_GOAL = help

# The @ makes sure that the command itself isn't echoed in the terminal
help:
	@echo "---------------HELP-----------------"
	@echo "To setup the project type make setup"
	@echo "To create virtual env ype make env"
	@echo "To run the project type make run"
	@echo "------------------------------------"

setup:
	pip install -e .

env:
	pip install -r requirements.txt

