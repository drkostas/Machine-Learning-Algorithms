# Makefile for the custom_libs module

SHELL=/bin/bash
PYTHON_VERSION=3.7

# You can use either venv (virtualenv) or conda env by specifying the correct argument
ifeq ($(env),conda)
	# Use Conda
	BASE=~/anaconda3/envs/cosc522
	BIN=$(BASE)/bin
	CLEAN_COMMAND="conda env remove -p $(BASE)"
	CREATE_COMMAND="conda create --prefix $(BASE) python=$(PYTHON_VERSION) -y"
	SETUP_FLAG=
	DEBUG=False
else ifeq ($(env),venv)
	# Use Venv
	BASE=venv
	BIN=$(BASE)/bin
	CLEAN_COMMAND="rm -rf $(BASE)"
	CREATE_COMMAND="python$(PYTHON_VERSION) -m venv $(BASE)"
	SETUP_FLAG=
	DEBUG=True
else
	# Use Conda
	BASE=~/anaconda3/envs/cosc522
	BIN=$(BASE)/bin
	CLEAN_COMMAND="conda env remove -p $(BASE)"
	CREATE_COMMAND="conda create --prefix $(BASE) python=$(PYTHON_VERSION) -y"
#	SETUP_FLAG='--local' # If you want to use this, you change it in setup.py too
	DEBUG=True
endif

all:
	$(MAKE) help
help:
	@echo
	@echo "-----------------------------------------------------------------------------------------------------------"
	@echo "                                              DISPLAYING HELP                                              "
	@echo "-----------------------------------------------------------------------------------------------------------"
	@echo "Use make <make recipe> [env=<conda|venv>] to specify the server"
	@echo
	@echo "make help"
	@echo "       Display this message"
	@echo "make install [env=<conda|venv>]"
	@echo "       Call clean delete_conda_env create_conda_env setup"
	@echo "make clean [env=<conda|venv>]"
	@echo "       Delete all './build ./dist ./*.pyc ./*.tgz ./*.egg-info' files"
	@echo "make delete_env [env=<conda|venv>]"
	@echo "       Delete the current conda env or virtualenv"
	@echo "make create_env [env=<conda|venv>]"
	@echo "       Create a new conda env or virtualenv for the specified python version"
	@echo "make setup [env=<conda|venv>]"
	@echo "       Call setup.py install"
	@echo "-----------------------------------------------------------------------------------------------------------"
install:
	$(MAKE) clean
	$(MAKE) delete_env
	$(MAKE) create_env
	$(MAKE) setup
	@echo "Installation Successful!"
clean:
	$(PYTHON_BIN)python setup.py clean
delete_env:
	@echo "Deleting virtual environment.."
	eval $(DELETE_COMMAND)
create_env:
	@echo "Creating virtual environment.."
	eval $(CREATE_COMMAND)
setup:
	$(BIN)/python setup.py install $(SETUP_FLAG)


.PHONY: help install clean delete_env create_env setup