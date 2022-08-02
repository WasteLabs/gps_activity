# Project utilities
env_create:
	conda create -n gps_activity_dev python=3.10 -y

env_configure: env_install_dependencies env_install_precommit_hooks
	echo "Environment is configured"

env_install_precommit_hooks:
	pre-commit install && pre-commit install --hook-type commit-msg

env_install_dependencies:
	pip3 install --upgrade pip \
	&& pip3 install wheel poetry \
	&& poetry install

env_install_jupyter_extensions:
	jupyter contrib nbextension install --sys-prefix \
	&& jupyter nbextension install --user https://rawgithub.com/minrk/ipython_extensions/master/nbextensions/toc.js \
	&& jupyter nbextension enable --py widgetsnbextension \
	&& jupyter nbextension enable codefolding/main \
	&& jupyter nbextension enable --py keplergl \
	&& jupyter nbextension enable spellchecker/main \
	&& jupyter nbextension enable toggle_all_line_numbers/main \
	&& jupyter nbextension enable hinterland/hinterland \
	&& pip install jupyterthemes && jt -t oceans16

env_delete:
	conda remove --name gps_activity_dev --all -y

run_test:
	pytest

run_precommit:
	pre-commit run --all-files

build:
	python3 -m build

publish:
	python3 -m twine upload --repository testpypi dist/*
