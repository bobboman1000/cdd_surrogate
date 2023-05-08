
dataset:
	cd ./scripts/extract_from_cdd/ && \
		python extract_and_collect.py "../../data/raw/$n/" "../../data/processed/"
config:
	python3 ./scripts/generate_config/generate_config_files.py ./scripts/generate_config data/cdd_config/configurations

create_venv:
	python3 -m pip install virtualenv
	python3 -m virtualenv env --python=python3.9

