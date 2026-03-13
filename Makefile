.PHONY: install test run-api docker-build

install:
	python -m pip install -U pip
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t fraud_detection_api:latest .

docker-run:
	docker run -p 8000:8000 fraud_detection_api:latest
