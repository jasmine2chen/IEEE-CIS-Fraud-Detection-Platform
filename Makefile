.PHONY: install install-api install-training test lint tune train tune-then-train register promote \
        batch-score monitor run-api docker-build docker-build-api docker-build-training docker-run \
        pipeline-run prefect-deploy stack-up stack-down \
        benchmark ablation

MODEL   ?= xgboost
TRIALS  ?= 50
OUTPUT  ?= data/predictions/batch_$(shell date +%Y%m%d).parquet
MODELS  ?= xgboost mlp_xgboost transformer_xgboost gnn_xgboost
BENCH_OUT ?= reports/benchmark
ABL_OUT   ?= reports/ablation

install:
	python -m pip install -U pip
	pip install -r requirements/dev.txt
	pre-commit install

install-api:
	python -m pip install -U pip
	pip install -r requirements/api.txt

install-training:
	python -m pip install -U pip
	pip install -r requirements/training.txt

test:
	python -m pytest tests/ -v --tb=short

lint:
	pre-commit run --all-files

tune:
	python -m src.training.tune \
		--trans  data/raw/train_transaction.csv \
		--id     data/raw/train_identity.csv \
		--model  $(MODEL) \
		--trials $(TRIALS)

train:
	python -m src.training.train \
		--trans data/raw/train_transaction.csv \
		--id    data/raw/train_identity.csv \
		--model $(MODEL)

tune-then-train: tune train

register:
	@echo "Registration happens automatically after training. Use 'make promote' to set @champion."

promote:
	python -c "\
from src.config import load_config; \
from src.deployment import registry; \
cfg = load_config(); \
uri = cfg['training']['mlflow_tracking_uri']; \
client = __import__('mlflow').MlflowClient(tracking_uri=uri); \
name = registry.get_model_name('$(MODEL)'); \
versions = client.search_model_versions(f\"name='{name}'\"); \
latest = max(versions, key=lambda v: int(v.version)); \
registry.promote_to_champion('$(MODEL)', latest.version, tracking_uri=uri); \
print(f'Promoted $(MODEL) version {latest.version} to @champion')"

batch-score:
	python -m src.deployment.batch_score \
		--trans  data/raw/test_transaction.csv \
		--id     data/raw/test_identity.csv \
		--output $(OUTPUT) \
		--model  $(MODEL)

monitor:
	python -m src.monitoring.drift \
		--current $(OUTPUT) \
		--model   $(MODEL)

benchmark:
	python -m src.evaluation.benchmark \
		--trans  data/raw/train_transaction.csv \
		--id     data/raw/train_identity.csv \
		--models $(MODELS) \
		--output $(BENCH_OUT)

ablation:
	python -m src.evaluation.ablation \
		--trans  data/raw/train_transaction.csv \
		--id     data/raw/train_identity.csv \
		--output $(ABL_OUT)

pipeline-run:
	python pipelines/training_pipeline.py \
		--model  $(MODEL) \
		--trials $(TRIALS)

run-api:
	uvicorn src.serving.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build: docker-build-api

docker-build-api:
	docker build -f docker/Dockerfile.api -t fraud_detection_api:latest .

docker-build-training:
	docker build -f docker/Dockerfile.training -t fraud_detection_training:latest .

docker-run:
	docker run -p 8000:8000 \
		-e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
		fraud_detection_api:latest

prefect-deploy:
	prefect deploy --all --prefect-file pipelines/prefect.yaml

stack-up:
	docker compose up -d
	@echo "Services:"
	@echo "  API:     http://localhost:8000/docs"
	@echo "  MLflow:  http://localhost:5000"
	@echo "  Grafana: http://localhost:3000  (admin/admin)"

stack-down:
	docker compose down
