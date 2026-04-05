# System Architecture

This document describes the end-to-end architecture of the fraud detection platform using diagrams that can be rendered in any Mermaid-compatible viewer (GitHub, VS Code, Notion, etc.).

---

## 1. End-to-End Data & Training Pipeline

```mermaid
flowchart TD
    subgraph Ingestion
        A[train_transaction.csv] --> C[data_loader.load_data]
        B[train_identity.csv]    --> C
        C --> D[reduce_mem_usage\nfloat64→float32, int64→int32]
        D --> E[clean_data\ndrop cols >70% missing]
    end

    subgraph Feature Engineering
        E --> F[build_features\nMagic UID = card1 + addr1 + floor day-D1]
        F --> G[normalize_d_columns\nD2..D15 relative to D1]
        G --> H[uid aggregations\nmean/std/min/max of C-cols, M-cols, D-norm-cols]
        H --> I[get_full_pipeline\nFrequencyEncoder + SimpleImputer + StandardScaler]
    end

    subgraph OOT Split
        I --> J{time_consistency_split}
        J -->|months 0-5| K[X_train / y_train]
        J -->|month 6|   L[X_test  / y_test]
    end

    subgraph Model Training
        K --> M1[XGBoost\nFPR early stopping]
        K --> M2[MLP Encoder\nFocalLoss → embeddings]
        M2 --> M2b[XGBoost on\nenriched features]
        K --> M3[GraphSAGE\ndirected 7-day graph]
        M3 --> M3b[XGBoost on\nnode embeddings]
        K --> M4[TabTransformer\nCLS-token attention]
        M4 --> M4b[XGBoost on\nCLS embeddings]
    end

    subgraph Evaluation
        M1  --> E1[FPR sweep\n0.1%→25% FPR]
        M2b --> E1
        M3b --> E1
        M4b --> E1
        L   --> E1
        E1  --> E2[partial AUC @5% FPR\ndollar recall\nMLflow logging]
    end

    subgraph Artefacts
        M1  --> AR[models/xgboost_fraud_model.joblib]
        I   --> AR2[models/feature_pipeline.joblib]
    end
```

---

## 2. Model Architecture Detail

### 2a. Baseline — XGBoost with FPR Early Stopping

```mermaid
flowchart LR
    FP[Feature Pipeline\nsklearn Pipeline] --> XGB[XGBoost Classifier\nn_est=500, depth=9, lr=0.05]
    XGB --> ES{Early Stopping\nmonitor FPR@threshold}
    ES -->|FPR rising| STOP[stop, restore best]
    ES -->|FPR stable| XGB
    STOP --> OUT[fraud_probability ∈ 0,1]
```

### 2b. MLP → XGBoost Hybrid

```mermaid
flowchart LR
    FP[Feature Pipeline] --> MLP
    subgraph MLP Encoder
        MLP[Input Layer] --> H1[Hidden 256 + BN + ReLU]
        H1 --> H2[Hidden 128 + BN + ReLU]
        H2 --> H3[Hidden 64]
    end
    H3 --> EMB[64-dim Embedding]
    FP  --> CAT[Concat: original + embedding]
    EMB --> CAT
    CAT --> XGB2[XGBoost\non enriched features]
    XGB2 --> OUT[fraud_probability]

    note1[FocalLoss α=0.25 γ=2.0\naddresses class imbalance]
```

### 2c. GraphSAGE → XGBoost Hybrid

```mermaid
flowchart TD
    subgraph Graph Construction
        TX[Transactions] --> ED[Directed edges\npast→future within 7-day window]
        ED --> EF[6-D edge features\ntemporal_decay, uid_flag, card1_flag,\nDeviceInfo_flag, email_flag, addr1_flag]
        EF --> EG[EdgeGate: learned\naggregation weights]
    end

    subgraph GNN Encoder
        EG --> GS1[GraphSAGE Layer 1\nhidden=128]
        GS1 --> GS2[GraphSAGE Layer 2\nhidden=128]
    end

    subgraph Hybrid Output
        GS2 --> NODE[Node Embeddings]
        TX  --> CAT2[Concat: tabular + node emb]
        NODE --> CAT2
        CAT2 --> XGB3[XGBoost]
        XGB3 --> OUT2[fraud_probability]
    end

    note2[K-NN cosine-similarity fallback\nfor isolated nodes]
```

### 2d. TabTransformer → XGBoost Hybrid

```mermaid
flowchart LR
    FP2[Feature Pipeline] --> TOK[Token Embeddings\nd_token=64 per feature]
    TOK --> CLS[Prepend CLS token]
    CLS --> ATT1[Multi-Head Attention\nnhead=4, 2 layers]
    ATT1 --> CLS2[Extract CLS token\n64-dim summary]
    FP2  --> CAT3[Concat: original + CLS]
    CLS2 --> CAT3
    CAT3 --> XGB4[XGBoost\non enriched features]
    XGB4 --> OUT3[fraud_probability]
```

---

## 3. API Serving Architecture

```mermaid
flowchart LR
    subgraph Client
        C1[Mobile / Web App]
        C2[Payment Gateway]
        C3[Batch Job]
    end

    subgraph Fraud API  port 8000
        GW[FastAPI\nuvicorn workers=4]
        GW --> R1[POST /predict\nsingle transaction]
        GW --> R2[POST /predict_batch\nlist of transactions]
        GW --> R3[GET /health]
        R1 --> DI[Dependency Injection\nload pipeline + model once at startup]
        R2 --> DI
        DI --> FP3[feature_pipeline.joblib]
        DI --> XGB5[xgboost_fraud_model.joblib]
        XGB5 --> THR{fraud_prob ≥ 0.85?}
        THR -->|yes| FRAUD[is_fraud: true]
        THR -->|no|  OK[is_fraud: false]
    end

    subgraph Observability
        GW --> MLF[MLflow UI\nport 5000]
    end

    C1 --> GW
    C2 --> GW
    C3 --> GW
```

---

## 4. Hyperparameter Tuning Flow

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Opt as Optuna
    participant Train as train.py
    participant YAML as model_config.yaml
    participant MLF as MLflow

    Dev->>Opt: make tune MODEL=gnn TRIALS=100
    loop 100 trials
        Opt->>Train: suggest params (hidden_dim, lr, depth, ...)
        Train->>Train: OOT split + feature engineering
        Train->>Train: train encoder → extract embeddings → train XGBoost
        Train->>MLF: log trial params + FPR metric
        Train-->>Opt: return FPR@threshold (objective to minimise)
    end
    Opt->>YAML: write best params to configs/model_config.yaml
    Dev->>Train: make train MODEL=gnn
    Train->>YAML: read best params
    Train->>MLF: log final run
```

---

## 5. CI/CD Pipeline

```mermaid
flowchart LR
    PR[Pull Request\nor push to main] --> GHA[GitHub Actions]
    GHA --> T1[Test matrix\nPython 3.9 + 3.11\npytest tests/ -v]
    GHA --> T2[Type check\nmypy src/ api/]
    T1 -->|pass| MERGE[Merge allowed]
    T2 -->|pass| MERGE
    MERGE --> DOCK[docker build\nfraud_detection_api:latest]
    DOCK --> DEPLOY[Deploy to staging\ndocker-compose up]
```
