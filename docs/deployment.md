# Real-Time Recommendation Engine Deployment Guide

## Overview

This guide covers deploying the Real-Time Recommendation Engine from development to production environments. The system is designed to be cloud-agnostic and can be deployed on AWS, GCP, Azure, or on-premises infrastructure.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
            ┌─────────┴─────────┐
            │                   │
┌───────────▼─────────┐ ┌──────▼──────────┐
│   FastAPI Server    │ │  FastAPI Server │
│   (Recommendations) │ │  (Recommendations)│
└─────────┬───────────┘ └──────┬──────────┘
          │                    │
          └─────────┬──────────┘
                    │
    ┌───────────────▼────────────────┐
    │         Redis Cluster          │
    │    (Caching & Real-time)       │
    └───────────────┬────────────────┘
                    │
    ┌───────────────▼────────────────┐
    │       Kafka Cluster           │
    │    (Event Streaming)          │
    └───────────────┬────────────────┘
                    │
    ┌───────────────▼────────────────┐
    │    PostgreSQL Cluster         │
    │  (User/Item/Interaction Data) │
    └────────────────────────────────┘
```

## Prerequisites

### System Requirements

**Minimum Hardware (Development):**

- CPU: 4 cores
- RAM: 16 GB
- Storage: 100 GB SSD
- Network: 1 Gbps

**Production Hardware (per service):**

- CPU: 8-16 cores
- RAM: 32-64 GB
- Storage: 500 GB SSD (database), 100 GB SSD (application)
- Network: 10 Gbps

### Software Dependencies

```bash
# Required software versions
Docker >= 20.10.0
Docker Compose >= 2.0.0
Python >= 3.11
PostgreSQL >= 14
Redis >= 7.0
Apache Kafka >= 3.0
```

## Development Environment Setup

### 1. Clone and Initial Setup

```bash
# Clone the repository
git clone https://github.com/salgue441/recommendation-engine.git
cd recommendation-engine

# Copy environment configuration
cp .env.example .env

# Edit environment variables
nano .env
```

### 2. Environment Configuration

```bash
# .env file configuration
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=recommendation_engine
POSTGRES_USER=rec_user
POSTGRES_PASSWORD=secure_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_INTERACTIONS=user_interactions
KAFKA_TOPIC_RECOMMENDATIONS=recommendations

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# ML Configuration
MODEL_UPDATE_INTERVAL=3600  # seconds
MIN_INTERACTIONS_FOR_CF=10
DEFAULT_RECOMMENDATION_COUNT=10

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
```

### 3. Local Development with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api

# Run database migrations
docker-compose exec api python scripts/setup_database.py

# Generate sample data for testing
docker-compose exec api python scripts/generate_sample_data.py
```

**docker-compose.yml:**

```yaml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://rec_user:secure_password@postgres:5432/recommendation_engine
      - REDIS_URL=redis://:redis_password@redis:6379/0
    depends_on:
      - postgres
      - redis
      - kafka
    volumes:
      - ./app:/app/app
      - ./data:/app/data

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: recommendation_engine
      POSTGRES_USER: rec_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass redis_password
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  minio_data:
  grafana_data:
```

## Production Deployment

### 1. Container Orchestration (Kubernetes)

#### Namespace and Configuration

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: recommendation-engine

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rec-engine-config
  namespace: recommendation-engine
data:
  LOG_LEVEL: "INFO"
  MODEL_UPDATE_INTERVAL: "3600"
  DEFAULT_RECOMMENDATION_COUNT: "10"
  PROMETHEUS_ENABLED: "true"
```

#### Secrets Management

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rec-engine-secrets
  namespace: recommendation-engine
type: Opaque
stringData:
  POSTGRES_PASSWORD: "production_password"
  REDIS_PASSWORD: "redis_production_password"
  SECRET_KEY: "production-secret-key"
  JWT_SECRET: "jwt-production-secret"
```

#### Database Deployment

```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: recommendation-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:14
          env:
            - name: POSTGRES_DB
              value: recommendation_engine
            - name: POSTGRES_USER
              value: rec_user
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: rec-engine-secrets
                  key: POSTGRES_PASSWORD
            - name: POSTGRES_REPLICATION_MODE
              value: master
          ports:
            - containerPort: 5432
          volumeMounts:
            - name: postgres-storage
              mountPath: /var/lib/postgresql/data
          resources:
            requests:
              memory: "4Gi"
              cpu: "2"
            limits:
              memory: "8Gi"
              cpu: "4"
      volumes:
        - name: postgres-storage
          persistentVolumeClaim:
            claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: recommendation-engine
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
  type: ClusterIP
```

#### API Deployment

```yaml
# api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rec-engine-api
  namespace: recommendation-engine
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  selector:
    matchLabels:
      app: rec-engine-api
  template:
    metadata:
      labels:
        app: rec-engine-api
    spec:
      containers:
        - name: api
          image: rec-engine:latest
          ports:
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              value: "postgresql://rec_user:$(POSTGRES_PASSWORD)@postgres-service:5432/recommendation_engine"
            - name: REDIS_URL
              value: "redis://:$(REDIS_PASSWORD)@redis-service:6379/0"
          envFrom:
            - configMapRef:
                name: rec-engine-config
            - secretRef:
                name: rec-engine-secrets
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: rec-engine-api-service
  namespace: recommendation-engine
spec:
  selector:
    app: rec-engine-api
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
```

#### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rec-engine-api-hpa
  namespace: recommendation-engine
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rec-engine-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### 2. Cloud-Specific Deployments

#### AWS Deployment with EKS

```bash
# Install AWS CLI and eksctl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# Create EKS cluster
eksctl create cluster \
  --name recommendation-engine \
  --region us-west-2 \
  --nodegroup-name workers \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name recommendation-engine

# Deploy application
kubectl apply -f k8s/
```

**AWS-specific services integration:**

```yaml
# aws-services.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aws-config
data:
  AWS_REGION: "us-west-2"
  S3_BUCKET: "rec-engine-models"
  CLOUDWATCH_LOG_GROUP: "/aws/eks/recommendation-engine"
```

#### Google Cloud Deployment with GKE

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Create GKE cluster
gcloud container clusters create recommendation-engine \
  --machine-type=n1-standard-4 \
  --num-nodes=3 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --region=us-central1

# Get credentials
gcloud container clusters get-credentials recommendation-engine --region=us-central1

# Deploy application
kubectl apply -f k8s/
```

### 3. Database Setup and Migration

#### Initial Database Schema

```sql
-- scripts/init.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Users table
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preferences JSONB,
    demographic_data JSONB
);

-- Items table
CREATE TABLE items (
    id BIGSERIAL PRIMARY KEY,
    item_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    categories TEXT[],
    tags TEXT[],
    features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Interactions table (partitioned by date)
CREATE TABLE interactions (
    id BIGSERIAL,
    user_id VARCHAR(255) NOT NULL,
    item_id VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    rating FLOAT,
    context JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE interactions_2025_06 PARTITION OF interactions
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');

-- Indexes for performance
CREATE INDEX idx_users_user_id ON users(user_id);
CREATE INDEX idx_items_item_id ON items(item_id);
CREATE INDEX idx_items_categories ON items USING GIN(categories);
CREATE INDEX idx_interactions_user_id ON interactions(user_id);
CREATE INDEX idx_interactions_item_id ON interactions(item_id);
CREATE INDEX idx_interactions_timestamp ON interactions(timestamp);
```

#### Migration Scripts

```python
# scripts/migrate.py
import asyncio
import asyncpg
from app.config import settings

async def run_migration(migration_file):
    """Run a single migration file"""
    conn = await asyncpg.connect(settings.DATABASE_URL)

    with open(f"migrations/{migration_file}", 'r') as file:
        migration_sql = file.read()

    try:
        await conn.execute(migration_sql)
        print(f"Migration {migration_file} completed successfully")
    except Exception as e:
        print(f"Migration {migration_file} failed: {e}")
        raise
    finally:
        await conn.close()

async def run_all_migrations():
    """Run all pending migrations"""
    migration_files = [
        "001_initial_schema.sql",
        "002_add_user_preferences.sql",
        "003_partition_interactions.sql",
        "004_add_recommendation_cache.sql"
    ]

    for migration_file in migration_files:
        await run_migration(migration_file)

if __name__ == "__main__":
    asyncio.run(run_all_migrations())
```

### 4. Monitoring and Observability

#### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: "rec-engine-api"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: "/metrics"
    scrape_interval: 10s

  - job_name: "postgres"
    static_configs:
      - targets: ["postgres_exporter:9187"]

  - job_name: "redis"
    static_configs:
      - targets: ["redis_exporter:9121"]

  - job_name: "kafka"
    static_configs:
      - targets: ["kafka_exporter:9308"]

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093
```

#### Custom Metrics Implementation

```python
# app/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# API Metrics
REQUEST_COUNT = Counter(
    'recommendation_requests_total',
    'Total recommendation requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'recommendation_request_duration_seconds',
    'Recommendation request duration',
    ['method', 'endpoint']
)

RECOMMENDATION_LATENCY = Histogram(
    'recommendation_generation_latency_seconds',
    'Time to generate recommendations',
    ['algorithm']
)

# Business Metrics
ACTIVE_USERS = Gauge(
    'active_users_total',
    'Number of active users'
)

RECOMMENDATION_CTR = Gauge(
    'recommendation_click_through_rate',
    'Recommendation click-through rate',
    ['algorithm']
)

CACHE_HIT_RATE = Gauge(
    'cache_hit_rate',
    'Cache hit rate percentage'
)

def track_request(method, endpoint, status, duration):
    """Track API request metrics"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

def track_recommendation_generation(algorithm, duration):
    """Track recommendation generation metrics"""
    RECOMMENDATION_LATENCY.labels(algorithm=algorithm).observe(duration)
```

#### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Recommendation Engine Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(recommendation_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(recommendation_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "cache_hit_rate",
            "legendFormat": "Cache Hit Rate"
          }
        ]
      }
    ]
  }
}
```

### 5. CI/CD Pipeline

#### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy Recommendation Engine

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest tests/ --cov=app --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: |
          docker build -t rec-engine:${{ github.sha }} .
          docker tag rec-engine:${{ github.sha }} rec-engine:latest

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push rec-engine:${{ github.sha }}
          docker push rec-engine:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v3

      - name: Deploy to Kubernetes
        run: |
          echo ${{ secrets.KUBE_CONFIG }} | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

          # Update image in deployment
          kubectl set image deployment/rec-engine-api api=rec-engine:${{ github.sha }} -n recommendation-engine

          # Wait for rollout
          kubectl rollout status deployment/rec-engine-api -n recommendation-engine
```

### 6. Security Configuration

#### Network Security

```yaml
# network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rec-engine-network-policy
  namespace: recommendation-engine
spec:
  podSelector:
    matchLabels:
      app: rec-engine-api
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: load-balancer
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379
```

#### SSL/TLS Configuration

```yaml
# tls-config.yaml
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
  namespace: recommendation-engine
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTi... # base64 encoded certificate
  tls.key: LS0tLS1CRUdJTi... # base64 encoded private key

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rec-engine-ingress
  namespace: recommendation-engine
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - api.recommendation-engine.com
      secretName: tls-secret
  rules:
    - host: api.recommendation-engine.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: rec-engine-api-service
                port:
                  number: 80
```

### 7. Backup and Disaster Recovery

#### Database Backup Strategy

```bash
#!/bin/bash
# scripts/backup_database.sh

BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="rec_engine_backup_${TIMESTAMP}.sql"

# Create backup
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Upload to cloud storage (AWS S3 example)
aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" s3://rec-engine-backups/database/

# Clean up old backups (keep last 30 days)
find $BACKUP_DIR -name "rec_engine_backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

#### Model and Data Backup

```python
# scripts/backup_models.py
import boto3
import os
from datetime import datetime

def backup_models():
    """Backup trained models to S3"""
    s3_client = boto3.client('s3')
    bucket_name = 'rec-engine-model-backups'

    model_dir = '/app/data/models'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for model_file in os.listdir(model_dir):
        local_path = os.path.join(model_dir, model_file)
        s3_key = f"models/{timestamp}/{model_file}"

        s3_client.upload_file(local_path, bucket_name, s3_key)
        print(f"Uploaded {model_file} to s3://{bucket_name}/{s3_key}")

if __name__ == "__main__":
    backup_models()
```

### 8. Scaling Strategies

#### Vertical Scaling

```yaml
# Update resource limits for higher capacity
resources:
  requests:
    memory: "8Gi"
    cpu: "4"
  limits:
    memory: "16Gi"
    cpu: "8"
```

#### Horizontal Scaling

```bash
# Scale API pods
kubectl scale deployment rec-engine-api --replicas=10 -n recommendation-engine

# Auto-scaling based on custom metrics
kubectl autoscale deployment rec-engine-api \
  --cpu-percent=70 \
  --min=3 \
  --max=50 \
  -n recommendation-engine
```

#### Database Scaling

```sql
-- Read replicas configuration
CREATE PUBLICATION rec_engine_pub FOR ALL TABLES;

-- On replica servers
CREATE SUBSCRIPTION rec_engine_sub
CONNECTION 'host=primary-db port=5432 user=replication_user dbname=recommendation_engine'
PUBLICATION rec_engine_pub;
```

This deployment guide provides a comprehensive foundation for deploying the recommendation engine in various environments while maintaining security, scalability, and reliability.
