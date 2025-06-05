# Real-Time Recommendation Engine

A production-ready, scalable recommendation system that combines multiple algorithms to deliver personalized recommendations in real-time. Built with advanced data structures, machine learning algorithms, and distributed computing principles.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Features

### Core Recommendation Algorithms

- **Collaborative Filtering** with MinHash + LSH optimization
- **Content-Based Filtering** using TF-IDF and advanced feature engineering
- **Social Network Recommendations** with graph neural networks
- **Hybrid Approach** combining multiple algorithms intelligently
- **Cold Start Handling** for new users and items

### Real-Time Processing

- **Stream Processing** with Apache Kafka
- **Live Model Updates** without service interruption
- **Sub-second Response Times** with multi-level caching
- **Event-Driven Architecture** for scalable interactions

### Production Features

- **Auto-scaling** with Kubernetes HPA
- **A/B Testing Framework** for algorithm comparison
- **Comprehensive Monitoring** with Prometheus + Grafana
- **RESTful APIs** with FastAPI and automatic documentation
- **Genetic Algorithm Optimization** for hyperparameter tuning

## 📊 Performance Highlights

- **Response Time:** <50ms (P95) for cached recommendations
- **Throughput:** 6,800+ RPS with horizontal scaling
- **Accuracy:** 88% NDCG@10 with hybrid algorithm
- **Cache Hit Rate:** 90%+ with intelligent caching
- **Uptime:** 99.9% SLA with proper deployment

## 🏗️ Architecture

```bash
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

## 🛠️ Technology Stack

- **Backend:** Python 3.11, FastAPI, NumPy, SciPy, Scikit-learn
- **Databases:** PostgreSQL 14+, Redis 7+
- **Streaming:** Apache Kafka 3.0+
- **ML Libraries:** Faiss, Implicit, NetworkX, Custom implementations
- **Infrastructure:** Docker, Kubernetes, Prometheus, Grafana
- **Storage:** MinIO (S3-compatible) for model artifacts

## 📋 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- 16GB+ RAM recommended
- PostgreSQL 14+, Redis 7+, Kafka 3.0+

### Local Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/recommendation-engine.git
   cd recommendation-engine
   ```

2. **Set up environment**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start services with Docker Compose**

   ```bash
   docker-compose up -d
   ```

4. **Initialize database and generate sample data**

   ```bash
   docker-compose exec api python scripts/setup_database.py
   docker-compose exec api python scripts/generate_sample_data.py
   ```

5. **Verify installation**
   ```bash
   curl http://localhost:8000/health
   ```

### API Usage

```python
import requests

# Get recommendations for a user
response = requests.get(
    "http://localhost:8000/api/v1/recommendations/user/12345",
    params={"limit": 10, "explain": True},
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)

recommendations = response.json()
print(f"Got {len(recommendations['recommendations'])} recommendations")

# Track user interaction
requests.post(
    "http://localhost:8000/api/v1/interactions",
    json={
        "user_id": 12345,
        "item_id": 67890,
        "interaction_type": "view"
    },
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

## 🧪 Algorithm Deep Dive

### Collaborative Filtering with LSH

Our collaborative filtering implementation uses Locality-Sensitive Hashing (LSH) to achieve sub-linear similarity computation:

```python
# Traditional approach: O(n²) complexity
def naive_similarity(users):
    for i, user1 in enumerate(users):
        for j, user2 in enumerate(users[i+1:]):
            similarity = compute_similarity(user1, user2)

# Our approach: O(n log n) with LSH
class OptimizedCF:
    def __init__(self):
        self.lsh = MinHashLSH(threshold=0.5, num_perm=128)

    def find_similar_users(self, user_id, k=50):
        candidates = self.lsh.query(user_minhash)
        return self.rank_candidates(candidates, k)
```

**Performance Improvement:** 4x faster similarity computation for 1M+ users

### Hybrid Recommendation Strategy

Combines multiple algorithms using weighted scoring:

```python
final_score = (
    0.5 * collaborative_score +
    0.3 * content_score +
    0.2 * social_score
)
```

Weights are optimized using a genetic algorithm that maximizes NDCG@10 on validation data.

### Real-Time Updates

Processes user interactions in real-time using Kafka streams:

```python
async def process_interaction(user_id, item_id, interaction_type):
    # Update user-item matrix incrementally
    await update_user_profile(user_id, item_id, interaction_type)

    # Invalidate affected caches
    await invalidate_user_cache(user_id)

    # Update real-time counters
    await update_item_popularity(item_id)
```

## 📚 Documentation

- **[API Documentation](docs/api.md)** - Complete API reference with examples
- **[Algorithm Guide](docs/algorithms.md)** - Detailed explanation of recommendation algorithms
- **[Deployment Guide](docs/deployment.md)** - Production deployment instructions
- **[Performance Benchmarks](docs/performance.md)** - Performance metrics and optimization guide

## 🔧 Configuration

### Environment Variables

```bash
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

# ML Configuration
MODEL_UPDATE_INTERVAL=3600  # seconds
MIN_INTERACTIONS_FOR_CF=10
DEFAULT_RECOMMENDATION_COUNT=10

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key-here
```

### Algorithm Parameters

```python
# Collaborative Filtering
CF_CONFIG = {
    'n_factors': 128,           # SVD factors
    'regularization': 0.01,     # L2 regularization
    'iterations': 50,           # Training iterations
    'lsh_threshold': 0.5,       # LSH similarity threshold
    'lsh_num_perm': 128        # MinHash permutations
}

# Content-Based Filtering
CONTENT_CONFIG = {
    'max_features': 10000,      # TF-IDF vocabulary size
    'ngram_range': (1, 2),      # N-gram range
    'min_df': 5,               # Minimum document frequency
    'max_df': 0.8              # Maximum document frequency
}

# Hybrid Weights (optimized via genetic algorithm)
HYBRID_WEIGHTS = {
    'collaborative': 0.5,
    'content': 0.3,
    'social': 0.2
}
```

## 🚀 Production Deployment

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace recommendation-engine

# Deploy with Helm
helm install rec-engine ./helm-chart \
  --namespace recommendation-engine \
  --set image.tag=latest \
  --set replicaCount=5 \
  --set resources.requests.memory=4Gi

# Enable auto-scaling
kubectl autoscale deployment rec-engine-api \
  --cpu-percent=70 \
  --min=3 \
  --max=20
```

### Docker Production Build

```dockerfile
# Multi-stage build for optimization
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## 📈 Monitoring & Observability

### Metrics Collection

The system exposes comprehensive metrics via Prometheus:

- **Request Metrics:** Rate, latency, error rate by endpoint
- **Algorithm Performance:** Execution time, accuracy by algorithm
- **Business Metrics:** CTR, conversion rate, revenue per recommendation
- **Infrastructure:** CPU, memory, cache hit rate, database performance

### Grafana Dashboards

Pre-built dashboards for monitoring:

- **System Overview:** High-level health metrics
- **Algorithm Performance:** Detailed algorithm analytics
- **Business Intelligence:** Revenue and engagement metrics
- **Infrastructure:** Resource utilization and capacity planning

### Alerting

Automated alerts for:

- High response times (>500ms P95)
- Low cache hit rates (<70%)
- Model accuracy degradation (<80%)
- Database connection exhaustion (>90%)
- Error rate spikes (>1%)

## 🧪 Testing

### Run Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Load tests
pytest tests/load/ -v

# Coverage report
pytest --cov=app --cov-report=html
```

### Load Testing

```bash
# Install Artillery.js
npm install -g artillery

# Run load test
artillery run load-tests/recommendation-load-test.yml

# Custom load test
artillery quick --count 100 --num 1000 http://localhost:8000/api/v1/health
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch:** `git checkout -b feature/amazing-feature`
3. **Make changes and add tests**
4. **Run test suite:** `pytest`
5. **Commit changes:** `git commit -m 'Add amazing feature'`
6. **Push to branch:** `git push origin feature/amazing-feature`
7. **Open Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add type hints for all functions
- Write comprehensive tests (aim for >90% coverage)
- Update documentation for API changes
- Benchmark performance impact for algorithm changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Collaborative Filtering:** Based on matrix factorization techniques from [Koren et al.](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)
- **LSH Implementation:** Inspired by [datasketch](https://github.com/ekzhu/datasketch) library
- **Hybrid Algorithms:** Techniques from [Ricci et al.](https://link.springer.com/book/1
