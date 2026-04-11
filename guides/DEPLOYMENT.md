# LexGuard MLOps & Deployment Guide

## Overview
This document covers deploying the LexGuard compliance engine as a containerized service for production use.

## Architecture

```
┌─────────────────────────────────────┐
│     FastAPI REST Service            │
│    (src/api/app.py)                 │
│  - /health                          │
│  - /compliance-check (POST)         │
└────────────┬────────────────────────┘
             │
      ┌──────▼──────────┐
      │  Models         │
      ├─────────────────┤
      │ - Static Graph  │
      │ - Eventic Graph │
      │ - SBERT Model   │
      │ - FAISS Index   │
      └─────────────────┘
```

## Quickstart

### Local Development (without Docker)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the API server:
   ```bash
   python -m src.api.app
   ```

3. Access the API:
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health
   - Compliance check: POST http://localhost:8000/compliance-check

### Run with Docker Compose

1. Build and start the service:
   ```bash
   docker-compose up --build
   ```

2. Access the service:
   - http://localhost:8000/docs
   - http://localhost:8000/health

3. Stop the service:
   ```bash
   docker-compose down
   ```

4. View logs:
   ```bash
   docker-compose logs -f
   ```

## API Endpoints

### 1. Health Check
```
GET /health
```

Response:
```json
{
  "status": "ok",
  "loaded_models": {
    "static_graph": true,
    "eventic_graph": true,
    "embeddings_model": true,
    "faiss_index": true
  }
}
```

### 2. Compliance Check
```
POST /compliance-check
Content-Type: application/json
```

Request body:
```json
{
  "text": "Your policy text here...",
  "lambda_thresh": 0.75,
  "hop_k": 1,
  "max_triples": 60,
  "prefer_local": false,
  "openai_model": "gpt-3.5-turbo"
}
```

Response:
```json
{
  "verdict": "pass",
  "evidence": [],
  "triples_text": "- Entity1 relatedTo Entity2\n...",
  "llm_reply": "...",
  "hits": ["node1", "node2"],
  "P": ["node3"],
  "N": ["node4", "node5"]
}
```

## Deployment Options

### Option 1: Local Docker (Development)
Use `docker-compose up` for local testing and development.

### Option 2: Cloud VM (AWS EC2, GCP Compute, Azure VM)
1. Push your image to a container registry (Docker Hub, ECR, GCR):
   ```bash
   docker build -t lexguard:latest .
   docker tag lexguard:latest docker.io/username/lexguard:latest
   docker push docker.io/username/lexguard:latest
   ```

2. Deploy on VM:
   ```bash
   # SSH into VM
   ssh user@vm-ip
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Pull and run
   docker run -p 8000:8000 \
     -v /path/to/data:/app/data:ro \
     docker.io/username/lexguard:latest
   ```

### Option 3: Managed Container Service
**AWS ECS**, **Google Cloud Run**, **Azure Container Instances**, or **Railway**:

Example for Railway.app:
1. Connect your GitHub repo
2. Railway auto-detects Dockerfile
3. Set environment variables (if needed)
4. Deploy with `git push`

Example for Google Cloud Run:
```bash
gcloud run deploy lexguard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Option 4: Kubernetes
For larger deployments, use Kubernetes:

1. Create a simple deployment manifest:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lexguard
spec:
  replicas: 2
  selector:
    matchLabels:
      app: lexguard
  template:
    metadata:
      labels:
        app: lexguard
    spec:
      containers:
      - name: lexguard
        image: username/lexguard:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        volumeMounts:
        - name: data
          mountPath: /app/data
          readOnly: true
      volumes:
      - name: data
        hostPath:
          path: /path/to/data
```

2. Deploy:
```bash
kubectl apply -f deployment.yaml
kubectl port-forward svc/lexguard 8000:8000
```

## Environment Variables

- **PORT**: API server port (default: 8000)
- **OPENAI_API_KEY**: OpenAI API key (optional, for cloud-based inference)
- **PYTHONUNBUFFERED**: Set to 1 for real-time logs in containers

## Monitoring & Logging

### Local Logging
The API logs to stdout with timestamps and levels. Docker Compose captures these:
```bash
docker-compose logs -f lexguard-api
```

### Production Logging
For production, consider:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **CloudWatch** (AWS)
- **Stackdriver** (Google Cloud)
- **Application Insights** (Azure)

### Metrics
Basic health metrics via `/health` endpoint. For advanced monitoring:
- Add **Prometheus** for metrics export
- Use **Grafana** for visualization
- Monitor API response times, error rates, model latency

## Performance Tuning

1. **Model Loading**: Pre-load models on startup (already done)
2. **Caching**: Consider caching embeddings for repeated texts
3. **Batch Processing**: For high throughput, add batch compliance check endpoint
4. **GPU**: If available, leverage GPU for embedding and LLM inference
   - Set `CUDA_VISIBLE_DEVICES` in docker-compose environment

## CI/CD Pipeline (GitHub Actions)

Create `.github/workflows/deploy.yml`:
```yaml
name: Build and Deploy

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: docker/setup-buildx-action@v2
      - uses: docker/build-push-action@v4
        with:
          push: true
          tags: username/lexguard:${{ github.sha }},username/lexguard:latest
          registry: docker.io
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
```

## Troubleshooting

### Models not loading
- Check data directory path in `src/api/app.py`
- Verify files exist: `data/static_graph.gpickle`, `data/eventic_graph.gpickle`

### Out of memory
- Run with limited model: Use `prefer_local: true` with smaller model
- Increase Docker memory limit in docker-compose.yml `mem_limit`

### Slow inference
- Enable GPU support
- Consider model quantization
- Add caching layer

## Next Steps

1. ✅ FastAPI service created
2. ✅ Dockerfile created
3. ✅ Docker Compose configured
4. ⏳ Add monitoring (Prometheus + Grafana)
5. ⏳ Set up CI/CD (GitHub Actions)
6. ⏳ Deploy to cloud provider
7. ⏳ Add API authentication/rate limiting
