# MLOps for Beginners - Step by Step

## What Is MLOps?

Think of it like this:

### Before MLOps (Just a Script)
```
You: python rag_runner.py
Output: results.json ✓
But: Only works on YOUR computer, hard to use, not scalable
```

### After MLOps (Production Ready)
```
Anyone: Makes HTTP request
  ↓
Your service answers via API
  ↓
Works everywhere (laptop, server, cloud)
  ↓
Easy to scale, monitor, update
```

---

## The 3 Layers (Simple Version)

### Layer 1: Application (What You Built)
```
YOU ALREADY HAVE THIS ✓
├── src/retriever_rag/rag_runner.py  ← The core logic
└── Works locally: python src/retriever_rag/rag_runner.py
```

### Layer 2: API Service (We Added This)
```
NEW: Makes the script accessible
├── src/api/app.py  ← Wraps rag_runner.py as a web service
└── Access via: http://localhost:8000
```

### Layer 3: Container (We Added This)
```
NEW: Makes it portable
├── Dockerfile  ← Instructions to package everything
└── Result: Works on any computer/server
```

---

## Simple Analogy

**Without Docker:**
```
Your Laptop: Python 3.12, sentence-transformers 5.1, torch 2.8, ...
Friend's Laptop: Python 3.10, sentence-transformers 4.0, cuda 11, ...
Result: ❌ "It works on my machine but not yours"
```

**With Docker:**
```
Dockerfile says: "Use Python 3.12, install these exact versions"
My Laptop: ✓ Works
Friend's Laptop: ✓ Works
AWS Server: ✓ Works
Result: ✓ Consistency everywhere
```

---

## Quick Start (Follow This Exactly)

### Step 1: Install Docker
- Windows/Mac: Download [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Linux:
  ```bash
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  ```

### Step 2: Navigate to Project
```bash
cd /workspaces/comp_check
```

### Step 3: Build & Run (One Command)
```bash
docker-compose up --build
```

That's it! After a few minutes (first time is slow), you'll see:
```
lexguard-api | INFO:     Application startup complete
lexguard-api | INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Test It Works
Open a browser: http://localhost:8000/docs

You should see an interactive API documentation page.

### Step 5: Try the API
Click the big green "Try it out" button on the `/compliance-check` endpoint.

### Step 6: Stop It
Press `Ctrl+C` in terminal, or run:
```bash
docker-compose down
```

---

## What Each Command Does

| Command | What It Does |
|---------|-------------|
| `docker-compose up --build` | Build container + start service |
| `docker-compose logs -f` | Watch live logs |
| `docker-compose ps` | Show running containers |
| `docker-compose down` | Stop everything |
| `docker build .` | Just build (no run) |

---

## Understanding the Files (In Plain English)

### `src/api/app.py`
**What it is**: A web version of your script
```python
@app.post("/compliance-check")
def check(text):
    # This is what rag_runner.py does, but via HTTP
    return results
```
**How to use**: Browser or Python requests library

### `Dockerfile`
**What it is**: Recipe to make your app portable
```
1. Start with Python 3.12
2. Install required system packages
3. Install Python libraries
4. Copy your code
5. Run the app
```
**Why**: So it works the same everywhere

### `docker-compose.yml`
**What it is**: Shortcut to run Docker without typing long commands
```yaml
- Instead of: docker run -p 8000:8000 -v data:/app/data ...
- Just use: docker-compose up
```

### `.github/workflows/ci.yml`
**What it is**: Automatic testing when you push to GitHub
**Why**: Catches bugs before deployment

---

## The Flow (Visual)

```
┌─────────────────────────────────┐
│   Your Browser / Mobile App     │
│   Makes HTTP request            │
└────────────┬────────────────────┘
             │
             │ GET /health
             │ POST /compliance-check
             ↓
┌─────────────────────────────────┐
│   FastAPI Service (app.py)      │
│   - Receives request            │
│   - Processes it                │
│   - Returns JSON response       │
└────────────┬────────────────────┘
             │
             ↓
┌─────────────────────────────────┐
│   RAG Pipeline                  │
│   - rag_runner.py logic         │
│   - Embeddings, graphs, LLM     │
└────────────┬────────────────────┘
             │
             ↓
┌─────────────────────────────────┐
│   Data Files                    │
│   - chunks.json                 │
│   - static_graph.gpickle        │
│   - faiss.index                 │
└─────────────────────────────────┘
```

---

## Common Questions for Beginners

### Q: Why do I need Docker?
**A**: Same setup works on laptop, server, and cloud. No "works on my machine" problems.

### Q: Why FastAPI?
**A**: Makes your Python script accessible as a web service anyone can use.

### Q: Can I just run the script without Docker?
**A**: Yes! For development:
```bash
python -m src.api.app
# Then visit http://localhost:8000/docs
```
(But Docker is better for production)

### Q: What's the difference between docker-compose and Docker?
**A**: 
- **Docker**: Container technology (the engine)
- **docker-compose**: Shortcut to manage containers (the remote control)

### Q: How do I deploy this to the internet?
**A**: Several options (pick one):
1. **Beginner**: Use Railway.app (auto-deploys from GitHub)
2. **Intermediate**: AWS EC2 + Docker
3. **Advanced**: Kubernetes

---

## Your Next Steps

### Now (Today)
1. ✅ Read this file
2. ✅ Make sure Docker is installed
3. ✅ Run `docker-compose up --build`
4. ✅ Visit http://localhost:8000/docs

### Tomorrow
- Learn the `/health` endpoint (checks if service is alive)
- Learn the `/compliance-check` endpoint (your main API)
- Test with `python test_api.py`

### This Week
- Deploy to a free cloud service (Railway, Render, or Google Cloud Run)
- Monitor the logs
- Make a simple frontend to use the API

### This Month
- Add authentication (API keys)
- Add metrics/monitoring
- Learn Kubernetes basics

---

## Debugging Tips

### Build fails?
```bash
docker-compose down -v  # Clean everything
docker-compose up --build  # Fresh build
```

### Port 8000 already in use?
```bash
# Kill whatever's using port 8000
lsof -i :8000
kill -9 <PID>

# Or use different port in docker-compose.yml
ports:
  - "8001:8000"  # Change first 8000 to 8001
```

### Want to see what's in the container?
```bash
docker-compose exec lexguard-api bash
ls -la /app/data  # Explore inside
```

---

## Most Important Thing

**You don't need to understand everything right now.**

Just focus on:
1. Run `docker-compose up`
2. Visit http://localhost:8000/docs
3. Click "Try it out" on an endpoint
4. Watch the magic happen

The rest will make sense with time. MLOps is a journey, not a sprint.

---

## Questions?

Ask me about:
- ❓ What does this error mean?
- ❓ How do I deploy to AWS?
- ❓ What's a healthcheck?
- ❓ How do I add authentication?

I'll explain in simple terms. 🚀
