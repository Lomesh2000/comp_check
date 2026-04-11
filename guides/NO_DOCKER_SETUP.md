# MLOps Without Docker (Codespace Edition)

## The Problem
Docker builds are resource-heavy and dying in your codespace.

## The Solution
Run the API directly - it's simpler and works perfectly for learning!

---

## Setup (5 minutes)

### Step 1: Install Dependencies
```bash
cd /workspaces/comp_check
pip install -r requirements.txt
```

(This uses installation cache and takes 2-3 minutes)

### Step 2: Start the API
```bash
python -m src.api.app
```

### Step 3: Test in Browser
Open: http://localhost:8000/docs

---

## What You Get

✅ **Same API** - All endpoints work  
✅ **Same MLOps** - Just without Docker container  
✅ **Easier** - Fewer things to debug  
✅ **Good for Learning** - Understand the concepts first  

---

## Commands You'll Use

```bash
# Start API
python -m src.api.app

# In another terminal, test it
python test_api.py --url http://localhost:8000

# Stop API
Ctrl + C
```

---

## Deploy to Cloud (When Ready)

Once you understand MLOps on codespace, deploy to:

**Option 1: Railway.app** (Easiest)
- Create account (free)
- Connect GitHub
- Railway auto-detects Python + deploys
- No Docker needed

**Option 2: Google Cloud Run** (Also Easy)
- Upload code to GitHub
- Cloud Run handles deployment
- 2 million free requests/month

**Option 3: Heroku** (Classic)
- Old but reliable
- 1 free tier per account

---

## Why This Is Better for NOW

| Aspect | Docker | Direct |
|--------|--------|--------|
| Setup time | 15+ min | 3 min |
| Disk space | 4-5 GB | 500 MB |
| Debugging | Harder | Easy |
| Learning | Indirect | Direct |
| Production? | Yes | Use the other methods |

---

## Getting Started

```bash
cd /workspaces/comp_check

# Install
pip install -r requirements.txt

# Run
python -m src.api.app

# Then in browser
http://localhost:8000/docs
```

That's it! 🚀

---

## When You Need Docker

Docker is needed when:
- ✅ Deploying to production servers
- ✅ Running on different machines (Mac/Windows/Linux differences)
- ✅ Complex system dependencies
- ✅ Scaling to multiple servers

**For learning? Start without it.**

---

## Next Steps

1. Get the API running locally
2. Understand each endpoint
3. Make requests from Python
4. Then deploy to cloud
5. Then learn Docker for production

You'll understand MLOps better this way! 📚
