# 🚀 START HERE - MLOps Setup (30 minutes)

## IF THIS IS YOUR FIRST TIME, FOLLOW THIS EXACTLY

### Step 1: Check Docker Is Installed (2 min)

```bash
docker --version
docker-compose --version
```

**If you see version numbers, you're good.** ✓

**If you see "command not found":**
- Download [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Install it
- Restart terminal

---

### Step 2: Build & Run (10 min)

Go to project folder:
```bash
cd /workspaces/comp_check
```

Run this ONE command:
```bash
docker-compose up --build
```

**Wait for these messages:**
```
lexguard-api | INFO:     Uvicorn running on http://0.0.0.0:8000
lexguard-api | INFO:     Application startup complete
```

**If it gets stuck on "Installing requirements...": Don't worry, just wait. First build takes 5-10 minutes.**

---

### Step 3: Open Your Browser (1 min)

While the command is running, open:
```
http://localhost:8000/docs
```

You should see an interactive API page with this:
- **GET /health** - Check if service is alive
- **POST /compliance-check** - Test the compliance check
- **GET /** - API info

---

### Step 4: Test the API (2 min)

1. Click on **POST /compliance-check**
2. Click the blue **"Try it out"** button
3. You'll see a text box with sample data
4. Click **"Execute"** button
5. Scroll down to see the **Response**

Should see something like:
```json
{
  "verdict": "pass",
  "evidence": [],
  "triples_text": "...",
  "hits": [...]
}
```

✅ **Congratulations! Your MLOps setup is working!**

---

### Step 5: Stop the Service (1 min)

Go back to terminal and press:
```
Ctrl + C
```

---

## What You Just Did

| What | Explanation |
|------|-------------|
| `docker-compose up --build` | Create a container + run your app inside |
| `http://localhost:8000/docs` | Web interface to interact with your API |
| Container | A box that has Python, your code, and everything needed |

---

## Now What?

### Option A: Keep Learning (Recommended)
Read: `MLOPS_BEGINNER_GUIDE.md` in this folder

### Option B: Deploy to Internet (Advanced)
Jump to: `DEPLOYMENT.md`

### Option C: Test with a Script
```bash
# (while your service is running in another terminal)
python test_api.py --url http://localhost:8000
```

---

## If Something Goes Wrong

### Error: "Port 8000 already in use"
```bash
# Solution 1: Use a different port
# Edit docker-compose.yml, change this line:
ports:
  - "8001:8000"  # Use 8001 instead

# Solution 2: Kill what's using port 8000
lsof -i :8000
kill -9 <PID>
```

### Error: "Docker daemon not running"
- On Windows/Mac: Open Docker Desktop app
- On Linux: Run `sudo systemctl start docker`

### Error: "Installation failed"
```bash
# Clean and rebuild
docker-compose down -v
docker-compose up --build
```

### Logs are confusing?
That's normal! Just wait for "`Application startup complete`"

---

## Common Questions

**Q: Does my laptop need a GPU?**
A: No! It works fine on CPU (slower, but fine for learning).

**Q: Can I edit the code?**
A: Yes! Just edit files, then `docker-compose down` and `docker-compose up --build` again.

**Q: Why does it take so long first time?**
A: Downloading Python packages (1 GB+ of files). Second build is faster (uses cache).

**Q: After I stop it, do I lose my data?**
A: No, your data is safe. Just restart with `docker-compose up`.

---

## Next Steps (Learning Path)

### Week 1 (YOU ARE HERE ✓)
- [x] Get service running with Docker
- [ ] Test API endpoints

### Week 2
- [ ] Read MLOPS_BEGINNER_GUIDE.md
- [ ] Understand what each file does
- [ ] Try calling API from Python script

### Week 3
- [ ] Deploy to free cloud service (Railway.app or Render.com)
- [ ] Learn about monitoring/logs
- [ ] Add API documentation to your GitHub

### Week 4+
- [ ] Learn Kubernetes (if interested)
- [ ] Add authentication
- [ ] Build a simple frontend

---

## Cheat Sheet (Commands You'll Use)

```bash
# Start service
docker-compose up --build

# Stop service
Ctrl + C

# View logs
docker-compose logs -f

# Enter container terminal
docker-compose exec lexguard-api bash

# Clean everything
docker-compose down -v

# Just build (don't run)
docker build .

# Show running containers
docker-compose ps
```

---

## You're Ready! 🎉

Now go to browser and visit:
```
http://localhost:8000/docs
```

Enjoy your MLOps journey!
