# Deployment Guide: Eurostat Dashboard

Deploy backend to Railway, frontend to Vercel, with custom domain `visiometrica.com/eurostat`.

---

## Prerequisites

- GitHub account (you have this)
- Railway account (free tier works)
- Vercel account (free tier works)
- Access to name.com DNS settings

---

## Part 1: Push Code to GitHub

### 1.1 Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `eurostat-dashboard` (or your preference)
3. Keep it **Public** or **Private** (your choice)
4. **Do NOT** initialize with README (we already have one)
5. Click **Create repository**

### 1.2 Push Your Code

Run these commands in your terminal:

```bash
cd /Users/victor/Documents/Projects/eurostat_dashboard

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Eurostat dashboard with GDP, productivity, and GVA charts"

# Add your GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/eurostat-dashboard.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Part 2: Deploy Backend to Railway

### 2.1 Create Railway Account

1. Go to https://railway.app
2. Click **Login** → **Login with GitHub**
3. Authorize Railway to access your GitHub

### 2.2 Create New Project

1. Click **New Project**
2. Select **Deploy from GitHub repo**
3. Find and select `eurostat-dashboard`
4. Railway will ask which directory to deploy

### 2.3 Configure Backend Service

1. After selecting the repo, click **Add Service** → **GitHub Repo**
2. Select your `eurostat-dashboard` repo again
3. Click on the newly created service to configure it
4. Go to **Settings** tab:
   - **Root Directory**: `backend`
   - **Build Command**: (leave empty, Railway auto-detects)
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

5. Go to **Variables** tab and add:
   - `PYTHON_VERSION` = `3.12`

6. Click **Deploy**

### 2.4 Get Your Railway URL

1. Once deployed, go to **Settings** → **Networking**
2. Click **Generate Domain**
3. You'll get a URL like: `eurostat-dashboard-production-xxxx.up.railway.app`
4. **Copy this URL** - you'll need it for the frontend

### 2.5 Test Backend

Open in browser:
```
https://YOUR_RAILWAY_URL.railway.app/api/v1/health
```

Should return: `{"status":"ok"}`

---

## Part 3: Update Frontend with Backend URL

### 3.1 Update the API URL

Edit `frontend/index.html` and replace:
```javascript
: 'RAILWAY_API_URL_PLACEHOLDER/api/v1';
```

With your actual Railway URL:
```javascript
: 'https://eurostat-dashboard-production-xxxx.up.railway.app/api/v1';
```

### 3.2 Commit and Push

```bash
git add frontend/index.html
git commit -m "Configure production API URL"
git push
```

---

## Part 4: Deploy Frontend to Vercel

### 4.1 Create Vercel Account

1. Go to https://vercel.com
2. Click **Sign Up** → **Continue with GitHub**
3. Authorize Vercel

### 4.2 Import Project

1. Click **Add New...** → **Project**
2. Find and **Import** `eurostat-dashboard`

### 4.3 Configure Build Settings

1. **Framework Preset**: `Other`
2. **Root Directory**: Click **Edit** → select `frontend`
3. **Build Command**: (leave empty)
4. **Output Directory**: `.` (just a dot)
5. **Install Command**: (leave empty)

6. Click **Deploy**

### 4.4 Get Your Vercel URL

After deployment, you'll get a URL like:
`eurostat-dashboard.vercel.app`

Test it - you should see your dashboard with live data!

---

## Part 5: Configure Custom Domain

### 5.1 Add Domain in Vercel

1. In Vercel, go to your project → **Settings** → **Domains**
2. Add: `visiometrica.com`
3. Vercel will show you DNS records to add

### 5.2 Configure DNS at name.com

1. Log in to https://www.name.com
2. Go to **My Domains** → **visiometrica.com** → **DNS Records**

3. Add these records (Vercel will tell you the exact values):

   **For root domain (visiometrica.com):**
   | Type | Host | Value |
   |------|------|-------|
   | A | @ | 76.76.21.21 |

   **For www redirect (optional):**
   | Type | Host | Value |
   |------|------|-------|
   | CNAME | www | cname.vercel-dns.com |

4. Wait for DNS propagation (5-30 minutes)

### 5.3 Verify in Vercel

1. Go back to Vercel **Domains** settings
2. The domain should show a green checkmark when ready
3. Vercel automatically provisions SSL/HTTPS

---

## Part 6: Configure CORS on Backend

The backend needs to allow requests from your domain. Update the CORS settings:

### 6.1 Update Backend CORS

Edit `backend/main.py`, find the CORS middleware section and update:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://visiometrica.com",
        "https://www.visiometrica.com",
        "https://eurostat-dashboard.vercel.app",  # your Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 6.2 Commit and Push

```bash
git add backend/main.py
git commit -m "Update CORS for production domains"
git push
```

Railway will auto-redeploy.

---

## Verification Checklist

- [ ] Backend health check works: `https://YOUR_RAILWAY_URL/api/v1/health`
- [ ] Backend data works: `https://YOUR_RAILWAY_URL/api/v1/data?dataset=sdg_08_10&geo=BG&time=2020-2024`
- [ ] Frontend loads on Vercel URL
- [ ] Frontend loads on `https://visiometrica.com`
- [ ] All 4 charts display data correctly

---

## Troubleshooting

### Charts show "Error: API error"
- Check browser console (F12) for CORS errors
- Verify the API_BASE URL in frontend/index.html is correct
- Check Railway logs for backend errors

### Domain not working
- DNS propagation can take up to 48 hours (usually 5-30 min)
- Use https://dnschecker.org to verify DNS records
- Ensure you added the correct records at name.com

### Railway deployment fails
- Check that Root Directory is set to `backend`
- Verify requirements.txt is present
- Check Railway build logs for specific errors

---

## Costs

- **Railway**: Free tier includes $5/month credit (sufficient for low traffic)
- **Vercel**: Free tier is generous for static sites
- **Total**: $0/month for low usage

---

## Auto-Deployments

Both Railway and Vercel will auto-deploy when you push to the `main` branch on GitHub.
