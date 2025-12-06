# Deployment Guide - FeatureBeats

This guide covers different deployment options for the FeatureBeats Hit Song Predictor web application.

---

## Quick Start (Local Development)

```bash
# 1. Clone the repository
git clone https://github.com/jairajsaraf/FeatureBeats.git
cd FeatureBeats

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the web app
streamlit run app.py

# 4. Open browser
# Navigate to: http://localhost:8501
```

---

## Option 1: Streamlit Cloud (Recommended - Free)

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Steps

1. **Push to GitHub** (if not already done)
   ```bash
   git push origin main
   ```

2. **Login to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

3. **Deploy App**
   - Click "New app"
   - Repository: `jairajsaraf/FeatureBeats`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

4. **Configure (if needed)**
   - Add secrets (if using APIs)
   - Configure custom subdomain

5. **Access Your App**
   - URL: `https://your-app-name.streamlit.app`
   - Share link with others!

### Advantages
✅ Free hosting
✅ Auto-updates on git push
✅ HTTPS enabled
✅ Easy sharing
✅ No DevOps required

### Limitations
⚠️ Limited resources (1 GB RAM)
⚠️ Apps sleep after inactivity
⚠️ Public by default

---

## Option 2: Docker Container

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t featurebeats .

# Run container
docker run -p 8501:8501 featurebeats

# Access at http://localhost:8501
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  featurebeats:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_PORT=8501
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./figures:/app/figures
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

### Advantages
✅ Consistent environment
✅ Easy deployment to cloud
✅ Scalable
✅ Isolated dependencies

---

## Option 3: Heroku

### Prerequisites
- Heroku account
- Heroku CLI installed

### Files Needed

1. **Procfile**
```
web: sh setup.sh && streamlit run app.py
```

2. **setup.sh**
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

3. **runtime.txt**
```
python-3.11.0
```

### Deploy

```bash
# Login to Heroku
heroku login

# Create app
heroku create featurebeats

# Deploy
git push heroku main

# Open app
heroku open
```

### Advantages
✅ Free tier available
✅ Custom domain support
✅ Add-ons ecosystem
✅ Simple scaling

### Limitations
⚠️ Dyno sleeps after 30 min (free tier)
⚠️ Limited free hours/month

---

## Option 4: AWS EC2

### Steps

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04
   - Instance type: t2.micro (free tier) or t2.small
   - Security group: Allow port 8501

2. **Connect and Setup**
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3-pip -y

# Clone repository
git clone https://github.com/jairajsaraf/FeatureBeats.git
cd FeatureBeats

# Install dependencies
pip3 install -r requirements.txt

# Run with nohup (background)
nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &
```

3. **Access**
   - URL: `http://your-ec2-ip:8501`

### Production Setup (with Nginx)

1. **Install Nginx**
```bash
sudo apt install nginx -y
```

2. **Configure Nginx** (`/etc/nginx/sites-available/featurebeats`)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

3. **Enable and Start**
```bash
sudo ln -s /etc/nginx/sites-available/featurebeats /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

4. **Process Manager (PM2)**
```bash
# Install PM2
sudo npm install -g pm2

# Create ecosystem file
pm2 start streamlit --name featurebeats -- run app.py

# Save configuration
pm2 save
pm2 startup
```

### Advantages
✅ Full control
✅ Scalable
✅ Custom configuration
✅ Can add databases, APIs

### Disadvantages
⚠️ Requires DevOps knowledge
⚠️ Manual maintenance
⚠️ Cost (after free tier)

---

## Option 5: Google Cloud Run

### Prerequisites
- Google Cloud account
- gcloud CLI installed

### Dockerfile

(Same as Option 2)

### Deploy

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/featurebeats

# Deploy to Cloud Run
gcloud run deploy featurebeats \
  --image gcr.io/YOUR_PROJECT_ID/featurebeats \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501

# Get URL
gcloud run services describe featurebeats --region us-central1 --format 'value(status.url)'
```

### Advantages
✅ Serverless (pay per use)
✅ Auto-scaling
✅ HTTPS by default
✅ Custom domains

---

## Option 6: Azure Web Apps

### Steps

1. **Create Web App**
```bash
# Login
az login

# Create resource group
az group create --name FeatureBeats --location eastus

# Create App Service plan
az appservice plan create --name FeatureBeatsplan --resource-group FeatureBeats --sku B1 --is-linux

# Create web app
az webapp create --resource-group FeatureBeats --plan FeatureBeatsplan --name featurebeats --runtime "PYTHON:3.11"
```

2. **Deploy Code**
```bash
# Configure deployment
az webapp deployment source config-local-git --name featurebeats --resource-group FeatureBeats

# Push code
git remote add azure <deployment-url>
git push azure main
```

3. **Configure Startup**
   - Set startup command: `streamlit run app.py --server.port=8000 --server.address=0.0.0.0`

### Advantages
✅ Microsoft ecosystem
✅ Enterprise support
✅ Integrated with Azure services

---

## Environment Variables

For all deployment options, you may need to set:

```bash
# Streamlit config
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_THEME_BASE=light

# Optional: Custom settings
MODEL_PATH=/app/models
DATA_PATH=/app/data
```

---

## Performance Optimization

### 1. Model Loading
```python
# Use caching to load model once
@st.cache_resource
def load_model():
    return joblib.load('models/final_xgboost.pkl')
```

### 2. Data Loading
```python
# Cache data loading
@st.cache_data
def load_data():
    return pd.read_csv('data/processed/hits_dataset.csv')
```

### 3. Resource Limits
For containers, set limits:
```yaml
deploy:
  resources:
    limits:
      cpus: '0.5'
      memory: 512M
```

---

## Monitoring & Maintenance

### Health Checks
```python
# Add health endpoint in app.py
import streamlit as st

if st.sidebar.checkbox("Health Check"):
    st.success("App is running!")
```

### Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### Error Tracking
Consider integrating:
- Sentry for error tracking
- Google Analytics for usage
- Prometheus for metrics

---

## Cost Comparison

| Platform | Free Tier | Paid (Monthly) | Best For |
|----------|-----------|----------------|----------|
| **Streamlit Cloud** | Yes (unlimited) | N/A | Quick sharing, demos |
| **Heroku** | 550-1000 hrs/month | $7+ | Simple deployment |
| **AWS EC2** | 750 hrs/month (12 mo) | $5-50+ | Full control |
| **Google Cloud Run** | 2M requests/month | $0.01/req | Scalability |
| **Azure** | $200 credit | $13+ | Enterprise |
| **Docker (Self-hosted)** | Free | $5+ (VPS) | Maximum control |

---

## Recommended Deployment Path

### For Demos & Prototypes
→ **Streamlit Cloud** (easiest, free)

### For Small Projects
→ **Heroku** or **Docker** (simple, affordable)

### For Production/Scale
→ **AWS/GCP/Azure** with Docker (professional, scalable)

### For Learning
→ **AWS EC2 Free Tier** (hands-on experience)

---

## Security Considerations

1. **Environment Variables**
   - Never commit secrets to Git
   - Use environment variables or secrets management

2. **HTTPS**
   - Always use HTTPS in production
   - Free with Let's Encrypt or cloud platforms

3. **Rate Limiting**
   - Add rate limiting to prevent abuse
   - Use API gateways or Nginx

4. **Authentication** (if needed)
   ```python
   # Simple auth in Streamlit
   import streamlit as st

   def check_password():
       def password_entered():
           if st.session_state["password"] == "your-password":
               st.session_state["password_correct"] = True
           else:
               st.session_state["password_correct"] = False

       if "password_correct" not in st.session_state:
           st.text_input("Password", type="password", on_change=password_entered, key="password")
           return False
       elif not st.session_state["password_correct"]:
           st.text_input("Password", type="password", on_change=password_entered, key="password")
           st.error("Password incorrect")
           return False
       else:
           return True

   if check_password():
       # Show app
       pass
   ```

---

## Troubleshooting

### Common Issues

**Issue:** App won't start
```bash
# Check logs
docker logs featurebeats
# or
heroku logs --tail
# or
pm2 logs
```

**Issue:** Port already in use
```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9
```

**Issue:** Model not found
```bash
# Verify model files are included
ls -la models/
# Check .gitignore doesn't exclude models
```

**Issue:** Out of memory
- Reduce model size
- Use smaller batch sizes
- Increase container memory

---

## Next Steps After Deployment

1. **Share URL** with stakeholders
2. **Gather feedback** from users
3. **Monitor usage** and errors
4. **Iterate** based on feedback
5. **Scale** as needed

---

## Support

For deployment issues:
- Check logs first
- Review platform documentation
- Open GitHub issue
- Contact support

---

**Deployment Guide Version:** 1.0
**Last Updated:** 2025-12-03
