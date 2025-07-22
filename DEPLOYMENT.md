# 🚀 Deployment Guide - Hebrew Content Intelligence Service

## Coolify Deployment

This guide will help you deploy the Hebrew Content Intelligence Service on Coolify via GitHub.

### Prerequisites

- Coolify instance running and accessible
- GitHub repository with this code
- (Optional) DataForSEO API key for real search data

### 📋 Deployment Steps

#### 1. GitHub Repository Setup

1. Push this code to your GitHub repository
2. Ensure all files are committed including:
   - `Dockerfile`
   - `docker-compose.yml`
   - `.coolify` (Coolify configuration)
   - `requirements.txt`
   - All application code

#### 2. Coolify Configuration

1. **Create New Project** in Coolify
2. **Connect GitHub Repository**
   - Select your repository
   - Choose the main/master branch
   - Set build context to root directory

3. **Environment Variables** (Set in Coolify):
   ```bash
   SERVICE_NAME=Hebrew Content Intelligence
   VERSION=1.0.0
   DEBUG=false
   LOG_LEVEL=INFO
   HOST=0.0.0.0
   PORT=5000
   WORKERS=4
   HEBSPACY_MODEL=he
   HEBSPACY_CACHE_DIR=/app/models/model_cache
   CACHE_TYPE=redis
   REDIS_URL=redis://redis:6379/0
   CACHE_TTL=3600
   DATAFORSEO_API_URL=https://api.dataforseo.com/v3
   DATAFORSEO_API_KEY=your_actual_api_key_here
   MAX_CONTENT_LENGTH=50000
   REQUEST_TIMEOUT=30
   MAX_CONCURRENT_REQUESTS=100
   ```

4. **Secrets** (Set in Coolify Secrets):
   - `DATAFORSEO_API_KEY`: Your actual DataForSEO API key
   - (Optional) `JWT_SECRET_KEY`: For future authentication
   - (Optional) `SENTRY_DSN`: For error monitoring

#### 3. Service Configuration

1. **Main Application Service**:
   - **Build Command**: `docker build -t hebrew-intel .`
   - **Start Command**: `python app.py`
   - **Port**: `5000`
   - **Health Check**: `/health/live`

2. **Redis Service**:
   - **Image**: `redis:7-alpine`
   - **Port**: `6379`
   - **Persistent Volume**: `/data`

#### 4. Domain & SSL

1. **Custom Domain**: Set your domain in Coolify
2. **SSL Certificate**: Coolify will auto-generate Let's Encrypt certificate
3. **Port Mapping**: Ensure port 5000 is mapped correctly

### 🔧 Advanced Configuration

#### Resource Limits
```yaml
# In Coolify resource settings
CPU: 1-2 cores
Memory: 2-4 GB (HebSpacy models require memory)
Storage: 5-10 GB (for models and cache)
```

#### Scaling
- **Horizontal**: Multiple app instances behind load balancer
- **Vertical**: Increase CPU/Memory for single instance
- **Redis**: Single instance with persistence

#### Monitoring
- **Health Checks**: Built-in endpoints at `/health/*`
- **Logs**: Available in Coolify logs section
- **Metrics**: Optional Prometheus integration

### 📊 Post-Deployment Verification

#### 1. Health Check
```bash
curl https://your-domain.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "Hebrew Content Intelligence",
  "version": "1.0.0",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### 2. API Test
```bash
curl -X POST https://your-domain.com/analysis/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "זהו טקסט בדיקה בעברית",
    "options": {
      "include_keywords": true,
      "include_roots": true
    }
  }'
```

#### 3. Performance Test
```bash
# Run the benchmark script against your deployed service
python scripts/benchmark.py --url https://your-domain.com
```

### 🐛 Troubleshooting

#### Common Issues

1. **Model Download Failure**:
   - Check internet connectivity in container
   - Increase startup timeout in Coolify
   - Monitor logs during first deployment

2. **Redis Connection Issues**:
   - Verify Redis service is running
   - Check network connectivity between services
   - Validate Redis URL environment variable

3. **Memory Issues**:
   - HebSpacy models require ~1-2GB RAM
   - Increase memory allocation in Coolify
   - Monitor memory usage in logs

4. **Slow Startup**:
   - First deployment takes longer (model download)
   - Subsequent deployments use cached models
   - Increase health check start period

#### Log Analysis
```bash
# Check application logs
docker logs hebrew-intel-app

# Check Redis logs
docker logs hebrew-intel-redis

# Monitor real-time logs in Coolify dashboard
```

### 🔄 Updates & Maintenance

#### Code Updates
1. Push changes to GitHub
2. Coolify auto-deploys on push (if configured)
3. Or manually trigger deployment in Coolify

#### Model Updates
```bash
# SSH into container and update models
python -c "import hebspacy; hebspacy.download('he', force=True)"
```

#### Database Maintenance
```bash
# Redis maintenance (if needed)
redis-cli FLUSHALL  # Clear cache
redis-cli BGSAVE   # Backup data
```

### 📈 Scaling Recommendations

#### Production Load
- **Small**: 1 CPU, 2GB RAM, handles ~100 req/min
- **Medium**: 2 CPU, 4GB RAM, handles ~500 req/min  
- **Large**: 4 CPU, 8GB RAM, handles ~1000+ req/min

#### High Availability
- Multiple app instances with load balancer
- Redis with persistence and backup
- Health monitoring and auto-restart

### 🔐 Security Considerations

1. **Environment Variables**: Use Coolify secrets for sensitive data
2. **API Keys**: Rotate DataForSEO keys regularly
3. **Network**: Restrict Redis access to app only
4. **Updates**: Keep base images and dependencies updated

### 📞 Support

If you encounter issues:
1. Check Coolify logs first
2. Review this deployment guide
3. Test locally with `docker-compose up`
4. Check GitHub issues for known problems

---

**Ready for Production!** 🎉

Your Hebrew Content Intelligence Service is now deployed and ready to analyze Hebrew content at scale!
