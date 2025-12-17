# Docker Setup for TikTok Fake News Detection API

This guide explains how to set up and run the TikTok Fake News Detection API using Docker and Docker Compose.

## Prerequisites

- Docker and Docker Compose installed on your system
- At least 4GB of RAM available for Docker
- Supabase project URL and API key

## Quick Start

1. **Clone and navigate to the project:**
   ```bash
   git clone <your-repo-url>
   cd vn_fake_news_tiktok/backend
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual configuration
   ```

3. **Build and start the services:**
   ```bash
   docker-compose up --build -d
   ```

4. **Check the services:**
   ```bash
   docker-compose ps
   ```

5. **Access the API:**
   - Main API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - Via Nginx: http://localhost (if nginx is enabled)

## Services

The Docker Compose setup includes:

- **api**: Main FastAPI application (port 8000)
- **worker**: News crawler worker that runs continuously
- **redis**: Redis cache for performance (port 6379)
- **nginx**: Reverse proxy (ports 80, 443)

## Configuration

### Required Environment Variables

Edit the `.env` file with your configuration:

```bash
# Database Configuration (Required)
SUPABASE_URL=https://ynkrpoozrpydjliuoxkw.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlua3Jwb296cnB5ZGpsaXVveGt3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ0MTA0NDQsImV4cCI6MjA3OTk4NjQ0NH0.76YAmB0Ypi0hHLOYGtVL5AMLlAHWgU55khYt0wUlpX8

# Model Configuration
MODEL_PATH=./models/han_rag_model.onnx
```

### Optional Configuration

The `.env.example` file includes all available configuration options. Key optional settings:

- `REDIS_URL`: Redis connection string
- `RAG_THRESHOLD`: Similarity threshold for RAG verification
- `CRAWL_INTERVAL_HOURS`: How often the worker crawls news
- `MAX_VIDEO_DURATION`: Maximum video length to process (seconds)

## Volume Mounts

The following directories are mounted as volumes:

- `./models`: For storing ML models
- `./temp`: For temporary media files
- `./data`: For persistent data storage

## Development

### Running in Development Mode

To run with auto-reload:

```bash
# Edit docker-compose.yml and add to api service:
# environment:
#   - RELOAD=true

docker-compose up --build
```

### Viewing Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs api
docker-compose logs worker
docker-compose logs redis

# Follow logs in real-time
docker-compose logs -f api
```

### Executing Commands in Containers

```bash
# Access the API container
docker-compose exec api bash

# Access the worker container
docker-compose exec worker bash

# Run Python commands
docker-compose exec api python -c "import torch; print(torch.__version__)"
```

## Production Deployment

### Security Considerations

1. **Never commit `.env` files** to version control
2. **Use strong Supabase keys** with appropriate permissions
3. **Enable HTTPS** in production by configuring SSL certificates
4. **Set `CORS_ORIGINS`** to your specific domains instead of `*`

### Scaling

To scale the API service:

```bash
# Scale to 3 API instances
docker-compose up --scale api=3 -d
```

### Monitoring

The application includes health checks:

```bash
# Check health status
curl http://localhost:8000/health

# Docker health status
docker-compose ps
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8000, 6379, and 80 are not in use
2. **Memory issues**: Increase Docker memory allocation to at least 4GB
3. **Model loading**: Ensure model files are in the `./models` directory
4. **Permission issues**: Check file permissions for mounted volumes

### Debug Steps

1. **Check container status:**
   ```bash
   docker-compose ps
   ```

2. **View logs for errors:**
   ```bash
   docker-compose logs api
   ```

3. **Test database connection:**
   ```bash
   docker-compose exec api python -c "
   from services.supabase_client import SupabaseService
   db = SupabaseService()
   print('Database connected successfully')
   "
   ```

4. **Check model files:**
   ```bash
   docker-compose exec api ls -la ./models/
   ```

### Cleanup

To stop and remove all containers, networks, and volumes:

```bash
docker-compose down -v
docker system prune -f
```

## API Usage

Once running, you can:

1. **View API documentation:** http://localhost:8000/docs
2. **Test prediction endpoint:**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "video_id": "test123",
       "video_url": "https://example.com/video.mp4",
       "caption": "Test video caption"
     }'
   ```

## Support

For issues related to:
- **Docker setup**: Check this guide and Docker documentation
- **Application functionality**: Check the main README.md
- **Configuration**: Review the .env.example file