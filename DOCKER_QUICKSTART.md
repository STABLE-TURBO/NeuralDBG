# Docker Quick Start Guide

This guide will get you up and running with Neural DSL using Docker in under 5 minutes.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 10GB free disk space

## Quick Start

### 1. Clone Repository (if not already)

```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))" > secret.txt

# Edit .env and set SECRET_KEY from secret.txt
# On Windows: notepad .env
# On Linux/Mac: nano .env
```

Minimum required changes in `.env`:
```bash
SECRET_KEY=<paste-from-secret.txt>
REDIS_PASSWORD=<choose-a-password>
POSTGRES_PASSWORD=<choose-a-password>
```

### 3. Start Services

**For Development:**
```bash
docker-compose -f docker-compose.dev.yml up -d
```

**For Production:**
```bash
docker-compose up -d
```

### 4. Verify Services

```bash
# Check if all containers are running
docker-compose ps

# Check logs
docker-compose logs -f
```

### 5. Access Services

Once services are running, access them at:

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | REST API endpoint |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| Dashboard | http://localhost:8050 | NeuralDbg debugging dashboard |
| No-Code | http://localhost:8051 | Visual model builder |
| Aquarium IDE | http://localhost:8052 | Integrated development environment |
| Flower | http://localhost:5555 | Celery task monitoring |
| Nginx | http://localhost | Reverse proxy (all services) |

## Common Commands

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f worker
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart api
```

### Scale Workers

```bash
# Scale to 5 workers
docker-compose up -d --scale worker=5
```

### Stop Services

```bash
# Stop (containers remain)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes (WARNING: deletes data)
docker-compose down -v
```

### Execute Commands in Containers

```bash
# API shell
docker-compose exec api bash

# Run Python in API container
docker-compose exec api python

# Check database
docker-compose exec postgres psql -U neural -d neural_db

# Check Redis
docker-compose exec redis redis-cli
```

## Building Images

### Build All Services

```bash
# Windows
scripts\build-images.bat

# Linux/Mac
chmod +x scripts/build-images.sh
./scripts/build-images.sh
```

### Build Specific Service

```bash
docker-compose build api
docker-compose build worker
```

## Troubleshooting

### Services Won't Start

```bash
# Check logs
docker-compose logs

# Check if ports are already in use
# On Linux/Mac:
netstat -an | grep 8000
# On Windows:
netstat -an | findstr 8000
```

### Database Connection Errors

```bash
# Wait for database to be ready
docker-compose exec postgres pg_isready -U neural

# Reset database (WARNING: deletes data)
docker-compose down -v
docker-compose up -d
```

### Redis Connection Errors

```bash
# Test Redis
docker-compose exec redis redis-cli ping

# Check Redis logs
docker-compose logs redis
```

### Out of Memory

```bash
# Check resource usage
docker stats

# Increase Docker memory limit in Docker Desktop settings
# Recommended: 8GB for all services
```

### Permission Errors

```bash
# Fix volume permissions
docker-compose exec --user root api chown -R neural:neural /app/data
```

## Production Checklist

Before deploying to production:

- [ ] Change `SECRET_KEY` in `.env`
- [ ] Change `REDIS_PASSWORD` in `.env`
- [ ] Change `POSTGRES_PASSWORD` in `.env`
- [ ] Change `FLOWER_USER` and `FLOWER_PASSWORD` in `.env`
- [ ] Set `DEBUG=false` in `.env`
- [ ] Configure `CORS_ORIGINS` in `.env`
- [ ] Set up SSL certificates (place in `./ssl/`)
- [ ] Configure proper `API_WORKERS` count
- [ ] Set up log rotation
- [ ] Configure backups for volumes
- [ ] Set up monitoring

## Next Steps

- Read the [full deployment guide](DEPLOYMENT.md)
- Explore the [API documentation](http://localhost:8000/docs)
- Try the [no-code interface](http://localhost:8051)
- Check out [example models](examples/)

## Support

- Documentation: https://github.com/Lemniscate-world/Neural
- Issues: https://github.com/Lemniscate-world/Neural/issues
- Email: Lemniscate_zero@proton.me
