# Neural Aquarium - Setup Guide

Complete setup instructions for the Shape Propagation Panel.

## Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- Neural DSL installed

## Quick Start

### 1. Backend Setup

```bash
cd neural/aquarium

# Install Python dependencies
pip install -r requirements.txt

# Start the API server
python api/shape_api.py
```

The API will be available at `http://localhost:5002`

### 2. Frontend Setup

```bash
cd neural/aquarium

# Install Node.js dependencies
npm install

# Start the development server
npm start
```

The UI will be available at `http://localhost:3000`

## Detailed Setup

### Python Backend

The backend provides REST API endpoints for shape propagation.

**Install dependencies:**

```bash
pip install flask flask-cors numpy plotly
```

**Run the server:**

```bash
# Development mode (with auto-reload)
python api/shape_api.py

# Or use environment variable for port
export FLASK_PORT=5002
python api/shape_api.py
```

**API Endpoints:**

- `GET /api/shape-propagation` - Get current shape history
- `POST /api/shape-propagation/propagate` - Propagate shapes through a model
- `POST /api/shape-propagation/reset` - Reset propagator
- `GET /api/shape-propagation/layer/<id>` - Get layer details

### React Frontend

The frontend provides interactive visualizations.

**Install dependencies:**

```bash
npm install
```

**Development mode:**

```bash
npm start
```

This will:
- Start the dev server on port 3000
- Auto-reload on file changes
- Enable source maps for debugging

**Production build:**

```bash
npm run build
```

This creates an optimized build in the `build/` directory.

## Configuration

### Backend Configuration

Edit `api/shape_api.py`:

```python
# Change port
def run_server(host='0.0.0.0', port=5002, debug=False):
    app.run(host=host, port=port, debug=debug)
```

### Frontend Configuration

Edit `package.json` to add proxy:

```json
{
  "proxy": "http://localhost:5002"
}
```

Or use environment variables:

```bash
# .env.local
REACT_APP_API_URL=http://localhost:5002
```

## Testing

### Backend Tests

```bash
cd neural/aquarium
python -m pytest tests/ -v
```

### Frontend Tests

```bash
cd neural/aquarium
npm test
```

## Troubleshooting

### Backend Issues

**Port already in use:**

```bash
# Find process using port 5002
lsof -i :5002

# Kill the process
kill -9 <PID>
```

**CORS errors:**

Ensure `flask-cors` is installed and configured:

```python
from flask_cors import CORS
CORS(app)
```

### Frontend Issues

**API connection refused:**

- Check backend is running on port 5002
- Verify API URL in frontend code
- Check browser console for errors

**Build errors:**

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**D3.js or Plotly not rendering:**

- Check browser console for errors
- Ensure D3 and Plotly are installed
- Verify data format is correct

## Integration with Neural Dashboard

To integrate with the existing Neural Dashboard:

```python
from neural.aquarium.api import initialize_propagator, app as aquarium_app

# In your dashboard initialization
initialize_propagator(model_data)

# Or mount the API
from flask import Flask
main_app = Flask(__name__)
main_app.register_blueprint(aquarium_app, url_prefix='/aquarium')
```

## Development Workflow

1. **Start backend** in one terminal:
   ```bash
   cd neural/aquarium
   python api/shape_api.py
   ```

2. **Start frontend** in another terminal:
   ```bash
   cd neural/aquarium
   npm start
   ```

3. **Make changes** - both will auto-reload

4. **Test your changes**:
   ```bash
   # Backend
   pytest tests/

   # Frontend
   npm test
   ```

## Production Deployment

### Backend

Use a production WSGI server:

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5002 "neural.aquarium.api.shape_api:app"
```

### Frontend

Build and serve:

```bash
npm run build

# Serve with any static file server
npx serve -s build -l 3000
```

Or integrate with your existing web server (Nginx, Apache, etc.)

## Environment Variables

**Backend:**

- `FLASK_PORT` - Server port (default: 5002)
- `FLASK_ENV` - Environment (development/production)

**Frontend:**

- `REACT_APP_API_URL` - Backend API URL
- `PORT` - Dev server port (default: 3000)

## Additional Resources

- [React Documentation](https://react.dev/)
- [D3.js Documentation](https://d3js.org/)
- [Plotly.js Documentation](https://plotly.com/javascript/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Neural DSL Documentation](../../docs/)
