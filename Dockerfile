# Use a base image with both Python and Node.js installed
FROM python:3.8-slim AS backend

# Set working directory for backend
WORKDIR /app/backend

# Copy requirements file
COPY backend/requirements.txt .

# Install backend dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend .

# Expose port for Flask app
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]

# Use a base image with Node.js
FROM node:14 AS frontend

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend source code
COPY frontend .

# Install frontend dependencies
RUN npm install

# Expose port for frontend (if needed)
# EXPOSE <frontend_port>

# Build frontend (replace 'npm run build' with your build command if needed)
RUN npm run build

# Use nginx base image to serve frontend
FROM nginx:alpine AS production

# Copy built frontend files from the frontend build stage
COPY --from=frontend /app/frontend/build /usr/share/nginx/html

# Expose port for Nginx
EXPOSE 80

# Start Nginx to serve the frontend
CMD ["nginx", "-g", "daemon off;"]
