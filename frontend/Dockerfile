# Build stage
FROM node:20-alpine AS build

WORKDIR /app

# Copy dependency files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production serve stage (Nginx)
FROM nginx:alpine

# Copy custom nginx configuration for routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy build files to Nginx directory
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
