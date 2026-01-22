# Base image with Node
FROM node:20

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy all source code
COPY . .

# Expose React dev server port
EXPOSE 3000

# Start React development server
CMD ["npm", "start"]
