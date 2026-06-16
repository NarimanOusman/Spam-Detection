FROM python:3.11-slim

# ── Stage 2: Set working directory ───────────────────────────────────────────
# All subsequent commands will run from /app inside the container.
WORKDIR /app

# ── Stage 3: Install Python dependencies ─────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 4: Copy project files into the container ───────────────────────────
# Copy the entire project directory into the container's /app directory.
COPY . .

# Train the model during image build so the .pkl files are generated and packaged
RUN python train.py

# ── Stage 5: Expose the application port ─────────────────────────────────────
# Inform Docker that the container listens on port 5000 at runtime.
EXPOSE 5000

# ── Stage 6: Start the Flask application ─────────────────────────────────────
# Use Gunicorn as the production-grade WSGI server (better than Flask's built-in server).
# - -w 2      : Use 2 worker processes for handling requests.
# - -b 0.0.0.0:5000 : Listen on all network interfaces, port 5000.
# - app:app   : The module is "app.py" and the Flask object is named "app".
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
