#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

echo "Starting Soybrary..."
echo ""

# Start the server in the background
.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload &
SERVER_PID=$!

# Wait for the server to be ready
echo "Waiting for server to be ready..."
until curl -s http://localhost:8000 > /dev/null 2>&1; do
    sleep 0.2
done

echo "Server is ready."

# Open the URL in the default browser
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8000
elif command -v open &> /dev/null; then
    open http://localhost:8000
fi

# Keep script running (so server stays alive)
wait $SERVER_PID
