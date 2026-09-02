#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

if [ -z "$PYTHON" ]; then
    echo "Python not found. Install Python 3.10+ and try again."
    exit 1
fi

HOST=$("$PYTHON" -c "import json;c=json.load(open('config.json'));print(c.get('host','127.0.0.1'))" 2>/dev/null || echo 127.0.0.1)
PORT=$("$PYTHON" -c "import json;c=json.load(open('config.json'));print(c.get('port',8000))" 2>/dev/null || echo 8000)
URL="http://${HOST}:${PORT}"

echo "Starting Soybrary on ${URL}..."
echo ""

# No --reload: the reloader would watch every file in data/, which holds one
# file per scraped post.
"$PYTHON" server.py &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

echo "Waiting for server to be ready..."
until curl -s "$URL" > /dev/null 2>&1; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server exited before becoming ready."
        exit 1
    fi
    sleep 0.2
done

echo "Server is ready."

# Open the URL in the default browser
if command -v xdg-open &> /dev/null; then
    xdg-open "$URL"
elif command -v open &> /dev/null; then
    open "$URL"
fi

# Keep script running (so server stays alive)
wait $SERVER_PID
