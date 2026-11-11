#!/usr/bin/env bash
# Foreground end-to-end check: start the Gradio app, poll it, exercise the
# named API endpoint, then stop the server. Exits non-zero on any failure.
set -u
cd /mnt/d/Thien/Code/mini-gpt
source ~/ai-roadmap/.venv/bin/activate

PORT=7863
LOG=/tmp/mg_server.log
GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=$PORT python app.py >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

up=0
for _ in $(seq 1 30); do
  sleep 2
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" || true)
  if [ "$code" = "200" ]; then up=1; break; fi
done
if [ "$up" != "1" ]; then
  echo "FAIL: server never became ready (last code: $code)"
  tail -20 "$LOG"
  exit 1
fi
echo "PASS: server up on :$PORT"

NAMED=$(curl -s "http://127.0.0.1:$PORT/gradio_api/info" | python -c "import json,sys; print(' '.join(json.load(sys.stdin).get('named_endpoints',{}).keys()))")
echo "named endpoints: $NAMED"
case "$NAMED" in
  *generate*) echo "PASS: /generate endpoint registered" ;;
  *) echo "FAIL: /generate not found"; exit 1 ;;
esac

SUBMIT=$(curl -s -X POST "http://127.0.0.1:$PORT/gradio_api/call/generate" \
  -H 'Content-Type: application/json' \
  -d '{"data": ["ROMEO:", 48, 0.0, 0, 1337, true]}')
echo "submit response: $SUBMIT"
EVENT_ID=$(printf '%s' "$SUBMIT" | sed -n 's/.*"event_id":"\([^"]*\)".*/\1/p')
if [ -z "$EVENT_ID" ]; then
  EVENT_ID=$(printf '%s' "$SUBMIT" | python -c "import json,sys; print(json.load(sys.stdin)['event_id'])")
fi
if [ -z "$EVENT_ID" ]; then
  echo "FAIL: no event_id in submit response"
  tail -20 "$LOG"
  exit 1
fi

sleep 5
RESULT=$(curl -s -N --max-time 60 "http://127.0.0.1:$PORT/gradio_api/call/generate/$EVENT_ID")
echo "--- result stream (first 500 chars) ---"
printf '%s' "$RESULT" | head -c 500
echo
case "$RESULT" in
  *ROMEO*) echo "PASS: generated text contains the prompt" ;;
  *) echo "FAIL: unexpected result"; exit 1 ;;
esac
echo "ALL CHECKS PASSED"
