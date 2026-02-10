# Remote Capture Setup & Maintenance Guide

iPad photo capture system that relays photos to the local robot machine via Heroku.

## Architecture

```
iPad Camera → Heroku (relay) → Local Machine → OpenAI → Contour Extraction → GCode → Robot Drawing
                ↑                                                                         ↓
            Status updates ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

---

## Deployment Setup

### Heroku

1. Create the Heroku app: `heroku create your-app-name`
2. Set required environment variables:
   - `heroku config:set CAPTURE_PASSWORD=<your-password>`
   - `heroku config:set SECRET_KEY=<random-secret-string>`
   - Optionally: `heroku config:set ROBOT_TOKEN=<token>` (if you want to authenticate the robot connection)
3. Deploy: `git push heroku main`
4. Verify it's running: `heroku logs --tail`

### Local Machine (Robot Side)

5. Install the new dependency: `pip install "python-socketio[client]>=5.0.0"`
6. Set `OPENAI_API_KEY` in your `.env` file (already done if you use the system today)
7. Test with mock AI first:
   ```
   python -m src.web.remote_client --server https://your-app.herokuapp.com --mock
   ```
8. Test with real AI:
   ```
   python -m src.web.remote_client --server https://your-app.herokuapp.com
   ```

### iPad Testing

9. Open `https://your-app.herokuapp.com` on the iPad
10. Enter the password — confirm session persists (no re-auth on refresh)
11. Grant camera permission when prompted
12. Test the full flow: take photo, pick style, submit, watch status updates through to "Complete!"

---

## Robustness Considerations

### Network Resilience

13. **Test reconnection** — Kill the local client, restart it, confirm it reconnects to Heroku and can receive jobs again
14. **Test with poor network** — The `python-socketio` client has auto-reconnect built in (configured with exponential backoff in `src/web/remote_client.py`), but test it on real Wi-Fi
15. **Consider a process manager** — Use `systemd`, `supervisor`, or `pm2` to auto-restart the local client if it crashes:
    ```ini
    # /etc/supervisor/conf.d/portrait-client.conf
    [program:portrait-client]
    command=python -m src.web.remote_client --server https://your-app.herokuapp.com
    directory=/path/to/aicentre-art-orchestration
    autorestart=true
    ```

### Heroku-Specific

16. **Heroku free tier dynos sleep after 30 min** — If using free/eco tier, the first request after sleep takes ~10s to wake. Consider a paid dyno for events, or use a keep-alive ping
17. **WebSocket support** — Heroku supports WebSockets out of the box, but confirm your app uses `wss://` (HTTPS) in production
18. **Session storage** — Sessions are stored in signed cookies (no server-side state), so they survive dyno restarts. No Redis/database needed

### Configuration Sync

19. **Style definitions are duplicated** — The `ART_STYLES` dict in `src/web/capture_app.py` and the styles in `config/settings.yaml` must stay in sync. If you add/remove/rename a style in `settings.yaml`, update `ART_STYLES` in `capture_app.py` too
20. **Consider loading styles from a shared source** — Long-term, you could have the relay server read style names/descriptions from a JSON file or have the robot client announce its available styles on connect

### Security

21. **Change the default password** — The fallback password in code is `portrait` — always set `CAPTURE_PASSWORD` env var
22. **Change the SECRET_KEY** — The fallback is `dev-secret-change-me` — always set `SECRET_KEY` env var to a random string
23. **HTTPS is automatic** on Heroku — no extra setup needed
24. **Rate limiting** — Currently not implemented. If this is public-facing at a busy event, consider adding `flask-limiter` to prevent photo spam

### Error Recovery

25. **Stuck "busy" state** — If the local client crashes mid-job, the Heroku server will detect the robot disconnect and reset the job state automatically (handled in `src/web/capture_app.py` robot disconnect handler)
26. **OpenAI API failures** — The existing retry logic (3 attempts, 5s delay) in `src/ai/openai_client.py` applies. If all retries fail, an error status is sent to the iPad
27. **Serial/hardware failures** — If the DexArm or MyCobot fail, the error propagates back through the status callback chain to the iPad

### Monitoring

28. **Heroku logs** — `heroku logs --tail` shows relay server activity
29. **Local client logs** — Printed to stdout with timestamps
30. **Health check** — The `/api/status` endpoint on Heroku returns the current state — you could poll this from a monitoring tool

---

## Future Improvements (Optional)

- **Job queue** — Allow queuing multiple photos instead of rejecting when busy
- **Photo gallery** — Show completed drawings on the web interface
- **QR code** — Display a QR code on a screen near the robot for easy iPad access
- **Multiple robot support** — Allow multiple local machines to connect (requires job routing)
- **Job timeout** — Add a max processing time (e.g. 10 min) in the remote client to auto-reset if a job hangs

---

## Key Files

| File | Purpose |
|------|---------|
| `src/web/capture_app.py` | Heroku relay server (Flask + SocketIO) |
| `src/web/remote_client.py` | Local machine client (connects to Heroku) |
| `src/web/templates/capture.html` | iPad camera + style picker + status UI |
| `src/web/templates/login.html` | Password gate |
| `src/main.py` | `run_pipeline_from_image()` method for remote submissions |
| `Procfile` | Heroku process definition |
| `runtime.txt` | Python version for Heroku |

## Environment Variables

| Variable | Where | Required | Description |
|----------|-------|----------|-------------|
| `CAPTURE_PASSWORD` | Heroku | Yes | Password for iPad access |
| `SECRET_KEY` | Heroku | Yes | Flask session signing key |
| `ROBOT_TOKEN` | Heroku + Local | No | Optional auth token for robot connection |
| `OPENAI_API_KEY` | Local machine | Yes | OpenAI API key (unless using `--mock`) |
