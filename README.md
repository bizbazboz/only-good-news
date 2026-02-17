# Only Good News

Only Good News is split into:

- `backend`: FastAPI backend exposing only `GET /news`
  - Stores filtered API articles in `backend/data/news_archive.db`
  - Archives all fetched raw posts in `backend/data/archive.db` for future fine-tuning
- `frontend`: Next.js SSR frontend

## Configuration

You can run with direct environment variables (recommended for Docker/CI), or use local env files.

Backend variables:

- `ALLOWED_ORIGINS`: CORS allowlist (comma-separated values or JSON array)

Frontend variables:

- `API_BASE_URL`: Server-side API base URL for Next.js SSR fetches
- `NEXT_PUBLIC_API_BASE_URL`: Client-visible API base URL

Model/filter/feed settings remain in `backend/config/config.json`:

- `SENTIMENT_MODEL`
- `MIN_CONFIDENCE`
- `UNSURE_CONFIDENCE_THRESHOLD`
- `MIN_TITLE_WORDS`
- `UPDATE_CHECK_INTERVAL_MINUTES`
- feed list and `banned_keywords_file` (text file, one keyword per line)
- `archive_enabled` (`true`/`false`) to turn raw-post archiving on or off
- `archive_db_path` for archive DB location (default: `data/archive.db`)

## Run Locally (Non-Docker)

```bash
ALLOWED_ORIGINS=http://localhost:3000 \
python3 backend/src/main.py --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
cd frontend
API_BASE_URL=http://localhost:8000 \
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 \
npm install && npm run build && npm run start
```

Or use the helper script:

```bash
./dev.sh
```

## Run With Docker

From repo root:

```bash
ALLOWED_ORIGINS=http://localhost:3000 \
API_BASE_URL=http://backend:8000 \
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 \
docker compose up --build
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000/news`
- Backend SQLite data persists in Docker volume `backend_data`

## Deployment

See `DEPLOYMENT.md` for platform-agnostic deployment steps, including Docker and env-var-only configuration.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International license (`CC BY-NC 4.0`).

- Human-readable summary: https://creativecommons.org/licenses/by-nc/4.0/
- Full legal code: https://creativecommons.org/licenses/by-nc/4.0/legalcode
