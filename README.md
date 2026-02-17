# Only Good News

Only Good News is split into:

- `backend`: FastAPI backend exposing only `GET /news`
- `frontend`: Next.js SSR frontend

## Environment Config

Backend (`backend/.env`, copy from `backend/.env.example`):

```bash
cp backend/.env.example backend/.env
```

- `ALLOWED_ORIGINS`: CORS allowlist. Use comma-separated values or JSON array.

Frontend (`frontend/.env.local`, copy from `frontend/.env.example`):

```bash
cp frontend/.env.example frontend/.env.local
```

- `API_BASE_URL`: Server-side API base URL for Next.js SSR fetches.
- `NEXT_PUBLIC_API_BASE_URL`: Client-visible API base URL (kept in sync with backend URL).

Model/filter/feed settings remain in `backend/config/config.json`:

- `SENTIMENT_MODEL`
- `MIN_CONFIDENCE`
- `UNSURE_CONFIDENCE_THRESHOLD`
- `MIN_TITLE_WORDS`
- `UPDATE_CHECK_INTERVAL_MINUTES`
- feed list and banned keywords

## Run Locally

```bash
./dev.sh
```

- Frontend runs at `http://localhost:3000`
- Backend runs at `http://localhost:8000`
- API endpoint: `GET /news`

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Roadmap
- Integrate more APIs
   - https://www.cnbc.com/id/100727362/device/rss/rss.html
   - https://abcnews.go.com/abcnews/internationalheadlines
   - https://www.buzzfeed.com/in/world.xml
   - https://www.nytimes.com/svc/collections/v1/publish/
   - https://www.nytimes.com/section/world/rss.xml
- Develop frontend
- Public Deployment
- Custom Sentiment Analysis Model
