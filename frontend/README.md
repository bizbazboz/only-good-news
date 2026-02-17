# Frontend (Next.js)

This frontend is deployed separately from the FastAPI backend.

## Run locally

Use root `dev.sh` to run backend + frontend together:

- `python -m pip install -r requirements.txt`
- `./dev.sh`

Or run frontend only:

- `npm install`
- `npm run dev`

Default frontend URL: `http://localhost:3000`  
Default backend API URL: `http://localhost:8000`

## Environment variables

- `API_BASE_URL`: Server-side API URL used by Next.js SSR.
- `NEXT_PUBLIC_API_BASE_URL`: Optional client fallback.
