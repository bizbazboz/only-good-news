# Only Good News

Only Good News is an in progress service that use sentiment analysis of a variety of news sources to only present positive news.

## Configuration

Set the sentiment model in `config.json`:

```json
{
  "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
  "unsure_confidence_threshold": 0.75,
  "headline_flag_confidence_threshold": 0.85
}
```

Temporary admin review page:

- `http://localhost:3000/admin`
- Shows positive, negative, and unsure headlines
- Includes keyword-fail status and flag reasons

Frontend rendering:

- Frontend is a standalone Next.js app in `frontend/`
- FastAPI backend serves API endpoints at `/api/*`
- Add your frontend domain(s) to `allowed_origins` in `config.json` for CORS (for example your Appwrite site URL)

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

(DEV COMMANDS)
python -m pip install -r requirements.txt
./dev.sh
