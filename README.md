# Only Good News

Only Good News is an in progress service that use sentiment analysis of a variety of news sources to only present positive news.

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
python -m http.server 8080 --directory frontend
python api.py --port 8000