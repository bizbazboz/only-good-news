import Link from "next/link";
import { fetchApi, normalizeThumbnailUrl } from "lib/api";

function readMinutes(article) {
  const text = `${article?.title || ""} ${article?.description || ""}`.trim();
  const words = text.split(/\s+/).filter(Boolean).length;
  return Math.max(3, Math.min(9, Math.ceil(words / 35)));
}

function imageForArticle(article, fallbackSeed) {
  const thumbnail = normalizeThumbnailUrl(article?.thumbnail);
  if (thumbnail) return thumbnail;
  const seed = encodeURIComponent(`${article?.title || "story"}-${fallbackSeed}`);
  return `https://picsum.photos/seed/${seed}/1200/760`;
}

export default async function HomePage() {
  let news = null;
  let error = "";
  try {
    news = await fetchApi("/api/news");
  } catch (err) {
    error = err instanceof Error ? err.message : "Could not load stories.";
  }

  const merged = news
    ? [
        ...(news.positive_articles || []),
        ...(news.neutral_articles || []),
        ...(news.negative_articles || [])
      ]
    : [];

  const featured = merged[0];
  const cards = merged.slice(1, 15);
  const total = Number(news?.total || 0);
  const positives = Number(news?.positive_articles?.length || 0);

  return (
    <main className="wrap">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">☀</div>
          <strong>ONLY GOOD NEWS</strong>
        </div>
        <Link href="/admin">Admin</Link>
      </header>

      <section
        className="hero"
        style={{ backgroundImage: `url('${featured ? imageForArticle(featured, 0) : ""}')` }}
      >
        <div className="hero-content">
          <span className="eyebrow">Good News of the Day</span>
          <h1>{featured?.title || "No stories available right now"}</h1>
          <p>{featured?.description || "Please check back soon for fresh updates."}</p>
          {featured?.link ? (
            <a className="hero-link" href={featured.link} target="_blank" rel="noopener noreferrer">
              Read Full Story
            </a>
          ) : null}
        </div>
      </section>

      <section className="summary">
        <strong>Smart News Analysis</strong>
        <div>{`${positives} uplifting stories surfaced from ${total} analyzed headlines.`}</div>
        <div>{news?.cached ? "Using cached analysis" : "Freshly updated"}</div>
      </section>

      {error ? <p>{error}</p> : null}

      <section className="section-head">
        <h2>Trending Positivity</h2>
      </section>

      <section className="cards">
        {cards.length ? (
          cards.map((article, idx) => (
            <article key={`${article.link || article.title || "story"}-${idx}`} className="card">
              <div
                className="card-media"
                style={{ backgroundImage: `url('${imageForArticle(article, idx + 1)}')` }}
              >
                <span className="chip">Verified Positive</span>
              </div>
              <h3>{article.title || "Positive story"}</h3>
              <p>{article.description || "Read the full article for details."}</p>
              <div className="meta">
                {article.source || "Global Desk"} · {article.pub_date || "Fresh update"} ·{" "}
                {readMinutes(article)} min read
              </div>
              {article.link ? (
                <a className="link" href={article.link} target="_blank" rel="noopener noreferrer">
                  Read original story
                </a>
              ) : null}
            </article>
          ))
        ) : (
          <p>No stories are available right now.</p>
        )}
      </section>
    </main>
  );
}
