import { fetchApi, normalizeThumbnailUrl } from "../lib/api";

function readMinutes(article) {
  const text = `${article?.title || ""} ${article?.description || ""}`.trim();
  const words = text.split(/\s+/).filter(Boolean).length;
  return Math.max(3, Math.min(9, Math.ceil(words / 35)));
}

function formatDateTime(value) {
  const dt = new Date(value || "");
  if (Number.isNaN(dt.getTime())) return "Fresh update";
  const dd = String(dt.getDate()).padStart(2, "0");
  const mm = String(dt.getMonth() + 1).padStart(2, "0");
  const yyyy = dt.getFullYear();
  const hh = String(dt.getHours()).padStart(2, "0");
  const min = String(dt.getMinutes()).padStart(2, "0");
  return `${dd}/${mm}/${yyyy} - ${hh}:${min}`;
}

export default async function HomePage() {
  let news = null;
  let error = "";
  try {
    news = await fetchApi("/news");
  } catch (err) {
    error = err instanceof Error ? err.message : "Could not load stories.";
  }

  const merged = news
    ? [...(news.positive_articles || [])].filter(
        (article) => Number(article?.confidence || 0) >= Number(news?.applied_min_confidence || 0)
      ).sort((a, b) => {
        const da = Date.parse(a?.pub_date || "") || 0;
        const db = Date.parse(b?.pub_date || "") || 0;
        return db - da;
      })
    : [];

  const featured = merged[0];
  const cards = merged.slice(1);
  const featuredThumbnail = normalizeThumbnailUrl(featured?.thumbnail);

  return (
    <main className="wrap">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">☀</div>
          <strong>ONLY GOOD NEWS</strong>
        </div>
      </header>

      <section
        className="hero"
        style={{ backgroundImage: featuredThumbnail ? `url('${featuredThumbnail}')` : "none" }}
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

      {error ? <p>{error}</p> : null}

      <section className="section-head">
        <h2>Trending Positivity</h2>
      </section>

      <section className="cards">
        {cards.length ? (
          cards.map((article, idx) => {
            const thumbnail = normalizeThumbnailUrl(article?.thumbnail);
            return (
            <article key={`${article.link || article.title || "story"}-${idx}`} className="card">
              {article.link ? (
                <a
                  className="link"
                  href={article.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ display: "block", textDecoration: "none", color: "inherit" }}
                >
                  <div
                    className="card-media"
                    style={{ backgroundImage: thumbnail ? `url('${thumbnail}')` : "none" }}
                  />
                  <h3>{article.title || "Positive story"}</h3>
                  <p>{article.description || "Read the full article for details."}</p>
                  <div className="meta">
                    {article.source || "Global Desk"} · {formatDateTime(article.pub_date)} ·{" "}
                    {readMinutes(article)} min read
                  </div>
                </a>
              ) : (
                <>
                  <div
                    className="card-media"
                    style={{ backgroundImage: thumbnail ? `url('${thumbnail}')` : "none" }}
                  />
                  <h3>{article.title || "Positive story"}</h3>
                  <p>{article.description || "Read the full article for details."}</p>
                  <div className="meta">
                    {article.source || "Global Desk"} · {formatDateTime(article.pub_date)} ·{" "}
                    {readMinutes(article)} min read
                  </div>
                </>
              )}
            </article>
          )})
        ) : (
          <p>No stories are available right now.</p>
        )}
      </section>
    </main>
  );
}
