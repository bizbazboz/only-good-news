import Link from "next/link";
import { fetchApi } from "lib/api";

function sentimentPill(sentiment) {
  if (sentiment === "positive") return <span className="pill positive">positive</span>;
  if (sentiment === "negative") return <span className="pill negative">negative</span>;
  return <span className="pill unsure">unsure</span>;
}

function ReviewTable({ title, rows }) {
  return (
    <section className="table-wrap">
      <h2>{title}</h2>
      {rows?.length ? (
        <table>
          <thead>
            <tr>
              <th>Source</th>
              <th>Headline</th>
              <th>Sentiment</th>
              <th>Keyword Fail</th>
              <th>Flags</th>
              <th>Link</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((item, idx) => (
              <tr key={`${item.link || item.title || "headline"}-${idx}`}>
                <td>{item.source}</td>
                <td>{item.title}</td>
                <td>
                  {sentimentPill(item.sentiment)} {item.label} ({Number(item.confidence || 0).toFixed(3)})
                </td>
                <td>
                  {item.keyword_fail ? "YES" : "NO"}
                  <div className="meta">{item.matched_keywords?.length ? item.matched_keywords.join(", ") : "-"}</div>
                </td>
                <td>
                  {item.flagged ? "FLAGGED" : "OK"}
                  <div className="meta">{item.flag_reasons?.length ? item.flag_reasons.join(", ") : "-"}</div>
                </td>
                <td>
                  {item.link ? (
                    <a href={item.link} target="_blank" rel="noopener noreferrer">
                      Open
                    </a>
                  ) : (
                    "-"
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="panel">No headlines.</p>
      )}
    </section>
  );
}

export default async function AdminPage() {
  let data = null;
  let error = "";
  try {
    data = await fetchApi("/api/admin/review");
  } catch (err) {
    error = err instanceof Error ? err.message : "Failed to load admin data.";
  }

  return (
    <main className="wrap">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">⚙</div>
          <strong>ADMIN REVIEW</strong>
        </div>
        <Link href="/">Back to homepage</Link>
      </header>

      {error ? (
        <p>{error}</p>
      ) : (
        <>
          <section className="summary">
            <div>
              Updated: {data?.timestamp || "-"} | Unsure threshold=
              {data?.thresholds?.unsure_confidence_threshold} | Flag threshold=
              {data?.thresholds?.headline_flag_confidence_threshold}
            </div>
          </section>

          <section className="grid4">
            <article className="panel">
              <span>Positive</span>
              <strong>{data?.counts?.positive ?? 0}</strong>
            </article>
            <article className="panel">
              <span>Negative</span>
              <strong>{data?.counts?.negative ?? 0}</strong>
            </article>
            <article className="panel">
              <span>Unsure</span>
              <strong>{data?.counts?.unsure ?? 0}</strong>
            </article>
            <article className="panel">
              <span>Total</span>
              <strong>{data?.counts?.total ?? 0}</strong>
            </article>
          </section>

          <ReviewTable title="Positive" rows={data?.positive_articles || []} />
          <ReviewTable title="Negative" rows={data?.negative_articles || []} />
          <ReviewTable title="Unsure" rows={data?.unsure_articles || []} />
        </>
      )}
    </main>
  );
}
