import { FormEvent, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ClipboardCheck,
  Database,
  History,
  Play,
  RefreshCw,
  Search,
  Send,
  Settings as SettingsIcon,
  Trash2,
  Upload
} from "lucide-react";

type Tab = "ask" | "corpus" | "runs" | "eval" | "settings";

type Doc = {
  doc_id: string;
  file_sha?: string;
  source: string;
  basename: string;
  chunks: number;
  strategy?: string;
  embedding_model?: string;
  chunking_version?: string;
};

type ManifestDoc = {
  doc_id: string;
  parse_report?: { warnings?: string[] } | null;
};

type RunRow = {
  id: number;
  timestamp: string;
  question: string;
  verifier?: { score?: number; verdict?: string };
  citation_validation?: { citation_valid?: boolean };
  iterations?: number;
  latency_ms?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
};

type QueryResponse = {
  run_id?: number;
  answer: string;
  verifier?: { score?: number; verdict?: string; issues?: string[] };
  citation_validation?: {
    citation_valid: boolean;
    citation_count: number;
    citation_errors: string[];
    citations: string[];
  };
  chunks?: Array<{ id: string; citation?: string; source?: string; text?: string }>;
  trace?: unknown[];
  latency_ms?: number;
  usage?: { prompt_tokens?: number; completion_tokens?: number };
};

const tabs: Array<[Tab, string]> = [
  ["ask", "Ask"],
  ["corpus", "Corpus"],
  ["runs", "Runs"],
  ["eval", "Eval"],
  ["settings", "Settings"]
];

const tabIcons = {
  ask: Search,
  corpus: Database,
  runs: History,
  eval: ClipboardCheck,
  settings: SettingsIcon
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) message = body.detail;
    } catch {
      /* noop */
    }
    throw new Error(message);
  }
  return res.json();
}

function Badge({ ok, label }: { ok?: boolean; label: string }) {
  return <span className={`badge ${ok === undefined ? "" : ok ? "ok" : "bad"}`}>{label}</span>;
}

export function App() {
  const [tab, setTab] = useState<Tab>("ask");
  return (
    <main>
      <header>
        <div>
          <h1>RAG Lab</h1>
          <StatusLine />
        </div>
        <nav>
          {tabs.map(([id, label]) => {
            const Icon = tabIcons[id];
            return (
              <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
                <Icon size={15} strokeWidth={1.8} aria-hidden="true" />
                {label}
              </button>
            );
          })}
        </nav>
      </header>
      {tab === "ask" && <AskView />}
      {tab === "corpus" && <CorpusView />}
      {tab === "runs" && <RunsView />}
      {tab === "eval" && <EvalView />}
      {tab === "settings" && <SettingsView />}
    </main>
  );
}

function StatusLine() {
  const stats = useQuery({ queryKey: ["stats"], queryFn: () => api<any>("/api/stats") });
  if (!stats.data) return <p className="muted">Loading collection</p>;
  return (
    <p className="muted">
      {stats.data.collection} · {stats.data.chunk_count} chunks · {stats.data.documents} docs
    </p>
  );
}

function AskView() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [rerank, setRerank] = useState(true);
  const [smallToBig, setSmallToBig] = useState(true);
  const mutation = useMutation({
    mutationFn: () =>
      api<QueryResponse>("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          mode,
          use_reranker: rerank,
          small_to_big: smallToBig
        })
      })
  });

  function submit(e: FormEvent) {
    e.preventDefault();
    if (question.trim()) mutation.mutate();
  }

  const result = mutation.data;
  const citationOk = result?.citation_validation?.citation_valid;

  return (
    <section className="grid two">
      <form className="panel" onSubmit={submit}>
        <h2>Ask</h2>
        <textarea value={question} onChange={(e) => setQuestion(e.target.value)} />
        <div className="controls">
          <label>
            Mode
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="hybrid">Hybrid</option>
              <option value="vector">Vector</option>
              <option value="lexical">Lexical</option>
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
            Rerank
          </label>
          <label className="check">
            <input type="checkbox" checked={smallToBig} onChange={(e) => setSmallToBig(e.target.checked)} />
            Small-to-big
          </label>
        </div>
        <button disabled={mutation.isPending}>
          <Send size={15} strokeWidth={1.8} aria-hidden="true" />
          {mutation.isPending ? "Thinking" : "Ask"}
        </button>
        {mutation.error && <p className="error">{(mutation.error as Error).message}</p>}
      </form>

      <section className="panel">
        <h2>Answer</h2>
        {!result && <p className="muted">No run selected</p>}
        {result && (
          <>
            <div className="meta">
              <Badge ok={citationOk} label={`Citations ${citationOk ? "valid" : "invalid"}`} />
              <Badge label={`Verifier ${result.verifier?.score ?? "—"}`} />
              <Badge label={`Run ${result.run_id ?? "—"}`} />
            </div>
            <pre className="answer">{result.answer}</pre>
            {!!result.citation_validation?.citation_errors?.length && (
              <ul className="errors">
                {result.citation_validation.citation_errors.map((e) => <li key={e}>{e}</li>)}
              </ul>
            )}
            <h3>Context</h3>
            <div className="stack">
              {(result.chunks || []).map((chunk) => (
                <details key={chunk.id}>
                  <summary>{chunk.citation || chunk.id}</summary>
                  <pre>{chunk.text}</pre>
                </details>
              ))}
            </div>
            <h3>Trace</h3>
            <pre className="json">{JSON.stringify(result.trace || [], null, 2)}</pre>
          </>
        )}
      </section>
    </section>
  );
}

function CorpusView() {
  const qc = useQueryClient();
  const docs = useQuery({ queryKey: ["docs"], queryFn: () => api<{ documents: Doc[]; manifest?: ManifestDoc[] }>("/api/docs") });
  const warningsById = new Map(
    (docs.data?.manifest || [])
      .filter((m) => m.parse_report?.warnings?.length)
      .map((m) => [m.doc_id, m.parse_report!.warnings!])
  );
  const del = useMutation({
    mutationFn: (id: string) => api(`/api/docs/${encodeURIComponent(id)}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["docs"] });
      qc.invalidateQueries({ queryKey: ["stats"] });
    }
  });
  const reingest = useMutation({
    mutationFn: (id: string) =>
      api(`/api/docs/${encodeURIComponent(id)}/reingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy: "sentence" })
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["docs"] });
      qc.invalidateQueries({ queryKey: ["stats"] });
    }
  });

  return (
    <section className="panel">
      <h2>Corpus</h2>
      <UploadBox />
      <table>
        <thead><tr><th>Document</th><th>Chunks</th><th>Strategy</th><th>Embedder</th><th /></tr></thead>
        <tbody>
          {(docs.data?.documents || []).map((doc) => (
            <tr key={doc.doc_id}>
              <td>
                <code>{doc.doc_id}</code><br /><span>{doc.source}</span>
                {warningsById.get(doc.doc_id)?.map((w, i) => (
                  <div key={i} className="warn">⚠ {w}</div>
                ))}
              </td>
              <td>{doc.chunks}</td>
              <td>{doc.strategy || "—"}</td>
              <td>{doc.embedding_model || "—"}</td>
              <td>
                <div className="actions">
                  <button className="ghost" title="Reingest" disabled={reingest.isPending} onClick={() => reingest.mutate(doc.doc_id)}>
                    <RefreshCw size={14} strokeWidth={1.8} aria-hidden="true" />
                    Reingest
                  </button>
                  <button className="danger" title="Delete" disabled={del.isPending} onClick={() => del.mutate(doc.doc_id)}>
                    <Trash2 size={14} strokeWidth={1.8} aria-hidden="true" />
                    Delete
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {del.error && <p className="error">{(del.error as Error).message}</p>}
      {reingest.error && <p className="error">{(reingest.error as Error).message}</p>}
    </section>
  );
}

function UploadBox() {
  const qc = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const upload = useMutation({
    mutationFn: async () => {
      if (!file) return null;
      const form = new FormData();
      form.append("file", file);
      return api<{ warnings?: string[] }>("/api/ingest", { method: "POST", body: form });
    },
    onSuccess: (res) => {
      setFile(null);
      setWarnings(res?.warnings || []);
      qc.invalidateQueries({ queryKey: ["docs"] });
      qc.invalidateQueries({ queryKey: ["stats"] });
    }
  });
  return (
    <div className="upload">
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
      <button disabled={!file || upload.isPending} onClick={() => upload.mutate()}>
        <Upload size={15} strokeWidth={1.8} aria-hidden="true" />
        Upload
      </button>
      {upload.error && <span className="error">{(upload.error as Error).message}</span>}
      {warnings.map((w, i) => (
        <span key={i} className="warn">⚠ {w}</span>
      ))}
    </div>
  );
}

function RunsView() {
  const [selected, setSelected] = useState<number | null>(null);
  const runs = useQuery({ queryKey: ["runs"], queryFn: () => api<{ runs: RunRow[] }>("/api/runs") });
  const detail = useQuery({
    queryKey: ["run", selected],
    queryFn: () => api<any>(`/api/runs/${selected}`),
    enabled: selected !== null
  });
  return (
    <section className="grid two">
      <div className="panel">
        <h2>Runs</h2>
        <div className="list">
          {(runs.data?.runs || []).map((run) => (
            <button key={run.id} className="rowButton" onClick={() => setSelected(run.id)}>
              <span>#{run.id} {run.question}</span>
              <span>{run.verifier?.score ?? "—"}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="panel">
        <h2>Run Detail</h2>
        <pre className="json">{JSON.stringify(detail.data || {}, null, 2)}</pre>
      </div>
    </section>
  );
}

function EvalView() {
  const [full, setFull] = useState(false);
  const evals = useQuery({ queryKey: ["evals"], queryFn: () => api<any>("/api/evals") });
  const runEval = useMutation({
    mutationFn: () => api<any>("/api/eval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ retrieval_only: !full, variant: full ? "dashboard-full" : "dashboard-retrieval" })
    })
  });
  const latest = useMemo(() => runEval.data || evals.data?.evals?.[0], [runEval.data, evals.data]);
  return (
    <section className="grid two">
      <div className="panel">
        <h2>Eval</h2>
        <label className="check">
          <input type="checkbox" checked={full} onChange={(e) => setFull(e.target.checked)} />
          Full API eval
        </label>
        <button disabled={runEval.isPending} onClick={() => runEval.mutate()}>
          <Play size={15} strokeWidth={1.8} aria-hidden="true" />
          {runEval.isPending ? "Running" : "Run eval"}
        </button>
        {runEval.error && <p className="error">{(runEval.error as Error).message}</p>}
        <h3>History</h3>
        <div className="list">
          {(evals.data?.evals || []).map((ev: any) => (
            <div className="row" key={ev.id}>#{ev.id} {ev.variant} · {Math.round((ev.summary?.hit_rate || 0) * 100)}%</div>
          ))}
        </div>
      </div>
      <div className="panel">
        <h2>Summary</h2>
        <pre className="json">{JSON.stringify(latest?.summary || {}, null, 2)}</pre>
      </div>
    </section>
  );
}

function SettingsView() {
  const cfg = useQuery({ queryKey: ["config"], queryFn: () => api<any>("/api/config") });
  const collections = useQuery({ queryKey: ["collections"], queryFn: () => api<any>("/api/collections") });
  return (
    <section className="grid two">
      <div className="panel">
        <h2>Settings</h2>
        <pre className="json">{JSON.stringify(cfg.data || {}, null, 2)}</pre>
      </div>
      <div className="panel">
        <h2>Collections</h2>
        <pre className="json">{JSON.stringify(collections.data || {}, null, 2)}</pre>
      </div>
    </section>
  );
}
