import { useState } from "react";
import { QueryClient, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, Trash2, Upload } from "lucide-react";

import { api, Doc, ManifestDoc } from "../api";

export function CorpusView() {
  const qc = useQueryClient();
  const docs = useQuery({ queryKey: ["docs"], queryFn: () => api<{ documents: Doc[]; manifest?: ManifestDoc[] }>("/api/docs") });
  const warningsById = new Map(
    (docs.data?.manifest || [])
      .filter((m) => m.parse_report?.warnings?.length)
      .map((m) => [m.doc_id, m.parse_report!.warnings!])
  );
  const del = useMutation({
    mutationFn: (id: string) => api(`/api/docs/${encodeURIComponent(id)}`, { method: "DELETE" }),
    onSuccess: () => refresh(qc)
  });
  const reingest = useMutation({
    mutationFn: (id: string) =>
      api(`/api/docs/${encodeURIComponent(id)}/reingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy: "sentence" })
      }),
    onSuccess: () => refresh(qc)
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

function refresh(qc: QueryClient) {
  qc.invalidateQueries({ queryKey: ["docs"] });
  qc.invalidateQueries({ queryKey: ["stats"] });
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
      refresh(qc);
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
