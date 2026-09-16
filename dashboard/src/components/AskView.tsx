import { FormEvent, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Send } from "lucide-react";

import { api, QueryResponse } from "../api";
import { Badge } from "../components/Badge";

export function AskView() {
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
