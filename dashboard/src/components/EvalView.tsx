import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Play } from "lucide-react";

import { api } from "../api";

export function EvalView() {
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
