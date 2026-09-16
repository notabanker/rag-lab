import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { api, RunRow } from "../api";

export function RunsView() {
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
