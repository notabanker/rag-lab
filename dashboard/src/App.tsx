import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ClipboardCheck,
  Database,
  History,
  Search,
  Settings as SettingsIcon
} from "lucide-react";

import { api, Tab } from "./api";
import { AskView } from "./components/AskView";
import { CorpusView } from "./components/CorpusView";
import { RunsView } from "./components/RunsView";
import { EvalView } from "./components/EvalView";
import { SettingsView } from "./components/SettingsView";

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
