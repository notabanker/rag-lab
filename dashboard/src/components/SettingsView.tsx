import { useQuery } from "@tanstack/react-query";

import { api } from "../api";

export function SettingsView() {
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
