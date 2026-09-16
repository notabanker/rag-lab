export function Badge({ ok, label }: { ok?: boolean; label: string }) {
  return <span className={`badge ${ok === undefined ? "" : ok ? "ok" : "bad"}`}>{label}</span>;
}
