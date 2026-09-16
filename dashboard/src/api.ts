export type Tab = "ask" | "corpus" | "runs" | "eval" | "settings";

export type Doc = {
  doc_id: string;
  file_sha?: string;
  source: string;
  basename: string;
  chunks: number;
  strategy?: string;
  embedding_model?: string;
  chunking_version?: string;
};

export type ManifestDoc = {
  doc_id: string;
  parse_report?: { warnings?: string[] } | null;
};

export type RunRow = {
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

export type QueryResponse = {
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

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
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
