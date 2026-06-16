import hashlib
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse

from .parsers import pick_parser
from . import chunker, embedder, vector_store
from .retriever import retrieve

app = FastAPI(title="rag-lab test console")

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>rag-lab test console</title>
<style>
:root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#c9d1d9;--dim:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149;--yellow:#d2991d}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;padding:20px;max-width:960px;margin:0 auto}
h1{font-size:20px;margin-bottom:20px;color:var(--accent)}
h2{font-size:14px;text-transform:uppercase;letter-spacing:0.5px;color:var(--dim);margin-bottom:8px}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:16px;margin-bottom:16px}
.row{display:flex;gap:12px;flex-wrap:wrap}
.col{flex:1;min-width:200px}
label{display:block;font-size:12px;color:var(--dim);margin-bottom:4px}
input,select,textarea{width:100%;padding:8px 10px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:13px;font-family:inherit}
textarea{resize:vertical;min-height:60px}
button{padding:8px 16px;background:var(--accent);color:#fff;border:none;border-radius:4px;font-size:13px;cursor:pointer;font-weight:600}
button:hover{opacity:0.85}
button:disabled{opacity:0.4;cursor:not-allowed}
.status{padding:8px 12px;border-radius:4px;font-size:12px;margin-top:8px;display:none}
.status.info{background:#1f2937;color:var(--accent);display:block}
.status.ok{background:#0d3320;color:var(--green);display:block}
.status.err{background:#3d1214;color:var(--red);display:block}
.answer-box{background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:12px;margin-top:8px;font-size:13px;white-space:pre-wrap}
.meta{font-size:11px;color:var(--dim);margin-top:4px}
.chunk-ref{color:var(--yellow);font-weight:600}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;margin-right:4px}
.tag.grounded{background:#0d3320;color:var(--green)}
.tag.partial{background:#2e2500;color:var(--yellow)}
.tag.ungrounded{background:#3d1214;color:var(--red)}
.trace-entry{margin-top:8px;padding:8px;background:var(--bg);border-radius:4px;font-size:12px;cursor:pointer}
.trace-entry summary{color:var(--dim)}
.trace-entry pre{font-size:11px;white-space:pre-wrap;margin-top:4px;max-height:200px;overflow-y:auto}
.file-drop{border:2px dashed var(--border);border-radius:6px;padding:24px;text-align:center;color:var(--dim);font-size:13px;cursor:pointer;transition:border-color 0.2s}
.file-drop:hover,.file-drop.dragover{border-color:var(--accent)}
.spinner{display:inline-block;width:12px;height:12px;border:2px solid var(--dim);border-top-color:var(--accent);border-radius:50%;animation:spin 0.6s linear infinite;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<h1>rag-lab test console</h1>

<div class="panel">
  <h2>Ingest</h2>
  <div class="file-drop" id="dropzone">Drop a PDF, EPUB, or Markdown file here — or click to browse</div>
  <input type="file" id="fileinput" accept=".pdf,.epub,.md,.markdown" style="display:none">
  <div class="row" style="margin-top:8px">
    <div class="col"><label>Strategy</label><select id="strategy"><option value="sentence">Sentence</option><option value="fixed">Fixed</option></select></div>
    <div class="col"><label>Chunk size</label><input id="chunksize" type="number" value="512" min="64" max="4096"></div>
    <div class="col"><label>Overlap</label><input id="overlap" type="number" value="64" min="0" max="2048"></div>
  </div>
  <button id="ingestbtn" style="margin-top:8px">Ingest</button>
  <div id="ingeststatus" class="status"></div>
</div>

<div class="panel">
  <h2>Query</h2>
  <div class="row">
    <div class="col" style="flex:2"><label>Question</label><textarea id="question" placeholder="Ask something about the ingested documents..."></textarea></div>
    <div class="col"><label>Top-K</label><input id="topk" type="number" value="20" min="1" max="100"></div>
    <div class="col"><label>Min Score</label><input id="minscore" type="number" value="8" min="1" max="10"></div>
  </div>
  <button id="querybtn" style="margin-top:8px">Query</button>
  <div id="querystatus" class="status"></div>
  <div id="answerarea" style="display:none;margin-top:8px">
    <div class="answer-box" id="answertext"></div>
    <div class="meta" id="verdictmeta"></div>
    <div id="traces"></div>
  </div>
</div>

<div class="panel">
  <h2>Stats</h2>
  <div id="statsbox"><span class="spinner"></span> loading...</div>
</div>

<script>
const $=s=>document.getElementById(s);
const status=(el,cls,msg)=>{el.className='status '+cls;el.textContent=msg};
const spin=(btn,on)=>{btn.disabled=on;btn.innerHTML=on?'<span class="spinner"></span> Working...':btn.dataset.label};

$('fileinput').dataset.label||($('fileinput').dataset.label='');
$('ingestbtn').dataset.label='Ingest';
$('querybtn').dataset.label='Query';

$('dropzone').onclick=()=>$('fileinput').click();
$('fileinput').onchange=()=>{
  const f=$('fileinput').files[0];
  if(f) $('dropzone').textContent=f.name+' ('+(f.size/1024).toFixed(1)+' KB)';
};
['dragenter','dragover'].forEach(e=>$('dropzone').addEventListener(e,ev=>{ev.preventDefault();$('dropzone').classList.add('dragover')}));
['dragleave','drop'].forEach(e=>$('dropzone').addEventListener(e,ev=>{ev.preventDefault();$('dropzone').classList.remove('dragover')}));
$('dropzone').addEventListener('drop',ev=>{
  const f=ev.dataTransfer.files[0];
  if(f){$('fileinput').files=ev.dataTransfer.files;$('dropzone').textContent=f.name+' ('+(f.size/1024).toFixed(1)+' KB)'}
});

$('ingestbtn').onclick=async()=>{
  const f=$('fileinput').files[0];
  if(!f){status($('ingeststatus'),'err','No file selected');return}
  spin($('ingestbtn'),true);
  status($('ingeststatus'),'info','Ingesting...');
  const fd=new FormData();
  fd.append('file',f);
  fd.append('strategy',$('strategy').value);
  fd.append('chunk_size',$('chunksize').value);
  fd.append('overlap',$('overlap').value);
  try{
    const r=await fetch('/api/ingest',{method:'POST',body:fd});
    const j=await r.json();
    if(r.ok)status($('ingeststatus'),'ok','Ingested '+j.chunks+' chunks from '+f.name);
    else status($('ingeststatus'),'err',j.error||'Ingest failed');
  }catch(e){status($('ingeststatus'),'err','Error: '+e.message)}
  spin($('ingestbtn'),false);
  loadStats();
};

$('querybtn').onclick=async()=>{
  const q=$('question').value.trim();
  if(!q){status($('querystatus'),'err','Enter a question');return}
  spin($('querybtn'),true);
  status($('querystatus'),'info','Querying...');
  $('answerarea').style.display='none';
  try{
    const r=await fetch('/api/query',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q,top_k:+$('topk').value,min_score:+$('minscore').value})
    });
    const j=await r.json();
    if(!r.ok){status($('querystatus'),'err',j.error||'Query failed');spin($('querybtn'),false);return}
    status($('querystatus'),'ok','Done in '+j.iterations+' iteration(s)');
    $('answertext').textContent=j.answer;
    const v=j.verifier||{};
    const vcls=v.verdict==='GROUNDED'?'grounded':v.verdict==='PARTIAL'?'partial':'ungrounded';
    $('verdictmeta').innerHTML='<span class="tag '+vcls+'">'+v.verdict+'</span> score: '+v.score+'/10'+(j.partial?' <span class="tag ungrounded">max iters</span>':'');
    $('traces').innerHTML='';
    if(j.trace) j.trace.forEach((t,i)=>{
      const d=document.createElement('details');d.className='trace-entry';
      d.innerHTML='<summary>Iter '+t.iter+' — score '+t.verifier_score+'</summary><pre>Query: '+esc(t.query)+'\\n\\nAnswer: '+esc(t.answer||'')+'\\n\\nIssues: '+esc(JSON.stringify(t.issues||[]))+'</pre>';
      $('traces').appendChild(d);
    });
    $('answerarea').style.display='block';
  }catch(e){status($('querystatus'),'err','Error: '+e.message)}
  spin($('querybtn'),false);
};

async function loadStats(){
  try{
    const r=await fetch('/api/stats');
    const j=await r.json();
    $('statsbox').innerHTML='Chunks: <b>'+j.chunk_count+'</b> &nbsp;|&nbsp; DB: <b>'+j.db_path+'</b> &nbsp;|&nbsp; Embedder: all-MiniLM-L6-v2 (CPU) &nbsp;|&nbsp; LLM: DeepSeek deepseek-chat';
  }catch(e){$('statsbox').textContent='Stats unavailable'}
}
function esc(s){return(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
loadStats();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE

@app.post("/api/ingest")
async def api_ingest(
    file: UploadFile = File(...),
    strategy: str = Form("sentence"),
    chunk_size: int = Form(512),
    overlap: int = Form(64),
):
    try:
        suffix = Path(file.filename).suffix
        if suffix.lower() not in {".pdf", ".epub", ".md", ".markdown"}:
            return {"error": f"Unsupported file type: {suffix}"}

        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        parser = pick_parser(tmp_path)
        text = parser(tmp_path)
        Path(tmp_path).unlink()

        if strategy == "fixed":
            chunks = chunker.chunk_fixed(text, size=chunk_size, overlap=overlap)
        else:
            chunks = chunker.chunk_sentence(text, target_size=chunk_size, overlap=max(1, overlap // 64))

        if not chunks:
            return {"error": "No text extracted from file"}

        vecs = embedder.embed([c.text for c in chunks])
        sha = hashlib.sha256(content).hexdigest()[:10]
        metadatas = [{"source": file.filename, "chunk_idx": i, "strategy": strategy, "file_sha": sha} for i in range(len(chunks))]
        ids = [f"{sha}-{i}" for i in range(len(chunks))]
        vector_store.upsert(chunks, vecs, metadatas, ids)
        return {"status": "ok", "chunks": len(chunks), "filename": file.filename}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Ingest failed: {e}"}

@app.post("/api/query")
async def api_query(data: dict):
    question = data.get("question", "")
    top_k = data.get("top_k", 20)
    min_score = data.get("min_score", 8)
    if not question.strip():
        return {"error": "Question is required"}
    try:
        result = retrieve(question, top_k=top_k, min_score=min_score)
        return {
            "answer": result["answer"],
            "verifier": result["verifier"],
            "iterations": result["iterations"],
            "trace": result.get("trace", []),
            "partial": result.get("partial", False),
        }
    except Exception as e:
        return {"error": f"Query failed: {e}"}

@app.get("/api/stats")
async def api_stats():
    return {
        "chunk_count": vector_store.count(),
        "db_path": vector_store._PERSIST_DIR,
    }
