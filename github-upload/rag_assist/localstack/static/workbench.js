"use strict";
const $ = id => document.getElementById(id);
let products = [], last = null, lastProduct = null;
const names = {text:"文本",structured:"条款结构",layout:"表格与脚注",graph:"术语图谱"};
const statuses = {ANSWERED:"已找到引用",GENERAL:"通用解释",CLARIFY:"需要补充",HUMAN_REVIEW:"需人工核对"};
async function api(path, options = {}) {
  const res = await fetch(path, {credentials:"same-origin", ...options, headers:{"Content-Type":"application/json",...options.headers}});
  let body; try { body = await res.json(); } catch { throw new Error("服务返回异常，请检查本机服务。"); }
  if (res.status === 401) { $("login-panel").hidden = false; $("workspace").hidden = true; $("logout").hidden = true; }
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : "请求未通过检查，请核对输入。");
  return body;
}
function element(tag, text, cls) { const e = document.createElement(tag); if(text !== undefined) e.textContent = text; if(cls) e.className=cls; return e; }
function scope() {
  const p = products.find(p => p.product_id === $("product").value);
  $("scope").textContent = p ? `本次收录物理页：${p.selected_pages.join("、")}。其余页面尚未纳入本次检索。` : "通用概念可以直接提问。具体条款需要选择产品。";
}
async function loadCatalog() {
  const data = await api("/api/catalog"); products = data.products;
  $("product").replaceChildren(new Option("尚未选择产品", ""));
  products.forEach(p => $("product").add(new Option(p.title, p.product_id)));
  $("login-panel").hidden=true; $("workspace").hidden=false; $("logout").hidden=false; scope();
}
async function health() {
  try {const h=await api("/health"); const ready=h.backends.ready; $("health-text").textContent=ready ? "资料服务已就绪" : "资料服务尚未全部就绪"; $("health-dot").classList.toggle("ok",!!ready);} catch {$("health-text").textContent="本机服务未连接";}
}
async function openSource(citation, productId) {
  try {
    const saved=last?.contexts?.find(p=>p.source_id===citation.source_id && p.page_number===citation.page_number && p.raw_pdf_sha256===citation.raw_pdf_sha256);
    const page=saved || await api(`/api/pages/${encodeURIComponent(productId)}/${citation.page_number}`);
    if(citation.raw_pdf_sha256 && page.raw_pdf_sha256!==citation.raw_pdf_sha256) throw new Error("原文版本已变化，请查看该次查询保全的上下文。");
    $("source-title").textContent=page.title || citation.title;
    $("source-meta").textContent=`物理页 ${page.page_number} · 原文件 SHA256 ${page.raw_pdf_sha256}`;
    const box=$("source-body"); box.replaceChildren();
    const quote=citation.quote || ""; const at=quote ? page.text.indexOf(quote) : -1;
    if(at>=0) box.append(document.createTextNode(page.text.slice(0,at)),element("mark",quote),document.createTextNode(page.text.slice(at+quote.length)));
    else box.textContent=page.text;
    const url=page.source_url || citation.source_url; $("source-link").hidden=!/^https?:\/\//.test(url || "");
    if(!$("source-link").hidden) $("source-link").href=url;
    $("source-dialog").showModal(); const mark=box.querySelector("mark"); if(mark) mark.scrollIntoView({block:"center"});
  } catch(error) { $("request-error").textContent=error.message; }
}
function render(data, fallbackProduct) {
  last=data; lastProduct=data.product_id || fallbackProduct;
  $("empty").hidden=true; $("result").hidden=false; $("result-status").textContent=statuses[data.status] || data.status;
  const answer=$("answer"); answer.replaceChildren();
  if(data.claims?.length) data.claims.forEach(c => {const p=element("p",c.text,"claim"); for(const i of c.citation_indices || [c.citation_index]) {const cite=data.citations[i]; if(cite) {const b=element("button",`[${i+1}]`,"cite-ref"); b.onclick=()=>openSource(cite,lastProduct); p.append(b);}} answer.append(p);});
  else answer.append(element("p",data.answer));
  $("missing").replaceChildren(...(data.missing_info || []).map(v=>element("li",String(v)))); $("missing-wrap").hidden=!data.missing_info?.length;
  const list=$("citations"); list.replaceChildren();
  (data.citations || []).forEach((c,i)=>{const card=element("article",undefined,"citation-card"); card.append(element("div",`[${i+1}] ${c.title}`,"cite-title"),element("blockquote",c.quote)); const f=element("div",undefined,"cite-footer"); const button=element("button","查看完整原文页"); button.onclick=()=>openSource(c,lastProduct); f.append(element("span",`物理页 ${c.page_number} · 字符位置 ${c.start}–${c.end}`),button); card.append(f); list.append(card);});
  $("evidence-wrap").hidden=data.status!=="HUMAN_REVIEW" || !data.evidence?.length;
  $("evidence").replaceChildren(); if(!$("evidence-wrap").hidden) {const seen=new Set(); data.evidence.forEach(e=>{if(seen.has(e.page_number)) return; seen.add(e.page_number); const b=element("button",`${e.title} · 第${e.page_number}页`,"quiet"); b.onclick=()=>openSource(e,lastProduct); $("evidence").append(b);});}
  $("run-id").textContent=`记录 ${data.run_id || "未生成"}`;
  $("route-summary").replaceChildren(...Object.entries(data.routes || {}).map(([k,v])=>element("span",`${names[k] || k}：${v.status} / ${v.candidates?.length || 0}条`)));
  $("trace").textContent=JSON.stringify({actual_model:data.actual_model,bundle_id:data.bundle_id,timings:data.timings,diagnostics:data.diagnostics,context_page_numbers:data.context_page_numbers,routes:data.routes,answer_ceiling:data.answer_ceiling,semantic_complete:data.semantic_complete},null,2);
}
$("login-form").onsubmit=async event=>{event.preventDefault(); $("login-error").textContent=""; try {await api("/api/login",{method:"POST",body:JSON.stringify({token:$("token").value})}); $("token").value=""; await loadCatalog();} catch(error){$("login-error").textContent=error.message;}};
$("logout").onclick=async()=>{await api("/api/logout",{method:"POST"}); $("workspace").hidden=true; $("login-panel").hidden=false; $("logout").hidden=true;};
$("product").onchange=scope;
$("ask-form").onsubmit=async event=>{event.preventDefault(); const product=$("product").value || null; $("request-error").textContent=""; $("ask-button").disabled=true; $("loading").hidden=false; $("empty").hidden=true; $("result").hidden=true; $("result-status").textContent=""; try{const data=await api("/api/ask",{method:"POST",body:JSON.stringify({question:$("question").value,product_id:product})}); render(data,product);}catch(error){$("request-error").textContent=error.message; $("empty").hidden=false;}finally{$("ask-button").disabled=false; $("loading").hidden=true;}};
$("question").onkeydown=event=>{if(event.ctrlKey && event.key==="Enter" && !$("ask-button").disabled) $("ask-form").requestSubmit();};
document.querySelectorAll("[data-example]").forEach(b=>b.onclick=()=>{const kind=b.dataset.example; if(kind==="general"){$("product").value=""; $("question").value="什么是医疗险的免赔额？查具体产品时还需要提供什么信息？";}else{$("product").value="cpic_b_2024"; $("question").value=kind==="waiting" ? "太保长期B款2024的等待期如何规定？哪些情形不受等待期限制？" : "条款中的‘合理且必要’是什么意思？请列出定义和相关条件。";}scope(); $("question").focus();});
$("replay").onclick=async()=>{if(!last?.run_id) return; try{const value=await api(`/api/runs/${encodeURIComponent(last.run_id)}`); render(value,lastProduct);}catch(error){$("request-error").textContent=error.message;}};
$("close-source").onclick=()=>$("source-dialog").close();
health(); loadCatalog().catch(error=>{$("login-panel").hidden=false;if(!error.message.includes("登录")) $("login-error").textContent=error.message;});
