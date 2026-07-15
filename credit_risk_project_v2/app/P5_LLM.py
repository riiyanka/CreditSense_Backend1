import os, json, re
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

def llm(prompt, max_tokens=300):
    r = client.chat.completions.create(model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content.strip()

def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return {}
    try: return json.loads(m.group())
    except json.JSONDecodeError: return {}

GUIDELINES_PATH = os.path.join(os.path.dirname(__file__), "rbi_guidelines.txt")

def _load_chunks(path, size=120, overlap=20):
    with open(path, encoding="utf-8") as f:
        words = f.read().split()
    step = size - overlap
    return [" ".join(words[i:i+size]) for i in range(0, len(words), step)]

try:
    CHUNKS = _load_chunks(GUIDELINES_PATH)
    _VEC = TfidfVectorizer(stop_words="english").fit(CHUNKS)
    _MATRIX = _VEC.transform(CHUNKS)
    RAG_READY = True
    print(f"RAG ready: {len(CHUNKS)} chunks loaded")
except FileNotFoundError:
    CHUNKS, RAG_READY = [], False
    print("rbi_guidelines.txt not found")

def retrieve(query, k=3):
    if not RAG_READY: return []
    sims = (_MATRIX @ _VEC.transform([query]).T).toarray().ravel()
    return [CHUNKS[i] for i in sims.argsort()[-k:][::-1]]

# ── Agent 2: Explainer ──
def explainer_agent(question, profile, decision):
    prompt = f"""You are a friendly loan-decision explainer for an Indian borrower.
Decision from the model: {json.dumps(decision)}
Applicant profile: {json.dumps(profile)}
Borrower's question: "{question}"
Rules: 2-3 short plain sentences, second person, use the exact numbers above,
never invent or recalculate any number."""
    return llm(prompt)

# ── Agent 3: Negotiation ──
def negotiation_agent(question, profile, decision, run_cascade):
    extract = f"""Current applicant profile: {json.dumps(profile)}
The borrower proposes: "{question}"
Return ONLY a JSON object of the profile fields that should change with new
numeric values (e.g. {{\"existing_emi\": 4500}}). Use only existing field names.
If relative ("5000 lower"), compute the new absolute value. Else return {{}}"""
    changes = parse_json(llm(extract, max_tokens=120))
    changes = {k: v for k, v in changes.items() if k in profile}
    if not changes:
        return {"answer": "I couldn't map that to a concrete change. Try 'what if my EMI were 5,000 lower?'",
                "changes": {}, "new_decision": None}
    modified = {**profile, **changes}
    new_decision = run_cascade(modified)
    summary = f"""Borrower asked: "{question}"
Old decision: {json.dumps(decision)}
Changes: {json.dumps(changes)}
New decision: {json.dumps(new_decision)}
In 2 short sentences, tell the borrower what changed and the new grade/rate. Numbers above only."""
    return {"answer": llm(summary), "changes": changes, "new_decision": new_decision}

# ── Agent 4: Compliance (RAG) ──
def compliance_agent(decision, profile):
    if not RAG_READY:
        return {"flag": False, "reason": "Corpus not loaded.", "cited_text": ""}
    query = (f"fair lending discrimination interest rate creditworthiness "
             f"{profile.get('sector','')} {profile.get('rural_urban','')} "
             f"{profile.get('gender','')} rate {decision.get('interest_rate_pct')}")
    passages = retrieve(query, 3)
    numbered = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    prompt = f"""RBI guideline passages:
{numbered}
Loan decision: {json.dumps(decision)}
Applicant: sector={profile.get('sector')}, rural_urban={profile.get('rural_urban')}, gender={profile.get('gender')}
Based ONLY on the passages, does this decision risk violating fair-lending norms?
Return ONLY JSON: {{\"flag\": true/false, \"reason\": \"one sentence\", \"cited_passage_number\": 1}}"""
    out = parse_json(llm(prompt))
    n = out.get("cited_passage_number")
    out["cited_text"] = passages[n-1][:300] if isinstance(n, int) and 1 <= n <= len(passages) else ""
    out.setdefault("flag", False); out.setdefault("reason", "")
    return out
