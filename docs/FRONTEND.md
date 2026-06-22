# Frontend Integration Guide — Enervera Medical Assistant API

How to talk to the assistant API from a web/mobile frontend. The contract is
**identical across specialties** (pulmonology today, cardiology/nephrology/… later),
so build against this once and switch specialties by changing the base URL.

- **Live base URL (pulmonology):** `https://rag-pulmonology.onrender.com`
- **Interactive API explorer (Swagger):** `https://rag-pulmonology.onrender.com/docs`
- **Content type:** `application/json` for all POST requests.

---

## 1. Auth

If the service has an `API_KEY` configured, every `POST` must include the header:

```
X-API-Key: <the key>
```

- Missing/wrong key → **401** `{"detail":"Invalid or missing X-API-Key."}`.
- `GET /` and `GET /health` are **public** (no key needed).
- **Never ship the API key in client-side code.** Call the API from your **backend/BFF**, or put it behind a server-side proxy. A public SPA should not hold the key.

---

## 2. Endpoints

### `GET /health` — readiness
```
GET https://rag-pulmonology.onrender.com/health
→ 200 {"status":"ok"}
```
Use this to warm the service (see Cold starts) and for uptime checks.

### `POST /chat` — send a message, get an answer
```jsonc
// Request body
{
  "message": "I have a cough and breathlessness for a week",  // required, non-empty
  "session_id": "conv-abc123",   // optional, default "default" — see Sessions
  "user_id": "patient-789"       // optional — enables long-term (episodic) memory
}
```
```jsonc
// 200 Response
{
  "answer": "…assistant's reply (markdown-friendly text)…",
  "session_id": "conv-abc123"
}
```

### `POST /session/end` — close the conversation
Call this **once, when the user closes/leaves the chat.** It consolidates the
whole conversation into the patient's long-term memory. **This is the only time
long-term memory is written** (never per message).
```jsonc
// Request body
{ "session_id": "conv-abc123", "user_id": "patient-789" }
```
```jsonc
// 200 Response (shape varies)
{ "stored": true,  "episode_id": "…", "category": "symptom" }
{ "stored": false, "reason": "no clinical content extracted" }
{ "stored": false, "reason": "no user_id — episodic memory is per-user only" }
```
Fire-and-forget is fine (e.g., on chat close / tab unload). No `user_id` → it's a
safe no-op.

---

## 3. Sessions & users (important)

| Field | Meaning | How to set it |
|---|---|---|
| `session_id` | One **conversation**. Same id across turns = continuous memory (the assistant remembers earlier symptoms). New id = fresh conversation. | Generate a UUID when a chat opens; reuse it for every `/chat` in that chat; send it to `/session/end` at the end. |
| `user_id` | One **patient**, for long-term memory across conversations. | Use your stable auth/patient id. Leave empty/omit for anonymous chats (no long-term memory). |

**Lifecycle:**
```
open chat        → make a session_id (and use the patient's user_id)
each user msg    → POST /chat   { message, session_id, user_id }   (repeat)
close chat       → POST /session/end { session_id, user_id }       (once)
```
Sessions expire server-side after ~2h of inactivity, so don't rely on a
`session_id` living forever.

---

## 4. Rendering the answer

- `answer` is **human-readable text that may contain light markdown** (bold,
  bullet lists) and emojis. **Render it as markdown** for best results.
- All assistant outcomes come back as **200** with `{answer}` — you don't need to
  branch on status for these. Just display `answer`. They include:
  - a normal clinical assessment,
  - a brief greeting,
  - an **out-of-specialty** notice (e.g. "🫁 I'm focused on pulmonology…") when the
    question isn't in this assistant's domain,
  - a **non-medical refusal** ("❌ I can only answer healthcare-related questions…"),
  - an **emergency** response that leads with "seek care now" guidance.
- If `answer` is an empty string, treat it as a transient backend hiccup — show a
  generic "please try again" and let the user resend.

> Streaming: not yet supported — `/chat` returns the full answer in one response.
> (SSE/token streaming is on the roadmap; the contract won't break when added.)

---

## 5. Errors & edge cases

| Status | Meaning | Frontend action |
|---|---|---|
| 401 | Missing/invalid `X-API-Key` | Fix the key (server-side). |
| 422 | Validation error (e.g. empty `message`) | Validate inputs before sending; show a field error. |
| 405 | Wrong method/path | Use **POST** to `/chat` and `/session/end`; **GET** for `/health`. |
| 500 | `{"detail":"Pipeline error: …"}` | Show "something went wrong, try again"; log the detail. |
| 502 / 503 | Service starting or cold | Retry with backoff (see Cold starts). |

**Cold starts:** on Render the instance may spin down when idle; the **first
request after idle can take ~30–60s** (or briefly return 502). Strategy:
- Call `GET /health` when the chat UI mounts to warm the service.
- Use a generous client timeout (≥ 60s) for the first `/chat`.
- Retry 502/503 a couple of times with backoff before showing an error.

---

## 6. Examples

### JavaScript (call from YOUR backend, where the key is safe)
```js
const BASE = "https://rag-pulmonology.onrender.com";
const HEADERS = { "Content-Type": "application/json", "X-API-Key": process.env.ASSISTANT_API_KEY };

async function chat(message, sessionId, userId = "") {
  const r = await fetch(`${BASE}/chat`, {
    method: "POST", headers: HEADERS,
    body: JSON.stringify({ message, session_id: sessionId, user_id: userId }),
  });
  if (!r.ok) throw new Error(`chat ${r.status}: ${await r.text()}`);
  return (await r.json()).answer;
}

async function endSession(sessionId, userId) {
  await fetch(`${BASE}/session/end`, {
    method: "POST", headers: HEADERS,
    body: JSON.stringify({ session_id: sessionId, user_id: userId }),
  }); // fire-and-forget
}
```

### curl
```bash
curl -X POST https://rag-pulmonology.onrender.com/chat \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"message":"cough and breathlessness for a week","session_id":"conv-1","user_id":"p-1"}'

curl -X POST https://rag-pulmonology.onrender.com/session/end \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"session_id":"conv-1","user_id":"p-1"}'
```

---

## 7. Multiple specialties (future-proofing) ⭐

Each specialty is a **separate deployment** of the **same API** (its own base URL,
its own knowledge base). The **request/response contract above does not change** —
so the frontend is written once and parameterized by base URL.

**Recommended frontend design:**
```js
// One config map; add a row per specialty as they go live.
const ASSISTANTS = {
  pulmonology: "https://rag-pulmonology.onrender.com",
  // cardiology: "https://rag-cardiology.onrender.com",   // when deployed
  // nephrology: "https://rag-nephrology.onrender.com",
};
const BASE = ASSISTANTS[selectedSpecialty];
```

- **Same endpoints, same payloads** for every specialty — only `BASE` differs.
- **`session_id` / `user_id` are per-specialty.** If your patient switches
  specialty mid-visit, start a new `session_id` for the new base URL (memory is
  scoped to each assistant). The same `user_id` can be reused across specialties.
- **Scope handling:** each assistant only answers its own specialty and returns a
  polite out-of-scope message otherwise (see §4). If you offer a specialty picker,
  route the question to the matching base URL; if a user asks something off-topic,
  the assistant will say so — you can surface a "switch specialty?" prompt.
- **Branding:** `GET /` returns a `service` name (e.g. "Enervera Pulmonology
  Assistant") you can show as the assistant's title; it updates per deployment.

**Net:** build to this one contract now; onboarding a new specialty is just adding
its base URL to your config — no frontend code changes.

---

## 8. Quick reference

| Method | Path | Auth | Body | Returns |
|---|---|---|---|---|
| GET | `/` | no | — | `{service, status, docs}` |
| GET | `/health` | no | — | `{status:"ok"}` |
| POST | `/chat` | X-API-Key* | `{message, session_id?, user_id?}` | `{answer, session_id}` |
| POST | `/session/end` | X-API-Key* | `{session_id?, user_id?}` | `{stored, …}` |

\* required only if the service has `API_KEY` configured.
