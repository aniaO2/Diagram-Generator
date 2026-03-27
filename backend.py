import requests
import json
import re

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "llama3:latest"

SYSTEM_PROMPT = """### ROLE
You are DiagramAI, a specialist system that converts natural-language requests into production-quality Mermaid flowcharts. Your goal is to map user logic into a coherent, valid, and visually readable diagram.

### OUTPUT RULES
1. STRICT JSON ONLY: Your entire response must be a single, valid JSON object.
2. NO MARKDOWN: Do not use markdown blocks (```json) or prose before/after the JSON.
3. SEMANTIC SHAPES:
   - Start/End/Terminal: ([Label])
   - Process/Action: [Label]
   - Decision/Branch: {Label}
   - Database/Storage: [(Label)]
4. FLOW: Every diagram must end with a terminal node (e.g., "User Response" or "Result").
5. ALWAYS generate a diagram, even if you ask for clarification.

### MERMAID SYNTAX CONSTRAINTS
- The first line must be the diagram type (e.g., "flowchart TD" or "flowchart LR").
- Use double backslashes (\\n) for newlines within the "mermaid_code" string.
- Edge labels must be actions (e.g., "validates", "stores", "calls").
- Never use color names or styling syntax inside the mermaid code.

### CLARIFICATION & ASSUMPTION POLICY
- If a prompt is underspecified, make a logical architectural assumption.
- Record all assumptions in the "assumptions_made" array.
- Always generate a best-effort diagram; never return an empty diagram.

### JSON SCHEMA
{
  "guardrail_triggered": boolean,
  "guardrail_reason": string | null,
  "needs_clarification": boolean,
  "clarification_question": string | null,
  "diagram_type": "flowchart",
  "diagram_description": "Single line summary",
  "mermaid_code": "flowchart TD\\n    A([Start]) --> B{Decision}\\n    B -- Yes --> C[Process]\\n    C --> D([Result])",
  "corrections_made": [],
  "assumptions_made": [],
  "confidence": number,
  "node_count": number,
  "user_message": "Success message"
}

### TASK
Convert the following user request into the JSON format above:
"""

def call_ollama(
    history: list,
    user_input: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL
) -> tuple:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history:
        if "raw" in msg:
            messages.append({"role": msg["role"], "content": msg["raw"]})

    messages.append({"role": "user", "content": user_input})

    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 4096
                },
            },
            timeout=120
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return {
            "guardrail_triggered": False,
            "guardrail_reason": None,
            "needs_clarification": False,
            "clarification_question": None,
            "diagram_type": "flowchart",
            "diagram_description": "Connection error",
            "mermaid_code": "flowchart TD\\n    A[Cannot connect to Ollama] --> B[Check that ollama serve is running] --> C([Return Result to User])",
            "corrections_made": [],
            "assumptions_made": [],
            "confidence": 0,
            "node_count": 3,
            "user_message": "Cannot connect to Ollama. Verify that the Ollama server is running."
        }, ""

    except requests.exceptions.RequestException as e:
        return {
            "guardrail_triggered": False,
            "guardrail_reason": None,
            "needs_clarification": False,
            "clarification_question": None,
            "diagram_type": "flowchart",
            "diagram_description": "HTTP error",
            "mermaid_code": f"flowchart TD\\n    A[Ollama request failed] --> B[{str(e)}] --> C([Return Result to User])",
            "corrections_made": [],
            "assumptions_made": [],
            "confidence": 0,
            "node_count": 3,
            "user_message": "The request to Ollama failed."
        }, ""

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
        return parsed, raw
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return parsed, raw
            except json.JSONDecodeError:
                pass

    return {
        "guardrail_triggered": False,
        "guardrail_reason": None,
        "needs_clarification": False,
        "clarification_question": None,
        "diagram_type": "flowchart",
        "diagram_description": "Parse error",
        "mermaid_code": "flowchart TD\\n    A[Model returned invalid JSON] --> B[Try regenerate or use another model] --> C([Return Result to User])",
        "corrections_made": [],
        "assumptions_made": [],
        "confidence": 0,
        "node_count": 3,
        "user_message": "The model returned invalid JSON. Try regenerating or switching models."
    }, raw