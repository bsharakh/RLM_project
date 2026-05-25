"""Prompts for the unified-reasoning RLM."""

INTERN_PROMPT = """You read ONE passage and extract facts relevant to the question.

Rules:
- Only state what is literally in the passage. Never guess or use outside knowledge.
- If the passage uses a different name for the same entity (e.g., "German Aerospace Center" = "DLR"), note both names.
- If nothing in the passage relates to the question, say exactly: "Nothing relevant."
- Keep it short: one fact per line.

Question: {question}

Passage:
{context}

Relevant facts (one per line, short):"""


BOSS_UPDATE_PROMPT = """You are reasoning step-by-step to answer a question. You see your previous reasoning and a new finding from a research assistant.

Your job: continue your reasoning. Think out loud. Connect new facts to what you already know. If you can now answer the question (even partially), state your answer. If the new finding changes your best guess, update it.

RULES:
- Build on your previous reasoning — never start over.
- You may ONLY use facts reported by the assistant. Do NOT use any outside knowledge or make assumptions beyond what the assistant has explicitly told you. If the assistant said "Nothing relevant", you learned nothing new — do not guess.
- Connect facts into chains: if you know A relates to B, and now learn B relates to C, write "So A relates to C through B."
- If two names refer to the same thing (abbreviations, aliases), note the connection explicitly.
- Always end with exactly this line: CURRENT BEST ANSWER: [your best guess or "none yet"]
- A partial or uncertain answer is ALWAYS better than "none yet". Only say "none yet" if you have truly zero relevant information.
- Once you have an answer, you may only change it to something MORE SPECIFIC or BETTER SUPPORTED — never go back to "none yet".
- Keep your reasoning SHORT — max 10 lines. Drop old reasoning that turned out to be irrelevant.

Question: {question}

Your reasoning so far:
{working_answer}

New finding from assistant:
{finding}

Continue your reasoning (end with CURRENT BEST ANSWER: ...):"""