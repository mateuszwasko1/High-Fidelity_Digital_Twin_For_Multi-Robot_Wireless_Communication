# Report Prompt Templates

These templates help an external AI model generate a high‑quality report using the provided context files. Paste `docs/ARCHITECTURE.md`, `docs/METHODS.md`, `docs/LIMITATIONS.md`, `docs/FUTURE_WORK.md`, and optionally key snippets from `pipeline.py`.

> Tip: First paste ARCHITECTURE.md alone and prompt the model to reply with "Context loaded" only. Then progressively add more docs and ask for specific sections to avoid context dilution.

---

## 1) Context‑loading prompt
"""
You are assisting with a technical report about a PyBullet-based robot sorting system. I will paste project context files. Reply only with: Context loaded.
"""

## 2) Section drafting prompts
- Abstract (150–200 words):
"""
Using the loaded context, write a concise Abstract that states the problem, approach (overhead camera + CLIP + IK), key contributions (fast, modular pipeline), and outcomes (sorting demo) with no citations.
"""

- Introduction (0.5–1 page):
"""
Draft an Introduction motivating tabletop sorting with robot arms, challenges in perception and planning, and the rationale for PyBullet + CLIP. End with a bulleted summary of contributions.
"""

- Related Work (short, optional):
"""
Compare our CLIP-based shape classification and IK control to alternative approaches (classical CV, learned policies, multimodal VLMs). Keep high‑level, cite generically without URLs.
"""

- Methods:
"""
Using the context, write the Methods section with subsections: Simulation Setup, Camera & Depth Conversion, Object Detection via Depth Clustering, CLIP Shape Classification, IK Motion Planning, Gripper Control, Sorting Policy. Be precise and align with the code behavior.
"""

- Results & Evaluation plan:
"""
Propose an evaluation plan and report placeholder results: grasp success rate, classification accuracy, cycle time. Include a table layout and what figures we would show.
"""

- Discussion:
"""
Discuss strengths (speed, modularity) and weaknesses (approximate calibration, no collision checking). Tie observations to design choices.
"""

- Limitations & Future Work:
"""
Summarize limitations and propose concrete next steps (VLM motion planner, better calibration, RL integration, metrics & CI).
"""

- Conclusion:
"""
Write a clear Conclusion that restates the objective, summarizes the method, and highlights the path to a more capable system.
"""

## 3) Style and polish prompts
- Academic tone:
"""
Rewrite the section in an academic tone, reduce colloquialisms, keep technical specificity.
"""

- Brevity:
"""
Shorten the section by 20% while preserving all key technical details.
"""

- Figures & captions:
"""
Suggest 3 figures and their captions (camera view, detection mask, pick‑and‑place trajectory). Use Mermaid for one architecture diagram.
"""

## 4) Outline prompt (for table of contents)
"""
Generate a report outline with sections and subsections tailored to this project. Use 2 levels deep.
"""

## 5) Export prompts (optional)
- Slide deck bullets:
"""
Create slide bullets for a 10‑minute talk: 6–8 slides with titles and 3–4 bullets each.
"""

- Executive summary (200–300 words):
"""
Write an executive summary for a non‑technical audience.
"""
