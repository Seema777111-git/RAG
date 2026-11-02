---
format: 1920x1080
duration: 90s
message: "Upload a PDF, ask a question, and see exactly how the answer was built, then how it is measured and tested."
arc: Hook -> Build the index -> Read the answer -> How it works -> Proof
audience: recruiters and engineers viewing a portfolio
---

## Frame 1: The hook (0-7s)
- act: Hook
- status: built
- src: compositions/s01.html
- duration: 7s
- scene: The real answer card fills the window; its citations and guardrail badges light up as the title lands.
- voiceover: "Ask a question about any PDF, and get an answer that shows exactly where it came from."
- poster: 5.5

## Frame 2: Upload and ingest (7-17s)
- act: Build the index
- status: built
- src: compositions/s02.html
- duration: 10s
- scene: HRPolicy.pdf goes in. The camera pans across the result as 2 pages, 6 chunks and 48 graph facts count up.
- voiceover: "Upload a PDF. It's split into six chunks, then indexed twice: as vectors in FAISS, and as forty-eight facts in a knowledge graph."
- poster: 8.5

## Frame 3: Ask (17-25s)
- act: Build the index
- status: built
- src: compositions/s03.html
- duration: 8s
- scene: The question sits in the form; Hybrid mode and the Ask button are ringed in turn.
- voiceover: "Ask in plain English. Hybrid mode searches both indexes at the same time."
- poster: 6

## Frame 4: The answer (25-35s)
- act: Read the answer
- status: built
- src: compositions/s04.html
- duration: 10s
- scene: Push into the answer card. C1, K1/K2 and both rail badges ring as the voice names them.
- voiceover: "Every claim is cited. C marks a passage, K marks a graph fact. Guardrails checked the question, and the answer."
- poster: 7.5

## Frame 5: Vector retrieval (35-45s)
- act: Read the answer
- status: built
- src: compositions/s05.html
- duration: 10s
- scene: Push to the top FAISS passage: 0.558 counts up; the 'also cited by a graph fact' badge rings in apricot.
- voiceover: "FAISS ranks passages by meaning. The top match scores point five six, and a graph fact cites it too."
- poster: 8

## Frame 6: Knowledge graph (45-55s)
- act: Read the answer
- status: built
- src: compositions/s06.html
- duration: 10s
- scene: The camera steps from entities to facts to the graph drawing, each ringed in apricot.
- voiceover: "The graph links words in the question to entities, then follows their relations to exact facts."
- poster: 8.5

## Frame 7: Under the hood (55-68s)
- act: How it works
- status: built
- src: compositions/s07.html
- duration: 13s
- scene: One diagram grows part by part as the voice names it: ingestion, input rail, both searches, fuse, gpt-4o, output rail.
- voiceover: "Under the hood: a NeMo guardrail screens the question. Both searches run in parallel. The evidence is fused, gpt-4o answers, and a second rail checks it's grounded."
- poster: 12.5

## Frame 8: Evaluation (68-80s)
- act: Proof
- status: built
- src: compositions/s08.html
- duration: 12s
- scene: The real DeepEval report: 1.00 counts up for four metrics; context relevancy 0.10 is flagged as the next fix.
- voiceover: "DeepEval measures quality, with gpt-4o as the judge. Faithfulness, relevancy, precision and recall all score one. Context relevancy is the next thing to tune."
- poster: 11.5

## Frame 9: Tests and close (80-90s)
- act: Proof
- status: built
- src: compositions/s09.html
- duration: 10s
- scene: 77 counts up, the stack lands, then the closing line holds.
- voiceover: "Seventy-seven automated tests cover the pipeline, served with FastAPI, Streamlit and Docker. Every answer, with its evidence."
- poster: 8.5
