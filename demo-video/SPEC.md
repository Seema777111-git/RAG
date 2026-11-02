GOAL:    A recruiter ends the film knowing this RAG system shows its evidence for every answer, and is measured and tested.
OUTPUT:  HyperFrames, 1920 x 1080, 30 fps
TIME:    ~90 s x 30 = ~2700 frames. Narrated: the voice is the clock (about 2.2 words/s, max ~19 words a scene).
LOOK:    ground #F5F7FB with two soft light pools (blue rgba(124,155,255,.20) top-left, apricot rgba(217,130,43,.10) top-right);
         ink #1B2540, muted #66708A. Accents by meaning only: vector #3D63E0, graph #D9822B, guardrail #1E9E6A, lilac #8B6CF0
         only on the answer border. Display Outfit 800, 64 px headline, -2.5% tracking; labels Outfit 800 26 px;
         body Manrope 700 28 px. Medium: the product's own UI, shown as a white browser window (radius 20, 1 px #C7D5FF,
         glow 0 0 0 1px rgba(61,99,224,.16) + 0 0 28px rgba(61,99,224,.18)). No grain, no vignette.
OBJECTS: the real app window with its real screens; highlight rings drawn on the exact UI region in page coordinates;
         a pipeline diagram built from the app's own vocabulary (PDF, chunks, FAISS, graph, rails, gpt-4o).
MOTION:  camera = power2.inOut 1.1 s pushes across the captured page (one per scene); enters expo.out 0.6 s, 4-frame
         staggers; numbers count up 1.2-1.4 s then hold; rings draw 0.5 s; diagram parts appear when the voice names them.
SHOTS:   01 0-7     hook: the real answer card fills the window, citations light up; title lands
         02 7-17    upload: Documents tab, HRPolicy.pdf, 2 pages / 6 chunks / 48 facts count up
         03 17-25   ask: the question in the form, Hybrid mode ringed, Ask pressed
         04 25-35   answer: push into the answer card; C1, K1, K2 and both rail pills ringed in turn
         05 35-45   vector: push to the FAISS panel; top passage 0.558 and "also cited by a graph fact"
         06 45-55   graph: push to the graph panel; entities, facts, then the drawing
         07 55-68   flow: one diagram that grows, part by part, as the voice names it
         08 68-80   evaluation: real report; 1.00 x4 count up, 0.10 flagged as the next fix
         09 80-90   tests and close: 77 tests, FastAPI, Streamlit, Docker; closing line lands and holds
BANNED:  dark grounds anywhere; glow orbs, particles; three cards in a row; bullet slides; titles that drift after landing;
         everything fading up the same way; logo docked in a corner; numbers not in the source.
QA:      2 s (answer + title readable), 13 s (48 facts), 30 s (citations ringed), 40 s (0.558), 50 s (graph drawing),
         64 s (full diagram), 74 s (metrics), 88 s (closing line held).
