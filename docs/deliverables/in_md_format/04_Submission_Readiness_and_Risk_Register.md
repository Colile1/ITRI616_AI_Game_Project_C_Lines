Submission Readiness Checklist
and Risk Register — ITRI 616 Mini-Project


Author: Colile Sibanda
Document purpose: A pre-submission audit instrument: a checklist mapped 1-to-1 to the official rubric, plus a risk register naming the top reasons a submission of this project could lose marks unintentionally.
Use: Walk through every checkbox the night before submission. Record each item as Done, Partial, or Not Done. Treat any "Not Done" in the rubric-aligned section as a blocker.

How to use this document
This is a practical instrument, not a report. The first section is the checklist, organised under the same five rubric headings the marker will use. The second section is the risk register: the named ways a submission like this can lose marks accidentally, ranked by likelihood × impact, with a single concrete mitigation for each. The third section is the final-night runbook — the exact, ordered shell commands to execute before zipping the project and uploading it.
 
1. Rubric-aligned Checklist
1.1 Formal TEP Definitions (15%)
•	Mitchell's TEP quote is reproduced verbatim somewhere in the report.
•	T = (S, A, δ) is stated formally and the symbols are defined.
•	Task category (decision-making) is explicit and justified.
•	State space is described with its concrete cardinality estimate (≈ 3^24).
•	Action space size (576) is stated and the encoding is shown.
•	Experience is named (self-play with replay buffer) and contrasted briefly against recorded play and simulated episodes.
•	Performance metric W(k) is defined as a function of training experience k.
•	Reward function is shown as a table.
•	tep_definitions.md is present in docs/ and is referenced from the main report.
1.2 Learning Algorithm Implementation (30%)
•	DQN choice is justified relative to tabular Q-learning.
•	Network architecture (75 → 128 → 128 → 576) is documented.
•	Action masking is implemented and explicitly mentioned in the report.
•	Target network sync schedule is documented (every 200 gradient steps).
•	Replay buffer capacity, batch size, and sampling strategy are documented.
•	Epsilon schedule (start, end, decay) is documented.
•	Self-play opponent freshness is correct (snapshot every N episodes, not per-episode deepcopy) — see risk R1.
•	Hyperparameters are centralised in get_training_config().
1.3 Experimental Evaluation (30%)
•	All five required figures exist in results/figures/ as PNGs.
•	Each figure is referenced by name in the report.
•	Each figure has a caption that names the X axis, the Y axis, and what is plotted.
•	Win rate trace crosses and ends above the 0.50 random baseline.
•	Numerical claims in the report (final win rate, peak win rate, episode of peak) match the CSV.
•	training_log.csv contains exactly one coherent run, no concatenated re-runs.
•	results/logs/config.json records the hyperparameters used.
•	(Stretch) Confidence band across at least three seeds is plotted.
•	(Stretch) A second baseline opponent (depth-2 minimax) is reported.
1.4 Critical Analysis (25%)
•	Section explicitly answers "did performance improve with experience?"
•	Section explicitly answers "was the problem well-posed?"
•	Assumptions are listed and named, not just implied.
•	At least three concrete limitations are stated, each with a one-sentence rationale.
•	The bias-variance tradeoff is mentioned in the context of the chosen architecture and dataset.
•	Failures or plateaus visible in the figures are surfaced and explained, not hidden.
•	The reward-curve regression at episode 5 000 is explicitly discussed.
•	(Stretch) At least one ablation result is referenced.
1.5 Code Quality (10%)
•	Project follows the directory structure in plan.md.
•	Each package has __init__.py and README.md.
•	No file exceeds 120 lines by a meaningful margin.
•	All public functions have docstrings.
•	Pure functions (board_rules, game_phases) contain no I/O.
•	requirements.txt is present and minimal.
•	README.md gives a one-line install + one-line run instruction.
•	docs/starter_guide.md gives the full reproduction recipe.
•	Tests under tests/ pass: python -m pytest tests/ -v.
•	Code is PEP-8 compliant.
1.6 Documentation and Submission Hygiene
•	docs/report.md is updated with results from the final run (not stub text).
•	docs/log.md has at least one entry per work session.
•	docs/todo.md is current — no "in progress" items remain on submission day.
•	README.md links to docs/starter_guide.md and docs/report.md.
•	__pycache__ directories and .pyc files are excluded from the submission archive.
•	.git directory is included in the archive only if the brief asks for it.
•	results/figures/ contains the five PNGs and nothing extraneous.
•	Plagiarism declaration is signed if the institution requires one.
•	Final filename matches the institution's naming convention.
 
2. Risk Register
This is the list of named ways the submission could lose marks accidentally — i.e., for reasons unrelated to the technical quality of the work. Each risk has an ID, a likelihood (Low/Med/High), an impact (Low/Med/High), and a single concrete mitigation.
ID	Risk	Likelihood	Impact	Mitigation
R1	Marker reads training_log.csv and sees concatenated re-runs	High	Medium	Truncate the CSV before submission so it contains only the final run's rows.
R2	results/figures/ ships an old PNG that contradicts the report's numbers	Medium	High	Re-run plotter.py from the cleaned CSV the night before submission; verify file timestamps.
R3	Marker tries to reproduce results and the run errors out	Low	High	Run pip install -r requirements.txt && python -m pytest tests/ -v on a fresh clone the night before.
R4	Self-play deepcopy bug is read as carelessness, not a deliberate choice	Medium	Medium	Either fix it (P0 of improvement plan) or document it explicitly in section 5.4 of the report.
R5	Report's reward-curve discussion is read as defensive rather than analytical	Medium	Medium	Lead the section with the diagnosis (curriculum switch) before describing the symptom.
R6	Win-rate confidence interval not stated, marker assumes no awareness	High	Low	Add one sentence to section 4.1: "95% CI on a 0.5 success rate over 200 games is ±0.07".
R7	Project loses code-quality marks for committed __pycache__ folders	Medium	Low	Add __pycache__ to .gitignore and delete checked-in copies.
R8	Tests pass on author's machine but fail elsewhere due to relative imports	Low	High	Run from a fresh terminal with no PYTHONPATH set; confirm python -m pytest works.
R9	Hyperparameters in report don't match config.json	Medium	Medium	Generate the hyperparameter table in the report by reading config.json, not by retyping.
R10	Marker can't open .docx version of report	Low	High	Submit both report.md and report.pdf (LibreOffice convert) so there is no ambiguity.
R11	Word count or page limit exceeded	Low	Low	Check the brief — the brief here does not specify a limit, but if it does, count words before submission.
R12	Citation format inconsistent	Medium	Low	Use Harvard or APA throughout; do not mix.
R13	Game choice questioned (Morabaraba vs Nine Men's Morris confusion)	Low	Medium	Add one sentence in section 1: "Morabaraba shares its board topology with Nine Men's Morris but is the indigenous Southern African form of the game; this project implements the rules as played in South Africa."
R14	TEP definitions not on the first page of the report	High	Low	The brief says lead with TEP. Move section 2 to immediately after the abstract — it already is, so verify.
R15	Brief asks for "comparison of experience types"; report does not	High	Medium	Add a half-page subsection contrasting self-play / recorded / simulated, even if cursorily.
 
3. Final-Night Runbook
The literal sequence of commands and edits to execute the night before submission. Steps are deliberately small so that any failure can be diagnosed without backtracking.
3.1 Repository hygiene
1.	git status — confirm working tree is clean.
2.	Delete every __pycache__ directory: find . -type d -name __pycache__ -exec rm -rf {} +.
3.	Confirm .gitignore contains __pycache__/, *.pyc, results/logs/*.npz, .DS_Store.
3.2 Reproducibility check
4.	Create a fresh virtual environment: python -m venv .venv-check && source .venv-check/bin/activate.
5.	pip install -r requirements.txt.
6.	python -m pytest tests/ -v — every test must pass.
7.	python -c "from src.training.train import get_training_config; print(get_training_config())" — must print the config.
3.3 Final results regeneration
8.	If implementation plan items P0/P1 were applied, run python -m src.training.train one more time.
9.	Verify results/figures/ contains exactly: win_rate.png, reward_curve.png, episode_length.png, epsilon_decay.png, loss_curve.png.
10.	Open each PNG and confirm the X and Y axis labels match what the report claims.
3.4 Document regeneration
11.	Update docs/report.md with the new numerical claims from training_log.csv.
12.	Re-export the report .docx if the institution requires it.
13.	Re-export to PDF: libreoffice --headless --convert-to pdf docs/report.md (or pandoc).
14.	Open the PDF and verify all figures render.
3.5 Archive and submit
15.	Create the submission archive: zip -r ITRI616_Sibanda_Morabaraba.zip 616_AI_Project -x "*.git/*" -x "*__pycache__/*" -x "*.pyc" -x "*.venv*".
16.	Verify the archive: unzip -l ITRI616_Sibanda_Morabaraba.zip | head -50.
17.	Confirm the archive size is reasonable (< 50 MB).
18.	Upload to the institution's submission portal.
19.	Take a screenshot of the submission confirmation page.
3.6 Self-attestation
By the time of upload the student should be able to honestly answer "yes" to all of the following questions:
•	If the marker re-runs python -m src.training.train, will the result be qualitatively the same as what the report claims?
•	If the marker challenges any single number in the report, can I point to its source in training_log.csv or config.json?
•	If the marker asks "why DQN and not supervised learning?", do I have a one-paragraph answer in the report?
•	If the marker asks "what would you do next?", do I have at least three concrete answers in section 5 or in the improvement plan?
•	If the marker says the win-rate curve is noisy, do I have a sentence in the report acknowledging the 200-game evaluation noise floor?
 
4. One-page Self-Audit
Read each statement aloud. If you cannot answer a clean "true" to all twelve, the submission is not ready.
20.	The five required figures are in results/figures/ and they are the most recent run.
21.	The numbers in the report match the numbers in the CSV.
22.	The TEP definitions are explicit, formal, and lead the report.
23.	The reward function is shown as a table.
24.	The hyperparameter table in the report matches config.json.
25.	Every public function in src/ has a docstring.
26.	python -m pytest tests/ -v passes from a fresh clone.
27.	There is no committed __pycache__ in the archive.
28.	The report names at least three limitations of the work.
29.	The report explains the regression at episode 5 000.
30.	The submission file name matches the institution's convention.
31.	A backup copy of the submission archive is stored somewhere other than the work machine.
Appendix A — Mapping from this document to the rubric
Rubric item	Where to verify
Formal TEP definitions (15%)	§1.1 of this document
Learning algorithm (30%)	§1.2 of this document; risk R4
Experimental evaluation (30%)	§1.3 of this document; risks R1, R2, R6, R9
Critical analysis (25%)	§1.4 of this document; risk R5
Code quality (10%)	§1.5 of this document; risk R7
Submission hygiene (zero-marks-but-don't-lose)	§1.6 of this document; §3 runbook
Closing Note
A submission rarely fails for technical reasons. It usually fails for a combination of small, named, mostly preventable risks accumulating in the last 24 hours. The function of this document is to make those risks legible early enough that they can be retired one at a time. If every checkbox above is ticked, the submission is doing its job.
