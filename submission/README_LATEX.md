# How to compile `report.tex`

Three options, easiest first.

---

## Option 1 — Overleaf (recommended, zero install)

1. Go to https://overleaf.com → New Project → Blank Project
2. Delete the default `main.tex`
3. Upload `submission/report.tex` (drag-drop)
4. Upload the 5 figure PNGs (drag-drop into the project root):
   - `extras/plots/report/trajectory.png`
   - `extras/plots/report/max_len_fix.png`
   - `extras/plots/report/scaling_corrected.png`
   - `extras/plots/report/submission_quality.png`
   - (optional) `extras/plots/report/phase_comparison.png`
5. Click **Recompile** → download the PDF
6. Total time: ~5 minutes

The `\graphicspath{}` line in `report.tex` looks for figures in three places, so as long as the PNGs sit next to the .tex file OR at `extras/plots/report/`, they'll render.

---

## Option 2 — pdflatex locally

If you have a TeX distribution (TeX Live on Linux/Mac, MiKTeX on Windows):

```bash
cd submission
# Copy figures next to the .tex
cp ../extras/plots/report/*.png .

# Compile twice (first pass writes refs, second resolves them)
pdflatex report.tex
pdflatex report.tex
```

Result: `report.pdf` next to `report.tex`.

If you get missing-package errors, install via your package manager:
```bash
# TeX Live
sudo tlmgr install booktabs caption enumitem geometry hyperref microtype subcaption
```

---

## Option 3 — VS Code with LaTeX Workshop extension

1. Install the LaTeX Workshop extension
2. Open `submission/report.tex`
3. Make sure the figure PNGs are in `extras/plots/report/` (they already are)
4. Save the file — LaTeX Workshop auto-compiles on save
5. PDF preview shows in a side panel

---

## What the PDF looks like

- Single-column, 11pt, A4, 1-inch margins
- ~3 pages of content (4 with figures rendered full-width)
- 4 inline figures (scaling, max_len fix, trajectory, submission quality)
- 2 numbered tables (training phases, LoFO results)
- Hyperlinked references to sections + the GitHub URL

---

## Tweaks you might want before submitting

### Authors line

Top of `report.tex`, around line 38:
```latex
\author{Team \texttt{abb}\\
\small Zero One Hack\_01 --- Industrial AI (Infineon) Track\\
\small\texttt{https://github.com/Julian-AT/zero\_one\_hack\_01}}
```

Add real names if you want:
```latex
\author{Alice Foo, Bob Bar, Carol Qux \\
\small Team \texttt{abb} --- Zero One Hack\_01 --- Industrial AI (Infineon) Track}
```

### Adjust the headline submission

The paper currently presents our v3-medium recipe as the result. If the team ships **main's SSL Transformer + reranker** or the **neurosymbolic** approach as the official submission, swap the numbers in `\subsection{Leave-one-family-out generalisation}` (Table \ref{tab:lofo}) and in the abstract.

For main (SSL Transformer + reranker):
- ID Top-1: 0.804
- LoFO: not measured; cite the team's cross-architecture benchmark

For neurosymbolic:
- ID Top-1: 0.696
- OOD Top-1: 0.681, drop: +0.015

### Two-column version

If you want the more "conference paper" look, change line 28:
```latex
\documentclass[11pt,a4paper]{article}
```
to:
```latex
\documentclass[10pt,a4paper,twocolumn]{article}
```
And remove the `\begin{abstract}...\end{abstract}` block (two-column abstracts need different markup).

### NeurIPS / ICML / arXiv style

If you want a known conference template, replace `\documentclass{article}` with the appropriate class file (`neurips_2023`, `icml2024`, etc.). The body content stays the same; just the title-block and bibliography style change.

---

*The .tex compiles cleanly out-of-the-box on Overleaf with no extra setup. Time to PDF: ~2 minutes once the figures are uploaded.*
