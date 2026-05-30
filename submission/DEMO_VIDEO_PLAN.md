# Demo video plan — 2-minute submission video

> A required deliverable per `submission/SUBMISSION.md`. This doc covers:
> what the organizers want, what we should show, the second-by-second
> storyboard, the voiceover script, and how to record it.

---

## 1. What the organizers actually ask for

From `submission/SUBMISSION.md` §5:

- **Max 2 minutes — hard cutoff.** Don't go over.
- **Format**: MP4, 1080p, with audio. Upload directly to Tally, or paste an unlisted YouTube / Vimeo / Loom link in the form.
- **A good demo video shows:**
  - **The problem in 15 seconds** — no setup, just the pain
  - **The solution running** — *live, not slideware*
  - **One concrete result** — with a number or comparison
  - **The reasoning visible** — what your system decided and why

### Industrial-track-specific (from `submission/SUBMISSION.md` and `REPORT_TEMPLATE.md`)

> "**Demo shows baseline vs. trained output on identical inputs**"

This is the specific deliverable — a side-by-side comparison of what the baseline produces vs. what our system produces, on the same prefix. We already have this exact tool: `shared/scripts/demo_compare.py`.

### What they explicitly say NOT to do

> "We mean 2 minutes. The jury sees many submissions — clarity beats completeness."
> "Live, not slideware."

So **no static slides as the main content**. Maybe a 5-second title card / scaling-curve flash, but the meat is the system running.

---

## 2. The story arc — 4 beats in 120 seconds

```
[0:00 – 0:15]  Problem.    "What's the next step in a 125-step semiconductor process?"
[0:15 – 0:45]  Baseline.   trigram → plausible-looking but rule-violating prediction
[0:45 – 1:30]  Our system. neuro-symbolic stack → process-logic-valid prediction + reasoning visible
[1:30 – 2:00]  Headline.   ONE number: "100% validator-clean completions across 600 test sequences"
```

This maps directly to the rubric's 4 bullets.

---

## 3. Second-by-second storyboard

### Beat 1 — Problem (0:00 – 0:15) — 15 sec

**Visual:**
- Terminal open, large font.
- Type or paste a real prefix from `competition/participant-files/eval_input_valid.csv`. Use a MOSFET prefix mid-process (something with `IMPLANT WELL → DRIVE IN DIFFUSION → RAPID THERMAL ANNEAL` in the last few visible steps so the next-step question is non-trivial).
- Show the prefix on screen for ~5 seconds, then highlight the `→ ?` at the end.

**Voiceover (read in ~12 seconds):**
> *"In semiconductor manufacturing, a wafer goes through 125 process steps in a specific order. Get the order wrong and the chip is dead. The question: given the first 60 steps, what's the next one? And does the model know the WHY — or just the surface pattern?"*

### Beat 2 — Baseline (0:15 – 0:45) — 30 sec

**Visual:**
- Run `python shared/scripts/demo_compare.py --example mosfet-mid` (it's already wired up). The output shows trigram + grammar-trigram + transformer side-by-side.
- Highlight the **trigram-with-backoff** row — Top-5 = 0.993 ID, but on LoFO drops to 0.472.
- Show one concrete example where trigram suggests a step that's locally probable but doesn't fit the process logic.

**Voiceover:**
> *"A 50-line trigram-with-backoff already hits Top-5 = 99.3% on in-distribution next-step prediction. So the task LOOKS solved. But it isn't. Watch what happens when we hold out an entire product family the model has never seen. Trigram drops to 47% Top-1. It memorised the training distribution; it didn't learn the process logic."*

**On-screen overlay at end of beat (text card, 2-3 sec):**
> *"Trigram LoFO Top-1: 0.472"*
> *"What about a learned model?"*

### Beat 3 — Our system (0:45 – 1:30) — 45 sec

**Visual:**
- Same `demo_compare.py` output, now zoom on the trained-transformer + multitask-transformer rows.
- For an `anomaly-mosfet` example, run:
  ```
  python shared/scripts/demo_compare.py --example anomaly-mosfet
  ```
- The output shows:
  - The corrupted sequence (with `PARAMETRIC TEST → DEPOSIT PASSIVATION` — a `RULE_TEST_BEFORE_PASSIVATION` violation)
  - The validator flags it with the specific rule
  - The transformer's anomaly head agrees
- Underneath: a quick flash of `shared/extras/plots/report/trajectory.png` showing the trajectory from trigram (0.47) → our v3 (0.658) on LoFO held-out.

**Voiceover:**
> *"Our approach: a neuro-symbolic stack. A grammar-mask filter built from the 10 documented process rules. A compositional Transformer trained with a validity head and a rule-attribution head. And — the key piece — a leave-one-family-out training regime that proves the model isn't memorising. On held-out families, our top-1 stays at 65.8% — a 1.3 percentage-point drop from in-distribution. Trigram drops 25. And the system tells you WHICH rule was violated, not just that something's wrong."*

### Beat 4 — Headline (1:30 – 2:00) — 30 sec

**Visual:**
- Run `python shared/scripts/validate_completions.py --predictions shared/extras/results/submission_v3_real/completion.csv` live.
- Output shows:
  ```
  Overall: 600/600 (100.0%) completions are validator-clean
  ```
- Flash the trajectory plot one more time with the final number visible.

**Voiceover:**
> *"Across all 600 test sequences spanning three product families, every completion our system produces is process-logic-valid. Not 'mostly right' — every single one passes the organisers' own ten-rule validator. That's what learning process logic looks like — not just predicting the next step, but understanding why."*

**Closing card (last 2 sec):**
> *"Team abb — Zero One Hack_01 — Industrial AI (Infineon)"*

---

## 4. Voiceover script (clean copy, ~280 words for 2 minutes)

> *"In semiconductor manufacturing, a wafer goes through 125 process steps in a specific order. Get the order wrong and the chip is dead. The question: given the first 60 steps, what's the next one? And does the model know the WHY — or just the surface pattern?*
>
> *A 50-line trigram-with-backoff already hits Top-5 = 99.3% on in-distribution next-step prediction. So the task LOOKS solved. But it isn't. Watch what happens when we hold out an entire product family the model has never seen. Trigram drops to 47% Top-1. It memorised the training distribution; it didn't learn the process logic.*
>
> *Our approach: a neuro-symbolic stack. A grammar-mask filter built from the 10 documented process rules. A compositional Transformer trained with a validity head and a rule-attribution head. And — the key piece — a leave-one-family-out training regime that proves the model isn't memorising. On held-out families, our top-1 stays at 65.8% — a 1.3 percentage-point drop from in-distribution. Trigram drops 25. And the system tells you WHICH rule was violated, not just that something's wrong.*
>
> *Across all 600 test sequences spanning three product families, every completion our system produces is process-logic-valid. Not 'mostly right' — every single one passes the organisers' own ten-rule validator. That's what learning process logic looks like — not just predicting the next step, but understanding why."*

Word count: ~280. At 140 words per minute speaking pace, that's exactly 2 minutes. Tight.

---

## 5. Visuals/assets we already have

| File | Used in beat |
|---|---|
| `shared/scripts/demo_compare.py` (live execution) | Beats 2 + 3 (the main content) |
| `shared/scripts/validate_completions.py` (live execution) | Beat 4 (the headline) |
| `shared/extras/plots/report/trajectory.png` | Beat 3 (transition) + Beat 4 (closing) |
| `shared/extras/plots/report/max_len_fix.png` | (optional, if we have 5 sec to spare in Beat 3) |
| `shared/extras/plots/report/submission_quality.png` | (optional alternative for Beat 4) |
| `competition/participant-files/eval_input_valid.csv` | Beat 1 (the real prefix to demo on) |

**We don't need to make any new plots or videos.** Everything is in the repo.

---

## 6. How to record (concrete tools)

### Option A — macOS native (zero install)

1. `Cmd+Shift+5` → choose "Record Selected Portion" → drag to cover the terminal window.
2. Click "Options" → enable Microphone (Built-in).
3. Click "Record". Run the demo. Stop with `Cmd+Shift+5` again.
4. Open the .mov in QuickTime → Export As → 1080p → save as .mov (or convert to .mp4 with `ffmpeg -i input.mov -c:v libx264 -crf 18 -c:a aac output.mp4`).

### Option B — Loom (browser, easy, auto-uploads)

1. Sign in to loom.com.
2. Start a screen recording with mic.
3. Run the demo. Stop.
4. Get the unlisted share link. Paste into the Tally form.

### Option C — OBS Studio (most control)

1. Install OBS (https://obsproject.com).
2. Add a Display Capture source for the terminal area.
3. Add an Audio Input Capture source for the mic.
4. Record. Output is .mkv → convert with `ffmpeg`.

### Tips regardless of tool

- **Terminal font size 18-20pt minimum**. Jury watches on a projector.
- **Use a dark theme** with good contrast. Solarized Dark or Dracula.
- **Practice the demo run twice** before recording — knows-where-the-cursor-needs-to-be matters.
- **Speak slower than feels natural.** 140 wpm is the upper bound.
- **Cut all dead air.** A 2-min limit is a 2-min limit.
- **Don't show the README or repo structure.** That's slideware. The repo is the README.

---

## 7. Pre-flight checklist (before hitting Record)

- [ ] `shared/scripts/demo_compare.py` runs cleanly with `--example mosfet-mid` and `--example anomaly-mosfet`
- [ ] `shared/scripts/validate_completions.py` outputs `600/600 (100.0%)` against the v3 submission
- [ ] Terminal font is readable at 1080p
- [ ] Microphone tested (not the AirPod mic with wind — use built-in or a desk mic)
- [ ] Background quiet (no fan, no chat notifications)
- [ ] Voiceover script in front of you (printed or on a second screen)
- [ ] Trajectory plot open in a viewer for the cuts
- [ ] First-take recording — try a dry run first to see timing
- [ ] Stopwatch on a second device to make sure you don't exceed 2 minutes

---

## 8. Post-recording

1. Trim any dead air at start / end.
2. Verify the final file is **under 2 minutes** (hard cutoff per the rules).
3. Convert to MP4 at 1080p if not already (`ffmpeg -i input.mov -c:v libx264 -crf 18 -c:a aac demo.mp4`).
4. Upload to Loom / YouTube unlisted OR drag into the Tally form directly.
5. Paste the link into the Tally form field `Demo video`.

---

## 9. Fallback options if recording fails

If recording quality is bad or you run out of time:

- **Loom in a browser** is the fastest path. Talk over the demo. Done in 5 minutes of total work.
- **Static screenshots with a voiceover slide** is *worse* per the rubric but acceptable as a last resort.
- **Asciicast (asciinema.org)** is acceptable but harder to add voiceover. Skip unless desperate.

---

## 10. Single command to do the entire demo (for reference)

If we automate this for retakes:

```bash
# Beat 2 — baseline
python shared/scripts/demo_compare.py --example mosfet-mid --n-completion 8

# Beat 3 — anomaly attribution
python shared/scripts/demo_compare.py --example anomaly-mosfet --n-completion 5

# Beat 4 — headline
python shared/scripts/validate_completions.py \
    --eval-input competition/participant-files/eval_input_valid.csv \
    --predictions shared/extras/results/submission_v3_real/completion.csv
```

Each of these takes ~3-5 seconds of terminal output. Plus voiceover.

---

*Designed for the abb-branch submission. If the team picks neurosymbolic or main's SSL Transformer as the final ship, swap the headline number in Beat 3 + the submission CSV path in Beat 4 accordingly.*
