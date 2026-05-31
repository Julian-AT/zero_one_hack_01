export const PROSE_MAX_WIDTH = "max-w-[775px]";
export const FIGURE_MAX_WIDTH = "max-w-[820px]";

export const GITHUB_URL =
  "https://github.com/Julian-AT/zero_one_hack_01";

export const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? "https://zeroone.julianschmidt.cv";

export const ARTICLE_URL =
  "https://github.com/Julian-AT/zero_one_hack_01";

export const ARTICLE = {
  title: "Learning and Benchmarking Process Logic in Semiconductor Fabrication Routes",
  description:
    "A hybrid neurosymbolic system for modeling semiconductor manufacturing process-flow sequences, developed for the Industrial AI track (Infineon) of Zero One Hack_01. The work combines a symbolic rule validator, a grammar-constrained decoder, k-nearest-neighbor retrieval, and a compositionally-tokenized multi-task decoder-only transformer (RoPE, RMSNorm, SwiGLU) with auxiliary validity and rule-attribution heads. It addresses next-step prediction, sequence completion, and anomaly detection with rule attribution, and targets out-of-distribution generalization to an unseen product family via leave-one-family-out evaluation and parsed physics-parameter features. Models were trained from scratch on the EuroHPC Leonardo cluster (CINECA).",
  datePublished: "2026-05-31",
  authors: [
    "Julian Schmidt",
    "Abdul Basit Banbhan",
    "Kyrillus Mehanni",
    "Emil Kasper"
  ],
} as const;

export const TOC_SECTIONS = [
  { id: "summary", label: "Summary" },
  { id: "problem", label: "Problem" },
  { id: "approach", label: "Approach" },
  { id: "how-to-run-it", label: "How to run it" },
  { id: "results", label: "Results" },
  { id: "what-worked", label: "What worked" },
  { id: "what-did-not-work", label: "What did not work" },
  { id: "another-36-hours", label: "Another 36 hours" },
  { id: "deliverables", label: "Deliverables" },
  { id: "a-note-on-honesty", label: "A note on honesty" },
] as const;
