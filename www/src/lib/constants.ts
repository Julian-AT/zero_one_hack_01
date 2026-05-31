export const PROSE_MAX_WIDTH = "max-w-[775px]";
export const FIGURE_MAX_WIDTH = "max-w-[820px]";

export const GITHUB_URL =
  "https://github.com/Julian-AT/zero_one_hack_01";

export const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

export const ARTICLE_URL =
  "https://www.anthropic.com/research/economic-index-march-2026-report";

export const ARTICLE = {
  title: "A Neurosymbolic Stack for Semiconductor Process-Flow Sequence Modeling and OOD Generalization",
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
  { id: "what-has-changed-since-our-last-report", label: "What has changed since our last report" },
  { id: "learning-to-use-ai", label: "Learning to use AI" },
  { id: "discussion", label: "Discussion" },
  { id: "appendix", label: "Appendix" },
  { id: "authors-and-acknowledgements", label: "Authors and acknowledgements" },
] as const;
