# Attention Seekers, Industrial AI submission site

A self-contained Next.js site that presents the team's submission report for the
Industrial AI track (Infineon) of Zero One Hack_01: learning and benchmarking
process logic in semiconductor fabrication routes.

The report body is authored in **Markdown MDX** (`src/content/report.mdx`) so the
prose stays readable and figures, tables, and code blocks drop straight into the
content. It is a replica of the repository's `SUBMISSION.md`, with every
illustration included; the figures are produced by `shared/benchmark/report.py`
from the official scorer output.

## Tech stack

- **Next.js 16** with the App Router, React 19, TypeScript strict
- **MDX** (`@next/mdx`) for the article content, with `rehype-slug` for heading anchors
- **Tailwind CSS v4** with a small custom token palette
- **Geist Sans / Mono** fonts

## Commands

```bash
bun install        # or: npm install
bun run dev        # start the dev server
bun run build      # production build
bun run lint       # ESLint
bun run typecheck  # TypeScript
bun run check      # lint + typecheck + build
```

## Project structure

```
src/
  app/              # layout, page, globals.css, sitemap, robots
  content/          # report.mdx (the report source)
  components/       # kebab-case component files
    article-hero.tsx
    citation-block.tsx   # code blocks
    data-table.tsx       # result tables
    figure.tsx           # figures with captions
    site-header.tsx
    site-footer.tsx
    table-of-contents.tsx
    icons.tsx
    ui/                  # shadcn primitives
  lib/              # cn() utility + site constants
mdx-components.tsx  # maps markdown elements to styled reading-column primitives
public/
  images/           # benchmark figures (fig1..fig7)
  zeroone.png       # hero image
```

## License

MIT
