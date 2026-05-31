# Anthropic Economic Index — Learning Curves

A faithful, self-contained Next.js rebuild of the [Anthropic Economic Index report: Learning curves](https://www.anthropic.com/research/economic-index-march-2026-report) research article.

The article body is authored in **Markdown MDX** so prose stays readable and future interactive charts drop straight into the content. Layout, typography, and color match the source page.

## Tech Stack

- **Next.js 16** — App Router, React 19, TypeScript strict
- **MDX** (`@next/mdx`) — article content as markdown with embedded components
- **shadcn/ui** — Button, Sheet on Tailwind v4 + the `cn()` utility
- **Tailwind CSS v4** — CSS-first config with an Anthropic-brand token palette
- Self-hosted Anthropic Sans / Serif / Mono woff2 fonts

## Commands

### useful

```bash
bun run dev        # Start dev server
bun run build      # Production build
bun run lint       # ESLint
bun run typecheck  # TypeScript
bun run check      # lint + typecheck + build
```

## Project Structure

```
src/
  app/              # layout, page, globals.css
  content/          # learning-curves.mdx (article source)
  components/       # kebab-case component files
    article-hero.tsx
    citation-block.tsx
    figure.tsx
    footnotes.tsx
    mobile-menu.tsx
    related-content.tsx
    site-footer.tsx
    site-header.tsx
    table-of-contents.tsx
    icons.tsx
    ui/             # shadcn primitives
  lib/              # cn(), constants
mdx-components.tsx  # maps markdown elements to styled reading-column primitives
public/
  images/           # figures and tables
  fonts/            # Anthropic woff2 files
  seo/              # favicons
```

## License

MIT
