<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

# Attention Seekers, Industrial AI submission site

A self-contained Next.js site presenting the team's submission report for the
Industrial AI track (Infineon) of Zero One Hack_01. Article content lives in
Markdown MDX (`src/content/report.mdx`) so figures, tables, and code blocks drop
straight into the prose.

## Tech Stack
- **Framework:** Next.js 16 (App Router, React 19, TypeScript strict)
- **Content:** MDX via `@next/mdx` (`src/content/*.mdx`), `rehype-slug` for anchors
- **UI:** shadcn/ui primitives, Tailwind CSS v4
- **Fonts:** Geist Sans / Mono

## Commands
- `bun run dev` — Start dev server
- `bun run build` — Production build
- `bun run lint` — ESLint check
- `bun run typecheck` — TypeScript check
- `bun run check` — Run lint + typecheck + build

## Code Style
- TypeScript strict mode, no `any`
- Named exports, PascalCase components, camelCase utils
- Component files use kebab-case (`site-header.tsx`)
- Tailwind utility classes, no inline styles
- 2-space indentation, mobile-first responsive
- No comments in source files

## Project Structure
```
src/
  app/            # layout, page, globals.css, sitemap, robots
  content/        # report.mdx (the report source)
  components/     # kebab-case component files
  lib/            # cn() utility + site constants
mdx-components.tsx
public/
  images/  zeroone.png
```
