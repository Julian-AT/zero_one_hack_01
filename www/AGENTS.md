<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

# Economic Index — Learning Curves

A faithful rebuild of the Anthropic Economic Index report "Learning curves" as a
self-contained Next.js site. Article content lives in Markdown MDX so future interactive
charts drop straight into the prose.

## Tech Stack
- **Framework:** Next.js 16 (App Router, React 19, TypeScript strict)
- **Content:** MDX via `@next/mdx` (`src/content/*.mdx`)
- **UI:** shadcn/ui primitives styled to the Anthropic palette, Tailwind CSS v4
- **Fonts:** self-hosted AnthropicSans / Serif / Mono (`public/fonts/`)

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
  app/            # layout, page, globals.css
  content/        # article MDX (markdown)
  components/     # kebab-case component files
  lib/            # cn() utility + constants
mdx-components.tsx
public/
  fonts/  images/  seo/
```
