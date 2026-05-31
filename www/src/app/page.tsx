import { SiteHeader } from "@/components/site-header";
import { ArticleHero } from "@/components/article-hero";
import Article from "@/content/learning-curves.mdx";
import { TableOfContents } from "@/components/table-of-contents";
import { ARTICLE, SITE_URL, TOC_SECTIONS } from "@/lib/constants";

const articleJsonLd = {
  "@context": "https://schema.org",
  "@type": "Article",
  headline: ARTICLE.title,
  description: ARTICLE.description,
  datePublished: ARTICLE.datePublished,
  author: ARTICLE.authors.map((name) => ({ "@type": "Person", name })),
  publisher: {
    "@type": "Organization",
    name: "Anthropic",
  },
  mainEntityOfPage: SITE_URL,
};

export default function Home() {
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(articleJsonLd) }}
      />
      <SiteHeader />
      <main id="main-content">
        <div className="mx-auto w-full max-w-[1400px] px-8 lg:px-16">
          <ArticleHero />
          <div className="flex">
            <div className="w-0 shrink-0">
              <TableOfContents sections={[...TOC_SECTIONS]} />
            </div>
            <article className="mt-12 min-w-0 flex-1 pb-8">
              <Article />
            </article>
          </div>
        </div>
      </main>
    </>
  );
}
