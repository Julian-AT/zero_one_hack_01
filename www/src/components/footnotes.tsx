import { TwitterIcon, LinkedInIcon } from "@/components/icons";
import { ARTICLE_URL, PROSE_MAX_WIDTH } from "@/lib/constants";

const footnotes = [
  {
    id: 1,
    content: (
      <>
        &ldquo;First-party API&rdquo; or 1P API refers to developer traffic routed
        directly through Anthropic&rsquo;s own programming interface, which is
        distinct from both Anthropic&rsquo;s consumer-facing Claude.ai application and
        third-party platforms such as Amazon Bedrock or Google Cloud Vertex.
      </>
    ),
  },
  {
    id: 2,
    content: <>This includes data from Claude Code.</>,
  },
  {
    id: 3,
    content: (
      <>
        This number uses 2019 O*NET-SOC codes, while previous reports use the 2010
        vintage.
      </>
    ),
  },
  {
    id: 4,
    content: (
      <>
        The drop in coursework conversations was 5 percentage points in countries
        where the school term was active and 12 percentage points in the countries
        where most students were on break.
      </>
    ),
  },
  {
    id: 5,
    content: <>See the Appendix for definitions of the interaction types.</>,
  },
  {
    id: 6,
    content: (
      <>
        For example, the task &ldquo;Compute moisture or salt content, percentages of
        ingredients, formulas, or other product factors, using mathematical and
        chemical procedures.&rdquo; is done only by Food Science Technicians, who have
        an average wage of $26.15, so this is the value of that task. The data source
        for this exercise is the{" "}
        <a
          href="https://www.bls.gov/oes/tables.htm"
          target="_blank"
          rel="noopener noreferrer"
          className="underline decoration-[1.36px] underline-offset-[3.06px] transition-opacity hover:opacity-70"
        >
          May 2024 BLS Occupational Employment and Wage Statistics (OEWS) Tables
        </a>
        . When multiple workers do the same task, we average their wages weighting by
        employment and the fraction of time spent on that task.
      </>
    ),
  },
  {
    id: 7,
    content: (
      <>
        To find the emerging patterns, we filtered for O*NET tasks that (i) appeared
        at least 300 times in the current data and (ii) showed at least 2x growth
        compared to the previous report.
      </>
    ),
  },
  {
    id: 8,
    content: (
      <>
        The range is given to reflect the different estimates from running the model
        in our{" "}
        <a
          href="https://www-cdn.anthropic.com/096d94c1a91c6480806d8f24b2344c7e2a4bc666.pdf"
          target="_blank"
          rel="noopener noreferrer"
          className="underline decoration-[1.36px] underline-offset-[3.06px] transition-opacity hover:opacity-70"
        >
          previous report
        </a>{" "}
        with (5 years) or without (9 years) weights.
      </>
    ),
  },
  {
    id: 9,
    content: (
      <>
        In this analysis, we use log-level data to estimate the models with the same
        privacy thresholds. See the Appendix for more on the methodology.
      </>
    ),
  },
  {
    id: 10,
    content: <>These results are similar however we define high tenure.</>,
  },
  {
    id: 11,
    content: (
      <>
        Our sampling period overlapped with the release of our Super Bowl
        advertisements, which brought many first-time users.
      </>
    ),
  },
];

export function ArticleFootnotes() {
  return (
    <section className={`mx-auto w-full ${PROSE_MAX_WIDTH}`}>
      <h4 className="mt-8 mb-2 font-sans text-[25px] font-semibold leading-[30px] text-ink">
        Footnotes
      </h4>
      <ol className="list-decimal pl-5 font-serif text-[16px] leading-[24.8px] text-ink marker:text-ink/60">
        {footnotes.map(({ id, content }) => (
          <li key={id} id={`footnote-${id}`} className="mb-2 scroll-mt-28 pl-1">
            {content}{" "}
            <a
              href={`#footnote-ref-${id}`}
              className="ml-1 font-sans text-[12px] text-muted-ink no-underline transition-opacity hover:opacity-70"
            >
              ↩
            </a>
          </li>
        ))}
      </ol>
      <div className="mt-8 flex items-center gap-3 border-t border-ink pt-8">
        <a
          href={`https://twitter.com/intent/tweet?text=${encodeURIComponent(ARTICLE_URL)}`}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Share on Twitter"
          className="text-ink transition-opacity hover:opacity-70"
        >
          <TwitterIcon className="h-5 w-5" />
        </a>
        <a
          href={`https://www.linkedin.com/shareArticle?mini=true&url=${encodeURIComponent(ARTICLE_URL)}`}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Share on LinkedIn"
          className="text-ink transition-opacity hover:opacity-70"
        >
          <LinkedInIcon className="h-5 w-5" />
        </a>
      </div>
    </section>
  );
}
