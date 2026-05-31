import { ArrowRightIcon } from "@/components/icons";
import { cn } from "@/lib/utils";

const items: { title: string; description: string; href: string }[] = [
  {
    title: "Coding agents in the social sciences",
    description:
      "Results from a survey of 1,260 social scientists about AI and coding agent use.",
    href: "https://www.anthropic.com/research/coding-agents-social-sciences",
  },
  {
    title: "Project Glasswing: An initial update",
    description:
      "An early update on what we've learned from Project Glasswing.",
    href: "https://www.anthropic.com/research/glasswing-initial-update",
  },
  {
    title: "2028: Two scenarios for global AI leadership",
    description: "Our views on the AI competition between the US and China.",
    href: "https://www.anthropic.com/research/2028-ai-leadership",
  },
];

export function RelatedContent({ className }: { className?: string }) {
  return (
    <section className={cn(className)}>
      <h2
        className="font-sans text-[32px] font-semibold leading-[38.4px] text-ink mb-8"
      >
        Related content
      </h2>
      <div className="flex flex-col gap-8 lg:flex-row">
        {items.map((item) => (
          <div key={item.href} className="flex flex-1 flex-col gap-2">
            <h3 className="font-sans text-[19px] font-semibold leading-[22.8px] text-ink">
              {item.title}
            </h3>
            <p className="font-serif text-[15px] leading-5 text-ink">
              {item.description}
            </p>
            <a
              href={item.href}
              className="group mt-2 inline-flex items-center gap-2 font-serif text-base text-muted-ink transition-colors duration-150 hover:text-ink"
            >
              Read more
              <ArrowRightIcon className="h-5 w-5 transition-transform duration-150 group-hover:translate-x-1" />
            </a>
          </div>
        ))}
      </div>
    </section>
  );
}
