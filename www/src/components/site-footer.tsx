import Link from "next/link";
import { Logomark, GitHubIcon } from "@/components/icons";
import { GITHUB_URL } from "@/lib/constants";

type FooterColumn = {
  heading: string;
  links: { label: string; href: string }[];
};

const columns: FooterColumn[] = [
  {
    heading: "Report",
    links: [
      { label: "Summary", href: "#summary" },
      { label: "Results", href: "#results" },
      { label: "What worked", href: "#what-worked" },
      { label: "A note on honesty", href: "#a-note-on-honesty" },
    ],
  },
  {
    heading: "Systems",
    links: [
      { label: "Transformer xLSTM", href: "#transformer-xlstm" },
      { label: "Self supervised hybrid", href: "#self-supervised-hybrid" },
      { label: "Neurosymbolic engine", href: "#neurosymbolic-engine" },
      { label: "Benchmark", href: "#results" },
    ],
  },
  {
    heading: "Project",
    links: [
      { label: "GitHub repository", href: GITHUB_URL },
      { label: "How to run it", href: "#how-to-run-it" },
      { label: "Deliverables", href: "#deliverables" },
    ],
  },
];

export function SiteFooter() {
  return (
    <footer id="footer" className="w-full bg-ink font-sans text-cream">
      <div className="mx-auto w-full max-w-[1400px] px-8 py-16 lg:px-16">
        <div className="flex flex-col gap-12 lg:flex-row lg:justify-between">
          <div className="flex flex-col gap-6">
            <Link
              href="/"
              aria-label="Attention Seekers home"
              className="inline-flex items-center gap-2.5"
            >
              <Logomark className="h-7 w-7" />
              <span className="text-[16px] font-semibold tracking-tight">
                Attention Seekers
              </span>
            </Link>
            <p className="max-w-xs text-[14px] leading-relaxed text-cream/70">
              Learning and benchmarking process logic in semiconductor
              fabrication routes. Industrial AI track, Zero One Hack_01.
            </p>
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              aria-label="View source on GitHub"
              className="inline-flex w-fit items-center gap-2 text-cream/70 transition-colors hover:text-cream"
            >
              <GitHubIcon className="h-5 w-5" />
              <span className="text-[14px]">Source on GitHub</span>
            </a>
            <p className="mt-auto text-[13px] text-cream/60">
              &copy; 2026 Team Attention Seekers
            </p>
          </div>

          <nav
            aria-label="Footer"
            className="grid grid-cols-2 gap-x-12 gap-y-10 sm:grid-cols-3 lg:gap-x-16"
          >
            {columns.map((col) => (
              <div key={col.heading}>
                <h3 className="mb-4 text-[13px] font-semibold uppercase tracking-wide text-cream/60">
                  {col.heading}
                </h3>
                <ul className="space-y-3">
                  {col.links.map((link) => (
                    <li key={`${col.heading}-${link.label}`}>
                      <a
                        href={link.href}
                        className="text-[14px] text-cream/90 transition-colors hover:text-cream hover:underline"
                      >
                        {link.label}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </nav>
        </div>
      </div>
    </footer>
  );
}
