import Link from "next/link";
import {
  AnthropicWordmark,
  LinkedInIcon,
  TwitterIcon,
  YouTubeIcon,
} from "@/components/icons";

type FooterColumn = {
  heading: string;
  links: { label: string; href: string }[];
};

const columns: FooterColumn[] = [
  {
    heading: "Product",
    links: [
      { label: "Claude", href: "https://claude.com/product/overview" },
      { label: "Claude Code", href: "https://claude.com/product/claude-code" },
      { label: "Claude Developer Platform", href: "https://claude.com/platform/api" },
      { label: "Pricing", href: "https://claude.com/pricing" },
      { label: "Download app", href: "https://claude.ai/download" },
    ],
  },
  {
    heading: "Research",
    links: [
      { label: "Economic Index", href: "https://www.anthropic.com/economic-index" },
      { label: "Research overview", href: "https://www.anthropic.com/research" },
      { label: "Economic Futures", href: "https://www.anthropic.com/economic-futures" },
      { label: "News", href: "https://www.anthropic.com/news" },
    ],
  },
  {
    heading: "Company",
    links: [
      { label: "About", href: "https://www.anthropic.com/company" },
      { label: "Careers", href: "https://www.anthropic.com/careers" },
      { label: "Transparency", href: "https://www.anthropic.com/transparency" },
      { label: "Security and compliance", href: "https://trust.anthropic.com/" },
    ],
  },
  {
    heading: "Terms and policies",
    links: [
      { label: "Privacy policy", href: "https://www.anthropic.com/legal/privacy" },
      { label: "Usage policy", href: "https://www.anthropic.com/legal/aup" },
      { label: "Terms of service", href: "https://www.anthropic.com/legal/consumer-terms" },
      { label: "Responsible disclosure", href: "https://www.anthropic.com/responsible-disclosure-policy" },
    ],
  },
];

const socials = [
  {
    href: "https://www.linkedin.com/company/anthropicresearch",
    label: "LinkedIn",
    Icon: LinkedInIcon,
  },
  { href: "https://x.com/AnthropicAI", label: "X (formerly Twitter)", Icon: TwitterIcon },
  { href: "https://www.youtube.com/@anthropic-ai", label: "YouTube", Icon: YouTubeIcon },
];

export function SiteFooter() {
  return (
    <footer id="footer" className="w-full bg-ink font-sans text-cream">
      <div className="mx-auto w-full max-w-[1400px] px-8 py-16 lg:px-16">
        <div className="flex flex-col gap-12 lg:flex-row lg:justify-between">
          <div className="flex flex-col gap-6">
            <Link href="/" aria-label="Anthropic home" className="inline-block">
              <AnthropicWordmark className="h-4 w-auto" />
            </Link>
            <div className="flex items-center gap-4">
              {socials.map(({ href, label, Icon }) => (
                <a
                  key={href}
                  href={href}
                  aria-label={label}
                  className="text-cream/70 transition-colors hover:text-cream"
                >
                  <Icon className="h-5 w-5" />
                </a>
              ))}
            </div>
            <p className="mt-auto text-[13px] text-cream/60">&copy; 2026 Anthropic PBC</p>
          </div>

          <nav
            aria-label="Footer"
            className="grid grid-cols-2 gap-x-12 gap-y-10 sm:grid-cols-4 lg:gap-x-16"
          >
            {columns.map((col) => (
              <div key={col.heading}>
                <h3 className="mb-4 text-[13px] font-semibold uppercase tracking-wide text-cream/60">
                  {col.heading}
                </h3>
                <ul className="space-y-3">
                  {col.links.map((link) => (
                    <li key={link.href}>
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
