"use client";

import {
  Sheet,
  SheetContent,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { GitHubIcon, MenuIcon } from "@/components/icons";
import { GITHUB_URL } from "@/lib/constants";
import { cn } from "@/lib/utils";

const navItems: { label: string; href: string }[] = [
  { label: "Summary", href: "#summary" },
  { label: "Approach", href: "#approach" },
  { label: "Results", href: "#results" },
  { label: "What worked", href: "#what-worked" },
  { label: "A note on honesty", href: "#honesty" },
];

export function MobileMenu({ className }: { className?: string }) {
  return (
    <Sheet>
      <SheetTrigger
        aria-label="Open menu"
        className={cn("lg:hidden", className)}
      >
        <MenuIcon className="h-10 w-10 text-ink" />
      </SheetTrigger>
      <SheetContent side="right" className="w-[280px] bg-cream">
        <SheetTitle className="sr-only">Navigation</SheetTitle>
        <nav aria-label="Mobile" className="mt-12 flex flex-col">
          {navItems.map((item) => (
            <a
              key={item.label}
              href={item.href}
              className="border-rule border-b py-4 font-sans text-[16px] text-ink-soft transition-opacity hover:opacity-70"
            >
              {item.label}
            </a>
          ))}
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="View source on GitHub"
            className="mt-6 inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-ink-soft px-4 font-sans text-[16px] text-ink-soft transition-opacity hover:opacity-90"
          >
            <GitHubIcon className="h-5 w-5" />
            GitHub
          </a>
        </nav>
      </SheetContent>
    </Sheet>
  );
}
