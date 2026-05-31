"use client";

import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

interface Section {
  id: string;
  label: string;
}

const SCROLL_THRESHOLD = 130;

export function TableOfContents({ sections }: { sections: Section[] }) {
  const [activeId, setActiveId] = useState<string>(sections[0]?.id ?? "");

  useEffect(() => {
    if (sections.length === 0) return;

    let frame = 0;

    const updateActive = () => {
      frame = 0;
      let current = sections[0]?.id ?? "";
      for (const { id } of sections) {
        const el = document.getElementById(id);
        if (!el) continue;
        if (el.getBoundingClientRect().top <= SCROLL_THRESHOLD) current = id;
        else break;
      }
      setActiveId(current);
    };

    const onScroll = () => {
      if (frame) return;
      frame = requestAnimationFrame(updateActive);
    };

    updateActive();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [sections]);

  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    e.preventDefault();
    setActiveId(id);
    document
      .getElementById(id)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <aside className="sticky top-[92px] mt-10 hidden w-[172px] flex-col self-start xl:flex">
      {sections.map(({ id, label }) => {
        const isActive = activeId === id;
        return (
          <a
            key={id}
            href={`#${id}`}
            aria-current={isActive ? "true" : undefined}
            onClick={(e) => handleClick(e, id)}
            className={cn(
              "w-full border-b border-spacing-0.5 py-3 text-left font-sans text-[12px] font-medium leading-[14.4px] tracking-[0.15px] transition-colors duration-150",
              isActive ? "font-bold text-ink" : "text-muted-ink hover:text-ink",
            )}
          >
            {label}
          </a>
        );
      })}
    </aside>
  );
}
