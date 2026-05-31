import Image from "next/image";
import { ARTICLE, FIGURE_MAX_WIDTH } from "@/lib/constants";

export function ArticleHero() {
  return (
    <header className="flex flex-col items-center gap-8 text-center mt-12">
      <div className="flex flex-col items-center gap-8 text-center">
        <p className="font-sans text-[15px] font-bold leading-[21px] text-ink">
          Industrial AI (Infineon) · Zero One Hack_01
        </p>
        <h1 className="font-sans text-[32px] font-bold leading-[1.1] text-ink lg:text-[52px] lg:leading-[57.2px] max-w-3xl">
          Learning and Benchmarking Process Logic in Semiconductor Fabrication Routes
        </h1>
        <div>
          <span className="font-sans text-[15px] leading-[21px] text-ink">
            Team Attention Seekers · May 31, 2026
          </span>
        </div>
      </div>

      <div
        className={`relative mx-auto aspect-[1200/630] w-full ${FIGURE_MAX_WIDTH} overflow-hidden rounded-[24px]`}
      >
        <Image
          src="/zeroone.png"
          alt={ARTICLE.title}
          fill
          priority
          sizes="(max-width: 820px) 100vw, 820px"
          className="object-cover"
        />
      </div>
    </header>
  );
}
