import { Fragment, type ReactNode } from "react";
import { FIGURE_MAX_WIDTH } from "@/lib/constants";

interface DataTableProps {
  head: string[];
  rows: string[][];
  label?: string;
  caption?: string;
}

function renderCell(value: string): ReactNode {
  const parts = value.split(/(\*[^*]+\*)/g).filter(Boolean);
  return parts.map((part, i) => {
    if (part.startsWith("*") && part.endsWith("*")) {
      return (
        <strong key={i} className="font-semibold text-ink">
          {part.slice(1, -1)}
        </strong>
      );
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

export function DataTable({ head, rows, label, caption }: DataTableProps) {
  return (
    <figure className={`mx-auto my-12 flex w-full ${FIGURE_MAX_WIDTH} flex-col`}>
      <div className="overflow-x-auto rounded-2xl border border-code-border">
        <table className="w-full border-collapse font-sans text-[14px] leading-[1.4]">
          <thead>
            <tr className="border-b border-code-border bg-code-bg">
              {head.map((cell, i) => (
                <th
                  key={i}
                  scope="col"
                  className={`px-4 py-3 font-semibold text-ink ${
                    i === 0 ? "text-left" : "text-right"
                  }`}
                >
                  {renderCell(cell)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, r) => (
              <tr
                key={r}
                className="border-b border-code-border/60 last:border-b-0"
              >
                {row.map((cell, c) => (
                  <td
                    key={c}
                    className={`px-4 py-2.5 text-ink ${
                      c === 0 ? "text-left font-medium" : "text-right tabular-nums"
                    }`}
                  >
                    {renderCell(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(label || caption) && (
        <figcaption className="mt-2 font-sans text-[14px] leading-[16.8px] tracking-[0.15px] text-muted-ink">
          {label && <strong className="font-semibold italic">{label} </strong>}
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
