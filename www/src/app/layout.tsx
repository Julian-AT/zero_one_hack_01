import type { Metadata } from "next";
import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import { ARTICLE, SITE_URL } from "@/lib/constants";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: `${ARTICLE.title} \\ Anthropic`,
  description: ARTICLE.description,
  alternates: {
    canonical: "/",
  },
  openGraph: {
    title: ARTICLE.title,
    description: ARTICLE.description,
    url: "/",
    type: "article",
    publishedTime: ARTICLE.datePublished,
    images: [
      {
        url: "/zeroone.png",
        width: 1200,
        height: 630,
        alt: ARTICLE.title,
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: ARTICLE.title,
    description: ARTICLE.description,
    images: ["/zeroone.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${GeistSans.variable} ${GeistMono.variable} h-full`}
    >
      <body className={`${GeistSans.className} min-h-full`}>{children}</body>
    </html>
  );
}
