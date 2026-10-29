import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ARC-AGI-3 Comparison",
  description: "Compare LLM, classical QAHS, and quantum QAHS experiment runs.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
