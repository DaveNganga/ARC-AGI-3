import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Quantum-Assisted Hypothesis Search",
  description: "Testing whether quantum-assisted hypothesis selection can strengthen AI reasoning.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <nav className="site-nav" aria-label="Primary navigation">
          <Link className="brand" href="/">QAHS</Link>
          <div><Link href="/">About</Link><Link href="/results">Results</Link></div>
        </nav>
        {children}
      </body>
    </html>
  );
}
