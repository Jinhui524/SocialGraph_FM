import Markdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

function safeUrlTransform(url: string): string {
  const trimmed = url.trim();
  if (trimmed.startsWith("//") || trimmed.startsWith("\\\\")) return "";
  if (trimmed.startsWith("#")) return trimmed;
  if (trimmed.startsWith("/") && !trimmed.startsWith("//") && !trimmed.startsWith("/\\")) return trimmed;
  if (trimmed.startsWith("./") || trimmed.startsWith("../")) return trimmed;
  try {
    const parsed = new URL(trimmed);
    return parsed.protocol === "https:" ? parsed.toString() : "";
  } catch {
    return defaultUrlTransform(trimmed);
  }
}

export function SafeMarkdown({ text }: { readonly text: string }) {
  return (
    <div className="safe-markdown">
      <Markdown
        remarkPlugins={[remarkGfm]}
        urlTransform={safeUrlTransform}
        components={{
          img: () => null,
          a: ({ children, href }) => href
            ? <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>
            : <span>{children}</span>,
        }}
      >
        {text}
      </Markdown>
    </div>
  );
}
