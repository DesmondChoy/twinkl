interface FullQuotation {
  quotation: string;
  source: string;
}

/** Expand saved, clipped quotations only when their source is unambiguous. */
export function expandCoachQuotations(text: string, sources: string[]): {
  text: string;
  fullQuotations: FullQuotation[];
} {
  const fullQuotations: FullQuotation[] = [];
  const expandedText = text.replace(/(?:“([^”]+)”|"([^"\n]+)")(\.)?/g, (quoted, curly, straight, period, offset) => {
    const phrase = (curly ?? straight) as string;
    if (!/(?:\.{3}|…)\s*$/.test(phrase)) return quoted;
    const prefix = phrase.replace(/(?:\.{3}|…)\s*$/, "").trim();
    if (prefix.split(/\s+/).length < 2) return quoted;

    const pattern = prefix.split(/\s+/)
      .map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("\\s+");
    const matches = [...new Set(sources)].flatMap((source) =>
      [...source.matchAll(new RegExp(pattern, "g"))]
        .filter((match) => match.index === 0 || !/[\p{L}\p{N}_]/u.test(source[match.index - 1]))
        .filter((match) => !/[\p{L}\p{N}_]/u.test(source[match.index + match[0].length] ?? ""))
        .map((match) => ({
          source,
          start: match.index,
          end: match.index + match[0].length,
        })),
    );
    if (matches.length !== 1) return quoted;

    const { source, start, end } = matches[0];
    // An ellipsis written by the author is not a clipped evidence excerpt.
    if (/^\s*(?:\.{3}|…)/.test(source.slice(end))) return quoted;
    const remainder = source.slice(end);
    const sentenceEnd = /[.!?](?=\s|$)/.exec(remainder);
    const fullQuote = source.slice(
      start,
      sentenceEnd ? end + sentenceEnd.index + 1 : source.length,
    ).trimEnd();
    const continuation = text.slice(offset + quoted.length - (period?.length ?? 0));
    if (/^(?:\s+[a-z]|\s*[,;:—–-])/.test(continuation)) {
      // Completing a sentence inside a fragment would break the surrounding prose.
      fullQuotations.push({ quotation: fullQuote, source });
      const fragment = source.slice(start, end);
      return curly === undefined ? `"${fragment}"` : `“${fragment}”`;
    }
    const expandedQuote = curly === undefined ? `"${fullQuote}"` : `“${fullQuote}”`;
    return expandedQuote + (/[.!?]$/.test(fullQuote) ? "" : period ?? "");
  });
  return { text: expandedText, fullQuotations };
}
