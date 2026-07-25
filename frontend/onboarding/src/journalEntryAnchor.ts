export function journalEntryAnchorId(journalEntryId: string): string {
  return `journal-entry-${journalEntryId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}
