import type { TraceEventContract } from "../demoContracts";
import type { CoachComparison, CoachComparisonNarrative } from "../coachComparison";
import type { NorthStarRecord } from "../northStar";

export function comparisonEvent(
  moment: TraceEventContract,
  without: CoachComparisonNarrative,
  withMoment: CoachComparisonNarrative = without,
): TraceEventContract {
  const record = moment.details.record as NorthStarRecord;
  const selected = record.selected!;
  const source = record.sources.find((item) => item.entry_id === selected.entry_id)!;
  const context = {
    source_id: selected.entry_id, source_type: selected.quote_source, date: selected.date,
    exact_quote: selected.evidence_quote,
    source_text: source[selected.quote_source === "nudge_response" ? "nudge_response" : "journal_entry"],
    mode: record.mode, core_value_phrase: record.value_phrase,
  };
  const prompt = (northStar: unknown) => "Message contract: live-prompt-boundary-v1\n\nTRUSTED INSTRUCTIONS\nWrite a reflection from this writing.\n\nUNTRUSTED INPUT DATA\n"
    + JSON.stringify({ evidence_lines: ["A fixture weekly excerpt."], north_star_context: northStar,
      week_window: `${record.week_start} to ${record.week_end}` });
  const arm = (narrative: CoachComparisonNarrative, northStar: unknown) => ({
    narrative,
    validation: { grounded_quotes: [], word_count: 50,
      checks: ["groundedness", "non_circularity", "value_leakage", "state_claims", "length",
        "conversational_voice", "weekly_mirror_verbatim", "all_quotes_grounded",
        "one_reflective_question", "comparison_metadata_hidden"]
        .map((name) => ({ name, passed: true, details: "Fixture check passed." })) },
    base_prompt: prompt(northStar), prompt: prompt(northStar), raw_output: JSON.stringify(narrative),
    repair_requirements: [], provider: "openai", model: "gpt-5.6-luna", reasoning_effort: "none",
    service_tier: "default", prompt_name: "demo_coach_nsm_comparison", prompt_version: "1.0",
    call_metrics: [], diagnostic_paths: [], base_prompt_sha256: "a".repeat(64),
    prompt_sha256: "a".repeat(64), response_sha256: "a".repeat(64), raw_output_sha256: "a".repeat(64),
  });
  const comparison: CoachComparison = {
    schema_version: "coach-digest-nsm-comparison-v1",
    scenario_id: record.session_id.replace(/^scenario-session:/, ""), persona_id: record.owner_id,
    week_start: record.week_start, week_end: record.week_end,
    weekly_drift_input_sha256: "a".repeat(64), north_star_input_hash: record.input_hash,
    north_star_context_sha256: "a".repeat(64), north_star_context: context,
    without_north_star: arm(without, null), with_north_star: arm(withMoment, context),
  };
  return { ...moment, event_id: `${moment.event_id}-coach`, event_type: "weekly_coach_generated",
    source: "saved_replay", status: "complete", parent_event_id: null,
    prompt: comparison.without_north_star.prompt, raw_response: comparison.without_north_star.raw_output,
    validation: { valid: true }, details: { narrative: without,
      validation: comparison.without_north_star.validation, comparison } };
}
