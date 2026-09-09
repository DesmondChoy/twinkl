import type { TraceEventContract } from "./demoContracts";
import type { NorthStarRecord, NorthStarSelection } from "./northStar";

type JsonObject = Record<string, unknown>;

export interface CoachComparisonNarrative extends JsonObject {
  weekly_mirror: string;
  tension_explanation: string;
  reflective_question: string;
}

export interface CoachComparisonArm extends JsonObject {
  narrative: CoachComparisonNarrative;
  validation: JsonObject;
  base_prompt: string;
  prompt: string;
  raw_output: string;
  repair_requirements: string[];
  provider: string;
  model: string;
  reasoning_effort: string;
  service_tier: string;
  prompt_name: string;
  prompt_version: string;
  call_metrics: JsonObject[];
  diagnostic_paths: string[];
  base_prompt_sha256: string;
  prompt_sha256: string;
  response_sha256: string;
  raw_output_sha256: string;
}

export interface CoachComparison extends JsonObject {
  schema_version: "coach-digest-nsm-comparison-v1";
  scenario_id: string;
  persona_id: string;
  week_start: string;
  week_end: string;
  weekly_drift_input_sha256: string;
  north_star_input_hash: string;
  north_star_context_sha256: string;
  north_star_context: JsonObject;
  without_north_star: CoachComparisonArm;
  with_north_star: CoachComparisonArm;
}

function object(value: unknown): JsonObject | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonObject : null;
}

function sameJson(left: unknown, right: unknown): boolean {
  const canonical = (value: unknown): string => {
    if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
    const item = object(value);
    return item ? `{${Object.keys(item).sort().map((key) => `${JSON.stringify(key)}:${canonical(item[key])}`).join(",")}}`
      : JSON.stringify(value);
  };
  return canonical(left) === canonical(right);
}

export function coachPromptParts(prompt: string): { instructions: string; input: JsonObject } | null {
  const marker = "\n\nUNTRUSTED INPUT DATA\n";
  const index = prompt.indexOf(marker);
  if (index < 0) return null;
  try {
    const input = object(JSON.parse(prompt.slice(index + marker.length)));
    return input ? { instructions: prompt.slice(0, index), input } : null;
  } catch {
    return null;
  }
}

function validArm(value: unknown): value is CoachComparisonArm {
  const arm = object(value);
  const narrative = object(arm?.narrative);
  if (!arm || !narrative || !["weekly_mirror", "tension_explanation", "reflective_question"]
    .every((key) => typeof narrative[key] === "string" && narrative[key].trim())) return false;
  if (!["base_prompt", "prompt", "raw_output", "provider", "model", "reasoning_effort", "service_tier", "prompt_name", "prompt_version"]
    .every((key) => typeof arm[key] === "string" && arm[key].length > 0)) return false;
  if (arm.provider !== "openai" || arm.model !== "gpt-5.6-luna" || arm.reasoning_effort !== "none"
    || arm.service_tier !== "default" || arm.prompt_name !== "demo_coach_nsm_comparison"
    || arm.prompt_version !== "1.0") return false;
  if (!Array.isArray(arm.repair_requirements) || !arm.repair_requirements.every((item) => typeof item === "string")) return false;
  const checks = object(arm.validation)?.checks;
  if (!Array.isArray(checks) || !checks.every((item) => object(item)?.passed === true)
    || !["groundedness", "non_circularity", "value_leakage", "state_claims", "length",
      "conversational_voice", "weekly_mirror_verbatim", "all_quotes_grounded",
      "one_reflective_question", "comparison_metadata_hidden"]
      .every((name) => checks.some((item) => object(item)?.name === name))) return false;
  const base = coachPromptParts(arm.base_prompt as string);
  const accepted = coachPromptParts(arm.prompt as string);
  if (!base || !accepted || !sameJson(base.input, accepted.input)) return false;
  try {
    return sameJson(JSON.parse(arm.raw_output as string), narrative);
  } catch {
    return false;
  }
}

export function savedCoachComparison(event: TraceEventContract | null): CoachComparison | null {
  if (!event || event.event_type !== "weekly_coach_generated" || event.source !== "saved_replay"
    || !["complete", "reused"].includes(event.status) || event.validation?.valid === false) return null;
  const pair = object(event.details.comparison);
  if (!pair || pair.schema_version !== "coach-digest-nsm-comparison-v1"
    || !["scenario_id", "persona_id", "week_start", "week_end"].every((key) => typeof pair[key] === "string")
    || !["weekly_drift_input_sha256", "north_star_input_hash", "north_star_context_sha256"]
      .every((key) => typeof pair[key] === "string" && /^[a-f0-9]{64}$/.test(pair[key]))
    || !object(pair.north_star_context) || !validArm(pair.without_north_star) || !validArm(pair.with_north_star)) return null;
  const without = pair.without_north_star;
  const withMoment = pair.with_north_star;
  const baseWithout = coachPromptParts(without.base_prompt)!;
  const baseWith = coachPromptParts(withMoment.base_prompt)!;
  // The typed receipt serializes an absent optional parent as null; the model input omits it.
  const context = { ...pair.north_star_context as JsonObject };
  if (context.parent_journal_entry === null) delete context.parent_journal_entry;
  if (baseWithout.instructions !== baseWith.instructions || baseWithout.input.north_star_context !== null
    || !sameJson(baseWith.input.north_star_context, context)
    || !sameJson(baseWithout.input, { ...baseWith.input, north_star_context: null })
    || !["provider", "model", "reasoning_effort", "service_tier", "prompt_name", "prompt_version"]
      .every((key) => without[key] === withMoment[key])) return null;
  return pair as CoachComparison;
}

export function comparisonMatchesMoment(
  pair: CoachComparison,
  result: { event: TraceEventContract; record: NorthStarRecord },
  selected: NorthStarSelection,
): boolean {
  const { record } = result;
  const source = record.sources.find((item) => item.entry_id === selected.entry_id);
  const context = pair.north_star_context;
  return pair.persona_id === record.owner_id && pair.week_start === record.week_start
    && pair.week_end === record.week_end && pair.north_star_input_hash === record.input_hash
    && `scenario-session:${pair.scenario_id}` === record.session_id
    && context.source_id === selected.entry_id && context.source_type === selected.quote_source
    && context.date === selected.date && context.exact_quote === selected.evidence_quote
    && context.mode === record.mode && context.core_value_phrase === record.value_phrase
    && context.source_text === source?.[selected.quote_source === "nudge_response" ? "nudge_response" : "journal_entry"]
    && (context.parent_journal_entry == null || (selected.quote_source === "nudge_response"
      && sameJson(context.parent_journal_entry, {
        source_id: selected.entry_id, date: selected.date, source_text: source?.journal_entry,
      })));
}
