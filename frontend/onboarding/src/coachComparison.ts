import type { TraceEventContract } from "./demoContracts";
import type { NorthStarRecord, NorthStarSelection } from "./northStar";

type JsonObject = Record<string, unknown>;

export function northStarAbsenceExplanation(
  result: { event: TraceEventContract; record: NorthStarRecord } | null,
): { title: string; reason: string } {
  const unavailable = {
    title: "Comparison unavailable for this week",
    reason: "The saved comparison could not be loaded or verified. This does not mean there was no supportive action in the writing.",
  };
  if (!result || !["complete", "reused", "not_eligible"].includes(result.event.status)
    || result.event.validation?.valid === false
    || !["complete", "not_eligible"].includes(result.record.status)
    || result.record.selected !== null) return unavailable;
  const { record } = result;
  let reason: string;
  switch (record.reason) {
    case "no_eligible_writing":
      reason = record.onset_t_index !== null
        ? "For Active Drift, a North Star Moment must come from before the Drift began. No eligible writing was available from that earlier period."
        : "There was no eligible writing available for this week’s North Star Moment review.";
      break;
    case "insufficient_evidence":
      reason = "There isn’t enough evidence to determine this week’s Drift state. Twinkl skips North Star Moments in this situation.";
      break;
    case "no_supportive_source": {
      reason = "The saved AI review did not find an action clearly supporting the reviewed Core Value in the eligible writing.";
      const reviews = object(object(object(record.experiment)?.output)?.source_reviews);
      const decisions = Object.values(reviews ?? {}).flatMap((review) => {
        const results = object(review)?.results;
        return Array.isArray(results) ? results.map(object) : [];
      });
      if (decisions.length && decisions.every((decision) => decision?.reason_code === "wrong_value")) {
        reason = "The saved AI review did not find a clear enough connection between the action described and the reviewed Core Value.";
      } else if (decisions.length && decisions.every((decision) => decision?.reason_code === "same_value_conflict")) {
        reason = "The saved AI review found behavior conflicting with the same Core Value in the eligible writing, so it did not select a North Star Moment.";
      }
      break;
    }
    default:
      return unavailable;
  }
  return { title: "Why there’s no North Star Moment", reason };
}

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
    || !["1.0", "1.1"].includes(arm.prompt_version as string)) return false;
  if (!Array.isArray(arm.repair_requirements) || !arm.repair_requirements.every((item) => typeof item === "string")) return false;
  const checks = object(arm.validation)?.checks;
  if (!Array.isArray(checks) || !checks.every((item) => object(item)?.passed === true)
    || !["groundedness", "non_circularity", "value_leakage", "state_claims", "length",
      "conversational_voice", "weekly_mirror_verbatim", "all_quotes_grounded",
      "one_reflective_question", "comparison_metadata_hidden"]
      .every((name) => checks.some((item) => object(item)?.name === name))) return false;
  if (arm.prompt_version === "1.1"
    && !checks.some((item) => object(item)?.name === "natural_reflection_voice")) return false;
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
