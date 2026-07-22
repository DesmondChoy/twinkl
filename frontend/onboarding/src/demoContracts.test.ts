import { describe, expect, it } from "vitest";
import fixtureJson from "./contracts/experience_inspect_v1.fixture.json";
import schemaJson from "./contracts/experience_inspect_v1.schema.json";
import {
  EXPERIENCE_INSPECT_CONTRACT_VERSION,
  validateExperienceInspectFixture,
} from "./demoContracts";
import { validateProfile } from "./domain";

function fixtureCopy(): unknown {
  return structuredClone(fixtureJson) as unknown;
}

describe("Experience and Inspect v1 contract", () => {
  it("validates the canonical Python-generated fixture and Profile", () => {
    const fixture = validateExperienceInspectFixture(fixtureCopy());

    expect(fixture.schema_version).toBe(EXPERIENCE_INSPECT_CONTRACT_VERSION);
    expect(validateProfile(fixture.session.profile)).toEqual(fixture.session.profile);
    expect(new Set(fixture.trace_events.map((event) => event.event_type)).size).toBe(10);
    expect(new Set(fixture.trace_events.map((event) => event.status))).toEqual(
      new Set(["complete", "reused", "refused", "invalid", "failed"]),
    );
    expect(
      fixture.trace_events
        .filter((event) => fixture.session.trace_event_ids.includes(event.event_id))
        .every((event) => event.input_refs.length > 0 || event.result_refs.length > 0),
    ).toBe(true);
  });

  it("ships the generated versioned JSON Schema", () => {
    expect(schemaJson.$id).toBe(
      "https://twinkl.local/contracts/experience-inspect-v1.schema.json",
    );
    expect(schemaJson.title).toBe("ContractFixtureSet");
  });

  it("rejects incompatible versions and missing trace fields", () => {
    const wrongVersion = fixtureCopy() as { schema_version: string };
    wrongVersion.schema_version = "experience-inspect-v2";
    expect(() => validateExperienceInspectFixture(wrongVersion)).toThrow("schema_version");

    const missingHash = fixtureCopy() as { trace_events: Array<Record<string, unknown>> };
    delete missingHash.trace_events[0].input_hash;
    expect(() => validateExperienceInspectFixture(missingHash)).toThrow("incompatible fields");
  });

  it("keeps Weekly Drift Reviewer Decisions distinct from VIF Critic Predictions", () => {
    const fixture = fixtureCopy() as {
      session: { weekly_reviewer_decisions: Array<Record<string, unknown>> };
    };
    fixture.session.weekly_reviewer_decisions[0].prediction = -1;
    fixture.session.weekly_reviewer_decisions[0].uncertainty = 0.08;

    expect(() => validateExperienceInspectFixture(fixture)).toThrow("incompatible fields");
  });

  it("rejects a changed Weekly Drift Reviewer model contract", () => {
    const fixture = fixtureCopy() as {
      trace_events: Array<Record<string, unknown>>;
    };
    const event = fixture.trace_events.find(
      (candidate) => candidate.event_type === "weekly_review_completed",
    );
    if (!event) throw new Error("Fixture lacks weekly_review_completed");
    (event.model_contract as Record<string, unknown>).reasoning_effort = "medium";

    expect(() => validateExperienceInspectFixture(fixture)).toThrow("Luna-low");
  });
});
