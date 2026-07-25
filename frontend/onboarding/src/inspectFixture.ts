import fixturePayload from "./contracts/experience_inspect_v1.fixture.json";
import { validateExperienceInspectFixture } from "./demoContracts";

export const canonicalInspectFixture = validateExperienceInspectFixture(fixturePayload);
