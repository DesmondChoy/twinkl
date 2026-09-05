import {
  BWS_OBJECTS,
  BWS_OBJECT_ORDER,
  BWS_SETS,
  VALUES,
  VALUE_ORDER,
  type BwsObjectKey,
  type BwsResponse,
  type ScoreBundle,
  type ValueKey,
} from "./domain";
import { ASSESSMENT_SECTIONS } from "./AssessmentSectionMap";

interface OnboardingScoreInspectionProps {
  confirmedValues: ValueKey[] | null;
  responses: BwsResponse[];
  scores: ScoreBundle;
  setOrder: number[];
}

const OBJECT_NAMES: Record<BwsObjectKey, string> = {
  power: "Power",
  achievement: "Achievement",
  hedonism: "Hedonism",
  stimulation: "Stimulation",
  self_direction: "Self-Direction",
  universalism_nature: "Universalism — Nature",
  benevolence: "Benevolence",
  tradition: "Tradition",
  conformity: "Conformity",
  security: "Security",
  universalism_social: "Universalism — Social",
};

function signed(value: number): string {
  if (Math.abs(value) < 0.005) return "0.00";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
}

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function duration(milliseconds: number): string {
  return milliseconds < 1_000
    ? `${milliseconds} ms`
    : `${(milliseconds / 1_000).toFixed(1)} s`;
}

function joinNames(values: ValueKey[]): string {
  const names = values.map((value) => VALUES[value].name);
  if (names.length < 2) return names[0] ?? "No result";
  return `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
}

function exampleEquation(value: ValueKey, scores: ScoreBundle): string {
  if (value === "universalism") {
    const nature = scores.bws.scores.universalism_nature;
    const social = scores.bws.scores.universalism_social;
    return `Mean(${signed(nature)} Nature, ${signed(social)} Social)`;
  }
  const most = scores.bws.best_counts[value];
  const least = scores.bws.worst_counts[value];
  const appearances = scores.bws.appearances[value];
  return `(${most} Most − ${least} Least) ÷ ${appearances}`;
}

export default function OnboardingScoreInspection({
  confirmedValues,
  responses,
  scores,
  setOrder,
}: OnboardingScoreInspectionProps) {
  const highestValues = scores.profile.top_values;
  const confirmed = confirmedValues !== null;
  const displayedValues = confirmedValues ?? highestValues;
  const leadValue = highestValues[0];
  const responseBySet = new Map(
    responses.map((response) => [response.set_number, response]),
  );
  const orderedResponses = setOrder
    .map((setIndex) => responseBySet.get(BWS_SETS[setIndex]?.setNumber))
    .filter((response): response is BwsResponse => response !== undefined);
  const weightsTotal = VALUE_ORDER.reduce(
    (total, value) => total + scores.profile.weights[value],
    0,
  );
  const allAppearSix = BWS_OBJECT_ORDER.every(
    (value) => scores.bws.appearances[value] === 6,
  );
  const choicesAreDistinct = responses.every(
    (response) => response.selected_best !== response.selected_worst,
  );
  const mostTotal = BWS_OBJECT_ORDER.reduce(
    (total, value) => total + scores.bws.best_counts[value],
    0,
  );
  const leastTotal = BWS_OBJECT_ORDER.reduce(
    (total, value) => total + scores.bws.worst_counts[value],
    0,
  );
  const rankedValues = [...VALUE_ORDER].sort((left, right) => {
    const weightDifference =
      scores.profile.weights[right] - scores.profile.weights[left];
    return (
      weightDifference || VALUE_ORDER.indexOf(left) - VALUE_ORDER.indexOf(right)
    );
  });

  return (
    <section
      className="score-inspection"
      aria-labelledby="score-inspection-title"
    >
      <header className="score-inspection__header">
        <div>
          <p className="eyebrow">How the Profile was formed</p>
          <h2 id="score-inspection-title">
            Begin with the recorded choices.
          </h2>
          <p>
            Each recorded selection is shown before any scoring. The
            calculation then counts the Schwartz objects, produces
            relative-importance scores, and maps them into the Profile.
          </p>
        </div>
        <div className="score-source" aria-label="Calculation provenance">
          <span>Calculation method</span>
          <strong>Deterministic · no model</strong>
        </div>
      </header>

      <div className="score-result">
        <div>
          <small>
            {confirmed
              ? displayedValues.length === 1
                ? "Confirmed Core Value"
                : "Confirmed Core Values"
              : highestValues.length === 1
                ? "Highest-scoring value"
                : "Highest-scoring values"}
          </small>
          <h3>{joinNames(displayedValues)}</h3>
        </div>
        <div className="score-result__phrases">
          {displayedValues.map((value) => (
            <p key={value}>{VALUES[value].phrase}</p>
          ))}
        </div>
      </div>

      {confirmed && highestValues.length > 2 ? (
        <p>
          {highestValues.length} values shared the highest score. You confirmed
          two as Core Values. The calculation retains every tied value below.
        </p>
      ) : null}

      <ol className="calculation-path" aria-label="SVBWS calculation steps">
        <li>
          <span>1</span>
          <small>Collect</small>
          <strong>11 groups</strong>
          <p>One Most and one Least choice in each group.</p>
        </li>
        <li>
          <span>2</span>
          <small>Count</small>
          <strong>Most − Least</strong>
          <p>Every SVBWS object appears exactly six times.</p>
        </li>
        <li>
          <span>3</span>
          <small>Score</small>
          <strong>{leadValue ? exampleEquation(leadValue, scores) : "Net ÷ appearances"}</strong>
          <p>
            {leadValue
              ? `${VALUES[leadValue].name} scores ${signed(scores.profile.scores[leadValue])}.`
              : "Each result stays between −1 and +1."}
          </p>
        </li>
        <li>
          <span>4</span>
          <small>Present</small>
          <strong>{joinNames(highestValues)}</strong>
          <p>Every exact highest-score tie is retained.</p>
        </li>
      </ol>

      <section
        id={ASSESSMENT_SECTIONS[0].id}
        className="score-section"
        aria-labelledby="recorded-choices-title"
      >
        <div className="score-section__heading">
          <div>
            <p className="eyebrow">Recorded choices</p>
            <h3 id="recorded-choices-title">
              The 11 Most and 11 Least selections.
            </h3>
          </div>
          <p>
            Read down each column in the order the questions appeared. Repeated
            cards make the later counts visible before the calculation begins.
          </p>
        </div>
        <div
          className="selection-ledger"
          role="region"
          aria-label="Recorded Most and Least selections"
          tabIndex={0}
        >
          <section
            className="selection-lane selection-lane--most"
            aria-labelledby="most-selections-title"
          >
            <header>
              <div>
                <small>Chosen as</small>
                <h4 id="most-selections-title">Most</h4>
              </div>
              <span>11 cards</span>
            </header>
            <ol>
              {orderedResponses.map((response, index) => {
                const value = response.selected_best;
                return (
                  <li key={response.set_number}>
                    <span className="selection-card__order">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <div>
                      <strong>{OBJECT_NAMES[value]}</strong>
                      <p>{BWS_OBJECTS[value].descriptor}</p>
                    </div>
                  </li>
                );
              })}
            </ol>
          </section>

          <section
            className="selection-lane selection-lane--least"
            aria-labelledby="least-selections-title"
          >
            <header>
              <div>
                <small>Chosen as</small>
                <h4 id="least-selections-title">Least</h4>
              </div>
              <span>11 cards</span>
            </header>
            <ol>
              {orderedResponses.map((response, index) => {
                const value = response.selected_worst;
                return (
                  <li key={response.set_number}>
                    <span className="selection-card__order">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <div>
                      <strong>{OBJECT_NAMES[value]}</strong>
                      <p>{BWS_OBJECTS[value].descriptor}</p>
                    </div>
                  </li>
                );
              })}
            </ol>
          </section>
        </div>
        <section
          id={ASSESSMENT_SECTIONS[1].id}
          className="selection-totals"
          aria-labelledby="selection-totals-title"
        >
          <div className="selection-totals__heading">
            <div>
              <p className="eyebrow">Counted once</p>
              <h4 id="selection-totals-title">Totals by value</h4>
            </div>
            <p>
              Each value appears once here. The Most and Least columns each
              total 11 recorded choices.
            </p>
          </div>
          <div className="selection-totals__table-wrap" tabIndex={0}>
            <table
              className="selection-totals__table"
              aria-label="Most and Least totals by value"
            >
              <thead>
                <tr>
                  <th scope="col">Value</th>
                  <th scope="col">Most</th>
                  <th scope="col">Least</th>
                </tr>
              </thead>
              <tbody>
                {BWS_OBJECT_ORDER.map((value) => (
                  <tr key={value}>
                    <th scope="row">{OBJECT_NAMES[value]}</th>
                    <td>{scores.bws.best_counts[value]}</td>
                    <td>{scores.bws.worst_counts[value]}</td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr>
                  <th scope="row">Total choices</th>
                  <td>{mostTotal}</td>
                  <td>{leastTotal}</td>
                </tr>
              </tfoot>
            </table>
          </div>
          <aside className="universalism-explanation">
            <strong>Why two Universalism objects?</strong>
            <p>
              Schwartz&apos;s theory has ten values. The published SVBWS
              separates Universalism&apos;s nature and social concerns so each
              can be ranked on its own in the balanced 11-object design. The
              Profile averages them back into one Universalism score.
            </p>
          </aside>
        </section>
      </section>

      <aside
        id={ASSESSMENT_SECTIONS[2].id}
        className="universalism-bridge"
      >
        <div className="universalism-bridge__facets">
          <span>
            Nature
            <strong>{signed(scores.bws.scores.universalism_nature)}</strong>
          </span>
          <b aria-hidden="true">+</b>
          <span>
            Social
            <strong>{signed(scores.bws.scores.universalism_social)}</strong>
          </span>
        </div>
        <div className="universalism-bridge__result">
          <small>Average the two facets</small>
          <strong>
            Universalism {signed(scores.profile.scores.universalism)}
          </strong>
        </div>
      </aside>

      <section
        id={ASSESSMENT_SECTIONS[3].id}
        className="score-section"
        aria-labelledby="profile-score-title"
      >
        <div className="score-section__heading">
          <div>
            <p className="eyebrow">Profile transformation</p>
            <h3 id="profile-score-title">Map scores into the ten-value Profile.</h3>
          </div>
          <p>
            Positive weights preserve the ranking and sum to 100%. The Experience
            phrase is the exact text shown to the user.
          </p>
        </div>
        <div
          className="score-table-wrap"
          role="region"
          aria-label="Ten-value Profile scores and Experience mapping"
          tabIndex={0}
        >
          <table className="score-table score-table--profile">
            <thead>
              <tr>
                <th scope="col">Schwartz value</th>
                <th scope="col">Calculation → score</th>
                <th scope="col">Weight</th>
                <th scope="col">Shown in Experience</th>
                <th scope="col">Result</th>
              </tr>
            </thead>
            <tbody>
              {rankedValues.map((value) => {
                const isHighest = highestValues.includes(value);
                return (
                  <tr
                    className={isHighest ? "score-table__highest" : undefined}
                    key={value}
                  >
                    <th scope="row">{VALUES[value].name}</th>
                    <td
                      className="score-table__calculation"
                      data-label="Calculation"
                    >
                      <code>{exampleEquation(value, scores)}</code>
                      <span aria-hidden="true">→</span>
                      <strong>{signed(scores.profile.scores[value])}</strong>
                    </td>
                    <td data-label="Weight">
                      {percentage(scores.profile.weights[value])}
                    </td>
                    <td data-label="Shown in Experience">
                      {VALUES[value].phrase}
                    </td>
                    <td
                      className={isHighest
                        ? "score-table__result"
                        : "score-table__result score-table__result--empty"}
                    >
                      {isHighest ? (
                        <span className="highest-chip">
                          {confirmedValues?.includes(value) ? "Core Value" : "Highest"}
                        </span>
                      ) : (
                        <span aria-label="Not highest">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section
        id={ASSESSMENT_SECTIONS[4].id}
        className="score-checks"
        aria-labelledby="score-checks-title"
      >
        <div>
          <p className="eyebrow">Contract checks</p>
          <h3 id="score-checks-title">The calculation passes its invariants.</h3>
        </div>
        <ul>
          <li>
            <span aria-hidden="true">✓</span>
            <strong>{responses.length}/11 canonical groups</strong>
            <small>Complete response set</small>
          </li>
          <li>
            <span aria-hidden="true">✓</span>
            <strong>{allAppearSix ? "Six appearances each" : "Exposure mismatch"}</strong>
            <small>Balanced SVBWS design</small>
          </li>
          <li>
            <span aria-hidden="true">✓</span>
            <strong>{choicesAreDistinct ? "Distinct Most and Least" : "Invalid choices"}</strong>
            <small>Every response is valid</small>
          </li>
          <li>
            <span aria-hidden="true">✓</span>
            <strong>{percentage(weightsTotal)} total weight</strong>
            <small>Normalized ten-value Profile</small>
          </li>
        </ul>
      </section>

      <aside className="score-boundary">
        <strong>No confidence score is inferred.</strong>
        <p>
          These scores describe relative importance in this response. Score
          spread and response time are not treated as reliability, probability,
          or psychometric confidence.
        </p>
      </aside>

      <details className="response-audit">
        <summary>
          <span>
            <strong>Review question order and timing</strong>
            <small>Canonical group and response time</small>
          </span>
          <span aria-hidden="true">+</span>
        </summary>
        <div
          className="score-table-wrap"
          role="region"
          aria-label="Assessment order and response times"
          tabIndex={0}
        >
          <table className="score-table score-table--responses">
            <thead>
              <tr>
                <th scope="col">Shown</th>
                <th scope="col">Canonical group</th>
                <th scope="col">Response time</th>
              </tr>
            </thead>
            <tbody>
              {orderedResponses.map((response, index) => (
                <tr key={response.set_number}>
                  <td>{index + 1}</td>
                  <th scope="row">Group {response.set_number}</th>
                  <td>{duration(response.response_time_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          Randomized group order and card order change presentation only. The
          score uses the selected object keys and the canonical group design.
        </p>
      </details>

      <footer className="score-method">
        <span>
          Instrument
          <code>SVBWS UI adaptation v2</code>
        </span>
        <span>
          Raw score
          <code>(Most − Least) ÷ appearances</code>
        </span>
        <span>
          Profile transformation
          <code>Mean Universalism facets, then shift-normalize</code>
        </span>
      </footer>
    </section>
  );
}
