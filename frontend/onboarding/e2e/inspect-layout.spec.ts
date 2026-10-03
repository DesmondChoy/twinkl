import { expect, test, type Locator, type Page } from "@playwright/test";

const samples = [
  { persona: "Nisha Agarwal", week: 1, state: "No Active Drift", comparison: true },
  { persona: "Nisha Agarwal", week: 4, state: "Active Drift", comparison: true },
  { persona: "Noor Haddad", week: 3, state: "No Active Drift", comparison: true },
  { persona: "Lukas Vetter", week: 3, state: "Active Drift", comparison: true },
  { persona: "Wei Jun Chen", week: 5, state: "Insufficient Evidence", comparison: false },
  { persona: "Meera Krishnamurthy", week: 1, state: "Active Drift", comparison: false },
  { persona: "Meera Krishnamurthy", week: 4, state: "No Active Drift", comparison: true },
] as const;

let browserErrors: string[];
let failedRequests: string[];
let externalRequests: string[];

test.beforeEach(async ({ page, request }) => {
  browserErrors = [];
  failedRequests = [];
  externalRequests = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("response", (response) => {
    if (response.status() >= 400) failedRequests.push(`${response.status()} ${response.url()}`);
  });
  await page.route("**/*", (route) => {
    if (new URL(route.request().url()).origin === "http://127.0.0.1:8765") {
      return route.continue();
    }
    externalRequests.push(route.request().url());
    return route.abort();
  });
  expect((await request.post("/qc/mode/omission")).ok()).toBe(true);
  expect((await request.post("/qc/coach/success")).ok()).toBe(true);
});

test.afterEach(() => {
  expect(browserErrors).toEqual([]);
  expect(failedRequests).toEqual([]);
  expect(externalRequests).toEqual([]);
});

async function openInspect(page: Page, sample: typeof samples[number]) {
  await page.goto("/");
  await page.getByRole("button", { name: "Try the Demo", exact: true }).click();
  await page.getByRole("radio", { name: sample.persona, exact: true }).check();
  await page.getByRole("button", { name: "Start at week 1", exact: true }).click();
  if (sample.week !== 1) {
    await page.getByRole("button", { name: `Show week ${sample.week}, outcome hidden`, exact: true }).click();
  }
  await page.getByRole("button", { name: "Review Weekly Drift Detection", exact: true }).click();
  await expect(page.getByRole("article", { name: sample.state, exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Inspect decision", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Follow the work, step by step." })).toBeVisible();
  return page.getByRole("list", { name: "Current week events", exact: true });
}

function eventSummary(timeline: Locator, name: string) {
  return timeline.locator(`.trace-event__card > summary[aria-label$=": ${name}"]`);
}

async function expectFocusedHeadingClearOfHeader(page: Page, summary: Locator) {
  await expect(summary).toBeFocused();
  await expect.poll(async () => {
    const header = await page.locator(".topbar").boundingBox();
    const heading = await summary.locator(".trace-event__name").boundingBox();
    const viewport = page.viewportSize();
    return !!header && !!heading && !!viewport
      && heading.y >= header.y + header.height
      && heading.y + heading.height <= viewport.height;
  }).toBe(true);
}

async function expectFullComparisonWidth(comparison: Locator) {
  await expect(comparison).toBeVisible();
  const widthRatio = await comparison.evaluate((element) => {
    const parent = element.parentElement!;
    const style = getComputedStyle(parent);
    const available = parent.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    return element.getBoundingClientRect().width / available;
  });
  expect(widthRatio).toBeGreaterThan(0.98);
  expect(widthRatio).toBeLessThanOrEqual(1.01);
}

async function expectNoHorizontalOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
}

async function exerciseDisclosure(summary: Locator, contents: Locator) {
  await expect(summary).toBeVisible();
  const markerVisible = await summary.evaluate((element) => {
    const style = getComputedStyle(element);
    const marker = getComputedStyle(element, "::marker");
    return style.display === "list-item" && style.listStyleType !== "none"
      && marker.visibility === "visible" && parseFloat(marker.fontSize) > 0;
  });
  expect(markerVisible, "Nested disclosures need a visible expand/collapse marker").toBe(true);
  const details = summary.locator("..");
  await summary.focus();
  if (await details.evaluate((element) => (element as HTMLDetailsElement).open)) {
    await summary.press("Space");
  }
  await expect(contents).toBeHidden();
  await summary.press("Enter");
  await expect(details).toHaveJSProperty("open", true);
  await expect(contents).toBeVisible();
  await expect(summary).toBeFocused();
  await summary.press("Space");
  await expect(details).toHaveJSProperty("open", false);
  await expect(contents).toBeHidden();
}

async function expectReadableTechnicalHeading(comparison: Locator) {
  const heading = comparison.getByRole("heading", { name: "Without North Star Moment: north_star_context", exact: true });
  await expect(heading).toBeVisible();
  const layout = await heading.evaluate((element) => {
    const box = element.getBoundingClientRect();
    const range = document.createRange();
    range.selectNodeContents(element);
    const textFits = [...range.getClientRects()].every((rect) => rect.left >= box.left - 1 && rect.right <= box.right + 1);
    const body = element.closest(".inspect-coach-comparison")!.querySelector("p")!;
    return {
      textFits,
      fontSize: parseFloat(getComputedStyle(element).fontSize),
      fontFamily: getComputedStyle(element).fontFamily,
      bodyFontSize: parseFloat(getComputedStyle(body).fontSize),
      bodyFontFamily: getComputedStyle(body).fontFamily,
    };
  });
  expect(layout.textFits, "The full technical label must wrap inside its column").toBe(true);
  expect(layout.fontSize).toBeGreaterThanOrEqual(12);
  expect(layout.fontSize).toBeLessThanOrEqual(layout.bodyFontSize * 1.125);
  expect(layout.fontFamily).toBe(layout.bodyFontFamily);
}

for (const sample of samples) {
  test(`Inspect remains usable for ${sample.persona}, week ${sample.week}: ${sample.state}`, async ({ page }) => {
    const timeline = await openInspect(page, sample);
    const coachSummary = eventSummary(timeline, "Coach Digest response generated");
    const coachCard = coachSummary.locator("..");
    const comparison = coachCard.getByRole("region", { name: "Coach Digest prompt and response comparison" });
    const comparisonLink = page.getByRole("button", { name: "Inspect Coach Digest comparison", exact: true });

    if (sample.comparison) {
      await comparisonLink.click();
      await expectFocusedHeadingClearOfHeader(page, coachSummary);
      await expectFullComparisonWidth(comparison);
      await expectReadableTechnicalHeading(comparison);
      const shared = comparison.locator("details").filter({ has: page.locator("summary", { hasText: /^Shared instructions and unchanged weekly input$/ }) });
      await exerciseDisclosure(shared.locator(":scope > summary"), shared.getByLabel("Exact common initial instructions", { exact: true }));
      const receipt = comparison.locator("details").filter({ has: page.locator("summary", { hasText: /^Without North Star Moment: validation and generation receipt$/ }) });
      await exerciseDisclosure(receipt.locator(":scope > summary"), receipt.getByLabel("Without North Star Moment: generation receipt", { exact: true }));
    } else {
      await expect(comparisonLink).toHaveCount(0);
      await coachSummary.click();
      await expect(coachCard).toHaveJSProperty("open", true);
      await expect(comparison).toHaveCount(0);
    }

    const technical = coachCard.locator(".trace-event__details > .inspect-technical--group");
    await exerciseDisclosure(technical.locator(":scope > summary"), technical.locator(".trace-facts"));
    await expect(coachCard).toHaveJSProperty("open", true);
    await expectNoHorizontalOverflow(page);

    await page.getByRole("button", { name: "Inspect North Star Moment", exact: true }).click();
    const northStarSummary = eventSummary(timeline, "North Star Moment reviewed");
    await expectFocusedHeadingClearOfHeader(page, northStarSummary);
    const northStar = northStarSummary.locator("..").locator(".nsm-inspect");
    await expect(northStar.getByRole("heading", { name: "How this North Star Moment was derived" })).toBeVisible();
    if (sample.comparison) {
      await expect(northStar.locator(".nsm-inspect__composition")).toContainText("above the with-context response");
      await expect(northStar.locator(".nsm-inspect__composition")).not.toContainText("replaces both");
      await expect(northStar.locator(".nsm-inspect__composition")).not.toContainText("Introduction:");
      const source = northStar.locator(".nsm-inspect__source").first();
      await exerciseDisclosure(source.locator(":scope > summary"), source.locator(".nsm-inspect__assessments"));
      await expect(northStarSummary.locator("..")).toHaveJSProperty("open", true);
    } else {
      await expect(northStar.locator(".nsm-inspect__outcome")).toHaveText("This weekly result was not eligible for a North Star Moment.");
      await expect(northStar.getByLabel("Exact selected quotation")).toHaveCount(0);
    }
    await expectNoHorizontalOverflow(page);

    await page.getByRole("navigation", { name: "Filter Inspect events" }).getByRole("button", { name: "Weekly Drift Reviewer", exact: true }).click();
    await expect(timeline.locator(".trace-event__card")).toHaveCount(2);
    await expect(eventSummary(timeline, "Weekly review completed")).toBeVisible();
    await page.getByRole("navigation", { name: "Filter Inspect events" }).getByRole("button", { name: "All steps", exact: true }).click();
    await expect(coachSummary).toBeVisible();
    if (sample.week > 1) {
      const history = page.locator(".inspect-history");
      await history.locator(":scope > summary").click();
      await expect(history.getByRole("list", { name: "Earlier events", exact: true })).toBeVisible();
    }
    await expectNoHorizontalOverflow(page);
    await page.getByRole("button", { name: "Return to Experience", exact: true }).click();
    await expect(page.locator(".replay-week-heading__position")).toContainText(`Week ${sample.week} of`);
    await page.getByRole("button", { name: "Review Weekly Drift Detection", exact: true }).click();
    await expect(page.getByRole("article", { name: sample.state, exact: true })).toBeVisible();
  });
}

test("Coach Digest comparison fills laptop and desktop widths with readable headings", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "The sample matrix covers the narrow viewport.");
  const timeline = await openInspect(page, samples[1]);
  const summary = eventSummary(timeline, "Coach Digest response generated");
  for (const width of [1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await page.getByRole("button", { name: "Inspect Coach Digest comparison", exact: true }).click();
    await expectFocusedHeadingClearOfHeader(page, summary);
    const comparison = summary.locator("..").getByRole("region", { name: "Coach Digest prompt and response comparison" });
    await expectFullComparisonWidth(comparison);
    await expectReadableTechnicalHeading(comparison);
    await expectNoHorizontalOverflow(page);
  }
});
