import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import { validateExperienceApiResponse } from "../src/demoContracts";
import { BWS_OBJECT_ORDER } from "../src/domain";
import { SESSION_STORAGE_KEY } from "../src/session";

const quote = "I helped my sister carry the groceries upstairs and stayed to cook dinner with her.";
let browserErrors: string[];
let externalRequests: string[];

test.beforeEach(async ({ page, request }) => {
  browserErrors = [];
  externalRequests = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
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
  expect(externalRequests).toEqual([]);
});

async function trace(request: APIRequestContext, sessionId: string) {
  const response = await request.post("/api/experience", {
    data: {
      schema_version: "experience-inspect-v1",
      operation: "read_trace",
      request_id: crypto.randomUUID(),
      session_id: sessionId,
      after_event_id: null,
    },
  });
  expect(response.ok()).toBe(true);
  const payload = validateExperienceApiResponse(await response.json());
  if (payload.operation !== "read_trace") throw new Error("Expected recorded Experience work");
  return payload.events;
}

async function completeOnboarding(page: Page) {
  await page.getByRole("button", { name: "Try Onboarding", exact: true }).click();
  await page.getByRole("textbox", { name: "Preferred name" }).fill("Casey");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  for (let group = 1; group <= 11; group += 1) {
    await expect(page.getByLabel(`Values · ${group} of 11`, { exact: true })).toBeVisible();
    const cards = page.locator('[data-testid="value-card"][data-location="pool"]');
    await expect(cards).toHaveCount(6);
    const shown = await cards.evaluateAll((elements) => elements.map((element) => element.getAttribute("data-value")));
    const ordered = BWS_OBJECT_ORDER.filter((value) => shown.includes(value));
    // Fixed choices preserve the Profile despite randomized card and group order.
    await page.locator(`[data-testid="value-card"][data-value="${ordered[0]}"]`).press("m");
    await page.locator(`[data-testid="value-card"][data-value="${ordered.at(-1)}"]`).press("l");
  }
  await expect(page.getByRole("heading", { name: "What sits at the center." })).toBeVisible();
  const ties = page.locator(".core-value-choice");
  if (await ties.count()) {
    await ties.nth(0).click();
    await ties.nth(1).click();
  }
  const created = page.waitForResponse((response) =>
    response.url().endsWith("/api/experience")
    && response.request().postDataJSON()?.operation === "create_session");
  await page.getByRole("button", { name: "Confirm my compass" }).click();
  const payload = validateExperienceApiResponse(await (await created).json());
  if (payload.operation !== "create_session") throw new Error("Profile was not accepted by Python");
  return payload.session.session_id;
}

test("saved replay preserves the selected week and Inspect context across reload", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Try the Demo", exact: true }).click();
  await page.getByRole("button", { name: "Start at week 1", exact: true }).click();
  await expect(page.getByText("Week 1 of 5", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Inspect decision", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Show Active Drift — week 4" }).click();
  await page.getByRole("button", { name: "Review Weekly Drift Detection", exact: true }).click();
  await expect(page.getByRole("article", { name: "Active Drift", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Inspect decision", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Follow the work, step by step." })).toBeVisible();
  const selection = page.getByTestId("inspect-selection");
  await expect(selection).toBeVisible();
  const selectedEvent = await selection.textContent();
  await page.reload();
  await expect(page.getByRole("button", { name: "Inspect", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(selection).toHaveText(selectedEvent!);
  await page.getByRole("button", { name: "Experience", exact: true }).click();
  await expect(page.getByText("Week 4 of 5", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Journal Entries", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Inspect decision", exact: true })).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("manual writing closes a week, recovers only Coach Digest, and confirms session deletion", async ({ page, request }) => {
  await page.goto("/");
  const sessionId = await completeOnboarding(page);
  await page.getByRole("button", { name: "Start my first Journal Entry" }).click();
  await expect(page.getByRole("heading", { name: "Know where your text goes." })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "First Journal Entry", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Continue with manual demo" }).click();
  await page.getByRole("textbox", { name: "First Journal Entry", exact: true }).fill(quote);
  await page.getByRole("button", { name: "Save Journal Entry", exact: true }).click();
  await expect(page.getByRole("button", { name: "Close week and review" })).toBeEnabled();
  const openEvents = await trace(request, sessionId);
  expect(openEvents.filter((event) => event.event_type === "journal_entry_submitted")).toHaveLength(1);
  expect(openEvents.some((event) => event.event_type === "weekly_review_completed")).toBe(false);

  expect((await request.post("/qc/coach/failure")).ok()).toBe(true);
  await page.getByRole("button", { name: "Close week and review" }).click();
  await expect(page.getByRole("heading", { name: "Your weekly response could not be prepared." })).toBeVisible();
  await expect(page.getByText("The Weekly Drift Detection result remains available.", { exact: true })).toBeVisible();
  const failedEvents = await trace(request, sessionId);
  expect(failedEvents.filter((event) => event.event_type === "weekly_review_completed")).toHaveLength(1);
  const failedCoach = failedEvents.find((event) => event.event_type === "weekly_coach_generated");
  expect(failedCoach?.status).toBe("failed");

  expect((await request.post("/qc/coach/success")).ok()).toBe(true);
  await page.getByRole("button", { name: "Retry Coach Digest" }).click();
  await expect(page.getByText("What did making that time mean to you?", { exact: true })).toBeVisible();
  const recoveredEvents = await trace(request, sessionId);
  expect(recoveredEvents.filter((event) => event.event_type === "weekly_review_completed")).toHaveLength(1);
  expect(recoveredEvents.filter((event) => event.event_type === "journal_entry_submitted")).toHaveLength(1);
  expect(recoveredEvents.filter((event) => event.event_type === "weekly_coach_generated").map((event) => event.status))
    .toEqual(["failed", "complete"]);

  page.once("dialog", (dialog) => dialog.accept());
  const deleted = page.waitForResponse((response) =>
    response.url().endsWith("/api/experience")
    && response.request().postDataJSON()?.operation === "delete_session");
  await page.getByRole("button", { name: "Delete session", exact: true }).click();
  expect((await (await deleted).json()).deleted).toBe(true);
  await expect(page.getByRole("heading", { name: "Choose how to explore Twinkl." })).toBeVisible();
  const stored = await page.evaluate((key) => JSON.parse(localStorage.getItem(key)!), SESSION_STORAGE_KEY);
  expect(stored.confirmed_profile).toBeNull();
  expect(stored.session_id).not.toBe(sessionId);
  const missing = await request.post("/api/experience", {
    data: { schema_version: "experience-inspect-v1", operation: "read_trace",
      request_id: crypto.randomUUID(), session_id: sessionId, after_event_id: null },
  });
  expect(missing.status()).toBe(404);
  expect((await missing.json()).error.code).toBe("session_not_found");
});
