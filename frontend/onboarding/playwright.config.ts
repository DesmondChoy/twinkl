import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  use: {
    ...devices["Desktop Chrome"],
    baseURL: "http://127.0.0.1:8765",
    viewport: { width: 390, height: 844 },
    reducedMotion: "reduce",
    timezoneId: "Asia/Singapore",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "npm run build && cd ../.. && . .venv/bin/activate && uv run --no-sync uvicorn scripts.demo_north_star_qc:app --host 127.0.0.1 --port 8765",
    env: { TWINKL_QC_STATIC_ROOT: "frontend/onboarding/dist" },
    url: "http://127.0.0.1:8765/health",
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
