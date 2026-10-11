import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "../.agents/tests",
  testMatch: "*.spec.ts",
  workers: 1,
  timeout: 30000,
  use: {
    baseURL: process.env.MJCF_TEST_URL || "http://127.0.0.1:8765",
    viewport: { width: 1440, height: 1000 },
    headless: true,
    launchOptions: {
      args: [
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
      ],
    },
  },
});
