import { expect, test } from "@playwright/test";

test("loads the WASM modem and validates UTF-8 byte length", async ({ page }) => {
  await page.goto("./");
  await expect(page.getByRole("heading", { name: /Send text/ })).toBeVisible();
  const message = page.getByRole("textbox", { name: "Message" });
  await message.fill("é🌍");
  await expect(page.getByText("6 / 235 bytes")).toBeVisible();
  await expect(page.getByRole("button", { name: "Start broadcasting" })).toBeEnabled();
});

test("does not request microphone access before the receive action", async ({ page }) => {
  await page.addInitScript(() => {
    let requests = 0;
    Object.defineProperty(window, "__microphoneRequests", { get: () => requests });
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: async () => {
          requests += 1;
          throw new DOMException("denied", "NotAllowedError");
        },
      },
    });
  });
  await page.goto("./");
  expect(await page.evaluate(() => (window as typeof window & { __microphoneRequests: number }).__microphoneRequests)).toBe(0);
  await page.getByRole("tab", { name: "Receive" }).click();
  expect(await page.evaluate(() => (window as typeof window & { __microphoneRequests: number }).__microphoneRequests)).toBe(0);
  await page.getByRole("button", { name: "Start listening" }).click();
  await expect(page.getByRole("status")).toContainText("Microphone permission was denied");
  expect(await page.evaluate(() => (window as typeof window & { __microphoneRequests: number }).__microphoneRequests)).toBe(1);
});

test("registers the offline service worker", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "WebKit headless does not expose service workers consistently");
  await page.goto("./");
  await expect.poll(() => page.evaluate(async () => Boolean(await navigator.serviceWorker.ready))).toBe(true);
});
