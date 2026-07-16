import { beforeEach, describe, expect, it } from "vitest";
import { GA_MEASUREMENT_ID, initializeAnalytics } from "./analytics";

describe("Google Analytics", () => {
  beforeEach(() => {
    document.head.innerHTML = "";
    delete window.dataLayer;
    delete window.gtag;
  });

  it("loads the mligent Equidistant tag with storage denied", () => {
    initializeAnalytics();

    const script = document.querySelector<HTMLScriptElement>(
      "#equidistant-google-analytics"
    );
    expect(script?.src).toBe(
      `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`
    );
    expect(script?.async).toBe(true);
    expect(window.dataLayer?.[0]).toEqual([
      "consent",
      "default",
      expect.objectContaining({
        analytics_storage: "denied",
        ad_storage: "denied"
      })
    ]);
    expect(window.dataLayer?.[2]).toEqual([
      "config",
      GA_MEASUREMENT_ID,
      { anonymize_ip: true }
    ]);
  });

  it("does not add the tag twice", () => {
    initializeAnalytics();
    initializeAnalytics();

    expect(document.querySelectorAll("#equidistant-google-analytics")).toHaveLength(1);
  });
});
