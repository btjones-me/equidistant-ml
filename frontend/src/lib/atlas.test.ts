import { describe, expect, it } from "vitest";
import { combineValues, standardDeviation } from "./atlas";

describe("offline atlas group scoring", () => {
  it("matches the backend sample standard deviation", () => {
    expect(standardDeviation([20, 30])).toBeCloseTo(Math.sqrt(50));
    expect(standardDeviation([20])).toBe(0);
  });

  it("supports every group combination strategy", () => {
    const values = [20, 30, 40];
    expect(combineValues(values, "mean")).toBe(30);
    expect(combineValues(values, "max")).toBe(40);
    expect(combineValues(values, "fairness")).toBe(10);
    expect(combineValues(values, "balanced")).toBe(35);
  });

  it("uses the participant surface unchanged for one person", () => {
    expect(combineValues([27.5], "balanced")).toBe(27.5);
    expect(combineValues([27.5], "fairness")).toBe(27.5);
  });
});
