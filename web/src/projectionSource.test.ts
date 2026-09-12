import { describe, expect, it } from "vitest";
import {
  boardSourceToProjectionSource,
  isBoardSource,
  labelForBoardSource,
} from "./projectionSource";

describe("projectionSource seam", () => {
  it("maps UI boards onto APP_PROJECTION_SOURCE values", () => {
    expect(boardSourceToProjectionSource("vegas_props")).toBe("weekly_props");
    expect(boardSourceToProjectionSource("league_value")).toBe("sealed_release");
  });

  it("labels boards for the toggle hint", () => {
    expect(labelForBoardSource("vegas_props")).toBe("Vegas Props");
    expect(labelForBoardSource("league_value")).toBe("League Value");
  });

  it("accepts only known board ids", () => {
    expect(isBoardSource("vegas_props")).toBe(true);
    expect(isBoardSource("league_value")).toBe(true);
    expect(isBoardSource("sealed_release")).toBe(false);
    expect(isBoardSource(null)).toBe(false);
  });
});
