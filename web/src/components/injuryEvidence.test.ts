import { describe, expect, it } from "vitest";
import { isActionableInjuryEvidence } from "./injuryEvidence";

describe("isActionableInjuryEvidence", () => {
  it("rejects unknown / no-evidence placeholders", () => {
    expect(
      isActionableInjuryEvidence({
        player_id: "x",
        status: "unknown",
        summary: "No evidence",
        sources: [],
        meta: { data_as_of: "", projection_run_id: "" },
      }),
    ).toBe(false);
  });

  it("accepts cited or non-healthy statuses", () => {
    expect(
      isActionableInjuryEvidence({
        player_id: "x",
        status: "Out",
        summary: "Knee",
        sources: [],
        meta: { data_as_of: "", projection_run_id: "" },
      }),
    ).toBe(true);
    expect(
      isActionableInjuryEvidence({
        player_id: "x",
        status: "unknown",
        summary: "No evidence",
        sources: [{ title: "Report", url: "https://x.test", publisher: "x.test" }],
        meta: { data_as_of: "", projection_run_id: "" },
      }),
    ).toBe(true);
  });
});
