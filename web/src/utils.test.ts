import { describe, expect, it } from "vitest";
import { utf8ByteLength } from "./utils";

describe("utf8ByteLength", () => {
  it("counts encoded bytes rather than JavaScript characters", () => {
    expect(utf8ByteLength("abc")).toBe(3);
    expect(utf8ByteLength("é")).toBe(2);
    expect(utf8ByteLength("🌍")).toBe(4);
  });
});

