import init, { Decoder } from "./wasm/bytewaves_wasm";

let decoder: Decoder | undefined;

self.onmessage = async (message: MessageEvent<{ type: string; sampleRate?: number; samples?: Float32Array }>) => {
  try {
    if (message.data.type === "init") {
      await init();
      decoder = new Decoder(message.data.sampleRate ?? 48_000);
      self.postMessage({ type: "ready" });
    } else if (message.data.type === "samples" && decoder && message.data.samples) {
      const events = decoder.pushSamples(message.data.samples);
      if (events.length) self.postMessage({ type: "events", events });
    } else if (message.data.type === "reset") {
      decoder?.reset();
    }
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};

