import "./style.css";
import { registerSW } from "virtual:pwa-register";
import init, { encodeText, maxPayloadBytes } from "./wasm/bytewaves_wasm";
import { microphoneError, utf8ByteLength } from "./utils";

type DecoderEvent =
  | { type: "synchronized" }
  | { type: "frameProgress"; received: number; total: number }
  | { type: "decodeFailed"; reason: string }
  | { type: "messageDecoded"; text: string; corrected_errors?: number };

const byId = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing #${id}`);
  return element as T;
};

const sendTab = byId<HTMLButtonElement>("send-tab");
const receiveTab = byId<HTMLButtonElement>("receive-tab");
const sendPanel = byId("send-panel");
const receivePanel = byId("receive-panel");
const message = byId<HTMLTextAreaElement>("message");
const byteCount = byId("byte-count");
const sendButton = byId<HTMLButtonElement>("send-button");
const receiveButton = byId<HTMLButtonElement>("receive-button");
const status = byId("status");
const decodedCard = byId("decoded-card");
const decodedText = byId("decoded-text");

let ready = false;
let mode: "idle" | "sending" | "listening" = "idle";
let audioContext: AudioContext | undefined;
let sourceNode: AudioBufferSourceNode | undefined;
let microphoneStream: MediaStream | undefined;
let captureNode: AudioWorkletNode | undefined;
let decoderWorker: Worker | undefined;

function setStatus(text: string, kind: "normal" | "active" | "error" = "normal"): void {
  status.className = `status ${kind}`;
  status.querySelector("span")!.textContent = text;
}

function showTab(tab: "send" | "receive"): void {
  if (mode !== "idle") stopAll();
  const sending = tab === "send";
  sendPanel.classList.toggle("hidden", !sending);
  receivePanel.classList.toggle("hidden", sending);
  sendTab.classList.toggle("active", sending);
  receiveTab.classList.toggle("active", !sending);
  sendTab.setAttribute("aria-selected", String(sending));
  receiveTab.setAttribute("aria-selected", String(!sending));
  setStatus("Modem ready");
}

function updateByteCount(): void {
  const length = utf8ByteLength(message.value);
  const maximum = ready ? maxPayloadBytes() : 235;
  byteCount.textContent = `${length} / ${maximum} bytes`;
  byteCount.classList.toggle("over", length > maximum);
  sendButton.disabled = !ready || length === 0 || length > maximum;
}

async function getAudioContext(): Promise<AudioContext> {
  if (!audioContext || audioContext.state === "closed") audioContext = new AudioContext();
  await audioContext.resume();
  return audioContext;
}

async function startSending(): Promise<void> {
  if (mode === "sending") {
    stopAll();
    return;
  }
  try {
    stopAll();
    const context = await getAudioContext();
    const samples = encodeText(message.value, context.sampleRate);
    const buffer = context.createBuffer(1, samples.length, context.sampleRate);
    buffer.getChannelData(0).set(samples);
    sourceNode = context.createBufferSource();
    sourceNode.buffer = buffer;
    sourceNode.loop = true;
    sourceNode.connect(context.destination);
    sourceNode.start();
    mode = "sending";
    sendButton.innerHTML = '<span class="button-icon">■</span><span>Stop broadcasting</span>';
    setStatus("Broadcasting repeated protected frames", "active");
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), "error");
  }
}

async function startListening(): Promise<void> {
  if (mode === "listening") {
    stopAll();
    return;
  }
  try {
    stopAll();
    const context = await getAudioContext();
    microphoneStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    await context.audioWorklet.addModule(`${import.meta.env.BASE_URL}capture-worklet.js`);
    decoderWorker = new Worker(new URL("./decoder-worker.ts", import.meta.url), { type: "module" });
    decoderWorker.onmessage = handleWorkerMessage;
    decoderWorker.postMessage({ type: "init", sampleRate: context.sampleRate });

    const microphone = context.createMediaStreamSource(microphoneStream);
    captureNode = new AudioWorkletNode(context, "bytewaves-capture");
    const silentGain = context.createGain();
    silentGain.gain.value = 0;
    captureNode.port.onmessage = (event: MessageEvent<Float32Array>) => {
      decoderWorker?.postMessage({ type: "samples", samples: event.data }, [event.data.buffer]);
    };
    microphone.connect(captureNode).connect(silentGain).connect(context.destination);
    mode = "listening";
    receiveButton.innerHTML = '<span class="button-icon">■</span><span>Stop listening</span>';
    receivePanel.classList.add("receiving");
    decodedCard.classList.add("hidden");
    setStatus("Listening for the training signal", "active");
  } catch (error) {
    stopAll();
    setStatus(microphoneError(error), "error");
  }
}

function handleWorkerMessage(message: MessageEvent<{ type: string; events?: DecoderEvent[]; message?: string }>): void {
  if (message.data.type === "error") {
    setStatus(message.data.message ?? "Decoder error", "error");
    return;
  }
  for (const event of message.data.events ?? []) {
    if (event.type === "synchronized") setStatus("Signal found—receiving frame", "active");
    if (event.type === "frameProgress") setStatus(`Receiving ${event.received} / ${event.total} protected bytes`, "active");
    if (event.type === "decodeFailed") setStatus("Frame damaged; waiting for the next repetition", "active");
    if (event.type === "messageDecoded") {
      decodedText.textContent = event.text;
      decodedCard.classList.remove("hidden");
      setStatus("Message decoded successfully");
    }
  }
}

function stopAll(): void {
  sourceNode?.stop();
  sourceNode?.disconnect();
  sourceNode = undefined;
  captureNode?.disconnect();
  captureNode = undefined;
  microphoneStream?.getTracks().forEach((track) => track.stop());
  microphoneStream = undefined;
  decoderWorker?.terminate();
  decoderWorker = undefined;
  mode = "idle";
  sendButton.innerHTML = '<span class="button-icon">▶</span><span>Start broadcasting</span>';
  receiveButton.innerHTML = '<span class="button-icon">●</span><span>Start listening</span>';
  receivePanel.classList.remove("receiving");
}

sendTab.addEventListener("click", () => showTab("send"));
receiveTab.addEventListener("click", () => showTab("receive"));
message.addEventListener("input", updateByteCount);
sendButton.addEventListener("click", startSending);
receiveButton.addEventListener("click", startListening);
window.addEventListener("pagehide", stopAll);

registerSW({ immediate: true });
init()
  .then(() => {
    ready = true;
    updateByteCount();
  })
  .catch((error) => setStatus(`Could not load modem: ${error}`, "error"));
