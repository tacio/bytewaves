export function utf8ByteLength(value: string): number {
  return new TextEncoder().encode(value).length;
}

export function microphoneError(error: unknown): string {
  if (error instanceof DOMException && error.name === "NotAllowedError") {
    return "Microphone permission was denied. Allow access in your browser settings and try again.";
  }
  if (error instanceof DOMException && error.name === "NotFoundError") {
    return "No microphone is available on this device.";
  }
  return error instanceof Error ? error.message : "The microphone could not be started.";
}

