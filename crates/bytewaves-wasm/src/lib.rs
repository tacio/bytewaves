use bytewaves_core::StreamingDecoder;
use wasm_bindgen::prelude::*;

fn js_error(error: impl ToString) -> JsValue {
    js_sys::Error::new(&error.to_string()).into()
}

#[wasm_bindgen(js_name = encodeText)]
pub fn encode_text(text: &str, sample_rate: u32) -> Result<Vec<f32>, JsValue> {
    bytewaves_core::encode_text(text, sample_rate).map_err(js_error)
}

#[wasm_bindgen(js_name = maxPayloadBytes)]
pub fn max_payload_bytes() -> usize {
    bytewaves_core::MAX_PAYLOAD_BYTES
}

#[wasm_bindgen(js_name = Decoder)]
pub struct WasmDecoder {
    inner: StreamingDecoder,
}

#[wasm_bindgen(js_class = Decoder)]
impl WasmDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: u32) -> Result<WasmDecoder, JsValue> {
        Ok(Self {
            inner: StreamingDecoder::new(sample_rate).map_err(js_error)?,
        })
    }

    #[wasm_bindgen(js_name = pushSamples)]
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.push_samples(samples)).map_err(js_error)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}
