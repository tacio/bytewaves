use bytewaves_core::{DecoderEvent, StreamingDecoder, MAX_PAYLOAD_BYTES};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction(signature = (text, sample_rate = 44_100))]
fn encode_text<'py>(
    py: Python<'py>,
    text: &str,
    sample_rate: u32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    bytewaves_core::encode_text(text, sample_rate)
        .map(|wave| wave.into_pyarray(py))
        .map_err(value_error)
}

#[pyfunction]
fn encode_packet(text: &str) -> PyResult<Vec<u8>> {
    bytewaves_core::encode_packet(text).map_err(value_error)
}

#[pyfunction]
fn decode_packet(packet: Vec<u8>) -> PyResult<(String, usize)> {
    bytewaves_core::decode_packet(&packet).map_err(value_error)
}

#[pyclass(name = "Decoder")]
struct PyDecoder {
    inner: StreamingDecoder,
}

#[pymethods]
impl PyDecoder {
    #[new]
    #[pyo3(signature = (sample_rate = 44_100))]
    fn new(sample_rate: u32) -> PyResult<Self> {
        Ok(Self {
            inner: StreamingDecoder::new(sample_rate).map_err(value_error)?,
        })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn push_samples_json(&mut self, samples: PyReadonlyArray1<'_, f32>) -> PyResult<String> {
        let contiguous = samples.as_slice().map_err(value_error)?;
        let events: Vec<DecoderEvent> = self.inner.push_samples(contiguous);
        serde_json::to_string(&events).map_err(value_error)
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(encode_text, module)?)?;
    module.add_function(wrap_pyfunction!(encode_packet, module)?)?;
    module.add_function(wrap_pyfunction!(decode_packet, module)?)?;
    module.add_class::<PyDecoder>()?;
    module.add("MAX_PAYLOAD_BYTES", MAX_PAYLOAD_BYTES)?;
    Ok(())
}
