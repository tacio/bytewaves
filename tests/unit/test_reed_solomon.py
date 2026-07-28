import numpy as np
import pytest

from bytewaves._core import decode_packet, encode_packet
from bytewaves.modem import MAX_PAYLOAD_BYTES, text_to_sound


class TestReedSolomon:
    def test_packet_round_trip(self):
        packet = encode_packet("Hello 世界 🌍")
        text, corrected = decode_packet(packet)
        assert text == "Hello 世界 🌍"
        assert corrected == 0

    def test_corrects_eight_bad_bytes(self):
        packet = bytearray(encode_packet("eight correctable errors"))
        for index in range(3, 11):
            packet[index] ^= 0x5A
        text, corrected = decode_packet(packet)
        assert text == "eight correctable errors"
        assert corrected == 8

    def test_rejects_corrupt_header(self):
        packet = bytearray(encode_packet("header"))
        packet[2] ^= 1
        with pytest.raises(ValueError, match="header"):
            decode_packet(packet)

    def test_utf8_payload_limit(self):
        assert len(text_to_sound("a" * MAX_PAYLOAD_BYTES)) > 0
        assert text_to_sound("a" * (MAX_PAYLOAD_BYTES + 1)).size == 0
        assert text_to_sound("é" * MAX_PAYLOAD_BYTES).size == 0

    def test_waveform_is_float32(self):
        wave = text_to_sound("ECC Test")
        assert isinstance(wave, np.ndarray)
        assert wave.dtype == np.float32
        assert np.isfinite(wave).all()
