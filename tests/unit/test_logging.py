"""Unit tests for StructuredLogger."""

import json
import logging
from io import StringIO

import pytest

from ml_lambda.utils.logging import StructuredLogger


@pytest.fixture
def capture_logs():
    """Fixture to capture log output."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return stream, handler


@pytest.fixture
def logger_with_capture(capture_logs):
    """Logger with captured output."""
    stream, handler = capture_logs
    logger = StructuredLogger("test_logger", level="DEBUG")
    # Clear existing handlers and add our capture handler
    logger._logger.handlers.clear()
    logger._logger.addHandler(handler)
    return logger, stream


class TestStructuredLoggerOutput:
    """Tests for JSON output format."""

    def test_info_outputs_valid_json(self, logger_with_capture):
        """Verify info() outputs valid JSON."""
        logger, stream = logger_with_capture
        logger.info("test message")
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        assert isinstance(parsed, dict)

    def test_error_outputs_valid_json(self, logger_with_capture):
        """Verify error() outputs valid JSON."""
        logger, stream = logger_with_capture
        logger.error("error message")
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        assert isinstance(parsed, dict)

    def test_debug_outputs_valid_json(self, logger_with_capture):
        """Verify debug() outputs valid JSON."""
        logger, stream = logger_with_capture
        logger.debug("debug message")
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        assert isinstance(parsed, dict)

    def test_warning_outputs_valid_json(self, logger_with_capture):
        """Verify warning() outputs valid JSON."""
        logger, stream = logger_with_capture
        logger.warning("warning message")
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        assert isinstance(parsed, dict)


class TestStructuredLoggerFields:
    """Tests for required fields in log output."""

    def test_info_contains_required_fields(self, logger_with_capture):
        """Verify info() contains timestamp, level, message."""
        logger, stream = logger_with_capture
        logger.info("test message")
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "message" in parsed
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "test message"

    def test_error_contains_required_fields(self, logger_with_capture):
        """Verify error() contains timestamp, level, message."""
        logger, stream = logger_with_capture
        logger.error("error occurred")
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "message" in parsed
        assert parsed["level"] == "ERROR"
        assert parsed["message"] == "error occurred"

    def test_debug_contains_required_fields(self, logger_with_capture):
        """Verify debug() contains timestamp, level, message."""
        logger, stream = logger_with_capture
        logger.debug("debug info")
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "message" in parsed
        assert parsed["level"] == "DEBUG"

    def test_warning_contains_required_fields(self, logger_with_capture):
        """Verify warning() contains timestamp, level, message."""
        logger, stream = logger_with_capture
        logger.warning("warning info")
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "message" in parsed
        assert parsed["level"] == "WARNING"

    def test_timestamp_is_iso_format(self, logger_with_capture):
        """Verify timestamp is ISO 8601 format."""
        from datetime import datetime
        
        logger, stream = logger_with_capture
        logger.info("test")
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        # Should parse without error
        datetime.fromisoformat(parsed["timestamp"])

    def test_extra_kwargs_included(self, logger_with_capture):
        """Verify extra kwargs are included in output."""
        logger, stream = logger_with_capture
        logger.info("test", request_id="123", latency_ms=45.2)
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        assert parsed["request_id"] == "123"
        assert parsed["latency_ms"] == 45.2

    def test_logger_name_included(self, logger_with_capture):
        """Verify logger name is included in output."""
        logger, stream = logger_with_capture
        logger.info("test")
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        assert "logger" in parsed
        assert parsed["logger"] == "test_logger"
