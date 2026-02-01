"""Property-based tests for StructuredLogger.

Feature: ml-lambda-deployment, Property 10: Logs Estructurados son JSON Válido
"""

import json
import logging
from datetime import datetime
from io import StringIO

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from ml_lambda.utils.logging import StructuredLogger


# Strategy for generating valid log messages (ASCII only for simplicity)
message_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"), whitelist_characters=" "),
    min_size=0,
    max_size=100
)

# Strategy for log levels
level_strategy = st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR"])

# Simplified strategy for extra kwargs
extra_kwargs_strategy = st.fixed_dictionaries({}).flatmap(
    lambda _: st.dictionaries(
        keys=st.sampled_from(["request_id", "latency_ms", "user_id", "count", "flag"]),
        values=st.one_of(
            st.integers(min_value=-1000, max_value=1000),
            st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=10),
            st.booleans(),
        ),
        max_size=3
    )
)


@pytest.fixture
def logger_with_capture():
    """Create a logger with captured output."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    
    logger = StructuredLogger("property_test_logger", level="DEBUG")
    logger._logger.handlers.clear()
    logger._logger.addHandler(handler)
    
    return logger, stream


class TestLoggingProperties:
    """Property 10: Logs Estructurados son JSON Válido.
    
    **Validates: Requirements 11.1, 11.2**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(message=message_strategy, level=level_strategy, extra=extra_kwargs_strategy)
    def test_logs_are_valid_json(self, message, level, extra):
        """
        Property 10: Para cualquier evento de log emitido por el sistema,
        debe ser JSON válido (parseable).
        
        **Validates: Requirements 11.1, 11.2**
        """
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        
        logger = StructuredLogger("prop_test", level="DEBUG")
        logger._logger.handlers.clear()
        logger._logger.addHandler(handler)
        
        # Call the appropriate log method based on level
        log_method = getattr(logger, level.lower())
        log_method(message, **extra)
        
        output = stream.getvalue().strip()
        
        # Must be valid JSON
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    @settings(max_examples=100)
    @given(message=message_strategy, level=level_strategy)
    def test_logs_contain_required_fields(self, message, level):
        """
        Property 10: Para cualquier evento de log, debe contener campos:
        timestamp, level, message.
        
        **Validates: Requirements 11.1, 11.2**
        """
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        
        logger = StructuredLogger("prop_test", level="DEBUG")
        logger._logger.handlers.clear()
        logger._logger.addHandler(handler)
        
        log_method = getattr(logger, level.lower())
        log_method(message)
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        # Must contain required fields
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "message" in parsed

    @settings(max_examples=100)
    @given(message=message_strategy, level=level_strategy)
    def test_timestamp_is_iso8601(self, message, level):
        """
        Property 10: El timestamp debe ser formato ISO 8601.
        
        **Validates: Requirements 11.1, 11.2**
        """
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        
        logger = StructuredLogger("prop_test", level="DEBUG")
        logger._logger.handlers.clear()
        logger._logger.addHandler(handler)
        
        log_method = getattr(logger, level.lower())
        log_method(message)
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        # Timestamp must be valid ISO 8601
        timestamp = parsed["timestamp"]
        datetime.fromisoformat(timestamp)  # Raises if invalid

    @settings(max_examples=100)
    @given(message=message_strategy, level=level_strategy)
    def test_level_is_valid(self, message, level):
        """
        Property 10: El level debe ser uno de: DEBUG, INFO, WARNING, ERROR.
        
        **Validates: Requirements 11.1, 11.2**
        """
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        
        logger = StructuredLogger("prop_test", level="DEBUG")
        logger._logger.handlers.clear()
        logger._logger.addHandler(handler)
        
        log_method = getattr(logger, level.lower())
        log_method(message)
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        # Level must be one of the valid levels
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        assert parsed["level"] in valid_levels
