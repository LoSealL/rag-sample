"""Tests for project processing sanitization rules."""

import pytest

from sec_rag.project_processing import SensitiveInfoChecker, SensitiveInfoError


class TestSensitiveInfoChecker:
    def test_phone_detection_with_separators(self):
        checker = SensitiveInfoChecker()
        text = "owner_contact: +86 138-0013-8000"

        with pytest.raises(SensitiveInfoError):
            checker.check_text(text, context="unit-test")

    def test_numeric_constants_are_not_phone(self):
        checker = SensitiveInfoChecker()
        text = """
        #define SYSTEM_CLOCK_HZ     100000000UL
        #define PERIPHERAL_CLOCK_HZ 50000000UL
        #define DEFAULT_BAUD_RATE   115200
        #define UART_RX_BUFFER_SIZE 512
        """

        checker.check_text(text, context="unit-test")
