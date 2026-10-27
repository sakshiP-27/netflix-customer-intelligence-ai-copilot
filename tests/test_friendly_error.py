"""A failed startup must not show the underlying error text."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

PAGE = """
import streamlit as st
from components.helpers import prepare_app

st.set_page_config(page_title="Customer Intelligence", layout="wide")
prepare_app()
st.write("loaded")
"""


class FriendlyErrorTests(unittest.TestCase):
    def test_database_failure_stays_generic(self):
        hidden = "HIDDEN-TOKEN"
        app = AppTest.from_string(PAGE, default_timeout=30)
        with patch("components.helpers.get_engine", side_effect=RuntimeError(hidden)):
            app.run()
        messages = [item.value for item in app.error]
        self.assertTrue(messages)
        self.assertIn("could not be loaded", messages[0])
        self.assertNotIn(hidden, messages[0])
        self.assertFalse(any(getattr(item, "value", "") == "loaded" for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
