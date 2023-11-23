import os
import logging

import streamlit as st

def load_env() -> None:
    assert os.environ["OPENAI_API_KEY"]
    logging.basicConfig(level=logging.INFO)