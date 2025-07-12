# src/my_app/__main__.py
from pathlib import Path
import sys
from streamlit.web import cli as stcli

def main():
    script = Path(__file__).with_name('app') / 'ui.py'
    sys.argv = ['streamlit', 'run', str(script)]
    sys.exit(stcli.main())
