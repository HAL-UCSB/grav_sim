# src/my_app/__main__.py
from pathlib import Path
import sys
from streamlit.web import cli as stcli

def main():
    target_script = f'{sys.argv[-1] if len(sys.argv) > 1 else "simulate"}.py'
    script = Path(__file__).with_name('app') / target_script
    sys.argv = ['streamlit', 'run', str(script)]
    sys.exit(stcli.main())
