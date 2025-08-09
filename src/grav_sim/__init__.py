import os
import pathlib
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    assets: pathlib.Path
    mano_assets: pathlib.Path
    ycb_aff_assets: pathlib.Path
    rom_csv: pathlib.Path
    hand_segments_assets: pathlib.Path

    _ENV_VAR_GRASPR_DOT_ENV: ClassVar[str] = 'GRASPR_DOT_ENV'
    _DEFAULT_GRASPR_DOT_ENV: ClassVar[pathlib.Path] = pathlib.Path('default.env')

    model_config = SettingsConfigDict(
        env_file=os.environ.get(
            _ENV_VAR_GRASPR_DOT_ENV,
            _DEFAULT_GRASPR_DOT_ENV),
        env_file_encoding='utf-8',
        extra='ignore')


settings = Settings()



