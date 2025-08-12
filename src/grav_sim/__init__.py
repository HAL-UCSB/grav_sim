import os
import pathlib
from typing import ClassVar
from importlib import resources

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    assets: pathlib.PurePosixPath
    mano_assets: pathlib.PurePosixPath
    ycb_aff_assets: pathlib.PurePosixPath
    rom_csv: pathlib.PurePosixPath
    hand_segments_assets: pathlib.PurePosixPath

    @computed_field
    @property
    def assets_path(self)->pathlib.Path:
        return pathlib.Path(self.assets)

    @computed_field
    @property
    def mano_assets_path(self) -> pathlib.Path:
        return pathlib.Path(self.mano_assets)

    @computed_field
    @property
    def ycb_aff_assets_path(self) -> pathlib.Path:
        return pathlib.Path(self.ycb_aff_assets)

    @computed_field
    @property
    def rom_csv_path(self) -> pathlib.Path:
        return pathlib.Path(self.rom_csv)

    @computed_field
    @property
    def hand_segments_assets_path(self) -> pathlib.Path:
        return pathlib.Path(self.hand_segments_assets)

    _ENV_VAR_GRASPR_DOT_ENV: ClassVar[str] = 'GRAV_DOT_ENV'
    _DEFAULT_GRASPR_DOT_ENV: ClassVar[pathlib.Path] = resources.files('grav_sim') / 'default.env'

    model_config = SettingsConfigDict(
        env_file=os.environ.get(
            _ENV_VAR_GRASPR_DOT_ENV,
            _DEFAULT_GRASPR_DOT_ENV),
        env_file_encoding='utf-8',
        extra='ignore')


settings = Settings()
