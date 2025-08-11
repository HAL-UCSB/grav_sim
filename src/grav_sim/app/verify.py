import io
import os
import pathlib
from io import BytesIO
from zipfile import ZipFile
from urllib import request

import gdown
import streamlit as st

from grav_sim import settings, Settings


def _rm_rf(path: pathlib.Path):
    if type(path) is str:
        path = pathlib.Path(path)
    if path.is_dir():
        for sub in path.iterdir():
            _rm_rf(sub)
        path.rmdir()
    else:
        path.unlink()


def download_zip(zip_url):
    response = request.urlopen(zip_url)
    zip_data = io.BytesIO(response.read())
    return ZipFile(zip_data)


def resolve_rom_csv():
    if st.button('Download ROM CSV'):
        url = r'https://github.com/HAL-UCSB/grav_sim/blob/main/assets/eatonhand_rom.csv'
        with request.urlopen(url) as response:
            content = response.read()
            with settings.rom_csv.open('wb') as f:
                f.write(content)
                st.rerun()


def resolve_hand_segments():
    if st.button('Download Hand Segments'):
        url = r'https://raw.githubusercontent.com/HAL-UCSB/grav_sim/refs/heads/main/assets/hand_segments.zip'
        hand_segments_zip = download_zip(url)
        hand_segments_zip.extractall(settings.assets)
        unzipped = settings.assets / hand_segments_zip.filelist[0].filename
        st.rerun()


def resolve_mano():
    instructions_url = 'https://github.com/lixiny/manotorch?tab=readme-ov-file#download-mano-pickle-data-structures'
    st.markdown(f'[Download Instructions]({instructions_url})')
    mano_zip_path = pathlib.Path(st.text_input('MANO zip path:'))
    if not mano_zip_path.exists() or not mano_zip_path.match('*.zip'):
        st.error(f'{mano_zip_path} must be an existing zip file')
    elif st.button(f'Unzip {mano_zip_path.stem}'):
        with ZipFile(mano_zip_path) as mano_zip:
            mano_zip.extractall(settings.assets)
            unzipped = settings.assets / mano_zip.filelist[0].filename
            settings.mano_assets = unzipped.rename(settings.assets / 'mano')
            unnecessary_files = '._.DS_Store', 'webuser', '__init__.py'
            for file in unnecessary_files:
                _rm_rf(settings.mano_assets / file)
            st.rerun()


def resolve_ycb_aff():
    # https://github.com/enriccorona/YCB_Affordance?tab=readme-ov-file#download-data
    models_zip_path = settings.assets / 'models.zip'
    if st.button('Download YCB Affordances'):
        with st.spinner('Downloading YCB Affordances Models'):
            gdown.download(
                id='1FdAWKpZTJBYctLNOZmlXGP7FGhE4etf0',
                output=str(models_zip_path.absolute()))

        with st.spinner('Unzipping Models'):
            with ZipFile(models_zip_path) as models_zip:
                print(models_zip_path)
                models_zip.extractall(settings.ycb_aff_assets)

        with st.spinner('Clonning YCB_Affordance Repo'):
            ycb_aff_repo_zip_url = 'https://github.com/enriccorona/YCB_Affordance/archive/refs/heads/master.zip'
            ycb_aff_zip = download_zip(ycb_aff_repo_zip_url)
            ycb_aff_zip.extractall(settings.ycb_aff_assets)

        ycb_aff_repo_path = settings.ycb_aff_assets / ycb_aff_zip.filelist[0].filename
        grasps_path = ycb_aff_repo_path / 'data' / 'grasps'
        grasps_path.rename(settings.ycb_aff_assets / grasps_path.name)
        _rm_rf(models_zip_path)
        _rm_rf(ycb_aff_repo_path)

        st.rerun()


st.write(
    f'you can set a custom a path to a custom .env on the environment variable `{Settings._ENV_VAR_GRASPR_DOT_ENV}`')
st.write(f'`{Settings._ENV_VAR_GRASPR_DOT_ENV}={os.environ.get(Settings._ENV_VAR_GRASPR_DOT_ENV, None)}`')
settings_path = pathlib.Path(settings.model_config['env_file'])
st.markdown(f'Using settings located at `{settings_path.absolute()}`')
st.code(settings_path.read_text(), language='bash')
settings.assets.mkdir(exist_ok=True)

st.markdown(f'ROM CSV {"✅" if settings.rom_csv.exists() else "❌"}')
if not settings.rom_csv.exists():
    resolve_rom_csv()

st.markdown(f'Hand Segments {"✅" if settings.hand_segments_assets.exists() else "❌"}')
if not settings.hand_segments_assets.exists():
    resolve_hand_segments()

st.markdown(f'MANO {"✅" if settings.mano_assets.exists() else "❌"}')
if not settings.mano_assets.exists():
    resolve_mano()

st.markdown(f'YCB Affordances {"✅" if settings.ycb_aff_assets.exists() else "❌"}')
if not settings.ycb_aff_assets.exists():
    resolve_ycb_aff()
