from io import BytesIO
from zipfile import ZipFile

import gdown
import requests
import streamlit as st

import pathlib

from grav_sim import settings


def _rm_rf(path: pathlib.Path):
    if type(path) is str:
        path = pathlib.Path(path)
    if path.is_dir():
        for sub in path.iterdir():
            _rm_rf(sub)
        path.rmdir()
    else:
        path.unlink()


def all_assets_exist():
    return settings.mano_assets.exists() and settings.ycb_aff_assets.exists()


def resolve_mano():
    instructions_url = 'https://github.com/lixiny/manotorch?tab=readme-ov-file#download-mano-pickle-data-structures'
    st.markdown(f'[Download Instructions]({instructions_url})')
    mano_zip_path = pathlib.Path(st.text_input('MANO zip path:'))
    if not mano_zip_path.exists() or not mano_zip_path.match('*.zip'):
        st.error(f'{mano_zip_path} must be an existing zip file')
    elif st.button('Unzip'):
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
    if st.button('Download'):
        with st.spinner('Downloading YCB Affordances Models'):
            gdown.download(
                id='1FdAWKpZTJBYctLNOZmlXGP7FGhE4etf0',
                output=str(models_zip_path.absolute()))
            settings.ycb_aff_assets = settings.assets / 'ycb_aff'

        with st.spinner('Unzipping Models'):
            with ZipFile(models_zip_path) as models_zip:
                print(models_zip_path)
                models_zip.extractall(settings.ycb_aff_assets)

        with st.spinner('Clonning YCB_Affordance Repo'):
            ycb_aff_repo_zip_url = 'https://github.com/enriccorona/YCB_Affordance/archive/refs/heads/master.zip'
            response = requests.get(ycb_aff_repo_zip_url)
            response_bytes = BytesIO(response.content)
            with ZipFile(response_bytes) as ycb_aff_zip:
                ycb_aff_zip.extractall(settings.ycb_aff_assets)
            ycb_aff_repo_path = settings.ycb_aff_assets / ycb_aff_zip.filelist[0].filename
            grasps_path = ycb_aff_repo_path / 'data' / 'grasps'
            grasps_path.rename(settings.ycb_aff_assets / grasps_path.name)
            _rm_rf(models_zip_path)
            _rm_rf(ycb_aff_repo_path)

        st.rerun()



settings_path = pathlib.Path(settings.model_config['env_file'])
st.markdown(f'Settings located at {settings_path.absolute()}')
st.json(settings.model_dump_json())
settings.assets.mkdir(exist_ok=True)

st.markdown(f'MANO {"✅" if settings.mano_assets.exists() else "❌"}')
if not settings.mano_assets.exists():
    resolve_mano()

st.markdown(f'YCB Affordances {"✅" if settings.ycb_aff_assets.exists() else "❌"}')
if not settings.ycb_aff_assets.exists():
    resolve_ycb_aff()
