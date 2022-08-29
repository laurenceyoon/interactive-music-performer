# interactive-music-performer
Interactive music performer based on FastAPI & online time warping (OLTW) algorithm

Tested on Python 3.10 (conda)

```bash
$ pyenv install miniforge3 && pyenv activate miniforge3
$ conda env create -f environment.yml
$ conda activate imp
```

if there's portaudio installation issue on Mac M1, please refer to [here](https://stackoverflow.com/a/68822818)

```bash
# rebuild db
$ sqlite3 ./sql_app.db < initial_data.sql

# start app
$ uvicorn app.main:app --reload
```

Go to 127.0.0.1:8000 to test APIs
