# interactive-music-performer
Interactive music performer based on FastAPI & online time warping (OLTW) algorithm

```bash
# create env
$ conda create --name <env> --file requirements.txt

# rebuild db
$ sqlite3 ./sql_app.db < initial_data.sql

# start app
$ uvicorn app.main:app
```

Go to 127.0.0.1:8000/docs to test APIs