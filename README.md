# Historical Text Dating 



## Overview 


## Development 

### Configurations 
This project uses `hydra` for configuration management. The default configuration file is located at `configs/defaults.yaml`.
You can override the default configuration by passing command line arguments. 
If there's a permanent change you want to make, change the config files in the `configs` directory directly.

To run the project on mac (windows requires other path), use the following command:
```bash
PYTHONPATH="$PWD/src:$PYTHONPATH" python src/main.py
```

If you wish to override default configs, override according to the config file structure. For example, this will override the logging level
to `DEBUG`:

```bash
PYTHONPATH="$PWD/src:$PYTHONPATH" python src/main.py hydra.job_logging.root.level=DEBUG
```
There's also an option to add configurations on the fly using `+`
```bash
PYTHONPATH="$PWD/src:$PYTHONPATH" python src/main.py +model.new_key=value
````

### Tracker 
This project uses `wandb` for tracking.
To login with your credentials, run the following command:
```bash
wandb login
``` 
You can set the `WANDB_MODE` environment variable to control the tracking mode, or control it from hydra.
By default, it is set to `online`, which means it will log to the cloud.
If you wish to disable tracking, simply run with
```bash
PYTHONPATH="$PWD/src:$PYTHONPATH" python src/main.py tracker.mode=disabled
```
Or
```bash
PYTHONPATH="$PWD/src:$PYTHONPATH" WANDB_MODE=disabled python src/main.py
```


### Formatter 
If you wish, you can use `black` to format the code upon pre-commit. 
To install the dev-dependencies, run:
```bash
pip-sync requirements-dev.txt
```

To enable pre-commit hooks, run:
```bash
pre-commit install
```

If you just want to run `black` manually, you can do so with:
```bash
python -m black src
```

