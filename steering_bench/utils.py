import pathlib 

project_dir = pathlib.Path(__file__).resolve().parents[1]
assets_dir = project_dir / 'assets'
raw_dataset_dir = project_dir / 'assets' / 'raw_datasets'