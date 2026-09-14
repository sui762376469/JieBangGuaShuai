from pathlib import Path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
map_data_path = project_root /'fangshan.json'
showtimes_path = project_root / 'fangshan_showtimes.json'
print(map_data_path, showtimes_path)
