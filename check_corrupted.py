from pathlib import Path
p = Path('output/gemma-4-12b-it')
min_size = 100 * 1024 * 1024  # 100 MB
for f in sorted(p.glob('*.gguf')):
    size = f.stat().st_size
    is_valid = size > min_size
    status = "VALID" if is_valid else "CORRUPTED -> will be deleted and re-quantized"
    print(f"{f.name}: {size/1e6:.1f} MB -> {status}")
