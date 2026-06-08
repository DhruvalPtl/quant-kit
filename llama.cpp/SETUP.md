# llama.cpp Setup Guide

Place your llama.cpp binaries in THIS folder.

## What to Download

1. Go to: https://github.com/ggerganov/llama.cpp/releases/latest

2. Download (for Windows with Intel Arc / AMD / NVIDIA):
   `llama-b*-bin-win-vulkan-x64.zip`

3. Extract the ZIP here. You should have:
   - llama-quantize.exe  ← used by quantize.py
   - llama-cli.exe       ← used by benchmark.py
   - (and many other .exe files)

4. Download convert_hf_to_gguf.py separately:
   https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py
   Save it here as: llama.cpp/convert_hf_to_gguf.py

5. Install convert script dependencies (in your venv):
   .venv\Scripts\pip install gguf numpy

## Final folder structure:
```
llama.cpp/
├── llama-quantize.exe      ← required
├── llama-cli.exe           ← required for benchmark.py
├── convert_hf_to_gguf.py  ← required
├── (other .exe files)
└── SETUP.md                ← this file
```

## Test it works:
```
.\llama.cpp\llama-quantize.exe --help
```
You should see the help output listing quant types.
