"""Deja las DLLs de CUDA/cuDNN visibles para onnxruntime-gpu.

Los paquetes `nvidia-*` (cublas, cudnn, nvrtc) instalan sus .dll DENTRO del
venv, en carpetas tipo `.../site-packages/nvidia/cu13/bin/x86_64`. En Windows
esas carpetas NO quedan en el PATH, así que onnxruntime-gpu no encuentra
`cublasLt64_13.dll` y cae a CPU **en silencio**: el runtime dice "no la
encuentra" y sigue por CPU sin avisar, y uno cree que la GPU está andando.

Esta función agrega esas carpetas al PATH del proceso (y a los directorios de
DLLs si estamos en Windows/Python 3.8+). Se llama ANTES de crear la primera
sesión de onnxruntime. Si algo falla (no están los paquetes, por ejemplo en
CPU-only), no rompe nada: solo se sigue sin GPU.

Uso:
    from app import cuda_dlls
    cuda_dlls.preparar()
"""

import os
import sys
from pathlib import Path


def _carpetas_bin(base):
    """Devuelve las carpetas bin del venv que contienen .dll de CUDA/cuDNN."""
    nvidia = base / "nvidia"
    if not nvidia.is_dir():
        return []
    encontradas = []
    for sub in nvidia.iterdir():
        if not sub.is_dir() or sub.name == "__pycache__":
            continue
        # layouts vistos: nvidia/<pkg>/bin y nvidia/<pkg>/bin/x86_64
        for rel in ("bin", os.path.join("bin", "x86_64")):
            d = sub / rel
            if d.is_dir():
                encontradas.append(d)
    return encontradas


def preparar():
    """Pone las DLLs de CUDA/cuDNN del venv en el PATH del proceso.

    Idempotente: llamarla varias veces no acumula entradas repetidas.
    Devuelve las carpetas que agregó (para diagnóstico)."""
    try:
        import nvidia  # noqa: F401
    except Exception:
        return []

    base = Path(__file__).resolve().parent.parent / "venv" / "Lib" / "site-packages"
    # el paquete nvidia puede estar en otra ruta (venv no estándar); usar la real
    try:
        import nvidia as _nv
        base = Path(_nv.__file__).resolve().parent.parent
    except Exception:
        pass

    carpetas = _carpetas_bin(base)
    if not carpetas:
        return []

    partes = os.environ.get("PATH", "").split(os.pathsep)
    agregadas = []
    for d in carpetas:
        s = str(d)
        if s not in partes:
            partes.insert(0, s)
            agregadas.append(s)
        # Python 3.8+: registro explícito para que el loader de DLLs lo vea
        if hasattr(os, "add_dll_directory") and os.name == "nt":
            try:
                os.add_dll_directory(s)
            except Exception:
                pass
    if agregadas:
        os.environ["PATH"] = os.pathsep.join(partes)
    return agregadas


if __name__ == "__main__":
    agregadas = preparar()
    print("carpetas agregadas al PATH:", agregadas or "(ninguna; ¿no hay nvidia-*?)")
    try:
        import onnxruntime as ort
        print("providers disponibles:", ort.get_available_providers())
    except Exception as e:
        print("onnxruntime no disponible:", e)
