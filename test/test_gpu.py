import subprocess
import platform


if __name__ == "__main__":
    print("=" * 55)
    print("GPU DETECTION TEST")
    print("=" * 55)

    print(f"Operating System: {platform.system()} {platform.release()}")

    print("\n[1] Detecting installed GPUs...")

    try:
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name, DriverVersion, AdapterRAM | "
            "Format-List",
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        print(result.stdout)

    except Exception as e:
        print(f"GPU detection failed: {e}")

    print("\n[2] Checking PyTorch GPU support...")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

            x = torch.rand(1000, 1000, device="cuda")
            y = torch.matmul(x, x)

            print("GPU computation: SUCCESS")
            print(f"Result device: {y.device}")
        else:
            print("CUDA unavailable (normal for Intel Iris Xe).")

    except ImportError:
        print("PyTorch not installed. Skipping.")

    print("\n[3] Checking OpenCL devices...")

    try:
        import pyopencl as cl

        for platform_info in cl.get_platforms():
            print(f"Platform: {platform_info.name}")

            for device in platform_info.get_devices():
                print(f"  Device: {device.name}")
                print(f"  Type: {cl.device_type.to_string(device.type)}")
                print(f"  Memory: {device.global_mem_size / 1024**3:.2f} GB")

    except ImportError:
        print("PyOpenCL not installed. Skipping.")
    except Exception as e:
        print(f"OpenCL detection error: {e}")

    print("\n" + "=" * 55)
    print("GPU TEST COMPLETED")
    print("=" * 55)
