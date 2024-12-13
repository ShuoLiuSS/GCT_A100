#!/usr/bin/env python

import sys
import torch
import platform

def check_system():
    print("System Information:")
    print("-" * 50)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Operating System: {platform.platform()}")
    print(f"PyTorch Version: {torch.__version__}")
    print("-" * 50)

def check_cuda():
    print("\nCUDA Information:")
    print("-" * 50)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {device.major}.{device.minor}")
            print(f"  Total Memory: {device.total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA is not available - Using CPU only")
        print("\nCPU Information:")
        print(f"Number of CPU threads: {torch.get_num_threads()}")
    print("-" * 50)

if __name__ == "__main__":
    try:
        check_system()
        check_cuda()
        print("\nCheck completed successfully!")
    except Exception as e:
        print(f"\nError during check: {str(e)}")
