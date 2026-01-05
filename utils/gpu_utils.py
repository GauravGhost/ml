#!/usr/bin/env python3
"""
GPU Utilities for Biometric Classification
Enhanced cross-platform GPU detection for Mac, Windows, and Linux
Designed to work seamlessly across platforms without code changes
"""

import platform
import sys
import os
import tensorflow as tf

def setup_gpu_acceleration():
    """
    Enhanced GPU detection and setup for Mac, Windows, and Linux
    Automatically detects platform and configures optimal GPU settings
    
    Returns:
        bool: True if GPU acceleration is available, False otherwise
    """
    system = platform.system().lower()
    
    print(f"🖥️  Detected OS: {platform.system()}")
    print(f"🔍 Python version: {sys.version}")
    print(f"🧠 TensorFlow version: {tf.__version__}")
    
    if system == "darwin":  # macOS
        print("🍎 Configuring Mac GPU (Apple Metal)...")
        return _setup_mac_gpu()
    
    elif system == "windows":  # Windows
        print("🪟 Configuring Windows GPU (NVIDIA CUDA)...")
        return _setup_windows_gpu()
    
    elif system == "linux":  # Linux
        print("🐧 Configuring Linux GPU (NVIDIA CUDA)...")
        return _setup_linux_gpu()
    
    else:
        print(f"❓ Unsupported OS: {system}")
        print("⚠️  Using CPU (GPU detection not implemented for this OS)")
        return False

def _setup_mac_gpu():
    """Setup GPU acceleration for macOS with comprehensive detection"""
    try:
        # Method 1: Check for modern GPU devices (TensorFlow 2.5+)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Configure GPU memory growth to prevent allocation issues
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                print(f"✅ Mac GPU enabled: {len(gpus)} GPU(s) found")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
                
                # Verify GPU is actually working with a test operation
                return _verify_gpu_functionality()
                
            except Exception as e:
                print(f"⚠️  GPU setup warning: {e}")
        
        # Method 2: Fallback to experimental method
        try:
            experimental_gpus = tf.config.experimental.list_physical_devices('GPU')
            if experimental_gpus:
                for gpu in experimental_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Mac GPU (experimental) enabled: {len(experimental_gpus)} GPU(s)")
                return _verify_gpu_functionality()
        except Exception as e:
            print(f"⚠️  Experimental GPU detection: {e}")
        
        # Method 3: Try to detect if tensorflow-metal is installed
        try:
            import tensorflow_metal
            print("✅ TensorFlow Metal plugin detected")
            return _verify_gpu_functionality()
        except ImportError:
            print("⚠️  TensorFlow Metal not installed")
        
        print("❌ No Apple GPU acceleration available")
        return False
        
    except Exception as e:
        print(f"❌ Mac GPU detection error: {e}")
        return False

def _setup_windows_gpu():
    """Setup GPU acceleration for Windows with NVIDIA CUDA"""
    try:
        # Check for NVIDIA GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Configure GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                print(f"✅ Windows GPU enabled: {len(gpus)} GPU(s) found")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
                
                # Verify GPU functionality
                return _verify_gpu_functionality()
                
            except RuntimeError as e:
                print(f"❌ GPU setup error: {e}")
                print("💡 Try restarting your environment or check CUDA installation")
                return False
        
        # Try experimental method as fallback
        try:
            exp_gpus = tf.config.experimental.list_physical_devices('GPU')
            if exp_gpus:
                for gpu in exp_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Windows GPU (experimental) enabled: {len(exp_gpus)} GPU(s)")
                return _verify_gpu_functionality()
        except Exception as e:
            print(f"⚠️  Experimental GPU detection: {e}")
        
        print("❌ No NVIDIA GPU found")
        return False
        
    except Exception as e:
        print(f"❌ Windows GPU detection error: {e}")
        return False

def _setup_linux_gpu():
    """Setup GPU acceleration for Linux with NVIDIA CUDA"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                print(f"✅ Linux GPU enabled: {len(gpus)} GPU(s) found")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
                
                return _verify_gpu_functionality()
                
            except RuntimeError as e:
                print(f"❌ Linux GPU setup error: {e}")
                return False
        
        print("❌ No GPU found on Linux")
        return False
        
    except Exception as e:
        print(f"❌ Linux GPU detection error: {e}")
        return False

def _verify_gpu_functionality():
    """Verify that GPU is actually functional with a test operation"""
    try:
        # Try to perform a simple GPU operation
        with tf.device('/GPU:0'):
            # Create test tensors and perform matrix multiplication
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
            c = tf.matmul(a, b)
            
            # Force execution
            result = c.numpy()
            
        print("✅ GPU functionality verified with test operation")
        return True
        
    except Exception as e:
        print(f"⚠️  GPU verification failed: {e}")
        print("⚠️  GPU detected but not functional, falling back to CPU")
        return False

def print_gpu_setup_guidance(gpu_available):
    """Print comprehensive setup guidance based on platform and GPU availability"""
    system = platform.system().lower()
    
    if gpu_available:
        print("🚀 GPU acceleration is ENABLED!")
        print("⚡ Training will be significantly faster")
        print("🎯 Optimal performance configured for your platform")
    else:
        print("🐌 Using CPU (slower but reliable)")
        print("💡 To enable GPU acceleration:")
        
        if system == "darwin":  # macOS
            print("\n🍎 For Mac (Apple Silicon M1/M2/M3):")
            print("   1. Install tensorflow-metal:")
            print("      pip install tensorflow-metal")
            print("   2. Ensure you have Apple Silicon Mac (not Intel)")
            print("   3. Restart your Python environment")
            
        elif system == "windows":  # Windows
            print("\n🪟 For Windows (NVIDIA GPU):")
            print("   1. Install NVIDIA GPU drivers from nvidia.com")
            print("   2. Install CUDA Toolkit (check TensorFlow compatibility)")
            print("   3. Install cuDNN library")
            print("   4. Restart your computer")
            print("   5. Verify with: nvidia-smi")
            
        elif system == "linux":  # Linux
            print("\n🐧 For Linux (NVIDIA GPU):")
            print("   1. Install NVIDIA drivers: sudo apt install nvidia-driver-xxx")
            print("   2. Install CUDA Toolkit")
            print("   3. Install cuDNN library")
            print("   4. Set up LD_LIBRARY_PATH")
            print("   5. Verify with: nvidia-smi")
        
        print("\n🔄 After GPU setup, restart your environment and run again")

def get_device_info():
    """Get comprehensive information about available compute devices"""
    print("\n📊 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   TensorFlow: {tf.__version__}")
    
    print("\n📱 Available TensorFlow devices:")
    try:
        devices = tf.config.list_physical_devices()
        if devices:
            for device in devices:
                print(f"   📱 {device}")
        else:
            print("   ⚠️  No physical devices detected")
            
        # Try to get logical devices too
        logical_devices = tf.config.list_logical_devices()
        if logical_devices:
            print("\n🧮 Logical devices:")
            for device in logical_devices:
                print(f"   🧮 {device}")
                
    except Exception as e:
        print(f"   ❌ Error listing devices: {e}")
    
    return devices if 'devices' in locals() else []

if __name__ == "__main__":
    # Test the GPU setup
    print("🧪 Testing GPU Detection System...")
    gpu_available = setup_gpu_acceleration()
    print_gpu_setup_guidance(gpu_available)
    get_device_info()

def _setup_mac_gpu():
    """Setup GPU acceleration for macOS"""
    try:
        # First, check for standard GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Configure GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Mac GPU enabled: {len(gpus)} GPU(s) found")
                
                # Print GPU details
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
                return True
            except Exception as e:
                print(f"⚠️  Mac GPU setup warning: {e}")
        
        # Check for Metal Performance Shaders (older method)
        try:
            if hasattr(tf.config.experimental, 'list_physical_devices'):
                experimental_gpus = tf.config.experimental.list_physical_devices('GPU')
                if experimental_gpus:
                    try:
                        for gpu in experimental_gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"✅ Mac GPU (experimental) enabled: {len(experimental_gpus)} GPU(s) found")
                        return True
                    except Exception as e:
                        print(f"⚠️  Experimental GPU setup warning: {e}")
        except Exception as e:
            print(f"⚠️  Experimental GPU detection: {e}")
        
        # Try to verify if Metal is working by creating a simple operation
        try:
            # Test if we can create tensors on GPU
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test_tensor, test_tensor)
                print("✅ Metal GPU acceleration verified and working!")
                return True
        except Exception as e:
            print(f"⚠️  Metal GPU verification failed: {e}")
        
        print("⚠️  No Apple GPU acceleration available")
        print("💡 Install tensorflow-metal: pip install tensorflow-metal")
        print("💡 Ensure you have Apple Silicon (M1/M2/M3) Mac")
        return False
        
    except Exception as e:
        print(f"❌ Mac GPU detection error: {e}")
        print("⚠️  Falling back to CPU")
        return False

def _setup_windows_gpu():
    """Setup GPU acceleration for Windows"""
    try:
        # NVIDIA CUDA setup for Windows
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Configure GPU memory growth to avoid allocation issues
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Optional: Set memory limit if needed
                # tf.config.experimental.set_memory_limit(gpus[0], 2048)  # 2GB limit
                
                print(f"✅ NVIDIA GPU enabled: {len(gpus)} GPU(s) found")
                
                # Print GPU details
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
                
                return True
                
            except RuntimeError as e:
                print(f"❌ NVIDIA GPU setup error: {e}")
                print("💡 Try restarting Python or check CUDA installation")
                return False
        else:
            print("⚠️  No NVIDIA GPU found")
            print("💡 Ensure NVIDIA drivers and CUDA are properly installed")
            return False
            
    except Exception as e:
        print(f"❌ Windows GPU detection error: {e}")
        print("⚠️  Falling back to CPU")
        return False

def _setup_linux_gpu():
    """Setup GPU acceleration for Linux"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Linux GPU enabled: {len(gpus)} GPU(s) found")
                
                # Print GPU details
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
                
                return True
            except RuntimeError as e:
                print(f"❌ Linux GPU setup error: {e}")
                return False
        else:
            print("⚠️  No GPU found on Linux")
            print("💡 Ensure NVIDIA drivers and CUDA are properly installed")
            return False
    except Exception as e:
        print(f"❌ Linux GPU detection error: {e}")
        return False

def print_gpu_setup_guidance(gpu_available):
    """Print helpful guidance based on GPU availability"""
    if gpu_available:
        print("🚀 Training will use GPU acceleration")
        print("⚡ Expected significant speedup compared to CPU")
    else:
        print("🐌 Training will use CPU (slower but will work)")
        print("💡 For faster training, ensure proper GPU setup:")
        
        system = platform.system()
        if system == "Darwin":
            print("   - Mac: Install TensorFlow-Metal with: pip install tensorflow-metal")
            print("   - Mac: Ensure you have Apple Silicon (M1/M2/M3) for best performance")
        elif system == "Windows":
            print("   - Windows: Install NVIDIA drivers from nvidia.com")
            print("   - Windows: Install CUDA Toolkit (compatible with your TensorFlow version)")
            print("   - Windows: Install cuDNN library")
        elif system == "Linux":
            print("   - Linux: Install NVIDIA drivers")
            print("   - Linux: Install CUDA Toolkit and cuDNN")
        
        print("   - Restart Python/VS Code after GPU setup")
        print("   - Training will still work on CPU, just slower")

def get_device_info():
    """Get information about available compute devices"""
    devices = tf.config.list_physical_devices()
    print("\n📊 Available TensorFlow devices:")
    for device in devices:
        print(f"   {device}")
    return devices

if __name__ == "__main__":
    # Test the GPU setup
    gpu_available = setup_gpu_acceleration()
    print_gpu_setup_guidance(gpu_available)
    get_device_info()