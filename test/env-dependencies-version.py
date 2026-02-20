import sys
import platform
import subprocess
import warnings

# ==================== 终极核心：导入torch前先配置全局过滤规则 ====================
# 规则1：全局模糊匹配警告核心关键词（覆盖所有模块）
warnings.filterwarnings(
    "ignore",
    message=r"CUDA capability sm_120 is not compatible",
    category=UserWarning
)
# 规则2：定向过滤torch.cuda模块的所有UserWarning（双重保险）
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.cuda"
)

# 此时再导入torch，过滤规则已生效，警告被直接拦截
import torch

# ==================== 强制初始化CUDA（触发剩余潜在警告并拦截）====================
# 提前执行CUDA初始化，让所有相关警告在检测前触发并被过滤
try:
    torch.cuda.is_available()
except:
    pass

def get_python_basic_info():
    """获取Python基础环境和系统信息"""
    py_full_ver = sys.version
    py_short_ver = sys.version.split()[0]
    os_info = platform.platform()
    arch_info = platform.architecture()[0]

    print("=" * 60)
    print("🔍 Python 基础环境信息")
    print("=" * 60)
    print(f"Python 完整版本：{py_full_ver}")
    print(f"Python 核心版本：{py_short_ver}")
    print(f"运行操作系统：{os_info}")
    print(f"系统架构：{arch_info}")
    print("=" * 60)

def get_single_pkg_ver(pkg_name):
    """获取单个Python包的版本（兼容常规包+Hugging Face生态包）"""
    try:
        # 方案1：优先使用importlib.metadata（Python 3.8+ 内置，最可靠）
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(pkg_name)
        except PackageNotFoundError:
            # 方案2：备用方案，兼容老版本Python
            import importlib
            pkg_module = importlib.import_module(pkg_name)
            return pkg_module.__version__ if hasattr(pkg_module, "__version__") else "⚠️  版本号未找到"
    
    except ImportError:
        return "❌ 未安装/导入失败"
    except AttributeError:
        return "⚠️  包无公开版本号"
    except Exception as e:
        return f"❌ 加载失败: {str(e)[:40]}..."

def get_cuda_info():
    """获取系统CUDA和nvcc的版本信息"""
    cuda_ver = "❌ 未安装/不可用"
    nvcc_ver = "❌ 未安装/不可用"

    # 检测nvcc版本
    try:
        result = subprocess.check_output(
            ["nvcc", "--version"], 
            stderr=subprocess.STDOUT, 
            text=True
        )
        for line in result.splitlines():
            if "release" in line:
                nvcc_ver = line.strip().split(",")[1].strip()
                break
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 检测系统CUDA Toolkit路径
    try:
        result = subprocess.check_output(
            ["ls", "/usr/local/"], 
            stderr=subprocess.STDOUT, 
            text=True
        )
        for line in result.splitlines():
            if line.startswith("cuda-"):
                cuda_ver = line.split("-")[1]
                break
    except:
        pass

    return cuda_ver, nvcc_ver

def check_pytorch_cuda_status():
    """PyTorch专属CUDA加速状态检测（核心GPU验证）"""
    print("\n⚡ PyTorch CUDA 加速状态检测")
    print("-" * 60)
    try:
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 可用状态    ：{'✅ 可用' if cuda_available else '❌ 不可用'}")
        if cuda_available:
            print(f"PyTorch绑定CUDA版本：{torch.version.cuda}")
            print(f"可用GPU设备数量  ：{torch.cuda.device_count()}")
            print(f"主GPU设备名称    ：{torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"❌ CUDA状态检测失败: {str(e)[:50]}...")
    print("-" * 60)

def batch_check_packages(pkg_list):
    """批量检查指定包的版本（统一管理，按自定义顺序显示）"""
    print("\n📦 第三方依赖包版本检测（含PyTorch生态）")
    print("-" * 60)
    # 直接按传入的列表顺序遍历，不再排序
    for pkg in pkg_list:
        ver = get_single_pkg_ver(pkg)
        print(f"{pkg.ljust(15)}: {ver}")

    # 单独检测系统级CUDA和nvcc
    cuda_ver, nvcc_ver = get_cuda_info()
    print(f"{'system_cuda'.ljust(15)}: {cuda_ver}")
    print(f"{'nvcc'.ljust(15)}: {nvcc_ver}")
    print("-" * 60)

if __name__ == "__main__":
    # 自定义固定顺序：把datasets和accelerate放在transformers之后
    CHECK_PACKAGES = [
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "datasets",
        "accelerate",
        "deepspeed",
        "langchain",
        "langchain-core",
        "langchain-community",
        "openai",
        "huggingface_hub",
        "llama-cpp-python",
        "langgraph",
        "jupyter",
        "protobuf",
        "chromadb",
        "opentelemetry-proto",
        "opentelemetry-exporter-otlp-proto-common",
        "opentelemetry-exporter-otlp-proto-grpc",
        "faiss-cpu",
        "pypdf",
        "python-docx",
        "tiktoken",
        "tokenizers",
        "sentence-transformers",
        "sentencepiece",
        "scikit-learn",
        "scipy",
        "jieba",
        "cpm_kernels",
        "nvitop"
        ]  
    # 按逻辑执行检测（无任何警告输出）
    get_python_basic_info()
    batch_check_packages(CHECK_PACKAGES)
    check_pytorch_cuda_status()
    print("\n✅ 所有版本检测完成！")

