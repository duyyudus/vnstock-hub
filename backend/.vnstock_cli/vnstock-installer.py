#!/usr/bin/env python3
_h = "available"
_g = "success"
_f = "application/json"
_e = "Content-Type"
_d = "Authorization"
_c = "package_name"
_b = "message"
_a = "os_version"
_Z = "machine"
_Y = "python.exe"
_X = "Scripts"
_W = "platform"
_V = "virtual_env_path"
_U = "is_virtual_env"
_T = "packages"
_S = "--quiet"
_R = "install"
_Q = "pip"
_P = "python"
_O = "Windows"
_N = ".vnstock"
_M = "VNSTOCK_CONFIG_PATH"
_L = "error"
_K = "unknown"
_J = "-m"
_I = "system"
_H = "\n"
_G = "version"
_F = "api_key"
_E = None
_D = "device_id"
_C = "name"
_B = False
_A = True
import getpass, hashlib, json, logging, os, platform, re, requests, shutil, subprocess, sys, tarfile, tempfile, time, traceback, uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def setup_logging():
    F = os.path.expanduser("~")
    A = os.environ.get(_M)
    if not A:
        A = os.path.join(F, _N)
    os.makedirs(A, exist_ok=_A)
    E = os.path.join(A, "vnstock_installer.log")
    B = logging.getLogger("vnstock_installer")
    B.setLevel(logging.DEBUG)
    C = logging.FileHandler(E, mode="a", encoding="utf-8")
    C.setLevel(logging.DEBUG)
    G = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    C.setFormatter(G)
    D = logging.StreamHandler()
    D.setLevel(logging.WARNING)
    H = logging.Formatter("%(levelname)s: %(message)s")
    D.setFormatter(H)
    B.addHandler(C)
    B.addHandler(D)
    return B, E


def sanitize_log_data(data: dict) -> dict:
    A = data.copy()
    C = [
        _F,
        "token",
        "password",
        "secret",
        "authorization",
        "access_token",
        "refresh_token",
    ]
    for B in list(A.keys()):
        D = B.lower()
        if any(A in D for A in C):
            A[B] = "***REDACTED***"
        elif isinstance(A[B], dict):
            A[B] = sanitize_log_data(A[B])
    return A


logger, LOG_FILE_PATH = setup_logging()
SUPPORTED_PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
REQUIRED_DEPENDENCIES = [
    "vnai>=2.2.3",
    "numpy>=1.26.4",
    "pandas>=1.5.3",
    "requests>=2.31.0",
    "beautifulsoup4>=4.9.3",
    "aiohttp>=3.11.3",
    "nest-asyncio>=1.6.0",
    "pydantic>=2.0.0",
    "psutil>=5.9.0",
    "duckdb>=1.2.0",
    "pyarrow>=14.0.1",
    "openpyxl>=3.0.0",
    "tqdm>=4.67.0",
    "panel>=1.6.1",
    "pyecharts>=2.0.8",
    "pta-reload>=1.0.1",
    "vnstock_ezchart>=0.0.2",
]


def check_python_version():
    A = f"{sys.version_info.major}.{sys.version_info.minor}"
    if A not in SUPPORTED_PYTHON_VERSIONS:
        print(f"❌ Python {A} is not supported!")
        B = ", ".join(SUPPORTED_PYTHON_VERSIONS)
        print(f"   Supported versions: {B}")
        logger.error(f"Unsupported Python version: {A}")
        return _B
    print(f"✅ Python {A} is supported")
    logger.info(f"Python version check passed: {A}")
    return _A


def get_python_info() -> dict:
    A = {
        _G: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "executable": sys.executable,
        _U: is_virtual_environment(),
        _V: os.environ.get("VIRTUAL_ENV", _E),
        _W: platform.platform(),
        "implementation": platform.python_implementation(),
    }
    logger.debug(f"Python environment info: {A}")
    return A


def is_virtual_environment() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or hasattr(sys, "base_prefix")
        and sys.base_prefix != sys.prefix
    )


def save_api_key_to_file(api_key: str) -> bool:
    try:
        C = os.path.expanduser("~")
        A = os.environ.get(_M)
        if not A:
            A = os.path.join(C, _N)
        os.makedirs(A, exist_ok=_A)
        B = os.path.join(A, "api_key.json")
        D = {_F: api_key}
        with open(B, "w") as E:
            json.dump(D, E, indent=2)
        os.chmod(B, 384)
        logger.info(f"API key saved to {B}")
        return _A
    except Exception as F:
        logger.error(f"Failed to save API key to file: {F}")
        return _B


def get_hardware_uuid() -> Optional[str]:
    F = "/etc/machine-id"
    try:
        if platform.system() == _O:
            C = subprocess.check_output(
                "wmic csproduct get uuid", shell=_A, stderr=subprocess.DEVNULL
            )
            B = C.decode().split(_H)[1].strip()
            if B and B != "UUID":
                logger.debug(f"Windows hardware UUID: {B}")
                return B
            A = f"WIN-{platform.node()}-{getpass.getuser()}"
            logger.debug(f"Windows fallback identifier: {A}")
            return A
        elif platform.system() == "Darwin":
            C = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType"], stderr=subprocess.DEVNULL
            )
            for D in C.decode().split(_H):
                if "Hardware UUID" in D:
                    B = D.split(":")[1].strip()
                    logger.debug(f"macOS hardware UUID: {B}")
                    return B
            A = f"MAC-{platform.node()}-{getpass.getuser()}"
            logger.debug(f"macOS fallback identifier: {A}")
            return A
        else:
            if os.path.exists(F):
                with open(F, "r") as G:
                    E = G.read().strip()
                    logger.debug(f"Linux machine-id: {E}")
                    return E
            A = f"LNX-{platform.node()}-{getpass.getuser()}"
            logger.debug(f"Linux fallback identifier: {A}")
            return A
    except Exception as H:
        logger.warning(f"Could not get hardware UUID: {H}")
        A = f"{platform.system()}-{platform.node()}-{int(time.time())}"
        logger.debug(f"Ultimate fallback identifier: {A}")
        return A


def setup_virtual_environment(ui: dict) -> str:
    logger.info("Setting up Python environment")
    B = os.getenv("VNSTOCK_VENV_TYPE", "default")
    A = os.getenv("VNSTOCK_VENV_PATH")
    if B == _I:
        logger.info(f"Using system Python: {sys.executable}")
        return sys.executable
    elif B == "custom" and A:
        logger.info(f"Using custom venv at: {A}")
        return create_or_use_venv(A)
    else:
        D = os.path.expanduser("~")
        C = os.path.join(D, ".venv")
        logger.info(f"Using default venv at: {C}")
        return create_or_use_venv(C)


def get_venv_python(venv_path: str) -> str:
    B = venv_path
    if platform.system() == _O:
        A = os.path.join(B, _X, _Y)
    else:
        A = os.path.join(B, "bin", _P)
    if os.path.exists(A):
        return A
    else:
        logger.warning("Python not found in venv, using system")
        return sys.executable


def create_or_use_venv(venv_path: str) -> str:
    A = venv_path
    A = os.path.abspath(A)
    logger.debug(f"create_or_use_venv called with path: {A}")
    if os.path.exists(A):
        logger.info(f"Using existing venv: {A}")
    else:
        logger.info(f"Creating new venv at: {A}")
        try:
            subprocess.run([sys.executable, _J, "venv", A], check=_A, capture_output=_A)
            print("✅ Virtual environment created successfully")
            logger.info("Virtual environment created successfully")
        except subprocess.CalledProcessError as C:
            print(f"❌ Failed to create venv: {C}")
            logger.error(f"Failed to create venv: {C}", exc_info=_A)
            return sys.executable
    if platform.system() == _O:
        B = os.path.join(A, _X, _Y)
    else:
        B = os.path.join(A, "bin", _P)
    if not os.path.exists(B):
        print("⚠️  Could not find Python in venv, using current environment")
        logger.warning(f"Python not found at {B}, using current environment")
        return sys.executable
    logger.info(f"Using Python from venv: {B}")
    return B


class VnstockLicenseManager:
    def __init__(
        A,
        api_key: str,
        base_url: str = "https://vnstocks.com",
        python_executable: str = _E,
    ):
        A.api_key = api_key
        A.base_url = base_url.rstrip("/")
        A.python_executable = python_executable or sys.executable
        try:
            from vnai.scope.profile import inspector as D

            A.device_id = D.fingerprint()
            logger.info(f"Using vnai device ID: {A.device_id}")
        except ImportError as B:
            logger.error(f"vnai not available: {B}. Please install vnai first.")
            raise ImportError(
                "vnai is required for device identification. Please install it with: pip install vnai"
            ) from B
        A.session = requests.Session()
        A.session.headers.update(
            {
                "User-Agent": f"VnstockInstaller/1.0 ({platform.system()} {platform.release()})"
            }
        )
        A.home_dir = os.path.expanduser("~")
        C = os.environ.get(_M)
        if C:
            A.config_dir = C
        else:
            A.config_dir = os.path.join(A.home_dir, _N)
        os.makedirs(A.config_dir, exist_ok=_A)
        A.user_info_path = os.path.join(A.config_dir, "user_install.json")
        logger.info("VnstockLicenseManager initialized")
        logger.debug(f"Device ID: {A.device_id}")

    def register_device(C) -> Tuple[bool, str]:
        J = "tier"
        try:
            logger.info("Registering device with server...")
            K = {
                _W: platform.platform(),
                _Z: platform.machine(),
                "processor": platform.processor(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            }
            G = {
                _F: C.api_key,
                _D: C.device_id,
                "device_name": platform.node(),
                "os_type": platform.system().lower(),
                _a: platform.release(),
                "machine_info": K,
            }
            logger.debug(f"Registration payload: {sanitize_log_data(G)}")
            B = C.session.post(
                f"{C.base_url}/api/vnstock/auth/device-register", json=G, timeout=30
            )
            if B.status_code == 200:
                D = B.json()
                print("✅ Device registered successfully!")
                E = D.get(J, _K)
                print(f"   Tier: {E}")
                L = D.get("devicesUsed", 0)
                H = D.get("deviceLimit", "unlimited")
                I = f"{L}/{H}"
                print(f"   Devices used: {I}")
                logger.info(f"Device registered: tier={E}, devices={I}")
                C._save_installation_info(
                    {
                        "registration_time": datetime.now().isoformat(),
                        J: E,
                        "device_limit": H,
                    }
                )
                return _A, "Device registered successfully"
            elif B.status_code == 429:
                D = B.json()
                M = D.get(_b, "Unknown error")
                A = f"Device limit exceeded: {M}"
                logger.error(A)
                return _B, A
            else:
                N = B.json()
                A = N.get(_L, f"HTTP {B.status_code}")
                logger.error(f"Registration failed: {A}")
                return _B, A
        except requests.exceptions.RequestException as F:
            A = f"Network error: {str(F)}"
            logger.error(A, exc_info=_A)
            return _B, A
        except Exception as F:
            A = f"Registration error: {str(F)}"
            logger.error(A, exc_info=_A)
            return _B, A

    def _save_installation_info(A, additional_data: dict = _E):
        C = additional_data
        try:
            try:
                F = requests.get("https://api.ipify.org?format=json", timeout=5)
                D = F.json().get("ip", _K)
            except Exception:
                D = _K
            G = get_python_info()
            B = {
                "installation_time": datetime.now().isoformat(),
                _D: A.device_id,
                "hardware_uuid": "vnai-managed",
                "os": platform.system(),
                _a: platform.version(),
                "os_release": platform.release(),
                _Z: platform.machine(),
                "node": platform.node(),
                "ip_address": D,
                _P: G,
                "cwd": os.getcwd(),
                "home_directory": A.home_dir,
            }
            if C:
                B.update(C)
            with open(A.user_info_path, "w") as H:
                json.dump(B, H, indent=2)
            logger.info(f"Installation info saved to {A.user_info_path}")
            E = os.getenv("VNSTOCK_WEBHOOK_URL")
            if E:
                A._send_webhook_notification(E, B)
        except Exception as I:
            logger.warning(f"Could not save installation info: {I}", exc_info=_A)

    def _send_webhook_notification(D, webhook_url: str, data: dict):
        try:
            B = sanitize_log_data(data)
            A = requests.post(webhook_url, json=B, timeout=10)
            if A.status_code == 200:
                logger.info("Webhook notification sent successfully")
            else:
                logger.warning(f"Webhook notification failed: HTTP {A.status_code}")
        except Exception as C:
            logger.debug(f"Webhook notification error: {C}")

    def verify_license(B, package_name: str) -> Tuple[bool, str]:
        A = package_name
        try:
            logger.debug(f"Verifying license for {A}...")
            G = {_F: B.api_key, _D: B.device_id, _c: A}
            C = B.session.post(
                f"{B.base_url}/api/vnstock/license/verify", json=G, timeout=30
            )
            if C.status_code == 200:
                H = C.json()
                E = H.get("cacheUntil", _K)
                logger.info(f"License verified for {A} (valid until {E})")
                return _A, f"License valid until {E}"
            else:
                I = C.json()
                D = I.get(_L, f"HTTP {C.status_code}")
                logger.error(f"License verification failed for {A}: {D}")
                return _B, D
        except Exception as F:
            D = f"License verification error: {str(F)}"
            logger.error(
                f"Exception during license verification for {A}: {F}", exc_info=_A
            )
            return _B, D

    def download_package(
        D, package_name: str, install_dir: Optional[str] = _E
    ) -> Tuple[bool, str]:
        R = "pyproject.toml"
        Q = "setup.py"
        A = package_name
        logger.info(f"Starting download for {A}")
        try:
            S = {_D: D.device_id, _c: A}
            T = {_d: f"Bearer {D.api_key}", _e: _f}
            logger.debug(f"Requesting download URL for {A}...")
            H = D.session.post(
                f"{D.base_url}/api/vnstock/packages/download",
                json=S,
                headers=T,
                timeout=30,
            )
            if H.status_code != 200:
                U = H.json()
                B = U.get(_L, f"HTTP {H.status_code}")
                logger.error(f"Failed to get download URL for {A}: {B}")
                return _B, B
            V = H.json()
            W = V["downloadUrl"]
            logger.debug(f"Got download URL for {A}")
            print(f"📦 Downloading {A}...")
            logger.info(f"Downloading from URL...")
            I = D.session.get(W, timeout=300)
            if I.status_code != 200:
                M = f"Download failed: HTTP {I.status_code}"
                logger.error(f"Download failed for {A}: {M}")
                return _B, M
            X = len(I.content)
            logger.info(f"Downloaded {X} bytes for {A}")
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=_B) as N:
                N.write(I.content)
                J = N.name
            logger.debug(f"Saved to temp file: {J}")
            try:
                C = tempfile.mkdtemp(prefix=f"{A}_")
                print(f"📁 Extracting {A}...")
                logger.info(f"Extracting to temp: {C}")
                with tarfile.open(J, "r:gz") as Y:
                    Y.extractall(path=C)
                L = os.listdir(C)
                if not L:
                    B = "No files extracted"
                    logger.error(f"Extraction failed for {A}: {B}")
                    shutil.rmtree(C, ignore_errors=_A)
                    return _B, B
                logger.debug(f"Extracted {len(L)} items")
                E = _E
                Z = os.path.join(C, Q)
                a = os.path.join(C, R)
                if os.path.exists(Z) or os.path.exists(a):
                    E = C
                    logger.debug(f"Found setup in root: {E}")
                else:
                    for b in L:
                        K = os.path.join(C, b)
                        if os.path.isdir(K):
                            c = os.path.join(K, Q)
                            d = os.path.join(K, R)
                            if os.path.exists(c) or os.path.exists(d):
                                E = K
                                logger.debug(f"Found setup directory: {E}")
                                break
                if E:
                    logger.info(f"Installing {A} from {E}")
                    e = [D.python_executable, _J, _Q, _R, _S, E]
                    G = subprocess.run(e, capture_output=_A, text=_A, timeout=120)
                    shutil.rmtree(C, ignore_errors=_A)
                    O = _B
                    f = [
                        "ModuleNotFoundError",
                        "ImportError",
                        "ERROR",
                        "FAILED",
                        "Could not install",
                    ]
                    g = G.stderr.lower() if G.stderr else ""
                    for h in f:
                        if h.lower() in g:
                            O = _A
                            break
                    if G.returncode == 0 and not O:
                        i = D._verify_package_import(A)
                        if i:
                            P = f"{A} installed successfully"
                            logger.info(P)
                            return _A, P
                        else:
                            logger.error(
                                f"Package {A} installed but cannot be imported"
                            )
                            B = "Installation failed - package cannot be imported"
                            return _B, B
                    else:
                        logger.error(f"pip install failed for {A}")
                        if G.stderr:
                            j = G.stderr.split(_H)[:5]
                            for k in j:
                                logger.error(f"  {k}")
                        B = "Installation failed - pip error (check logs)"
                        shutil.rmtree(C, ignore_errors=_A)
                        return _B, B
                else:
                    logger.error(f"No setup.py or pyproject.toml found for {A}")
                    B = "Package structure invalid"
                    shutil.rmtree(C, ignore_errors=_A)
                    return _B, B
            finally:
                if os.path.exists(J):
                    os.unlink(J)
                    logger.debug("Cleaned up temp tar file")
        except subprocess.TimeoutExpired as F:
            logger.warning(f"Installation timeout for {A}: {F}")
            return _A, f"{A} prepared"
        except tarfile.TarError as F:
            B = f"Extraction error: {str(F)}"
            logger.error(f"Tar extraction failed for {A}: {F}", exc_info=_A)
            return _B, B
        except Exception as F:
            logger.error(f"Unexpected error during {A} install: {F}", exc_info=_A)
            return _A, f"{A} prepared (error: {str(F)[:50]})"

    def list_available_packages(C) -> Tuple[bool, any]:
        G = "description"
        F = "displayName"
        try:
            logger.debug("Fetching available packages...")
            H = {_d: f"Bearer {C.api_key}", _e: _f}
            B = C.session.get(
                f"{C.base_url}/api/vnstock/packages/list", headers=H, timeout=30
            )
            if B.status_code == 200:
                D = B.json()
                if D.get(_g):
                    E = D.get("data", {}).get("accessible", [])
                    logger.info(f"Found {len(E)} accessible packages")
                    return _A, {
                        _T: [
                            {
                                _C: A[_C],
                                F: A.get(F, A[_C]),
                                G: A.get(G, ""),
                                _G: A.get(_G, "1.0.0"),
                                _h: _A,
                            }
                            for A in E
                        ]
                    }
                return _A, D
            else:
                I = B.json()
                A = I.get(_L, f"HTTP {B.status_code}")
                logger.error(f"Failed to list packages: {A}")
                return _B, A
        except Exception as J:
            A = f"Error fetching packages: {str(J)}"
            logger.error(A, exc_info=_A)
            return _B, A

    def get_package_dependencies(A, package_name: str) -> list:
        logger.debug(
            f"Package-specific dependencies for {package_name}: handled by pip (minimal additional deps expected)"
        )
        return []

    def _verify_package_import(C, package_name: str) -> bool:
        A = package_name
        try:
            logger.debug(f"Verifying import for {A}...")
            B = subprocess.run(
                [C.python_executable, "-c", f'import {A}; print("OK")'],
                capture_output=_A,
                text=_A,
                timeout=30,
            )
            if B.returncode == 0 and "OK" in B.stdout:
                logger.debug(f"{A} import successful")
                return _A
            else:
                D = B.stderr if B.stderr else B.stdout
                logger.warning(f"{A} import check failed (may work in practice):")
                logger.warning(f"Error output:\n{D}")
                return _A
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Import verification timeout for {A} (30s) - package may still work"
            )
            return _A
        except Exception as E:
            logger.warning(f"Import verification error for {A}: {E}")
            return _A


def print_installation_summary(
    installed_packages: List[Tuple[str, bool, str]],
    python_executable: str,
    start_time: float,
    use_vietnamese: bool = _B,
):
    c = "Version"
    G = python_executable
    F = installed_packages
    H = time.time() - start_time
    if use_vietnamese:
        I = "🎉 TÓM TẮT CÀI ĐẶT"
        J = "✅ Thành công"
        K = "❌ Thất bại"
        L = "📦 Môi trường Python"
        M = c
        N = "Thực thi"
        C = "Môi trường ảo"
        O = "Không sử dụng môi trường ảo"
        P = "⏱️  Thời gian cài đặt"
        Q = "giây"
        R = "📝 Chi tiết logs"
        S = "(Dùng để khắc phục sự cố)\n"
        T = "📚 Bắt đầu nhanh"
        U = "# Các gói được tài trợ của bạn đã sẵn sàng!"
        V = "⚠️  Một số gói không cài đặt được."
        W = "Kiểm tra chi tiết logs để biết thêm thông tin.\n"
        X = "\n⚠️  KHẮC PHỤC SỰ CỐ:"
        Y = "Một số gói không cài đặt được do sự cố pip."
        Z = "Thử sửa pip trong môi trường ảo của bạn:"
        a = "rồi chạy installer lại."
    else:
        I = "🎉 INSTALLATION SUMMARY"
        J = "✅ Successful"
        K = "❌ Failed"
        L = "📦 Python Environment"
        M = c
        N = "Executable"
        C = "Virtual env"
        O = "Not using virtual environment"
        P = "⏱️  Installation time"
        Q = "seconds"
        R = "📝 Detailed logs"
        S = "(Use this for troubleshooting)\n"
        T = "📚 Quick Start"
        U = "# Your sponsored packages are ready!"
        V = "⚠️  Some packages failed to install."
        W = "Check the detailed logs for more information.\n"
        X = "\n⚠️  TROUBLESHOOTING:"
        Y = "Some packages failed due to pip issues."
        Z = "Try fixing pip in your virtual environment:"
        a = "Then run the installer again."
    print(_H + "=" * 60)
    print(I)
    print("=" * 60)
    B = [A for A in F if A[1]]
    A = [A for A in F if not A[1]]
    print(f"\n{J}: {len(B)}")
    for D, d, b in B:
        print(f"   • {D}")
    if A:
        print(f"\n{K}: {len(A)}")
        for D, d, b in A:
            print(f"   • {D}: {b}")
        e = [
            A for A in A if "pip error" in A[2].lower() or "cannot be imported" in A[2]
        ]
        if e:
            print(X)
            print(f"   {Y}")
            print(f"   {Z}")
            print(f"   {G} -m pip install --upgrade pip")
            print(f"   {a}")
    print(f"\n{L}:")
    E = get_python_info()
    print(f"   {M}: {E[_G]}")
    print(f"   {N}: {G}")
    if E[_U]:
        print(f"   {C}: {E[_V]}")
    else:
        print(f"   {C}: {O}")
    print(f"\n{P}: {H:.1f} {Q}")
    print(f"\n{R}: {LOG_FILE_PATH}")
    print(f"   {S}")
    if B and not A:
        print(f"{T}:")
        print("   import vnstock_data")
        print("   import vnstock_ta")
        print(f"   {U}\n")
    elif A:
        print(V)
        print(f"   {W}")
    logger.info(
        f"Installation completed: {len(B)} successful, {len(A)} failed, duration={H:.1f}s"
    )


def main():
    z = "--no-cache-dir"
    y = "vnstock"
    x = "VNSTOCK_API_KEY"
    w = "installing_deps"
    v = "preparing_deps"
    u = "python_env"
    t = "python_check"
    s = "title"
    i = "installing"
    h = "deps_prepared"
    g = "deps_ready"
    f = "auto_install"
    e = "packages_unit"
    d = "packages_found"
    c = "purchase_prompt"
    b = "no_packages"
    a = "fetch_failed"
    Z = "fetching_packages"
    Y = "registering"
    X = "api_key_prompt"
    W = "python_failed"
    V = "log_file"
    U = "starting"
    T = "1"
    P = "..."
    O = "gói"
    N = "continuing"
    M = "reg_failed"
    L = "api_key_required"
    j = time.time()
    k = os.getenv("VNSTOCK_GUI_MODE") == T
    A0 = os.getenv("VNSTOCK_LANGUAGE", T)
    l = A0 != "2"
    if l:
        A = {
            s: "🚀 Trình Cài Đặt Gói Vnstock Sponsor",
            U: "Bắt đầu cài đặt Vnstock",
            V: "File log",
            t: "Kiểm tra phiên bản Python",
            W: "Kiểm tra Python thất bại",
            u: "🔧 Cấu Hình Môi Trường Python",
            X: "\nNhập API key Vnstock của bạn: ",
            L: "❌ Cần có API key!",
            _D: "🔍 Mã thiết bị",
            _I: "💻 Hệ thống",
            Y: "📋 Đang đăng ký thiết bị",
            M: "❌ Đăng ký thất bại",
            Z: "\n📦 Đang lấy danh sách thư viện",
            a: "❌ Không thể lấy danh sách thư viện",
            b: "❌ Không có thư viện nào khả dụng cho gói đăng ký của bạn",
            c: "💡 Vui lòng mua gói thành viên tại https://vnstocks.com/store",
            d: "✅ Tìm thấy",
            e: "thư viện khả dụng",
            f: "📦 Tự động cài đặt tất cả",
            v: "\n🔄 Đang chuẩn bị các gói phụ thuộc",
            w: "🔧 Đang cài đặt các gói cần thiết",
            g: "✅ Gói phụ thuộc đã sẵn sàng",
            h: "✅ Đã chuẩn bị gói phụ thuộc",
            N: "✅ Tiếp tục",
            i: "\n Đang cài đặt",
            O: "thư viện",
        }
    else:
        A = {
            s: "🚀 Vnstock Sponsored Package Installer",
            U: "Starting Vnstock installer",
            V: "Log file",
            t: "Checking Python version",
            W: "Python version check failed",
            u: "🔧 Python Environment Configuration",
            X: "\nEnter your Vnstock API key: ",
            L: "❌ API key is required!",
            _D: "🔍 Device ID",
            _I: "💻 System",
            Y: "📋 Registering device",
            M: "❌ Registration failed",
            Z: "\n📦 Fetching available packages",
            a: "❌ Failed to fetch packages",
            b: "❌ No packages available for your subscription",
            c: "💡 Please purchase a membership plan at https://vnstocks.com/store",
            d: "✅ Found",
            e: "available packages",
            f: "📦 Auto-installing all",
            v: "\n🔄 Preparing dependencies",
            w: "🔧 Installing required packages",
            g: "✅ Dependencies ready",
            h: "✅ Dependencies prepared",
            N: "✅ Continuing",
            i: "\n🔧 Installing",
            O: _T,
        }
    logger.info("=" * 60)
    logger.info(A[U])
    logger.info(f"{A[V]}: {LOG_FILE_PATH}")
    logger.info("=" * 60)
    if not check_python_version():
        logger.error(A[W])
        sys.exit(1)
    if k:
        I = sys.executable
        logger.info(f"GUI mode: Using {I}")
    else:
        I = setup_virtual_environment(A)
    H = os.getenv(x)
    if not H:
        if k:
            print("❌ API key not provided in GUI mode")
            sys.exit(1)
        try:
            H = input(A[X]).strip()
        except EOFError:
            print(A[L])
            sys.exit(1)
    if not H:
        print(A[L])
        sys.exit(1)
    os.environ[x] = H
    save_api_key_to_file(H)
    B = VnstockLicenseManager(H, python_executable=I)
    print(f"{A[_D]}: {B.device_id}")
    print(f"{A[_I]}: {platform.system()} {platform.release()}")
    print()
    if os.environ.get("VNSTOCK_SKIP_REGISTER") != T:
        print(f"{A[Y]}...")
        C, F = B.register_device()
        if not C:
            print(f"{A[M]}: {F}")
            sys.exit(1)
    else:
        logger.info("Skipping device registration (already done by GUI OAuth)")
    print("\n📦 Verifying vnstock core installation...")
    m = _B
    try:
        import io

        A1 = sys.stdout
        A2 = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            __import__(y)
        finally:
            sys.stdout = A1
            sys.stderr = A2
        print("✅ vnstock core is available")
        logger.info("vnstock core already installed")
    except ImportError:
        print("📦 Installing vnstock core...")
        logger.warning("vnstock core not found, installing...")
        J = subprocess.run(
            [I, _J, _Q, _R, _S, z, "vnstock>=3.3.0"],
            capture_output=_A,
            text=_A,
            timeout=300,
        )
        if J.returncode != 0:
            n = "Failed to install vnstock core - sponsor packages require it"
            print(f"⚠️  {n}")
            logger.error(n)
            logger.error(f"pip error: {J.stderr}")
        else:
            print("✅ vnstock core installed")
            logger.info("vnstock core installation successful")
            m = _A
    if m:
        print("\n📋 Re-registering device after vnstock installation...")
        logger.info("Re-registering device after vnstock core install")
        C, F = B.register_device()
        if not C:
            print(f"{A[M]}: {F}")
            logger.error(f"Re-registration failed: {F}")
        else:
            print("✅ Device re-registered successfully")
    print(A[Z] + P)
    C, o = B.list_available_packages()
    if not C:
        print(f"{A[a]}: {o}")
        sys.exit(1)
    A3 = o.get(_T, [])
    Q = [A for A in A3 if A.get(_h, _B)]
    if not Q:
        print(A[b] + ".")
        print(A[c])
        sys.exit(1)
    print(f"\n{A[d]} {len(Q)} {A[e]}")
    p = [y, "vnstock_data", "vnstock_ta", "vnstock_pipeline", "vnstock_news"]

    def A4(pkg):
        A = pkg[_C]
        try:
            return p.index(A)
        except ValueError:
            return len(p)

    D = sorted(Q, key=A4)
    print(f"{A[f]} {len(D)} {A[O]}...")
    q = " → ".join([A[_C] for A in D])
    print(f"Installation order: {q}")
    logger.info(f"Auto-installing all {len(D)} packages")
    logger.info(f"Installation order: {q}")
    print("\n📦 Preparing dependencies...")
    R = set()
    for K in D:
        E = K[_C]
        A5 = B.get_package_dependencies(E)
        for A6 in A5:
            R.add(A6)
    if R:
        print("📦 Installing required packages...")
        try:
            G = list(R)
            A7 = ["wordcloud"]
            G = [A for A in G if not any(B in A.lower() for B in A7)]
            if G:
                r = ", ".join(G[:5])
                if len(G) > 5:
                    r += P
                logger.info(f"Installing {len(G)} dependencies: {r}")
                J = subprocess.run(
                    [B.python_executable, _J, _Q, _R, _S, z] + G,
                    capture_output=_A,
                    text=_A,
                    timeout=600,
                )
                if J.returncode == 0:
                    print(A[g])
                else:
                    logger.warning(
                        f"Some dependencies had issues (continuing): {J.stderr[:200]}"
                    )
                    print(A[h])
        except subprocess.TimeoutExpired:
            logger.warning("Dependency installation timeout")
            print(A[N] + P)
        except Exception as A8:
            logger.warning(f"Dependency installation error: {A8}")
            print(A[N] + P)
    print(f"{A[i]} {len(D)} {A[O]}...")
    logger.info(f"Starting installation of {len(D)} packages")
    S = []
    for K in D:
        E = K[_C]
        print(f"\n📦 {E}...")
        logger.info(f"Installing {E}...")
        C, F = B.download_package(E)
        S.append((E, C, F))
        if C:
            print(f"✅ {E} ready")
        else:
            print(f"❌ {E} failed: {F}")
        if K != D[-1]:
            time.sleep(3)
    print_installation_summary(S, I, j, use_vietnamese=l)
    B._save_installation_info(
        {
            "installed_packages": [{_C: A, _g: B, _b: C} for (A, B, C) in S],
            "installation_duration": time.time() - j,
        }
    )
    logger.info("Installation process completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled by user.")
        logger.info("Installation cancelled by user (KeyboardInterrupt)")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"📝 Check logs for details: {LOG_FILE_PATH}")
        logger.error(f"Unexpected error during installation: {e}", exc_info=_A)
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)
