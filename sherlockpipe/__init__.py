__version__ = "1.2.2"

import shutil
import sys
import os
import subprocess

# Determine the path to libellc.so
ellc_path = os.path.join(os.path.dirname(__file__), 'ellc')
lib_path = os.path.join(ellc_path, 'ellc','libellc.so')
# Check if it exists
if not os.path.exists(lib_path):
    print("[ellc] libellc.so not found, running make...")
    try:
        subprocess.check_call(['make', '-B'], cwd=ellc_path)
        shutil.copy(ellc_path + '/libellc.so', os.path.join(ellc_path, 'ellc') + '/libellc.so')
        print("[ellc] libellc.so built successfully.")
    except Exception as e:
        print(f"Could not build libellc.so. Please ensure make and dependencies are available: {e}")

#Patching ellc with submodule
import sherlockpipe.ellc.ellc as _mypackage_ellc

# Override the 'ellc' name in sys.modules to point to your internal package module
sys.modules['ellc'] = _mypackage_ellc

# Patching all errors due to SSL certificates
import requests
import warnings
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Original request method
_original_request = requests.Session.request
# Overwrite to enforce verify=False by default
def no_verify_request(self, *args, **kwargs):
    kwargs.setdefault("verify", False)  # solo cambia si no se pasa ya 'verify'
    return _original_request(self, *args, **kwargs)
# Apply patch
requests.Session.request = no_verify_request
