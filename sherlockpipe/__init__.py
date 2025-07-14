__version__ = "1.0.3"

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
