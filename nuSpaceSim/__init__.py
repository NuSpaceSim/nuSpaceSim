import subprocess
sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
sha = sha.decode('ascii').strip()

major_minor_patch = '0.1.2'

__version__ = f'{major_minor_patch}-{sha}'
