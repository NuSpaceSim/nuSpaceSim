import subprocess
sha = subprocess.getoutput("git rev-parse --short HEAD").strip()

major_minor_patch = '0.1.3'

__version__ = f'{major_minor_patch}'
