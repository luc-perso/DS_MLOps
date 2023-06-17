import subprocess

def check_package_version(package, version):
    result = subprocess.run(['yolk', '-V', package], stdout=subprocess.PIPE)
    versions = result.stdout.decode('utf-8').split('\n')
    for vers in versions:
        if version in vers:
            return True
    return False

with open('requirements.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        package, version = line.strip().split('==')
        if check_package_version(package, version):
            print(f'La version {version} du package {package} est disponible.')
        else:
            print(f'La version {version} du package {package} n\'est pas disponible.')
