import toml
import subprocess
import os

with open('pyproject.toml', 'r') as f:
    t = toml.loads(f.read())

v = t['tool']['poetry']['version']
print(f'Old Version: {v}')
major, minor, patch = v.split('.')
incremented = str(int(patch) + 1)

new_version = '.'.join([major, minor, incremented])
print(f"New Version: {new_version}")
proceed = input('Proceed? y/n: ').upper() == 'Y'
if proceed:
    t['tool']['poetry']['version'] = new_version

    with open('pyproject.toml', 'w') as f:
        f.write(toml.dumps(t))

    print(subprocess.check_output('poetry build'.split()))
    print(subprocess.check_output(f'poetry publish -u {os.environ["PYPI_USERNAME"]} -p {os.environ["PYPI_PASS"]}'.split()))


else:
    print('Aborted')


