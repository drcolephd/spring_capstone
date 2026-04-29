import os

# Construct the absolute path to the icon file
paths = {
    'pkg': '', 
    'root': '', 
    'assets': '', 
    'data': '', 
    'lib': '', 
} 
paths['pkg'] = os.path.dirname(__file__)
paths['root'] = os.path.abspath(os.path.join(paths['pkg'], '..'))
paths['assets'] = os.path.join(paths['root'], 'assets')
paths['data'] = os.path.join(paths['root'], 'data')
paths['models'] = os.path.join(paths['root'], 'models')
paths['configurations'] = os.path.join(paths['root'], 'configurations')
paths['lib'] = os.path.join(paths['root'], 'lib')
