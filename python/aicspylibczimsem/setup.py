
from setuptools import setup

setup (name = 'aicspylibczimsem',
       version = '0.1',
       description = 'Extension of aicspylibczi that reads and processes msem metadata (xml)',
       author = 'Paul Watkins',
       author_email = 'pwatkins@gmail.com',
       url = '',
       long_description = '''
Msem czi files also contains polygons, scenes and ribbons describing slices. This pure python module extends aicspylibczi to process these files.
''',
       packages = ['aicspylibczimsem'],
       )
