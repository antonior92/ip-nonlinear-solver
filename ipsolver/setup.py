from __future__ import division, print_function, absolute_import


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
<<<<<<< HEAD
    config = Configuration('_lsq', parent_package, top_path)
    config.add_extension('_group_columns', sources=['_group_columns.c'],)
=======
    config = Configuration('ipsolver', parent_package, top_path)
    config.add_subpackage('_large_scale_constrained')
    config.add_extension('_group_columns', sources=['_group_columns.c'],)
    config.add_data_dir('tests')
>>>>>>> test-constraints
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
