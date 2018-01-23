
# This file is part of Photos-vectorizer.
#
# Copyright (C) 2017  Mike D'Arcy
#
# Photos-vectorizer is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Photos-vectorizer is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License
# for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import print_function
import sys

class VersionError(Exception):
	pass

def default_on_fail(needed_version):
	cur_version = sys.version_info
	cur_vers_str = '.'.join(str(x) for x in cur_version[:3])
	needed_vers_str = '.'.join(str(x) for x in needed_version[:3])
	msg = 'Python version ({}) is too low. Should be at least {}'.format(cur_vers_str, needed_vers_str)
	raise VersionError(msg)
	

def assert_min_version(version, on_fail=default_on_fail):
	version_tuple = None
	if isinstance(version, float):
		print('Warning: Passed a float as a version number; it is highly recommended to use a tuple instead', file=sys.stderr)
		version_str = '{:0.10f}'.format(version)
		version_tuple = tuple([int(x) for x in version_str.split('.')])
	elif isinstance(version, str):
		version_tuple = tuple([int(x) for x in version.split('.')])
		if len(version_tuple) > 3:
			print('Warning: Too many version fields found in version string. Only the first 3 (major, minor, micro) will be used', file=sys.stderr)
			version_tuple = version_tuple[:3]
	elif isinstance(version, tuple):
		version_tuple = version
	else:
		print('Error: Bad version type (expected tuple, got {})'.format(type(vesion)), file=sys.stderr)
		sys.exit(1)

	if sys.version_info < version_tuple:
		on_fail(version_tuple)
