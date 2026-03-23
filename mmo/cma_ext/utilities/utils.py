# -*- coding: utf-8 -*-
"""various utilities not related to optimization"""
from __future__ import (absolute_import, division, print_function)
import os, sys, time
import warnings
from collections import defaultdict  # since Python 2.5

from collections.abc import MutableMapping  # New (fix for Python 3.10+)
import ast  # ast.literal_eval is safe eval
import numpy as np
import math

# New helper function for sigma updates
def update_sigma(value, clip_bound=1.5):
    """
    Compute sigma change as np.exp(value) and clip it to be within
    np.exp(-clip_bound) and np.exp(clip_bound).

    Parameters
    ----------
    value : float
        The log-change value.
    clip_bound : float, optional
        The clipping bound in logarithmic space (default: 1.5).

    Returns
    -------
    sigma_change : float
        The clipped sigma change.
    """
    sigma_change = np.exp(value)
    if sigma_change > np.exp(clip_bound):
        sigma_change = np.exp(clip_bound)
        warnings.warn("sigma change " + str(np.exp(value)) +
                      " clipped to np.exp(+-" + str(clip_bound) + ")", UserWarning)
    elif sigma_change < np.exp(-clip_bound):
        sigma_change = np.exp(-clip_bound)
        warnings.warn("sigma change " + str(np.exp(value)) +
                      " clipped to np.exp(+-" + str(clip_bound) + ")", UserWarning)
    return sigma_change

# Remove future compatibility flags
del absolute_import, division, print_function

global_verbosity = 1

def is_(var):
    """intuitive handling of variable truth value also for `numpy` arrays.
    Return True for any non-empty container, otherwise the truth value of the
    scalar var.
    """
    try:  # cases: ('', (), [], {}, np.array([]))
        return True if len(var) else False
    except TypeError:  # cases None, False, 0
        return True if var else False

def is_not(var):
    """see `is_`"""
    return not is_(var)

def is_any(var_list):
    """return any(is_(v) for v in var_list)"""
    return any(is_(var) for var in var_list)

def is_all(var_list):
    """return all(is_(v) for v in var_list)"""
    return all(is_(var) for var in var_list)

def is_str(var):
    """bytes (in Python 3) also fit the bill."""
    types_ = (bytes, str)
    PY2 = sys.version_info[0] == 2
    if PY2:
        types_ = types_ + (basestring, unicode)
    return any(isinstance(var, type_) for type_ in types_)

def is_nan(var):
    """return np.isnan(var) or False if var is not numeric"""
    try:
        return np.isnan(var)
    except TypeError:
        return False

def is_vector_list(x):
    """make an educated guess whether x is a list of vectors."""
    try:
        return np.isscalar(x[0][0])
    except:
        return False

def as_vector_list(X):
    """a tool to handle a vector or a list of vectors in the same way,
    return a list of vectors and a function to revert the "list making".
    """
    if is_vector_list(X):
        return X, lambda x: x
    else:
        return [X], lambda *args: args[0][0] if len(args) == 1 else (arg[0] for arg in args)

def rglen(ar):
    """return generator range(len(.)) with shortcut rglen(.)"""
    return range(len(ar))

def recycled(vec, dim=None, as_=None):
    """return vec with the last element recycled to dim if len(vec) doesn't fail, else vec."""
    try:
        len_ = len(vec)
    except TypeError:
        return vec
    if dim is None:
        try:
            dim = len(as_)
        except TypeError:
            return vec[0]
    if dim == len_:
        return vec
    elif dim < len_:
        return vec[:dim]
    elif dim > len_:
        return np.asarray(list(vec) + (dim - len_) * [vec[-1]])

def argsort(a, reverse=False):
    """return index list to get a in order"""
    return sorted(range(len(a)), key=a.__getitem__, reverse=reverse)

def ranks(a, reverse=False):
    """return ranks of entries starting with zero"""
    idx = argsort(a)
    return [len(idx) - 1 - idx.index(i) if reverse else idx.index(i)
            for i in range(len(idx))]

def zero_values_indices(diffs):
    """generate increasing index pairs (i, j) with all(diffs[i:j] == 0)."""
    i = 0
    while i < len(diffs):
        if diffs[i] == 0:
            j = i
            while j < len(diffs) and diffs[j] == 0:
                j += 1
            yield i, j
            i = j + 1
        else:
            i += 1

def pprint(to_be_printed):
    """nicely formatted print"""
    try:
        import pprint as pp
        pp.pprint(to_be_printed)
    except ImportError:
        if isinstance(to_be_printed, dict):
            print('{')
            for k, v in to_be_printed.items():
                print("'" + k + "'" if str(k) == k else k,
                      ': ',
                      "'" + v + "'" if str(v) == v else v,
                      sep="")
            print('}')
        else:
            print('could not import pprint module, applying regular print')
            print(to_be_printed)

def num2str(val, significant_digits=2, force_rounding=False,
            max_predecimal_digits=5, max_postdecimal_leading_zeros=1,
            remove_trailing_zeros=True, desired_length=None):
    """returns the shortest string representation."""
    if val == 0:
        return '0'
    if not significant_digits > 0:
        raise ValueError('need significant_digits=%s > 0' % str(significant_digits))
    is_negative = val < 0
    original_value = val
    val = float(np.abs(val))
    order_of_magnitude = int(np.floor(np.log10(val)))
    fac = 10**(significant_digits - 1 - order_of_magnitude)
    val_rounded = np.round(fac * val) / fac
    if order_of_magnitude + 1 >= significant_digits:
        s = str(int(val_rounded if force_rounding else np.round(val)))
    else:
        s = str(val_rounded)
        idx1 = 0
        while idx1 < len(s) and s[idx1] in ('-', '0', '.'):
            idx1 += 1
        idx2 = idx1 + significant_digits + (s.find('.') > idx1)
        if val != val_rounded:
            if len(s) < idx2:
                s += '0' * (idx2 - len(s))
        if val == val_rounded and remove_trailing_zeros:
            while s[-1] == '0':
                s = s[0:-1]
        if s[-1] == '.':
            s = s[0:-1]
    s_float = ('-' if is_negative else '') + s
    s = ('%.' + str(significant_digits - 1) + 'e') % val
    if eval(s) == val and s.find('.') > 0:
        while s.find('0e') > 0:
            s = s.replace('0e', 'e')
    s = s.replace('.e', 'e')
    s = s.replace('e+', 'e')
    while s.find('e0') > 0:
        s = s.replace('e0', 'e')
    while s.find('e-0') > 0:
        s = s.replace('e-0', 'e-')
    if s[-1] == 'e':
        s = s[:-1]
    s_exp = ('-' if is_negative else '') + s
    if (len(s_exp) < len(s_float) or
        s_float.find('0.' + '0' * (max_postdecimal_leading_zeros + 1)) > -1 or
        np.abs(val_rounded) >= 10**(max_predecimal_digits + 1)
        ):
        s_ret = s_exp
    else:
        s_ret = s_float
    if desired_length:
        s_old = ''
        while len(s_ret) < desired_length and len(s_old) < len(s_ret):
            s_old = s_ret
            s_ret = num2str(original_value,
                           significant_digits + desired_length - len(s_ret),
                           force_rounding,
                           max_predecimal_digits,
                           max_postdecimal_leading_zeros,
                           remove_trailing_zeros,
                           desired_length=None)
    return s_ret

def print_warning(msg, method_name=None, class_name=None, iteration=None, verbose=None, maxwarns=None):
    if verbose is None:
        verbose = global_verbosity
    if maxwarns is not None and iteration is None:
        raise ValueError('iteration must be given to activate maxwarns')
    if verbose >= -2 and (iteration is None or maxwarns is None or iteration <= maxwarns):
        warnings.warn(msg + ' (' +
                      ('class=%s ' % str(class_name) if class_name else '') +
                      ('method=%s ' % str(method_name) if method_name else '') +
                      ('iteration=%s' % str(iteration) if iteration else '') +
                      ')')

def print_message(msg, method_name=None, class_name=None, iteration=None, verbose=None):
    if verbose is None:
        verbose = global_verbosity
    if verbose >= 0 and 0:
        print('NOTE (module=cma): ' +
              (', class=' + str(class_name) if class_name else '') +
              (', method=' + str(method_name) if method_name else '') +
              (', iteration=' + str(iteration) if iteration is not None else '') +
              ': ', msg)

def set_attributes_from_dict(self, dict_, initial_params_dict_name=None):
    if initial_params_dict_name:
        setattr(self, initial_params_dict_name, dict_.copy())
        getattr(self, initial_params_dict_name).pop('self', None)
    for key, val in dict_.items():
        if key != 'self':
            setattr(self, key, val)

def download_file(url, target_dir='.', target_name=None):
    import urllib2
    if target_name is None:
        target_name = url.split(os.path.sep)[-1]
    with open(os.path.join(target_dir, target_name), 'wb') as f:
        f.write(urllib2.urlopen(url).read())

def extract_targz(tarname, filename=None, target_dir='.'):
    import tarfile
    tmp_dir = '._tmp_'
    if filename is None:
        tarfile.TarFile.gzopen(tarname).extractall(target_dir)
    else:
        import shutil
        tarfile.TarFile.gzopen(tarname).extractall(tmp_dir)
        shutil.copy2(os.path.join(tmp_dir, filename),
                     os.path.join(target_dir, filename.split(os.path.sep)[-1]))
        shutil.rmtree(tmp_dir)

class BlancClass(object):
    """Blanc container class to hold a collection of attributes."""
    pass

class DictClass(dict):
    """A class wrapped over `dict` to use class .-notation."""
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
    def __dir__(self):
        return self.keys()

class DerivedDictBase(MutableMapping):
    def __init__(self, *args, **kwargs):
        super(DerivedDictBase, self).__init__()
        self.data = dict()
        self.data.update(dict(*args, **kwargs))
    def __len__(self):
        return len(self.data)
    def __contains__(self, key):
        return key in self.data
    def __iter__(self):
        return iter(self.data)
    def __setitem__(self, key, value):
        self.data[key] = value
    def __getitem__(self, key):
        return self.data[key]
    def __delitem__(self, key):
        del self.data[key]

class SolutionDict(DerivedDictBase):
    def __init__(self, *args, **kwargs):
        super(SolutionDict, self).__init__(*args, **kwargs)
        self.data_with_same_key = {}
        self.last_iteration = 0
    def key(self, x):
        try:
            return tuple(x)
        except TypeError:
            return x
    def __setitem__(self, key, value):
        key = self.key(key)
        if key in self.data_with_same_key:
            self.data_with_same_key[key] += [self.data[key]]
        elif key in self.data:
            self.data_with_same_key[key] = [self.data[key]]
        self.data[key] = value
    def __getitem__(self, key):
        return self.data[self.key(key)]
    def __delitem__(self, key):
        key = self.key(key)
        if key in self.data_with_same_key:
            if len(self.data_with_same_key[key]) == 1:
                self.data[key] = self.data_with_same_key.pop(key)[0]
            else:
                self.data[key] = self.data_with_same_key[key].pop(-1)
        else:
            del self.data[key]
    def truncate(self, max_len, min_iter):
        if len(self) > max_len:
            for k in list(self.keys()):
                if self[k]['iteration'] < min_iter:
                    del self[k]
            self._lenhist = []

class DataDict(defaultdict):
    def __init__(self, filename='_data.py'):
        self.filename = filename
        defaultdict.__init__(self, list)
        self.load()
    def load(self):
        with open(self.filename, 'rt') as f:
            dd = ast.literal_eval(f.read())
        self.update(dd)
        return self
    def update(self, dict_):
        for k in dict_:
            self[k] += dict_[k]
        return self
    def save(self):
        with open(self.filename, 'wt') as f:
            f.write(repr(dict(self)))
    def clear(self):
        for key in list(self.keys()):
            del self[key]
        return self

class ExclusionListOfVectors(list):
    def __contains__(self, vec):
        for v in self:
            if 1 - 1e-9 < np.dot(v, vec) / (sum(np.asarray(v)**2) * sum(np.asarray(vec)**2))**0.5 < 1 + 1e-9:
                return True
        return False

class ElapsedWCTime(object):
    def __init__(self, time_offset=0):
        self._time_offset = time_offset
        self.reset()
    def reset(self):
        self.cum_time = self._time_offset
        self.paused = 0
        self.last_tic = time.time()
        return self
    def pause(self):
        if not self.paused:
            self.paused = time.time()
        return self
    def __call__(self):
        raise DeprecationWarning()
        return self.elapsed
    @property
    def tic(self):
        return_ = self.toc
        if self.paused:
            if self.paused < self.last_tic:
                print_warning("paused time=%f < last_tic=%f, which should never happen." % (self.paused, self.last_tic),
                              "tic", "ElapsedWCTime")
                self.paused = self.last_tic
            self.cum_time += self.paused - self.last_tic
        else:
            self.cum_time += time.time() - self.last_tic
        self.paused = 0
        self.last_tic = time.time()
        return return_
    @property
    def elapsed(self):
        return self.cum_time + self.toc
    @property
    def toc(self):
        if self.paused:
            return self.paused - self.last_tic
        return time.time() - self.last_tic

class TimingWrapper(object):
    def __init__(self, callable_):
        self._callable = callable_
        self.timer = ElapsedWCTime().pause()
    def __call__(self, *args, **kwargs):
        self.timer.tic
        res = self._callable(*args, **kwargs)
        self.timer.pause()
        return res

class DictFromTagsInString(dict):
    def __init__(self, *args, **kwargs):
        super(DictFromTagsInString, self).__init__()
        self.tag_string = "python"
        if is_(args) or is_(kwargs):
            self.update(*args, **kwargs)
    def update(self, string_=None, filename=None, file_=None, dict_=None, tag_string=None):
        args = 4 - ((string_ is None) + (filename is None) +
                    (file_ is None) + (dict_ is None))
        if not args:
            raise ValueError('nothing to update')
        if args > 1:
            raise ValueError('use either string_ or filename or file_ or dict_ as input, but not several of them')
        if tag_string is not None:
            self.tag_string = tag_string
        if filename is not None:
            string_ = open(filename, 'r').read()
        elif file_ is not None:
            string_ = file_.read()
        elif dict_ is not None:
            super(DictFromTagsInString, self).update(dict_)
            return self
        super(DictFromTagsInString, self).update(self._eval_python_tag(string_))
        return self
    @property
    def as_python_tag(self):
        return self._start + repr(dict(self)) + self._end
    def __repr__(self):
        return self.as_python_tag
    @property
    def _start(self):
        return '<' + self.tag_string + '>'
    @property
    def _end(self):
        return '</' + self.tag_string + '>'
    def _eval_python_tag(self, str_):
        values = {}
        str_lower = str_.lower()
        start = str_lower.find(self._start)
        while start >= 0:
            start += len(self._start)
            end = str_lower.find(self._end, start)
            values.update(ast.literal_eval(str_[start:end].strip()))
            start = str_lower.find(self._start, start + 1)
        return values

class MoreToWrite(list):
    def __init__(self):
        self._lenhist = []
    def check(self):
        self._lenhist += [len(self)]
        if len(self._lenhist) > 3:
            if all(np.diff(self._lenhist) > 0):
                del self[:]
            self._lenhist = []

class DefaultSettings(object):
    def __init__(self, params, number_of_params, obj):
        self.inparams = dict(params)
        self._number_of_params = number_of_params
        self.obj = obj
        self.inparams.pop('self', None)
        self._set_from_defaults()
        self._set_from_input()
    def __str__(self):
        return ("{" + '\n'.join(r"%s: %s" % (str(k), str(v)) for k, v in self.items()) + "}")
    def _set_from_defaults(self):
        self.__dict__.update(((key, val)
                              for (key, val) in type(self).__dict__.items()
                              if not key.startswith('_')))
    def _set_from_input(self):
        discarded = {}
        for key in list(self.inparams):
            if key not in self.__dict__ or key in self.obj.__dict__:
                discarded[key] = self.inparams.pop(key)
            elif self.inparams[key] is not None:
                setattr(self, key, self.inparams[key])
        if len(self.inparams) != self._number_of_params:
            warnings.warn("%s: %d parameters desired; remaining: %s; discarded: %s " %
                          (str(type(self)), self._number_of_params, str(self.inparams), str(discarded)))
        delattr(self, 'obj')

# End of file.
