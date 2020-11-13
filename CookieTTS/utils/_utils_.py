import types

def get_args(*func):
    args = list()
    if type(func) in [list, tuple]:
        for f in func:
            args.extend(f.__code__.co_varnames[:f.__code__.co_argcount])
    elif isinstance(func, types.ClassType):
        args.extend(func.__call__.__code__.co_varnames[:func.__call__.__code__.co_argcount])
    elif isinstance(func, types.FunctionType):
        args.extend(func.__code__.co_varnames[:func.__code__.co_argcount])
    elif isinstance(func, types.CodeType):
        args.extend(func.co_varnames[:func.co_argcount])
    args = tuple(set(args))
    return args

def replace_args(args, replace_dict={}):
    new_args = list()
    for arg in args:
        new_arg = replace_dict[arg] if arg in replace_dict.keys() else arg
        new_args.append(new_arg)
    return tuple(set(new_args))

def rename_argdict(argdict, replace_dict={}, invert=True):
    if invert:
        replace_args = {v:k for k,v in replace_dict.items()}
    argdict = {replace_dict[k] if k in replace_dict else k: v for k,v in argdict.items()}
    return argdict

def force(func, valid_kwargs=None, *args, **kwargs):
    if valid_kwargs is True:
        return func(*args, **kwargs)
    elif valid_kwargs is None:
        valid_kwargs = get_args(func)
    return func(*args, **{k:v for k,v in kwargs.items() if k in valid_kwargs})
