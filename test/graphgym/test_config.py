from dataclasses import dataclass


def _lazy_from_config(*args, **kwargs):
    # Defer importing heavy third-party code until the function is actually called
    from torch_geometric.graphgym.config import from_config as _real_from_config

    # Replace the proxy with the real function for subsequent calls
    globals()['from_config'] = _real_from_config
    return _real_from_config(*args, **kwargs)


from_config = _lazy_from_config


@dataclass
class MyConfig:
    a: int
    b: int = 4


def my_func(a: int, b: int = 2) -> str:
    return f'a={a},b={b}'


def test_from_config():
    assert my_func(a=1) == 'a=1,b=2'

    assert my_func.__name__ == from_config(my_func).__name__
    assert from_config(my_func)(cfg=MyConfig(a=1)) == 'a=1,b=4'
    assert from_config(my_func)(cfg=MyConfig(a=1, b=1)) == 'a=1,b=1'
    assert from_config(my_func)(2, cfg=MyConfig(a=1, b=3)) == 'a=2,b=3'
    assert from_config(my_func)(cfg=MyConfig(a=1), b=3) == 'a=1,b=3'
