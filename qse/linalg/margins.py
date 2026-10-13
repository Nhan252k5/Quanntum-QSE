"""
Theo dõi các quyết định số học sát ngưỡng.

Mỗi lần code so một dãy giá trị với một ngưỡng (kernel, rank, nullspace...),
gọi `check(tag, values, tol, config)`. Nếu có giá trị nằm trong
[tol / 10^m, tol * 10^m] (m = margin_decades) thì ghi cảnh báo AMBIGUOUS.
"""
import numpy as np

_LOG: list[dict] = []


def reset():
    _LOG.clear()


def entries():
    return list(_LOG)


def check(tag, values, tol, config, **info):
    v = np.abs(np.asarray(values, dtype=float)).ravel()
    if v.size == 0:
        return
    f = 10.0 ** config.margin_decades
    near = v[(v > tol / f) & (v < tol * f)]
    if near.size:
        _LOG.append(dict(tag=tag, tol=float(tol),
                         nearest=float(near[np.argmin(np.abs(np.log10(near / tol)))]),
                         **info))
