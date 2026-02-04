"""
sklearn 兼容层。

处理 OneHotEncoder 等 API 变更：sklearn>=1.2 将 sparse 参数更名为 sparse_output，
而 synthcity 仍使用旧 API。在 synthcity 调用前 monkey-patch 以兼容 sklearn 1.0 ~ 最新版本。
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def apply_sklearn_compat() -> None:
    """
    Monkey-patch OneHotEncoder，使 sparse 参数在 sklearn>=1.2 下映射为 sparse_output。

    必须在 synthcity 首次导入前调用（通常在 synthetic_backend 加载时执行）。
    兼容 sklearn 1.0 ~ 最新版本。
    """
    try:
        import sklearn.preprocessing
        from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
    except ImportError:
        return

    sig = inspect.signature(_OneHotEncoder.__init__)
    if "sparse" in sig.parameters:
        return

    _Original = _OneHotEncoder

    class _OneHotEncoderCompat(_Original):
        """兼容层：将 synthcity 使用的 sparse 参数映射为 sparse_output。"""

        def __init__(self, *args: object, sparse=None, sparse_output=None, **kwargs: object) -> None:
            if "sparse" in kwargs:
                sparse = kwargs.pop("sparse")
            if sparse is not None and sparse_output is None:
                sparse_output = sparse
            if sparse_output is not None:
                kwargs["sparse_output"] = sparse_output
            super().__init__(*args, **kwargs)

    sklearn.preprocessing.OneHotEncoder = _OneHotEncoderCompat
