"""
第三方库兼容层。

所有对 synthcity / sklearn / pydantic 等第三方 API 的适配与 hack 集中放置于此。
禁止在业务代码中直接假设第三方 API 稳定，必须经此模块显式过滤/适配。
"""

from synthgen.compat.sklearn import apply_sklearn_compat
from synthgen.compat.synthcity import filter_plugin_params

__all__ = ["apply_sklearn_compat", "filter_plugin_params"]
