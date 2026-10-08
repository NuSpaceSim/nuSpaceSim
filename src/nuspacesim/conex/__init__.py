r"""CONEX-format ROOT output of simulated shower profiles.

* :mod:`~nuspacesim.conex.format` -- the fixed-shape output schema.
* :mod:`~nuspacesim.conex.writer` -- :class:`ConexWriter`, an
  ``on_shower_profile`` callable for :func:`nuspacesim.compute.compute`.
"""

from .writer import ConexWriter, build_showers, conex_path

__all__ = ["ConexWriter", "build_showers", "conex_path"]
