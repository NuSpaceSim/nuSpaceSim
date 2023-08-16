# The Clear BSD License
#
# Copyright (c) 2023 Alexander Reustle and the NuSpaceSim Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

__all__ = ["MonoCloud", "NoCloud", "PressureMapCloud"]


@dataclass
class NoCloud:
    pass


@dataclass
class MonoCloud:
    altitude: float = -np.inf
    """Altitude of monoheight cloud."""


@dataclass
class PressureMapCloud:
    month: int = 1
    """Cloud Map Month integer 1-12 inclusive."""

    version: str = "0"
    """Cloud Map File Version."""


def parse_month(date: str | int | datetime) -> int:
    if isinstance(date, datetime):
        return date.month
    if isinstance(date, int):
        if date < 1 or date > 12:
            raise RuntimeError(f"Provided month {date} is invalid")
        return date
    if isinstance(date, str):
        try:
            return (datetime.strptime(date, "%m")).month
        except ValueError:
            pass
        try:
            return (datetime.strptime(date, "%B")).month
        except ValueError:
            pass
        try:
            return (datetime.strptime(date, "%b")).month
        except ValueError:
            pass
        raise ValueError(
            f"date {date} does not match valid month patterns (%m, %B, %b)"
        )
