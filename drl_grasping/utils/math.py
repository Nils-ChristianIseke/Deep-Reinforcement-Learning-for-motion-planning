## BSD 3-Clause License
##
## Copyright (c) 2021, Andrej Orsula
## All rights reserved.

## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:

## 1. Redistributions of source code must retain the above copyright notice, this
##   list of conditions and the following disclaimer.
##
## 2. Redistributions in binary form must reproduce the above copyright notice,
##   this list of conditions and the following disclaimer in the documentation
##   and/or other materials provided with the distribution.
##
## 3. Neither the name of the copyright holder nor the names of its
##   contributors may be used to endorse or promote products derived from
##   this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
## FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
## DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
## CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
## OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Tuple


def quat_mul(quat_0: Tuple[float, float, float, float],
             quat_1: Tuple[float, float, float, float],
             xyzw: bool = True) -> Tuple[float, float, float, float]:
    """
    Multiply two quaternions
    """
    if xyzw:
        x0, y0, z0, w0 = quat_0
        x1, y1, z1, w1 = quat_1
        return (x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)
    else:
        w0, x0, y0, z0 = quat_0
        w1, x1, y1, z1 = quat_1
        return (-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0)
