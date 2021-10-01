#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <utility>
#include <vector>

namespace py = pybind11;

template <typename T>
auto py_zsteps(T z, T sinThetView, T RadE, T zMaxZ, T zmax, T dL, T pi)
    -> std::pair<py::array_t<T>, py::array_t<T>> {

  auto zsave = std::vector<T>();
  auto delzs = std::vector<T>();

  auto RadMax = RadE + zmax;
  auto pi_2   = pi / 2.0;

  while (z <= zMaxZ) {

    auto Rad      = z + RadE;
    auto tp       = RadMax / Rad;
    auto ThetProp = acos(sinThetView * tp);
    auto delz     = sqrt((Rad * Rad) + (dL * dL) -
                     (2.0 * Rad * dL * cos(pi_2 + ThetProp))) -
                Rad;

    delzs.push_back(delz);
    zsave.push_back(z + (delz / 2.0));

    z += delz;
  }

  auto py_zsave = py::array_t<T>(zsave.size());
  auto zsavebuf = py_zsave.request();
  auto zsaveptr = zsavebuf.ptr;
  std::memcpy(zsaveptr, zsave.data(), zsave.size() * sizeof(T));

  auto py_delzs = py::array_t<T>(delzs.size());
  auto delzsbuf = py_delzs.request();
  auto delzsptr = delzsbuf.ptr;
  std::memcpy(delzsptr, delzs.data(), delzs.size() * sizeof(T));

  return { py_zsave, py_delzs };
}

PYBIND11_MODULE(zsteps, m) {
  m.doc() = "Iterative z propagation through atmosphere.";
  m.def("zsteps",
        &py_zsteps<double>,
        "Given a double precision starting altitude and associated constants, "
        "compute the array of z steps and the array of deltas for the full "
        "trajectory up to a maximum altitude cutoff.");
  m.def("zsteps",
        &py_zsteps<float>,
        "Given a single precision starting altitude and associated constants, "
        "compute the array of z steps and the array of deltas for the full "
        "trajectory up to a maximum altitude cutoff.");
}
