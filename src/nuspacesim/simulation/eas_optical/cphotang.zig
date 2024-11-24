const py = @cImport({
    @cDefine("Py_LIMITED_API", "3");
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
    @cInclude("numpy/arrayobject.h");
    @cInclude("numpy/ndarrayobject.h");
});

const std = @import("std");

const pw7 = @import("src/cubeptwt").square_deg7;

const PyArg_ParseTuple = py.PyArg_ParseTuple;
const PyArrayObject = py.PyArrayObject;
const PyMethodDef = py.PyMethodDef;
const PyModuleDef = py.PyModuleDef;
const PyModuleDef_Base = py.PyModuleDef_Base;
const PyModule_Create = py.PyModule_Create;
const PyObject = py.PyObject;
const Py_BuildValue = py.Py_BuildValue;
const METH_NOARGS = py.METH_NOARGS;

fn zaml_load(self: [*c]PyObject, args: [*c]PyObject) callconv(.C) [*]PyObject {
    _ = self;
    var beta_arr: [*c]py.PyArrayObject = undefined;
    var altDec_arr: [*c]py.PyArrayObject = undefined;
    var Eshowp100_arr: [*c]py.PyArrayObject = undefined;

    if (py.PyArrayObject(args, "O!O!O!", //
        &py.PyArray_Type, &beta_arr, //
        &py.PyArray_Type, &altDec_arr, //
        &py.PyArray_Type, &Eshowp100_arr) == 0) //
    { //
        return null;
    }

    if (py.PyArray_NDIM(beta_arr) != 1 //
    or py.PyArray_Type(beta_arr) != py.NPY_FLOAT32 //
    or py.PyArray_NDIM(altDec_arr) != 1 //
    or py.PyArray_Type(altDec_arr) != py.NPY_FLOAT32 //
    or py.PyArray_NDIM(Eshowp100_arr) != 1 //
    or py.PyArray_Type(Eshowp100_arr) != py.NPY_FLOAT32 //
    ) {
        _ = py.PyErr_SetString(py.PyExc_TypeError, //
            "Input arrays must be 1D and of type float32.");
        return null;
    }

    // Check that the lengths of the arrays are the same
    const length = py.PyArray_SIZE(beta_arr);
    if (length != py.PyArray_SIZE(altDec_arr) //
    or length != py.PyArray_SIZE(Eshowp100_arr)) //
    {
        _ = py.PyErr_SetString(py.PyExc_ValueError, //
            "Input arrays must have the same length.");
        return null;
    }

    // Create two new 1D output arrays
    var dims: [1]py.npy_intp = [length]py.npy_intp;
    const cang_arr = py.PyArray_SimpleNew(1, &dims[0], py.NPY_FLOAT32);
    const dphot_arr = py.PyArray_SimpleNew(1, &dims[0], py.NPY_FLOAT32);
    if (cang_arr == null or dphot_arr == null) {
        py.Py_XDECREF(cang_arr);
        py.Py_XDECREF(dphot_arr);
        return null;
    }

    // Retrieve pointers to the data of input and output arrays
    const betas_data: [*]const f32 = @ptrCast(py.PyArray_DATA(beta_arr));
    const altdec_data: [*]const f32 = @ptrCast(py.PyArray_DATA(altDec_arr));
    const Eshowp100_data: [*]const f32 = @ptrCast(py.PyArray_DATA(Eshowp100_arr));
    var cang_data: [*]f32 = @ptrCast(py.PyArray_DATA(cang_arr));
    var dphot_data: [*]f32 = @ptrCast(py.PyArray_DATA(dphot_arr));

    const betas: []const f32 = betas_data[0..length];
    const altdec: []const f32 = altdec_data[0..length];
    const Eshowp100: []const f32 = Eshowp100_data[0..length];
    var cang: []f32 = cang_data[0..length];
    var dphot: []f32 = dphot_data[0..length];

    cphot_framed(betas, altdec, Eshowp100, &cang, &dphot);

    // Return a tuple of the output arrays
    return py.PyTuple_Pack(2, cang_arr, dphot_arr);
}

var ZamlMethods = [_]PyMethodDef{
    PyMethodDef{
        .ml_name = "load",
        .ml_meth = zaml_load,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Perform batched EAS Optical shower propagation.",
    },
    PyMethodDef{
        .ml_name = null,
        .ml_meth = null,
        .ml_flags = 0,
        .ml_doc = null,
    },
};

var zamlmodule = PyModuleDef{
    .m_base = PyModuleDef_Base{
        .ob_base = PyObject{
            .ob_refcnt = 1,
            .ob_type = null,
        },
        .m_init = null,
        .m_index = 0,
        .m_copy = null,
    },
    .m_name = "zaml",
    .m_doc = null,
    .m_size = -1,
    .m_methods = &ZamlMethods,
    .m_slots = null,
    .m_traverse = null,
    .m_clear = null,
    .m_free = null,
};

pub export fn PyInit_zaml() [*]PyObject {
    return PyModule_Create(&zamlmodule);
}

// SPYield [zs, w] Scaled photon yeild per wavelength bin.
// DistStep = [zs] distance to detector.
// thetaC = [zs] cherenkov threshold angle.
// eCthres = [zs] cherenkov threshold energy.
// Tfrac = [zs] tracklen ratio for crad forming particles.
// E0 [zs] s dependent Tfrac parameter.
// s [zs] shower age.
// Eshow [1] Shower energy in GeV.

/// Hillas dndu integrand, scalar implementation.
fn dndu(comptime T: type, E: T, t: T, E2: T) T {
    const v = E / E2;
    const wr = (1.0 + v + 8.3 * v * v) / (5.4e-3 * E * (1.0 + v));
    const w = 2.0 * (1.0 - @cos(t)) * E * E * (1.0 / 441.0);
    const u = w * wr;
    const z = @sqrt(u);
    const z0 = 0.59;

    // const lambda: f32 = if (z < z0) 1.0 / 0.413 else 1.0 / 0.380;
    const bl = z < z0;
    const lambda = (bl * (1.0 / 0.413)) + (~bl * (1.0 / 0.380));

    const A = 0.777;
    return A * @exp(-(z - z0) * lambda);
}

fn Nhill_cubature(Eshow: f32, thetaC: f32, s: f32) f32 {
    const E2 = 1150 + 454 * @log(s);
    var nhill = 0.0;
    // pw7 imported from above is a cubature point, weight set.
    for (pw7) |pw| {
        const E = 0.5 * Eshow * (1.0 + pw[0]);
        const t = 0.5 * thetaC * (1.0 + pw[1]);
        nhill += pw[2] * dndu(f32, E, t, E2);
    }
    const Det = 0.25 * Eshow * thetaC;
    return nhill * Det;
}

// Optionally define dndu as an integral over cos(t). The E input remains [0,Eshow]
// The t input becomes cos(t) from [0,cos(thetaC)]. The new mapping to the unit square
// nodes is:
//
// const E: f32 = 0.5 * Eshow * (1 + pw[0]);
// const cost: f32 = 0.5 * (1 - cos(thetaC))*pw[1] + (1 + cos(thetaC));
// const Det: f32 = 0.25 * Eshow * (1 - cos(thetaC));
// n += dndu_scalar_cost(E, cost, E2) * Det;
//
// This moves the cos(t) computation out of the integrand, saving some compute cycles.

const Line = struct {
    const size: u8 = 32;
    z: [size]f32,
    X: [size]f32,
    gm: [size]f32,
};

const Frame = struct {
    const itrlen: u8 = 32;
    data: [itrlen]Line,
    beta: [Line.size]f32,
    eshow: [Line.size]f32,
};

fn cphot_framed(
    betaE: []const f32,
    alt: []const f32,
    EshowGeV: []const f32,
    cang: []f32,
    dphot: []f32,
) !void {
    // [x] Compute Memory requirements?
    // [x] Allocate Memory for Frame.
    // [ ] Allocate Memory for materialized intermediates.
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // The operation frame
    var frame: Frame = try allocator.alloc(Frame, 1);

    const chunk_count: u32 = betaE.len() / Line.size;
    for (0..chunk_count) |i| {
        const j = i * Line.size;

        // Initialize Frame.
        std.mem.copy(f32, &frame.data[0].zs, &alt[j .. j + Line.size]);
        std.mem.copy(f32, &frame.beta, &betaE[j .. j + Line.size]);
        std.mem.copy(f32, &frame.eshow, &EshowGeV[j .. j + Line.size]);


        // [ ] simulate frame.
        //
        // [ ] store frame intermediates.
        //
        // [ ] output frame intermediates.
    }
    // const elem_remain: u32 = betaE.len() % Line.size;
    // for (0..elem_remain) |i| {}
    //

    _ = cang;
    _ = dphot;
}

fn run(frame: *Frame) void{
    for (1..Frame.itrlen) |i| {
        frame.data[i]
    }
}
