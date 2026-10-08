r"""Minimal ROOT file writer for the CONEX ``Header`` and ``Shower`` trees.

Writes flat TTrees from NumPy structured arrays -- one branch per field, a
``(k,)`` subarray field as a fixed ``name[k]`` leaf -- with no ROOT/uproot
dependency. Scope is exactly what the CONEX schema needs: int8/int32/float32/
float64 leaves, uncompressed, one basket per branch, files under 2 GB.

Layout (written front to back)::

    0    file header (fBEGIN = 100)
    100  TFile key + top directory, (empty) StreamerInfo record
         per tree: one TBasket per branch, then the TTree
         KeysList, FreeSegments
    fEND

Object layouts follow ROOT 6.24 (TTree v20, TBranch v13, TLeaf v2) as
written by uproot; see ROOT's ``io/doc/TFile`` format notes.
"""

from __future__ import annotations

import datetime
import os
import struct
import uuid

import numpy as np

__all__ = ["write_trees"]

_BEGIN = 100
_LIMIT = 2_000_000_000  # 4-byte seek limit of the small file format
_BYTECOUNT = 0x40000000
_NEW_CLASS = 0xFFFFFFFF
_ON_HEAP_NOT_DELETED = 0x03000000

# dtype -> (TLeaf letter, fMinimum/fMaximum format)
_LEAF = {"i1": ("B", ">bb"), "i4": ("I", ">ii"), "f4": ("F", ">ff"), "f8": ("D", ">dd")}

_KEY = struct.Struct(">ihiIhhii")  # small TKey: 4-byte seeks
_KEY_BIG = struct.Struct(">ihiIhhqq")  # TTree/TBasket keys (as ROOT/uproot write them)
_TIOFEATURES = b"@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00"
_EMPTY_TOBJARRAY = b"@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x03" + b"\x00" * 12


def _tstring(s: str) -> bytes:
    b = s.encode()
    assert len(b) < 255
    return bytes([len(b)]) + b


class _Buffer:
    """Big-endian builder with back-patched ROOT byte counts."""

    def __init__(self):
        self.data = bytearray()

    def put(self, b):
        self.data += b

    def pack(self, fmt, *values):
        self.data += struct.pack(fmt, *values)

    def open(self, version=None, classname=None):
        pos = len(self.data)
        if classname is None:
            self.pack(">IH", 0, version)
        else:
            self.pack(">II", 0, _NEW_CLASS)
            self.put(classname.encode() + b"\x00")
        return pos

    def close(self, pos):
        count = (len(self.data) - pos - 4) | _BYTECOUNT
        struct.pack_into(">I", self.data, pos, count)

    def tnamed(self, name, title, bits=0):
        pos = self.open(1)
        self.pack(">HII", 1, 0, bits | _ON_HEAP_NOT_DELETED)
        self.put(_tstring(name) + _tstring(title))
        self.close(pos)

    def tobjarray_header(self, size):
        self.pack(">HII", 1, 0, _ON_HEAP_NOT_DELETED)
        self.put(b"\x00")  # fName ""
        self.pack(">ii", size, 0)


def _branches(records):
    out = []
    for name in records.dtype.names:
        field = records.dtype[name]
        base, shape = field.subdtype or (field, ())
        code = f"{base.kind}{base.itemsize}"
        if code not in _LEAF or len(shape) > 1:
            raise TypeError(f"branch {name!r}: unsupported field type {field}")
        out.append((name, base.newbyteorder(">"), shape))
    return out


def _ttree(name, title, n, branches, baskets, keylen):
    """Serialize a TTree; ``baskets[name] = (seek, nbytes, totbytes)``."""
    buf = _Buffer()
    tree = buf.open(20)
    buf.tnamed(name, title, bits=0x8)  # kMustCleanup
    buf.put(b"@\x00\x00\x08\x00\x02" + struct.pack(">hhh", 602, 1, 1))  # TAttLine
    buf.put(b"@\x00\x00\x06\x00\x02" + struct.pack(">hh", 0, 1001))  # TAttFill
    buf.put(b"@\x00\x00\x0a\x00\x02" + struct.pack(">hhf", 1, 1, 1.0))  # TAttMarker
    tot = sum(b[2] for b in baskets.values())
    zipped = sum(b[1] for b in baskets.values())
    # fEntries .. fEstimate (TTree v20 members), then empty cluster arrays.
    buf.pack(
        ">qqqqqdiiiiIqqqqqq",
        n,
        tot,
        zipped,
        0,
        0,
        1.0,
        0,
        25,
        0,
        1000,
        0,
        10**12,
        10**12,
        0,
        -300_000_000,
        -30_000_000,
        1_000_000,
    )
    buf.put(b"\x00\x00" + _TIOFEATURES)

    arr = buf.open(3)
    buf.tobjarray_header(len(branches))
    leaf_refs = []
    for bname, dtype, shape in branches:
        letter, minmax = _LEAF[f"{dtype.kind}{dtype.itemsize}"]
        dims = "".join(f"[{d}]" for d in shape)
        seek, nbytes, totbytes = baskets[bname]
        nb = 1 if n else 0

        obj = buf.open(classname="TBranch")
        ver = buf.open(13)
        buf.tnamed(bname, f"{bname}{dims}/{letter}", bits=0x00400000)
        buf.put(b"@\x00\x00\x06\x00\x02" + struct.pack(">hh", 0, 1001))  # TAttFill
        buf.pack(">iiiiq", 0, 32000, 0, nb, n)  # fCompress .. fEntryNumber
        buf.put(_TIOFEATURES)
        buf.pack(">iIiqqqq", 0, 2, 0, n, 0, totbytes, nbytes)  # fOffset .. fZipBytes
        buf.put(_EMPTY_TOBJARRAY)  # sub-branches

        leaves = buf.open(3)
        buf.tobjarray_header(1)
        leaf_refs.append(keylen + len(buf.data) + 2)  # +kMapOffset
        leaf = buf.open(classname=f"TLeaf{letter}")
        typed = buf.open(1)
        base = buf.open(2)
        buf.tnamed(bname, f"{bname}{dims}")
        buf.pack(">iii??", int(np.prod(shape)), dtype.itemsize, 0, False, False)
        buf.put(b"\x00" * 4)  # no fLeafCount: fixed length
        buf.close(base)
        buf.pack(minmax, 0, 0)
        buf.close(typed)
        buf.close(leaf)
        buf.close(leaves)

        buf.put(_EMPTY_TOBJARRAY)  # fBaskets
        buf.put(b"\x01" + struct.pack(">ii", nbytes if n else 0, 0))  # fBasketBytes
        buf.put(b"\x01" + struct.pack(">qq", 0, n))  # fBasketEntry
        buf.put(b"\x01" + struct.pack(">qq", seek if n else 0, 0))  # fBasketSeek
        buf.put(b"\x00")  # fFileName
        buf.close(ver)
        buf.close(obj)
    buf.close(arr)

    leaves = buf.open(3)
    buf.tobjarray_header(len(leaf_refs))
    buf.pack(f">{len(leaf_refs)}I", *leaf_refs)
    buf.close(leaves)
    buf.put(b"\x00" * 28)  # fAliases .. fBranchRef: all empty
    buf.close(tree)
    return bytes(buf.data)


def write_trees(path, trees):
    """Write ``{name: (title, structured array)}`` as TTrees to a new ROOT file."""
    filename = os.path.basename(os.fspath(path))
    t = datetime.datetime.now()
    datime = (
        (t.year - 1995) << 26
        | t.month << 22
        | t.day << 17
        | t.hour << 12
        | t.minute << 6
        | t.second
    )
    file_uuid = uuid.uuid4().bytes
    # Empty StreamerInfo TList: ROOT and uproot have built-in layouts for
    # TTree/TBranch/TLeaf, so no class descriptions are needed.
    payload = struct.pack(">IHHIIBI", 17 | _BYTECOUNT, 5, 1, 0, 0x02000000, 0, 0)
    names = _tstring(filename) + _tstring("")

    def key(fmt, cls, name, title, objlen, seek, pdir=_BEGIN):
        strings = _tstring(cls) + _tstring(name) + _tstring(title)
        keylen = fmt.size + len(strings)
        version = 1004 if fmt is _KEY_BIG else 4
        head = fmt.pack(keylen + objlen, version, objlen, datime, keylen, 1, seek, pdir)
        return head + strings

    with open(path, "wb") as fh:
        fh.write(b"\x00" * _BEGIN)
        dir_rec = 60  # TDirectory record (fits the big variant too)
        top = key(_KEY, "TFile", filename, "", len(names) + dir_rec, _BEGIN, 0)
        fh.write(top + names)
        dir_at = fh.tell()
        fh.write(b"\x00" * dir_rec)

        seek_info = fh.tell()
        fh.write(
            key(
                _KEY,
                "TList",
                "StreamerInfo",
                "Doubly linked list",
                len(payload),
                seek_info,
            )
            + payload
        )
        nbytes_info = fh.tell() - seek_info

        top_keys = []
        for tname, (title, records) in trees.items():
            branches = _branches(records)
            baskets = {}
            for bname, dtype, _ in branches:
                raw = np.ascontiguousarray(records[bname], dtype=dtype).tobytes()
                strings = _tstring("TBasket") + _tstring(bname) + _tstring(tname)
                keylen = _KEY_BIG.size + len(strings) + 19  # + TBasket header, flag
                seek = fh.tell()
                entry_bytes = len(raw) // max(len(records), 1)
                fh.write(
                    _KEY_BIG.pack(
                        keylen + len(raw),
                        1004,
                        len(raw),
                        datime,
                        keylen,
                        1,
                        seek,
                        _BEGIN,
                    )
                    + strings
                )
                fh.write(
                    struct.pack(
                        ">Hiiii", 3, 32000, entry_bytes, len(records), keylen + len(raw)
                    )
                    + b"\x00"
                    + raw
                )
                baskets[bname] = (seek, keylen + len(raw), keylen + len(raw))
            strings = _tstring("TTree") + _tstring(tname) + _tstring(title)
            body = _ttree(
                tname,
                title,
                len(records),
                branches,
                baskets,
                _KEY_BIG.size + len(strings),
            )
            k = key(_KEY_BIG, "TTree", tname, title, len(body), fh.tell())
            fh.write(k + body)
            top_keys.append(k)

        seek_keys = fh.tell()
        body = struct.pack(">i", len(top_keys)) + b"".join(top_keys)
        fh.write(key(_KEY, "TFile", filename, "", len(body), seek_keys) + body)
        nbytes_keys = fh.tell() - seek_keys

        seek_free = fh.tell()
        end = seek_free + _KEY.size + len(_tstring("TFile") + names) + 10
        if end >= _LIMIT:
            raise ValueError(f"{path}: CONEX file would exceed 2 GB")
        fh.write(
            key(_KEY, "TFile", filename, "", 10, seek_free)
            + struct.pack(">HII", 1, end, _LIMIT)
        )

        fh.seek(dir_at)
        fh.write(
            struct.pack(
                ">hIIiiiii",
                5,
                datime,
                datime,
                nbytes_keys,
                len(top) + len(names),
                _BEGIN,
                0,
                seek_keys,
            )
            + b"\x00\x01"
            + file_uuid
        )
        fh.seek(0)
        fh.write(
            struct.pack(
                ">4siiiiiiiBiiiH16s",
                b"root",
                62400,
                _BEGIN,
                end,
                seek_free,
                end - seek_free,
                1,
                len(top) + len(names),
                4,
                0,
                seek_info,
                nbytes_info,
                1,
                file_uuid,
            )
        )
