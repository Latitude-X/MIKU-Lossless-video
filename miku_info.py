#!/usr/bin/env python3
"""Independent reader for .miku v1/v2/v3: prints a structural summary and checks invariants.
usage: miku_info.py FILE [--frames]"""
import struct, sys

def read(path):
    d = open(path, "rb").read()
    assert d[:4] == b"MIKU", "bad magic"
    ver, fn, fd = struct.unpack_from("<HHH", d, 4)
    nframes, = struct.unpack_from("<I", d, 10)
    idx_off, = struct.unpack_from("<Q", d, 30)
    codec = d[45]
    video_off, = struct.unpack_from("<Q", d, 46)
    esz = 24 if codec == 3 else 16
    ents = []
    for i in range(nframes):
        o, sz, dur = struct.unpack_from("<QII", d, idx_off + i * esz)
        ref = struct.unpack_from("<i", d, idx_off + i * esz + 16)[0] if esz == 24 else -1
        ents.append(dict(i=i, off=o, size=sz & 0x7FFFFFFF, key=bool(sz & 0x80000000) if codec in (2, 3) else True, dur=dur, ref=ref))
    ao, asz, asamp = struct.unpack_from("<QII", d, idx_off + nframes * esz)
    return dict(data=d, ver=ver, codec=codec, n=nframes, idx_off=idx_off, video_off=video_off, esz=esz, ents=ents, audio_off=ao, audio_size=asz, flen=len(d))

def main():
    path = sys.argv[1]
    m = read(path)
    e = m["ents"]
    nk = sum(1 for x in e if x["key"])
    nr = sum(1 for x in e if not x["key"] and x["ref"] >= 0)
    nd = len(e) - nk - nr
    uniq = {}
    for x in e:
        uniq.setdefault(x["off"], x["size"])
    vbytes = sum(uniq.values())
    print(f"{path}: version={m['ver']} codec_flag={m['codec']} entry={m['esz']}B frames={m['n']} file={m['flen']}B")
    print(f"  keyframes={nk} delta_prev={nd} delta_ref={nr}  unique_packets={len(uniq)} video_bytes={vbytes} audio_bytes={m['audio_size']}")
    bad = 0
    for x in e:
        if not (m["video_off"] <= x["off"] and x["off"] + x["size"] <= m["audio_off"]):
            print("  !! frame", x["i"], "outside the video region"); bad += 1
        if x["ref"] >= 0:
            t = e[x["ref"]] if x["ref"] < m["n"] else None
            if t is None or not t["key"] or x["key"]:
                print("  !! frame", x["i"], "has an invalid ref", x["ref"]); bad += 1
    # worst-case seek cost: chain length from the nearest root (key or ref frame)
    worst = 0; run = 0
    for x in e:
        run = 0 if (x["key"] or x["ref"] >= 0) else run + 1
        worst = max(worst, run)
    print(f"  longest delta-prev run after a root: {worst}   invariants: {'OK' if not bad else 'BROKEN'}")
    if "--frames" in sys.argv:
        for x in e:
            role = "K" if x["key"] else (f"R{x['ref']}" if x["ref"] >= 0 else "D")
            print(f"   {x['i']:5d} {role:>6} off={x['off']} size={x['size']}")
    sys.exit(1 if bad else 0)
main()
