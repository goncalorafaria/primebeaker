"""Ragged sampler rows: int32 counts and contiguous typed values on the wire."""
import base64
import numpy as np


def encode(counts, values, dtype, mode=b"topk"):
    if mode not in (b"topk", b"mc32"):
        raise ValueError("Unknown sampler mode")
    return {'mode': mode, 'counts': np.asarray(counts, dtype=np.int32).tobytes(),
            'values': np.asarray(values, dtype=dtype).tobytes()}


def arrays(rows, dtype):
    counts = np.frombuffer(rows['counts'], dtype=np.int32)
    values = np.frombuffer(rows['values'], dtype=dtype)
    if (counts < 0).any() or counts.sum(dtype=np.int64) != values.size:
        raise ValueError('Malformed packed sampler rows')
    return counts, values


def pack(rows, dtype, n=0):
    if rows is None:
        return encode(np.zeros(n, dtype=np.int32), [], dtype)
    if isinstance(rows, dict):
        arrays(rows, dtype)
        return rows
    counts = np.fromiter((len(row) for row in rows), dtype=np.int32, count=len(rows))
    values = np.concatenate([np.asarray(row, dtype=dtype) for row in rows]) if counts.sum() else []
    return encode(counts, values, dtype)


def row_count(rows):
    return len(rows['counts']) // 4 if isinstance(rows, dict) else len(rows)


def concat(parts):
    modes = {p.get('mode', b'topk') for p in parts if len(p['values'])}
    if len(modes) > 1:
        raise ValueError('Cannot mix top-k and MC32 samples')
    mode = next(iter(modes), parts[0].get('mode', b'topk') if parts else b'topk')
    return {'mode': mode, 'counts': b''.join(p['counts'] for p in parts),
            'values': b''.join(p['values'] for p in parts)}


def prefix(rows, stop, dtype):
    counts, values = arrays(rows, dtype)
    return encode(counts[:stop], values[:int(counts[:stop].sum(dtype=np.int64))], dtype, rows.get('mode', b'topk'))


def pad(rows, n):
    return {'mode': rows.get('mode', b'topk'), 'counts': rows['counts'] + bytes(4*n), 'values': rows['values']}


def align(rows, mask, dtype):
    mask = np.asarray(mask, dtype=np.bool_)
    if rows is None:
        return pack(None, dtype, len(mask))
    rows = pack(rows, dtype)
    counts, values = arrays(rows, dtype)
    if len(counts) != int(mask.sum()):
        raise ValueError('Sampler head rows must match sampled node tokens')
    aligned = np.zeros(len(mask), dtype=np.int32)
    aligned[mask] = counts
    return encode(aligned, values, dtype, rows.get('mode', b'topk'))


def validate(ids, logps, n):
    counts, iv = arrays(ids, np.int32)
    lc, lv = arrays(logps, np.float32)
    if len(counts) != n or not np.array_equal(counts, lc):
        raise ValueError('Sampler head rows must align with tokens and logprobs')
    if (iv < 0).any() or not np.isfinite(lv).all() or (lv > 0).any():
        raise ValueError('Invalid sampler head ids or logprobs')
    mode = ids.get('mode', b'topk')
    if mode != logps.get('mode', b'topk'):
        raise ValueError('Sampler modes disagree')
    if mode == b'mc32':
        if ((counts != 0) & (counts != 32)).any():
            raise ValueError('MC32 rows must contain exactly 32 draws or be padding')
        return
    if mode != b'topk':
        raise ValueError('Unknown sampler mode')
    rows = np.repeat(np.arange(n, dtype=np.int64), counts)
    order = np.lexsort((iv, rows))
    sr, si = rows[order], iv[order]
    if ((sr[1:] == sr[:-1]) & (si[1:] == si[:-1])).any():
        raise ValueError('Duplicate sampler head token id')
    mass = np.bincount(rows, weights=np.exp(lv.astype(np.float64)), minlength=n)
    if (mass > 1.0 + 1e-5).any():
        raise ValueError('Sampler head probability mass exceeds one')


def dense(ids, logps, n):
    ids, logps = pack(ids, np.int32), pack(logps, np.float32)
    counts, iv = arrays(ids, np.int32)
    lc, lv = arrays(logps, np.float32)
    if len(counts) != n or not np.array_equal(counts, lc):
        raise ValueError('Sampler head rows must align with tokens and logprobs')
    k = max(1, int(counts.max(initial=0)))
    valid = np.arange(k)[None, :] < counts[:, None]
    out_ids = np.full((n, k), -1, dtype=np.int64)
    out_logps = np.zeros((n, k), dtype=np.float32)
    out_ids[valid], out_logps[valid] = iv, lv
    return out_ids, out_logps


def decode(wire, completion_ids, sampled_logprobs, sentinel):
    counts = np.frombuffer(base64.b64decode(wire['counts'], validate=True), dtype=np.int32)
    ids = np.frombuffer(base64.b64decode(wire['ids'], validate=True), dtype=np.int32)
    logps = np.frombuffer(base64.b64decode(wire['logprobs'], validate=True), dtype=np.float32)
    n = len(completion_ids)
    if len(counts) != n or (counts < 0).any() or counts.sum(dtype=np.int64) != len(ids) or len(ids) != len(logps):
        raise ValueError('Packed sampler head does not align with completion tokens')
    if wire.get('mode') == 'mc32':
        if not (counts == 32).all():
            raise ValueError('MC32 requires exactly 32 draws at every generated position')
        sampled_lp = np.asarray(sampled_logprobs, dtype=np.float64)
        if sampled_lp.shape != (n,) or not np.isfinite(sampled_lp).all() or (sampled_lp > 0).any():
            raise ValueError('Invalid sampled-token logprob')
        probes = ids.reshape(n, 32)
        match = probes == np.asarray(completion_ids, dtype=np.int32)[:, None]
        if not np.isclose(logps.reshape(n, 32), sampled_lp[:, None], rtol=1e-5, atol=1e-5)[match].all():
            raise ValueError('MC32 and sampled-token probabilities disagree')
        ip = encode(counts, ids, np.int32, b'mc32')
        lp = encode(counts, logps, np.float32, b'mc32')
        validate(ip, lp, n)
        return ip, lp
    if wire.get('mode', 'topk') != 'topk':
        raise ValueError('Unknown sampler wire mode')
    if np.isnan(logps).any() or (logps > 0).any():
        raise ValueError('Invalid sampler head logprob')
    rows = np.repeat(np.arange(n, dtype=np.int64), counts)
    keep = np.isfinite(logps) & (logps > sentinel)
    rows, ids, logps = rows[keep], ids[keep], logps[keep]
    counts = np.bincount(rows, minlength=n).astype(np.int32)
    if (counts == 0).any():
        raise ValueError('Missing sampler head probabilities')
    sampled = np.asarray(completion_ids, dtype=np.int32)
    sampled_lp = np.asarray(sampled_logprobs, dtype=np.float64)
    if sampled_lp.shape != (n,) or not np.isfinite(sampled_lp).all() or (sampled_lp > 0).any():
        raise ValueError('Invalid sampled-token logprob')
    match = ids == sampled[rows]
    if not (np.abs(logps[match] - sampled_lp[rows[match]]) <= np.maximum(1e-5, 1e-5 * np.maximum(np.abs(logps[match]), np.abs(sampled_lp[rows[match]])))).all():
        raise ValueError('Sampler head and sampled-token logprobs disagree')
    present = np.bincount(rows[match], minlength=n) > 0
    missing = ~present
    new_counts = counts + missing.astype(np.int32)
    starts = np.cumsum(counts, dtype=np.int64) - counts
    new_starts = np.cumsum(new_counts, dtype=np.int64) - new_counts
    positions = np.arange(len(ids)) - starts[rows] + new_starts[rows]
    out_ids = np.empty(int(new_counts.sum()), dtype=np.int32)
    out_logps = np.empty(len(out_ids), dtype=np.float32)
    out_ids[positions], out_logps[positions] = ids, logps
    extra = (new_starts + counts)[missing]
    out_ids[extra], out_logps[extra] = sampled[missing], sampled_lp[missing]
    ip, lp = encode(new_counts, out_ids, np.int32), encode(new_counts, out_logps, np.float32)
    validate(ip, lp, n)
    return ip, lp
