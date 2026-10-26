#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
#  AS Blink
#  Copyright (c) 2026 Stewart Oliver, AstroShed (astroshed.co.uk)
#  All rights reserved.
#
#  FREEWARE LICENCE
#  1. AS Blink is free. You may use it without charge, for personal or
#     commercial imaging work.
#  2. You may copy and share AS Blink, unchanged, as long as this licence and
#     copyright notice stay with it and nobody is charged for it.
#  3. You may not sell AS Blink, include it in a paid product or bundle, or
#     share changed versions of it, without written permission from the author.
#  4. You may read the code and change it for your own private use.
#  5. Copyright stays with the author. This licence gives you the right to use
#     the program; it does not give you ownership of it.
#  6. AS Blink is provided "as is", without warranty of any kind. The author is
#     not liable for any loss or damage from using it, including lost or moved
#     files. Sorting into folders moves your files: keep backups of your data.
#  7. AS Blink uses third-party libraries (PySide6/Qt, NumPy, OpenCV, Pillow,
#     tifffile, rawpy/LibRaw, lz4, zstandard, astropy), each under its own
#     licence. This licence does not change their terms.
# ---------------------------------------------------------------------------
"""
AS Blink - a fast blink comparator for astro sub-frames and videos.
AstroShed - astroshed.co.uk

Needs:  PySide6  numpy  opencv-python-headless  Pillow  tifffile
Optional: rawpy (camera RAW), lz4 + zstandard (compressed XISF),
          astropy (tile-compressed FITS)
"""
APP_NAME = 'AS Blink'
APP_VERSION = '1.0.1'

import os
import re
import sys
import json
import math
import time
import zlib
import base64
import struct
import shutil
import datetime
import threading
import traceback
import collections
import logging
import logging.handlers
import platform
import xml.etree.ElementTree as ET

import numpy as np
import cv2

try:
    from PIL import Image
except Exception:
    Image = None
try:
    import tifffile
except Exception:
    tifffile = None
try:
    import rawpy
except Exception:
    rawpy = None
try:
    import lz4.block as lz4block
except Exception:
    lz4block = None
try:
    import zstandard as zstd
except Exception:
    zstd = None

cv2.setNumThreads(max(1, (os.cpu_count() or 2) // 2))



def n_frames(n):
    return '%d frame%s' % (n, '' if n == 1 else 's')



# ---------------------------------------------------------------------------
#  LOG  (for beta testers to send in when something goes wrong)
# ---------------------------------------------------------------------------
LOG = logging.getLogger('asblink')
LOG_FILE = None


def log_dir():
    if sys.platform.startswith('win'):
        base = os.environ.get('LOCALAPPDATA') or os.path.join(os.path.expanduser('~'), 'AppData', 'Local')
    elif sys.platform == 'darwin':
        base = os.path.join(os.path.expanduser('~'), 'Library', 'Logs')
    else:
        base = os.environ.get('XDG_STATE_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'state')
    return os.path.join(base, 'AstroShed', 'AS Blink')


def _private(text):
    """Take the user's home folder and user name out of a log line."""
    home = os.path.expanduser('~')
    if home and len(home) > 3:
        text = text.replace(home, '~').replace(home.replace('\\', '/'), '~')
    user = os.environ.get('USERNAME') or os.environ.get('USER') or ''
    if len(user) >= 3:
        text = re.sub(r'([\\/])' + re.escape(user) + r'([\\/])', r'\1<user>\2', text)
    return text


class _PrivateFormatter(logging.Formatter):
    def format(self, record):
        return _private(super().format(record))


def setup_log():
    global LOG_FILE
    if LOG.handlers:
        return
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    try:
        d = log_dir()
        os.makedirs(d, exist_ok=True)
        LOG_FILE = os.path.join(d, 'AS_Blink.log')
        h = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1000000, backupCount=2, encoding='utf-8')
        h.setFormatter(_PrivateFormatter('%(asctime)s  %(levelname)-7s %(message)s'))
        LOG.addHandler(h)
    except Exception:
        LOG_FILE = None
        LOG.addHandler(logging.NullHandler())


def log_files():
    """Oldest first: AS_Blink.log.2, .log.1, .log"""
    if not LOG_FILE:
        return []
    return [p for p in (LOG_FILE + '.2', LOG_FILE + '.1', LOG_FILE) if os.path.exists(p)]


def fname(path):
    """Only the file name goes in the log, never the full path."""
    try:
        return os.path.basename(str(path).rstrip('\\/')) or str(path)
    except Exception:
        return '?'


def _mod_version(name):
    try:
        m = __import__(name)
        for part in name.split('.')[1:]:
            m = getattr(m, part)
        return str(getattr(m, '__version__', 'yes'))
    except Exception:
        return 'not installed'


def system_summary():
    lines = ['%s %s' % (APP_NAME, APP_VERSION),
             'Started from: %s' % ('packaged app' if getattr(sys, 'frozen', False) else 'Python script'),
             'Python %s (%s)' % (platform.python_version(), platform.architecture()[0]),
             'System: %s' % platform.platform(),
             'CPU threads: %s' % os.cpu_count(),
             'numpy %s, OpenCV %s' % (np.__version__, cv2.__version__)]
    try:
        import PySide6
        from PySide6 import QtCore
        lines.append('PySide6 %s, Qt %s' % (PySide6.__version__, QtCore.qVersion()))
    except Exception:
        lines.append('PySide6: not found')
    lines.append('Pillow %s, tifffile %s, rawpy %s, lz4 %s, zstandard %s, astropy %s' % tuple(
        _mod_version(m) for m in ('PIL', 'tifffile', 'rawpy', 'lz4', 'zstandard', 'astropy')))
    return lines
# ---------------------------------------------------------------------------
#  FILE READERS
#  Every reader returns an Frame-data dict:
#     data    : numpy array, (h, w) mono/CFA or (h, w, 3) colour, top-down rows
#     kind    : 'mono', 'cfa' or 'rgb'
#     pattern : Bayer pattern for 'cfa' (RGGB, BGGR, GRBG, GBRG) after any
#               row-order / offset correction, else ''
#     header  : list of (key, value, comment)
#     scale   : factor that maps the data to a 0..65535 working range
#     satlevel: value (after scale) treated as saturated
# ---------------------------------------------------------------------------

FITS_EXT = ('.fit', '.fits', '.fts', '.fz')
XISF_EXT = ('.xisf',)
TIFF_EXT = ('.tif', '.tiff')
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
RAW_EXT = ('.cr2', '.cr3', '.crw', '.nef', '.nrw', '.arw', '.srf', '.sr2',
           '.dng', '.orf', '.raf', '.rw2', '.pef', '.srw', '.3fr', '.iiq',
           '.erf', '.mef', '.mos', '.kdc', '.raw')
SER_EXT = ('.ser',)
VIDEO_EXT = ('.avi', '.mp4', '.mov')
MULTI_EXT = SER_EXT + VIDEO_EXT          # files holding many frames
ALL_EXT = FITS_EXT + XISF_EXT + TIFF_EXT + IMG_EXT + RAW_EXT + SER_EXT + VIDEO_EXT

PATTERNS = ('RGGB', 'BGGR', 'GRBG', 'GBRG')
# OpenCV names its Bayer codes from the second row, so RGGB -> BayerBG etc.
# (checked against a test mosaic, and these names work on old OpenCV too)
_CV_BAYER = {'RGGB': 'BG', 'BGGR': 'RG', 'GRBG': 'GB', 'GBRG': 'GR'}


class ReadError(Exception):
    pass


def _pattern_flip_rows(p):
    return p[2:4] + p[0:2]


def _pattern_flip_cols(p):
    return p[1] + p[0] + p[3] + p[2]


def _pattern_shift(p, xoff, yoff):
    if yoff % 2:
        p = _pattern_flip_rows(p)
    if xoff % 2:
        p = _pattern_flip_cols(p)
    return p


def _clean_pattern(s):
    s = str(s or '').strip().strip("'").upper().replace(' ', '')
    return s if s in PATTERNS else ''


def debayer(cfa, pattern, quality=True):
    """CFA (h, w) uint16 -> RGB (h, w, 3) uint16."""
    name = 'COLOR_Bayer%s2RGB%s' % (_CV_BAYER[pattern], '_EA' if quality else '')
    code = getattr(cv2, name, None)
    if code is None:
        code = getattr(cv2, 'COLOR_Bayer%s2RGB' % _CV_BAYER[pattern])
    if cfa.dtype != np.uint16:
        cfa = np.clip(cfa, 0, 65535).astype(np.uint16)
    return cv2.cvtColor(np.ascontiguousarray(cfa), code)


def superpixel(cfa, pattern):
    """CFA -> half size RGB float32 (each 2x2 cell -> one RGB pixel)."""
    h, w = cfa.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    c = cfa[:h2, :w2].astype(np.float32)
    cells = [c[0::2, 0::2], c[0::2, 1::2], c[1::2, 0::2], c[1::2, 1::2]]
    out = {'R': [], 'G': [], 'B': []}
    for ch, cell in zip(pattern, cells):
        out[ch].append(cell)
    r = out['R'][0]
    b = out['B'][0]
    g = (out['G'][0] + out['G'][1]) * 0.5
    return np.dstack([r, g, b])


def _hdr_get(header, *keys):
    for key, val, _c in header:
        if key in keys:
            return val
    return None


def hdr_dict(header):
    d = {}
    for key, val, _c in header:
        if key and key not in d:
            d[key] = val
    return d


def _to_number(v):
    if isinstance(v, (int, float)):
        return v
    try:
        s = str(v).strip().strip("'").strip()
        if s.upper() in ('T', 'TRUE'):
            return True
        if s.upper() in ('F', 'FALSE'):
            return False
        if re.fullmatch(r'[+-]?\d+', s):
            return int(s)
        return float(s)
    except Exception:
        return v


def _working_scale(arr):
    """Factor that maps native values to the 0..65535 working range."""
    dt = arr.dtype
    if dt == np.uint8:
        return 257.0
    if dt in (np.uint16, np.int16):
        return 1.0
    if dt in (np.uint32, np.int32, np.uint64, np.int64):
        sample = arr[::max(1, arr.shape[0] // 256), ::max(1, arr.shape[1] // 256)]
        mx = float(np.max(sample)) if sample.size else 0.0
        return 1.0 if mx <= 65535 else 1.0 / 65536.0
    # floating point
    sample = arr[::max(1, arr.shape[0] // 256), ::max(1, arr.shape[1] // 256)]
    finite = sample[np.isfinite(sample)]
    mx = float(np.percentile(finite, 99.99)) if finite.size else 1.0
    if mx <= 1.5:
        return 65535.0
    if mx <= 70000:
        return 1.0
    return 65535.0 / mx


# ------------------------------- FITS --------------------------------------

def _parse_fits_card(card):
    key = card[:8].strip()
    if key in ('COMMENT', 'HISTORY', '') or card[8:10] != '= ':
        return key, None, card[8:].rstrip()
    rest = card[10:]
    rest_s = rest.lstrip()
    if rest_s.startswith("'"):
        # string, '' is an escaped quote
        i = rest.index("'") + 1
        val = ''
        while i < len(rest):
            if rest[i] == "'":
                if i + 1 < len(rest) and rest[i + 1] == "'":
                    val += "'"
                    i += 2
                    continue
                i += 1
                break
            val += rest[i]
            i += 1
        comment = rest[i:].split('/', 1)[1].strip() if '/' in rest[i:] else ''
        return key, val.rstrip(), comment
    if '/' in rest:
        v, c = rest.split('/', 1)
    else:
        v, c = rest, ''
    return key, _to_number(v.strip()), c.strip()


def _read_fits_header(f):
    cards = []
    raw = []
    while True:
        block = f.read(2880)
        if len(block) < 2880:
            raise ReadError('FITS header is cut short')
        text = block.decode('ascii', errors='replace')
        done = False
        for i in range(36):
            card = text[i * 80:(i + 1) * 80]
            raw.append(card)
            if card[:8].strip() == 'END':
                done = True
                break
            cards.append(_parse_fits_card(card))
        if done:
            return cards


def _fits_data_size(h):
    d = hdr_dict(h)
    naxis = int(d.get('NAXIS', 0) or 0)
    if naxis == 0:
        return 0
    n = 1
    for i in range(1, naxis + 1):
        n *= int(d.get('NAXIS%d' % i, 0) or 0)
    n = abs(int(d.get('BITPIX', 8))) // 8 * int(d.get('GCOUNT', 1) or 1) * (n + int(d.get('PCOUNT', 0) or 0))
    return n


def read_fits(path, want_data=True):
    with open(path, 'rb') as f:
        header = _read_fits_header(f)
        d = hdr_dict(header)
        offset = f.tell()
        # primary with no data -> look at the first image extension
        if int(d.get('NAXIS', 0) or 0) == 0 or d.get('EXTEND') and int(d.get('NAXIS', 0) or 0) == 0:
            size = _fits_data_size(header)
            f.seek(offset + (size + 2879) // 2880 * 2880)
            try:
                ext = _read_fits_header(f)
            except ReadError:
                raise ReadError('No image found in this FITS file')
            ed = hdr_dict(ext)
            if ed.get('ZIMAGE') or ed.get('XTENSION', '').strip() == 'BINTABLE':
                return _read_fits_astropy(path, header, want_data)
            d = ed
            offset = f.tell()
            header = header + [('', None, '---- extension ----')] + ext
    naxis = int(d.get('NAXIS', 0) or 0)
    if naxis < 2:
        raise ReadError('FITS image has no 2D data')
    w = int(d['NAXIS1'])
    h = int(d['NAXIS2'])
    planes = int(d.get('NAXIS3', 1) or 1) if naxis >= 3 else 1
    bitpix = int(d['BITPIX'])
    dt = {8: '>u1', 16: '>i2', 32: '>i4', 64: '>i8', -32: '>f4', -64: '>f8'}.get(bitpix)
    if dt is None:
        raise ReadError('Unsupported BITPIX %s' % bitpix)
    info = {'header': header, 'width': w, 'height': h, 'planes': planes}
    if not want_data:
        return info
    nplanes = 3 if planes == 3 else 1
    count = w * h * nplanes
    bzero = float(d.get('BZERO', 0) or 0)
    bscale = float(d.get('BSCALE', 1) or 1)
    if bitpix == 16 and bzero == 32768 and bscale == 1:
        dt = '>u2'      # unsigned 16 bit: flipping the top bit is the same as adding 32768
    raw = np.fromfile(path, dtype=dt, count=count, offset=offset)
    if raw.size < count:
        raise ReadError('FITS data is cut short')
    if bitpix == 16 and bzero == 32768 and bscale == 1:
        raw ^= np.array(0x8000, dtype='>u2')
        arr = raw.astype(np.uint16)
    elif bitpix == 8 and bzero == 0 and bscale == 1:
        arr = raw.astype(np.uint8)
    elif bitpix == 32 and bzero == 2147483648 and bscale == 1:
        arr = (raw.astype(np.int64) + 2147483648).astype(np.uint32)
    elif bitpix < 0 and bzero == 0 and bscale == 1:
        arr = raw.astype(np.float32)
    else:
        arr = (raw.astype(np.float64) * bscale + bzero).astype(np.float32)
        if bitpix > 0:
            mn = float(arr.min()) if arr.size else 0
            mx = float(arr.max()) if arr.size else 0
            if mn >= 0 and mx <= 65535:
                arr = arr.astype(np.uint16)
    if nplanes == 3:
        arr = arr.reshape(3, h, w).transpose(1, 2, 0)
    else:
        arr = arr.reshape(h, w)
    info['data'] = arr
    info['source_dtype'] = 'BITPIX %d' % bitpix
    return info


def _read_fits_astropy(path, primary_header, want_data):
    try:
        from astropy.io import fits as afits
    except Exception:
        raise ReadError('This FITS file is tile compressed - install astropy to read it')
    with afits.open(path, memmap=False) as hl:
        hdu = None
        for x in hl:
            if x.is_image and x.data is not None and getattr(x.data, 'ndim', 0) >= 2:
                hdu = x
                break
        if hdu is None:
            raise ReadError('No image found in this FITS file')
        header = list(primary_header)
        header.append(('', None, '---- compressed image ----'))
        for card in hdu.header.cards:
            header.append((card.keyword, card.value if card.keyword not in ('COMMENT', 'HISTORY') else None,
                           card.comment if card.keyword not in ('COMMENT', 'HISTORY') else str(card.value)))
        data = hdu.data
        if data.ndim == 3:
            if data.shape[0] == 3:
                data = np.transpose(data, (1, 2, 0))
            else:
                data = data[0]
        info = {'header': header, 'width': data.shape[1], 'height': data.shape[0],
                'planes': 3 if data.ndim == 3 else 1}
        if want_data:
            if data.dtype.kind == 'f':
                data = data.astype(np.float32)
            elif data.dtype.kind in 'iu' and data.min() >= 0 and data.max() <= 65535:
                data = data.astype(np.uint16)
            else:
                data = data.astype(np.float32)
            info['data'] = np.ascontiguousarray(data)
            info['source_dtype'] = 'FITS compressed'
        return info


# ------------------------------- XISF --------------------------------------

def _xisf_decompress(buf, codec, usize, itemsize, subblocks=None):
    codec = codec.lower()
    shuffled = codec.endswith('+sh')
    base = codec.replace('+sh', '')

    def one(b, n):
        if base == 'zlib':
            return zlib.decompress(b)
        if base in ('lz4', 'lz4hc'):
            if lz4block is None:
                raise ReadError('This XISF file uses LZ4 - install the lz4 package')
            return lz4block.decompress(b, uncompressed_size=n)
        if base == 'zstd':
            if zstd is None:
                raise ReadError('This XISF file uses Zstandard - install the zstandard package')
            return zstd.ZstdDecompressor().decompress(b, max_output_size=n)
        raise ReadError('Unknown XISF compression %s' % codec)

    if subblocks:
        parts = []
        pos = 0
        for cs, us in subblocks:
            parts.append(one(buf[pos:pos + cs], us))
            pos += cs
        out = b''.join(parts)
    else:
        out = one(buf, usize)
    if shuffled and itemsize > 1:
        a = np.frombuffer(out, dtype=np.uint8)
        n = a.size // itemsize
        body = a[:n * itemsize].reshape(itemsize, n).T.reshape(-1)
        out = body.tobytes() + a[n * itemsize:].tobytes()
    return out


def read_xisf(path, want_data=True):
    with open(path, 'rb') as f:
        sig = f.read(8)
        if sig != b'XISF0100':
            raise ReadError('Not an XISF file')
        head = f.read(8)
        if len(head) < 8:
            raise ReadError('XISF header is cut short')
        hlen = struct.unpack('<I', head[:4])[0]
        xml_bytes = f.read(hlen)
        if len(xml_bytes) < hlen:
            raise ReadError('XISF header is cut short')
    try:
        root = ET.fromstring(xml_bytes.decode('utf-8', errors='replace').rstrip('\x00 \r\n\t'))
    except ET.ParseError as e:
        raise ReadError('XISF header could not be read (%s)' % e)

    def strip(tag):
        return tag.split('}', 1)[-1]

    img = None
    for el in root.iter():
        if strip(el.tag) == 'Image':
            img = el
            break
    if img is None:
        raise ReadError('No image found in this XISF file')
    geom = [int(x) for x in img.get('geometry', '0:0:1').split(':')]
    w, h = geom[0], geom[1]
    ch = geom[2] if len(geom) > 2 else 1
    header = []
    pattern = ''
    for el in img:
        t = strip(el.tag)
        if t == 'FITSKeyword':
            header.append((el.get('name', ''), _to_number(el.get('value', '').strip().strip("'").strip())
                           if el.get('value', '').strip()[:1] != "'" else el.get('value', '').strip().strip("'").rstrip(),
                           el.get('comment', '')))
        elif t == 'ColorFilterArray':
            pattern = _clean_pattern(el.get('pattern', ''))
    for el in img:
        if strip(el.tag) == 'Property':
            val = el.get('value')
            if val is None:
                val = (el.text or '').strip()
            header.append(('[' + el.get('id', '') + ']', val, 'XISF property'))
    info = {'header': header, 'width': w, 'height': h, 'planes': ch, 'xisf_pattern': pattern}
    if not want_data:
        return info
    fmt = img.get('sampleFormat', 'UInt16')
    dt = {'UInt8': np.uint8, 'UInt16': np.uint16, 'UInt32': np.uint32, 'UInt64': np.uint64,
          'Float32': np.float32, 'Float64': np.float64}.get(fmt)
    if dt is None:
        raise ReadError('Unsupported XISF sample format %s' % fmt)
    dtype = np.dtype(dt)
    if img.get('byteOrder', 'little') == 'big':
        dtype = dtype.newbyteorder('>')
    loc = img.get('location', '')
    parts = loc.split(':')
    if parts[0] == 'attachment':
        pos, size = int(parts[1]), int(parts[2])
        with open(path, 'rb') as f:
            f.seek(pos)
            buf = f.read(size)
    elif parts[0] == 'inline':
        enc = parts[1] if len(parts) > 1 else 'base64'
        txt = (img.text or '').strip()
        buf = base64.b64decode(txt) if enc == 'base64' else bytes.fromhex(txt)
    elif parts[0] == 'embedded':
        data_el = None
        for el in img:
            if strip(el.tag) == 'Data':
                data_el = el
        if data_el is None:
            raise ReadError('XISF embedded data missing')
        enc = data_el.get('encoding', 'base64')
        txt = (data_el.text or '').strip()
        buf = base64.b64decode(txt) if enc == 'base64' else bytes.fromhex(txt)
    else:
        raise ReadError('Unsupported XISF data location %s' % loc)
    comp = img.get('compression', '')
    if comp:
        cp = comp.split(':')
        codec = cp[0]
        usize = int(cp[1])
        isz = int(cp[2]) if len(cp) > 2 else dtype.itemsize
        sb = img.get('subblocks')
        subblocks = None
        if sb:
            subblocks = [tuple(int(v) for v in s.split(',')) for s in sb.split(':') if s]
        buf = _xisf_decompress(buf, codec, usize, isz, subblocks)
    arr = np.frombuffer(buf, dtype=dtype, count=w * h * ch)
    if dtype.byteorder == '>':
        arr = arr.astype(dtype.newbyteorder('='))
    storage = img.get('pixelStorage', 'Planar')
    if ch == 1:
        arr = arr.reshape(h, w)
    elif storage.lower() == 'normal':
        arr = arr.reshape(h, w, ch)[:, :, :3]
    else:
        arr = arr.reshape(ch, h, w)[:3].transpose(1, 2, 0)
    if ch == 2:
        arr = arr[:, :, 0] if arr.ndim == 3 else arr
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    lb = img.get('bounds')
    if lb and arr.dtype.kind == 'f':
        try:
            lo, hi = [float(x) for x in lb.split(':')]
            if (lo, hi) != (0.0, 1.0) and hi > lo:
                arr = ((arr - lo) / (hi - lo)).astype(np.float32)
        except Exception:
            pass
    info['data'] = np.ascontiguousarray(arr)
    info['source_dtype'] = fmt
    return info


# ------------------------------- TIFF / PNG / JPG --------------------------

def read_tiff(path, want_data=True):
    header = []
    if tifffile is None:
        return read_plain(path, want_data)
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        for tag in page.tags.values():
            v = tag.value
            if isinstance(v, (bytes, bytearray)):
                if len(v) > 200:
                    continue
                v = v.decode('latin-1', errors='replace')
            s = str(v)
            if len(s) > 300:
                s = s[:300] + '...'
            header.append((tag.name, s, 'TIFF tag'))
        # FITS keywords some programs put in the description
        desc = page.description or ''
        for line in desc.splitlines():
            if len(line) >= 10 and line[8:10] == '= ':
                header.append(_parse_fits_card(line.ljust(80)))
        info = {'header': header, 'width': page.shape[1] if len(page.shape) > 1 else 0,
                'height': page.shape[0], 'planes': 1}
        if not want_data:
            return info
        arr = page.asarray()
    if arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
            arr = arr.transpose(1, 2, 0)
        arr = arr[:, :, :3]
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    if arr.dtype == np.float16:
        arr = arr.astype(np.float32)
    info['data'] = np.ascontiguousarray(arr)
    info['width'], info['height'] = arr.shape[1], arr.shape[0]
    info['source_dtype'] = str(arr.dtype)
    return info


def read_plain(path, want_data=True):
    raw = np.fromfile(path, dtype=np.uint8)
    arr = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ReadError('Image could not be decoded')
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    header = [('FORMAT', os.path.splitext(path)[1][1:].upper(), ''),
              ('BITDEPTH', arr.dtype.itemsize * 8, 'bits per sample')]
    if Image is not None:
        try:
            with Image.open(path) as im:
                ex = im.getexif()
                from PIL import ExifTags
                for k, v in ex.items():
                    name = ExifTags.TAGS.get(k, str(k))
                    s = str(v)
                    if len(s) < 200:
                        header.append((name, s, 'EXIF'))
                try:
                    sub = ex.get_ifd(0x8769)
                    for k, v in sub.items():
                        name = ExifTags.TAGS.get(k, str(k))
                        s = str(v)
                        if len(s) < 200:
                            header.append((name, s, 'EXIF'))
                except Exception:
                    pass
        except Exception:
            pass
    return {'header': header, 'width': arr.shape[1], 'height': arr.shape[0],
            'planes': 3 if arr.ndim == 3 else 1, 'data': arr, 'source_dtype': str(arr.dtype),
            'lossy': os.path.splitext(path)[1].lower() in ('.jpg', '.jpeg', '.webp')}


# ------------------------------- camera RAW --------------------------------

def _exif_header(path):
    header = []
    if Image is None:
        return header
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            from PIL import ExifTags
            items = list(ex.items())
            try:
                items += list(ex.get_ifd(0x8769).items())
            except Exception:
                pass
            for k, v in items:
                name = ExifTags.TAGS.get(k, str(k))
                s = str(v)
                if len(s) < 200 and name not in ('MakerNote',):
                    header.append((name, s, 'EXIF'))
    except Exception:
        pass
    return header


def read_raw(path, want_data=True):
    if rawpy is None:
        raise ReadError('Camera RAW needs the rawpy package')
    header = []
    with rawpy.imread(path) as raw:
        cd = raw.color_desc.decode('ascii', errors='replace')
        pat = raw.raw_pattern
        sizes = raw.sizes
        header += [('CAMERA', '', ''),
                   ('WIDTH', int(sizes.width), 'visible width'),
                   ('HEIGHT', int(sizes.height), 'visible height'),
                   ('BLACKLVL', int(np.mean(raw.black_level_per_channel)), 'black level'),
                   ('WHITELVL', int(raw.white_level), 'white level'),
                   ('CAMWB', ' '.join('%.3f' % v for v in raw.camera_whitebalance), 'camera white balance'),
                   ('COLORS', cd, 'CFA colours')]
        header = [hh for hh in header if hh[0] != 'CAMERA']
        pattern = ''
        if pat is not None and pat.shape == (2, 2) and raw.num_colors >= 3:
            letters = ''.join(cd[int(pat[y, x])] for y in range(2) for x in range(2))
            letters = letters.replace('g', 'G')
            if letters in PATTERNS:
                pattern = letters
        header.append(('BAYERPAT', pattern or 'not Bayer', 'from RAW file'))
        info = {'width': int(sizes.width), 'height': int(sizes.height), 'planes': 1}
        if want_data:
            if pattern:
                a = raw.raw_image_visible.astype(np.float32)
                bl = float(np.mean(raw.black_level_per_channel))
                wl = float(raw.white_level)
                a = np.clip((a - bl) / max(1.0, wl - bl), 0, 1) * 65535.0
                info['data'] = a.astype(np.uint16)
                info['raw_pattern'] = pattern
            else:
                rgb = raw.postprocess(output_bps=16, no_auto_bright=True, gamma=(1, 1),
                                      use_camera_wb=True)
                info['data'] = rgb
            info['source_dtype'] = 'camera RAW'
    info['header'] = header + _exif_header(path)
    return info


# ------------------------------- SER ---------------------------------------

def read_ser_info(path):
    with open(path, 'rb') as f:
        hdr = f.read(178)
    if len(hdr) < 178 or not hdr[:14].startswith(b'LUCAM'):
        raise ReadError('Not a SER file')
    (lu, color, little, w, h, depth, count) = struct.unpack('<7i', hdr[14:42])
    observer = hdr[42:82].split(b'\0')[0].decode('latin-1').strip()
    instr = hdr[82:122].split(b'\0')[0].decode('latin-1').strip()
    scope = hdr[122:162].split(b'\0')[0].decode('latin-1').strip()
    dt, dtu = struct.unpack('<qq', hdr[162:178])
    planes = 3 if color in (100, 101) else 1
    bpp = 1 if depth <= 8 else 2
    pattern = {8: 'RGGB', 9: 'GRBG', 10: 'GBRG', 11: 'BGGR'}.get(color, '')
    date = ''
    if dtu > 0:
        try:
            date = (datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=dtu / 10)).isoformat()
        except Exception:
            date = ''
    header = [('FORMAT', 'SER', ''), ('COLORID', color, 'SER colour id'),
              ('BAYERPAT', pattern or ('RGB' if planes == 3 else 'mono'), ''),
              ('NAXIS1', w, ''), ('NAXIS2', h, ''), ('BITDEPTH', depth, ''),
              ('FRAMES', count, 'frames in file'), ('OBSERVER', observer, ''),
              ('INSTRUME', instr, ''), ('TELESCOP', scope, ''), ('DATE-OBS', date, 'file start (UTC)')]
    # SER files written by some programs say little-endian = 0 but are little-endian.
    return {'width': w, 'height': h, 'planes': planes, 'depth': depth, 'bpp': bpp,
            'count': count, 'color': color, 'pattern': pattern, 'header': header,
            'little': little, 'frame_bytes': w * h * planes * bpp, 'container': 'ser'}


def read_ser_frame(path, info, index):
    off = 178 + index * info['frame_bytes']
    n = info['width'] * info['height'] * info['planes']
    dt = np.uint8 if info['bpp'] == 1 else np.dtype('<u2')
    arr = np.fromfile(path, dtype=dt, count=n, offset=off)
    if arr.size < n:
        raise ReadError('SER frame %d is cut short' % (index + 1))
    if info['planes'] == 3:
        arr = arr.reshape(info['height'], info['width'], 3)
        if info['color'] == 101:
            arr = arr[:, :, ::-1]
    else:
        arr = arr.reshape(info['height'], info['width'])
    if info['bpp'] == 2 and 8 < info['depth'] < 16:
        arr = (arr.astype(np.uint32) << (16 - info['depth'])).clip(0, 65535).astype(np.uint16)
    return np.ascontiguousarray(arr)


def ser_frame_times(path, info):
    """Per-frame UTC timestamps from the trailer, if present."""
    try:
        size = os.path.getsize(path)
        trailer = 178 + info['count'] * info['frame_bytes']
        if size >= trailer + 8 * info['count']:
            t = np.fromfile(path, dtype='<i8', count=info['count'], offset=trailer)
            return t
    except Exception:
        pass
    return None


def write_ser(src_path, info, indices, out_path):
    with open(src_path, 'rb') as f:
        hdr = bytearray(f.read(178))
    struct.pack_into('<i', hdr, 38, len(indices))
    times = ser_frame_times(src_path, info)
    with open(src_path, 'rb') as f, open(out_path, 'wb') as o:
        o.write(hdr)
        for i in indices:
            f.seek(178 + i * info['frame_bytes'])
            o.write(f.read(info['frame_bytes']))
        if times is not None:
            o.write(np.asarray([times[i] for i in indices], dtype='<i8').tobytes())


# ------------------------------- AVI / MP4 / MOV ---------------------------

class _VideoReader:
    """One open video. Frames are decoded in order where possible (seeking in
    a compressed video is slow), with the last few kept, and one thread at a
    time, because an OpenCV capture cannot be shared between threads."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.cap = None
        self.pos = 0                      # the frame the next read() returns
        self.cache = collections.OrderedDict()

    def _open(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ReadError('This video could not be opened (its codec may not be supported)')
        self.cap = cap
        self.pos = 0

    def close(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.cache.clear()

    def frame(self, i):
        with self.lock:
            hit = self.cache.get(i)
            if hit is not None:
                self.cache.move_to_end(i)
                return hit
            if self.cap is None:
                self._open()
            if i < self.pos or i > self.pos + 60:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                self.pos = i
            while self.pos < i:
                if not self.cap.grab():
                    raise ReadError('Video frame %d could not be read' % (i + 1))
                self.pos += 1
            ok, fr = self.cap.read()
            if not ok or fr is None:
                self.cap.release()
                self.cap = None
                raise ReadError('Video frame %d could not be read' % (i + 1))
            self.pos = i + 1
            self.cache[i] = fr
            while len(self.cache) > 24:
                self.cache.popitem(last=False)
            return fr


_videos = collections.OrderedDict()
_videos_lock = threading.Lock()


def _video(path):
    with _videos_lock:
        r = _videos.get(path)
        if r is None:
            r = _videos[path] = _VideoReader(path)
            while len(_videos) > 4:
                _p, old = _videos.popitem(last=False)
                old.close()
        else:
            _videos.move_to_end(path)
        return r


def forget_video(path):
    """Close a video (before it is moved, or when the list is cleared)."""
    with _videos_lock:
        r = _videos.pop(path, None)
    if r is not None:
        r.close()


def read_video_info(path):
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ReadError('This video could not be opened (its codec may not be supported)')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = ''.join(chr((cc >> (8 * k)) & 255) for k in range(4)).strip('\0 ') if cc else ''
        ok, first = cap.read()
        if not ok or first is None:
            raise ReadError('No frames could be read from this video')
        if count <= 0:
            # the file does not say: count them
            count = 1
            while cap.grab():
                count += 1
        looks = [first]
        if count > 2:
            # a second look half way in, in case the first frame is blank
            cap.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
            ok, mid = cap.read()
            if ok and mid is not None:
                looks.append(mid)
    finally:
        cap.release()

    def grey(a):
        # compressed video (MJPEG, H.264) puts small colour errors into grey
        # pictures, so 'grey' means the channels differ by a level or two at most
        if a.ndim == 2:
            return True
        s = a[::2, ::2].astype(np.int16)
        d = np.maximum(np.abs(s[:, :, 0] - s[:, :, 1]), np.abs(s[:, :, 1] - s[:, :, 2]))
        lit = s.max(axis=2) > 24          # judge on the object, not the black sky round it
        q = float(np.percentile(d[lit], 99)) if lit.sum() > 50 else float(np.percentile(d, 99.9))
        return float(np.mean(d)) < 3 and q <= 6
    mono = all(grey(a) for a in looks)
    h, w = first.shape[:2]
    header = [('FORMAT', os.path.splitext(path)[1][1:].upper() + ' video', ''),
              ('CODEC', codec or '?', 'video codec (FourCC)'),
              ('NAXIS1', w, ''), ('NAXIS2', h, ''), ('BITDEPTH', 8, ''),
              ('COLOR', 'mono' if mono else 'colour', ''),
              ('FRAMES', count, 'frames in file'),
              ('FPS', round(fps, 3) if fps else '?', 'frames per second')]
    return {'width': w, 'height': h, 'planes': 1 if mono else 3, 'depth': 8, 'bpp': 1,
            'count': count, 'pattern': '', 'header': header, 'fps': fps, 'mono': mono,
            'container': 'video'}


def read_video_frame(path, info, index):
    fr = _video(path).frame(index)
    if fr.ndim == 2:
        return np.ascontiguousarray(fr)
    if info.get('mono'):
        return np.ascontiguousarray(fr[:, :, 1])
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def read_multi_info(path):
    """SER or video: the details every frame inside it shares."""
    if path.lower().endswith(SER_EXT):
        return read_ser_info(path)
    return read_video_info(path)


def is_video(src):
    return src.ser is not None and src.ser.get('container') == 'video'


# ------------------------------- dispatcher --------------------------------

class Source:
    """One blinkable frame: a file, or one frame inside a SER or video file."""
    __slots__ = ('path', 'index', 'ser', 'name')

    def __init__(self, path, index=-1, ser=None):
        self.path = path
        self.index = index
        self.ser = ser
        base = os.path.basename(path)
        self.name = base if index < 0 else '%s [%d]' % (base, index + 1)

    @property
    def key(self):
        return (self.path, self.index)


def read_source(src, want_data=True):
    """Read one frame and work out mono / CFA / RGB and the Bayer pattern."""
    ext = os.path.splitext(src.path)[1].lower()
    xisf_pat = ''
    raw_pat = ''
    if src.ser is not None and src.ser.get('container') == 'video':
        info = dict(src.ser)
        info['header'] = list(src.ser['header']) + [('FRAME', src.index + 1, 'frame number in the video')]
        if want_data:
            info['data'] = read_video_frame(src.path, src.ser, src.index)
            info['source_dtype'] = 'video 8 bit'
    elif src.ser is not None:
        info = dict(src.ser)
        info['header'] = list(src.ser['header']) + [('FRAME', src.index + 1, 'frame number in SER')]
        if want_data:
            info['data'] = read_ser_frame(src.path, src.ser, src.index)
            info['source_dtype'] = 'SER %d bit' % src.ser['depth']
        raw_pat = src.ser['pattern']
    elif ext in FITS_EXT:
        info = read_fits(src.path, want_data)
    elif ext in XISF_EXT:
        info = read_xisf(src.path, want_data)
        xisf_pat = info.get('xisf_pattern', '')
    elif ext in TIFF_EXT:
        info = read_tiff(src.path, want_data)
    elif ext in RAW_EXT:
        info = read_raw(src.path, want_data)
        raw_pat = info.get('raw_pattern', '')
    else:
        info = read_plain(src.path, want_data)
    header = info['header']
    d = hdr_dict(header)
    # Bayer pattern from the header
    pattern = raw_pat or xisf_pat
    note = ''
    if not pattern:
        pattern = _clean_pattern(d.get('BAYERPAT') or d.get('COLORTYP') or d.get('CFA-PAT') or '')
        if pattern:
            xo = int(_to_number(d.get('XBAYROFF', 0)) or 0) if isinstance(_to_number(d.get('XBAYROFF', 0)), (int, float)) else 0
            yo = int(_to_number(d.get('YBAYROFF', 0)) or 0) if isinstance(_to_number(d.get('YBAYROFF', 0)), (int, float)) else 0
            pattern = _pattern_shift(pattern, xo, yo)
    roworder = str(d.get('ROWORDER', '') or '').strip().upper()
    info['roworder'] = roworder
    info['header_pattern'] = pattern
    info['note'] = note
    if want_data:
        data = info['data']
        if roworder == 'BOTTOM-UP':
            # stored bottom row first: turn it top-down; the pattern in the
            # header describes the top-down picture, so it stays as it is
            data = data[::-1]
        if data.dtype.kind == 'f' and not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if data.dtype.kind in 'fi':
            # signed or float data can go below zero (e.g. calibrated subs);
            # lift it so nothing is cut off when it is put on the 0-65535 scale
            mn = float(np.min(data))
            if mn < 0:
                data = data.astype(np.float32) - np.float32(mn)
                info['note'] = 'Values below zero in this file; lifted by %.4g before measuring' % -mn
        info['data'] = np.ascontiguousarray(data)
        info['scale'] = _working_scale(info['data'])
    return info

# ---------------------------------------------------------------------------
#  ANALYSIS: preview, statistics, stars, trails, alignment
# ---------------------------------------------------------------------------

def _rel_median_mad(a):
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    return med, mad


def _sample(a, n=400000):
    step = max(1, int(math.sqrt(a.size / n)))
    return a[::step, ::step]


def check_pattern(cfa, pattern):
    """Compare the four CFA sub-grids. Green pixels should be the matching
    pair. If the header pattern puts green on the wrong diagonal (the usual
    row order mistake) return the row-flipped pattern and a note."""
    h, w = cfa.shape
    y0, y1 = h // 4 // 2 * 2, 3 * h // 4 // 2 * 2
    x0, x1 = w // 4 // 2 * 2, 3 * w // 4 // 2 * 2
    c = cfa[y0:y1, x0:x1]
    m = [float(np.median(_sample(c[dy::2, dx::2], 100000))) for dy in (0, 1) for dx in (0, 1)]
    if pattern[0] == 'G':
        gpair, other = (m[0], m[3]), (m[1], m[2])
    else:
        gpair, other = (m[1], m[2]), (m[0], m[3])
    mean = max(1.0, sum(m) / 4.0)
    dg = abs(gpair[0] - gpair[1]) / mean
    do = abs(other[0] - other[1]) / mean
    if dg > 0.01 and do < 0.3 * dg:
        newp = _pattern_flip_rows(pattern)
        return newp, ('Header says %s but the green pixels are on the other diagonal - using %s'
                      % (pattern, newp))
    return pattern, ''


def mtf(m, x):
    """Midtones transfer function (PixInsight style)."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)
    y = np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, y))
    return y


def stretch_params(stats_c, target=0.25, shadows=-2.8):
    """stats_c: list of (median, mad) per channel in 0..1. Returns (c0, m) per channel."""
    out = []
    for med, mad in stats_c:
        c0 = min(max(0.0, med + shadows * mad), 0.99)
        x = (med - c0) / (1.0 - c0) if med > c0 else 0.0
        if x <= 0:
            m = 0.5
        else:
            # midtones balance that sends x to target
            m = float(mtf(target, x))
        out.append((c0, m))
    return out


def build_lut(c0, m, white=1.0):
    x = np.arange(65536, dtype=np.float64) / 65535.0
    span = max(1e-6, white - c0)
    y = mtf(m, np.clip((x - c0) / span, 0, 1))
    return (y * 255.0 + 0.5).astype(np.uint8)


def linear_lut(lo, hi):
    x = np.arange(65536, dtype=np.float64) / 65535.0
    y = np.clip((x - lo) / max(1e-6, hi - lo), 0, 1)
    return (y * 255.0 + 0.5).astype(np.uint8)


def preview_stats(prev):
    """Median and MAD (0..1) per channel of a uint16 preview."""
    p = prev if prev.ndim == 3 else prev[:, :, None]
    out = []
    for ch in range(p.shape[2]):
        a = _sample(p[:, :, ch], 200000).astype(np.float32) / 65535.0
        out.append(_rel_median_mad(a))
    return out


def _clipped_sigma(d):
    """Sigma of d: MAD first, then the standard deviation inside +-3 sigma
    (gives smooth values even for whole-number data)."""
    med = float(np.median(d))
    s0 = float(np.median(np.abs(d - med))) * 1.4826
    if s0 <= 0:
        s0 = float(np.std(d)) or 1e-6
    k = d[np.abs(d - med) < 3.0 * s0]
    if k.size < 10:
        return s0
    return float(np.std(k)) / 0.9866


def _noise_diff(L):
    """Gaussian noise sigma from neighbour differences (ignores gradients)."""
    s = _sample(L, 1500000)
    d = (s[:, 1:].astype(np.float32) - s[:, :-1].astype(np.float32)).ravel()
    return _clipped_sigma(d) / math.sqrt(2.0)


def _background(L):
    s = _sample(L, 1000000).astype(np.float32).ravel()
    med = float(np.median(s))
    sig = float(np.median(np.abs(s - med))) * 1.4826 or 1.0
    k = s[np.abs(s - med) < 2.5 * sig]
    return float(np.mean(k)) if k.size else med


def _bg_map(L, cell=32):
    h, w = L.shape
    sh, sw = max(4, h // cell), max(4, w // cell)
    small = cv2.resize(L, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small.astype(np.float32), 5)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def adaptive_moments(cuts):
    """cuts: (N, S, S) background-subtracted star cut-outs.
    Gaussian weighted (adaptive) moments; returns fwhm, ecc, ok mask."""
    N, S, _ = cuts.shape
    r = (S - 1) / 2.0
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    xx = xx[None] - r
    yy = yy[None] - r
    cx = np.zeros(N, np.float32)
    cy = np.zeros(N, np.float32)
    sw2 = np.full(N, 4.0, np.float32)
    I = np.clip(cuts, 0, None)
    for _it in range(12):
        dx = xx - cx[:, None, None]
        dy = yy - cy[:, None, None]
        wgt = np.exp(-(dx * dx + dy * dy) / (2.0 * sw2[:, None, None])) * I
        s = wgt.sum(axis=(1, 2)) + 1e-12
        ncx = (wgt * xx).sum(axis=(1, 2)) / s
        ncy = (wgt * yy).sum(axis=(1, 2)) / s
        cx = np.clip(ncx, -r / 2, r / 2)
        cy = np.clip(ncy, -r / 2, r / 2)
        dx = xx - cx[:, None, None]
        dy = yy - cy[:, None, None]
        mxx = (wgt * dx * dx).sum(axis=(1, 2)) / s
        myy = (wgt * dy * dy).sum(axis=(1, 2)) / s
        mxy = (wgt * dx * dy).sum(axis=(1, 2)) / s
        w_used = sw2
        new = np.clip(mxx + myy, 0.15, (r * 0.6) ** 2)      # = 2 * mean moment
        sw2 = new
        if np.allclose(new, w_used, rtol=1e-3):
            break
    # undo the round Gaussian weight along each principal axis:
    # measured m = s2 * w2 / (s2 + w2)  ->  s2 = m * w2 / (w2 - m)
    a, b, c = mxx, myy, mxy
    tr = a + b
    disc = np.sqrt(np.maximum(0, (a - b) ** 2 / 4.0 + c * c))
    m1 = tr / 2 + disc
    m2 = np.maximum(tr / 2 - disc, 1e-6)
    w2 = w_used
    m1 = np.minimum(m1, w2 * 0.95)
    m2 = np.minimum(m2, w2 * 0.95)
    l1 = m1 * w2 / (w2 - m1)
    l2 = m2 * w2 / (w2 - m2)
    fwhm = 2.3548 * np.sqrt(np.maximum(1e-6, (l1 + l2) / 2.0))
    ecc = np.sqrt(np.clip(1.0 - l2 / np.maximum(l1, 1e-6), 0, 1))
    ok = (fwhm > 0.9) & (fwhm < S * 0.7) & np.isfinite(fwhm) & (np.abs(cx) < r / 2) & (np.abs(cy) < r / 2)
    return fwhm, ecc, ok


def _cutouts(L, ys, xs, r):
    h, w = L.shape
    S = 2 * r + 1
    keep = (ys >= r) & (ys < h - r) & (xs >= r) & (xs < w - r)
    ys, xs = ys[keep], xs[keep]
    if ys.size == 0:
        return np.zeros((0, S, S), np.float32), ys, xs
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    cuts = L[ys[:, None, None] + oy[None], xs[:, None, None] + ox[None]].astype(np.float32)
    border = np.concatenate([cuts[:, 0, :], cuts[:, -1, :], cuts[:, :, 0], cuts[:, :, -1]], axis=1)
    cuts -= np.median(border, axis=1)[:, None, None]
    return cuts, ys, xs


def hot_pixel_peaks(L, ys, xs, b, limit=6.0):
    """True for peaks that are a single bright pixel (hot pixel, cosmic ray).
    Compares the brightest full-resolution pixel with a Gaussian (sigma 1.2)
    weighted mean around it: stars give about 1-4, lone pixels 7-10."""
    h, w = L.shape
    r = 5
    out = np.zeros(ys.size, bool)
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    for s0 in range(0, ys.size, 4000):
        cy = np.clip(ys[s0:s0 + 4000] * b + b // 2, r + 1, h - r - 2)
        cx = np.clip(xs[s0:s0 + 4000] * b + b // 2, r + 1, w - r - 2)
        n = cy.size
        cut = L[cy[:, None, None] + oy, cx[:, None, None] + ox].astype(np.float32)
        ring = np.concatenate([cut[:, 0, :], cut[:, -1, :], cut[:, :, 0], cut[:, :, -1]], 1)
        cut -= np.median(ring, 1)[:, None, None]
        inner = cut[:, r - 1:r + 2, r - 1:r + 2].reshape(n, -1)
        k = np.argmax(inner, 1)
        py = k // 3 - 1
        px = k % 3 - 1
        pk = inner[np.arange(n), k]
        g = np.exp(-((oy[None] - py[:, None, None]) ** 2 + (ox[None] - px[:, None, None]) ** 2) / (2 * 1.44))
        g /= g.sum(axis=(1, 2), keepdims=True)
        smv = (cut * g).sum(axis=(1, 2))
        out[s0:s0 + 4000] = pk > limit * np.maximum(smv, 1e-6)
    return out


def find_stars(Ld, noise_d, thresh=5.0, max_measure=400, sn_floor=0.05):
    """Ld: detection image (float32). Returns residual, smooth noise and peaks."""
    bg = _bg_map(Ld)
    resid = Ld - bg
    sm = cv2.GaussianBlur(resid, (0, 0), 1.2)
    # floor of 0.05 (on the 0-65535 scale): real data is never quieter than
    # this, and a noise-free image must not turn every pixel into a 'star'
    sn = max(_rel_median_mad(_sample(sm, 400000))[1], sn_floor)
    dil = cv2.dilate(sm, np.ones((5, 5), np.uint8))
    pk = (sm >= dil) & (sm > thresh * sn)
    pk[:4, :] = False
    pk[-4:, :] = False
    pk[:, :4] = False
    pk[:, -4:] = False
    ys, xs = np.nonzero(pk)
    vals = sm[ys, xs]
    return resid, sm, sn, ys, xs, vals


def thin_structure(sm_s, sn, line_w):
    """Keep only thin structure (trails, stars); remove anything wider, such as
    nebula edges and dust lanes. A median filter wipes out thin lines but keeps
    broad steps and bands, so subtracting it leaves only the thin parts."""
    q = np.clip(sm_s / max(sn, 1e-9) * 6.0 + 40.0, 0, 255).astype(np.uint8)
    k = 2 * int(max(3, round(2.5 * line_w))) + 1
    med = cv2.medianBlur(q, min(k, 51))
    thin = (q.astype(np.float32) - med.astype(np.float32)) * (sn / 6.0)
    sn_t = _rel_median_mad(_sample(thin, 300000))[1] or sn
    return thin, sn_t


def set_trails(an, keep_idx):
    """Use only the listed trail candidates; star count gets back any peaks
    that sat on a dropped one."""
    keep_idx = list(keep_idx)
    an.trails = [an.trails_all[i] for i in keep_idx]
    an.stats['trails'] = len(an.trails)
    if an.quick:
        return
    if keep_idx:
        on = np.unique(np.concatenate([an.trail_peaks[i] for i in keep_idx]))
        an.stats['stars'] = int(an.nstars_raw - on.size)
    else:
        an.stats['stars'] = int(an.nstars_raw)


def _seg_match(a, b, tol):
    """True when two segments lie along the same line and overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    La = math.hypot(ax2 - ax1, ay2 - ay1)
    Lb = math.hypot(bx2 - bx1, by2 - by1)
    if La < 1 or Lb < 1:
        return False
    ang_a = math.atan2(ay2 - ay1, ax2 - ax1) % math.pi
    ang_b = math.atan2(by2 - by1, bx2 - bx1) % math.pi
    da = abs(ang_a - ang_b)
    da = min(da, math.pi - da)
    if da > math.radians(4):
        return False
    ux, uy = (ax2 - ax1) / La, (ay2 - ay1) / La
    nx, ny = -uy, ux
    d1 = abs((bx1 - ax1) * nx + (by1 - ay1) * ny)
    d2 = abs((bx2 - ax1) * nx + (by2 - ay1) * ny)
    if (d1 + d2) / 2 > tol:
        return False
    t1 = (bx1 - ax1) * ux + (by1 - ay1) * uy
    t2 = (bx2 - ax1) * ux + (by2 - ay1) * uy
    lo, hi = max(0.0, min(t1, t2)), min(La, max(t1, t2))
    return (hi - lo) > 0.3 * min(La, Lb)


def repeating_trails(frames_segs, diag):
    """frames_segs: {frame id: [segments in reference-frame pixels]}.
    A real trail crosses once; a 'trail' in the same place in two or more
    frames is sky structure. Returns {frame id: set of indices to drop}."""
    tol = max(8.0, diag * 0.004)
    drop = {k: set() for k in frames_segs}
    keys = list(frames_segs)
    for i, ka in enumerate(keys):
        for ia, sa in enumerate(frames_segs[ka]):
            for kb in keys[i + 1:]:
                for ib, sb in enumerate(frames_segs[kb]):
                    if _seg_match(sa, sb, tol) or _seg_match(sb, sa, tol):
                        drop[ka].add(ia)
                        drop[kb].add(ib)
    return drop


def detect_trails(resid_s, sm_s, sn, fwhm_d, bright_xy=None):
    """Find straight trails (satellites, aircraft) in a star-field residual.
    Works on a down-sized copy (about 1500 px). Returns list of segments
    (x1, y1, x2, y2) in that copy's pixels, plus the copy's scale."""
    h, w = sm_s.shape
    mask = (sm_s > 3.0 * sn).astype(np.uint8)
    n, lab, st, _cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return []
    bw = st[:, cv2.CC_STAT_WIDTH]
    bh = st[:, cv2.CC_STAT_HEIGHT]
    area = st[:, cv2.CC_STAT_AREA]
    big = np.maximum(bw, bh)
    small = np.minimum(bw, bh)
    star_size = max(6.0, fwhm_d * 5.0)
    # round, compact blobs are stars; elongated pieces may be trail parts
    elong = big >= np.maximum(2.2 * small, small + 4)
    keep = elong | (big > star_size * 4)
    keep[0] = False
    # very fat blobs (nebula, galaxy) are not trails
    fill = area / np.maximum(1, bw * bh)
    keep &= ~((small > star_size * 2.5) & (fill > 0.35))
    m2 = keep[lab].astype(np.uint8) * 255
    diag = math.hypot(w, h)
    lines = cv2.HoughLinesP(m2, 1, np.pi / 360, threshold=max(40, int(diag * 0.03)),
                            minLineLength=max(60, int(diag * 0.08)), maxLineGap=max(10, int(diag * 0.012)))
    if lines is None:
        return []
    segs = []
    # OpenCV 4 returns N x 1 x 4, OpenCV 5 returns N x 4 - accept either
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 1:
            continue
        # check the line is brighter than both sides along most of its length
        nx, ny = -(y2 - y1) / L, (x2 - x1) / L
        t = np.linspace(0.02, 0.98, max(20, int(L / 3)))
        px = x1 + (x2 - x1) * t
        py = y1 + (y2 - y1) * t
        off = max(5.0, fwhm_d * 3.0)

        def samp(dx, dy):
            xi = np.clip(np.round(px + dx).astype(int), 0, w - 1)
            yi = np.clip(np.round(py + dy).astype(int), 0, h - 1)
            return sm_s[yi, xi]
        on = samp(0, 0)
        side = np.minimum(samp(nx * off, ny * off), samp(-nx * off, -ny * off))
        frac = float(np.mean(on - side > 2.0 * sn))
        if frac < 0.55:
            continue
        # diffraction spikes: short lines running through a bright star
        if bright_xy is not None and len(bright_xy) and L < diag * 0.3:
            bx, by = bright_xy[:, 0], bright_xy[:, 1]
            d = np.abs(-(bx - x1) * (y2 - y1) / L + (by - y1) * (x2 - x1) / L)
            along = ((bx - x1) * (x2 - x1) + (by - y1) * (y2 - y1)) / L
            hit = (d < 4) & (along > -L * 0.2) & (along < L * 1.2)
            if np.any(hit):
                continue
        segs.append((float(x1), float(y1), float(x2), float(y2)))
    return _merge_segments(segs, diag)


def _radon_prep(resid, star_xy, star_r, size):
    """Downsize, remove broad structure, scale to noise units, blank stars."""
    h, w = resid.shape
    f = size / float(max(h, w))
    sw, sh = max(16, int(w * f)), max(16, int(h * f))
    img = cv2.resize(resid.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    img = img - cv2.GaussianBlur(img, (0, 0), max(3.0, size / 128.0))
    s = _rel_median_mad(_sample(img, 300000))[1] or 1e-6
    img /= s
    valid = np.ones((sh, sw), np.float32)
    if star_xy is not None and len(star_xy):
        # blank every star in one go: mark the centres, then grow them
        r = max(1, int(round(star_r * f)) + 1)
        cx = np.clip(np.round(star_xy[:, 0] * f).astype(int), 0, sw - 1)
        cy = np.clip(np.round(star_xy[:, 1] * f).astype(int), 0, sh - 1)
        centres = np.zeros((sh, sw), np.uint8)
        centres[cy, cx] = 1
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        valid[cv2.dilate(centres, disk) > 0] = 0.0
    valid[np.abs(img) > 8.0] = 0.0
    med = float(np.median(img[valid > 0])) if (valid > 0).any() else 0.0
    img = np.clip(img - med, -4, 4) * valid           # zero baseline, stars blanked
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    vblur = cv2.GaussianBlur(valid, (0, 0), 0.8)
    return img, vblur, f


def _running_median(a, win):
    from numpy.lib.stride_tricks import sliding_window_view
    p = win // 2
    ap = np.pad(a, (p, p), mode='edge')
    return np.median(sliding_window_view(ap, win), axis=1)


class _LineSums:
    """Sums of the image along every straight line at a given angle (a Radon
    transform done by sorting pixels into lines - no image rotation)."""

    def __init__(self, img, vblur):
        sh, sw = img.shape
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        self.cx, self.cy = sw / 2.0, sh / 2.0
        self.X = (xx - self.cx).ravel()
        self.Y = (yy - self.cy).ravel()
        self.I = img.ravel()
        self.V = vblur.ravel()
        self.n = int(math.ceil(math.hypot(sw, sh))) + 2
        self.half = self.n / 2.0
        self.minpix = 0.25 * min(sw, sh)

    def z(self, ang):
        """High-passed line-sum profile scored against its own spread: a trail
        is a narrow spike; broad structure, gradients and correlated noise are not."""
        t = math.radians(ang)
        b = (self.X * math.cos(t) + self.Y * math.sin(t) + self.half + 0.5).astype(np.int32)
        cs = np.bincount(b, self.I, self.n + 1)
        cn = np.bincount(b, self.V, self.n + 1)
        ok = cn > self.minpix
        z = np.zeros(self.n + 1, np.float32)
        if ok.sum() > 30:
            prof = np.where(ok, cs / np.sqrt(np.maximum(cn, 1.0)), 0.0)
            hp = prof - _running_median(prof, 15)
            v = hp[ok]
            s = float(np.median(np.abs(v - np.median(v)))) * 1.4826
            # never below the spread plain random noise gives (about 0.5 here):
            # very smooth images (synthetic, noise-reduced) must not inflate scores
            s = max(s, 0.5)
            z = np.where(ok, hp / s, 0.0).astype(np.float32)
        return z

    def line_points(self, ang, j, sw, sh):
        """Pixel positions along line j at this angle, inside the image."""
        t = math.radians(ang)
        r = j - self.half
        ux, uy = -math.sin(t), math.cos(t)
        L = self.n
        s = np.arange(-L / 2.0, L / 2.0, 1.0, dtype=np.float32)
        x = self.cx + r * math.cos(t) + s * ux
        y = self.cy + r * math.sin(t) + s * uy
        inside = (x >= 0) & (x < sw - 1) & (y >= 0) & (y < sh - 1)
        return x[inside], y[inside]


def detect_faint_trails(resid, star_xy, star_r, min_score=16.0, size=512):
    """Faint trails: add up the brightness along every straight line. A trail
    too faint to see pixel by pixel adds up along its length while noise does
    not. resid: background-removed luminance; star_xy: star positions (same
    pixels) to blank; star_r: their radius. Returns segments in resid's pixels."""
    h, w = resid.shape
    img, val, ff = _radon_prep(resid, star_xy, star_r, size)
    sh, sw = img.shape
    LS = _LineSums(img, val)
    cands = []
    for ang in np.arange(0.0, 180.0, 0.5):
        z = LS.z(ang)
        j = int(np.argmax(z))
        if z[j] >= 0.75 * min_score:
            cands.append((float(z[j]), float(ang), j))
    if not cands:
        return []
    cands.sort(key=lambda c: -c[0])
    picked = []
    for sc, ang, j in cands:
        if any(min(abs(ang - a2), 180 - abs(ang - a2)) < 3.0 and abs(j - j2) < 5 for _s, a2, j2 in picked):
            continue
        picked.append((sc, ang, j))
        if len(picked) >= 5:
            break
    segs = []
    for _sc, ang0, _j in picked:
        best = None
        for ang in np.arange(ang0 - 0.5, ang0 + 0.55, 0.1):
            z = LS.z(ang)
            j = int(np.argmax(z))
            if best is None or z[j] > best[0]:
                best = (float(z[j]), float(ang), j)
        sc, ang, j = best
        if sc < min_score:
            continue
        # a trail is a thin ridge; the edge of a nebula is a step (bright one
        # side, dark the other) - reject steps
        z = LS.z(ang)
        left = z[max(0, j - 9):max(0, j - 3)]
        right = z[j + 4:j + 10]
        if left.size and right.size:
            lm, rm = float(np.mean(left)), float(np.mean(right))
            if abs(lm - rm) > 0.3 * sc or min(lm, rm) < -0.25 * sc:
                continue
        # walk along the line to find where the trail actually is
        x, y = LS.line_points(ang, j, sw, sh)
        if x.size < 20:
            continue
        xi, yi = np.round(x).astype(int), np.round(y).astype(int)
        vals = img[yi, xi]
        vv = val[yi, xi]
        ker = np.ones(31, np.float32)
        run = np.convolve(vals, ker, mode='same') / (0.35 * np.sqrt(np.maximum(np.convolve(vv, ker, mode='same'), 1.0)))
        on = run > 2.5
        if not on.any():
            continue
        idx = np.nonzero(on)[0]
        a, b = idx[0], idx[-1]
        if (b - a) < 0.08 * math.hypot(sw, sh):
            continue
        if on[a:b + 1].mean() < 0.5:          # signal along most of it, not two blobs
            continue
        segs.append((x[a] / ff, y[a] / ff, x[b] / ff, y[b] / ff))
    return _merge_segments(segs, math.hypot(w, h))


def _merge_segments(segs, diag):
    """Join segments that lie on the same line."""
    out = []
    for s in sorted(segs, key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1])):
        x1, y1, x2, y2 = s
        ang = math.atan2(y2 - y1, x2 - x1) % math.pi
        joined = False
        for i, o in enumerate(out):
            ox1, oy1, ox2, oy2 = o
            oang = math.atan2(oy2 - oy1, ox2 - ox1) % math.pi
            da = abs(ang - oang)
            da = min(da, math.pi - da)
            if da > math.radians(3):
                continue
            L = math.hypot(ox2 - ox1, oy2 - oy1)
            nx, ny = -(oy2 - oy1) / L, (ox2 - ox1) / L
            dist = abs((x1 - ox1) * nx + (y1 - oy1) * ny) + abs((x2 - ox1) * nx + (y2 - oy1) * ny)
            if dist / 2 > max(6, diag * 0.004):
                continue
            # extend along the line
            ux, uy = (ox2 - ox1) / L, (oy2 - oy1) / L
            ts = [0, L, (x1 - ox1) * ux + (y1 - oy1) * uy, (x2 - ox1) * ux + (y2 - oy1) * uy]
            t0, t1 = min(ts), max(ts)
            out[i] = (ox1 + ux * t0, oy1 + uy * t0, ox1 + ux * t1, oy1 + uy * t1)
            joined = True
            break
        if not joined:
            out.append(s)
    return out


def align_thumb(Ld, sn_hint=None, size=1024):
    """High-passed, clipped small image for phase correlation (uint8)."""
    h, w = Ld.shape
    f = size / float(max(h, w))
    # a 3x3 median first, so hot pixels (which sit still) cannot pull the match to zero
    src = cv2.medianBlur(np.ascontiguousarray(Ld, dtype=np.float32), 3)
    small = cv2.resize(src, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    small = small - cv2.GaussianBlur(small, (0, 0), 8)
    med, mad = _rel_median_mad(_sample(small, 100000))
    mad = mad or 1e-6
    small = np.clip((small - med) / mad, -2, 40)
    small = np.log1p(small + 2)
    return np.clip(small * 64.0, 0, 255).astype(np.uint8), f


def phase_shift(ref, img, flip=True):
    """Shift (dx, dy) of img relative to ref, response; checks a 180 turn too."""
    if ref is None or img is None or ref.shape != img.shape:
        return 0.0, 0.0, 0.0, False
    ref = ref.astype(np.float32)
    img = img.astype(np.float32)
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ref, img, win)
    if not flip:
        return float(dx), float(dy), float(resp), False
    rot = np.ascontiguousarray(img[::-1, ::-1])
    (rdx, rdy), rresp = cv2.phaseCorrelate(ref, rot, win)
    if rresp > resp * 1.5 and rresp > 0.05:
        return float(rdx), float(rdy), float(rresp), True
    return float(dx), float(dy), float(resp), False


def header_values(header):
    d = hdr_dict(header)

    def g(*keys):
        for k in keys:
            if k in d and d[k] not in (None, ''):
                return d[k]
        return None

    def num(v):
        v = _to_number(v)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    out = {}
    out['date'] = str(g('DATE-OBS', 'DATE-LOC', 'DATE', 'DateTimeOriginal', 'DateTime',
                         '[Observation:Time:Start]') or '')
    out['exposure'] = num(g('EXPTIME', 'EXPOSURE', 'EXPOINUS', 'ExposureTime', '[Instrument:ExposureTime]'))
    if out['exposure'] is None:
        et = g('ExposureTime')
        if et is not None:
            try:
                out['exposure'] = float(str(et).split()[0]) if '/' not in str(et) else \
                    float(str(et).split('/')[0]) / float(str(et).split('/')[1])
            except Exception:
                pass
    out['filter'] = str(g('FILTER', 'FILTER1', '[Instrument:Filter:Name]') or '').strip()
    out['temp'] = num(g('CCD-TEMP', 'SET-TEMP', 'CCDTEMP', 'TEMPERAT', '[Instrument:Sensor:Temperature]',
                        '[Instrument:Sensor:TargetTemperature]'))
    out['gain'] = num(g('GAIN', 'EGAIN_SET', 'ISOSPEED', 'ISO', 'ISOSpeedRatings', 'PhotographicSensitivity',
                        '[Instrument:Camera:Gain]', '[Instrument:Camera:ISOSpeed]'))
    out['alt'] = num(g('OBJCTALT', 'CENTALT', 'ALTITUDE', 'ALT-OBJ'))
    out['object'] = str(g('OBJECT', '[Observation:Object:Name]') or '').strip()
    out['frametype'] = str(g('IMAGETYP', 'FRAMETYP', 'FRAME') or '').strip()
    xp = num(g('XPIXSZ', 'PIXSIZE1', 'PIXSIZE'))
    fl = num(g('FOCALLEN', 'FOCAL'))
    if not xp:
        xp = num(g('[Instrument:Sensor:XPixelSize]'))       # micrometres
    if not fl:
        flm = num(g('[Instrument:Telescope:FocalLength]'))  # metres in XISF
        if flm:
            fl = flm * 1000.0 if flm < 50 else flm
    xb = num(g('XBINNING', 'BINX', '[Instrument:Camera:XBinning]')) or 1.0
    scale = None
    if xp and fl:
        scale = 206.265 * xp / fl          # XPIXSZ already includes binning in most writers
    ps = num(g('PIXSCALE', 'SCALE', 'SECPIX'))
    if ps:
        scale = ps
    elif not scale:
        cd = num(g('CDELT1', 'CD1_1'))
        if cd:
            scale = abs(cd) * 3600.0
    out['pixscale'] = scale
    out['binning'] = xb
    return out


def _dt_of(datestr):
    s = datestr.strip().replace('T', ' ')
    s = s.replace(':', '-', 2) if re.match(r'^\d{4}:\d{2}:\d{2}', s) else s
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            frac = ''
            if '.' in s and fmt.endswith('%f'):
                head, frac = s.split('.', 1)
                frac = re.sub(r'\D.*$', '', frac)[:6]
                return datetime.datetime.strptime(head + '.' + (frac or '0'), fmt)
            return datetime.datetime.strptime(s[:19], fmt)
        except Exception:
            continue
    return None


class Analysis:
    """Everything worked out for one frame."""

    def __init__(self):
        self.ok = False
        self.error = ''
        self.kind = 'mono'
        self.pattern = ''
        self.pattern_note = ''
        self.width = 0
        self.height = 0
        self.header = []
        self.hv = {}
        self.preview = None         # uint16, top-down
        self.prev_scale = 1.0       # full pixels per preview pixel
        self.athumb = None          # float16 thumb for alignment
        self.athumb_f = 1.0
        self.stats = {}
        self.trails = []            # segments in full-res pixels
        self.trails_all = []        # every candidate before the repeat check
        self.trail_peaks = []
        self.nstars_raw = 0
        self.shift = (0.0, 0.0)     # full-res pixels, vs reference
        self.flipped = False
        self.align_resp = 0.0
        self.when = None
        self.quick = False          # SER / video frame: only looked at, no star measuring
        self.data_note = ''


def to_rgb_or_mono(info, debayer_mode='auto', check=True):
    """Returns (full image uint16 or float, kind, pattern, note)."""
    data = info['data']
    scale = info['scale']
    pattern = ''
    note = ''
    if data.ndim == 2:
        if debayer_mode == 'off':
            pattern = ''
        elif debayer_mode in PATTERNS:
            pattern = debayer_mode
        else:
            pattern = info.get('header_pattern', '')
            if pattern and check:
                pattern, note = check_pattern(data, pattern)
    if data.dtype == np.uint16 and scale == 1.0:
        work = data
    else:
        work = np.clip(data.astype(np.float32) * scale, 0, 65535).astype(np.uint16)
    if data.ndim == 2 and pattern:
        rgb = debayer(work, pattern)
        return rgb, 'cfa', pattern, note
    if data.ndim == 3:
        return work, 'rgb', '', ''
    return work, 'mono', '', ''


def analyse(src, opts):
    """Read a frame and measure everything. opts: dict of settings."""
    an = Analysis()
    t0 = time.time()
    try:
        info = read_source(src, True)
    except Exception as e:
        an.error = str(e) or e.__class__.__name__
        if not isinstance(e, ReadError):
            LOG.exception('Reading %s failed', fname(getattr(src, 'path', src)))
        return an
    an.header = info['header']
    an.data_note = info.get('note', '')
    an.hv = header_values(an.header)
    an.when = _dt_of(an.hv.get('date') or '')
    img, kind, pattern, note = to_rgb_or_mono(info, opts.get('debayer', 'auto'),
                                             opts.get('check_pattern', True))
    an.kind, an.pattern, an.pattern_note = kind, pattern, note
    h, w = img.shape[:2]
    an.width, an.height = w, h
    # preview
    pmax = int(opts.get('preview_max', 1600))
    f = min(1.0, pmax / float(max(h, w)))
    if f < 1.0:
        pw, ph = max(1, int(round(w * f))), max(1, int(round(h * f)))
        an.preview = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
    else:
        an.preview = img.copy()
    an.prev_scale = w / float(an.preview.shape[1])
    # luminance at full resolution
    if img.ndim == 3:
        L = img.astype(np.float32).mean(axis=2)
    else:
        L = img.astype(np.float32)
    del img
    # saturation on the raw values
    raw = info['data']
    sat_level = 65535.0 * 0.98
    rs = _sample(raw if raw.ndim == 2 else raw[:, :, 1], 2000000)
    if raw.dtype.kind == 'f' or info['scale'] != 1.0:
        rs = rs.astype(np.float32) * info['scale']
    sat = float(np.mean(rs >= sat_level)) * 100.0
    if getattr(src, 'ser', None) is not None and not opts.get('video_stars', False):
        return _quick_measure(an, L, sat, t0)
    # detection image: 2x2 binned if large
    b = 2 if max(h, w) > 3000 else 1
    if b == 2:
        Ld = cv2.resize(L, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    else:
        Ld = L
    bgval = _background(L)
    noise = _noise_diff(L)
    sn_floor = 0.05
    if info.get('lossy'):
        # JPEG flattens the sky into 8x8 blocks and rings around stars; those
        # edges look like faint stars. Soften them and do not let the noise
        # estimate drop below about one 8-bit step.
        Ld = cv2.GaussianBlur(Ld, (0, 0), 1.0)
        sn_floor = 0.5 * float(info.get('scale', 257.0))
    resid, sm, sn, ys, xs, vals = find_stars(Ld, noise, opts.get('star_sigma', 5.0), sn_floor=sn_floor)
    if ys.size:
        keep = ~hot_pixel_peaks(L, ys, xs, b)
        ys, xs, vals = ys[keep], xs[keep], vals[keep]
    nstars = int(ys.size)
    # measure on full resolution cut-outs
    fwhm_med = ecc_med = None
    fwhm_d = 2.0
    if nstars >= 3:
        order = np.argsort(-vals)
        peak_full = np.clip(ys * b, 0, h - 1), np.clip(xs * b, 0, w - 1)
        pv = L[peak_full[0], peak_full[1]]
        unsat = pv < sat_level * 0.9
        cand = order[unsat[order]]
        # skip the very brightest (often bloated) and take the next ones
        cand = cand[min(len(cand) // 20, 10):][:int(opts.get('max_measure', 400))]
        if cand.size >= 3:
            cy, cx = ys[cand] * b + (b // 2), xs[cand] * b + (b // 2)
            # refine to the brightest full-res pixel nearby
            r0 = 10
            cuts, cy, cx = _cutouts(L, cy, cx, r0)
            if cuts.shape[0]:
                fw, ec, ok = adaptive_moments(cuts)
                if ok.sum() >= 3:
                    f0 = float(np.median(fw[ok]))
                    r1 = int(np.clip(round(f0 * 2.5), 6, 40))
                    cuts, cy2, cx2 = _cutouts(L, cy, cx, r1)
                    fw, ec, ok = adaptive_moments(cuts)
                    if ok.sum() >= 3:
                        fwhm_med = float(np.median(fw[ok]))
                        ecc_med = float(np.median(ec[ok]))
                        fwhm_d = fwhm_med / b
    # signal: mean absolute deviation above the background over the noise
    rs2 = _sample(resid, 500000)
    meandev = float(np.mean(np.abs(rs2)))
    snr = meandev / max(1e-6, _noise_diff(Ld))
    st = {'background': bgval, 'noise': noise, 'snr': snr, 'stars': nstars,
          'fwhm': fwhm_med, 'ecc': ecc_med, 'saturated': sat}
    ps = an.hv.get('pixscale')
    st['fwhm_arcsec'] = fwhm_med * ps if (fwhm_med and ps) else None
    # trails on a ~1500 px copy
    trails = []
    if opts.get('trails', True):
        ts = min(1.0, 1500.0 / max(Ld.shape))
        if ts < 1.0:
            sm_s = cv2.resize(sm, (int(Ld.shape[1] * ts), int(Ld.shape[0] * ts)), interpolation=cv2.INTER_AREA)
            sn_s = _rel_median_mad(_sample(sm_s, 300000))[1] or sn
        else:
            sm_s, sn_s = sm, sn
        thin, sn_t = thin_structure(sm_s, sn_s, max(1.0, fwhm_d * ts))
        bright = None
        if nstars:
            top = np.argsort(-vals)[:60]
            bright = np.stack([xs[top] * ts, ys[top] * ts], axis=1)
        segs = detect_trails(None, thin, sn_t, max(1.0, fwhm_d * ts), bright)
        k = b / ts
        trails = [(x1 * k, y1 * k, x2 * k, y2 * k) for (x1, y1, x2, y2) in segs]
        # faint trails, found by adding brightness along lines
        try:
            sxy = np.stack([xs, ys], axis=1).astype(np.float32) if nstars else None
            faint = detect_faint_trails(resid, sxy, max(2.0, fwhm_d * 1.5))
            faint = [(x1 * b, y1 * b, x2 * b, y2 * b) for (x1, y1, x2, y2) in faint]
            trails = _merge_segments(trails + faint, math.hypot(w, h))
        except Exception:
            LOG.exception('Faint trail search failed')
    # star peaks lying on each trail (so they can be given back if a
    # "trail" later turns out to be sky structure seen in other frames)
    peaks_on = []
    if trails and nstars:
        px, py = xs.astype(np.float64) * b, ys.astype(np.float64) * b
        tol = max(4.0, (fwhm_med or 3.0) * 2.0)
        for x1, y1, x2, y2 in trails:
            L = max(1e-6, math.hypot(x2 - x1, y2 - y1))
            d = np.abs(-(px - x1) * (y2 - y1) / L + (py - y1) * (x2 - x1) / L)
            t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / L
            peaks_on.append(np.nonzero((d < tol) & (t > -tol) & (t < L + tol))[0].astype(np.int32))
    else:
        peaks_on = [np.zeros(0, np.int32) for _t in trails]
    an.trails_all = list(trails)
    an.trail_peaks = peaks_on
    an.nstars_raw = nstars
    st['stars'] = nstars
    an.stats = st
    set_trails(an, list(range(len(trails))))
    th, tf = align_thumb(Ld)
    an.athumb = th
    an.athumb_f = tf / b       # thumb px per full px
    an.ok = True
    an.stats['time'] = time.time() - t0
    return an


def _quick_measure(an, L, sat, t0):
    """SER and video frames: there can be thousands, and stars mean little on a
    planet or the Moon, so only the brightness is measured - fast enough to
    start playing almost at once."""
    # every 4th row is plenty here (whole rows, so neighbour differences still work)
    rows = L[::4] if L.shape[0] >= 64 else L
    bgval = _background(_sample(rows, 120000))
    noise = _noise_diff(rows[:, :min(rows.shape[1], 2048)])
    sig = float(np.mean(np.abs(_sample(rows, 120000).astype(np.float32) - bgval)))
    an.quick = True
    an.stats = {'background': bgval, 'noise': noise, 'snr': sig / max(1e-6, noise), 'stars': None,
                'fwhm': None, 'ecc': None, 'saturated': sat, 'fwhm_arcsec': None, 'trails': 0}
    an.trails_all, an.trail_peaks, an.trails, an.nstars_raw = [], [], [], 0
    th, tf = align_thumb(L, size=min(1024, max(L.shape)))
    an.athumb = th
    an.athumb_f = tf
    an.ok = True
    an.stats['time'] = time.time() - t0
    return an


def full_image_for_display(src, opts):
    """Full resolution image (uint16) for zoomed viewing."""
    info = read_source(src, True)
    img, kind, pattern, note = to_rgb_or_mono(info, opts.get('debayer', 'auto'),
                                             opts.get('check_pattern', True))
    return img


# ------------------------------- auto reject -------------------------------

REJECT_RULES = [
    # key, label, direction (+1 high is bad, -1 low is bad), default sigma, minimum relative change, on
    ('fwhm', 'FWHM', +1, 2.0, 0.10, True),
    ('ecc', 'Eccentricity', +1, 2.0, 0.08, True),
    ('stars', 'Star count', -1, 2.0, 0.25, True),
    ('background', 'Background', +1, 2.5, 0.15, True),
    ('snr', 'SNR', -1, 2.5, 0.20, False),
]


def auto_reject(values, rules, reject_trails=False, trails=None):
    """values: {key: list of numbers or None}. rules: {key: (on, sigma)}.
    Returns (reasons, keys): per frame, a list of reason strings (empty = keep)
    and the set of measurements that caused them."""
    n = len(next(iter(values.values()))) if values else 0
    reasons = [[] for _ in range(n)]
    keys = [set() for _ in range(n)]
    for key, label, direction, _ds, minrel, _on in REJECT_RULES:
        on, sigma = rules.get(key, (False, 2.0))
        if not on:
            continue
        arr = np.array([np.nan if v is None else float(v) for v in values.get(key, [])], dtype=np.float64)
        good = np.isfinite(arr)
        if good.sum() < 4:
            continue
        med = float(np.median(arr[good]))
        mad = float(np.median(np.abs(arr[good] - med))) * 1.4826
        if key == 'ecc':
            tol = max(sigma * mad, minrel)          # eccentricity: absolute
        else:
            tol = max(sigma * mad, minrel * abs(med))
        for i in range(n):
            if not good[i]:
                continue
            dev = (arr[i] - med) * direction
            if dev > tol:
                keys[i].add(key)
                if key == 'ecc':
                    reasons[i].append('%s %.2f (session %.2f)' % (label, arr[i], med))
                elif key in ('stars', 'background'):
                    reasons[i].append('%s %d (session %d)' % (label, arr[i], med))
                else:
                    reasons[i].append('%s %.3g (session %.3g)' % (label, arr[i], med))
    if reject_trails and trails is not None:
        for i, t in enumerate(trails):
            if t:
                reasons[i].append('%d trail%s' % (t, 's' if t > 1 else ''))
                keys[i].add('trails')
    return reasons, keys


def reason_family(key):
    """FWHM in pixels and in arcseconds are the same measurement."""
    return 'fwhm' if key in ('fwhm', 'fwhm_arcsec') else key


def best_percent_cut(values, kind, drop_pct):
    """values: {frame id: number or None}. Keeps the best (100 - drop_pct)% for
    this measurement and returns (set of dropped ids, cut-off value).
    kind 'max': lower is better (FWHM, eccentricity, background);
    kind 'min': higher is better (star count, SNR). Unmeasured frames are dropped."""
    have = [(k, float(v)) for k, v in values.items() if v is not None and math.isfinite(float(v))]
    missing = {k for k, v in values.items() if v is None or not math.isfinite(float(v))}
    if not have or drop_pct <= 0:
        return set(), None
    have.sort(key=lambda kv: kv[1], reverse=(kind == 'min'))     # best first
    keep_n = max(1, int(math.ceil(len(have) * (100.0 - drop_pct) / 100.0 - 1e-9)))
    dropped = {k for k, _v in have[keep_n:]} | missing
    cut = have[keep_n - 1][1]
    return dropped, cut


def load_preview(src, opts):
    """Quick reload of just the preview (after it has been dropped from memory)."""
    info = read_source(src, True)
    data = info['data']
    pmax = int(opts.get('preview_max', 1600))
    pattern = ''
    if data.ndim == 2:
        mode = opts.get('debayer', 'auto')
        if mode in PATTERNS:
            pattern = mode
        elif mode != 'off':
            pattern = info.get('header_pattern', '')
            if pattern and opts.get('check_pattern', True):
                pattern, _n = check_pattern(data, pattern)
    h, w = data.shape[:2]
    scale = info['scale']
    if pattern and pmax <= max(h, w) // 2:
        # half-size colour straight from the 2x2 cells is plenty for a preview
        sp = superpixel(data, pattern) * scale
        img = np.clip(sp, 0, 65535).astype(np.uint16)
    else:
        img, _k, _p, _n = to_rgb_or_mono(info, opts.get('debayer', 'auto'), opts.get('check_pattern', True))
    f = min(1.0, pmax / float(max(img.shape[:2])))
    if f < 1.0:
        pw = max(1, int(round(w * pmax / float(max(h, w)))))
        ph = max(1, int(round(h * pmax / float(max(h, w)))))
        img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
    return img, w / float(img.shape[1])


def _digits(a, b):
    """Fewest decimals (2 to 4) that show two values as different."""
    for nd in (2, 3, 4):
        if '%.*f' % (nd, a) != '%.*f' % (nd, b):
            return nd
    return 4


def limit_reasons(v, key, vkey, kind, lim):
    """Reasons a frame fails one manual limit. kind 'max': fail when the value is
    ABOVE the limit (FWHM, eccentricity, background). kind 'min': fail when it is
    BELOW the limit (star count, SNR)."""
    names = {'fwhm': 'FWHM', 'ecc': 'Eccentricity', 'stars': 'Stars', 'background': 'Background', 'snr': 'SNR'}
    name = names.get(key, key)
    if v is None:
        return ['%s not measured' % name]
    v = float(v)
    if kind == 'max':
        if v > lim + 1e-9:
            if key == 'background':
                return ['%s %.0f above %.0f' % (name, v, lim)]
            unit = '"' if vkey == 'fwhm_arcsec' else (' px' if key == 'fwhm' else '')
            nd = _digits(v, lim)
            return ['%s %.*f%s above %.*f%s' % (name, nd, v, unit, nd, lim, unit)]
        return []
    if key == 'stars':
        need = int(math.ceil(lim - 1e-9))
        return ['%s %d below %d' % (name, v, need)] if v < need else []
    nd = _digits(v, lim)
    return ['%s %.*f below %.*f' % (name, nd, v, nd, lim)] if v < lim - 1e-9 else []


def suggested_limit(key, ys, sigma):
    """'Odd ones out' limit for one measurement: the typical value (median)
    plus sigma times the usual spread (robust), or the minimum gap if that is
    bigger. Same rule as the automatic tests used."""
    rule = [r for r in REJECT_RULES if r[0] == key][0]
    direction, minrel = rule[2], rule[4]
    a = np.asarray(ys, dtype=np.float64)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    gap = minrel if key == 'ecc' else minrel * abs(med)
    return med + direction * max(sigma * mad, gap)
# ---------------------------------------------------------------------------
#  QT APP
# ---------------------------------------------------------------------------
from PySide6.QtCore import (Qt, QObject, QRunnable, QThreadPool, Signal, QTimer, QPointF, QRectF,
                            QSettings, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QSize,
                            QItemSelectionModel, QRect, QUrl, qInstallMessageHandler, QtMsgType)
from PySide6.QtGui import (QImage, QPainter, QColor, QPen, QBrush, QFont, QKeySequence, QAction,
                           QPalette, QIcon, QPixmap, QShortcut, QFontMetrics, QDesktopServices)
from PySide6.QtWidgets import (QApplication, QProxyStyle, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QComboBox, QCheckBox, QSpinBox,
                               QDoubleSpinBox, QFileDialog, QTableView, QHeaderView, QAbstractItemView,
                               QSplitter, QTabWidget, QLineEdit, QTableWidget, QTableWidgetItem,
                               QMessageBox, QProgressBar, QStatusBar, QDialog, QDialogButtonBox,
                               QFormLayout, QGroupBox, QRadioButton, QGridLayout, QToolButton, QFrame,
                               QStyle, QTextBrowser, QToolBar, QMenu)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


class LRU:
    """Byte-budget cache."""

    def __init__(self, budget):
        self.budget = budget
        self.used = 0
        self.d = collections.OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def _size(v):
        if isinstance(v, np.ndarray):
            return v.nbytes
        if isinstance(v, tuple):
            return sum(LRU._size(x) for x in v)
        if isinstance(v, QImage):
            return v.sizeInBytes()
        return 64

    def get(self, k):
        with self.lock:
            v = self.d.get(k)
            if v is not None:
                self.d.move_to_end(k)
            return v

    def put(self, k, v, pinned=()):
        s = self._size(v)
        with self.lock:
            if k in self.d:
                self.used -= self._size(self.d.pop(k))
            self.d[k] = v
            self.used += s
            for key in list(self.d.keys()):
                if self.used <= self.budget:
                    break
                if key == k or key in pinned:
                    continue
                self.used -= self._size(self.d.pop(key))

    def drop(self, pred):
        with self.lock:
            for key in [k for k in self.d if pred(k)]:
                self.used -= self._size(self.d.pop(key))

    def clear(self):
        with self.lock:
            self.d.clear()
            self.used = 0

    def __contains__(self, k):
        with self.lock:
            return k in self.d


class Frame:
    def __init__(self, src, uid):
        self.src = src
        self.uid = uid
        self.an = None
        self.status = 'waiting'      # waiting, working, ok, error
        self.mark = ''               # '', 'good', 'bad'
        self.reasons = []
        self.reason_keys = set()     # which measurements caused the flag
        self.shift = None            # (dx, dy, flipped) in full pixels
        self.fine_shift = None
        self.pstats = None

    @property
    def name(self):
        return self.src.name

    def stat(self, key):
        if self.an is None or not self.an.ok:
            return None
        if key in self.an.stats:
            return self.an.stats[key]
        return self.an.hv.get(key)


class Signals(QObject):
    analysed = Signal(int, int, object)          # generation, uid, Analysis
    preview = Signal(int, int, object, float)    # generation, uid, preview, prev_scale
    full = Signal(int, int, object)              # generation, uid, full image
    failed = Signal(int, int, str)


def log_analysis(src, an, secs):
    try:
        name = fname(src.path) + ((' [frame %d]' % (src.index + 1)) if src.ser is not None else '')
        try:
            nb = os.path.getsize(src.path)
            size = ('%.1f MB' % (nb / 1e6)) if nb >= 1e6 else ('%d KB' % max(1, round(nb / 1e3)))
        except Exception:
            size = '? MB'
        if not an.ok:
            LOG.warning('FAILED  %s (%s): %s', name, size, an.error)
            return
        if an.quick:
            return          # thousands of video frames would flood the log; the summary covers them
        st = an.stats

        def f(k, fmt):
            v = st.get(k)
            return fmt % v if isinstance(v, (int, float)) else '-'
        hv = an.hv or {}
        extra = '; '.join(x for x in (
            ('filter %s' % hv['filter']) if hv.get('filter') else '',
            ('exp %gs' % hv['exposure']) if hv.get('exposure') else '',
            an.pattern_note, getattr(an, 'data_note', '')) if x)
        LOG.info('ok  %s (%s, %dx%d, %s%s) %.2fs | stars %s fwhm %s ecc %s bg %s noise %s snr %s sat %s trails %s%s',
                 name, size, an.width, an.height, an.kind, (' ' + an.pattern) if an.pattern else '', secs,
                 f('stars', '%d'), f('fwhm', '%.2f'), f('ecc', '%.2f'), f('background', '%.0f'),
                 f('noise', '%.1f'), f('snr', '%.2f'), f('saturated', '%.2f'), f('trails', '%d'),
                 (' | ' + extra) if extra else '')
    except Exception:
        LOG.exception('Could not log the result for a frame')


class AnalyseTask(QRunnable):
    def __init__(self, sig, gen, uid, src, opts, alive):
        super().__init__()
        self.sig, self.gen, self.uid, self.src, self.opts, self.alive = sig, gen, uid, src, opts, alive
        self.setAutoDelete(True)

    def run(self):
        if not self.alive(self.gen):
            return
        t0 = time.time()
        try:
            an = analyse(self.src, self.opts)
        except Exception as e:
            an = Analysis()
            an.error = '%s: %s' % (e.__class__.__name__, e)
            LOG.exception('Measuring %s crashed', fname(self.src.path))
        if an.ok and an.preview is not None:
            try:
                an.pstats = preview_stats(an.preview)   # here, not on the window's thread
            except Exception:
                an.pstats = None
        log_analysis(self.src, an, time.time() - t0)
        if self.alive(self.gen):
            try:
                self.sig.analysed.emit(self.gen, self.uid, an)
            except RuntimeError:
                pass


class PreviewTask(QRunnable):
    def __init__(self, sig, gen, uid, src, opts, alive, busy):
        super().__init__()
        self.sig, self.gen, self.uid, self.src, self.opts, self.alive, self.busy = \
            sig, gen, uid, src, opts, alive, busy
        self.setAutoDelete(True)

    def run(self):
        try:
            if not self.alive(self.gen):
                return
            pv, ps = load_preview(self.src, self.opts)
            if self.alive(self.gen):
                self.sig.preview.emit(self.gen, self.uid, pv, ps)
        except Exception as e:
            LOG.exception('%s for %s failed', self.__class__.__name__.replace('Task', ' image'), fname(self.src.path))
            try:
                if self.alive(self.gen):
                    self.sig.failed.emit(self.gen, self.uid, str(e))
            except RuntimeError:
                pass
        finally:
            self.busy.discard(('p', self.uid))


class FullTask(QRunnable):
    def __init__(self, sig, gen, uid, src, opts, alive, busy):
        super().__init__()
        self.sig, self.gen, self.uid, self.src, self.opts, self.alive, self.busy = \
            sig, gen, uid, src, opts, alive, busy
        self.setAutoDelete(True)

    def run(self):
        try:
            if not self.alive(self.gen):
                return
            img = full_image_for_display(self.src, self.opts)
            if self.alive(self.gen):
                self.sig.full.emit(self.gen, self.uid, img)
        except Exception as e:
            LOG.exception('%s for %s failed', self.__class__.__name__.replace('Task', ' image'), fname(self.src.path))
            try:
                if self.alive(self.gen):
                    self.sig.failed.emit(self.gen, self.uid, str(e))
            except RuntimeError:
                pass
        finally:
            self.busy.discard(('f', self.uid))

# ---------------------------------------------------------------------------
#  THEMES  (Dark, Light, Night red)
# ---------------------------------------------------------------------------
THEMES = {
    'Dark': {
        'window': (44, 44, 52), 'base': (25, 25, 30), 'alt': (33, 33, 40), 'text': (228, 228, 235),
        'button': (54, 54, 64), 'tooltip': (55, 55, 66), 'highlight': (70, 100, 170), 'hltext': (255, 255, 255),
        'disabled': (125, 125, 133), 'bright': (255, 80, 80), 'mid': (90, 90, 104),
        'icon': (228, 228, 235),
        'view_bg': (18, 18, 22), 'view_msg': (150, 150, 160), 'trail': (255, 60, 60, 200), 'loupe': (230, 200, 90),
        'badge_good': (40, 170, 80), 'badge_bad': (210, 50, 50), 'badge_text': (255, 255, 255),
        'label_box': (0, 0, 0, 130), 'label': (235, 235, 240),
        'row_good': (40, 150, 70), 'row_bad': (180, 45, 45), 'row_flag': (160, 98, 22),
        'err_text': (240, 110, 110), 'bad_text': (150, 150, 150),
        'graph_bg': (26, 26, 32), 'graph_frame': (70, 70, 82), 'graph_text': (170, 170, 185),
        'graph_median': (120, 120, 140), 'graph_auto': (255, 150, 45), 'graph_manual': (255, 70, 70),
        'graph_line': (90, 120, 170), 'pt_good': (70, 200, 100), 'pt_bad': (230, 70, 70),
        'pt_flag': (255, 165, 30), 'pt_plain': (140, 170, 235), 'pt_current': (255, 255, 255),
        'hover_bg': (0, 0, 0, 200), 'hover_text': (240, 240, 240),
        'note_warn': '#e0b060', 'note_dim': '#9a9aaa', 'header_hl': (110, 90, 20),
    },
    'Light': {
        'window': (238, 238, 242), 'base': (255, 255, 255), 'alt': (245, 245, 249), 'text': (22, 22, 28),
        'button': (226, 226, 232), 'tooltip': (255, 255, 230), 'highlight': (60, 110, 200), 'hltext': (255, 255, 255),
        'disabled': (150, 150, 156), 'bright': (200, 0, 0), 'mid': (190, 190, 198),
        'icon': (40, 40, 50),
        'view_bg': (205, 205, 212), 'view_msg': (70, 70, 80), 'trail': (230, 30, 30, 220), 'loupe': (200, 140, 0),
        'badge_good': (30, 150, 60), 'badge_bad': (200, 40, 40), 'badge_text': (255, 255, 255),
        'label_box': (0, 0, 0, 130), 'label': (240, 240, 245),
        'row_good': (160, 225, 170), 'row_bad': (240, 160, 160), 'row_flag': (252, 218, 160),
        'err_text': (190, 30, 30), 'bad_text': (140, 140, 140),
        'graph_bg': (255, 255, 255), 'graph_frame': (190, 190, 200), 'graph_text': (60, 60, 72),
        'graph_median': (140, 140, 152), 'graph_auto': (210, 110, 20), 'graph_manual': (215, 30, 30),
        'graph_line': (90, 120, 190), 'pt_good': (30, 150, 60), 'pt_bad': (205, 40, 40),
        'pt_flag': (225, 135, 20), 'pt_plain': (60, 95, 170), 'pt_current': (0, 0, 0),
        'hover_bg': (255, 255, 255, 235), 'hover_text': (20, 20, 25),
        'note_warn': '#a05a00', 'note_dim': '#666672', 'header_hl': (250, 228, 150),
    },
    'Night (red)': {
        'window': (24, 0, 0), 'base': (10, 0, 0), 'alt': (20, 2, 2), 'text': (205, 40, 40),
        'button': (40, 4, 4), 'tooltip': (40, 4, 4), 'highlight': (105, 10, 10), 'hltext': (255, 95, 95),
        'disabled': (105, 22, 22), 'bright': (255, 60, 60), 'mid': (80, 12, 12),
        'icon': (205, 40, 40),
        'view_bg': (6, 0, 0), 'view_msg': (170, 30, 30), 'trail': (255, 40, 40, 220), 'loupe': (210, 40, 40),
        'badge_good': (60, 0, 0), 'badge_bad': (175, 20, 20), 'badge_text': (255, 120, 120),
        'label_box': (0, 0, 0, 150), 'label': (220, 50, 50),
        'row_good': (55, 8, 8), 'row_bad': (125, 15, 15), 'row_flag': (85, 18, 10),
        'err_text': (255, 80, 80), 'bad_text': (110, 25, 25),
        'graph_bg': (12, 0, 0), 'graph_frame': (70, 10, 10), 'graph_text': (180, 40, 40),
        'graph_median': (110, 25, 25), 'graph_auto': (200, 50, 30), 'graph_manual': (255, 40, 40),
        'graph_line': (120, 22, 22), 'pt_good': (255, 90, 90), 'pt_bad': (110, 0, 0),
        'pt_flag': (210, 55, 30), 'pt_plain': (165, 40, 40), 'pt_current': (255, 140, 140),
        'hover_bg': (40, 0, 0, 230), 'hover_text': (235, 70, 70),
        'note_warn': '#e04030', 'note_dim': '#902020', 'header_hl': (75, 10, 10),
    },
}
THEME = {'name': 'Dark', 'c': THEMES['Dark']}


def C(role):
    v = THEME['c'][role]
    if isinstance(v, str):
        return QColor(v)
    return QColor(*v)


def night_mode():
    return THEME['name'].startswith('Night')


def night_tint(a8):
    """Night mode: show the picture in red only, so it does not spoil dark adaptation."""
    if not night_mode():
        return a8
    if a8.ndim == 3:
        lum = (a8[:, :, 0].astype(np.uint16) * 77 + a8[:, :, 1].astype(np.uint16) * 150
               + a8[:, :, 2].astype(np.uint16) * 29) >> 8
    else:
        lum = a8.astype(np.uint16)
    lum = ((lum * 205) >> 8).astype(np.uint8)      # also a little dimmer
    out = np.zeros(lum.shape + (3,), np.uint8)
    out[:, :, 0] = lum
    return out


class ThemeStyle(QProxyStyle):
    """Fusion, but with tick boxes and radio buttons drawn clearly in every theme
    (Fusion's own ones almost vanish on very dark or red backgrounds)."""

    def __init__(self):
        super().__init__('Fusion')

    def drawPrimitive(self, element, option, painter, widget=None):
        if element in (QStyle.PE_IndicatorCheckBox, QStyle.PE_IndicatorRadioButton):
            r = QRectF(option.rect).adjusted(1.5, 1.5, -1.5, -1.5)
            enabled = bool(option.state & QStyle.State_Enabled)
            on = bool(option.state & QStyle.State_On)
            edge = C('text') if enabled else C('disabled')
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(edge)
            pen.setWidthF(1.3)
            painter.setPen(pen)
            painter.setBrush(C('base'))
            if element == QStyle.PE_IndicatorRadioButton:
                painter.drawEllipse(r)
                if on:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(C('highlight').lighter(130) if not night_mode() else C('text'))
                    painter.drawEllipse(r.adjusted(r.width() * 0.27, r.height() * 0.27,
                                                   -r.width() * 0.27, -r.height() * 0.27))
            else:
                painter.drawRoundedRect(r, 2.5, 2.5)
                if on:
                    tick = QPen(C('text') if enabled else C('disabled'))
                    tick.setWidthF(max(1.8, r.width() * 0.16))
                    tick.setCapStyle(Qt.RoundCap)
                    tick.setJoinStyle(Qt.RoundJoin)
                    painter.setPen(tick)
                    painter.setBrush(Qt.NoBrush)
                    x, y, w, h = r.x(), r.y(), r.width(), r.height()
                    painter.drawPolyline([QPointF(x + w * 0.22, y + h * 0.52), QPointF(x + w * 0.42, y + h * 0.72),
                                          QPointF(x + w * 0.80, y + h * 0.28)])
            painter.restore()
            return
        super().drawPrimitive(element, option, painter, widget)


def apply_palette(app, name):
    THEME['name'] = name if name in THEMES else 'Dark'
    THEME['c'] = THEMES[THEME['name']]
    if THEME.get('style') is None:
        THEME['style'] = ThemeStyle()
        app.setStyle(THEME['style'])
    p = QPalette()
    q = C
    p.setColor(QPalette.Window, q('window'))
    p.setColor(QPalette.WindowText, q('text'))
    p.setColor(QPalette.Base, q('base'))
    p.setColor(QPalette.AlternateBase, q('alt'))
    p.setColor(QPalette.ToolTipBase, q('tooltip'))
    p.setColor(QPalette.ToolTipText, q('text'))
    p.setColor(QPalette.PlaceholderText, q('disabled'))
    p.setColor(QPalette.Text, q('text'))
    p.setColor(QPalette.Button, q('button'))
    p.setColor(QPalette.ButtonText, q('text'))
    p.setColor(QPalette.BrightText, q('bright'))
    p.setColor(QPalette.Highlight, q('highlight'))
    p.setColor(QPalette.HighlightedText, q('hltext'))
    p.setColor(QPalette.Mid, q('mid'))
    p.setColor(QPalette.Dark, q('mid').darker(130))
    p.setColor(QPalette.Light, q('button').lighter(130))
    p.setColor(QPalette.Midlight, q('button').lighter(115))
    p.setColor(QPalette.Shadow, QColor(0, 0, 0))
    p.setColor(QPalette.Link, q('highlight').lighter(140))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        p.setColor(QPalette.Disabled, role, q('disabled'))
    app.setPalette(p)
    app.setStyleSheet('QToolTip { color: %s; background-color: %s; border: 1px solid %s; }'
                      % (q('text').name(), q('tooltip').name(), q('mid').name()))


def draw_icon(kind, color, size=20):
    """Simple toolbar glyphs drawn in the theme's colour."""
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(2, 2)
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    s = float(size)

    def tri(x0, pointing_right):
        h = s * 0.62
        y0 = (s - h) / 2
        w = s * 0.36
        if pointing_right:
            pts = [QPointF(x0, y0), QPointF(x0 + w, s / 2), QPointF(x0, y0 + h)]
        else:
            pts = [QPointF(x0 + w, y0), QPointF(x0, s / 2), QPointF(x0 + w, y0 + h)]
        p.drawPolygon(pts)
    if kind == 'play':
        h = s * 0.7
        p.drawPolygon([QPointF(s * 0.28, (s - h) / 2), QPointF(s * 0.80, s / 2), QPointF(s * 0.28, (s + h) / 2)])
    elif kind == 'pause':
        p.drawRoundedRect(QRectF(s * 0.26, s * 0.18, s * 0.17, s * 0.64), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.57, s * 0.18, s * 0.17, s * 0.64), 1.5, 1.5)
    elif kind == 'next':
        tri(s * 0.12, True)
        tri(s * 0.46, True)
    elif kind == 'prev':
        tri(s * 0.18, False)
        tri(s * 0.52, False)
    p.end()
    pm.setDevicePixelRatio(2.0)
    icon = QIcon(pm)
    # a faded copy for when the button is disabled
    dim = QPixmap(pm.size())
    dim.fill(Qt.transparent)
    p = QPainter(dim)
    p.setOpacity(0.35)
    p.drawPixmap(0, 0, pm)
    p.end()
    dim.setDevicePixelRatio(2.0)
    icon.addPixmap(dim, QIcon.Disabled)
    return icon


def tinted_icon(icon, color, size=18):
    """A standard icon redrawn in one colour (used for Night mode)."""
    src = icon.pixmap(size * 2, size * 2)
    img = src.toImage().convertToFormat(QImage.Format_ARGB32)
    out = QImage(img.size(), QImage.Format_ARGB32)
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.drawImage(0, 0, img)
    p.setCompositionMode(QPainter.CompositionMode_SourceIn)
    p.fillRect(out.rect(), color)
    p.end()
    pm = QPixmap.fromImage(out)
    pm.setDevicePixelRatio(src.devicePixelRatio() or 1.0)
    return QIcon(pm)



def np_to_qimage(a8):
    """uint8 (h, w) or (h, w, 3) -> QImage that keeps its own copy."""
    a8 = np.ascontiguousarray(a8)
    h, w = a8.shape[:2]
    if a8.ndim == 2:
        q = QImage(a8.data, w, h, a8.strides[0], QImage.Format_Grayscale8)
    else:
        q = QImage(a8.data, w, h, a8.strides[0], QImage.Format_RGB888)
    return q.copy()


class Stretch:
    """Display stretch settings and LUT building."""
    MODES = ('Auto stretch (each frame)', 'Auto stretch (same for all)', 'Linear')

    def __init__(self):
        self.mode = 1
        self.linked = False
        self.target = 0.25
        self.ref_stats = None

    def key(self):
        return (self.mode, self.linked, round(self.target, 3),
                None if self.ref_stats is None or self.mode == 0 else tuple(map(tuple, self.ref_stats)))

    def luts(self, frame_stats):
        """frame_stats: list of (median, mad) per channel. Returns list of LUTs per channel."""
        stats = frame_stats
        if self.mode == 1 and self.ref_stats is not None and len(self.ref_stats) == len(frame_stats):
            stats = self.ref_stats
        if self.mode == 2:
            lo = min(s[0] - 3 * s[1] for s in stats)
            hi = max(min(1.0, s[0] + 60 * s[1]) for s in stats)
            lo = max(0.0, lo)
            return [linear_lut(lo, max(hi, lo + 1e-4))] * len(stats)
        if self.linked and len(stats) > 1:
            med = sum(s[0] for s in stats) / len(stats)
            mad = sum(s[1] for s in stats) / len(stats)
            (c0, m), = stretch_params([(med, mad)], self.target)
            lut = build_lut(c0, m)
            return [lut] * len(stats)
        return [build_lut(c0, m) for c0, m in stretch_params(stats, self.target)]


def apply_luts(img16, luts):
    if img16.ndim == 2:
        return luts[0][img16]
    if all(l is luts[0] for l in luts):
        return luts[0][img16]
    out = np.empty(img16.shape, np.uint8)
    for c in range(img16.shape[2]):
        out[:, :, c] = luts[c][img16[:, :, c]]
    return out

class ImageView(QWidget):
    pixelInfo = Signal(str)
    wantFull = Signal()
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 240)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.qimg = None            # stretched preview
        self.prev_scale = 1.0
        self.full = None            # uint16 full image of the current frame
        self.luts = None
        self.fw = self.fh = 0       # current frame size (full pixels)
        self.rw = self.rh = 0       # reference frame size
        self.shift = (0.0, 0.0)
        self.flipped = False
        self.trails = []
        self.show_trails = True
        self.show_info = True
        self.loupe = False
        self.label = ''
        self.sublabel = ''
        self.mark = ''
        self.message = 'Open a folder of subs to start  (Ctrl+O)'
        self.z = 1.0
        self.cx = self.cy = 0.0
        self.fit_mode = True
        self.mouse = None
        self.drag = None
        self.crop_cache = None

    # ---------------- state ------------------------------------------------
    def set_frame(self, qimg, prev_scale, fw, fh, luts, full, shift, flipped, trails,
                  label, sublabel, mark):
        new_size = (fw, fh) != (self.fw, self.fh)
        self.qimg, self.prev_scale, self.fw, self.fh = qimg, prev_scale, fw, fh
        self.luts, self.full = luts, full
        self.shift, self.flipped, self.trails = shift, flipped, trails
        self.label, self.sublabel, self.mark = label, sublabel, mark
        self.crop_cache = None
        self.message = ''
        if self.rw == 0 or new_size and self.fit_mode:
            self.rw, self.rh = fw, fh
        if self.fit_mode:
            self._fit()
        self.update()

    def set_reference_size(self, w, h):
        self.rw, self.rh = w, h
        if self.fit_mode:
            self._fit()

    def clear(self, message=''):
        self.qimg = None
        self.full = None
        self.fw = self.fh = self.rw = self.rh = 0
        self.message = message
        self.update()

    def needs_full(self):
        return self.qimg is not None and self.z * self.devicePixelRatioF() * self.prev_scale > 1.25

    def zoom_one(self):
        # one physical screen pixel per image pixel (allows for Windows display scaling)
        self.zoom_to(1.0 / max(1.0, self.devicePixelRatioF()))

    def _fit(self):
        if self.rw <= 0:
            return
        self.z = min(self.width() / float(self.rw), self.height() / float(self.rh)) * 0.98
        self.cx, self.cy = self.rw / 2.0, self.rh / 2.0
        self.zoomChanged.emit(self.z)

    def fit(self):
        self.fit_mode = True
        self._fit()
        self.update()

    def zoom_to(self, z, anchor=None):
        z = float(np.clip(z, 0.02, 32.0))
        if anchor is not None:
            # keep the point under the cursor still
            px, py = self._to_ref(anchor)
            self.z = z
            self.cx = px - (anchor.x() - self.width() / 2.0) / z
            self.cy = py - (anchor.y() - self.height() / 2.0) / z
        else:
            self.z = z
        self.fit_mode = False
        self.zoomChanged.emit(self.z)
        if self.needs_full() and self.full is None:
            self.wantFull.emit()
        self.update()

    def resizeEvent(self, e):
        if self.fit_mode:
            self._fit()
        super().resizeEvent(e)

    # ---------------- coordinates -------------------------------------------
    def _to_ref(self, p):
        return (self.cx + (p.x() - self.width() / 2.0) / self.z,
                self.cy + (p.y() - self.height() / 2.0) / self.z)

    def _ref_to_frame(self, x, y):
        dx, dy = self.shift
        u, v = x + dx, y + dy
        if self.flipped:
            u, v = self.fw - u, self.fh - v
        return u, v

    def _frame_transform(self, p):
        """Painter is in reference coordinates; move into frame coordinates."""
        dx, dy = self.shift
        p.translate(-dx, -dy)
        if self.flipped:
            p.translate(self.fw, self.fh)
            p.rotate(180)

    # ---------------- painting ----------------------------------------------
    def _draw_frame(self, p, zoom, device_rect):
        """Draw the current frame; painter already set to frame coordinates."""
        use_full = self.full is not None and zoom * self.devicePixelRatioF() * self.prev_scale > 1.25
        if not use_full:
            p.setRenderHint(QPainter.SmoothPixmapTransform, zoom * self.prev_scale < 2.0)
            p.drawImage(QRectF(0, 0, self.fw, self.fh), self.qimg)
            return
        inv, _ok = p.transform().inverted()
        vis = inv.mapRect(QRectF(device_rect))
        x0 = int(max(0, math.floor(vis.left()) - 2))
        y0 = int(max(0, math.floor(vis.top()) - 2))
        x1 = int(min(self.fw, math.ceil(vis.right()) + 2))
        y1 = int(min(self.fh, math.ceil(vis.bottom()) + 2))
        if x1 <= x0 or y1 <= y0:
            return
        # At 1:1 Qt snaps a plain translation to whole screen pixels, which
        # would undo the sub-pixel alignment; so shift the crop ourselves.
        T = p.transform()
        fx = fy = 0.0
        if abs(T.m11() - 1.0) < 1e-6 and abs(T.m22() - 1.0) < 1e-6 and abs(T.m12()) < 1e-9 and abs(T.m21()) < 1e-9:
            dev = T.map(QPointF(x0, y0))
            fx = dev.x() - math.floor(dev.x())
            fy = dev.y() - math.floor(dev.y())
            if fx < 0.02 or fx > 0.98:
                fx = 0.0
            if fy < 0.02 or fy > 0.98:
                fy = 0.0
        key = (id(self.full), x0, y0, x1, y1, id(self.luts), round(fx, 2), round(fy, 2))
        if self.crop_cache is None:
            self.crop_cache = {}
        q = self.crop_cache.get(key)
        if q is None:
            a8 = apply_luts(self.full[y0:y1, x0:x1], self.luts)
            if fx or fy:
                # content moves right/down by the fraction; drawn one fraction earlier
                M = np.array([[1, 0, fx], [0, 1, fy]], np.float32)
                a8 = cv2.warpAffine(a8, M, (a8.shape[1], a8.shape[0]), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            q = np_to_qimage(night_tint(a8))
            if len(self.crop_cache) > 3:
                self.crop_cache.clear()
            self.crop_cache[key] = q
        p.setRenderHint(QPainter.SmoothPixmapTransform, zoom < 4.0)
        p.drawImage(QRectF(x0 - fx, y0 - fy, x1 - x0, y1 - y0), q)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), C('view_bg'))
        if self.qimg is None:
            p.setPen(C('view_msg'))
            f = p.font()
            f.setPointSize(13)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.message)
            p.end()
            return
        p.save()
        p.translate(self.width() / 2.0, self.height() / 2.0)
        p.scale(self.z, self.z)
        p.translate(-self.cx, -self.cy)
        self._frame_transform(p)
        self._draw_frame(p, self.z, self.rect())
        if self.show_trails and self.trails:
            pen = QPen(C('trail'))
            pen.setCosmetic(True)
            pen.setWidthF(2.0)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            for x1, y1, x2, y2 in self.trails:
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        p.restore()
        # loupe
        if self.loupe and self.mouse is not None:
            self._draw_loupe(p)
        if self.show_info:
            self._draw_overlay(p)
        p.end()

    def _draw_loupe(self, p):
        size = 240
        m = self.mouse
        r = QRect(int(m.x() + 24), int(m.y() + 24), size, size)
        if r.right() > self.width():
            r.moveRight(int(m.x() - 24))
        if r.bottom() > self.height():
            r.moveBottom(int(m.y() - 24))
        zl = max(self.z * 4.0, 2.0) if self.full is not None else max(self.z * 4.0, 1.0 / self.prev_scale * 2)
        zl = min(zl, 16.0)
        rx, ry = self._to_ref(m)
        p.save()
        p.setClipRect(r)
        p.fillRect(r, C('view_bg'))
        p.translate(r.center().x(), r.center().y())
        p.scale(zl, zl)
        p.translate(-rx, -ry)
        self._frame_transform(p)
        self._draw_frame(p, zl, r)
        p.restore()
        pen = QPen(C('loupe'))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawRect(r)
        p.drawText(r.adjusted(6, 4, -6, -4), Qt.AlignTop | Qt.AlignLeft, 'x%.1f' % zl)

    def _draw_overlay(self, p):
        f = QFont(self.font())
        f.setPointSize(11)
        f.setBold(True)
        p.setFont(f)
        fm = QFontMetrics(f)
        x, y = 10, 10
        if self.mark:
            txt = 'GOOD' if self.mark == 'good' else 'BAD'
            col = C('badge_good') if self.mark == 'good' else C('badge_bad')
            w = fm.horizontalAdvance(txt) + 16
            p.fillRect(QRect(x, y, w, fm.height() + 8), col)
            p.setPen(C('badge_text'))
            p.drawText(QRect(x, y, w, fm.height() + 8), Qt.AlignCenter, txt)
            x += w + 8
        if self.label:
            w = fm.horizontalAdvance(self.label) + 14
            p.fillRect(QRect(x, y, w, fm.height() + 8), C('label_box'))
            p.setPen(C('label'))
            p.drawText(QRect(x, y, w, fm.height() + 8), Qt.AlignCenter, self.label)
        if self.sublabel:
            f.setBold(False)
            f.setPointSize(10)
            p.setFont(f)
            fm2 = QFontMetrics(f)
            w = min(self.width() - 12, fm2.horizontalAdvance(self.sublabel) + 14)
            r = QRect(6, self.height() - fm2.height() - 14, w, fm2.height() + 8)
            p.fillRect(r, C('label_box'))
            p.setPen(C('label'))
            p.drawText(r, Qt.AlignCenter, fm2.elidedText(self.sublabel, Qt.ElideRight, w - 10))
        f.setBold(True)
        f.setPointSize(11)
        p.setFont(f)
        zt = '%d%%' % round(self.z * self.devicePixelRatioF() * 100)
        w = fm.horizontalAdvance(zt) + 14
        r = QRect(self.width() - w - 8, y, w, fm.height() + 8)
        p.fillRect(r, C('label_box'))
        p.setPen(C('label'))
        p.drawText(r, Qt.AlignCenter, zt)

    # ---------------- mouse -------------------------------------------------
    def wheelEvent(self, e):
        if self.qimg is None:
            return
        steps = e.angleDelta().y() / 120.0
        self.zoom_to(self.z * (1.25 ** steps), e.position())

    def mousePressEvent(self, e):
        if e.button() in (Qt.LeftButton, Qt.MiddleButton):
            self.drag = (e.position(), self.cx, self.cy)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, e):
        self.drag = None
        self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, e):
        if self.fit_mode:
            self.zoom_to(1.0 / max(1.0, self.devicePixelRatioF()), e.position())
        else:
            self.fit()

    def leaveEvent(self, e):
        self.mouse = None
        if self.loupe:
            self.update()

    def mouseMoveEvent(self, e):
        self.mouse = e.position()
        if self.drag is not None:
            p0, cx, cy = self.drag
            d = e.position() - p0
            self.cx = cx - d.x() / self.z
            self.cy = cy - d.y() / self.z
            self.fit_mode = False
            self.update()
        elif self.loupe:
            self.update()
        if self.qimg is not None:
            rx, ry = self._to_ref(e.position())
            u, v = self._ref_to_frame(rx, ry)
            iu, iv = int(math.floor(u)), int(math.floor(v))
            if 0 <= iu < self.fw and 0 <= iv < self.fh:
                if self.full is not None:
                    val = self.full[iv, iu]
                    approx = ''
                else:
                    a = self._preview_array()
                    if a is None:
                        return
                    pu = min(a.shape[1] - 1, int(u / self.prev_scale))
                    pv = min(a.shape[0] - 1, int(v / self.prev_scale))
                    val = a[pv, pu]
                    approx = '~'
                if np.ndim(val):
                    vs = 'R %s%d  G %s%d  B %s%d' % (approx, val[0], approx, val[1], approx, val[2])
                else:
                    vs = '%s%d' % (approx, val)
                self.pixelInfo.emit('x %d  y %d   %s' % (iu, iv, vs))
            else:
                self.pixelInfo.emit('')

    def _preview_array(self):
        return getattr(self, 'preview16', None)

COLUMNS = [
    # key, title, tooltip
    ('idx', '#', 'Order the files were loaded in'),
    ('mark', 'Mark', 'G = good, B = bad, Ref = reference frame'),
    ('name', 'File', ''),
    ('date', 'Date / time', 'DATE-OBS from the header'),
    ('exposure', 'Exp (s)', 'Exposure time'),
    ('filter', 'Filter', ''),
    ('temp', 'Temp', 'Sensor temperature'),
    ('background', 'Background', 'Median level (0-65535 scale)'),
    ('noise', 'Noise', 'Background noise, from neighbouring pixel differences'),
    ('snr', 'SNR', 'Mean signal above the background divided by the noise. Higher is better'),
    ('stars', 'Stars', 'Stars found above 5 sigma'),
    ('fwhm', 'FWHM (px)', 'Median star FWHM in pixels (Gaussian-weighted moments)'),
    ('fwhm_arcsec', 'FWHM (")', 'FWHM in arcseconds, from the pixel size and focal length (or image scale) in the header'),
    ('ecc', 'Ecc', 'Median star eccentricity. 0 = round, above about 0.5 looks elongated'),
    ('saturated', 'Sat %', 'Percentage of saturated pixels'),
    ('trails', 'Trails', 'Straight trails found (satellites, aircraft)'),
    ('shift', 'Shift (px)', 'Offset from the reference frame'),
    ('reason', 'Why flagged', 'Why this frame is flagged: the measurements past their limits, or a trail'),
]
COL = {c[0]: i for i, c in enumerate(COLUMNS)}
GRAPH_TIPS = {
    'fwhm': 'Star size in pixels (width at half brightness).\nLower is better - rises with poor focus, seeing or cloud.',
    'fwhm_arcsec': 'Star size in arcseconds (FWHM x pixel scale).\nLower is better.',
    'ecc': 'How stretched the stars are: 0 = round, 1 = a line.\nLower is better - above about 0.5 looks '
           'elongated (tracking, wind, guiding).',
    'stars': 'Stars found in the frame.\nHigher is better - drops with cloud, haze, dew or poor focus.',
    'background': 'Sky brightness (0 to 65535 scale).\nLower is better - rises with moon, dawn, light '
                  'pollution or cloud.',
    'noise': 'Random pixel-to-pixel noise in the background.\nLower is better - rises with a brighter sky.',
    'snr': 'Signal above the background divided by the noise.\nHigher is better.',
    'saturated': 'Share of pixels at the camera\'s maximum (clipped).\nLower is better - small values from '
                 'bright stars are normal.',
    'temp': 'Sensor temperature from the header.\nShould stay steady with a cooled camera.',
    'alt': 'How high the target is above the horizon, from the header.\nLower means more atmosphere: '
           'softer stars and a brighter sky.',
}
GRAPH_KEYS = [('fwhm', 'FWHM (px)'), ('fwhm_arcsec', 'FWHM (")'), ('ecc', 'Eccentricity'),
              ('stars', 'Star count'), ('background', 'Background'), ('noise', 'Noise'),
              ('snr', 'SNR'), ('saturated', 'Saturated %'), ('temp', 'Temperature'),
              ('alt', 'Altitude')]


def _fmt(key, v):
    if v is None or v == '':
        return ''
    if key in ('stars', 'trails'):
        return '%d' % v
    if key in ('background',):
        return '%.0f' % v
    if key in ('noise',):
        return '%.2f' % v if v < 100 else '%.0f' % v
    if key in ('snr', 'fwhm', 'fwhm_arcsec', 'ecc'):
        return '%.2f' % v
    if key == 'saturated':
        return '%.3f' % v
    if key in ('exposure',):
        return ('%.3f' % v).rstrip('0').rstrip('.')
    if key in ('temp', 'alt'):
        return '%.1f' % v
    return str(v)


class FrameModel(QAbstractTableModel):
    def __init__(self, win):
        super().__init__()
        self.win = win

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.win.frames)

    def columnCount(self, parent=QModelIndex()):
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return COLUMNS[section][1]
            if role == Qt.ToolTipRole:
                return COLUMNS[section][2] or None
        return None

    def value(self, fr, key):
        if key == 'idx':
            return fr.uid + 1
        if key == 'mark':
            m = {'good': 'G', 'bad': 'B'}.get(fr.mark, '')
            if fr.uid == self.win.ref:
                m = (m + (' Ref' if self.win.ref_final else ' Ref?')).strip()
            return m
        if key == 'name':
            return fr.name
        if key == 'reason':
            if fr.status == 'error':
                return fr.an.error if fr.an else 'could not read'
            return '; '.join(fr.reasons)
        if key == 'shift':
            if fr.shift is None:
                return None
            dx, dy, fl = fr.shift
            return math.hypot(dx, dy)
        if fr.an is None or not fr.an.ok:
            return None
        if key in fr.an.stats:
            return fr.an.stats[key]
        return fr.an.hv.get(key)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        fr = self.win.frames[index.row()]
        key = COLUMNS[index.column()][0]
        if role == Qt.DisplayRole:
            v = self.value(fr, key)
            if key == 'shift' and fr.shift is not None:
                dx, dy, fl = fr.shift
                return '%.1f, %.1f%s' % (dx, dy, ' (turned)' if fl else '')
            if key == 'date' and v:
                # 2026-10-05T08:43:14.4321 -> 2026-10-05 08:43:14 (full value in the tooltip)
                return str(v).replace('T', ' ')[:19]
            if key == 'name' and fr.status in ('waiting', 'working'):
                return fr.name + ('  ...' if fr.status == 'working' else '')
            return _fmt(key, v) if key not in ('name', 'mark', 'reason', 'date', 'filter') else (v or '')
        if role == Qt.UserRole:
            v = self.value(fr, key)
            if v is None or v == '':
                return float('inf')
            if key == 'date':
                return v
            return v if not isinstance(v, str) else v.lower()
        if role == Qt.BackgroundRole:
            if key == 'mark' and fr.mark:
                return QBrush(C('row_good') if fr.mark == 'good' else C('row_bad'))
            if fr.reasons and not fr.mark:
                return QBrush(C('row_flag'))
        if role == Qt.ForegroundRole:
            if fr.status == 'error':
                return QBrush(C('err_text'))
            if fr.mark == 'bad' and key != 'mark':
                return QBrush(C('bad_text'))
        if role == Qt.FontRole and fr.uid == self.win.ref:
            f = QFont()
            f.setBold(True)
            return f
        if role == Qt.TextAlignmentRole:
            if key in ('mark',):
                return int(Qt.AlignCenter)
            if key not in ('name', 'date', 'filter', 'reason'):
                return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.ToolTipRole:
            if key == 'reason' and fr.reasons:
                return '\n'.join(fr.reasons)
            if key == 'mark' and fr.uid == self.win.ref:
                if not self.win.ref_final:
                    return ('Temporary reference while measuring - the proper one is chosen when every '
                            'frame is measured.\nPress R on a frame to choose it yourself.')
                return 'Reference frame - used for alignment and the "same for all" stretch.\nShow another frame and press R to change it.'
            if key == 'name':
                return fr.src.path
            if key == 'date':
                return self.value(fr, key) or None
        return None

    def refresh_row(self, row):
        self.dataChanged.emit(self.index(row, 0), self.index(row, len(COLUMNS) - 1))

    def refresh_all(self):
        if self.win.frames:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.win.frames) - 1, len(COLUMNS) - 1))

    def reset(self):
        self.beginResetModel()
        self.endResetModel()


class SortProxy(QSortFilterProxyModel):
    def lessThan(self, a, b):
        va = self.sourceModel().data(a, Qt.UserRole)
        vb = self.sourceModel().data(b, Qt.UserRole)
        try:
            return va < vb
        except TypeError:
            return str(va) < str(vb)


# ------------------------------- graphs ------------------------------------

class GraphPlot(QWidget):
    clicked = Signal(int)

    def __init__(self, win, key):
        super().__init__()
        self.win = win
        self.key = key
        self.setMinimumHeight(110)
        self.setMouseTracking(True)
        self.pts = []
        self.hover = None

    def sizeHint(self):
        return QSize(400, 150)

    def _series(self):
        frames = self.win.time_order()
        xs, ys, fs = [], [], []
        for i, fr in enumerate(frames):
            v = self.win.model.value(fr, self.key) if self.key != 'alt' else (
                fr.an.hv.get('alt') if fr.an and fr.an.ok else None)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                continue
            xs.append(i)
            ys.append(float(v))
            fs.append(fr)
        return xs, ys, fs, len(frames)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), C('graph_bg'))
        xs, ys, fs, n = self._series()
        f = p.font()
        f.setPointSize(9)
        p.setFont(f)
        title = dict(GRAPH_KEYS).get(self.key, self.key)
        R, T, B = 10, 8, 18
        top_txt = bot_txt = ''
        if ys:
            lo, hi = min(ys), max(ys)
            med = float(np.median(ys))
            if hi - lo < 1e-9:
                lo, hi = lo - 1, hi + 1
            pad = (hi - lo) * 0.08
            lo, hi = lo - pad, hi + pad
            top, bot = hi - pad, lo + pad
            digits = max(0, min(4, int(math.ceil(-math.log10(max(1e-9, top - bot)))) + 2)) if top > bot else 0
            top_txt, bot_txt = '%.*f' % (digits, top), '%.*f' % (digits, bot)
        # left margin wide enough for the scale numbers, however many digits
        fm = p.fontMetrics()
        L = max(52, max(fm.horizontalAdvance(top_txt), fm.horizontalAdvance(bot_txt)) + 12)
        W, H = self.width() - L - R, self.height() - T - B
        p.setPen(C('graph_frame'))
        p.drawRect(L, T, W, H)
        p.setPen(C('graph_text'))
        self.pts = []
        if len(ys) < 1:
            ok = [f for f in self.win.frames if f.status == 'ok']
            star_key = self.key in ('fwhm', 'ecc', 'stars', 'fwhm_arcsec')
            if star_key and ok and all(f.an.quick for f in ok):
                note = ' - not measured for video frames'
            else:
                note = ' - no data yet'
            p.drawText(QRect(L, T, W, H), Qt.AlignCenter, title + note)
            p.end()
            return

        def X(i):
            return L + (i + 0.5) * W / max(1, n)

        def Y(v):
            return T + H - (v - lo) / (hi - lo) * H
        # median and reject limit
        pen = QPen(C('graph_median'))
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(L, Y(med)), QPointF(L + W, Y(med)))
        lim = self.win.reject_limit(self.key, ys)
        if lim is not None:
            manual = self.win.reject_panel.is_manual()
            pen = QPen(C('graph_manual') if manual else C('graph_auto'))
            pen.setStyle(Qt.SolidLine if manual else Qt.DotLine)
            pen.setWidthF(1.5 if manual else (1.0 if THEME['name'] == 'Light' else 1.5))
            p.setPen(pen)
            if lo <= lim <= hi:
                p.drawLine(QPointF(L, Y(lim)), QPointF(L + W, Y(lim)))
            else:
                # off the graph: say so rather than pin a line to the edge
                where = 'above' if lim > hi else 'below'
                note = 'limit %s (%s)' % (_fmt(self.key, lim) or '%.3g' % lim, where)
                p.setPen(C('graph_manual') if manual else C('graph_auto'))
                # written under the plot, at the right, so it never sits on the dots
                p.drawText(QRect(L, T + H + 2, W, 14), Qt.AlignRight, note)
        # line
        p.setPen(QPen(C('graph_line'), 1))
        for k in range(1, len(xs)):
            p.drawLine(QPointF(X(xs[k - 1]), Y(ys[k - 1])), QPointF(X(xs[k]), Y(ys[k])))
        cur = self.win.current_frame()
        fam = reason_family(self.key)
        for x, y, fr in zip(xs, ys, fs):
            px, py = X(x), Y(y)
            ring = False
            if fr.mark == 'good':
                c = C('pt_good')
            elif fr.mark == 'bad':
                c = C('pt_bad')
            elif fr.reasons and fam in fr.reason_keys:
                c = C('pt_flag')              # flagged for THIS measurement
            elif fr.reasons:
                c = C('pt_plain')             # flagged for something else: orange ring
                ring = True
            else:
                c = C('pt_plain')
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawEllipse(QPointF(px, py), 3.2, 3.2)
            if ring:
                # thicker, wider ring on the dark themes so it stands out
                light = THEME['name'] == 'Light'
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(C('pt_flag'), 1.6 if light else 2.2))
                p.drawEllipse(QPointF(px, py), 3.6 if light else 4.6, 3.6 if light else 4.6)
            if fr is cur:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(C('pt_current'), 1.5))
                p.drawEllipse(QPointF(px, py), 6.5, 6.5)
            self.pts.append((px, py, fr, y))
        # labels
        p.setPen(C('graph_text'))
        p.drawText(QRect(2, T - 2, L - 6, 14), Qt.AlignRight, top_txt)
        p.drawText(QRect(2, T + H - 12, L - 6, 14), Qt.AlignRight, bot_txt)
        p.drawText(QRect(L + 6, T + 2, W - 12, 14), Qt.AlignLeft, title)
        p.drawText(QRect(L, T + H + 2, W, 14), Qt.AlignLeft, 'frames in time order')
        if self.hover is not None:
            px, py, fr, y = self.hover
            txt = '%s   %s' % (fr.name, _fmt(self.key, y))
            if fr.mark:
                txt += '   marked %s' % ('Good' if fr.mark == 'good' else 'Bad')
            elif fr.reasons:
                names = {'fwhm': 'FWHM', 'ecc': 'eccentricity', 'stars': 'star count', 'background': 'background',
                         'snr': 'SNR', 'trails': 'trails'}
                txt += '   flagged for ' + ', '.join(names.get(k, k) for k in sorted(fr.reason_keys))
            fm = p.fontMetrics()
            txt = fm.elidedText(txt, Qt.ElideMiddle, W - 10)
            w = fm.horizontalAdvance(txt) + 10
            bx = min(max(L, px - w / 2), L + W - w)
            p.fillRect(QRectF(bx, T + 16, w, 16), C('hover_bg'))
            p.setPen(C('hover_text'))
            p.drawText(QRectF(bx, T + 16, w, 16), Qt.AlignCenter, txt)
        p.end()

    def _nearest(self, pos):
        best, bd = None, 1e9
        for pt in self.pts:
            d = abs(pt[0] - pos.x()) + 0.3 * abs(pt[1] - pos.y())
            if d < bd:
                best, bd = pt, d
        return best if bd < 25 else None

    def mouseMoveEvent(self, e):
        h = self._nearest(e.position())
        if h is not self.hover:
            self.hover = h
            self.update()

    def leaveEvent(self, e):
        self.hover = None
        self.update()

    def mousePressEvent(self, e):
        h = self._nearest(e.position())
        if h is not None:
            self.clicked.emit(h[2].uid)


class GraphRow(QWidget):
    """Tick box + slider + value under one graph, for whichever measurement
    the graph shows. In Automatic mode the slider is that test's sigma, in
    Manual mode it is that measurement's limit. Drag right = stricter."""

    def __init__(self, panel, plot):
        super().__init__()
        self.panel = panel
        self.plot = plot
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        self.cb = QCheckBox()
        self.sl = QSlider(Qt.Horizontal)
        self.lab = QLabel('')
        self.lab.setMinimumWidth(190)
        lay.addWidget(self.cb)
        lay.addWidget(self.sl, 1)
        lay.addWidget(self.lab)
        self.sl.valueChanged.connect(self._moved)
        self.cb.toggled.connect(self._ticked)

    def _family(self):
        return reason_family(self.plot.key)

    def bind(self):
        """Show the current state (without sending it back)."""
        rp = self.panel.win.reject_panel
        f = self._family()
        manual = rp.is_manual()
        for w in (self.cb, self.sl):
            w.blockSignals(True)
        try:
            if not manual and f in rp.rules:
                cb, sp = rp.rules[f]
                self.cb.setVisible(True)
                self.sl.setVisible(True)
                self.sl.setRange(0, 500)
                self.sl.setValue(int(round((6.0 - sp.value()) * 100)))
                self.cb.setChecked(cb.isChecked())
                self.sl.setEnabled(cb.isChecked())
                self.lab.setText(self._auto_label(sp.value(), cb.isChecked()))
                self.lab.setEnabled(cb.isChecked())
                self.cb.setToolTip('Use this test')
                self.sl.setToolTip('How far from the session median a frame may be, in robust standard '
                                   'deviations.\nDrag right to be stricter.')
            elif manual and f in rp.limits:
                d = rp.limits[f]
                self.cb.setVisible(True)
                self.sl.setVisible(True)
                self.sl.setRange(0, 1000)
                self.sl.setValue(d['sl'].value())
                self.cb.setChecked(d['cb'].isChecked())
                self.sl.setEnabled(d['sl'].isEnabled())
                self.lab.setText(d['lab'].text())
                self.lab.setEnabled(d['cb'].isChecked())
                self.cb.setToolTip('Use this limit')
                self.sl.setToolTip('Every frame passes at the left. Drag right to be stricter - the worst '
                                   'frames for this measurement go first.')
            else:
                self.cb.setVisible(False)
                self.sl.setVisible(False)
                self.lab.setEnabled(True)
                self.lab.setText('No limit for this measurement')
        finally:
            for w in (self.cb, self.sl):
                w.blockSignals(False)

    def _auto_label(self, sigma, on):
        """'2.0 sigma -> limit 2.88 px': the sigma and the cut-off it gives."""
        txt = '%.1f sigma' % sigma
        if not on:
            return txt + '  (off)'
        win = self.panel.win
        key = self.plot.key
        ys = [win.model.value(f, key) for f in win.frames if f.status == 'ok']
        ys = [float(v) for v in ys if v is not None and math.isfinite(float(v))]
        if len(win._groups()) > 1:
            return txt + '  (limits per group)'
        info = win.auto_limit_info(key, ys) if len(ys) >= 4 else None
        if info is None:
            return txt
        lim, by_gap = info
        unit = {'fwhm': ' px', 'fwhm_arcsec': '"'}.get(key, '')
        if key in ('stars', 'background'):
            val = '%d' % round(lim)
        elif key == 'ecc':
            val = '%.2f' % lim
        else:
            val = '%.2f' % lim
        # on a steady night the minimum gap sets the limit, so the slider has
        # no effect until it is strict enough - say so
        return '%s \u2192 limit %s%s%s' % (txt, val, unit, '  (min. gap)' if by_gap else '')

    def _moved(self, pos):
        rp = self.panel.win.reject_panel
        f = self._family()
        if rp.is_manual():
            if f in rp.limits:
                rp.limits[f]['sl'].setValue(pos)      # rp handles ticking and re-flagging
        elif f in rp.rules:
            cb, sp = rp.rules[f]
            if not cb.isChecked():
                cb.setChecked(True)
            sp.setValue(6.0 - pos / 100.0)
            self.panel.win.schedule_auto_reject()
        self.panel.sync_rows()

    def _ticked(self, on):
        rp = self.panel.win.reject_panel
        f = self._family()
        if rp.is_manual():
            if f in rp.limits:
                rp.limits[f]['cb'].setChecked(on)
        elif f in rp.rules:
            rp.rules[f][0].setChecked(on)
            self.panel.win.schedule_auto_reject()
        self.panel.sync_rows()


class GraphPanel(QWidget):
    DEFAULTS = ['fwhm', 'ecc', 'stars', 'background']

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.lay = lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)
        self.plots = []
        self.combos = []
        self.rows = []
        self.head = QVBoxLayout()            # filled by attach_reject()
        lay.addLayout(self.head)
        for k in self.DEFAULTS:
            row = QHBoxLayout()
            cb = QComboBox()
            for i, (key, title) in enumerate(GRAPH_KEYS):
                cb.addItem(title, key)
                cb.setItemData(i, GRAPH_TIPS.get(key, ''), Qt.ToolTipRole)
            cb.setCurrentIndex([g[0] for g in GRAPH_KEYS].index(k))
            cb.setToolTip(GRAPH_TIPS.get(k, ''))
            plot = GraphPlot(win, k)
            plot.setMinimumHeight(90)
            plot.clicked.connect(win.go_uid)
            cb.currentIndexChanged.connect(lambda _i, c=cb, pl=plot: self._set(pl, c))
            row.addWidget(cb)
            row.addStretch(1)
            lay.addLayout(row)
            lay.addWidget(plot, 1)
            gr = GraphRow(self, plot)
            lay.addWidget(gr)
            self.plots.append(plot)
            self.combos.append(cb)
            self.rows.append(gr)

    def restyle(self):
        if getattr(self, 'hint', None) is not None:
            self.hint.setStyleSheet('color: %s' % C('note_dim').name())

    def attach_reject(self, rp):
        """Put the reject controls on this tab (the Reject panel itself stays hidden)."""
        win = self.win
        # row 1: mode, and the fold-out
        r1 = QHBoxLayout()
        self.hint = QLabel('Drag a slider right to be stricter')
        self.hint.setStyleSheet('color: %s' % C('note_dim').name())
        r1.addWidget(self.hint)
        r1.addStretch(1)
        self.more_btn = QToolButton()
        self.more_btn.setText('More options')
        self.more_btn.setCheckable(True)
        self.more_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.more_btn.setArrowType(Qt.RightArrow)
        self.more_btn.setAutoRaise(True)
        self.more_btn.toggled.connect(self._fold)
        r1.addWidget(self.more_btn)
        self.head.addLayout(r1)
        # row 2: actions
        r2 = QHBoxLayout()
        self.find_btn = QPushButton('Suggest limits')
        self.find_btn.setToolTip('Sets each ticked slider to tonight\'s "odd ones out" limit and flags the frames '
                                 'past it.\nYou can then drag any slider to adjust.')
        self.find_btn.clicked.connect(lambda: win.suggest_limits())
        self.mark_btn = QPushButton('Mark them Bad')
        self.mark_btn.setToolTip('Marks the flagged (orange) frames Bad. Nothing is moved.')
        self.mark_btn.clicked.connect(win.apply_auto_reject)
        self.mark_btn.setEnabled(False)
        r2.addWidget(self.find_btn)
        r2.addWidget(self.mark_btn)
        r2.addWidget(rp.rest_good)
        r2.addStretch(1)
        self.head.addLayout(r2)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        self.head.addWidget(self.status)
        # fold-out: the less used options
        self.more = QFrame()
        self.more.setFrameShape(QFrame.StyledPanel)
        ml = QVBoxLayout(self.more)
        ml.setContentsMargins(6, 4, 6, 4)
        ml.addWidget(rp.trails)
        ml.addWidget(rp.auto_run)
        self.video_stars = QCheckBox('Measure stars in SER and video frames (slow)')
        self.video_stars.setToolTip('Off: frames inside SER, AVI, MP4 and MOV files are only read for viewing and '
                                    'their brightness,\nso even long videos are ready to play in seconds. '
                                    'On: every frame gets the full star measurement.')
        self.video_stars.setChecked(bool(win.opts.get('video_stars')))
        self.video_stars.toggled.connect(win._video_stars_changed)
        ml.addWidget(self.video_stars)
        srow = QHBoxLayout()
        srow.addWidget(QLabel('Suggest strictness (sigma)'))
        srow.addWidget(rp.sigma_sp)
        srow.addStretch(1)
        ml.addLayout(srow)
        self.best_row = QWidget()
        bl = QGridLayout(self.best_row)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setColumnStretch(2, 1)
        bl.addWidget(rp.best_cb, 0, 0)
        bl.addWidget(rp.best_key, 0, 1)
        # the slider gives way to the label, so a long count is never cut off
        brow = QHBoxLayout()
        brow.addWidget(rp.best_sl, 1)
        brow.addWidget(rp.best_lab, 0)
        bl.addLayout(brow, 1, 0, 1, 3)
        ml.addWidget(self.best_row)
        r3 = QHBoxLayout()
        self.reset_btn = QPushButton('Reset the sliders')
        self.reset_btn.clicked.connect(rp.reset_limits)
        r3.addWidget(self.reset_btn)
        r3.addWidget(rp.clear_btn)
        r3.addStretch(1)
        ml.addLayout(r3)
        self.more.setVisible(False)
        self.head.addWidget(self.more)

    def _fold(self, on):
        self.more_btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.more.setVisible(on)

    def keys(self):
        return [p.key for p in self.plots]

    def set_keys(self, keys):
        valid = [g[0] for g in GRAPH_KEYS]
        for cb, k in zip(self.combos, keys):
            if k in valid:
                cb.setCurrentIndex(valid.index(k))

    def _set(self, plot, cb):
        plot.key = cb.currentData()
        cb.setToolTip(GRAPH_TIPS.get(plot.key, ''))
        plot.update()
        self.sync_rows()
        if hasattr(self.win, 'reject_panel'):
            self.win.reject_panel.update_ranges()
            self.win.graphs_changed()

    def refresh(self):
        for p in self.plots:
            p.update()

    def sync_rows(self):
        if not hasattr(self.win, 'reject_panel'):
            return
        for r in self.rows:
            r.bind()

    def sync(self):
        """Buttons, mode-dependent options, short result and the slider rows."""
        rp = self.win.reject_panel
        if not hasattr(self, 'find_btn'):
            return
        self.mark_btn.setEnabled(rp.apply.isEnabled())
        using = self.win.using_text()
        txt = rp.pass_label.text()
        lines = [t for t in (txt, using) if t]
        ng = len(self.win._groups())
        if ng > 1:
            lines.append('Note: this folder has %d groups (different filters, exposures or sizes). The limits '
                         'apply to all of them - open each filter on its own for the best results.' % ng)
        self.status.setText('\n'.join(lines))
        self.sync_rows()


# ------------------------------- header panel ------------------------------

class HeaderPanel(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search keywords or values')
        self.search.setClearButtonEnabled(True)
        self.only = QCheckBox('Only changed')
        self.only.setToolTip('Show only keywords whose value differs from the previous frame')
        row.addWidget(self.search, 1)
        row.addWidget(self.only)
        lay.addLayout(row)
        self.info = QLabel('')
        self.info.setWordWrap(True)
        self.info.setStyleSheet('color: %s' % C('note_warn').name())
        lay.addWidget(self.info)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Keyword', 'Value', 'Comment'])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(1, 180)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(20)
        lay.addWidget(self.table, 1)
        self.search.textChanged.connect(self._fill)
        self.only.toggled.connect(self._fill)
        self.header = []
        self.prev = {}

    def show_header(self, header, prev_header, note=''):
        self.header = header or []
        self.prev = hdr_dict(prev_header) if prev_header else {}
        self.info.setText(note)
        self.info.setVisible(bool(note))
        self._fill()

    def restyle(self):
        self.info.setStyleSheet('color: %s' % C('note_warn').name())
        self._fill()

    def _fill(self):
        q = self.search.text().strip().lower()
        only = self.only.isChecked()
        rows = []
        for key, val, com in self.header:
            sval = '' if val is None else (('T' if val else 'F') if isinstance(val, bool) else str(val))
            changed = bool(self.prev) and key and key not in ('COMMENT', 'HISTORY') and \
                str(self.prev.get(key, '\0')) != str(val if not isinstance(val, bool) else val)
            if only and not changed:
                continue
            if q and q not in (key or '').lower() and q not in sval.lower() and q not in (com or '').lower():
                continue
            rows.append((key, sval, com or '', changed))
        self.table.setUpdatesEnabled(False)
        self.table.setRowCount(len(rows))
        hl = QBrush(C('header_hl'))
        for r, (k, v, c, ch) in enumerate(rows):
            for col, t in enumerate((k, v, c)):
                it = QTableWidgetItem(t)
                if ch:
                    it.setBackground(hl)
                if col == 1:
                    it.setToolTip(t)
                self.table.setItem(r, col, it)
        self.table.setUpdatesEnabled(True)


# ------------------------------- auto reject panel -------------------------

class RejectPanel(QWidget):
    """Two ways to pick frames: automatic (sigma tests) or manual limits."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        self.m_auto = QRadioButton('Automatic')
        self.m_auto.setToolTip('Flags frames that stand out from the rest of the session')
        self.m_manual = QRadioButton('Manual limits')
        self.m_manual.setToolTip('You set the limits with sliders and see the result as you drag')
        self.m_auto.setChecked(True)
        row.addWidget(self.m_auto)
        row.addWidget(self.m_manual)
        row.addStretch(1)
        lay.addLayout(row)

        # ---------------- automatic ----------------
        self.auto_box = QWidget()
        al = QVBoxLayout(self.auto_box)
        al.setContentsMargins(0, 0, 0, 0)
        intro = QLabel('Flags frames that stand out from the rest of the session. '
                       'Nothing is marked or moved until you say so.')
        intro.setWordWrap(True)
        al.addWidget(intro)
        grid = QGridLayout()
        grid.addWidget(QLabel('<b>Test</b>'), 0, 0)
        grid.addWidget(QLabel('<b>Sigma</b>'), 0, 1)
        self.rules = {}
        tips = {'fwhm': 'Stars bigger than usual: poor seeing, focus drift, cloud',
                'ecc': 'Elongated stars: tracking, wind, guiding trouble',
                'stars': 'Fewer stars than usual: cloud, haze, dew',
                'background': 'Brighter sky than usual: moon, dawn, cloud lit from below',
                'snr': 'Less signal against the noise than usual'}
        for r, (key, label, direction, sig, minrel, on) in enumerate(REJECT_RULES, 1):
            cb = QCheckBox(label + (' too high' if direction > 0 else ' too low'))
            cb.setChecked(on)
            cb.setToolTip(tips.get(key, ''))
            sp = QDoubleSpinBox()
            sp.setRange(1.0, 6.0)
            sp.setSingleStep(0.25)
            sp.setDecimals(2)
            sp.setValue(sig)
            sp.setToolTip('How far from the session median, in robust standard deviations. '
                          'Lower = stricter.')
            grid.addWidget(cb, r, 0)
            grid.addWidget(sp, r, 1)
            self.rules[key] = (cb, sp)
            cb.toggled.connect(win.graphs_refresh)
            sp.valueChanged.connect(win.graphs_refresh)
        self.trails = QCheckBox('Frames with trails')
        self.trails.setToolTip('Satellite and aircraft trails. Usually removed by pixel rejection '
                               'when stacking, so this is off by default.')
        grid.addWidget(self.trails, len(REJECT_RULES) + 1, 0)
        al.addLayout(grid)
        b1 = QPushButton('Find frames to reject')
        b1.clicked.connect(lambda: win.run_auto_reject())
        al.addWidget(b1)
        self.auto_run = QCheckBox('Run as soon as measuring finishes')
        self.auto_run.setToolTip('Flags frames by itself once every frame is measured. '
                                 'Nothing is marked or moved until you say so.')
        al.addWidget(self.auto_run)
        lay.addWidget(self.auto_box)

        # ---------------- manual limits ----------------
        self.manual_box = QWidget()
        ml = QVBoxLayout(self.manual_box)
        ml.setContentsMargins(0, 0, 0, 0)
        intro2 = QLabel('Every slider starts with all frames passing. <b>Drag right to be stricter</b> - '
                        'the worst frames for that measurement go first. Frames that fail turn orange.')
        intro2.setWordWrap(True)
        ml.addWidget(intro2)
        self.limits = {}
        g2 = QGridLayout()
        g2.setColumnStretch(1, 1)
        tips2 = {'fwhm': 'Leaves out frames with stars bigger than this',
                 'ecc': 'Leaves out frames with stars more elongated than this (0 = round)',
                 'stars': 'Leaves out frames with fewer stars than this',
                 'background': 'Leaves out frames with a brighter sky than this',
                 'snr': 'Leaves out frames with a lower SNR than this'}
        for r, (key, kind) in enumerate(LIMITS):
            cb = QCheckBox()
            cb.setToolTip('Use this limit')
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 1000)
            sl.setValue(0)
            sl.setToolTip(tips2[key] + '.\nApplies to every loaded frame, so with several filters '
                          'loaded one limit covers them all.')
            lab = QLabel('')
            lab.setMinimumWidth(150)
            g2.addWidget(cb, r, 0)
            g2.addWidget(sl, r, 1)
            g2.addWidget(lab, r, 2)
            self.limits[key] = {'cb': cb, 'sl': sl, 'lab': lab, 'kind': kind,
                                'lo': None, 'hi': None, 'value': None, 'unit': key}
            sl.valueChanged.connect(lambda _v, k=key: self._slider_moved(k))
            cb.toggled.connect(lambda _v, k=key: self._limit_toggled(k))
        ml.addLayout(g2)
        # keep the best X% for one measurement (as in Siril, APP and DeepSkyStacker)
        bl = QGridLayout()
        bl.setColumnStretch(2, 1)
        self.best_cb = QCheckBox('Keep only the best')
        self.best_cb.setToolTip('Keeps the best part of the session for one measurement and flags the rest')
        self.best_key = QComboBox()
        for key, kind in LIMITS:
            self.best_key.addItem({'fwhm': 'FWHM', 'ecc': 'Eccentricity', 'stars': 'Star count',
                                   'background': 'Background', 'snr': 'SNR'}[key], key)
        self.best_sl = QSlider(Qt.Horizontal)
        self.best_sl.setRange(0, 90)
        self.best_sl.setValue(0)
        self.best_sl.setToolTip('Drag right to keep fewer frames - the worst for the chosen measurement go first')
        self.best_lab = QLabel('')
        self.best_lab.setMinimumWidth(150)
        bl.addWidget(self.best_cb, 0, 0)
        bl.addWidget(self.best_key, 0, 1)
        bl.addWidget(self.best_sl, 1, 0, 1, 2)
        bl.addWidget(self.best_lab, 1, 2)
        ml.addLayout(bl)
        self.best_cut = None
        self.best_sl.valueChanged.connect(self._best_moved)
        self.best_cb.toggled.connect(self._best_changed)
        self.best_key.currentIndexChanged.connect(self._best_changed)
        self._best_label()
        self.pass_label = QLabel('')
        f = self.pass_label.font()
        f.setBold(True)
        self.pass_label.setFont(f)
        ml.addWidget(self.pass_label)
        b2 = QPushButton('Reset the sliders')
        b2.clicked.connect(self.reset_limits)
        ml.addWidget(b2)
        lay.addWidget(self.manual_box)

        # ---------------- shared ----------------
        self.result = QLabel('')
        self.result.setWordWrap(True)
        lay.addWidget(self.result)
        row = QHBoxLayout()
        self.apply = QPushButton('Mark them Bad')
        self.apply.setEnabled(False)
        self.apply.clicked.connect(win.apply_auto_reject)
        self.rest_good = QCheckBox('and the rest Good')
        self.rest_good.setToolTip('Also mark every frame that passed as Good')
        row.addWidget(self.apply)
        row.addWidget(self.rest_good)
        row.addStretch(1)
        lay.addLayout(row)
        self.clear_btn = QPushButton('Clear all marks')
        self.clear_btn.clicked.connect(win.clear_marks)
        lay.addWidget(self.clear_btn)
        lay.addStretch(1)
        self.m_auto.toggled.connect(self._mode_changed)
        # one way of working now: real limits, which "Suggest limits" can set
        self.m_manual.setChecked(True)
        self.m_auto.setVisible(False)
        self.m_manual.setVisible(False)
        self.auto_run.setText('Suggest limits as soon as measuring finishes')
        self.auto_run.setToolTip('Sets the limits by itself once every frame is measured. '
                                 'Nothing is marked or moved until you say so.')
        self.trails.setText('Flag frames with trails')
        self.trails.setToolTip('Satellite and aircraft trails. Usually removed by pixel rejection when '
                               'stacking, so this is off by default.')
        self.trails.toggled.connect(lambda _v: win.apply_manual_limits())
        self.sigma_sp = QDoubleSpinBox()
        self.sigma_sp.setRange(1.0, 6.0)
        self.sigma_sp.setSingleStep(0.25)
        self.sigma_sp.setDecimals(2)
        self.sigma_sp.setValue(2.0)
        self.sigma_sp.setToolTip('How unusual a frame must be before Suggest limits draws the line '
                                 '(sigma = the usual spread of the session).\nLower = stricter. '
                                 'Background and SNR get 0.5 more.')
        # FWHM, eccentricity, star count and background in use from the start
        for k in ('fwhm', 'ecc', 'stars', 'background'):
            self.limits[k]['cb'].blockSignals(True)
            self.limits[k]['cb'].setChecked(True)
            self.limits[k]['cb'].blockSignals(False)
        self._mode_changed()

    # ---------------- mode ----------------
    def is_manual(self):
        return self.m_manual.isChecked()

    def _mode_changed(self, *_a):
        manual = self.is_manual()
        self.auto_box.setVisible(not manual)
        self.manual_box.setVisible(manual)
        self.result.setText('')
        self.apply.setEnabled(False)
        if not hasattr(self.win, 'graph_panel') or not hasattr(self.win, 'reject_panel'):
            return                       # still building the window
        if manual:
            self.update_ranges()
            self.win.apply_manual_limits()
        else:
            self.win.clear_reasons()
            self.win.schedule_auto_reject()      # Automatic is live too

    # ---------------- manual limits ----------------
    def _key_for(self, key):
        """FWHM is limited in the unit its graph shows: arcseconds only when a
        graph shows FWHM in arcseconds (and every frame has a pixel scale)."""
        if key != 'fwhm':
            return key
        shown = self.win.graph_panel.keys() if hasattr(self.win, 'graph_panel') else []
        if 'fwhm_arcsec' in shown and 'fwhm' not in shown:
            frs = [f for f in self.win.frames if f.status == 'ok' and f.an.stats.get('fwhm')]
            if frs and all(f.an.stats.get('fwhm_arcsec') for f in frs):
                return 'fwhm_arcsec'
        return 'fwhm'

    def update_ranges(self):
        """Slider ends = the session's best and worst values. Keeps any limit already set."""
        for key, d in self.limits.items():
            vk = self._key_for(key)
            if vk != d['unit'] and d['value'] is not None and key == 'fwhm':
                # graph switched between pixels and arcseconds: keep the same limit
                ratio = self._fwhm_ratio()
                if ratio:
                    d['value'] = d['value'] * ratio if vk == 'fwhm_arcsec' else d['value'] / ratio
                else:
                    d['value'] = None
            d['unit'] = vk
            vals = [self.win.model.value(f, vk) for f in self.win.frames if f.status == 'ok']
            vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
            if not vals:
                d['lo'] = d['hi'] = None
                d['sl'].setEnabled(False)
                d['lab'].setText(LIMIT_LABEL[key] + ': no data')
                continue
            self._best_label()
            d['lo'], d['hi'] = min(vals), max(vals)
            d['sl'].setEnabled(d['hi'] > d['lo'])
            d['sl'].blockSignals(True)
            if d['value'] is None:
                d['sl'].setValue(0)
            else:
                d['sl'].setValue(self._pos_for(d, d['value']))
            d['sl'].blockSignals(False)
            self._label(key)

    @staticmethod
    def _pos_for(d, value):
        lo, hi = d['lo'], d['hi']
        if hi <= lo:
            return 0
        t = (hi - value) / (hi - lo) if d['kind'] == 'max' else (value - lo) / (hi - lo)
        return int(round(min(max(t, 0.0), 1.0) * 1000))

    @staticmethod
    def _value_at(d, pos):
        """Slider at 0 = everything passes; at 1000 = only the best frame passes.
        'max' limits (high is bad) run from the worst value down to the best,
        'min' limits (low is bad) run from the worst value up to the best."""
        t = pos / 1000.0
        lo, hi = d['lo'], d['hi']
        if d['kind'] == 'max':
            return hi - t * (hi - lo)
        return lo + t * (hi - lo)

    def _label(self, key):
        d = self.limits[key]
        if d['lo'] is None:
            return
        v = self._value_at(d, d['sl'].value())
        unit = {'fwhm': ' px', 'fwhm_arcsec': '"'}.get(d['unit'], '')
        name = LIMIT_LABEL[key]
        if key == 'stars':
            txt = '%s %d' % (name, int(math.ceil(v)))
        elif key == 'background':
            txt = '%s %.0f' % (name, v)
        else:
            txt = '%s %.2f%s' % (name, v, unit)
        d['lab'].setText(txt)
        d['lab'].setEnabled(d['cb'].isChecked())

    def _slider_moved(self, key):
        d = self.limits[key]
        if d['lo'] is None:
            return
        d['value'] = self._value_at(d, d['sl'].value())
        if not d['cb'].isChecked() and d['sl'].value() > 0:
            d['cb'].blockSignals(True)
            d['cb'].setChecked(True)
            d['cb'].blockSignals(False)
        self._label(key)
        self.win.apply_manual_limits()

    def _limit_toggled(self, key):
        self._label(key)
        self.win.apply_manual_limits()

    def _best_label(self):
        drop = self.best_sl.value()
        n = sum(1 for f in self.win.frames if f.status == 'ok')
        keep_n = max(1, int(math.ceil(n * (100 - drop) / 100.0 - 1e-9))) if n else 0
        self.best_lab.setText('Keep best %d%%%s' % (100 - drop, ('  (%d of %d)' % (keep_n, n)) if n else ''))
        self.best_lab.setEnabled(self.best_cb.isChecked())

    def _best_moved(self, _v):
        if not self.best_cb.isChecked() and self.best_sl.value() > 0:
            self.best_cb.blockSignals(True)
            self.best_cb.setChecked(True)
            self.best_cb.blockSignals(False)
        self._best_changed()

    def _best_changed(self, *_a):
        self._best_label()
        self.win.apply_manual_limits()

    def best_setting(self):
        if not self.best_cb.isChecked() or self.best_sl.value() <= 0:
            return None
        key = self.best_key.currentData()
        kind = dict(LIMITS)[key]
        vk = self._key_for(key)
        return key, vk, kind, self.best_sl.value()

    def reset_limits(self):
        self.suggested = False
        for key, d in self.limits.items():
            d['value'] = None
            d['sl'].blockSignals(True)
            d['sl'].setValue(0)
            d['sl'].blockSignals(False)
            # the tick boxes say which measurements are used: leave them alone
            self._label(key)
        self.best_sl.blockSignals(True)
        self.best_sl.setValue(0)
        self.best_sl.blockSignals(False)
        self.best_cb.blockSignals(True)
        self.best_cb.setChecked(False)
        self.best_cb.blockSignals(False)
        self._best_label()
        self.win.apply_manual_limits()

    def active_limits(self):
        """[(key, value_key, kind, limit)] for the ticked sliders whose
        measurement is shown on a graph."""
        shown = self.win.shown_families() if hasattr(self.win, 'graph_panel') else set(self.limits)
        out = []
        for key, d in self.limits.items():
            if d['cb'].isChecked() and d['lo'] is not None and key in shown:
                out.append((key, d['unit'], d['kind'], self._value_at(d, d['sl'].value())))
        return out

    def _fwhm_ratio(self):
        """Arcseconds per pixel, from frames that have both FWHM values."""
        r = [f.an.stats['fwhm_arcsec'] / f.an.stats['fwhm'] for f in self.win.frames
             if f.status == 'ok' and f.an.stats.get('fwhm') and f.an.stats.get('fwhm_arcsec')]
        return float(np.median(r)) if r else None

    def limit_for_graph(self, graph_key):
        """The strictest manual cut-off for this graph's measurement, or None."""
        found = []
        for key, vk, kind, lim in self.active_limits():
            if vk == graph_key:
                found.append((kind, lim))
            elif key == 'fwhm' and graph_key in ('fwhm', 'fwhm_arcsec'):
                ratio = self._fwhm_ratio()
                if ratio:
                    found.append((kind, lim / ratio if graph_key == 'fwhm' else lim * ratio))
        b = self.best_setting()
        if b is not None and self.best_cut is not None and self.best_cut[0] == graph_key \
                and self.best_cut[1] is not None and b[1] == graph_key:
            found.append((b[2], self.best_cut[1]))
        if not found:
            return None
        kind = found[0][0]
        vals = [v for _k, v in found]
        return min(vals) if kind == 'max' else max(vals)

    # ---------------- settings ----------------
    def get_rules(self):
        return {k: (cb.isChecked(), sp.value()) for k, (cb, sp) in self.rules.items()}

    def save(self, st):
        for k, (cb, sp) in self.rules.items():
            st.setValue('reject/%s_on' % k, cb.isChecked())
            st.setValue('reject/%s_sig' % k, sp.value())
        st.setValue('reject/trails', self.trails.isChecked())
        st.setValue('reject/auto_run', self.auto_run.isChecked())
        st.setValue('reject/sigma', self.sigma_sp.value())
        for k, d in self.limits.items():
            st.setValue('limits/%s_on' % k, d['cb'].isChecked())

    def load(self, st):
        for k, (cb, sp) in self.rules.items():
            v = st.value('reject/%s_on' % k)
            if v is not None:
                cb.setChecked(str(v).lower() in ('true', '1'))
            v = st.value('reject/%s_sig' % k)
            if v is not None:
                try:
                    sp.setValue(float(v))
                except Exception:
                    pass
        v = st.value('reject/trails')
        if v is not None:
            self.trails.setChecked(str(v).lower() in ('true', '1'))
        v = st.value('reject/auto_run')
        if v is not None:
            self.auto_run.setChecked(str(v).lower() in ('true', '1'))
        v = st.value('reject/sigma')
        if v is not None:
            try:
                self.sigma_sp.setValue(float(v))
            except Exception:
                pass
        for k, d in self.limits.items():
            v = st.value('limits/%s_on' % k)
            if v is not None:
                d['cb'].blockSignals(True)
                d['cb'].setChecked(str(v).lower() in ('true', '1'))
                d['cb'].blockSignals(False)
                self._label(k)


# key, kind: 'max' = a high value is bad, 'min' = a low value is bad
LIMITS = [('fwhm', 'max'), ('ecc', 'max'), ('stars', 'min'), ('background', 'max'), ('snr', 'min')]
LIMIT_LABEL = {'fwhm': 'Max FWHM', 'ecc': 'Max eccentricity', 'stars': 'Min stars',
               'background': 'Max background', 'snr': 'Min SNR'}

class ExportDialog(QDialog):
    def __init__(self, parent, n_sel, n_good, n_all, settings):
        super().__init__(parent)
        self.setWindowTitle('Make a GIF or video')
        lay = QVBoxLayout(self)
        g1 = QGroupBox('Frames')
        v = QVBoxLayout(g1)
        self.r_sel = QRadioButton('Selected rows in the table (%d)' % n_sel)
        self.r_good = QRadioButton('Frames marked Good (%d)' % n_good)
        self.r_notbad = QRadioButton('All except Bad')
        self.r_all = QRadioButton('All frames (%d)' % n_all)
        for r in (self.r_sel, self.r_good, self.r_notbad, self.r_all):
            v.addWidget(r)
        self.r_sel.setEnabled(n_sel > 1)
        self.r_good.setEnabled(n_good > 1)
        (self.r_sel if n_sel > 1 else self.r_notbad).setChecked(True)
        lay.addWidget(g1)
        form = QFormLayout()
        self.fmt = QComboBox()
        self.fmt.addItems(['Animated GIF', 'MP4 video'])
        self.width = QSpinBox()
        self.width.setRange(200, 8000)
        self.width.setSingleStep(100)
        self.width.setValue(int(settings.value('export/width', 900)))
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.5, 60)
        self.fps.setValue(float(settings.value('export/fps', 4)))
        self.align = QCheckBox('Align frames to the reference')
        self.align.setChecked(True)
        self.labels = QCheckBox('Show file name and frame number')
        self.labels.setChecked(str(settings.value('export/labels', 'true')).lower() == 'true')
        self.crop = QCheckBox('Crop to the current view (what is on screen)')
        self.crop.setChecked(False)
        form.addRow('Format', self.fmt)
        form.addRow('Width (px)', self.width)
        form.addRow('Frames per second', self.fps)
        form.addRow('', self.align)
        form.addRow('', self.labels)
        form.addRow('', self.crop)
        lay.addLayout(form)
        note = QLabel('Uses the stretch you are viewing with. Large GIFs can be slow to make.')
        note.setWordWrap(True)
        note.setStyleSheet('color: %s' % C('note_dim').name())
        lay.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText('Make it')
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def which(self):
        if self.r_sel.isChecked():
            return 'selected'
        if self.r_good.isChecked():
            return 'good'
        if self.r_all.isChecked():
            return 'all'
        return 'notbad'


class MoveDialog(QDialog):
    def __init__(self, parent, n_good, n_bad, n_unmarked, n_ser, settings):
        super().__init__(parent)
        self.setWindowTitle('Sort into folders')
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel('%d Good, %d Bad, %d not marked.' % (n_good, n_bad, n_unmarked)))
        form = QFormLayout()
        self.good_name = QLineEdit(settings.value('move/good', 'Good'))
        self.bad_name = QLineEdit(settings.value('move/bad', 'Bad'))
        form.addRow('Good folder name', self.good_name)
        form.addRow('Bad folder name', self.bad_name)
        lay.addLayout(form)
        g = QGroupBox('Where')
        v = QVBoxLayout(g)
        self.w_own = QRadioButton("Inside each file's own folder")
        self.w_pick = QRadioButton('Inside this folder:')
        self.w_own.setChecked(True)
        row = QHBoxLayout()
        self.base = QLineEdit(settings.value('move/base', ''))
        b = QPushButton('Browse...')
        b.clicked.connect(self._browse)
        row.addWidget(self.base, 1)
        row.addWidget(b)
        v.addWidget(self.w_own)
        v.addWidget(self.w_pick)
        v.addLayout(row)
        lay.addWidget(g)
        g2 = QGroupBox('How')
        v2 = QVBoxLayout(g2)
        self.move = QRadioButton('Move the files (can be undone)')
        self.copy = QRadioButton('Copy the files (originals stay where they are)')
        (self.copy if settings.value('move/mode', 'move') == 'copy' else self.move).setChecked(True)
        v2.addWidget(self.move)
        v2.addWidget(self.copy)
        self.unmarked_good = QCheckBox('Treat frames that are not marked as Good')
        self.unmarked_good.setChecked(str(settings.value('move/unmarked_good', 'false')).lower() == 'true')
        v2.addWidget(self.unmarked_good)
        lay.addWidget(g2)
        if n_ser:
            n = QLabel('SER video frames cannot be moved one by one. For those, new SER files '
                       'holding just the Good and the Bad frames are written into the folders; '
                       'the original SER is left alone.')
            n.setWordWrap(True)
            n.setStyleSheet('color: %s' % C('note_warn').name())
            lay.addWidget(n)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText('Sort them')
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Folder for Good and Bad', self.base.text())
        if d:
            self.base.setText(d)
            self.w_pick.setChecked(True)

    def _ok(self):
        bad_chars = set('<>:"/\\|?*')
        for e in (self.good_name, self.bad_name):
            t = e.text().strip()
            if not t or set(t) & bad_chars:
                QMessageBox.warning(self, APP_NAME, 'Folder names cannot be empty or contain  < > : " / \\ | ? *')
                return
        if self.good_name.text().strip().lower() == self.bad_name.text().strip().lower():
            QMessageBox.warning(self, APP_NAME, 'The Good and Bad folders need different names.')
            return
        if self.w_pick.isChecked() and not os.path.isdir(self.base.text()):
            QMessageBox.warning(self, APP_NAME, 'Please choose a folder that exists.')
            return
        self.accept()


HELP_HTML = '''
<h2>AS Blink %s</h2>
<p>A fast blink comparator for checking your subs before stacking.</p>
<h3>Getting started</h3>
<p>Open a folder of subs (Ctrl+O) or add files (Ctrl+Shift+O). You can also drag files or a folder
onto the window. Every frame is measured in the background; the table and graphs fill in as it goes.</p>
<h3>Videos (SER, AVI, MP4, MOV)</h3>
<p>Every frame of a video becomes a row in the table, so you can step through it one frame at a time with the
<b>Left / Right</b> keys, or play it with <b>Space</b> at any speed from 0.5 to 30 frames a second. Video
frames are only read for viewing and their brightness, not measured for stars, so even a long video is ready
to play in seconds (to measure stars as well, tick the option in <b>More options</b>). Alignment, the
magnifier, marking and GIF / MP4 of selected frames all work on video frames. <b>Sort into folders</b> writes
new SER files for SER frames; frames inside AVI, MP4 and MOV files are left out of sorting.</p>
<h3>Keys</h3>
<table cellpadding="3">
<tr><td><b>Ctrl+O</b></td><td>Open a folder of subs</td></tr>
<tr><td><b>Ctrl+Shift+O</b></td><td>Add files</td></tr>
<tr><td><b>Space</b></td><td>Play / pause the blink</td></tr>
<tr><td><b>Left / Right</b></td><td>Previous / next frame</td></tr>
<tr><td><b>Home / End</b></td><td>First / last frame</td></tr>
<tr><td><b>G</b></td><td>Mark Good (and move on)</td></tr>
<tr><td><b>B</b></td><td>Mark Bad (and move on)</td></tr>
<tr><td><b>U</b></td><td>Clear the mark</td></tr>
<tr><td><b>R</b></td><td>Make this frame the reference (alignment and the shared stretch)</td></tr>
<tr><td><b>A</b></td><td>Align on / off</td></tr>
<tr><td><b>M</b></td><td>Magnifier on / off</td></tr>
<tr><td><b>T</b></td><td>Show trails on / off</td></tr>
<tr><td><b>I</b></td><td>Info overlay on / off</td></tr>
<tr><td><b>F</b> or double-click</td><td>Fit to window</td></tr>
<tr><td><b>1</b></td><td>100%% (one screen pixel per image pixel)</td></tr>
<tr><td><b>+ / -</b>, mouse wheel</td><td>Zoom</td></tr>
<tr><td><b>Delete</b></td><td>Remove the selected rows from the list (the files on disk are not touched)</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>Undo the last move into folders</td></tr>
<tr><td><b>F1</b></td><td>This help</td></tr>
</table>
<h3>Layout</h3>
<p>The window size and position, the dividers, the table's column widths and the four graph choices are
remembered when you close AS Blink. Tick <b>Lock layout</b> so the dividers and columns cannot be dragged by
accident, and use <b>Reset layout</b> to put everything back to how it starts.</p>
<h3>Stretch</h3>
<p><b>Same for all</b> uses one stretch, taken from the reference frame, for every frame, so cloud,
moon or dawn shows up as a brighter frame. <b>Each frame</b> stretches every frame on its own,
which is better for comparing detail. <b>Linked</b> uses one stretch for all colour channels;
unlinked removes the colour cast of raw colour subs.</p>
<h3>Colour (OSC) subs</h3>
<p>The Bayer pattern is read from the header (BAYERPAT, ROWORDER, XBAYROFF, YBAYROFF, or the XISF
colour filter array). AS Blink also checks the pattern against the data: if the header puts the green
pixels on the wrong rows it says so in the header tab and uses the corrected pattern. You can set the
pattern yourself with the Debayer list.</p>
<h3>Statistics</h3>
<p><b>FWHM</b> and <b>eccentricity</b> are medians over up to 400 unsaturated stars, measured at full
resolution with Gaussian-weighted moments. Use them to compare frames within a session; other programs
fit stars differently, so their numbers can differ a little. <b>SNR</b> is the mean signal above the
background divided by the noise. <b>Background</b> is on a 0-65535 scale for every file type.</p>
<h3>Rejecting frames (Graphs tab)</h3>
<p>Under each graph is a tick box and a slider. The slider is the actual limit for that measurement, for
example "Max FWHM 3.09 px", shown as a red line on the graph. <b>Dragging right is always stricter</b>: the
worst frames for that measurement fail first, whether a high value is bad (FWHM, eccentricity, background)
or a low one (star count, SNR). Frames past a limit turn orange.</p>
<p><b>Suggest limits</b> sets each ticked slider for you, to tonight's "odd ones out": frames that are worse
than the session's typical value by more than twice its usual spread (sigma), or by a minimum gap if that is
bigger (FWHM 10%%, star count 25%%, background 15%%, SNR 20%%, eccentricity 0.08). You can then drag any
slider to adjust. The strictness (sigma) is in <b>More options</b>.</p>
<p>Only measurements that are <b>shown on a graph and ticked</b> are used. To judge on FWHM and star count
only, untick the other graphs (or switch a graph to another measurement). The line under the buttons says what
is in use, for example "Using: FWHM, Star count". The limits apply to every frame in the folder, so keep each
filter in its own folder; AS Blink warns you if a folder mixes filters, exposures or frame sizes.</p>
<p><b>More options</b> also holds: flagging frames with trails; suggesting limits as soon as measuring
finishes; <b>Keep only the best</b> - the best part of the session for one measurement, like Siril's
percentage filter and the "best X%%" option in APP and DeepSkyStacker; resetting the sliders; and clearing all
marks.</p>
<p>Frames are only flagged (orange). <b>Mark them Bad</b> marks them, and nothing is moved until you use
<b>Sort into folders</b>.</p>
<h3>Graph colours</h3>
<p>Green = marked Good, red = marked Bad, solid orange = flagged for the measurement on that graph,
blue with an orange ring = flagged for a different measurement, blue = not flagged. Hover over a dot to see
why. If a limit is off the top or bottom of a graph, its value is written under the graph instead.</p>
<h3>Alignment</h3>
<p>The reference is chosen when measuring finishes: the frame with the most stars among the sharper half, leaving out any frame with a trail, flagged for rejection or marked Bad. Press <b>R</b> to choose another.</p>
<p>Shifts each frame to line up with the reference, including a 180 degree turn after a meridian flip.
It corrects drift between frames, not field rotation.</p>
<h3>Something not working?</h3>
<p>AS Blink keeps a log of what it does: the files it opens (names only, never the full path), how each one
was measured, any errors, and sorting and exports. Use <b>Log &gt; Save log for support...</b> to save it as one
text file and send it to AstroShed with a few words on what you were doing. <b>Log &gt; Open log folder</b>
shows where it is kept.</p>
<h3>Licence</h3>
<p>AS Blink is freeware, copyright Stewart Oliver, AstroShed. You may use it free of charge and share it
unchanged, but not sell it or share changed versions. It comes with no warranty: keep backups of your data.
The full licence is at the top of the AS_Blink.py file and in LICENCE.txt.</p>
<p>AstroShed - astroshed.co.uk - buymeacoffee.com/astroshed</p>
''' % APP_VERSION

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('%s %s' % (APP_NAME, APP_VERSION))
        self.settings = QSettings('AstroShed', 'AS Blink')
        self.frames = []
        self._pending = 0
        self._rows = {}
        self.by_uid = {}
        self.gen = 0
        self.cur = None             # uid of the frame on screen
        self.ref = None             # uid of the reference frame
        self.auto_ref = True        # True until the user picks one
        self.ref_final = False      # False while the reference is only a stand-in
        self.reject_summary = ''    # short result shown on the Graphs tab
        self.last_move = []         # [(new_path, old_path, uid)]
        self.busy = set()
        self.pending_show = None
        self.sig = Signals()
        self.sig.analysed.connect(self._on_analysed)
        self.sig.preview.connect(self._on_preview)
        self.sig.full.connect(self._on_full)
        self.sig.failed.connect(self._on_failed)
        ncpu = os.cpu_count() or 2
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(max(1, min(4, ncpu - 1)))
        self.dpool = QThreadPool()
        self.dpool.setMaxThreadCount(2)
        mem_mb = int(self.settings.value('cache_mb', 2000))
        self.pcache = LRU(mem_mb * 1024 * 1024 * 6 // 10)    # previews
        self.fcache = LRU(mem_mb * 1024 * 1024 * 3 // 10)    # full resolution
        self.rcache = LRU(mem_mb * 1024 * 1024 * 1 // 10)    # stretched 8-bit
        self.stretch = Stretch()
        self.opts = {'debayer': 'auto', 'check_pattern': True, 'trails': True,
                     'preview_max': int(self.settings.value('preview_max', 1600)),
                     'video_stars': str(self.settings.value('video_stars', 'false')).lower() == 'true'}
        # timers first: restoring the saved settings can fire slots that use them
        self.auto_timer = QTimer(self)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.timeout.connect(self._auto_rerun)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.graph_timer = QTimer(self)
        self.graph_timer.setSingleShot(True)
        self.graph_timer.timeout.connect(self._refresh_slow)
        self._build_ui()
        self._refresh_icons()
        self._load_settings()
        self.setAcceptDrops(True)

    # ===================================================================== UI
    def _build_ui(self):
        tb = QToolBar('Main')
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        st = self.style()

        self._std_icons = []

        def act(text, slot, icon=None, tip='', key=None):
            a = QAction(text, self)
            if icon is not None:
                a.setIcon(st.standardIcon(icon))
                self._std_icons.append((a, icon))
            a.setToolTip(tip or text)
            if key:
                a.setShortcut(QKeySequence(key))
            a.triggered.connect(slot)
            return a
        tb.addAction(act('Open folder', self.open_folder, QStyle.SP_DirOpenIcon, 'Open a folder of subs (Ctrl+O)', 'Ctrl+O'))
        tb.addAction(act('Add files', self.add_files, QStyle.SP_FileIcon, 'Add files (Ctrl+Shift+O)', 'Ctrl+Shift+O'))
        tb.addAction(act('Clear', self.clear_all, QStyle.SP_DialogResetButton, 'Clear the list'))
        tb.addSeparator()
        self.a_prev = act('Prev', self.prev_frame, None, 'Previous frame (Left)')
        self.a_play = act('Play', self.toggle_play, None, 'Play / pause (Space)')
        self.a_next = act('Next', self.next_frame, None, 'Next frame (Right)')
        tb.addAction(self.a_prev)
        tb.addAction(self.a_play)
        tb.addAction(self.a_next)
        tb.addWidget(QLabel(' fps '))
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.5, 30)
        self.fps.setValue(4)
        self.fps.setSingleStep(0.5)
        self.fps.setToolTip('Blink speed, frames per second')
        self.fps.valueChanged.connect(self._fps_changed)
        tb.addWidget(self.fps)
        self.skip_bad = QCheckBox('Skip Bad')
        self.skip_bad.setToolTip('Leave out frames marked Bad while blinking')
        self.skip_bad.setChecked(True)
        tb.addWidget(self.skip_bad)
        self.sel_only = QCheckBox('Selected only')
        self.sel_only.setToolTip('Blink only the rows selected in the table (select two or more)')
        tb.addWidget(self.sel_only)
        tb.addSeparator()
        tb.addAction(act('GIF / video', self.export_anim, QStyle.SP_DriveDVDIcon, 'Make an animated GIF or MP4'))
        tb.addAction(act('Sort into folders', self.move_dialog, QStyle.SP_DirLinkIcon,
                         'Put Good and Bad frames into separate folders'))
        self.a_undo = act('Undo move', self.undo_move, QStyle.SP_ArrowBack, 'Undo the last move (Ctrl+Z)', 'Ctrl+Z')
        self.a_undo.setEnabled(False)
        tb.addAction(self.a_undo)
        tb.addAction(act('CSV', self.export_csv, QStyle.SP_DialogSaveButton, 'Save the statistics table as CSV'))
        tb.addAction(act('Help', self.show_help, QStyle.SP_DialogHelpButton, 'Keys and notes (F1)', 'F1'))
        self.log_btn = QToolButton()
        self.log_btn.setText('Log')
        self.log_btn.setToolTip('Save the log to send to AstroShed if something goes wrong')
        self.log_btn.setPopupMode(QToolButton.InstantPopup)
        self.log_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.log_btn.setIcon(st.standardIcon(QStyle.SP_FileDialogDetailedView))
        self._std_icons.append((self.log_btn, QStyle.SP_FileDialogDetailedView))
        m = QMenu(self.log_btn)
        m.addAction('Save log for support...', self.save_log)
        m.addAction('Open log folder', self.open_log_folder)
        self.log_btn.setMenu(m)
        tb.addWidget(self.log_btn)
        self.addToolBarBreak()
        tb = QToolBar('View')
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        tb.addWidget(QLabel(' Stretch '))
        self.stretch_cb = QComboBox()
        self.stretch_cb.addItems(['Auto - same for all', 'Auto - each frame', 'Linear'])
        self.stretch_cb.setToolTip('Same for all: one stretch from the reference frame, so brightness '
                                   'changes (cloud, moon) show.\nEach frame: every frame stretched on its own.')
        self.stretch_cb.currentIndexChanged.connect(self._stretch_changed)
        tb.addWidget(self.stretch_cb)
        self.linked = QCheckBox('Linked')
        self.linked.setToolTip('One stretch for all colour channels (unlinked removes the colour cast)')
        self.linked.toggled.connect(self._stretch_changed)
        tb.addWidget(self.linked)
        tb.addWidget(QLabel(' Boost '))
        self.boost = QSlider(Qt.Horizontal)
        self.boost.setRange(5, 50)
        self.boost.setValue(25)
        self.boost.setFixedWidth(90)
        self.boost.setToolTip('How bright the stretched background is')
        self.boost.valueChanged.connect(self._stretch_changed)
        tb.addWidget(self.boost)
        tb.addSeparator()
        tb.addWidget(QLabel(' Debayer '))
        self.debayer_cb = QComboBox()
        self.debayer_cb.addItems(['Auto (header)', 'Off (mono)', 'RGGB', 'BGGR', 'GRBG', 'GBRG'])
        self.debayer_cb.setToolTip('Auto reads the Bayer pattern from the file header')
        self.debayer_cb.currentIndexChanged.connect(self._debayer_changed)
        tb.addWidget(self.debayer_cb)
        tb.addSeparator()
        self.align_cb = QCheckBox('Align')
        self.align_cb.setToolTip('Line every frame up with the reference (A)')
        self.align_cb.setChecked(True)
        self.align_cb.toggled.connect(lambda _v: self.show_frame(self.cur))
        tb.addWidget(self.align_cb)
        tb.addAction(act('Set reference', lambda: self.set_reference(), QStyle.SP_DialogApplyButton,
                         'Use this frame as the reference for alignment and the shared stretch (R)'))
        self.trails_cb = QCheckBox('Show trails')
        self.trails_cb.setToolTip('Draws found trails on the picture (T).\nTo reject frames with trails, use '
                                  'Flag frames with trails in More options on the Graphs tab.')
        self.trails_cb.setChecked(True)
        self.trails_cb.toggled.connect(self._view_opts)
        tb.addWidget(self.trails_cb)
        self.loupe_cb = QCheckBox('Magnifier')
        self.loupe_cb.setToolTip('Shows the area under the mouse enlarged, at full resolution (M)')
        self.loupe_cb.toggled.connect(self._view_opts)
        tb.addWidget(self.loupe_cb)
        tb.addAction(act('Fit', lambda: self.view.fit(), QStyle.SP_TitleBarNormalButton, 'Fit to window (F)'))
        tb.addAction(act('1:1', lambda: self.view.zoom_one(), None, '100% (1)'))
        tb.addSeparator()
        tb.addWidget(QLabel(' Theme '))
        self.theme_cb = QComboBox()
        self.theme_cb.addItems(list(THEMES.keys()))
        self.theme_cb.setToolTip('Dark, Light, or Night (red) to keep your night vision at the scope')
        self.theme_cb.setCurrentText(THEME['name'])
        self.theme_cb.currentTextChanged.connect(self.apply_theme)
        tb.addWidget(self.theme_cb)
        tb.addSeparator()
        self.lock_cb = QCheckBox('Lock layout')
        self.lock_cb.setToolTip('Stops the dividers and the table columns being dragged by accident.\n'
                                'Untick to adjust them, then tick again.')
        self.lock_cb.toggled.connect(self.apply_layout_lock)
        tb.addWidget(self.lock_cb)
        tb.addAction(act('Reset layout', self.reset_layout, None,
                         'Put the window, dividers, table columns and graphs back to how they start'))

        self.view = ImageView()
        self.view.pixelInfo.connect(lambda s: self.pix_label.setText(s))
        self.view.wantFull.connect(self._want_full)
        self.view.zoomChanged.connect(lambda z: QTimer.singleShot(0, self._want_full))

        self.model = FrameModel(self)
        self.proxy = SortProxy()
        self.proxy.setSourceModel(self.model)
        # the table order is asked for several times per blink step: keep it
        # until the table is sorted again or rows come or go
        self._order_cache = None
        for sgn in (self.proxy.layoutChanged, self.proxy.modelReset, self.proxy.rowsInserted,
                    self.proxy.rowsRemoved, self.proxy.rowsMoved):
            sgn.connect(self._order_dirty)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(-1, Qt.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.setWordWrap(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._default_columns()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setMinimumSectionSize(30)
        self.table.clicked.connect(self._table_clicked)
        self.table.selectionModel().currentRowChanged.connect(self._table_current)
        self.table.installEventFilter(self)
        self.proxy.layoutChanged.connect(lambda: self.graphs_refresh())

        self.tabs = QTabWidget()
        self.header_panel = HeaderPanel()
        self.graph_panel = GraphPanel(self)
        self.reject_panel = RejectPanel(self)
        self.reject_panel.setParent(self)
        self.reject_panel.hide()                 # its controls live on the Graphs tab
        self.graph_panel.attach_reject(self.reject_panel)
        self.tabs.addTab(self.graph_panel, 'Graphs')
        self.tabs.addTab(self.header_panel, 'Header')

        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.view)
        top.addWidget(self.tabs)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 1)
        top.setSizes([1100, 420])
        self.vsplit = QSplitter(Qt.Vertical)
        self.vsplit.addWidget(top)
        self.vsplit.addWidget(self.table)
        self.vsplit.setStretchFactor(0, 3)
        self.vsplit.setStretchFactor(1, 1)
        self.vsplit.setSizes([700, 260])
        self.hsplit = top
        self.setCentralWidget(self.vsplit)

        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status = QLabel('')
        self.pix_label = QLabel('')
        self.pix_label.setMinimumWidth(260)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.progress.setFormat('Measuring %v / %m')
        self.progress.setToolTip('The reference frame is chosen when measuring finishes')
        sb.addWidget(self.status, 1)
        sb.addPermanentWidget(self.pix_label)
        sb.addPermanentWidget(self.progress)

        keys = [('Space', self.toggle_play), ('Left', self.prev_frame), ('Right', self.next_frame),
                ('Home', self.first_frame), ('End', self.last_frame),
                ('G', lambda: self.mark_current('good')), ('B', lambda: self.mark_current('bad')),
                ('U', lambda: self.mark_current('')), ('R', lambda: self.set_reference()),
                ('A', lambda: self.align_cb.toggle()), ('M', lambda: self.loupe_cb.toggle()),
                ('T', lambda: self.trails_cb.toggle()), ('I', self._toggle_info),
                ('F', lambda: self.view.fit()), ('1', lambda: self.view.zoom_one()),
                ('+', lambda: self.view.zoom_to(self.view.z * 1.25)),
                ('=', lambda: self.view.zoom_to(self.view.z * 1.25)),
                ('-', lambda: self.view.zoom_to(self.view.z / 1.25)),
                ('Delete', self.remove_selected)]
        for k, fn in keys:
            s = QShortcut(QKeySequence(k), self)
            s.setContext(Qt.WindowShortcut)
            s.activated.connect(fn)

    def _default_columns(self):
        # snug starting widths (never narrower than the column title), so
        # 'Why flagged' - the last column - gets the rest of the width
        widths = {'idx': 34, 'mark': 54, 'name': 280, 'date': 148, 'exposure': 50, 'filter': 55,
                  'temp': 45, 'background': 70, 'noise': 50, 'snr': 42, 'stars': 50, 'fwhm': 64,
                  'fwhm_arcsec': 58, 'ecc': 40, 'saturated': 46, 'trails': 42, 'shift': 76}
        hfm = self.table.horizontalHeader().fontMetrics()
        for key, wdt in widths.items():
            need = hfm.horizontalAdvance(COLUMNS[COL[key]][1]) + 16
            self.table.setColumnWidth(COL[key], max(wdt, need))

    # ============================================================ layout lock / reset
    def apply_layout_lock(self, locked):
        """Stop (or allow) dragging the dividers and the table's column edges."""
        for sp in (self.hsplit, self.vsplit):
            for i in range(1, sp.count()):
                h = sp.handle(i)
                h.setEnabled(not locked)
                h.setCursor(Qt.ArrowCursor if locked else
                            (Qt.SplitHCursor if sp.orientation() == Qt.Horizontal else Qt.SplitVCursor))
        hh = self.table.horizontalHeader()
        last = len(COLUMNS) - 1
        for c in range(last):
            hh.setSectionResizeMode(c, QHeaderView.Fixed if locked else QHeaderView.Interactive)
        hh.setStretchLastSection(True)
        self.settings.setValue('layout_locked', bool(locked))

    def reset_layout(self):
        """Window, dividers, columns and graphs back to how they start."""
        self.showNormal()
        scr = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        w, h = min(1500, scr.width() - 40), min(950, scr.height() - 40)
        self.resize(w, h)
        self.move(scr.x() + (scr.width() - w) // 2, scr.y() + (scr.height() - h) // 2)
        self.hsplit.setSizes([int(w * 0.73), int(w * 0.27)])
        self.vsplit.setSizes([int(h * 0.73), int(h * 0.27)])
        self._default_columns()
        self.graph_panel.set_keys(GraphPanel.DEFAULTS)
        self.status.setText('Layout reset')

    def eventFilter(self, obj, ev):
        # keep Space / arrows for blinking even when the table has focus
        if obj is self.table and ev.type() == ev.Type.KeyPress:
            if ev.key() in (Qt.Key_Space,):
                self.toggle_play()
                return True
            if ev.key() == Qt.Key_Left:
                self.prev_frame()
                return True
            if ev.key() == Qt.Key_Right:
                self.next_frame()
                return True
        return super().eventFilter(obj, ev)

    def _toggle_info(self):
        self.view.show_info = not self.view.show_info
        self.view.update()

    def _view_opts(self, *_a):
        self.view.show_trails = self.trails_cb.isChecked()
        self.view.loupe = self.loupe_cb.isChecked()
        if self.view.loupe:
            self._want_full()
        self.view.update()

    # ============================================================ settings
    def _load_settings(self):
        s = self.settings
        g = s.value('geometry')
        if g is not None:
            self.restoreGeometry(g)
        else:
            self.resize(1500, 950)
        try:
            self.fps.setValue(float(s.value('fps', 4)))
            self.stretch_cb.setCurrentIndex(int(s.value('stretch', 0)))
            self.linked.setChecked(str(s.value('linked', 'false')).lower() == 'true')
            self.boost.setValue(int(s.value('boost', 25)))
            self.skip_bad.setChecked(str(s.value('skip_bad', 'true')).lower() == 'true')
            self.align_cb.setChecked(str(s.value('align', 'true')).lower() == 'true')
            self.trails_cb.setChecked(str(s.value('trails', 'true')).lower() == 'true')
        except Exception:
            pass
        # divider positions are put back once the window is on screen, when
        # the real sizes are known (doing it earlier can squash the table)
        self._split_restored = False
        try:
            cw = json.loads(str(s.value('col_widths', '') or '{}'))
            for key, wdt in cw.items():
                if key in COL and COL[key] < len(COLUMNS) - 1 and 20 <= int(wdt) <= 2000:
                    self.table.setColumnWidth(COL[key], int(wdt))
        except Exception:
            pass
        try:
            gk = json.loads(str(s.value('graphs', '') or '[]'))
            if gk:
                self.graph_panel.set_keys(gk)
        except Exception:
            pass
        locked = str(s.value('layout_locked', 'false')).lower() == 'true'
        self.lock_cb.setChecked(locked)
        self.apply_layout_lock(locked)
        self.reject_panel.load(s)
        self._stretch_changed()
        self.graph_panel.sync()

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, '_split_restored', True):
            self._split_restored = True
            QTimer.singleShot(0, self._restore_splitters)

    def _restore_splitters(self):
        try:
            hs = self.settings.value('hsplit')
            vs = self.settings.value('vsplit')
            if hs is not None:
                self.hsplit.restoreState(hs)
            if vs is not None:
                self.vsplit.restoreState(vs)
        except Exception:
            pass
        # restoring can recreate the divider handles, so apply the lock again
        self.apply_layout_lock(self.lock_cb.isChecked())

    def closeEvent(self, e):
        LOG.info('Closing AS Blink (%s loaded)', n_frames(len(self.frames)))
        s = self.settings
        s.setValue('geometry', self.saveGeometry())
        s.setValue('fps', self.fps.value())
        s.setValue('stretch', self.stretch_cb.currentIndex())
        s.setValue('linked', self.linked.isChecked())
        s.setValue('boost', self.boost.value())
        s.setValue('skip_bad', self.skip_bad.isChecked())
        s.setValue('align', self.align_cb.isChecked())
        s.setValue('trails', self.trails_cb.isChecked())
        s.setValue('hsplit', self.hsplit.saveState())
        s.setValue('vsplit', self.vsplit.saveState())
        s.setValue('col_widths', json.dumps({c[0]: self.table.columnWidth(i)
                                             for i, c in enumerate(COLUMNS[:-1])}))
        s.setValue('graphs', json.dumps(self.graph_panel.keys()))
        self.reject_panel.save(s)
        self.gen += 1
        self.timer.stop()
        self.pool.clear()
        self.dpool.clear()
        self.pool.waitForDone(5000)
        self.dpool.waitForDone(5000)
        super().closeEvent(e)

    # ============================================================ loading
    def alive(self, gen):
        return gen == self.gen

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        files = []
        for p in paths:
            if os.path.isdir(p):
                files += self._files_in(p)
            elif p.lower().endswith(ALL_EXT):
                files.append(p)
        if files:
            self.add_paths(files)

    def _files_in(self, folder):
        try:
            names = [n for n in os.listdir(folder) if n.lower().endswith(ALL_EXT)
                     and os.path.isfile(os.path.join(folder, n))]
        except OSError:
            return []
        names.sort(key=natural_key)
        return [os.path.join(folder, n) for n in names]

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, 'Open a folder of subs', self.settings.value('last_dir', ''))
        if not d:
            return
        self.settings.setValue('last_dir', d)
        files = self._files_in(d)
        if not files:
            QMessageBox.information(self, APP_NAME, 'No images found in that folder.')
            return
        self.clear_all()
        self.add_paths(files)

    def add_files(self):
        exts = ' '.join('*' + e for e in ALL_EXT)
        files, _f = QFileDialog.getOpenFileNames(self, 'Add images', self.settings.value('last_dir', ''),
                                                 'Images (%s);;All files (*)' % exts)
        if files:
            self.settings.setValue('last_dir', os.path.dirname(files[0]))
            files.sort(key=natural_key)
            self.add_paths(files)

    def add_paths(self, paths):
        have = {f.src.key for f in self.frames}
        new = []
        for p in paths:
            if p.lower().endswith(MULTI_EXT):
                try:
                    info = read_multi_info(p)
                except Exception as e:
                    self.status.setText('%s: %s' % (os.path.basename(p), e))
                    LOG.warning('%s could not be opened: %s', fname(p), e)
                    continue
                LOG.info('%s: %d frames, %dx%d, %s%s', fname(p), info['count'], info['width'], info['height'],
                         info.get('container'), (', %.1f fps' % info['fps']) if info.get('fps') else '')
                if info['count'] > 20000:
                    r = QMessageBox.question(self, APP_NAME, '%s has %d frames. Load them all?'
                                             % (os.path.basename(p), info['count']))
                    if r != QMessageBox.Yes:
                        continue
                for i in range(info['count']):
                    s = Source(p, i, info)
                    if s.key not in have:
                        new.append(s)
            else:
                s = Source(p)
                if s.key not in have:
                    new.append(s)
        if not new:
            return
        try:
            exts = collections.Counter(os.path.splitext(s.path)[1].lower() or '(none)' for s in new)
            folders = sorted({fname(os.path.dirname(s.path)) for s in new})
            LOG.info('Adding %s from folder %s - types: %s | settings: %s', n_frames(len(new)),
                     ', '.join(folders[:5]) + (' ...' if len(folders) > 5 else ''),
                     ', '.join('%s x%d' % kv for kv in sorted(exts.items())), self._log_settings())
        except Exception:
            LOG.exception('Could not log the new files')
        self.measure_t0 = time.time()
        self.model.beginInsertRows(QModelIndex(), len(self.frames), len(self.frames) + len(new) - 1)
        start = len(self.frames)
        for k, s in enumerate(new):
            fr = Frame(s, start + k)
            self.frames.append(fr)
            self.by_uid[fr.uid] = fr
        self.model.endInsertRows()
        for fr in self.frames[start:]:
            self._queue_analyse(fr)
        self._update_progress()

    def _log_settings(self):
        return 'debayer %s, check pattern %s, trails %s, star sigma %s, align %s, theme %s' % (
            self.opts.get('debayer'), self.opts.get('check_pattern'), self.opts.get('trails'),
            self.opts.get('star_sigma', 5.0), self.align_cb.isChecked() if hasattr(self, 'align_cb') else '?',
            THEME.get('name', '?'))

    def _queue_analyse(self, fr):
        if fr.status != 'working':
            self._pending += 1
        fr.status = 'working'
        self.pool.start(AnalyseTask(self.sig, self.gen, fr.uid, fr.src, dict(self.opts), self.alive))

    def clear_all(self):
        self.timer.stop()
        self._set_play_icon(False)
        self.gen += 1
        self.pool.clear()
        self.dpool.clear()
        self.busy.clear()
        for p in {f.src.path for f in self.frames if is_video(f.src)}:
            forget_video(p)
        self.frames = []
        self._pending = 0
        self._rows = {}
        self.by_uid = {}
        self.cur = None
        self.ref = None
        self.auto_ref = True
        self.ref_final = False
        self.stretch.ref_stats = None
        self.pcache.clear()
        self.fcache.clear()
        self.rcache.clear()
        self.last_move = []
        self.a_undo.setEnabled(False)
        self.model.reset()
        self.view.clear('Open a folder of subs to start  (Ctrl+O)')
        self.header_panel.show_header([], None)
        self.reject_panel.result.setText('')
        self.reject_panel.apply.setEnabled(False)
        self.reject_panel.reset_limits()
        self.reject_panel.update_ranges()
        self.reject_summary = ''
        self.graph_panel.sync()
        self.progress.setVisible(False)
        self.graphs_refresh()

    def remove_selected(self):
        rows = sorted({self.proxy.mapToSource(i).row() for i in self.table.selectionModel().selectedRows()})
        if not rows:
            return
        keep = [f for i, f in enumerate(self.frames) if i not in set(rows)]
        removed = {self.frames[i].uid for i in rows}
        self.model.beginResetModel()
        self._pending -= sum(1 for f in self.frames if f.uid in removed and f.status == 'working')
        self.frames = keep
        self._rows = {}
        for u in removed:
            self.by_uid.pop(u, None)
            self.pcache.drop(lambda k, u=u: k == u)
            self.fcache.drop(lambda k, u=u: k == u)
            self.rcache.drop(lambda k, u=u: isinstance(k, tuple) and k[0] == u)
        self.model.endResetModel()
        self.reject_panel.update_ranges()
        self.apply_manual_limits()
        ref_msg = ''
        if self.ref in removed:
            # the reference has gone: choose again, the same way as at the end of measuring
            self.ref = None
            self.auto_ref = True
            self.ref_final = False
            self.rcache.clear()
            best = None if self._still_measuring() else self._best_frame()
            if best is not None:
                keep_cur = self.cur
                self.cur = best.uid
                self.set_reference(quiet=True)
                self.cur = keep_cur
                ref_msg = 'The reference was removed. New reference: %s - press R to use another' % best.name
            else:
                self._pick_reference()
        if self.cur in removed:
            self.cur = None
            order = self.order()
            if order:
                self.show_frame(order[0])
            else:
                self.view.clear('No frames')
        self.graphs_refresh()
        if ref_msg:
            self.status.setText(ref_msg)

    def _still_measuring(self):
        # a running count, not a scan: a long video can have 20,000 frames
        return self._pending > 0

    def _update_progress(self):
        total = len(self.frames)
        done = total - self._pending
        if total and done < total:
            now = time.time()
            if now - getattr(self, '_prog_t', 0.0) < 0.1 and done > 0:
                return              # redrawing the bar for every frame slows long videos down
            self._prog_t = now
            self.progress.setVisible(True)
            self.progress.setMaximum(total)
            self.progress.setValue(done)
        else:
            self.progress.setVisible(False)

    def _row_of(self, uid):
        i = self._rows.get(uid)
        if i is not None and i < len(self.frames) and self.frames[i].uid == uid:
            return i
        self._rows = {f.uid: k for k, f in enumerate(self.frames)}
        i = self._rows.get(uid)
        if i is not None:
            return i
        return -1

    def _on_analysed(self, gen, uid, an):
        if gen != self.gen or uid not in self.by_uid:
            return
        fr = self.by_uid[uid]
        if fr.status == 'working':
            self._pending -= 1
        fr.an = an
        if an.ok:
            fr.status = 'ok'
            self.pcache.put(uid, (an.preview, an.prev_scale), pinned=self._pinned())
            fr.pstats = getattr(an, 'pstats', None) or preview_stats(an.preview)
            an.preview = None
            if self.ref is None:
                self._pick_reference()
            self._compute_shift(fr)
        else:
            fr.status = 'error'
        row = self._row_of(uid)
        if row >= 0:
            self.model.refresh_row(row)
        self._update_progress()
        if self.cur is None and fr.status == 'ok':
            self.show_frame(uid)
        elif self.cur == uid:
            self.show_frame(uid)
        if not self.graph_timer.isActive():
            self.graph_timer.start(400 if len(self.frames) < 1500 else 1500)
        if self._pending <= 0:
            n_err = sum(1 for f in self.frames if f.status == 'error')
            msg = '%s measured%s' % (n_frames(len(self.frames)), ', %d could not be read' % n_err if n_err else '')
            LOG.info('Measuring finished: %s in %.1fs, %d could not be read',
                     n_frames(len(self.frames)), time.time() - getattr(self, 'measure_t0', time.time()), n_err)
            if self.frames and n_err == len(self.frames):
                why = self.frames[0].an.error if self.frames[0].an is not None else ''
                self.status.setText('None of the %d file%s could be read%s. Click a row to see the reason for each.'
                                    % (n_err, '' if n_err == 1 else 's', (' (first: %s)' % why) if why else ''))
                self.view.clear('None of these files could be read.\n%s' % why)
                self.reject_panel.update_ranges()
                self.apply_manual_limits()
                self.graphs_refresh()
                return
            self._drop_repeating_trails()
            self.reject_panel.update_ranges()
            self.apply_manual_limits()
            if self.reject_panel.auto_run.isChecked():
                self.suggest_limits(quiet=True)
            if self.auto_ref:
                best = self._best_frame()
                if best is not None and best.uid == self.ref:
                    self.ref_final = True
                    self.model.refresh_all()
                    if self.cur is not None:
                        self.show_frame(self.cur, sync_table=False)
                if best is not None and best.uid != self.ref:
                    keep = self.cur
                    self.cur = best.uid
                    self.set_reference(quiet=True)
                    self.cur = keep
                    self.show_frame(keep)
                if best is not None and best.an.quick:
                    msg += '.  Reference: %s (stars are not measured in video frames) - press R to use another' % best.name
                elif best is not None:
                    msg += ('.  Reference: %s (sharpest with the most stars, no trails or flags) - press R to use another'
                            % best.name)
            self.status.setText(msg)

    def _refresh_slow(self):
        self.graphs_refresh()

    def _pinned(self):
        return tuple(u for u in (self.cur, self.ref) if u is not None)

    # ============================================================ reference & alignment
    def _pick_reference(self):
        for f in self.frames:
            if f.status == 'ok':
                self.ref = f.uid
                break
        if self.ref is None:
            return
        rf = self.by_uid[self.ref]
        self.stretch.ref_stats = getattr(rf, 'pstats', None)
        self.view.set_reference_size(rf.an.width, rf.an.height)
        for f in self.frames:
            if f.status == 'ok':
                self._compute_shift(f)
        self.model.refresh_all()

    def _compute_shift(self, fr):
        if self.ref is None or fr.an is None or not fr.an.ok:
            fr.shift = None
            return
        rf = self.by_uid.get(self.ref)
        if rf is None or rf.an is None or not rf.an.ok:
            fr.shift = None
            return
        if fr.uid == self.ref:
            fr.shift = (0.0, 0.0, False)
            return
        if (fr.an.width, fr.an.height) != (rf.an.width, rf.an.height):
            fr.shift = None
            return
        # no meridian flip inside one video, so skip that check there (it doubles the work)
        dx, dy, resp, flipped = phase_shift(rf.an.athumb, fr.an.athumb, flip=not (fr.an.quick and rf.an.quick))
        f = fr.an.athumb_f
        fr.shift = (dx / f, dy / f, flipped)
        fr.an.align_resp = resp

    def _best_frame(self):
        frs = [f for f in self.frames if f.status == 'ok' and f.src.ser is None] or \
              [f for f in self.frames if f.status == 'ok']
        if not frs:
            return None
        # stay within the biggest group (same filter, exposure, size)
        groups = collections.Counter(self._group_key(f) for f in frs)
        top = groups.most_common(1)[0][0]
        frs = [f for f in frs if self._group_key(f) == top]
        # never a frame with a trail, one flagged for rejection, or one marked Bad -
        # unless every frame is like that
        clean = [f for f in frs if not f.an.stats.get('trails') and not f.reasons and f.mark != 'bad']
        frs = clean or frs
        fw = [f.an.stats.get('fwhm') for f in frs if f.an.stats.get('fwhm')]
        if fw:
            lim = float(np.median(fw))
            sharp = [f for f in frs if f.an.stats.get('fwhm') and f.an.stats['fwhm'] <= lim]
            frs = sharp or frs
        return max(frs, key=lambda f: f.an.stats.get('stars') or 0)

    def _drop_repeating_trails(self):
        """A 'trail' found in the same place in two or more frames is sky
        structure (nebula edges, dust lanes), not a satellite: drop it."""
        rf = self.by_uid.get(self.ref)
        segs = {}
        for f in self.frames:
            if f.status != 'ok':
                continue
            an = f.an
            if f.shift is None or not an.trails_all:
                set_trails(an, range(len(an.trails_all)))
                if f.shift is not None:
                    segs[f.uid] = []
                continue
            dx, dy, flipped = f.shift
            out = []
            for x1, y1, x2, y2 in an.trails_all:
                if flipped:
                    x1, y1, x2, y2 = an.width - x1, an.height - y1, an.width - x2, an.height - y2
                out.append((x1 - dx, y1 - dy, x2 - dx, y2 - dy))
            segs[f.uid] = out
        if not segs or rf is None:
            return
        drop = repeating_trails(segs, math.hypot(rf.an.width, rf.an.height))
        for uid, bad in drop.items():
            an = self.by_uid[uid].an
            set_trails(an, [i for i in range(len(an.trails_all)) if i not in bad])
        self.model.refresh_all()
        if self.cur is not None:
            self.show_frame(self.cur, sync_table=False)

    def set_reference(self, quiet=False):
        if not quiet:
            self.auto_ref = False
        if self.cur is None:
            return
        fr = self.by_uid[self.cur]
        if fr.status != 'ok':
            return
        self.ref = fr.uid
        self.ref_final = True
        self.stretch.ref_stats = getattr(fr, 'pstats', None)
        self.view.set_reference_size(fr.an.width, fr.an.height)
        for f in self.frames:
            self._compute_shift(f)
        self.rcache.clear()
        self.model.refresh_all()
        if not quiet:
            self.status.setText('Reference is now %s' % fr.name)
            self.show_frame(self.cur)

    # ============================================================ order & navigation
    def _order_dirty(self, *_a):
        self._order_cache = None

    def order(self):
        """uids in the table's (sorted) order. Do not change the list returned."""
        c = self._order_cache
        if c is not None and len(c) == len(self.frames):
            return c
        uids = []
        for r in range(self.proxy.rowCount()):
            src = self.proxy.mapToSource(self.proxy.index(r, 0)).row()
            uids.append(self.frames[src].uid)
        self._order_cache = uids
        return uids

    def blink_order(self):
        uids = self.order()
        if self.sel_only.isChecked():
            sel = {self.frames[self.proxy.mapToSource(i).row()].uid
                   for i in self.table.selectionModel().selectedRows()}
            if len(sel) >= 2:
                uids = [u for u in uids if u in sel]
        # asked for several times per step, so kept quick for long videos
        by, cur = self.by_uid, self.cur
        if self.skip_bad.isChecked():
            return [u for u in uids if by[u].status != 'error' and (by[u].mark != 'bad' or u == cur)]
        return [u for u in uids if by[u].status != 'error']

    def time_order(self):
        frs = [f for f in self.frames if f.status == 'ok']
        if frs and all(f.an.when is not None for f in frs):
            return sorted(frs, key=lambda f: (f.an.when, f.uid))
        return frs

    def current_frame(self):
        return self.by_uid.get(self.cur)

    def _step(self, d):
        order = self.blink_order()
        if not order:
            return None
        if self.cur not in order:
            return order[0]
        i = order.index(self.cur)
        return order[(i + d) % len(order)]

    def next_frame(self):
        u = self._step(1)
        if u is not None:
            self.show_frame(u, sync_table=True)

    def prev_frame(self):
        u = self._step(-1)
        if u is not None:
            self.show_frame(u, sync_table=True)

    def first_frame(self):
        o = self.blink_order()
        if o:
            self.show_frame(o[0], sync_table=True)

    def last_frame(self):
        o = self.blink_order()
        if o:
            self.show_frame(o[-1], sync_table=True)

    def go_uid(self, uid):
        if uid in self.by_uid:
            self.show_frame(uid, sync_table=True)

    def _table_clicked(self, idx):
        src = self.proxy.mapToSource(idx).row()
        if 0 <= src < len(self.frames):
            self.show_frame(self.frames[src].uid, sync_table=False)

    def _table_current(self, cur, _prev):
        if getattr(self, '_syncing', False) or not cur.isValid() or self.timer.isActive():
            return
        src = self.proxy.mapToSource(cur).row()
        if 0 <= src < len(self.frames) and self.frames[src].uid != self.cur:
            self.show_frame(self.frames[src].uid, sync_table=False)

    def _sync_table(self):
        row = self._row_of(self.cur)
        if row < 0:
            return
        pidx = self.proxy.mapFromSource(self.model.index(row, 0))
        if not pidx.isValid():
            return
        sm = self.table.selectionModel()
        # Do NOT block the selection model's signals here: the table repaints
        # rows from those signals, and blocking them left every row it had
        # passed through still drawn as highlighted during playback.
        # A flag stops our own handler reacting to this change instead.
        self._syncing = True
        try:
            sel_rows = sm.selectedRows()
            if len(sel_rows) >= 2 and (self.sel_only.isChecked() or pidx.row() in {i.row() for i in sel_rows}):
                sm.setCurrentIndex(pidx, QItemSelectionModel.NoUpdate)
            else:
                sm.setCurrentIndex(pidx, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
            self.table.scrollTo(pidx, QAbstractItemView.EnsureVisible)
        finally:
            self._syncing = False

    # ============================================================ play
    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self._set_play_icon(False)
            self._sync_table()
        else:
            if len(self.blink_order()) < 2:
                return
            self.timer.start(int(1000 / self.fps.value()))
            self._set_play_icon(True)

    def _set_play_icon(self, playing):
        self._playing_icon = playing
        self.a_play.setIcon(draw_icon('pause' if playing else 'play', C('icon')))
        self.a_play.setText('Pause' if playing else 'Play')

    def _refresh_icons(self):
        col = C('icon')
        self.a_prev.setIcon(draw_icon('prev', col))
        self.a_next.setIcon(draw_icon('next', col))
        self._set_play_icon(getattr(self, '_playing_icon', False))
        for a, std in self._std_icons:
            icon = self.style().standardIcon(std)
            a.setIcon(tinted_icon(icon, col) if night_mode() else icon)

    def apply_theme(self, name):
        apply_palette(QApplication.instance(), name)
        self.settings.setValue('theme', THEME['name'])
        self._refresh_icons()
        self.header_panel.restyle()
        self.graph_panel.restyle()
        self.rcache.clear()
        self.view.crop_cache = None
        self.model.refresh_all()
        self.graphs_refresh()
        if self.cur is not None:
            self.show_frame(self.cur, sync_table=False)
        else:
            self.view.update()

    def _fps_changed(self, v):
        if self.timer.isActive():
            self.timer.setInterval(int(1000 / v))

    def _tick(self):
        u = self._step(1)
        if u is None:
            return
        if self.pcache.get(u) is None:
            # not in memory yet: ask for it and wait for the next tick
            self._request_preview(u)
            self._prefetch(u)
            return
        self.show_frame(u, sync_table=True)

    # ============================================================ display
    def _request_preview(self, uid):
        fr = self.by_uid.get(uid)
        if fr is None or fr.status != 'ok' or ('p', uid) in self.busy:
            return
        self.busy.add(('p', uid))
        self.dpool.start(PreviewTask(self.sig, self.gen, uid, fr.src, dict(self.opts), self.alive, self.busy))

    def _request_full(self, uid):
        fr = self.by_uid.get(uid)
        if fr is None or fr.status != 'ok' or ('f', uid) in self.busy or uid in self.fcache:
            return
        self.busy.add(('f', uid))
        self.dpool.start(FullTask(self.sig, self.gen, uid, fr.src, dict(self.opts), self.alive, self.busy))

    def _on_preview(self, gen, uid, pv, ps):
        if gen != self.gen or uid not in self.by_uid:
            return
        self.pcache.put(uid, (pv, ps), pinned=self._pinned())
        if uid == self.cur or uid == self.pending_show:
            self.show_frame(uid)

    def _on_full(self, gen, uid, img):
        if gen != self.gen or uid not in self.by_uid:
            return
        self.fcache.put(uid, img, pinned=(self.cur,) if self.cur is not None else ())
        if uid == self.cur:
            self.view.full = img
            self.view.crop_cache = None
            self.view.update()

    def _on_failed(self, gen, uid, msg):
        if gen != self.gen:
            return
        self.status.setText('Could not load %s: %s' % (self.by_uid[uid].name if uid in self.by_uid else '', msg))

    def _want_full(self):
        if self.cur is None:
            return
        if self.view.needs_full() or self.view.loupe:
            self._request_full(self.cur)
            if self.timer.isActive():
                for u in self._ahead(self.cur, 2):
                    self._request_full(u)

    def _ahead(self, uid, n):
        order = self.blink_order()
        if uid not in order or not order:
            return order[:n]
        i = order.index(uid)
        return [order[(i + k) % len(order)] for k in range(1, min(n, len(order) - 1) + 1)]

    def _prefetch(self, uid):
        for u in self._ahead(uid, 4):
            if self.pcache.get(u) is None:
                self._request_preview(u)

    def _render(self, fr, pv):
        key = (fr.uid, self.stretch.key())
        hit = self.rcache.get(key)
        if hit is not None:
            return hit
        luts = self.stretch.luts(fr.pstats)
        q = np_to_qimage(night_tint(apply_luts(pv, luts)))
        out = (q, luts)
        self.rcache.put(key, out)
        return out

    def show_frame(self, uid, sync_table=True):
        if uid is None or uid not in self.by_uid:
            return
        fr = self.by_uid[uid]
        if fr.status != 'ok':
            self.cur = uid
            self.view.clear('Measuring %s ...' % fr.name if fr.status in ('working', 'waiting')
                            else 'Could not read %s\n%s' % (fr.name, fr.an.error if fr.an else ''))
            self.header_panel.show_header(fr.an.header if fr.an else [], None)
            if sync_table:
                self._sync_table()
            return
        hit = self.pcache.get(uid)
        if hit is None:
            self.pending_show = uid
            self._request_preview(uid)
            if self.cur is None:
                self.cur = uid
            return
        self.pending_show = None
        self.cur = uid
        pv, ps = hit
        q, luts = self._render(fr, pv)
        full = self.fcache.get(uid)
        if self.align_cb.isChecked() and fr.shift is not None:
            dx, dy, flipped = fr.shift
            shift = (dx, dy)
        else:
            shift, flipped = (0.0, 0.0), False
        order = self.blink_order()
        pos = order.index(uid) + 1 if uid in order else 0
        label = '%d / %d   %s' % (pos, len(order), fr.name)
        if uid == self.ref:
            label += '   [reference]' if self.ref_final else '   [temporary reference]'
        st = fr.an.stats
        bits = []
        if st.get('fwhm'):
            bits.append('FWHM %.2f px' % st['fwhm'] + (' (%.2f")' % st['fwhm_arcsec'] if st.get('fwhm_arcsec') else ''))
        if st.get('ecc') is not None:
            bits.append('Ecc %.2f' % st['ecc'])
        if st.get('stars') is not None:
            bits.append('Stars %d' % st['stars'])
        bits.append('Bkg %.0f' % st['background'])
        bits.append('SNR %.2f' % st['snr'])
        if st.get('trails'):
            bits.append('Trails %d' % st['trails'])
        if fr.reasons:
            bits.append('Flagged: ' + '; '.join(fr.reasons))
        self.view.preview16 = pv
        self.view.set_frame(q, ps, fr.an.width, fr.an.height, luts, full, shift, flipped,
                            fr.an.trails, label, '   '.join(bits), fr.mark)
        if self.view.needs_full() or self.view.loupe:
            self._want_full()
        # header, compared with the previous frame in blink order
        prev_fr = None
        if order and uid in order:
            i = order.index(uid)
            pu = order[i - 1] if i > 0 else None
            prev_fr = self.by_uid.get(pu) if pu is not None else None
        note = ''
        if fr.an.kind == 'cfa':
            note = 'Colour sub, debayered as %s.' % fr.an.pattern
            if fr.an.pattern_note:
                note += ' ' + fr.an.pattern_note
        if getattr(fr.an, 'data_note', ''):
            note = (note + ' ' if note else '') + fr.an.data_note + '.'
        if self.tabs.currentWidget() is self.header_panel or not self.timer.isActive():
            self.header_panel.show_header(fr.an.header, prev_fr.an.header if prev_fr and prev_fr.an else None, note)
        if sync_table:
            self._sync_table()
        self.status.setText('%s   %d x %d   %s' % (fr.src.path, fr.an.width, fr.an.height,
                                                  {'cfa': 'colour (debayered %s)' % fr.an.pattern,
                                                   'rgb': 'colour', 'mono': 'mono'}[fr.an.kind]))
        self._prefetch(uid)
        if self.timer.isActive():
            if not self.graph_timer.isActive():
                # thousands of video frames make the graphs slow to draw: update less often
                self.graph_timer.start(300 if len(self.frames) < 1500 else 1000)
        else:
            self.graph_panel.refresh()

    # ============================================================ stretch / debayer
    def _stretch_changed(self, *_a):
        idx = self.stretch_cb.currentIndex()
        self.stretch.mode = {0: 1, 1: 0, 2: 2}[idx]
        self.stretch.linked = self.linked.isChecked()
        self.stretch.target = self.boost.value() / 100.0
        self.rcache.clear()
        if self.cur is not None:
            self.show_frame(self.cur)

    def _debayer_changed(self, idx):
        mode = ['auto', 'off', 'RGGB', 'BGGR', 'GRBG', 'GBRG'][idx]
        if mode == self.opts['debayer']:
            return
        self.opts['debayer'] = mode
        if not self.frames:
            return
        LOG.info('Debayer changed to %s - measuring all %s again', mode, n_frames(len(self.frames)))
        self._remeasure_all()

    def _video_stars_changed(self, on):
        self.opts['video_stars'] = bool(on)
        self.settings.setValue('video_stars', bool(on))
        if any(f.src.ser is not None for f in self.frames):
            LOG.info('Measure stars in SER/video frames: %s - measuring all %s again', on, n_frames(len(self.frames)))
            self._remeasure_all()

    def _remeasure_all(self):
        self.measure_t0 = time.time()
        # everything has to be measured again
        paths = [f.src for f in self.frames]
        marks = {f.src.key: f.mark for f in self.frames}
        self.gen += 1
        self.pool.clear()
        self.dpool.clear()
        self.busy.clear()
        self.pcache.clear()
        self.fcache.clear()
        self.rcache.clear()
        self.ref = None
        self.ref_final = False
        self.cur = None
        self.stretch.ref_stats = None
        self.model.beginResetModel()
        self.frames = []
        self._pending = 0
        self._rows = {}
        self.by_uid = {}
        for k, s in enumerate(paths):
            fr = Frame(s, k)
            fr.mark = marks.get(s.key, '')
            self.frames.append(fr)
            self.by_uid[k] = fr
        self.model.endResetModel()
        for fr in self.frames:
            self._queue_analyse(fr)
        self._update_progress()

    # ============================================================ marks
    def mark_current(self, mark):
        if self.cur is None:
            return
        rows = {self.proxy.mapToSource(i).row() for i in self.table.selectionModel().selectedRows()}
        targets = [self.frames[r] for r in rows] if len(rows) > 1 and not self.timer.isActive() else [self.by_uid[self.cur]]
        for fr in targets:
            fr.mark = mark
            r = self._row_of(fr.uid)
            if r >= 0:
                self.model.refresh_row(r)
        self.view.mark = self.by_uid[self.cur].mark
        self.view.update()
        self.graphs_refresh()
        if mark and len(targets) == 1 and not self.timer.isActive():
            self.next_frame()

    def clear_marks(self):
        if any(f.mark for f in self.frames):
            if QMessageBox.question(self, APP_NAME, 'Clear every Good and Bad mark?') != QMessageBox.Yes:
                return
        for f in self.frames:
            f.mark = ''
            f.reasons = []
            f.reason_keys = set()
        self.model.refresh_all()
        self.reject_panel.result.setText('')
        self.reject_panel.apply.setEnabled(False)
        self.reject_summary = ''
        if self.reject_panel.is_manual():
            self.apply_manual_limits()
        self.graph_panel.sync()
        self.show_frame(self.cur)

    # ============================================================ auto reject
    def _values(self, key, frames):
        return [self.model.value(f, key) for f in frames]

    def reject_limit(self, key, ys):
        if self.reject_panel.is_manual():
            return self.reject_panel.limit_for_graph(key)
        info = self.auto_limit_info(key, ys)
        return None if info is None else info[0]

    def auto_limit_info(self, key, ys):
        """Automatic limit for this graph's values: (limit, True if the minimum
        gap - not the sigma - decided it), or None. FWHM in arcseconds uses the
        FWHM test."""
        rules = self.reject_panel.get_rules()
        fam = reason_family(key)
        if fam not in rules or not rules[fam][0] or len(ys) < 4:
            return None
        if len(self._groups()) > 1:
            return None
        sigma = rules[fam][1]
        rule = [r for r in REJECT_RULES if r[0] == fam][0]
        direction, minrel = rule[2], rule[4]
        med = float(np.median(ys))
        mad = float(np.median(np.abs(np.array(ys) - med))) * 1.4826
        gap = minrel if fam == 'ecc' else minrel * abs(med)
        tol = max(sigma * mad, gap)
        return med + direction * tol, gap > sigma * mad

    @staticmethod
    def _group_key(f):
        hv = f.an.hv
        exp = hv.get('exposure')
        return (hv.get('filter') or '', round(exp, 2) if exp else None, f.an.width, f.an.height,
                f.an.kind, f.src.path if f.src.ser is not None else '')

    def _groups(self):
        groups = collections.OrderedDict()
        for f in self.frames:
            if f.status == 'ok':
                groups.setdefault(self._group_key(f), []).append(f)
        return groups

    def run_auto_reject(self, quiet=False):
        if self.reject_panel.is_manual():
            # the Graphs tab button always means the Automatic tests
            self.reject_panel.m_auto.setChecked(True)
        if any(f.status in ('waiting', 'working') for f in self.frames):
            if not quiet:
                QMessageBox.information(self, APP_NAME, 'Still measuring - try again when the progress bar has finished.')
            return
        groups = self._groups()
        if not any(len(g) >= 4 for g in groups.values()):
            if not quiet:
                QMessageBox.information(self, APP_NAME, 'Auto-reject needs at least 4 measured frames '
                                        'with the same filter, exposure and size.')
            return
        for f in self.frames:
            f.reasons = []
            f.reason_keys = set()
        keys = [r[0] for r in REJECT_RULES]
        n = total = 0
        skipped = 0
        for gk, frs in groups.items():
            if len(frs) < 4:
                skipped += len(frs)
                continue
            total += len(frs)
            values = {k: self._values(k, frs) for k in keys}
            reasons, rkeys = auto_reject(values, self.used_rules(), self.reject_panel.trails.isChecked(),
                                  [f.an.stats.get('trails', 0) for f in frs])
            for f, r, rk in zip(frs, reasons, rkeys):
                f.reasons = r
                f.reason_keys = {reason_family(k) for k in rk}
                if r:
                    n += 1
        self.model.refresh_all()
        self.graphs_refresh()
        ng = sum(1 for g in groups.values() if len(g) >= 4)
        extra = ''
        if ng > 1:
            extra = ' Frames were compared within %d groups (same filter, exposure and size).' % ng
        if skipped:
            extra += ' %d frame%s left out: fewer than 4 in their group.' % (skipped, '' if skipped == 1 else 's')
        if n:
            self.reject_panel.result.setText('%d of %d frames flagged (orange rows). Check them by '
                                             'blinking, then mark them Bad.%s' % (n, total, extra))
            self.reject_summary = '%d of %d flagged' % (n, total)
        else:
            self.reject_panel.result.setText('Nothing stands out - all %d frames look consistent.%s' % (total, extra))
            self.reject_summary = 'Nothing stands out'
        self.reject_panel.apply.setEnabled(n > 0)
        self.graph_panel.sync()
        if self.cur is not None:
            self.show_frame(self.cur)

    # ---- which measurements are used: only those shown on a graph and ticked
    def shown_families(self):
        return {reason_family(k) for k in self.graph_panel.keys()}

    def used_rules(self):
        shown = self.shown_families()
        return {k: (on and k in shown, sig) for k, (on, sig) in self.reject_panel.get_rules().items()}

    def using_text(self):
        """'Using: FWHM, Star count' - what can flag frames right now."""
        rp = self.reject_panel
        names = {'fwhm': 'FWHM', 'ecc': 'Eccentricity', 'stars': 'Star count', 'background': 'Background',
                 'snr': 'SNR'}
        order = ['fwhm', 'ecc', 'stars', 'background', 'snr']
        if rp.is_manual():
            used = {k for k, _vk, _kind, _lim in rp.active_limits()}
            parts = [names[k] for k in order if k in used]
            b = rp.best_setting()
            if b is not None:
                parts.append('best %d%% by %s' % (100 - b[3], names[b[0]]))
            if rp.trails.isChecked():
                parts.append('trails')
        else:
            used = {k for k, (on, _s) in self.used_rules().items() if on}
            parts = [names[k] for k in order if k in used]
            if rp.trails.isChecked():
                parts.append('trails')
        if not parts:
            return 'No tests in use - tick the box under a graph'
        return 'Using: ' + ', '.join(parts)

    def graphs_changed(self):
        """A graph now shows another measurement, so what is used has changed."""
        if self.reject_panel.is_manual():
            self.apply_manual_limits()
        elif any(f.reasons for f in self.frames) or self.reject_summary:
            self.schedule_auto_reject()
        self.graph_panel.sync()

    def suggest_limits(self, quiet=False):
        """Move each ticked slider to tonight's 'odd ones out' limit (the session's
        typical value, its usual spread and the minimum gaps), then flag."""
        if any(f.status in ('waiting', 'working') for f in self.frames):
            if not quiet:
                QMessageBox.information(self, APP_NAME, 'Still measuring - try again when the progress bar has finished.')
            return
        frs = [f for f in self.frames if f.status == 'ok']
        if len(frs) < 4:
            if not quiet:
                QMessageBox.information(self, APP_NAME, 'Suggest limits needs at least 4 measured frames.')
            return
        rp = self.reject_panel
        rp.update_ranges()
        base = rp.sigma_sp.value()
        shown = self.shown_families()
        names = []
        for key, d in rp.limits.items():
            if not d['cb'].isChecked() or key not in shown or d['lo'] is None:
                continue
            ys = [self.model.value(f, d['unit']) for f in frs]
            ys = [float(v) for v in ys if v is not None and math.isfinite(float(v))]
            if len(ys) < 4:
                continue
            sigma = base + (0.5 if key in ('background', 'snr') else 0.0)
            lim = suggested_limit(key, ys, sigma)
            d['sl'].blockSignals(True)
            d['sl'].setValue(rp._pos_for(d, lim))
            d['sl'].blockSignals(False)
            d['value'] = rp._value_at(d, d['sl'].value())
            rp._label(key)
            names.append(key)
        rp.suggested = bool(names)
        self.apply_manual_limits()
        if not names and not quiet:
            QMessageBox.information(self, APP_NAME, 'Tick the box under at least one graph first.')

    def schedule_auto_reject(self):
        """Re-run the Automatic tests shortly after a slider stops moving."""
        self.auto_timer.start(150)

    def _auto_rerun(self):
        if self.reject_panel.is_manual() or not self.frames:
            return
        if any(f.status in ('waiting', 'working') for f in self.frames):
            return
        self.run_auto_reject(quiet=True)

    def clear_reasons(self):
        for f in self.frames:
            f.reasons = []
            f.reason_keys = set()
        self.reject_summary = ''
        self.reject_panel.apply.setEnabled(False)
        self.model.refresh_all()
        self.graphs_refresh()
        self.graph_panel.sync()
        if self.cur is not None:
            self.show_frame(self.cur, sync_table=False)

    def apply_manual_limits(self):
        """Flag every frame that fails a ticked manual limit."""
        rp = self.reject_panel
        if not rp.is_manual():
            return
        limits = rp.active_limits()
        frs = [f for f in self.frames if f.status == 'ok']
        n_fail = 0
        for f in self.frames:
            f.reasons = []
            f.reason_keys = set()
        best = rp.best_setting()          # (key, value key, kind, drop %) or None
        best_drop, best_cut = set(), None
        if best is not None:
            bkey, bvk, bkind, bpct = best
            best_drop, best_cut = best_percent_cut({f.uid: self.model.value(f, bvk) for f in frs}, bkind, bpct)
            rp.best_cut = (bvk, best_cut)
        else:
            rp.best_cut = None
        for f in frs:
            for key, vk, kind, lim in limits:
                r = limit_reasons(self.model.value(f, vk), key, vk, kind, lim)
                if r:
                    f.reasons += r
                    f.reason_keys.add(reason_family(key))
            if rp.trails.isChecked() and f.an.stats.get('trails'):
                t = f.an.stats['trails']
                f.reasons.append('%d trail%s' % (t, 's' if t > 1 else ''))
                f.reason_keys.add('trails')
            if f.uid in best_drop:
                bkey, bvk, bkind, bpct = best
                v = self.model.value(f, bvk)
                name = LIMIT_LABEL[bkey].split(' ', 1)[1]
                if v is None:
                    f.reasons.append('%s not measured' % (name[0].upper() + name[1:]))
                else:
                    f.reasons.append('Not in the best %d%% for %s (%s; cut-off %s)'
                                     % (100 - bpct, name, _fmt(bvk, v), _fmt(bvk, best_cut)))
                f.reason_keys.add(reason_family(bkey))
            if f.reasons:
                n_fail += 1
        self.model.refresh_all()
        self.graphs_refresh()
        if not frs:
            rp.pass_label.setText('')
        elif n_fail == 0 and getattr(rp, 'suggested', False):
            rp.pass_label.setText('Suggested limits: nothing stands out - ' +
                                  ('all %d frames pass' % len(frs) if len(frs) != 1 else 'the frame passes'))
        elif n_fail == 0 and not any(rp.limits[k]['sl'].value() for k, _v, _k2, _l in limits) and best is None:
            rp.pass_label.setText(('All %d frames pass' % len(frs) if len(frs) != 1 else 'The frame passes') +
                                  ' - press Suggest limits, or drag a slider')
        else:
            rp.pass_label.setText('%d of %s pass, %d fail' % (len(frs) - n_fail, n_frames(len(frs)), n_fail))
        rp.result.setText('')
        rp.apply.setEnabled(n_fail > 0)
        self.graph_panel.sync()
        if self.cur is not None:
            self.show_frame(self.cur, sync_table=False)

    def apply_auto_reject(self):
        n = 0
        for f in self.frames:
            if f.reasons:
                f.mark = 'bad'
                n += 1
            elif self.reject_panel.rest_good.isChecked() and f.status == 'ok' and not f.mark:
                f.mark = 'good'
        self.model.refresh_all()
        self.graphs_refresh()
        self.reject_panel.result.setText('%d frame%s marked Bad. Use "Sort into folders" to move them.'
                                         % (n, '' if n == 1 else 's'))
        if self.reject_panel.is_manual():
            self.reject_panel.pass_label.setText('')
        self.reject_panel.apply.setEnabled(False)
        self.reject_summary = '%d marked Bad' % n
        self.graph_panel.sync()
        if self.cur is not None:
            self.show_frame(self.cur)

    def graphs_refresh(self, *_a):
        self.graph_panel.refresh()

    # ============================================================ move
    def _busy_measuring(self, what):
        if not self._still_measuring():
            return False
        QMessageBox.information(self, APP_NAME, 'Still measuring - please wait until every frame is done '
                                'before you %s.' % what)
        return True

    def move_dialog(self):
        frs = [f for f in self.frames]
        if not frs:
            return
        if self._busy_measuring('sort them into folders'):
            return
        n_good = sum(1 for f in frs if f.mark == 'good')
        n_bad = sum(1 for f in frs if f.mark == 'bad')
        n_un = len(frs) - n_good - n_bad
        n_ser = sum(1 for f in frs if f.src.ser is not None and not is_video(f.src))
        dlg = MoveDialog(self, n_good, n_bad, n_un, n_ser, self.settings)
        if dlg.exec() != QDialog.Accepted:
            return
        s = self.settings
        good_name = dlg.good_name.text().strip()
        bad_name = dlg.bad_name.text().strip()
        s.setValue('move/good', good_name)
        s.setValue('move/bad', bad_name)
        s.setValue('move/base', dlg.base.text())
        copy = dlg.copy.isChecked()
        s.setValue('move/mode', 'copy' if copy else 'move')
        s.setValue('move/unmarked_good', dlg.unmarked_good.isChecked())
        base = dlg.base.text() if dlg.w_pick.isChecked() else None

        def dest_mark(f):
            if f.mark:
                return f.mark
            return 'good' if dlg.unmarked_good.isChecked() and f.status == 'ok' else ''
        plan = []
        ser_groups = collections.defaultdict(lambda: {'good': [], 'bad': []})
        n_vid = 0
        for f in frs:
            m = dest_mark(f)
            if not m:
                continue
            if is_video(f.src):
                n_vid += 1          # frames inside AVI / MP4 / MOV cannot be sorted (yet)
                continue
            if f.src.ser is not None:
                ser_groups[f.src.path][m].append(f)
                continue
            folder = os.path.join(base or os.path.dirname(f.src.path), good_name if m == 'good' else bad_name)
            plan.append((f, folder))
        vid_note = ('\n\n%d marked frame%s inside video files (AVI, MP4, MOV) %s left out: frames inside a '
                    'video cannot be sorted into folders.' % (n_vid, '' if n_vid == 1 else 's',
                                                               'is' if n_vid == 1 else 'are')) if n_vid else ''
        if not plan and not ser_groups:
            QMessageBox.information(self, APP_NAME, ('No frames are marked, so there is nothing to sort.'
                                                     if not n_vid else 'Nothing to sort.' + vid_note))
            return
        n_files = len(plan)
        msg = '%s %d file%s into the %s and %s folders?' % ('Copy' if copy else 'Move', n_files,
                                                            '' if n_files == 1 else 's', good_name, bad_name)
        if ser_groups:
            n_new = sum(bool(g['good']) + bool(g['bad']) for g in ser_groups.values())
            msg += '\n\nAlso write %d new SER file%s.' % (n_new, '' if n_new == 1 else 's')
        msg += vid_note
        if QMessageBox.question(self, APP_NAME, msg) != QMessageBox.Yes:
            return
        self.timer.stop()
        self._set_play_icon(False)
        LOG.info('Sort into folders: %s %d files (Good folder "%s", Bad folder "%s", %s, unmarked as Good: %s, SER files: %d)',
                 'copying' if copy else 'moving', n_files, good_name, bad_name,
                 'inside a chosen folder' if base else "inside each file's own folder",
                 dlg.unmarked_good.isChecked(), len(ser_groups))
        done, errors = [], []
        for f, folder in plan:
            try:
                os.makedirs(folder, exist_ok=True)
                target = self._free_name(os.path.join(folder, os.path.basename(f.src.path)))
                if os.path.normcase(os.path.abspath(target)) == os.path.normcase(os.path.abspath(f.src.path)):
                    continue
                if copy:
                    shutil.copy2(f.src.path, target)
                else:
                    shutil.move(f.src.path, target)
                    done.append((target, f.src.path, f.uid))
                    f.src.path = target
            except Exception as e:
                errors.append('%s: %s' % (os.path.basename(f.src.path), e))
                LOG.warning('  could not %s %s: %s', 'copy' if copy else 'move', fname(f.src.path), e)
        n_ser_written = 0
        for path, g in ser_groups.items():
            info = read_ser_info(path)
            stem = os.path.splitext(os.path.basename(path))[0]
            for m, lst in g.items():
                if not lst:
                    continue
                folder = os.path.join(base or os.path.dirname(path), good_name if m == 'good' else bad_name)
                try:
                    os.makedirs(folder, exist_ok=True)
                    target = self._free_name(os.path.join(folder, '%s_%s.ser' % (stem, m)))
                    idx = sorted(f.src.index for f in lst)
                    write_ser(path, info, idx, target)
                    n_ser_written += 1
                except Exception as e:
                    errors.append('%s: %s' % (os.path.basename(path), e))
                    LOG.exception('  could not write the %s SER from %s', m, fname(path))
        if not copy:
            self.last_move = done
            self.a_undo.setEnabled(bool(done))
        self.model.refresh_all()
        txt = '%s %d file%s.' % ('Copied' if copy else 'Moved', len(plan) - len(errors),
                                 '' if len(plan) == 1 else 's')
        if n_ser_written:
            txt += ' Wrote %d SER file%s.' % (n_ser_written, '' if n_ser_written == 1 else 's')
        LOG.info('Sort finished: %s %d problem%s', txt, len(errors), '' if len(errors) == 1 else 's')
        if errors:
            QMessageBox.warning(self, APP_NAME, txt + '\n\nProblems:\n' + '\n'.join(errors[:20]))
        else:
            self.status.setText(txt + (' Ctrl+Z to undo.' if done else ''))

    @staticmethod
    def _free_name(path):
        if not os.path.exists(path):
            return path
        stem, ext = os.path.splitext(path)
        k = 1
        while os.path.exists('%s_%d%s' % (stem, k, ext)):
            k += 1
        return '%s_%d%s' % (stem, k, ext)

    def undo_move(self):
        if not self.last_move:
            return
        if self._busy_measuring('undo the move'):
            return
        if QMessageBox.question(self, APP_NAME, 'Put the %d moved files back where they were?'
                                % len(self.last_move)) != QMessageBox.Yes:
            return
        errors = []
        LOG.info('Undo move: putting back %d files', len(self.last_move))
        for new, old, uid in reversed(self.last_move):
            try:
                if os.path.exists(old):
                    raise OSError('a file with that name is already there')
                shutil.move(new, old)
                fr = self.by_uid.get(uid)
                if fr is not None:
                    fr.src.path = old
            except Exception as e:
                errors.append('%s: %s' % (os.path.basename(new), e))
                LOG.warning('  could not put back %s: %s', fname(new), e)
        LOG.info('Undo finished, %d problem%s', len(errors), '' if len(errors) == 1 else 's')
        self.last_move = []
        self.a_undo.setEnabled(False)
        self.model.refresh_all()
        if errors:
            QMessageBox.warning(self, APP_NAME, 'Some files could not be put back:\n' + '\n'.join(errors[:20]))
        else:
            self.status.setText('Files put back. The empty Good / Bad folders are left in place.')

    # ============================================================ export
    def export_csv(self):
        if not self.frames:
            return
        path, _f = QFileDialog.getSaveFileName(self, 'Save statistics', os.path.join(
            self.settings.value('last_dir', ''), 'AS_Blink_stats.csv'), 'CSV (*.csv)')
        if not path:
            return
        import csv
        keys = [c[0] for c in COLUMNS]
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow([c[1] for c in COLUMNS] + ['Path'])
            for u in self.order():
                f = self.by_uid[u]
                row = []
                for k in keys:
                    v = self.model.value(f, k)
                    row.append('' if v is None else (_fmt(k, v) if isinstance(v, (int, float)) and k != 'idx' else v))
                row.append(f.src.path)
                w.writerow(row)
        LOG.info('Saved statistics CSV %s (%s)', fname(path), n_frames(len(self.frames)))
        self.status.setText('Saved %s' % path)

    def export_anim(self):
        ok = [f for f in self.frames if f.status == 'ok']
        if len(ok) < 2:
            QMessageBox.information(self, APP_NAME, 'Need at least two measured frames.')
            return
        sel = {self.frames[self.proxy.mapToSource(i).row()].uid for i in self.table.selectionModel().selectedRows()}
        n_good = sum(1 for f in ok if f.mark == 'good')
        dlg = ExportDialog(self, len(sel), n_good, len(ok), self.settings)
        if dlg.exec() != QDialog.Accepted:
            return
        which = dlg.which()
        order = [u for u in self.order() if self.by_uid[u].status == 'ok']
        if which == 'selected':
            uids = [u for u in order if u in sel]
        elif which == 'good':
            uids = [u for u in order if self.by_uid[u].mark == 'good']
        elif which == 'notbad':
            uids = [u for u in order if self.by_uid[u].mark != 'bad']
        else:
            uids = order
        if len(uids) < 2:
            QMessageBox.information(self, APP_NAME, 'Need at least two frames for an animation.')
            return
        is_gif = dlg.fmt.currentIndex() == 0
        self.settings.setValue('export/width', dlg.width.value())
        self.settings.setValue('export/fps', dlg.fps.value())
        self.settings.setValue('export/labels', dlg.labels.isChecked())
        ext = '.gif' if is_gif else '.mp4'
        first = self.by_uid[uids[0]].src.path
        path, _f = QFileDialog.getSaveFileName(self, 'Save animation', os.path.join(
            os.path.dirname(first), 'AS_Blink' + ext), ('GIF (*.gif)' if is_gif else 'MP4 video (*.mp4)'))
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        crop = None
        if dlg.crop.isChecked():
            v = self.view
            x0, y0 = v._to_ref(QPointF(0, 0))
            x1, y1 = v._to_ref(QPointF(v.width(), v.height()))
            crop = (max(0.0, x0), max(0.0, y0), min(float(v.rw), x1), min(float(v.rh), y1))
        self.timer.stop()
        self._set_play_icon(False)
        LOG.info('Making %s: %s, width %d, %.1f fps, align %s, labels %s, crop %s', ext[1:].upper(),
                 n_frames(len(uids)), dlg.width.value(), dlg.fps.value(), dlg.align.isChecked(),
                 dlg.labels.isChecked(), crop is not None)
        try:
            n = self._write_anim(path, uids, dlg.width.value(), dlg.fps.value(), dlg.align.isChecked(),
                                 dlg.labels.isChecked(), crop, is_gif)
            if n:
                self.status.setText('Saved %s (%s)' % (path, n_frames(n)))
            LOG.info('  saved %s (%s)', fname(path), n_frames(n or 0))
        except Exception as e:
            LOG.exception('  making the animation failed')
            QMessageBox.warning(self, APP_NAME, 'Could not make the animation:\n%s' % e)

    def _frame_rgb8(self, fr, align, crop, out_w, label, idx, total):
        hit = self.pcache.get(fr.uid)
        if hit is None:
            pv, ps = load_preview(fr.src, self.opts)
            self.pcache.put(fr.uid, (pv, ps), pinned=self._pinned())
        else:
            pv, ps = hit
        luts = self.stretch.luts(fr.pstats)
        img = apply_luts(pv, luts)
        if img.ndim == 2:
            img = np.dstack([img] * 3)
        rf = self.by_uid.get(self.ref)
        rw = rf.an.width if rf else fr.an.width
        rh = rf.an.height if rf else fr.an.height
        # map into reference coordinates at preview scale
        s = 1.0 / ps
        dx = dy = 0.0
        flipped = False
        if align and fr.shift is not None:
            dx, dy, flipped = fr.shift
        W, H = int(round(rw * s)), int(round(rh * s))
        if flipped:
            # ref(x) = rot(x + d) with rot(u) = img(fw - u)
            M = np.array([[-1, 0, (fr.an.width - dx) * s], [0, -1, (fr.an.height - dy) * s]], np.float32)
        else:
            M = np.array([[1, 0, -dx * s], [0, 1, -dy * s]], np.float32)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if crop is not None:
            x0, y0, x1, y1 = [int(round(v * s)) for v in crop]
            if x1 - x0 > 8 and y1 - y0 > 8:
                img = img[y0:y1, x0:x1]
        h, w = img.shape[:2]
        f = out_w / float(w)
        oh = max(2, int(round(h * f)) // 2 * 2)
        ow = max(2, out_w // 2 * 2)
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        if label:
            txt = '%d/%d  %s' % (idx + 1, total, fr.name)
            scale = max(0.4, ow / 1600.0)
            while scale > 0.3 and cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > ow - 20:
                scale *= 0.9
            th = max(1, int(round(scale * 2)))
            cv2.putText(img, txt, (10, int(28 * scale) + 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(img, txt, (10, int(28 * scale) + 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), th, cv2.LINE_AA)
        return np.ascontiguousarray(img)

    def _write_anim(self, path, uids, out_w, fps, align, labels, crop, is_gif):
        from PySide6.QtWidgets import QProgressDialog
        prog = QProgressDialog('Making animation...', 'Cancel', 0, len(uids), self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        frames = []
        writer = None
        for i, u in enumerate(uids):
            prog.setValue(i)
            QApplication.processEvents()
            if prog.wasCanceled():
                if writer is not None:
                    writer.release()
                return 0
            img = self._frame_rgb8(self.by_uid[u], align, crop, out_w, labels, i, len(uids))
            if is_gif:
                frames.append(Image.fromarray(img).quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                                                             dither=Image.Dither.NONE))
            else:
                if writer is None:
                    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), float(fps),
                                             (img.shape[1], img.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('This system cannot write MP4 video')
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        prog.setValue(len(uids))
        if is_gif:
            if Image is None:
                raise RuntimeError('Pillow is needed for GIFs')
            dur = int(round(1000.0 / fps))
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0,
                           optimize=False, disposal=1)
        elif writer is not None:
            writer.release()
        return len(uids)

    def save_log(self):
        for h in LOG.handlers:
            try:
                h.flush()
            except Exception:
                pass
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        start = os.path.join(os.path.expanduser('~'), 'Desktop')
        if not os.path.isdir(start):
            start = os.path.expanduser('~')
        path, _f = QFileDialog.getSaveFileName(self, 'Save log for support',
                                               os.path.join(start, 'AS_Blink_log_%s.txt' % stamp), 'Text (*.txt)')
        if not path:
            return
        n_ok = sum(1 for f in self.frames if f.status == 'ok')
        n_err = sum(1 for f in self.frames if f.status == 'error')
        lines = ['AS Blink log for support - saved %s' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                 'Please send this file to AstroShed with a few words on what you were doing.', '']
        lines += system_summary()
        scr = QApplication.primaryScreen()
        if scr is not None:
            g = scr.geometry()
            lines.append('Screen: %d x %d, scale %.2f' % (g.width(), g.height(), scr.devicePixelRatio()))
        lines += ['Window: %d x %d' % (self.width(), self.height()),
                  'Frames loaded now: %d (%d measured, %d could not be read, %d still to do)'
                  % (len(self.frames), n_ok, n_err, len(self.frames) - n_ok - n_err),
                  'Settings now: ' + self._log_settings(),
                  'Limits now: ' + self.using_text(),
                  'Log folder: ' + _private(log_dir()), '', '=' * 70, '']
        body = []
        for p in log_files():
            try:
                with open(p, 'r', encoding='utf-8', errors='replace') as fh:
                    body.append(fh.read())
            except Exception as e:
                body.append('(could not read %s: %s)\n' % (fname(p), e))
        if not body:
            body = ['(no log file was found - it could not be written to %s)\n' % _private(log_dir())]
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(_private('\n'.join(lines)) + '\n' + ''.join(body))
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, 'Could not save the log:\n%s' % e)
            return
        self.status.setText('Log saved as %s - please send it to AstroShed' % os.path.basename(path))

    def open_log_folder(self):
        d = log_dir()
        if not os.path.isdir(d):
            QMessageBox.information(self, APP_NAME, 'There is no log folder yet.')
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def show_help(self):
        d = QDialog(self)
        d.setWindowTitle('%s help' % APP_NAME)
        d.resize(640, 720)
        lay = QVBoxLayout(d)
        tb = QTextBrowser()
        tb.setOpenExternalLinks(True)
        tb.setHtml(HELP_HTML)
        lay.addWidget(tb)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(d.reject)
        lay.addWidget(bb)
        d.exec()


def dark_palette(app):
    apply_palette(app, 'Dark')


ICON_PNG = (
    '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'
    'xIEAAAAASUVORK5CYII='
)


def app_icon():
    pm = QPixmap()
    try:
        pm.loadFromData(base64.b64decode(ICON_PNG), 'PNG')
    except Exception:
        pass
    return QIcon(pm)


_last_crash_box = [0.0]


def _excepthook(etype, value, tb):
    text = ''.join(traceback.format_exception(etype, value, tb))
    LOG.critical('Unexpected error:\n%s', text.rstrip())
    try:
        sys.__excepthook__(etype, value, tb)
    except Exception:
        pass
    app = QApplication.instance()
    if app is None or time.time() - _last_crash_box[0] < 10:
        return
    _last_crash_box[0] = time.time()
    win = next((w for w in app.topLevelWidgets() if isinstance(w, MainWindow)), None)
    try:
        box = QMessageBox(win)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(APP_NAME)
        box.setText('Something went wrong:\n%s: %s\n\nThe details have been written to the log. '
                    'If it keeps happening, please save the log and send it to AstroShed.'
                    % (etype.__name__, value))
        save = box.addButton('Save log...', QMessageBox.ActionRole) if win is not None else None
        box.addButton(QMessageBox.Ok)
        box.exec()
        if save is not None and box.clickedButton() is save:
            win.save_log()
    except Exception:
        pass


def _thread_excepthook(args):
    LOG.critical('Unexpected error in a background thread:\n%s', ''.join(
        traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)).rstrip())


def _qt_messages(mode, context, message):
    if 'propagateSizeHints' in message:
        return
    if mode == QtMsgType.QtFatalMsg or mode == QtMsgType.QtCriticalMsg:
        LOG.error('Qt: %s', message)
    elif mode == QtMsgType.QtWarningMsg:
        LOG.warning('Qt: %s', message)


def start_logging():
    setup_log()
    LOG.info('=' * 60)
    for line in system_summary():
        LOG.info(line)
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    try:
        qInstallMessageHandler(_qt_messages)
    except Exception:
        pass


def main():
    start_logging()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName('AstroShed')
    apply_palette(app, str(QSettings('AstroShed', 'AS Blink').value('theme', 'Dark')))
    app.setWindowIcon(app_icon())
    w = MainWindow()
    w.show()
    scr = app.primaryScreen()
    if scr is not None:
        g = scr.geometry()
        LOG.info('Screen %d x %d, scale %.2f, theme %s', g.width(), g.height(), scr.devicePixelRatio(), THEME.get('name'))
    args = [a for a in sys.argv[1:] if os.path.exists(a)]
    if args:
        files = []
        for a in args:
            files += w._files_in(a) if os.path.isdir(a) else [a]
        QTimer.singleShot(100, lambda: w.add_paths(files))
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
