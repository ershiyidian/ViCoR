import inspect
import ast
import asyncio
import base64
import collections
import copy
import heapq
import io
import json
import math
import os
import re
import time
import traceback
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw

from .roles import resolve_semantic_action
from .trigger_template import _detect as _ut_detect, _sig8 as _ut_sig8


COUNTERS = ('trigger_count', 'query_count', 'action_count', 'none_count',
            'escalate_count', 'error_count', 'stale_count')

TRIGGER_CONTRACT = """Python code defining `async def run(api)` and calling `await api.run_rules(rules)`.
Each rule matches a cropped image from `images` (or `"mode": "motion"` for moving objects) and uses a Python function `"prompt": decide` (`decide(target, player) -> str | None`) that returns `None` when out of range or a short instruction naming the single key/action to take:
```python
def decide(target, player):
    if target["dx"] < -15 or target["dx"] > 220:
        return None
    if target["dy"] < -15:
        return "High obstacle ahead. Press ArrowDown to duck."
    return "Ground obstacle ahead. Press Space to jump."

async def run(api):
    await api.run_rules([
        {
            "name": "dodge",
            "image": "obstacle",         # crop name in `images`
            "player": "player",          # optional player crop name in `images`, or (x, y)
            "roi": None,                 # optional (x, y, w, h) search region
            "cooldown_s": 0.20,
            "prompt": decide,
        }
    ])
```
`target` has `"cx"`, `"cy"`, `"dx"` (`target["cx"] - player["cx"]`), `"dy"` (`target["cy"] - player["cy"]`), `"vx"`, `"vy"`, and `"box"` `(x, y, w, h)`. `player` has `"cx"`, `"cy"`. Return `None` when no action is needed yet."""


def _observation_times(frame, previous):
    scene = getattr(frame, 'scene_time_s', None)
    prior_scene = getattr(previous, 'scene_time_s', None)
    # Keep unsampled scene timestamps as None rather than substituting wall-clock time.
    return {'time_s': frame.timestamp,
            'dt_s': frame.timestamp-previous.timestamp if previous is not None else 0.0,
            'scene_time_s': scene,
            'scene_dt_s': (None if scene is None else
                           scene-prior_scene if prior_scene is not None else 0.0)}


def _decode_program_images(images):
    if not isinstance(images, dict):
        raise ValueError('images must be a mapping of names to bound images')
    decoded = {}
    for name, bound in images.items():
        with Image.open(io.BytesIO(base64.b64decode(bound['png_base64'], validate=True))) as source:
            pixels = np.asarray(source.convert('RGB'))
        if pixels.shape[:2] != (bound['height'], bound['width']):
            raise ValueError(f'Bound image dimensions do not match: {name}')
        decoded[name] = pixels
    return decoded


def _image_references(images):
    fields = ('event', 'moment', 'view', 'region', 'width', 'height')
    return {name: copy.deepcopy({key: bound[key] for key in fields})
            for name, bound in images.items()}


_SAFE_FUNCTIONS = {
    # Scalar and arithmetic helpers.
    'abs': abs, 'bool': bool, 'complex': complex, 'divmod': divmod,
    'float': float, 'int': int, 'max': max, 'min': min, 'round': round, 'sum': sum,
    # Containers, indexing and iteration over values already available to the program.
    'all': all, 'any': any, 'enumerate': enumerate, 'isinstance': isinstance, 'hasattr': hasattr,
    'dict': dict, 'list': list, 'len': len, 'iter': iter, 'next': next, 'slice': slice,
    'range': range,
    'reversed': reversed, 'set': set, 'frozenset': frozenset,
    'sorted': sorted, 'str': str, 'tuple': tuple, 'zip': zip,
    'map': map, 'filter': filter, 'Exception': Exception, 'ValueError': ValueError,
    'TypeError': TypeError, 'IndexError': IndexError, 'KeyError': KeyError,
    'ArithmeticError': ArithmeticError, 'ZeroDivisionError': ZeroDivisionError,
    'FloatingPointError': FloatingPointError, 'OverflowError': OverflowError,
    'StopIteration': StopIteration,
}
# Expose pure array CV operations without file or camera I/O.
_CV_ARRAY_API = set('''
    add subtract multiply divide scaleAdd addWeighted absdiff convertScaleAbs LUT
    bitwise_and bitwise_or bitwise_xor bitwise_not compare min max inRange
    sqrt pow exp log magnitude phase cartToPolar polarToCart normalize norm
    mean meanStdDev sumElems countNonZero findNonZero minMaxLoc checkRange
    split merge mixChannels extractChannel insertChannel copyTo flip rotate repeat
    reduce reduceArgMin reduceArgMax sort sortIdx transpose hconcat vconcat
    copyMakeBorder transform perspectiveTransform determinant invert solve eigen
    dft idft dct idct mulSpectrums getOptimalDFTSize
    cvtColor cvtColorTwoPlane threshold adaptiveThreshold equalizeHist calcHist
    compareHist calcBackProject blur boxFilter sqrBoxFilter GaussianBlur medianBlur
    bilateralFilter filter2D sepFilter2D getGaussianKernel getDerivKernels
    Sobel Scharr Laplacian Canny spatialGradient morphologyEx dilate erode
    getStructuringElement resize pyrDown pyrUp buildPyramid remap convertMaps
    warpAffine warpPerspective getRotationMatrix2D getAffineTransform getPerspectiveTransform
    matchTemplate connectedComponents connectedComponentsWithStats findContours
    boundingRect contourArea moments arcLength approxPolyDP convexHull isContourConvex
    convexityDefects pointPolygonTest minAreaRect boxPoints minEnclosingCircle fitEllipse fitLine
    distanceTransform distanceTransformWithLabels floodFill watershed
    cornerHarris cornerMinEigenVal cornerEigenValsAndVecs cornerSubPix goodFeaturesToTrack
    HoughLines HoughLinesP HoughCircles calcOpticalFlowPyrLK calcOpticalFlowFarneback
    rectangle circle ellipse line polylines fillPoly fillConvexPoly drawContours
    imencode imdecode error
'''.split())
_NP_ARRAY_API = {
    'creation': '''array asarray asanyarray asarray_chkfinite ascontiguousarray asfortranarray
        require copy empty empty_like zeros zeros_like ones ones_like full full_like eye identity
        arange linspace logspace geomspace meshgrid frombuffer fromiter fromstring''',
    'shape': '''shape ndim size reshape ravel transpose matrix_transpose swapaxes moveaxis
        expand_dims squeeze atleast_1d atleast_2d atleast_3d broadcast_to broadcast_arrays broadcast_shapes
        stack hstack vstack dstack column_stack row_stack block concatenate concat unstack
        split array_split hsplit vsplit dsplit repeat tile resize pad roll flip fliplr flipud rot90''',
    'indexing': '''indices unravel_index ravel_multi_index ix_ newaxis ndindex ndenumerate
        take take_along_axis put put_along_axis putmask copyto place delete insert append trim_zeros
        where select choose compress extract nonzero flatnonzero argwhere count_nonzero
        diag diagonal diagflat tri tril triu diag_indices diag_indices_from fill_diagonal
        tril_indices tril_indices_from triu_indices triu_indices_from mask_indices''',
    'reductions': '''sum prod cumsum cumprod cumulative_sum cumulative_prod nancumsum nancumprod
        mean average median percentile quantile std var ptp min max amin amax
        nanmin nanmax nansum nanprod nanmean nanmedian nanpercentile nanquantile nanstd nanvar
        all any argmin argmax nanargmin nanargmax''',
    'ordering': '''sort argsort lexsort partition argpartition searchsorted unique
        unique_all unique_counts unique_inverse unique_values isin in1d intersect1d union1d setdiff1d setxor1d
        bincount digitize histogram histogram2d histogramdd histogram_bin_edges''',
    'arithmetic': '''clip round around fix real imag angle real_if_close nan_to_num
        isclose allclose array_equal array_equiv diff ediff1d gradient unwrap
        correlate convolve interp trapezoid trapz sinc i0 packbits unpackbits''',
    'polynomials': '''poly roots polyint polyder polyadd polysub
        polymul polydiv polyval poly1d polyfit''',
    'products': '''dot vdot vecdot inner outer matmul matvec vecmat tensordot einsum einsum_path
        trace cross kron cov corrcoef vander''',
    'types': '''dtype ndarray generic number integer floating complexfloating finfo iinfo
        result_type promote_types min_scalar_type can_cast issubdtype isscalar
        iscomplex isreal iscomplexobj isrealobj isfortran vectorize frompyfunc errstate''',
}
_EXTERNAL_ATTRIBUTES = {'ctypes', 'dump', 'tofile', 'save', 'load', 'read', 'write',
                        'format', 'format_map'}
_RUNTIME_ATTRIBUTE_PREFIXES = ('_', 'f_', 'gi_', 'cr_', 'ag_', 'tb_', 'co_')


def _image_modules():
    # Each trigger program receives an isolated capability object without host I/O.
    def namespace(module, names):
        return SimpleNamespace(**{name: getattr(module, name) for name in names
                                  if not name.startswith('_') and hasattr(module, name)})
    numeric = {name for name, value in vars(np).items() if not name.startswith('_') and (
        isinstance(value, (np.ufunc, int, float, complex)) or
        (isinstance(value, type) and (issubclass(value, np.number) or value is np.bool_)))}
    arrays = namespace(np, set(' '.join(_NP_ARRAY_API.values()).split()) | numeric)
    arrays.linalg = namespace(np.linalg, np.linalg.__all__)
    arrays.fft = namespace(np.fft, np.fft.__all__)
    constants = {name for name, value in vars(cv2).items() if isinstance(value, (int, float))}
    return {'numpy': arrays, 'cv2': namespace(cv2, _CV_ARRAY_API | constants),
            'math': namespace(math, vars(math)),
            'collections': namespace(collections, ('deque', 'defaultdict', 'Counter')),
            'heapq': namespace(heapq, ('heapify', 'heappush', 'heappop', 'heappushpop',
                                      'heapreplace', 'nlargest', 'nsmallest')),
            're': namespace(re, ('search', 'match', 'findall', 'sub', 'split', 'compile', 'IGNORECASE', 'I'))}


async def _cooperate():
    # Even a continue branch that skips user awaits must yield to cancellation and input timers.
    await asyncio.sleep(0)
    if asyncio.current_task().cancelling():
        raise asyncio.CancelledError()


class _CooperativeLoops(ast.NodeTransformer):
    asynchronous = False

    def visit_AsyncFunctionDef(self, node):
        previous, self.asynchronous = self.asynchronous, True
        self.generic_visit(node)
        self.asynchronous = previous
        return node

    def visit_FunctionDef(self, node):
        previous, self.asynchronous = self.asynchronous, False
        self.generic_visit(node)
        self.asynchronous = previous
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        if self.asynchronous:
            checkpoint = ast.Expr(ast.Await(ast.Call(ast.Name('__reflex_yield', ast.Load()), [], [])))
            node.body.insert(0, ast.copy_location(checkpoint, node))
        return node

    visit_While = visit_For


def compile_trigger(source: str):
    """Compile a cooperatively cancellable async visual trigger program."""
    if not isinstance(source, str) or not source.strip() or len(source) > 12000:
        raise ValueError('trigger_py must define async def run(api) (max 12k characters)')
    tree = ast.parse(source, '<reflex-trigger>', 'exec')
    declarations = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom,
                    ast.Assign, ast.AnnAssign)
    if any(not isinstance(node, declarations) and not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)) for node in tree.body):
        raise ValueError('Use async def run(api), constants, helpers and array imports')
    entries = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == 'run']
    if len(entries) != 1 or not isinstance(entries[0], ast.AsyncFunctionDef):
        raise ValueError('Define async def run(api)')
    args = entries[0].args
    if len(args.posonlyargs + args.args) != 1 or args.vararg or args.kwarg or args.kwonlyargs:
        raise ValueError('run must take exactly one argument')
    forbidden = (ast.AsyncWith, ast.AsyncFor, ast.ClassDef, ast.Yield, ast.YieldFrom)
    modules = _image_modules()
    roots = set(modules)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f'Unsupported trigger_py syntax: {type(node).__name__}')
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            raise ValueError('Catch Exception or a specific error; bare except prevents cancellation')
        if isinstance(node, ast.Import) and any(alias.name not in roots for alias in node.names):
            raise ValueError('Available imports: ' + ', '.join(sorted(roots)))
        if isinstance(node, ast.ImportFrom) and (node.level or node.module not in roots
                or any(alias.name.startswith('_') for alias in node.names)):
            raise ValueError('Import public capabilities from: ' + ', '.join(sorted(roots)))
        name = (node.id if isinstance(node, ast.Name) else node.arg if isinstance(node, ast.arg)
                else node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                else node.asname or '' if isinstance(node, ast.alias) else '')
        if name.startswith('__') or name in {'eval', 'exec', 'open', 'globals', 'locals', 'compile',
                                          'getattr', 'setattr', '__import__'}:
            raise ValueError('Execution, file and private runtime APIs are unavailable')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'observe' and (node.args or node.keywords):
            raise ValueError('api.observe() takes no arguments; use `await api.run_rules([{...}])` with declarative rule dicts instead')
        if isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value in ('action', 'action_id') for k in node.keys
        ):
            raise ValueError('Python cannot construct raw action dicts; write natural-language guidance (supporting if/else) in "prompt" to steer the fast visual model, which makes the final action decision')
        if isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value in ('prompt', 'guide', 'decide', 'condition')
            and isinstance(v, ast.Lambda) and isinstance(v.body, ast.Constant) and (v.body.value is None or v.body.value is False or v.body.value == '')
            for k, v in zip(node.keys, node.values)
        ):
            raise ValueError('Rule "prompt" lambda cannot unconditionally return None; return a natural-language instruction string for the fast visual model when an action may be needed')
        if isinstance(node, ast.Attribute) and (node.attr.startswith(_RUNTIME_ATTRIBUTE_PREFIXES)
                                               or node.attr in _EXTERNAL_ATTRIBUTES):
            raise ValueError(f'External or private attribute is unavailable: {node.attr}')
    def image_import(name, globals=None, locals=None, fromlist=(), level=0):
        if not level and name in modules:
            return modules[name]
        if not level and name.startswith(('numpy.', 'cv2.')):
            return __import__(name, globals, locals, fromlist or (), level)
        raise ImportError('Only declared array and algorithm capabilities are available')
    _def_fn = lambda target, player=None, state=None: 'React to the detected target.'
    scope = {'__builtins__': {**_SAFE_FUNCTIONS, '__import__': image_import},
             **modules, 'np': modules['numpy'], '__reflex_yield': _cooperate,
             'decide': _def_fn, 'guide': _def_fn}
    tree = ast.fix_missing_locations(_CooperativeLoops().visit(tree))
    exec(compile(tree, '<reflex-trigger>', 'exec'), scope)
    return scope['run']


def _normalize_hits(value, dimensions):
    """Convert one box, many boxes, or full-resolution binary mask to boxes."""
    if isinstance(value, dict):
        value = value['match']
    height, width = dimensions
    if value is None or (isinstance(value, (bool, np.bool_)) and not value):
        return []
    if isinstance(value, np.ndarray) and value.ndim == 2:
        if value.shape != (height, width):
            raise ValueError(f'Trigger mask must match original {(height, width)} frame')
        mask = np.ascontiguousarray((value != 0).astype(np.uint8))
        if not cv2.countNonZero(mask):
            return []
        # Disconnected objects need their OWN tight rectangles. A single
        # boundingRect(mask) may cover most of the static search corridor.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:32]
        return [tuple(map(int, cv2.boundingRect(contour))) for contour in contours]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (tuple, list)) and len(value) == 4 and all(
            isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
            for v in value):
        value = [value]
    if not isinstance(value, (tuple, list)):
        raise ValueError('match must be None, False, (x,y,w,h), boxes, or a full-frame mask')
    if len(value) > 32:
        raise ValueError('At most 32 hit regions per trigger')
    result = []
    for rect in value:
        if (not isinstance(rect, (tuple, list)) or len(rect) != 4
                or any(not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(v)
                       for v in rect)):
            raise ValueError('Each trigger region must contain four finite numbers')
        x, y, w, h = map(int, rect)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
        if x1 > x0 and y1 > y0:
            result.append((x0, y0, x1-x0, y1-y0))
    return result


def _render_reflex_image(current, boxes, *, scene=False, meta=None):
    # Pair full-scene context with motion-annotated local candidate crops.
    if scene:
        boxes = []
    current = np.asarray(current, dtype=np.uint8)
    source_height, source_width = current.shape[:2]
    left, top, right, bottom = 0, 0, source_width, source_height
    by_box = (meta or {}).get('by_box') or {}
    anc_meta = (meta or {}).get('anc')
    vis_mode = (meta or {}).get('vis_mode') or os.environ.get('VICOR_VIS_MODE', 'trail_ttc')
    if vis_mode == 'ghosting' and not scene:
        prevs = [p for p in ((meta or {}).get('rgb_hist') or ()) if isinstance(p, np.ndarray) and p.shape == current.shape]
        if len(prevs) >= 2:
            ghost = cv2.addWeighted(prevs[-1], 0.65, prevs[-2], 0.35, 0.0)
            current = cv2.addWeighted(current, 0.55, ghost, 0.45, 0.0)
        elif len(prevs) == 1:
            current = cv2.addWeighted(current, 0.65, prevs[-1], 0.35, 0.0)
    if boxes and not scene:
        x0, y0 = min(x for x, y, w, h in boxes), min(y for x, y, w, h in boxes)
        x1, y1 = max(x+w for x, y, w, h in boxes), max(y+h for x, y, w, h in boxes)
        for bx, by, bw, bh in (boxes if vis_mode == 'trail_ttc' else ()):
            g = by_box.get((int(bx), int(by), int(bw), int(bh)))
            if g:
                vx, vy = float(g.get('vx', 0.0) or 0.0), float(g.get('vy', 0.0) or 0.0)
                if math.hypot(vx, vy) >= 20.0:
                    px, py = bx + 0.5 * bw + vx * 0.20, by + 0.5 * bh + vy * 0.20
                    x0, y0 = min(x0, int(px - 0.5 * bw)), min(y0, int(py - 0.5 * bh))
                    x1, y1 = max(x1, int(px + 0.5 * bw)), max(y1, int(py + 0.5 * bh))
        if anc_meta and 'cx' in anc_meta and 'cy' in anc_meta:
            ax, ay = float(anc_meta['cx']), float(anc_meta['cy'])
            if math.hypot(ax - 0.5 * (x0 + x1), ay - 0.5 * (y0 + y1)) <= 360.0:
                x0, y0 = min(x0, int(ax - 24)), min(y0, int(ay - 24))
                x1, y1 = max(x1, int(ax + 24)), max(y1, int(ay + 24))
        margin = max(160, x1-x0, y1-y0)
        left, top = max(0, x0-margin), max(0, y0-margin)
        right, bottom = min(source_width, x1+margin), min(source_height, y1+margin)
    region = {'x': left, 'y': top, 'width': right-left, 'height': bottom-top}
    pixels = current[top:bottom, left:right]
    image = pixels.copy()
    height, width = image.shape[:2]
    max_width, max_height = (224, 140) if scene else (224, 224)
    scale = min(1, max_width/width, max_height/height)
    result = Image.fromarray(image)
    if scale < 1:
        result = result.resize((max(1, round(width*scale)), max(1, round(height*scale))),
                               Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(result)
    def _to_pt(sx, sy):
        return (round((sx - left) * result.width / max(1, width)),
                round((sy - top) * result.height / max(1, height)))
    if not scene and anc_meta and 'cx' in anc_meta and 'cy' in anc_meta:
        ax, ay = float(anc_meta['cx']), float(anc_meta['cy'])
        if left <= ax <= right and top <= ay <= bottom:
            sax, say = _to_pt(ax, ay)
            draw.ellipse((sax-4, say-4, sax+4, say+4), outline=(0, 230, 64), width=2)
            draw.line((sax-6, say, sax+6, say), fill=(0, 230, 64), width=1)
            draw.line((sax, say-6, sax, say+6), fill=(0, 230, 64), width=1)
    for index, (x, y, w, h) in enumerate(boxes, 1):
        x0, y0 = _to_pt(x, y)
        x1, y1 = _to_pt(x + w, y + h)
        x1, y1 = x1 - 1, y1 - 1
        g = by_box.get((int(x), int(y), int(w), int(h))) if (not scene and vis_mode == 'trail_ttc') else None
        if g is not None:
            vx, vy = float(g.get('vx', 0.0) or 0.0), float(g.get('vy', 0.0) or 0.0)
            spd = math.hypot(vx, vy)
            if spd >= 20.0:
                cx, cy = x + 0.5 * w, y + 0.5 * h
                for dt_back, col_b in ((0.16, (255, 140, 0)), (0.08, (255, 190, 0))):
                    bx_p, by_p = _to_pt(cx - vx * dt_back, cy - vy * dt_back)
                    draw.ellipse((bx_p-2, by_p-2, bx_p+2, by_p+2), fill=col_b)
                px, py = cx + vx * 0.20, cy + vy * 0.20
                scx, scy = _to_pt(cx, cy)
                spx, spy = _to_pt(px, py)
                px0, py0 = _to_pt(px - 0.5 * w, py - 0.5 * h)
                px1, py1 = _to_pt(px + 0.5 * w, py + 0.5 * h)
                draw.rectangle((px0, py0, max(px0, px1), max(py0, py1)), outline=(0, 220, 255), width=1)
                draw.line((scx, scy, spx, spy), fill=(255, 230, 0), width=2)
                ang = math.atan2(spy - scy, spx - scx)
                for da in (2.6, -2.6):
                    wx = round(spx + 6.0 * math.cos(ang + da))
                    wy = round(spy + 6.0 * math.sin(ang + da))
                    draw.line((spx, spy, wx, wy), fill=(255, 230, 0), width=2)
        draw.rectangle((x0-2, y0-2, max(x0, x1)+2, max(y0, y1)+2),
                       outline=(220, 32, 32), width=2)
        draw.text((max(0, x0-2), max(0, y0-14)), str(index), fill=(255, 255, 255),
                  stroke_width=1, stroke_fill=(0, 0, 0))
    output = io.BytesIO()
    result.save(output, format='PNG', optimize=False)
    return output.getvalue(), {
        'region': region, 'resampling': 'bilinear' if scale < 1 else 'none',
        'source_size': {'width': source_width, 'height': source_height}}


class _Operation:
    """Observe completion and await a managed task without exposing host capabilities."""
    __slots__ = ('_task',)

    def __init__(self, task):
        self._task = task

    def done(self):
        return self._task.done()

    async def _wait(self):
        # Native dispatch must finish its bookkeeping before source cleanup releases input.
        return copy.deepcopy(await asyncio.shield(self._task))

    def __await__(self):
        return self._wait().__await__()

    def __deepcopy__(self, memo):
        return '<finished operation>' if self.done() else '<pending operation>'


class _ProgramAPI:
    def __init__(self, engine, name, state, card, episode_id, lifecycle):
        self._engine, self._name, self._state, self._card = engine, name, state, card
        self._stats = engine.stats[name]
        self._episode_id, self._lifecycle = episode_id, lifecycle
        self._stopped = False

    @property
    def images(self):
        return self._state['images']

    @property
    def state(self):
        return self._state['memory']

    @property
    def context(self):
        observation = self._state.get('observation')
        if observation is None:
            return {}
        frame, timing = observation
        return {'shared': self._engine.shared_state, 'images': dict(self.images),
                'reflex': self._engine._feedback(self._name),
                'last_input': self._engine._last_input(frame),
                **{key: timing[key] for key in ('time_s', 'dt_s', 'scene_time_s', 'scene_dt_s')}}

    def _check(self):
        engine = self._engine
        if (self._stopped or engine.closed or engine.state.get(self._name) is not self._state
                or engine.frames.episode_id != self._episode_id
                or engine.actions.lifecycle_epoch != self._lifecycle
                or engine.frames.phase != 'playing' or not engine.actions.can_run_reflexes):
            raise asyncio.CancelledError()

    async def match_template(self, frame, template, method=cv2.TM_CCOEFF_NORMED):
        self._check()

        def match(image, target):
            scores = cv2.matchTemplate(image, target, method)
            lo, hi, lo_pos, hi_pos = cv2.minMaxLoc(scores)
            use_min = method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)
            x, y = lo_pos if use_min else hi_pos
            h, w = target.shape[:2]
            return {"score": lo if use_min else hi,
                    "box": (x, y, w, h), "scores": scores}

        result = await asyncio.to_thread(match, frame.copy(), template.copy())
        self._check()
        return result

    async def observe(self):
        self._check()
        observation = self._state.get('observation')
        previous = observation[0] if observation is not None else None
        prior_scene = getattr(previous, 'scene_time_s', None) if previous is not None else None
        cursor = previous.frame_id if previous is not None else 0
        while True:
            frame = await self._engine.frames.next(after=cursor, captured_after=self._state.get('captured_after'))
            self._check()
            cursor = frame.frame_id
            captured_after = self._state.get('captured_after')
            frame_scene = getattr(frame, 'scene_time_s', None)
            scene_advanced = (
                self._engine.actions.clock_env is None
                or prior_scene is None
                or frame_scene is None
                or frame_scene > prior_scene + 1e-4
            )
            # An input may finish while frames.next is waiting with an older time boundary.
            if (frame.phase == 'playing'
                    and scene_advanced
                    and (captured_after is None or frame.timestamp > captured_after)
                    and self._engine._after_latest_input(frame)):
                break
        pixels = np.asarray(Image.open(io.BytesIO(frame.image)).convert('RGB'))
        self._state['observation'] = (frame, {'match_boxes': [], **_observation_times(frame, previous)})
        return pixels

    async def confirm(self, condition, match):
        self._check()
        api = self._state['perception_api']
        active = self._state['api']
        if active is not api:
            await self._engine._finish_reaction(active, keep_waits=True)
            self._state['api'] = api
        api._check()
        observation = self._state.get('observation')
        if observation is None:
            raise ValueError('observe() before confirm()')
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError('A candidate needs a nonempty visual condition')
        frame, info = observation
        with Image.open(io.BytesIO(frame.image)) as picture:
            boxes = _normalize_hits(match, (picture.height, picture.width))
        if not boxes:
            return None
        info.update(match_boxes=boxes, memory_40w=condition)
        confirmed = await self._engine._confirm(api, frame, condition, boxes)
        if confirmed:
            reaction = _ReactionAPI(self._engine, self._name, self._state, self._card,
                                    self._episode_id, self._lifecycle)
            reaction._box = tuple(confirmed['box'])
            reaction._action = copy.deepcopy(confirmed.get('action'))
            self._state['api'] = reaction
            return reaction
        return None

    def note(self, key_or_text, value=None):
        if value is not None:
            self._state['memory'][str(key_or_text)] = value
        elif isinstance(key_or_text, dict):
            self._state['memory'].update(key_or_text)
        else:
            self._state['memory']['note'] = str(key_or_text)

    def escalate(self, reason, rule_name=None):
        actions = self._engine.actions
        now_m = time.monotonic()
        has_wait = any(w.get('source') in ('reflex_observe', 'main') for w in actions.passive_waits.values())
        if actions.main_busy and not has_wait:
            return
        if not has_wait and (now_m - self._state.get('_last_escalate_mono', -1e9)) < 1.5:
            return
        self._state['_last_escalate_mono'] = now_m
        msg = str(reason)
        self._state['memory']['escalation'] = msg
        if rule_name and isinstance(self._state['memory'].get(rule_name), dict):
            rmem = self._state['memory'][rule_name]
            sys_keys = {'detected', 'anchor', 'nearest', 'last_escalation'}
            user_snap = {k: copy.deepcopy(v) for k, v in rmem.items() if k not in sys_keys}
            rmem['last_escalation'] = {'reason': msg, **user_snap}
            for k in ('escalate', 'escalated'):
                rmem.pop(k, None)
        self._stats['escalate_count'] += 1
        self._engine._notify('reflex_escalation', rule=self._name, version=self._card['version'],
                             episode_id=self._episode_id, status='escalated', reason=msg,
                             memory=copy.deepcopy(self._state['memory']))
        async def _wake_main():
            async with actions.lock:
                timers = await actions._stop_passive_waits('reflex_escalated', {'reflex_observe', 'main'})
            await asyncio.gather(*timers, return_exceptions=True)
        asyncio.create_task(_wake_main())

    def wait(self, seconds):
        return self._act({'action': 'wait', 'duration': seconds})

    def _act(self, native_action):
        self._check()
        if isinstance(native_action, str):
            native_action = {'action_id': native_action} if native_action in self._engine.semantic_actions else {'action': 'press_key', 'key': native_action}
        if not isinstance(native_action, dict):
            raise ValueError('act needs a native action dictionary')
        action = self._engine._resolve(native_action)
        owner = self._state['perception_api'] if action['action'] == 'wait' else self
        obs_pair = self._state.get('observation')
        obs_frame = obs_pair[0] if obs_pair is not None else self._engine.frames.latest
        task = asyncio.create_task(self._engine._execute(owner, obs_frame, action),
                                   name='reflex-input-'+self._name)
        self._engine._track(task, self._state['inputs'], self._engine.input_tasks)
        return _Operation(task)


    async def run_rules(self, rules):
        trk = {'frame': None, 'gray': None, 'bg': None, 'dprev': None, 'dbg': None,
               'idx': 0, 'ts': 0.0, 'dt': 0.05, 'tracks': {}, 'ntid': 1}
        last_fire, cell_conf, pending, last_anc = {}, {}, {}, {}
        last_act, last_matched, last_confirmed, last_act_frame = {}, {}, {}, {}
        none_cache = deque(maxlen=24)
        while True:
            frame = await self.observe()
            is_lockstep = self._engine.actions.clock_env is not None
            if is_lockstep and (self._engine.actions._inference_count > 0 or not self._engine.actions.clock_actions):
                continue
            scene_ts = self.context.get('scene_time_s')
            now = float(scene_ts or self.context.get('time_s') or (trk['ts'] + 0.05))
            if is_lockstep and scene_ts is not None and trk['ts'] > 0.0 and now <= trk['ts'] + 1e-4:
                continue
            if trk['ts'] > 0.0 and now > trk['ts']:
                trk['dt'] = max(1e-3, min(0.5, now - trk['ts']))
            trk['ts'], trk['idx'], trk['frame'] = now, trk['idx'] + 1, frame
            trk.setdefault('rgb_hist', deque(maxlen=3)).append(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            trk['dprev'] = cv2.absdiff(gray, trk['gray']) if (trk['gray'] is not None and trk['gray'].shape == gray.shape) else np.zeros_like(gray)
            trk['gray'] = gray
            if trk['bg'] is None or trk['bg'].shape != gray.shape:
                trk['bg'], trk['dbg'] = gray.astype(np.float32), np.zeros_like(gray)
            else:
                cv2.accumulateWeighted(gray, trk['bg'], 0.08)
                trk['dbg'] = cv2.absdiff(gray, cv2.convertScaleAbs(trk['bg']))
            fh, fw = gray.shape[:2]
            is_lockstep = self._engine.actions.clock_env is not None
            for idx_r, rule in enumerate(rules or ()):
                rname = str(rule.get('name') or f'rule_{idx_r}')
                rule = {**rule, 'name': rname}
                cands = await asyncio.to_thread(_ut_detect, rule, trk, self.images)
                self._check()
                player_spec = rule.get('player')
                anc_img = rule.get('anchor_image') or (player_spec if isinstance(player_spec, str) else None)
                anc_xy_spec = rule.get('anchor_xy') or (tuple(player_spec) if isinstance(player_spec, (tuple, list)) and len(player_spec) == 2 else None)
                if not anc_img and not anc_xy_spec and not rule.get('anchor_rule'):
                    other_imgs = [k for k in self.images if k != rule.get('image')]
                    if len(other_imgs) == 1:
                        anc_img = other_imgs[0]
                has_explicit_anc = bool(anc_xy_spec or anc_img or rule.get('anchor_rule'))
                default_anc = (fw * 0.5, fh * 0.5)
                if anc_img and anc_img in self._card.get('images', {}):
                    reg = self._card['images'][anc_img].get('region') or {}
                    if 'x' in reg and 'y' in reg:
                        default_anc = (float(reg['x']) + 0.5 * float(reg.get('width', 0)),
                                       float(reg['y']) + 0.5 * float(reg.get('height', 0)))
                anc_xy = anc_xy_spec or last_anc.get(rname) or default_anc
                if anc_img and anc_img in self.images:
                    ax0, ay0 = int(anc_xy[0]), int(anc_xy[1])
                    aroi = (max(0, ax0 - 240), max(0, ay0 - 200), min(fw, ax0 + 240) - max(0, ax0 - 240), min(fh, ay0 + 200) - max(0, ay0 - 200))
                    ac = await asyncio.to_thread(_ut_detect, {'name': rname + '_anc', 'mode': 'template', 'image': anc_img,
                                     'roi': aroi, 'flip_h': False, 'is_anchor': True, 'threshold': 0.58, 'dynamics': 'any', 'max_k': 1}, trk, self.images)
                    if not ac:
                        ac = await asyncio.to_thread(_ut_detect, {'name': rname + '_anc', 'mode': 'template', 'image': anc_img,
                                         'flip_h': True, 'is_anchor': True, 'threshold': 0.52, 'dynamics': 'any', 'max_k': 1}, trk, self.images)
                    if ac:
                        anc_xy = (ac[0]['cx'], ac[0]['cy'])
                        last_anc[rname] = anc_xy
                elif rule.get('anchor_rule'):
                    ac = _ut_detect({**rule['anchor_rule'], 'name': rname + '_anc'}, trk, self.images)
                    if ac:
                        anc_xy = (ac[0]['cx'], ac[0]['cy'])
                        last_anc[rname] = anc_xy
                anc = {'cx': float(anc_xy[0]), 'cy': float(anc_xy[1])}
                rule_v = trk.setdefault('rule_vel', {})
                nz_vx = [float(c.get('vx', 0.0)) for c in cands if abs(float(c.get('vx', 0.0) or 0.0)) > 20.0]
                nz_vy = [float(c.get('vy', 0.0)) for c in cands if abs(float(c.get('vy', 0.0) or 0.0)) > 20.0]
                if nz_vx or nz_vy:
                    prev_vx, prev_vy = rule_v.get(rname, (0.0, 0.0))
                    med_vx = sorted(nz_vx)[len(nz_vx)//2] if nz_vx else prev_vx
                    med_vy = sorted(nz_vy)[len(nz_vy)//2] if nz_vy else prev_vy
                    rule_v[rname] = (0.5 * prev_vx + 0.5 * med_vx if prev_vx else med_vx,
                                     0.5 * prev_vy + 0.5 * med_vy if prev_vy else med_vy)
                for c in cands:
                    c['dx'], c['dy'] = c['cx'] - anc['cx'], c['cy'] - anc['cy']
                    vx_c = float(c.get('vx', 0.0) or 0.0)
                    vy_c = float(c.get('vy', 0.0) or 0.0)
                    if abs(vx_c) < 1.0 and abs(vy_c) < 1.0 and rname in rule_v and (rule.get('dynamics') == 'moving' or float(c.get('motion', 0.0) or 0.0) >= 4.0):
                        vx_c, vy_c = rule_v[rname]
                        c['vx'], c['vy'] = round(vx_c, 1), round(vy_c, 1)
                        c['pred_x'], c['pred_y'] = round(c['cx'] + vx_c * 0.20, 1), round(c['cy'] + vy_c * 0.20, 1)
                    px_c = float(c.get('pred_x', c['cx'] + vx_c * 0.20))
                    py_c = float(c.get('pred_y', c['cy'] + vy_c * 0.20))
                    c['pred_dx'], c['pred_dy'] = px_c - anc['cx'], py_c - anc['cy']
                    ttc_x = abs(c['dx'] / vx_c) if (abs(vx_c) > 15.0 and c['dx'] * vx_c < 0.0) else None
                    ttc_y = abs(c['dy'] / vy_c) if (abs(vy_c) > 15.0 and c['dy'] * vy_c < 0.0) else None
                    dist_c = math.hypot(c['dx'], c['dy'])
                    v_close = -(c['dx'] * vx_c + c['dy'] * vy_c) / max(1.0, dist_c)
                    ttc_r = (dist_c / v_close) if v_close > 15.0 else None
                    if ttc_x is not None and (ttc_y is None or abs(vx_c) >= 1.2 * abs(vy_c)):
                        c['ttc'] = round(ttc_x, 2)
                    elif ttc_y is not None and (ttc_x is None or abs(vy_c) >= 1.2 * abs(vx_c)):
                        c['ttc'] = round(ttc_y, 2)
                    elif ttc_r is not None:
                        c['ttc'] = round(ttc_r, 2)
                    else:
                        c['ttc'] = None
                rmem = self._state['memory'].setdefault(rname, {})
                if not isinstance(rmem, dict):
                    rmem = {'value': rmem}
                    self._state['memory'][rname] = rmem
                rmem['detected'] = len(cands)
                rmem['anchor'] = {'cx': round(anc['cx'], 1), 'cy': round(anc['cy'], 1)}
                if cands:
                    c0 = min(cands, key=lambda c: math.hypot(c['dx'], c['dy']))
                    rmem['nearest'] = {'cx': round(c0['cx'], 1), 'cy': round(c0['cy'], 1), 'dx': round(c0['dx'], 1), 'dy': round(c0['dy'], 1), 'vx': round(float(c0.get('vx', 0.0)), 1), 'vy': round(float(c0.get('vy', 0.0)), 1), 'ttc': c0.get('ttc')}
                def _call_rule_fn(fn, *args):
                    try:
                        return fn(*args)
                    except TypeError:
                        return fn(*args[:2])
                raw_cands = list(cands)
                step_ev = {'phase': 'step', 'time_s': now, 'frame_idx': trk['idx'], 'before_frame_id': last_act_frame.get(rname, trk['idx']), 'cands': raw_cands, 'anc': anc, 'passed': list(cands), 'gated': [], 'confirmed': last_confirmed.get(rname), 'matched': last_matched.get(rname), 'action': last_act.get(rname)}
                if callable(rule.get('on_step')):
                    try:
                        res_s = _call_rule_fn(rule['on_step'], rmem, step_ev, anc)
                        if inspect.isawaitable(res_s):
                            await res_s
                    except Exception:
                        pass
                icond = rule.get('intercept_when') or rule.get('when')
                if icond is not None:
                    passed = []
                    for c in cands:
                        try:
                            ok_icond = bool(_call_rule_fn(icond, c, anc, rmem))
                        except Exception:
                            ok_icond = False
                        if not ok_icond:
                            continue
                        passed.append(c)
                    cands = passed
                step_ev['passed'] = list(cands)
                if callable(rule.get('escalate_when')):
                    esc = None
                    try:
                        esc = _call_rule_fn(rule['escalate_when'], rmem, step_ev, anc)
                        if inspect.isawaitable(esc):
                            esc = await esc
                    except Exception:
                        esc = None
                    if esc:
                        if raw_cands and self._stats.get('confirm_attempts', 0) == 0:
                            raise ValueError('Do not use escalate_when to wake up and act manually when a candidate is aligned; put the alignment check in intercept_when and return a natural-language action prompt (e.g. "Press Space if aligned") so the fast visual model executes the action directly.')
                        self.escalate(esc if isinstance(esc, str) else f'{rname}_escalated', rule_name=rname)
                if 'action' in rule and rule['action'] is not None:
                    raise ValueError('Rule cannot specify a direct "action"; provide "prompt" (str or function returning str) to send natural-language guidance to the fast visual model, which makes the final action decision.')
                has_Explicit_prompt_key = any(k in rule for k in ('prompt', 'guide', 'decide', 'condition'))
                prompt_spec = rule.get('prompt') if 'prompt' in rule else (
                    rule.get('guide') if 'guide' in rule else (
                        rule.get('decide') if 'decide' in rule else (rule.get('condition') if 'condition' in rule else None)
                    )
                )
                is_tracker_rule = not has_Explicit_prompt_key or prompt_spec is None
                if is_tracker_rule and (callable(rule.get('on_step')) or callable(rule.get('on_confirm')) or callable(rule.get('escalate_when'))) and len(rules or ()) > 1:
                    if cands and callable(rule.get('on_confirm')):
                        top_hit = {k: v for k, v in cands[0].items() if k != 'sig'} if isinstance(cands[0], dict) else cands[0]
                        last_confirmed[rname] = top_hit
                        last_matched[rname] = True
                        conf_ev = {**step_ev, 'phase': 'confirm', 'gated': [top_hit], 'confirmed': top_hit, 'matched': True, 'action': None}
                        try:
                            res_c = _call_rule_fn(rule['on_confirm'], rmem, conf_ev, anc)
                            if inspect.isawaitable(res_c):
                                await res_c
                        except Exception:
                            pass
                    continue
                if prompt_spec is None or (isinstance(prompt_spec, str) and not prompt_spec.strip()):
                    raise ValueError('Each action rule in run_rules must provide a non-empty "prompt" (str or function returning str) to guide the fast visual model.')
                base_cond = rule.get('condition') if isinstance(rule.get('condition'), str) and prompt_spec is not rule.get('condition') else None
                valid_cands = []
                for c in cands:
                    if callable(prompt_spec):
                        p_val = _call_rule_fn(prompt_spec, c, anc, rmem)
                        if inspect.isawaitable(p_val):
                            p_val = await p_val
                    else:
                        p_val = prompt_spec
                    if isinstance(p_val, dict):
                        raise ValueError('Python cannot return an action dict; return a natural-language string prompt for the fast visual model, which makes the final action decision.')
                    if p_val is None or p_val is False or (isinstance(p_val, str) and not p_val.strip()):
                        continue
                    p_str = str(p_val).strip()
                    if not callable(prompt_spec):
                        w_m = re.search(r'within\s+(\d+(?:\.\d+)?)\s*(?:px|pixels)', p_str, re.I)
                        if w_m:
                            w_lim = float(w_m.group(1))
                            dist_c = abs(c['dx']) if 'horiz' in p_str.lower() else (abs(c['dy']) if 'vert' in p_str.lower() else math.hypot(c['dx'], c['dy']))
                            if dist_c > w_lim:
                                continue
                    # Skip candidates whose prompt explicitly asks for no action / observation only
                    if re.search(r'(?:\b(?:answer|reply|return|output)\s+`?none`?\s*[.!]*$|\breport\s+nothing\b|\bjust\s+observe\b|\bdo\s+nothing\b)', p_str, re.I):
                        if not re.search(r'\b(?:only\s+if|unless|otherwise|if\s+no|if\s+not|if\s+the|press\s+\w+|click|drag)\b', p_str, re.I):
                            continue
                    if base_cond and base_cond.strip():
                        p_str = f"{base_cond.strip()} | {p_str}"
                    valid_cands.append({**c, '_jev_prompt': p_str})
                if not valid_cands and cands and callable(rule.get('on_confirm')):
                    top_hit = {k: v for k, v in cands[0].items() if k != 'sig'} if isinstance(cands[0], dict) else cands[0]
                    last_confirmed[rname] = top_hit
                    last_matched[rname] = True
                    conf_ev = {**step_ev, 'phase': 'confirm', 'gated': [top_hit], 'confirmed': top_hit, 'matched': True, 'action': None}
                    try:
                        res_c = _call_rule_fn(rule['on_confirm'], rmem, conf_ev, anc)
                        if inspect.isawaitable(res_c):
                            await res_c
                    except Exception:
                        pass
                cands = valid_cands
                if len(cands) > 1:
                    if has_explicit_anc:
                        cands.sort(key=lambda c: math.hypot(c['dx'], c['dy']))
                    else:
                        cands.sort(key=lambda c: (c['cx'], c['cy']))
                cands = cands[:int(rule.get('max_k', 5))]
                if not cands or (now - last_fire.get(rname, -1e9)) < float(rule.get('cooldown_s', 0.20)):
                    continue
                reconf = 0.0 if is_lockstep else min(float(rule.get('reconfirm_s', 0.12)), float(rule.get('cooldown_s', 0.20)) * 0.75)
                gated = []
                for c in cands:
                    cx_c, cy_c = int(c['cx']) // 36, int(c['cy']) // 36
                    if reconf > 0.0 and (now - cell_conf.get((rname, cx_c, cy_c), -1e9)) < reconf:
                        continue
                    sig = _ut_sig8(gray, c['box'])
                    if sig is not None and any(
                        rn == rname and (now - nts) <= 3.5 and abs(cx_c - nx) <= 1 and abs(cy_c - ny) <= 1
                        and float(np.mean(np.abs(sig - nsig))) < 12.0
                        for rn, nx, ny, nsig, nts in none_cache
                    ):
                        continue
                    gated.append({**c, 'sig': sig, 'cell': (cx_c, cy_c)})
                if not gated:
                    continue
                gated = gated[:int(rule.get('confirm_k', 1))]
                vis_mode = os.environ.get('VICOR_VIS_MODE', 'trail_ttc')
                prompt_lines = [
                    f"Candidate {idx_g} [center=({g['cx']:.0f},{g['cy']:.0f}), "
                    f"dx={g['dx']:+.0f}, dy={g['dy']:+.0f}, "
                    f"vx={float(g.get('vx', 0.0)):+.0f}, vy={float(g.get('vy', 0.0)):+.0f}"
                    + (f", pred_dx={g['pred_dx']:+.0f}, ttc={g['ttc']:.2f}s" if (vis_mode == 'trail_ttc' and g.get('ttc') is not None) else "")
                    + f"]: {g.get('_jev_prompt') or rname}"
                    for idx_g, g in enumerate(gated, 1)
                ]
                jev_guidance = "\n".join(prompt_lines)
                self._state['_last_gated_meta'] = {
                    'by_box': {tuple(map(int, g['box'])): g for g in gated},
                    'anc': anc if has_explicit_anc else None,
                    'space': rule.get('space', '2d'),
                    'vis_mode': vis_mode,
                    'rgb_hist': list(trk.get('rgb_hist', ()))[:-1] if vis_mode == 'ghosting' else None,
                }
                rx = await self.confirm(jev_guidance, [c['box'] for c in gated])
                for c in gated:
                    cell_conf[(rname, c['cell'][0], c['cell'][1])] = now
                if rx is None:
                    for c in gated:
                        if c['sig'] is not None:
                            none_cache.append((rname, c['cell'][0], c['cell'][1], c['sig'], now))
                    last_act[rname] = None
                    last_matched[rname] = False
                    last_confirmed[rname] = None
                    last_act_frame[rname] = trk['idx']
                    conf_ev = {**step_ev, 'phase': 'confirm', 'gated': [{k: v for k, v in g.items() if k != 'sig'} for g in gated], 'confirmed': None, 'matched': False, 'action': None}
                    if callable(rule.get('on_confirm')):
                        try:
                            res_c = _call_rule_fn(rule['on_confirm'], rmem, conf_ev, anc)
                            if inspect.isawaitable(res_c):
                                await res_c
                        except Exception:
                            pass
                    if callable(rule.get('escalate_when')):
                        try:
                            esc = _call_rule_fn(rule['escalate_when'], rmem, conf_ev, anc)
                            if inspect.isawaitable(esc):
                                esc = await esc
                            if esc:
                                self.escalate(esc if isinstance(esc, str) else f'{rname}_escalated', rule_name=rname)
                        except Exception:
                            pass
                    continue
                bx, by, bw, bh = rx.box
                hit = next((c for c in gated if abs(c['box'][0] - bx) <= 4 and abs(c['box'][1] - by) <= 4), gated[0])
                act = _ActionDict(rx.action) if isinstance(rx.action, dict) else rx.action
                conf_hit = {k: v for k, v in hit.items() if k != 'sig'} if isinstance(hit, dict) else hit
                last_act[rname] = act
                last_matched[rname] = True
                last_confirmed[rname] = conf_hit
                last_act_frame[rname] = trk['idx']
                conf_ev = {**step_ev, 'phase': 'confirm', 'gated': [{k: v for k, v in g.items() if k != 'sig'} for g in gated], 'confirmed': conf_hit, 'matched': True, 'action': act}
                if callable(rule.get('on_confirm')):
                    try:
                        res_c = _call_rule_fn(rule['on_confirm'], rmem, conf_ev, anc)
                        if inspect.isawaitable(res_c):
                            await res_c
                    except Exception:
                        pass
                if callable(rule.get('escalate_when')):
                    try:
                        esc = _call_rule_fn(rule['escalate_when'], rmem, conf_ev, anc)
                        if inspect.isawaitable(esc):
                            esc = await esc
                        if esc:
                            self.escalate(esc if isinstance(esc, str) else f'{rname}_escalated', rule_name=rname)
                    except Exception:
                        pass
                if act is not None and not (isinstance(act, dict) and act.get('action') == 'wait'):
                    rep_map = trk.setdefault('visual_rep', {})
                    cur_cell = hit.get('cell') or (int(hit['cx']) // 36, int(hit['cy']) // 36)
                    cur_sig = hit.get('sig')
                    prev_rep = rep_map.get(rname)
                    if (prev_rep is not None and prev_rep['act'] == act
                            and abs(cur_cell[0] - prev_rep['cell'][0]) <= 1 and abs(cur_cell[1] - prev_rep['cell'][1]) <= 1
                            and cur_sig is not None and prev_rep['sig'] is not None
                            and float(np.mean(np.abs(cur_sig - prev_rep['sig']))) < 8.0
                            and float(np.mean(trk['dprev'])) < 2.5):
                        rep_cnt = prev_rep['count'] + 1
                    else:
                        rep_cnt = 1
                    rep_map[rname] = {'act': act, 'cell': cur_cell, 'sig': cur_sig, 'count': rep_cnt}
                    if rep_cnt >= 3 and cur_sig is not None:
                        none_cache.append((rname, cur_cell[0], cur_cell[1], cur_sig, now))
                        continue
                    last_fire[rname] = max(now, float(self.context.get('scene_time_s') or self.context.get('time_s') or now))
                    await rx.act()
                    trk['bg'] = None
                    break


class _ActionDict(dict):
    def _str_key(self):
        return str(self.get('key') or self.get('button') or self.get('action') or '')

    def __hash__(self):
        return hash(self._str_key())

    def __eq__(self, other):
        if isinstance(other, str):
            s = self._str_key()
            return other == s or other.lower() == s.lower() or other == self.get('action')
        return super().__eq__(other)


class _ReactionAPI(_ProgramAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._consumed = False

    @property
    def box(self):
        return self._box

    @property
    def action(self):
        return copy.deepcopy(getattr(self, '_action', None))

    def act(self, native_action=None):
        if native_action is not None:
            raise ValueError('Python cannot specify the action directly; call `await rx.act()` with no arguments to execute the final action decided by the fast visual model.')
        if self._consumed:
            async def _noop():
                await asyncio.sleep(0)
                return {'status': 'skipped', 'reason': 'reaction_already_consumed_call_confirm_again'}
            return _Operation(asyncio.create_task(_noop()))
        self._consumed = True
        self._state['api'] = self._state['perception_api']
        return self._act(self._action)


class VisualReflexEngine:
    def __init__(self, frames, actions, events, semantic_actions, semantic_specs,
                 directory, endpoint='http://127.0.0.1:8189/v1/chat/completions',
                 model='default', mode='full', bank_enabled=True):
        self.frames, self.actions, self.events = frames, actions, events
        self.semantic_actions, self.semantic_specs = semantic_actions, semantic_specs
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.endpoint, self.model, self.mode = endpoint, model, mode
        self.bank_enabled = bank_enabled and mode != 'no-bank'
        self.cards, self.stats, self.state, self.shared_state = {}, {}, {}, {}
        self.notifications = []
        self.program_tasks, self.input_tasks, self.effect_tasks = set(), set(), set()
        self.source_tasks = {}
        self.confirm_lock = asyncio.Lock()
        self.closed = False
        self.revision, self.request_sequence = 0, 0
        self.episode_id = None
        self.recent_previews = deque(maxlen=32)
        self.best = None
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=None, write=30, pool=10))

    @staticmethod
    def _track(task, *groups):
        for group in groups:
            group.add(task)
            task.add_done_callback(group.discard)
        return task

    def _stop_state(self, state):
        if state.get('stopped'):
            return
        state['stopped'] = True
        if state.get('api') is not None:
            state['api']._stopped = True
        task = state.get('task')
        if task is not None and not task.done():
            task.cancel()

    def _initial_state(self, name, reason='reflex_changed', compiled=None, decoded_images=None):
        previous = self.state.get(name, {})
        self._stop_state(previous)
        self.stats.setdefault(name, {key: 0 for key in COUNTERS})
        self.state[name] = {
            'memory': {k: copy.deepcopy(v) for k, v in (previous.get('memory') or {}).items() if k != 'escalation'}, 'inputs': set(),
            'compiled': compiled if compiled is not None else compile_trigger(self.cards[name]['trigger_py']),
            'images': decoded_images if decoded_images is not None else _decode_program_images(self.cards[name].get('images', {})),
        }
        if reason == 'episode_changed' and previous.get('error'):
            self.state[name]['error'] = previous['error']

    def _feedback(self, name):
        return copy.deepcopy({'status': 'idle', 'action': None, 'action_id': None,
                              'executed': False, 'action_completed': False, 'reason': None,
                              **(self.state[name].get('last_decision') or {})})

    def _last_input(self, frame):
        started = next((row for row in reversed(list(self.actions.history))
                        if row['event'] == 'action_started' and not row.get('passive')
                        and row.get('episode_id') == frame.episode_id
                        and row['monotonic'] < frame.timestamp), None)
        if started is None:
            return None
        completed = self.actions.completed.get(started['action_id'])
        if completed is not None and completed['monotonic'] > frame.timestamp:
            completed = None
        return copy.deepcopy({**started, 'completion': completed})

    def _save(self, snapshot=True):
        text = json.dumps({'revision': self.revision, 'cards': self.cards}, ensure_ascii=False, indent=2)
        target = self.directory / 'reflexes.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(target)
        if snapshot:
            folder = self.directory / 'versions'
            folder.mkdir(exist_ok=True)
            (folder / f'{self.revision:06d}.json').write_text(text, encoding='utf-8')

    def _resolve(self, value):
        if isinstance(value, str):
            if value not in self.semantic_actions:
                raise ValueError(f'Unknown semantic action {value!r}')
            return resolve_semantic_action(self.semantic_actions[value], self.semantic_specs[value], {})
        if not isinstance(value, dict):
            raise ValueError('Action must be a native action dictionary')
        if 'action' not in value and 'action_id' not in value:
            if 'key' in value:
                value = {'action': 'press_key', **value}
            elif 'keys' in value:
                value = {'action': 'press_keys', **value}
        elif value.get('action') in ('press', 'key', 'keydown'):
            value = {**value, 'action': 'press_key'}
        if 'action_id' in value:
            action_id = value['action_id']
            if action_id not in self.semantic_actions:
                raise ValueError(f'Unknown semantic action {action_id!r}')
            return resolve_semantic_action(self.semantic_actions[action_id], self.semantic_specs[action_id], value.get('parameters', {}))
        inspected = self.actions.validator.inspect_action(value)
        if not inspected['is_valid']:
            raise ValueError(f"Invalid reflex action {value!r}: {inspected['reason']}")
        return copy.deepcopy(inspected['normalized_action'])

    def set_reflex(self, name, trigger_py=None, memory_40w=None, direct_action=None, edits=None, images=None):
        if self.mode == 'pure-vlm':
            raise ValueError('This comparison arm has no reflex tools')
        if not isinstance(name, str) or not name or not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Use a short alphanumeric rule name')
        previous = self.cards.get(name, {})
        trigger_py = previous.get('trigger_py') if trigger_py is None else trigger_py
        memory_40w = previous.get('memory_40w', '') if memory_40w is None else memory_40w
        merged_imgs = copy.deepcopy(getattr(self, '_known_images', {}))
        merged_imgs.update(copy.deepcopy(previous.get('images', {})))
        if images:
            merged_imgs.update(copy.deepcopy(images))
        images = merged_imgs
        self._known_images = copy.deepcopy(images)
        for edit in edits or ():
            old, new = edit['old'], edit['new']
            if not isinstance(trigger_py, str) or not old or trigger_py.count(old) != 1:
                raise ValueError('Each edit.old must match exactly once in trigger_py')
            trigger_py = trigger_py.replace(old, new, 1)
        if not isinstance(memory_40w, str):
            raise ValueError('memory_40w must be a string')
        compiled, decoded_images = compile_trigger(trigger_py), _decode_program_images(images)
        frame = self.frames.latest
        version = previous.get('version', 0) + 1
        self.revision += 1
        self.cards[name] = {'name': name, 'trigger_py': trigger_py, 'memory_40w': memory_40w,
                            'images': images, 'version': version,
                            'registered_episode': frame.episode_id if frame else None,
                            'registered_frame': frame.frame_id if frame else None}
        # Preserve ablation metadata when operating in programmatic script mode.
        if self.mode == 'pure-python' and direct_action is not None:
            self.cards[name]['direct_action'] = direct_action
        self._initial_state(name, compiled=compiled, decoded_images=decoded_images)
        self._save()
        self.events.add('reflex_registered', rule=name, version=version, revision=self.revision,
                        frame_id=frame.frame_id if frame else None,
                        episode_id=frame.episode_id if frame else None,
                        card=self.cards[name], origin='main_agent')
        return {'status': 'registered', 'rule': name, 'version': version,
                'trigger': 'async def run(api)', 'images': _image_references(images)}

    def drop_reflex(self, name):
        if name not in self.cards:
            raise ValueError(f'Unknown reflex {name}')
        self._stop_state(self.state[name])
        del self.cards[name]
        del self.state[name]
        self.revision += 1
        self._save()
        self.events.add('reflex_dropped', rule=name, revision=self.revision)
        return {'status': 'dropped', 'rule': name}

    def list(self):
        return [{'name': name, **card, 'statistics': self.stats[name],
                 'images': _image_references(card.get('images', {})),
                'state': copy.deepcopy(self.state[name]['memory']),
                'finished': self.state[name].get('finished', False),
                'error': self.state[name].get('error'), 'reflex': self._feedback(name)}
                for name, card in self.cards.items()]

    def drain_notifications(self):
        messages, self.notifications = self.notifications, []
        return messages

    def _notify(self, kind, **values):
        row = self.events.add(kind, **values)
        self.notifications.append(row)
        return row

    async def wait_pending(self):
        # The coordinator calls this during intermission, before reset; never await an endless run.
        if self.closed or self.frames.phase != 'playing' or not self.actions.can_run_reflexes:
            for state in list(self.state.values()):
                self._stop_state(state)
            if self.program_tasks:
                await asyncio.gather(*list(self.program_tasks), return_exceptions=True)

    async def wait_effects(self):
        while self.input_tasks or self.effect_tasks:
            await asyncio.gather(*list(self.input_tasks | self.effect_tasks), return_exceptions=True)

    async def preview(self, name, frame):
        observation = self.state[name].get('observation')
        if observation is None:
            return None
        source, details = observation
        pixels = np.asarray(Image.open(io.BytesIO(source.image)).convert('RGB'))
        picture, transform = _render_reflex_image(pixels, details['match_boxes'])
        return picture, {**details, 'preview_transform': transform}, source

    def _parse_jev_decision(self, raw, boxes, condition_text=''):
        if not isinstance(raw, str) or not raw.strip() or not boxes:
            return 0, None
        controls = getattr(getattr(self.actions, 'validator', None), 'controls', None)
        allowed_keys = sorted(getattr(controls, 'allowed_keys', None) or ())
        allow_clicks = bool(getattr(controls, 'allow_clicks', False))
        key_map = {k.lower(): k for k in allowed_keys}
        aliases = {
            'spacebar': 'Space', 'space': 'Space', 'spc': 'Space',
            'up': 'ArrowUp', 'arrowup': 'ArrowUp',
            'down': 'ArrowDown', 'arrowdown': 'ArrowDown',
            'left': 'ArrowLeft', 'arrowleft': 'ArrowLeft',
            'right': 'ArrowRight', 'arrowright': 'ArrowRight',
            'enter': 'Enter', 'return': 'Enter',
        }
        for ak, target in aliases.items():
            if target.lower() in key_map:
                key_map[ak] = key_map[target.lower()]

        def _extract_action_from_str(s, box):
            if not s:
                return None
            s_clean = s.strip().strip('`')
            j_match = re.search(r'\{[^{}]+\}', s_clean)
            if j_match:
                try:
                    obj = json.loads(j_match.group(0))
                    return self._resolve(obj)
                except Exception:
                    pass
            low = s_clean.lower()
            if allow_clicks:
                if re.search(r'\bdrag\b', low):
                    nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', s_clean)]
                    if len(nums) >= 4:
                        try:
                            return self._resolve({'action': 'drag', 'x1': int(round(nums[0])), 'y1': int(round(nums[1])),
                                                  'x2': int(round(nums[2])), 'y2': int(round(nums[3]))})
                        except Exception:
                            pass
                if re.search(r'\bclick\b', low):
                    nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', s_clean)]
                    cx, cy = (int(round(nums[0])), int(round(nums[1]))) if len(nums) >= 2 else (
                        int(round(box[0] + 0.5 * box[2])), int(round(box[1] + 0.5 * box[3]))
                    )
                    try:
                        return self._resolve({'action': 'click', 'x': cx, 'y': cy})
                    except Exception:
                        pass
            dur = None
            dur_m = re.search(r'(?:duration\s*[:=]?\s*|hold\s+)?(\d+\.\d+|\.\d+)\s*s?\b', low)
            if dur_m:
                val = float(dur_m.group(1))
                if 0.0 < val <= 2.0:
                    dur = val
            tokens = re.findall(r'[A-Za-z0-9_]+', s_clean)
            exact_lower = {k.lower(): k for k in allowed_keys}
            matched_keys = []
            for tok in tokens:
                tl = tok.lower()
                if tl in exact_lower and exact_lower[tl] not in matched_keys:
                    matched_keys.append(exact_lower[tl])
            if not matched_keys:
                for tok in tokens:
                    tl = tok.lower()
                    if tl in key_map and key_map[tl] not in matched_keys:
                        matched_keys.append(key_map[tl])
            if len(matched_keys) >= 2:
                cand_act = {'action': 'press_keys', 'keys': matched_keys[:2]}
                if dur is not None:
                    cand_act['duration'] = dur
                try:
                    return self._resolve(cand_act)
                except Exception:
                    pass
            if len(matched_keys) >= 1:
                cand_act = {'action': 'press_key', 'key': matched_keys[0]}
                if dur is not None:
                    cand_act['duration'] = dur
                try:
                    return self._resolve(cand_act)
                except Exception:
                    pass
            return None

        first_line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), '').strip('`').strip()
        if not first_line or re.match(r'^(?:none|no|false|skip|wait|0)\b', first_line, re.IGNORECASE):
            return 0, None
        m = re.match(r'^(\d+)\b\s*(.*)$', first_line)
        if m:
            index = int(m.group(1))
            rest = m.group(2).strip()
            if index <= 0 or index > len(boxes):
                return 0, None
        else:
            index = 1
            rest = first_line
        box = boxes[index - 1]
        cand_text = condition_text or ''
        if condition_text:
            for ln in condition_text.splitlines():
                if re.match(rf'^\s*Candidate\s+{index}\b', ln, re.IGNORECASE):
                    cand_text = ln
                    break
        cand_body = re.sub(r'^\s*Candidate\s+\d+\s*\[[^\]]*\]\s*:\s*', '', cand_text, flags=re.IGNORECASE)
        action = _extract_action_from_str(rest, box)
        if action is None and cand_body:
            action = _extract_action_from_str(cand_body, box)
        if action is not None and action.get('action') in ('press_key', 'press_keys') and 'duration' not in action and cand_body:
            key_name = str(action.get('key') or (action.get('keys') or [''])[0])
            search_spans = [cl for cl in re.split(r'(?:;|\||\n|\.(?=\s|$))+', cand_body) if key_name.lower() in cl.lower()] if key_name else [cand_body]
            for sp in search_spans:
                dm = re.search(
                    r'(?:for\s+|hold(?:\s+[A-Za-z0-9_]+)?(?:\s+for)?\s+|duration\s*[:=]?\s*)(\d+\.\d+|\.\d+|\d+)\s*(?:s|sec|seconds)?\b'
                    r'|\b(\d+\.\d+)\s*(?:s|sec|seconds)\b',
                    sp, re.IGNORECASE
                )
                if dm:
                    val = float(dm.group(1) or dm.group(2))
                    if 0.02 <= val <= 2.0:
                        try:
                            action = self._resolve({**action, 'duration': val})
                        except Exception:
                            action['duration'] = val
                        break
        if action is None or action.get('action') == 'wait':
            return 0, None
        return index, action

    async def _infer(self, card, picture, boxes, transform, scene_picture):
        controls = getattr(getattr(self.actions, 'validator', None), 'controls', None)
        allowed_keys = sorted(getattr(controls, 'allowed_keys', None) or ())
        allow_clicks = bool(getattr(controls, 'allow_clicks', False))
        allowed_parts = []
        if allowed_keys:
            allowed_parts.append('Allowed keys: ' + ', '.join(allowed_keys) + ' (optional hold seconds, e.g. `1 Space` or `1 ArrowRight 0.35`)')
        if allow_clicks:
            allowed_parts.append('Mouse allowed: `1 click <x> <y>` or `1 drag <x1> <y1> <x2> <y2>`')
        controls_desc = ' | '.join(allowed_parts) if allowed_parts else 'Output `<box_number> <action>`'
        prompt = (
            'You are playing a live video game. '
            'A helper script marked numbered box(es) (1..N) on the image and provided a suggested action below, '
            'but the script\'s suggestion is only a rough guess and may be wrong (wrong timing, wrong direction, obstacle still too far away, or a false alarm).\n'
            'Look at the actual game image (red box = candidate; yellow arrow & cyan box = +0.2s motion trajectory; green circle = player/anchor; ttc = estimated seconds to contact) and decide flexibly:\n'
            '1. If the boxed region is just background/UI, not a real obstacle/target, or no action should be taken right now, reply ONLY: `NONE`\n'
            '2. If immediate action is needed, output the best action for what you see in the image. You may follow the script\'s suggestion when it fits the image, or change the key, hold duration, or reply `NONE` if the suggestion does not fit the actual scene.\n'
            f'{controls_desc}\n'
            'Reply with ONLY `NONE` or one line `<box_number> <action>` (for example `1 Space`, `1 ArrowRight 0.35`, or `1 click 400 300`). Do not output explanations.'
        )
        content = [{'type': 'text', 'text': 'Script suggestion (may be inaccurate; judge flexibly from the image):\n' + card['memory_40w']},
                        {'type': 'image_url', 'image_url': {
                            'url': 'data:image/png;base64,' + base64.b64encode(picture).decode()}},
                        {'type': 'text', 'text': 'Your decision (`NONE` or `<box_number> <action>`):'}]
        body = {'model': self.model, 'temperature': 0, 'max_tokens': 12,
                'chat_template_kwargs': {'enable_thinking': False},
                'messages': [{'role': 'system', 'content': prompt},
                             {'role': 'user', 'content': content}]}
        async with self.actions.inference():
            for _attempt in range(120):
                response = await self.http.post(self.endpoint, json=body)
                if response.status_code not in (429, 502, 503, 504):
                    break
                await asyncio.sleep(0.10)
            response.raise_for_status()
            data = response.json()
        raw = data['choices'][0]['message'].get('content')
        index, action = self._parse_jev_decision(raw, boxes, card.get('memory_40w', ''))
        observation = {'candidate': index, 'box': tuple(boxes[index-1]), 'action': action} if (index and action is not None) else None
        return ('MATCH' if observation is not None else 'NONE'), raw, data.get('usage'), observation, data.get('timings')

    def _unavailable_reason(self, api, frame):
        if self.closed or api._stopped:
            return 'runtime_stopped'
        if self.frames.episode_id != frame.episode_id:
            return 'episode_changed'
        if self.frames.phase != 'playing' or not self.actions.can_run_reflexes:
            return 'episode_finished_before_response'
        if self.actions.lifecycle_epoch != api._lifecycle:
            return 'episode_lifecycle_changed'
        if self.state.get(api._name) is not api._state or self.cards.get(api._name) is not api._card:
            return 'reflex_changed_before_response'
        return None

    async def _confirm(self, api, frame, condition, boxes):
        name, card, state = api._name, api._card, api._state
        if self._unavailable_reason(api, frame) is not None or not boxes:
            return None
        now_ts = float(getattr(frame, 'scene_time_s', None) or frame.timestamp)
        cell_conf = state.setdefault('cell_conf', {})
        none_patches = state.setdefault('none_patches', deque(maxlen=24))
        current_rgb = np.asarray(Image.open(io.BytesIO(frame.image)).convert('RGB'))
        current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
        filtered_boxes = []
        box_sigs = []
        for b in boxes:
            cx_c, cy_c = int(b[0] + 0.5 * b[2]) // 36, int(b[1] + 0.5 * b[3]) // 36
            if self.actions.clock_env is None and (now_ts - cell_conf.get((name, cx_c, cy_c), -1e9)) < 0.12:
                continue
            sig = _ut_sig8(current_gray, b)
            if sig is not None and any(
                cond == name and (now_ts - nts) <= 3.5 and abs(cx_c - nx) <= 1 and abs(cy_c - ny) <= 1
                and float(np.mean(np.abs(sig - nsig))) < 12.0
                for cond, nx, ny, nsig, nts in none_patches
            ):
                continue
            filtered_boxes.append(b)
            box_sigs.append((cx_c, cy_c, sig))
        if not filtered_boxes:
            return None
        boxes = filtered_boxes
        for cx_c, cy_c, _ in box_sigs:
            cell_conf[(name, cx_c, cy_c)] = now_ts
        stats = api._stats
        self.request_sequence += 1
        decision = {'rule': name, 'decision_type': 'confirmation', 'request_id': self.request_sequence,
                    'episode_id': frame.episode_id, 'frame_id': frame.frame_id,
                    'request_frame_id': frame.frame_id, 'request_time_s': frame.timestamp,
                    'version': card['version'], 'hit_boxes': boxes, 'memory_40w': condition,
                    'status': 'checking', 'reason': None, 'action': None,
                    'executed': False, 'action_completed': False}
        prev_last_decision = state.get('last_decision')
        state['last_decision'] = decision
        stats['trigger_count'] += bool(boxes)
        try:
            async with self.confirm_lock:
                async with self.actions.inference():
                    reason = self._unavailable_reason(api, frame)
                    if reason is not None or not boxes:
                        decision.update(status='stale' if reason else 'no_action', reason=reason or 'empty_match',
                                        finished_time_s=time.monotonic())
                        if reason:
                            stats['stale_count'] += 1
                        self.events.add('reflex_decision', **decision)
                        return None
                    began = time.monotonic()
                    decision['request_frame_age_before_confirmation_s'] = began-frame.timestamp
                    if self.mode == 'pure-python':
                        _, pp_action = self._parse_jev_decision('1', boxes, condition)
                        word = 'MATCH' if pp_action is not None else 'NONE'
                        raw, usage, timings = 'pure-python', None, None
                        observation = {'candidate': 1, 'box': tuple(boxes[0]), 'action': pp_action} if pp_action is not None else None
                        latency = time.monotonic()-began
                    else:
                        stats['query_count'] += 1
                        decision['request_frame_path'] = str(self.frames.save(frame, reason='reflex_trigger'))
                        current = np.asarray(Image.open(io.BytesIO(frame.image)).convert('RGB'))
                        picture, transform = _render_reflex_image(current, boxes, meta=state.pop('_last_gated_meta', None))
                        scene_picture, scene_transform = _render_reflex_image(current, boxes, scene=True)
                        folder = self.events.directory / 'reflex_previews'
                        folder.mkdir(exist_ok=True)
                        stem = f'{frame.frame_id:07d}_{name}_{decision["request_id"]}'
                        preview_path, scene_path = folder / (stem+'.png'), folder / (stem+'_scene.png')
                        preview_path.write_bytes(picture)
                        scene_path.write_bytes(scene_picture)
                        decision.update(preview_path=str(preview_path), preview_transform=transform,
                                        scene_preview_path=str(scene_path), scene_preview_transform=scene_transform)
                        self.events.add('reflex_request', **decision, endpoint=self.endpoint,
                                        reasoning_effort='none', max_tokens=16)
                        word, raw, usage, observation, timings = await self._infer(
                            {**card, 'memory_40w': condition}, picture, boxes, transform, scene_picture)
                        latency = time.monotonic()-began
            decided_action = observation.get('action') if isinstance(observation, dict) else None
            self.events.add('reflex_response', rule=name, request_id=decision['request_id'],
                            frame_id=frame.frame_id, episode_id=frame.episode_id,
                            output=word, raw_output=raw, action=decided_action, inference_s=latency,
                            observation_age_s=time.monotonic()-frame.timestamp, usage=usage, timings=timings,
                            version=card['version'], hit_boxes=boxes, memory_40w=condition,
                            observation=observation)
            reason = self._unavailable_reason(api, frame)
            matched = word == 'MATCH' and reason is None
            game_latency = 0.0 if self.actions.clock_env is not None else latency
            decision.update(output=word, inference_s=game_latency, wall_inference_s=latency,
                            action=decided_action, raw_output=raw,
                            confirmation_time_s=time.monotonic(),
                            finished_time_s=time.monotonic(), reason=reason or (None if matched else 'NONE'),
                            status='stale' if reason else 'confirmed' if matched else 'no_action')
            if observation:
                decision['observation'] = observation
            if self.mode != 'pure-python':
                self.recent_previews.append(decision)
            if reason:
                stats['stale_count'] += 1
                self._notify('reflex_stale_result', **decision)
            else:
                if not matched:
                    stats['none_count'] += 1
                    for cx_c, cy_c, sig in box_sigs:
                        if sig is not None:
                            none_patches.append((name, cx_c, cy_c, sig, now_ts))
                if self.mode == 'pure-python':
                    self.events.add('reflex_decision', **decision)
                else:
                    self._notify('reflex_visual_result', **decision)
            return observation if matched else None
        except asyncio.CancelledError:
            decision.update(status='cancelled', reason='program_stopped', finished_time_s=time.monotonic())
            if (prev_last_decision is not None
                    and prev_last_decision.get('episode_id') == frame.episode_id
                    and prev_last_decision.get('status') not in ('checking', 'cancelled')):
                state['last_decision'] = prev_last_decision
            self._notify('reflex_decision', **decision)
            raise
        except Exception as exc:
            stats['error_count'] += 1
            decision.update(status='error', reason=f'{type(exc).__name__}: {exc}',
                            finished_time_s=time.monotonic())
            self._notify('reflex_error', rule=name, episode_id=frame.episode_id, error=decision['reason'])
            return None

    async def _execute(self, api, frame, action):
        name, state, card = api._name, api._state, api._card
        reason = self._unavailable_reason(api, frame)
        if reason:
            return self.events.add('action_rejected', status='skipped', source='rule:'+name, reason=reason,
                                   frame_id=frame.frame_id, episode_id=frame.episode_id)
        if action.get('action') == 'wait':
            result = await self.actions.submit(action, source='rule:'+name, frame=frame,
                                               priority=102, expected_lifecycle_epoch=api._lifecycle)
            if result.get('status') != 'executing' or not isinstance(result.get('action_id'), int):
                return {'status': 'skipped', **result}
            completion = await self.actions.wait_for_completion(result['action_id'])
            finished = completion['monotonic']
            if state.get('captured_after') is None or finished > state['captured_after']:
                state['captured_after'] = finished
            completed = completion.get('reason') == 'duration_completed'
            return {**result, **completion, 'status': 'completed' if completed else 'interrupted', 'action_completed': completed}
        before = self.frames.latest
        if before is None or before.episode_id != frame.episode_id:
            before = frame
        prev_dec = state.get('last_decision') or {}
        inherited = ({k: prev_dec[k] for k in (
            'preview_path', 'preview_transform', 'scene_preview_path', 'scene_preview_transform',
            'request_frame_path', 'observation', 'hit_boxes', 'inference_s', 'wall_inference_s')
            if k in prev_dec} if prev_dec.get('episode_id') == frame.episode_id else {})
        decision = {'rule': name, 'decision_type': 'action', 'episode_id': frame.episode_id,
                    'frame_id': frame.frame_id, 'request_frame_id': frame.frame_id,
                    'observed_frame_id': frame.frame_id,
                    'observed_frame_path': str(self.frames.save(frame, reason='reflex_observed')),
                    'before_frame_id': before.frame_id,
                    'request_time_s': frame.timestamp, 'version': card['version'],
                    'action': action, 'status': 'submitting', 'output': 'DIRECT', 'inference_s': 0.0,
                    'memory_40w': card['memory_40w'], 'executed': False, 'action_completed': False,
                    'reason': None, **inherited}
        state['last_decision'] = decision
        result = await self.actions.submit(action, source='rule:'+name, frame=frame,
                                           priority=102, expected_lifecycle_epoch=api._lifecycle)
        if result.get('status') != 'executing' or not isinstance(result.get('action_id'), int):
            decision.update(status='not_executed', reason=result.get('reason', result.get('status')),
                            finished_time_s=time.monotonic())
            self._notify('reflex_action_not_executed', **decision, result=result)
            return {'status': 'skipped', **result}
        decision.update(status='executing', executed=True, action_id=result['action_id'],
                        execution_time_s=result['execution_started_at'],
                        execution_frame_id=before.frame_id,
                        observed_frame_to_keydown_s=(0.02 if self.actions.clock_env is not None else result['execution_started_at']-frame.timestamp),
                        execution_frame_to_keydown_s=(0.01 if self.actions.clock_env is not None else result['execution_started_at']-before.timestamp))
        api._stats['action_count'] += 1
        self.events.add('reflex_action_executed', **decision, control_epoch=self.actions.epoch)
        completion = await self.actions.wait_for_completion(result['action_id'])
        finished = completion['monotonic']
        if state.get('captured_after') is None or finished > state['captured_after']:
            state['captured_after'] = finished
        completed = completion.get('reason') == 'duration_completed'
        decision.update(status='completed' if completed else 'interrupted', reason=completion.get('reason'),
                        action_completed=completed, finished_time_s=completion['monotonic'])
        self.events.add('reflex_action_completed', **decision)
        task = asyncio.create_task(self._record_aftermath(before, result, completion, decision),
                                   name=f'reflex-effect-{result["action_id"]}')
        self._track(task, self.effect_tasks)
        return {**result, **completion, 'status': decision['status'], 'action_completed': completed}

    async def _record_aftermath(self, frame, result, completion, decision):
        name, action_id = decision['rule'], result['action_id']
        try:
            window = 0.0 if decision['action']['action'] == 'wait' or self.actions.clock_env is not None else 0.25
            if not window:
                after = (await self.frames.capture(source='reflex_input_after')
                         if frame.episode_id == self.frames.episode_id and not self.actions.intermission_active else None)
            else:
                after = await self.actions.capture_reflex_aftermath(self.frames, episode_id=frame.episode_id, duration_s=window)
            if after is None or after.episode_id != frame.episode_id:
                self._notify('reflex_effect_unavailable', rule=name, episode_id=frame.episode_id,
                             action_id=action_id, action=decision['action'], executed=True,
                             reason='native_episode_changed_during_after_window')
                return
            before_path = self.frames.save(frame, reason='reflex_before')
            after_path = self.frames.save(after, reason='reflex_after_250ms' if window else 'reflex_after_input')
            overlap = [row['action_id'] for row in self.events.rows
                       if row['event'] == 'action_started' and row.get('action_id') != action_id
                       and row.get('episode_id') == frame.episode_id
                       and result['execution_started_at'] <= row.get('monotonic', 0) <= after.timestamp]
            receipt = {**decision, 'trigger_frame_id': decision['observed_frame_id'], 'before_frame_id': frame.frame_id,
                       'frame_id': after.frame_id, 'before_frame_path': str(before_path),
                       'after_frame_path': str(after_path), 'frame_path': str(after_path),
                       'before_time_s': frame.timestamp, 'after_time_s': after.timestamp,
                       'action_completed_reason': completion.get('reason'), 'observation_window_s': window,
                       'after_elapsed_s': round(after.timestamp-completion['monotonic'], 3),
                       'concurrent_action_ids': overlap,
                       'message': 'Python requested native input; its actual completion and visual aftermath are archived.'}
            self._notify('reflex_receipt', **receipt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._notify('reflex_effect_error', rule=name, episode_id=frame.episode_id, action_id=action_id,
                         error=f'{type(exc).__name__}: {exc}')

    def _after_latest_input(self, frame):
        latest = next((row for row in reversed(self.actions.history)
                       if row['event'] == 'action_started' and not row.get('passive')
                       and row.get('episode_id') == frame.episode_id), None)
        return latest is None or frame.timestamp > latest['monotonic']

    async def _finish_reaction(self, api, *, keep_waits=False):
        api._stopped = True
        cleanup = asyncio.create_task(self.actions.disable_source('rule:'+api._name, keep_waits=keep_waits))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # Releasing native input must publish its completion before replacement proceeds.
            await cleanup
            raise
        if not keep_waits and api._state['inputs']:
            await asyncio.gather(*(asyncio.shield(task) for task in list(api._state['inputs'])),
                                 return_exceptions=True)

    async def _run_program(self, name, state, card, episode_id, lifecycle, predecessors):
        api = _ProgramAPI(self, name, state, card, episode_id, lifecycle)
        state['perception_api'] = state['api'] = api
        try:
            if predecessors:
                await asyncio.gather(*(asyncio.shield(task) for task in predecessors), return_exceptions=True)
            api._check()
            await state['compiled'](api)
            await asyncio.gather(*(asyncio.shield(task) for task in list(state['inputs'])),
                                 return_exceptions=True)
            state['finished'] = True
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            locations = [entry for entry in traceback.extract_tb(exc.__traceback__)
                         if entry.filename == '<reflex-trigger>']
            if locations:
                location = locations[-1]
                line = card['trigger_py'].splitlines()[location.lineno - 1]
                error = (f'File "<reflex-trigger>", line {location.lineno}, in {location.name}\n'
                         f'    {line.strip()}\n{error}')
            state['error'] = error
            api._stats['error_count'] += 1
            self._notify('reflex_error', rule=name, episode_id=episode_id, error=error,
                         message='The program is paused; its source is retained for set_reflex edits.')
        finally:
            api._stopped = True
            # A replacement waits for this whole cleanup before reusing the same rule source.
            if predecessors:
                await asyncio.gather(*(asyncio.shield(task) for task in predecessors), return_exceptions=True)
            await self._finish_reaction(state.get('api', api))

    async def run(self):
        last = 0
        while not self.closed:
            frame = await self.frames.next(after=last)
            last = frame.frame_id
            if frame.episode_id != self.episode_id:
                self.episode_id = frame.episode_id
                self.stats.clear()
                self.shared_state.clear()
                for name in list(self.cards):
                    self._initial_state(name, 'episode_changed')
            playing = frame.phase == 'playing' and self.frames.phase == 'playing' and self.actions.can_run_reflexes
            for name, state in list(self.state.items()):
                if not playing:
                    self._stop_state(state)
                elif (self.mode != 'pure-vlm' and not state.get('error') and not state.get('finished')
                      and (not state.get('task') or state['task'].done())):
                    state['stopped'] = False
                    source_tasks = self.source_tasks.setdefault(name, set())
                    task = asyncio.create_task(self._run_program(name, state, self.cards[name],
                                               frame.episode_id, self.actions.lifecycle_epoch, tuple(source_tasks)),
                                               name='reflex-program-'+name)
                    state['task'] = self._track(task, self.program_tasks, source_tasks)

    def after_episode(self, record):
        for state in self.state.values():
            self._stop_state(state)
            state.pop('observation', None)
        counters = copy.deepcopy(self.stats)
        previews = [dict(x) for x in self.recent_previews if x['episode_id'] == record['episode_id']][-3:]
        candidates = record.get('metrics', {}).get('objective_target_results', [])
        values = [float(row['actual']) for row in candidates if isinstance(row.get('actual'), (int, float))]
        score = values[0] if len(values) == 1 else sum(values)
        if self.bank_enabled and values and self.cards and (self.best is None or score > self.best['score']):
            self.best = {'score': score, 'episode_id': record['episode_id'], 'cards': copy.deepcopy(self.cards),
                         'revision': self.revision, 'objectives': candidates}
            (self.directory / 'best_reflex.json').write_text(json.dumps(self.best, ensure_ascii=False, indent=2))
            self.events.add('reflex_champion_saved', score=score, revision=self.revision, episode_id=record['episode_id'])
        for name, counts in counters.items():
            counts['active'] = name in self.cards
        return {'episode_id': record['episode_id'], 'status': record['status'], 'objectives': candidates,
                'reflex_statistics': counters, 'recent_reflex_previews': previews,
                'zero_trigger_rules': [name for name, counts in counters.items()
                                       if counts['active'] and counts['trigger_count'] == 0 and counts['action_count'] == 0],
                'best_score': self.best['score'] if self.best else None, 'restored_best': False}

    def restore_best(self):
        if self.best is None or not self.best.get('cards'):
            raise ValueError('No completed native episode has established a nonempty reflex champion yet')
        previous = self.state
        for state in previous.values():
            self._stop_state(state)
        self.cards = copy.deepcopy(self.best['cards'])
        self.revision += 1
        self.shared_state.clear()
        for name in self.cards:
            self._initial_state(name)
        self.state = {name: self.state[name] for name in self.cards}
        self._save()
        restored = copy.deepcopy(self.list())
        self._notify('reflex_champion_restored', score=self.best['score'], revision=self.revision,
                     episode_id=self.frames.episode_id, reflexes=restored)
        return {'status': 'restored', 'score': self.best['score'], 'rules': list(self.cards),
                'source_episode_id': self.best['episode_id'],
                'source_objectives': self.best.get('objectives', []), 'reflexes': restored}

    async def close(self):
        self.closed = True
        await self.wait_pending()
        await self.wait_effects()
        await self.http.aclose()
