import collections
import math
import cv2
import numpy as np


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return float(inter) / float(max(1, a[2] * a[3] + b[2] * b[3] - inter)) if inter > 0 else 0.0


def _nms(cands, iou_thr=0.35, max_k=12):
    kept = []
    for c in sorted(cands, key=lambda x: x["score"], reverse=True):
        if all(_iou(c["box"], p["box"]) < iou_thr for p in kept):
            kept.append(c)
            if len(kept) >= max_k:
                break
    return kept


def _crop_roi(arr, roi):
    if roi is None:
        return arr, 0, 0
    h, w = arr.shape[:2]
    x1, y1 = max(0, min(w - 1, int(roi[0]))), max(0, min(h - 1, int(roi[1])))
    x2, y2 = max(x1 + 1, min(w, int(roi[0] + roi[2]))), max(y1 + 1, min(h, int(roi[1] + roi[3])))
    return arr[y1:y2, x1:x2], x1, y1


def _fg_offset(patch):
    if patch is None or patch.ndim != 3 or patch.shape[0] < 6 or patch.shape[1] < 6:
        h, w = patch.shape[:2] if patch is not None else (0, 0)
        return 0.5 * w, 0.5 * h, None
    h, w = patch.shape[:2]
    border = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]], axis=0)
    bg = np.median(border, axis=0)
    diff = np.max(np.abs(patch.astype(np.float32) - bg), axis=2)
    fg = diff > 28.0
    ys, xs = np.where(fg)
    if xs.size < 8 or xs.size > 0.88 * (h * w):
        return 0.5 * w, 0.5 * h, None
    fg_rgb = np.median(patch[ys, xs], axis=0)
    return float(np.mean(xs)), float(np.mean(ys)), (bg, fg_rgb)


def _detect(rule, trk, images):
    frame, gray, dprev, dbg = trk["frame"], trk["gray"], trk["dprev"], trk["dbg"]
    roi, mode, space = rule.get("roi"), rule.get("mode"), rule.get("space", "2d")
    if mode is None:
        _img_k = str(rule.get("image") or rule.get("target_image") or "").strip().lower()
        mode = "motion" if (_img_k == "motion" and "motion" not in (images or {})) else "template" 
    dyn, mthr, max_k = rule.get("dynamics", "any"), float(rule.get("motion_thr", 4.5)), int(rule.get("max_k", 5))
    raw_limit = max(12, max_k * 4)
    raw = []
    if mode == "template":
        _ik = rule.get("image") or rule.get("target_image") or rule.get("template")
        patch = images.get(_ik) if _ik else rule.get("patch")
        if patch is None and _ik and images and _ik != "motion":
            anc_n = rule.get("anchor_image") or (rule.get("player") if isinstance(rule.get("player"), str) else None)
            cand_keys = [k for k in images if k != anc_n and "player" not in k.lower() and "anc" not in k.lower()]
            sub_m = [k for k in cand_keys if k.lower() in str(_ik).lower() or str(_ik).lower() in k.lower()]
            if sub_m:
                patch = images.get(sub_m[0])
            elif len(cand_keys) == 1:
                patch = images.get(cand_keys[0])
        if patch is None or patch.size == 0:
            return []
        vrgb, ox, oy = _crop_roi(frame, roi)
        vg, _, _ = _crop_roi(gray, roi)
        vh, vw = vg.shape[:2]
        use_half = (vh * vw > 160000)
        vrgb_s = cv2.resize(vrgb, (vw // 2, vh // 2), interpolation=cv2.INTER_AREA) if (use_half and vrgb.ndim == 3) else vrgb
        vg_s = cv2.resize(vg, (vw // 2, vh // 2), interpolation=cv2.INTER_AREA) if use_half else vg
        scales = rule.get("scales") or ((0.55, 0.75, 1.0, 1.35, 1.8) if space == "3d" else (1.0, 0.75, 1.3, 0.6, 1.6))
        flips = (False, True) if rule.get("flip_h", False) else (False,)
        ph0, pw0 = patch.shape[:2]
        use_rot90 = bool(rule.get("rotate_90", False) or (space != "3d" and rule.get("is_anchor", False) and max(pw0, ph0) >= 1.5 * max(1, min(pw0, ph0))))
        rots = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180) if use_rot90 else (None,)
        default_thr = 0.60 if space == "3d" else 0.65
        thr = float(rule.get("threshold", default_thr))
        patches_to_try = [(patch, 0, 0)]
        if patch.ndim == 3 and ph0 >= 12 and pw0 >= 12:
            border_p = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]], axis=0)
            bg_p = np.median(border_p, axis=0)
            fg_ys, fg_xs = np.where(np.max(np.abs(patch.astype(np.float32) - bg_p), axis=2) > 28.0)
            if 8 <= fg_xs.size <= 0.65 * (ph0 * pw0):
                tx1, ty1 = max(0, int(fg_xs.min()) - 2), max(0, int(fg_ys.min()) - 2)
                tx2, ty2 = min(pw0, int(fg_xs.max()) + 3), min(ph0, int(fg_ys.max()) + 3)
                if (tx2 - tx1) >= 8 and (ty2 - ty1) >= 8 and ((tx2 - tx1) * (ty2 - ty1)) < 0.80 * (pw0 * ph0):
                    patches_to_try.append((patch[ty1:ty2, tx1:tx2], tx1, ty1))
        for cur_patch, _px_off, _py_off in patches_to_try:
          for sc in scales:
            if sc != 1.0 and raw and space != "3d" and not rule.get("scales"):
                break
            for rot in rots:
                base_p = cv2.rotate(cur_patch, rot) if rot is not None else cur_patch
                for fl in flips:
                    if rot is not None and fl:
                        continue
                    p = cv2.flip(base_p, 1) if fl else base_p
                    if abs(sc - 1.0) > 1e-3:
                        p = cv2.resize(p, (max(8, round(p.shape[1] * sc)), max(8, round(p.shape[0] * sc))))
                    th, tw = p.shape[:2]
                    if th >= vh or tw >= vw or th < 6 or tw < 6:
                        continue
                    off_x, off_y, _ = _fg_offset(p) if p.ndim == 3 else (0.5 * tw, 0.5 * th, None)
                    step = 2 if (use_half and th >= 14 and tw >= 14) else 1
                    p_m = cv2.resize(p, (max(4, tw // 2), max(4, th // 2)), interpolation=cv2.INTER_AREA) if step == 2 else p
                    pg_m = cv2.cvtColor(p_m, cv2.COLOR_RGB2GRAY) if p_m.ndim == 3 else p_m
                    res_g = cv2.matchTemplate(vg_s if step == 2 else vg, pg_m, cv2.TM_CCOEFF_NORMED)
                    if p_m.ndim == 3 and vrgb.ndim == 3:
                        res_c = cv2.matchTemplate(vrgb_s if step == 2 else vrgb, p_m, cv2.TM_CCOEFF_NORMED)
                        res = np.maximum(res_c, 0.92 * res_g)
                    else:
                        res = res_g
                    ys, xs = np.where(res >= thr)
                    if ys.size:
                        sc_arr = res[ys, xs]
                        for idx in np.argsort(sc_arr)[::-1][:raw_limit]:
                            bx, by = ox + int(xs[idx]) * step, oy + int(ys[idx]) * step
                            raw.append({"box": (bx, by, tw, th), "cx": bx + off_x, "cy": by + off_y, "score": float(sc_arr[idx]), "scale": float(sc)})
        if not raw and rule.get("is_anchor", False) and patch.ndim == 3 and pw0 >= 2.2 * max(1, ph0):
            _, _, colors = _fg_offset(patch)
            if colors is not None:
                bg_c, fg_c = colors
                if float(np.max(np.abs(fg_c - bg_c))) > 35.0:
                    tgt = fg_c.astype(np.int16).reshape(1, 1, 3)
                    cmask = (np.max(np.abs(vrgb.astype(np.int16) - tgt), axis=2) <= 38).astype(np.uint8) * 255
                    cnts, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in cnts:
                        x, y, w, h = cv2.boundingRect(c)
                        if w * h >= 120 and not (y + oy < 140 and h < 45):
                            raw.append({"box": (ox + x, oy + y, w, h), "score": 0.55, "scale": 1.0})
    if (mode == "motion" or (mode in ("template", "color") and not raw and dyn == "moving")) and trk["idx"] >= 2:
        comb = cv2.min(dprev, cv2.scaleAdd(dbg, 1.25, np.zeros_like(dbg)))
        view, ox, oy = _crop_roi(comb, roi)
        gview, _, _ = _crop_roi(gray, roi)
        _, mask = cv2.threshold(view, int(rule.get("diff_thr", 15)), 255, cv2.THRESH_BINARY)
        row_fill = np.mean(mask > 0, axis=1)
        if np.any(row_fill > 0.45):
            vert_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 1), np.uint8))
            for r_idx in np.where(row_fill > 0.45)[0]:
                mask[r_idx, :] = vert_open[r_idx, :]
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a, max_a = int(rule.get("min_area", 140)), int(rule.get("max_area", 60000))
        min_w, min_h = int(rule.get("min_w", 14)), int(rule.get("min_h", 18))
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w >= min_w and h >= min_h and min_a <= w * h <= max_a and not (h < 28 and w > 2.4 * h):
                if float(np.std(gview[y : y + h, x : x + w])) < 8.0:
                    continue
                ms = float(np.mean(view[y : y + h, x : x + w]))
                sc = min(1.0, (ms / 36.0) * min(1.0, math.sqrt(w * h) / 26.0))
                raw.append({"box": (ox + x, oy + y, w, h), "score": sc, "scale": 1.0})
    elif mode == "color":
        vrgb, ox, oy = _crop_roi(frame, roi)
        _col = rule.get("color") or rule.get("rgb") or (255, 255, 255)
        _tol = rule.get("tol") if rule.get("tol") is not None else rule.get("tolerance", 38)
        tgt = np.array(_col, dtype=np.int16).reshape(1, 1, 3)
        mask = (np.max(np.abs(vrgb.astype(np.int16) - tgt), axis=2) <= int(_tol)).astype(np.uint8) * 255
        row_fill = np.mean(mask > 0, axis=1)
        if np.any(row_fill > 0.55):
            vert_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 1), np.uint8))
            for r_idx in np.where(row_fill > 0.55)[0]:
                mask[r_idx, :] = vert_open[r_idx, :]
        if dyn == "moving" and trk["idx"] >= 2:
            comb = cv2.addWeighted(dprev, 0.65, dbg, 0.35, 0.0)
            mview, _, _ = _crop_roi(comb, roi)
            _, mmask = cv2.threshold(mview, int(rule.get("diff_thr", 10)), 255, cv2.THRESH_BINARY)
            mmask = cv2.dilate(mmask, np.ones((7, 7), np.uint8), iterations=2)
            mask = cv2.bitwise_and(mask, mmask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a, max_a = int(rule.get("min_area", 90)), int(rule.get("max_area", 60000))
        min_w, min_h = int(rule.get("min_w", 10)), int(rule.get("min_h", 10))
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w >= min_w and h >= min_h and min_a <= w * h <= max_a and not (h < 16 and w > 5 * h):
                raw.append({"box": (ox + x, oy + y, w, h), "score": float(np.mean(mask[y : y + h, x : x + w])) / 255.0, "scale": 1.0})
        if not raw and images:
            anc_name = rule.get("anchor_image") or (rule.get("player") if isinstance(rule.get("player"), str) else None)
            fb_img = rule.get("image") if (rule.get("image") in images) else next((k for k in images if k != anc_name and "player" not in k.lower() and "anc" not in k.lower()), None)
            if fb_img and fb_img in images:
                raw.extend(_detect({**rule, "mode": "template", "image": fb_img}, trk, images))

    fh, fw = gray.shape[:2]
    flt = []
    for c in raw:
        bx, by, bw, bh = c["box"]
        x1, y1, x2, y2 = max(0, bx), max(0, by), min(fw, bx + bw), min(fh, by + bh)
        ms = 0.65 * float(np.mean(dprev[y1:y2, x1:x2])) + 0.35 * float(np.mean(dbg[y1:y2, x1:x2])) if (x2 > x1 and y2 > y1) else 0.0
        ms_prev = float(np.mean(dprev[y1:y2, x1:x2])) if (x2 > x1 and y2 > y1) else 0.0
        if dyn == "moving" and ms < mthr:
            continue
        cx, cy = c.get("cx", bx + 0.5 * bw), c.get("cy", by + 0.5 * bh)
        if ((mode == "color" and dyn == "moving") or (rule.get("is_anchor", False) and bw >= 2.6 * max(1, bh))) and max(bw, bh) >= 1.8 * max(1, min(bw, bh)) and trk["idx"] >= 2:
            sub_dp = dprev[y1:y2, x1:x2]
            ly, lx = np.where(sub_dp > 14)
            if lx.size >= 16:
                cx, cy = float(x1 + np.median(lx)), float(y1 + np.median(ly))
        c["x"], c["y"], c["w"], c["h"] = bx, by, bw, bh
        c["width"], c["height"], c["bbox"] = bw, bh, c["box"]
        c["motion"], c["cx"], c["cy"] = ms, cx, cy
        flt.append(c)

    kept = _nms(flt, iou_thr=0.35, max_k=raw_limit)
    tracks = trk["tracks"].setdefault(rule["name"], {})
    now, dt_def = trk["ts"], trk["dt"]
    for tid in [t for t, tr in list(tracks.items()) if (now - tr["ts"]) > 1.5]:
        tracks.pop(tid, None)
    used = set()
    assoc_r = max(125.0, min(220.0, 900.0 * dt_def))
    for c in kept:
        bid, bd = -1, assoc_r
        for tid, tr in tracks.items():
            if tid not in used and math.hypot(c["cx"] - tr["cx"], c["cy"] - tr["cy"]) < bd:
                bd, bid = math.hypot(c["cx"] - tr["cx"], c["cy"] - tr["cy"]), tid
        if bid < 0:
            bid = trk["ntid"]
            trk["ntid"] += 1
            tracks[bid] = {"h": collections.deque(maxlen=8), "cx": c["cx"], "cy": c["cy"], "ts": now}
        used.add(bid)
        tr = tracks[bid]
        tr["h"].append((now, c["cx"], c["cy"], c["scale"]))
        tr["cx"], tr["cy"], tr["ts"], c["tid"] = c["cx"], c["cy"], now, bid
        if len(tr["h"]) >= 2:
            t0, x0, y0, s0 = tr["h"][0]
            t1, x1, y1, s1 = tr["h"][-1]
            dt = max(1e-3, t1 - t0) if t1 > t0 else max(1e-3, (len(tr["h"]) - 1) * dt_def)
            c["vx"], c["vy"], c["vz"] = (x1 - x0) / dt, (y1 - y0) / dt, (s1 - s0) / dt
        else:
            c["vx"], c["vy"], c["vz"] = 0.0, 0.0, 0.0
        lead = float(rule.get("lead_s", 0.16))
        c["pred_x"], c["pred_y"] = c["cx"] + c["vx"] * lead, c["cy"] + c["vy"] * lead
    return kept


def _sig8(gray, box):
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[0] + box[2])), min(h, int(box[1] + box[3]))
    return cv2.resize(gray[y1:y2, x1:x2], (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) if (x2 - x1 >= 4 and y2 - y1 >= 4) else None
