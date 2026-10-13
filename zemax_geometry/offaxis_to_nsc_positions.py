import argparse
import csv
import math
import re
from pathlib import Path


def parse_plain_pairs(lines):
    pairs = []
    idx = 1
    num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("!") or line.startswith("#"):
            continue
        nums = num_re.findall(line)
        if len(nums) >= 2:
            x_deg = float(nums[0])
            y_deg = float(nums[1])
            pairs.append((idx, x_deg, y_deg))
            idx += 1
    return pairs


def parse_zpl_pairs(lines):
    pat = re.compile(
        r"^\s*SYSP\s+10([23])\s*,\s*(\d+)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$",
        re.IGNORECASE,
    )
    by_id = {}
    for raw in lines:
        m = pat.match(raw)
        if not m:
            continue
        axis = m.group(1)
        field_id = int(m.group(2))
        val = float(m.group(3))
        rec = by_id.setdefault(field_id, {"x": None, "y": None})
        if axis == "3":
            rec["x"] = val
        else:
            rec["y"] = val

    pairs = []
    for field_id in sorted(by_id.keys()):
        x_deg = by_id[field_id]["x"]
        y_deg = by_id[field_id]["y"]
        if x_deg is None or y_deg is None:
            continue
        pairs.append((field_id, x_deg, y_deg))
    return pairs


def parse_input_file(path, input_format):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if input_format == "zpl":
        return parse_zpl_pairs(lines)
    if input_format == "plain":
        return parse_plain_pairs(lines)

    zpl_pairs = parse_zpl_pairs(lines)
    if zpl_pairs:
        return zpl_pairs
    return parse_plain_pairs(lines)


def offaxis_to_positions(pairs, distance_mm, z_position_mm):
    out = []
    for field_id, x_deg, y_deg in pairs:
        x_mm = distance_mm * math.tan(math.radians(x_deg))
        y_mm = distance_mm * math.tan(math.radians(y_deg))
        out.append((field_id, x_deg, y_deg, x_mm, y_mm, z_position_mm))
    return out


def write_csv(rows, output_path):
    header = [
        "id",
        "offaxis_x_deg",
        "offaxis_y_deg",
        "X_position_mm",
        "Y_position_mm",
        "Z_position_mm",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def parse_angles_from_log(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    rot_matches = re.findall(r"ZEMAX\s+ROTATOR\s+ANGLE\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text, re.IGNORECASE)
    det_matches = re.findall(r"ZEMAX\s+DETECTOR\s+ANGLE\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text, re.IGNORECASE)

    if not rot_matches or not det_matches:
        raise ValueError("Nel file log non trovo entrambe le righe ZEMAX ROTATOR ANGLE / ZEMAX DETECTOR ANGLE.")

    rotator_deg = float(rot_matches[-1])
    detector_deg = float(det_matches[-1])
    return rotator_deg, detector_deg


def write_surface_updates(output_path, surface20, surface27, rotator_deg, detector_deg):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["surface", "rotation_deg"])
        w.writerow([surface20, rotator_deg])
        w.writerow([surface27, detector_deg])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="File con coppie X/Y off-axis in gradi (ZPL o plain).")
    parser.add_argument(
        "--input-format",
        choices=["auto", "zpl", "plain"],
        default="auto",
        help="Formato input: auto (default), zpl, plain.",
    )
    parser.add_argument(
        "--distance-mm",
        type=float,
        required=True,
        help="Distanza geometrica L (mm) usata nella conversione X=L*tan(theta_x), Y=L*tan(theta_y).",
    )
    parser.add_argument(
        "--z-position-mm",
        type=float,
        required=True,
        help="Valore costante da assegnare a Z position per tutte le source.",
    )
    parser.add_argument(
        "--output",
        default="nsc_source_positions.csv",
        help="CSV di output (default: nsc_source_positions.csv).",
    )
    parser.add_argument(
        "--angles-log",
        default=None,
        help="File di testo/log che contiene le righe stampate: ZEMAX ROTATOR ANGLE e ZEMAX DETECTOR ANGLE.",
    )
    parser.add_argument(
        "--rotator-angle-deg",
        type=float,
        default=None,
        help="Valore rotazione (deg) per la superficie 20 (override del log).",
    )
    parser.add_argument(
        "--detector-angle-deg",
        type=float,
        default=None,
        help="Valore rotazione (deg) per la superficie 27 (override del log).",
    )
    parser.add_argument(
        "--surface20",
        type=int,
        default=20,
        help="Numero superficie da aggiornare col valore ZEMAX ROTATOR ANGLE (default: 20).",
    )
    parser.add_argument(
        "--surface27",
        type=int,
        default=27,
        help="Numero superficie da aggiornare col valore ZEMAX DETECTOR ANGLE (default: 27).",
    )
    parser.add_argument(
        "--surface-output",
        default="zemax_surface_rotations.csv",
        help="CSV con rotazioni superfici (default: zemax_surface_rotations.csv).",
    )
    args = parser.parse_args()

    pairs = parse_input_file(args.input, args.input_format)
    if not pairs:
        raise SystemExit("Nessuna coppia X/Y trovata nel file di input.")

    rows = offaxis_to_positions(pairs, args.distance_mm, args.z_position_mm)
    write_csv(rows, args.output)

    rotator_deg = args.rotator_angle_deg
    detector_deg = args.detector_angle_deg

    if rotator_deg is None or detector_deg is None:
        if args.angles_log is not None:
            log_rot, log_det = parse_angles_from_log(args.angles_log)
            if rotator_deg is None:
                rotator_deg = log_rot
            if detector_deg is None:
                detector_deg = log_det

    if rotator_deg is not None and detector_deg is not None:
        write_surface_updates(
            args.surface_output,
            args.surface20,
            args.surface27,
            rotator_deg,
            detector_deg,
        )
        print(f"Rotazioni superfici scritte in: {args.surface_output}")
    else:
        print("Rotazioni superfici non scritte: passa --angles-log oppure --rotator-angle-deg e --detector-angle-deg.")

    print(f"Trovate {len(rows)} source. Output scritto in: {args.output}")
    print("id, offaxis_x_deg, offaxis_y_deg, X_position_mm, Y_position_mm, Z_position_mm")
    for r in rows:
        print(f"{r[0]}, {r[1]:.6f}, {r[2]:.6f}, {r[3]:.6f}, {r[4]:.6f}, {r[5]:.6f}")


if __name__ == "__main__":
    main()
