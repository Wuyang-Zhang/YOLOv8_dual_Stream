from argparse import ArgumentParser
from pathlib import Path

import cv2

from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(description="Run dual-stream visual prediction with RGB + depth/IR images.")
    parser.add_argument(
        "--model",
        default="model/yolov8m-seg-2stream-ref-postfusion.yaml",
        help="Dual-stream model YAML or trained weight file.",
    )
    parser.add_argument("--rgb", required=True, help="Path to the RGB image.")
    parser.add_argument("--aux", required=True, help="Path to the second-stream image, e.g. depth or IR.")
    parser.add_argument("--output", default="runs/predict/result.png", help="Path to save the plotted result.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, 0, 0,1.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--show-labels", action="store_true", help="Draw class labels on the output image.")
    parser.add_argument("--show-conf", action="store_true", help="Draw confidence scores on the output image.")
    return parser.parse_args()


def validate_path(path_str, description):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def main():
    args = parse_args()

    model_path = validate_path(args.model, "Model file")
    rgb_path = validate_path(args.rgb, "RGB image")
    aux_path = validate_path(args.aux, "Second-stream image")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    results = model(
        [str(rgb_path), str(aux_path)],
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
    )

    plotted = results[0].plot(labels=args.show_labels, conf=args.show_conf)
    if not cv2.imwrite(str(output_path), plotted):
        raise RuntimeError(f"Failed to save result image to: {output_path}")

    print(f"Saved prediction result to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
