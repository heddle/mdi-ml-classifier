"""Generate the deterministic ONNX model used by the Java integration test."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> None:
    model_input = helper.make_tensor_value_info(
        "image", TensorProto.FLOAT, [1, 3, 2, 2]
    )
    model_output = helper.make_tensor_value_info(
        "scores", TensorProto.FLOAT, [1, 3]
    )
    average_channels = helper.make_node(
        "ReduceMean",
        inputs=["image"],
        outputs=["scores"],
        axes=[2, 3],
        keepdims=0,
    )
    graph = helper.make_graph(
        [average_channels], "tiny_rgb_classifier", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="mdi-ml-classifier-tests",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    output = (
        Path(__file__).parents[1]
        / "resources"
        / "edu"
        / "cnu"
        / "mdi"
        / "mlclassifier"
        / "onnx"
        / "tiny-rgb-classifier.onnx"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
