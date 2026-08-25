# MDI Machine Learning Classifier

This capstone application combines the MDI desktop framework, Swing,
sPlot, and ONNX Runtime in an end-to-end image-classification workflow.

## Requirements

- Java 17 or newer
- A locally installed `io.github.heddle:mdi:1.2.2-SNAPSHOT`
- An ONNX image-classification model

The default model location is `models/mobilenetv2-12.onnx`; the ImageNet label
file is already stored at `models/imagenet_labels.txt`. Large model binaries are
not committed to this repository. Obtain MobileNet V2 from the ONNX Model Zoo
or supply another compatible single-input, single-output RGB classifier.

## Build and run

```text
mvn clean package
mvn exec:java \
  -Dexec.mainClass=edu.cnu.mdi.mlclassifier.app.ClassifierApp \
  -Dexec.args="--model=/path/to/model.onnx --labels=/path/to/labels.txt"
```

The application also accepts `--model=/path/to/model.onnx` and
`--labels=/path/to/labels.txt`. When omitted, paths are resolved relative to
the launch directory.

Open an image by dropping it onto the classifier view or by choosing
**Image > Open Image…**. Successfully opened files are retained in
**Image > Recent Images**.

Use **Model > Open ONNX Model…** to load or replace the classifier without
restarting the application. The view keeps a recent-model list and offers
separate controls for the labels file and input normalization. If the startup
model is missing, the application remains usable and immediately offers a model
chooser.

Each recent model retains its own labels path and normalization setting. The
last successfully loaded profile is restored on the next launch; explicit
`--model` or `--labels` command-line arguments take precedence.

Use **Results > Number of Classes** to request 1, 3, 5, 10, or 20 ranked
classifications. The selection is remembered, and changing it reclassifies the
current image without blocking the Swing event-dispatch thread.

Completed classifications are retained in a bounded, in-memory comparison
history. **Results > Comparison History…** compares model, prediction,
confidence, inference time, uncertainty, model size, and normalization.
The latest run can be copied as text, and the complete history can be saved as
CSV with one row per ranked class.

## Obtaining models

The historical ONNX Model Zoo is now preserved rather than actively updated,
and its model files have moved to the
[ONNX Model Zoo organization on Hugging Face](https://huggingface.co/onnxmodelzoo).
The default model used by this project is the 14 MB
[MobileNet V2 opset-12 model](https://huggingface.co/onnxmodelzoo/mobilenetv2-12/tree/main).
Download `mobilenetv2-12.onnx`, then open it from the **Model** menu or place it
in `models/` before starting the application. The repository already includes
the corresponding ImageNet labels file.

Model downloads are deliberately not initiated by the application. Models can
be large, have model-specific licenses and preprocessing requirements, and may
use external data files. Keeping acquisition explicit avoids an implicit network
dependency and lets the user verify the model card and license first.

This demo expects a floating-point RGB image classifier with one rank-4 input
and one output. A model must be paired with the labels and normalization its
model card specifies; merely having an `.onnx` extension does not guarantee that
it is compatible with this application.

## Pipeline

1. Inspect the ONNX input tensor and determine NCHW or NHWC layout.
2. Center-crop the source to the model aspect ratio.
3. Resize and apply the configured channel normalization.
4. Run inference on a dedicated worker thread.
5. Preserve probability outputs or apply stable softmax to logits.
6. Present top classes in an sPlot bar chart and expose diagnostics through
   MDI feedback.

The application assumes one rank-4 RGB input and one floating-point output.
The model's preprocessing requirements must match the selected normalization.

## Tests

The test suite includes a generated 177-byte ONNX RGB classifier under
`src/test/resources`. It exercises real ONNX Runtime session creation, tensor
layout discovery, preprocessing, synchronous and asynchronous inference,
labels, structured measurements, and resource cleanup without depending on the
large models in `models/`. The reproducible generator is retained under
`src/test/scripts` and requires the Python `onnx` package only when regenerating
the fixture.
