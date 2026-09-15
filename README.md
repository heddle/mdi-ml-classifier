# MDI Machine Learning Classifier

This capstone application combines the MDI desktop framework, Swing,
sPlot, and ONNX Runtime in an end-to-end image-classification workflow.

## Requirements

- Java 17 or newer
- Maven
- An ONNX image-classification model

MDI 1.2.2 and ONNX Runtime 1.23.2 are Maven dependencies and are obtained
automatically when the project is built. A separate local installation of MDI
is not required.

The default model location is `models/mobilenetv2-12.onnx`. The ImageNet label
file is included at `models/imagenet_labels.txt`. Large model binaries are not
committed to this repository. Obtain MobileNet V2 from the ONNX Model Zoo
distribution or supply another compatible single-input, single-output RGB
classifier.

## Build and run

Build the application with:

```text
mvn clean package
```

Run it with:

```text
mvn exec:java \
  -Dexec.mainClass=edu.cnu.mdi.mlclassifier.app.ClassifierApp
```

A model and labels file can optionally be supplied on the command line:

```text
mvn exec:java \
  -Dexec.mainClass=edu.cnu.mdi.mlclassifier.app.ClassifierApp \
  -Dexec.args="--model=/path/to/model.onnx --labels=/path/to/labels.txt"
```

When command-line paths are omitted, the default paths are resolved relative
to the launch directory.

If the startup model is missing or invalid, the application remains usable and
offers a model chooser rather than terminating.

## Images

Open an image by dropping it onto the classifier view or by choosing
**Image > Open Image…**. Successfully opened files are retained in
**Image > Recent Images**.

Image decoding and classification run in the background so the Swing
event-dispatch thread remains responsive. If a newer image or model request
supersedes work already in progress, stale results are discarded rather than
being allowed to replace the current display.

## Models and model profiles

Use **Model > Open ONNX Model…** to load or replace the classifier without
restarting the application.

The application maintains a recent-model list and supports separate controls
for the labels file and input normalization. Each recent model retains its own
labels path and normalization setting. The last successfully loaded profile is
restored on the next launch.

Explicit `--model` or `--labels` command-line arguments take precedence over
persisted settings.

Use **Model > Find Models Online…** to open the ONNX Model Zoo when desktop
browsing is supported.

## Results

Use **Results > Number of Classes** to request 1, 3, 5, 10, or 20 ranked
classifications. The selection is remembered, and changing it reclassifies the
current image without blocking the Swing event-dispatch thread.

Completed classifications are retained in a bounded, in-memory comparison
history. **Results > Comparison History…** compares model, prediction,
confidence, inference time, uncertainty, model size, and normalization.

The latest result can be copied as text, and the complete comparison history
can be exported as CSV with one row per ranked class.

Predicted probability is presented as model confidence, not as guaranteed
correctness or calibrated real-world accuracy.

The platform menu shortcut opens an image; adding Shift opens a model.

## Obtaining models

The historical ONNX Model Zoo is preserved rather than actively updated, and
its model files are available through the
[ONNX Model Zoo organization on Hugging Face](https://huggingface.co/onnxmodelzoo).

The default model used by this project is the approximately 14 MB
[MobileNet V2 opset-12 model](https://huggingface.co/onnxmodelzoo/mobilenetv2-12/tree/main).

Download `mobilenetv2-12.onnx`, then open it from the **Model** menu or place
it in `models/` before starting the application. The repository already includes
the corresponding ImageNet labels file.

Model downloads are deliberately not initiated by the application. Models can
be large, have model-specific licenses and preprocessing requirements, and may
use external data files. Keeping acquisition explicit avoids an implicit network
dependency and allows the user to review the model card and license first.

This demo expects a floating-point RGB image classifier with one rank-4 input
and one output. A model must be paired with the labels and normalization its
model card specifies; merely having an `.onnx` extension does not guarantee
compatibility with this application.

## Classification pipeline

1. Inspect the ONNX input tensor and determine NCHW or NHWC layout.
2. Center-crop the source image to the model aspect ratio.
3. Resize the image and apply the configured channel normalization.
4. Run inference on a dedicated worker thread.
5. Preserve probability outputs or apply stable softmax to logits.
6. Rank the requested number of classes.
7. Present the results in an sPlot bar chart and expose diagnostics through
   MDI feedback.

The application's preprocessing settings must match the requirements of the
selected model.

## Tests

The test suite includes a generated 177-byte ONNX RGB classifier under
`src/test/resources`. It exercises real ONNX Runtime session creation, tensor
layout discovery, preprocessing, synchronous and asynchronous inference,
labels, structured measurements, and resource cleanup without depending on the
large models in `models/`.

The reproducible generator is retained under `src/test/scripts` and requires
the Python `onnx` package only when regenerating the fixture.

Run the test suite with:

```text
mvn test
```
