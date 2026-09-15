# Changelog

All notable changes to the MDI Machine Learning Classifier are documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

No changelog was kept during initial development, so the 1.0.1 entry has been
reconstructed from the project history.

## [Unreleased]

## [1.0.1] - 2026-09-15

### Added

- End-to-end ONNX image classification with automatic input-layout discovery,
  model-appropriate image preprocessing, stable softmax conversion for logits,
  ranked results, and an sPlot confidence display.
- Drag-and-drop image loading, an Open Image command, and a persistent Recent
  Images menu.
- Runtime ONNX model switching with persistent recent-model profiles, labels
  paths, and normalization settings.
- Graceful recovery when the startup model is missing or invalid, leaving the
  application usable and offering a model chooser.
- Configurable top-K results for 1, 3, 5, 10, or 20 classes.
- A bounded model-comparison history reporting predictions, confidence,
  inference time, uncertainty, model size, and normalization.
- Clipboard export for the latest result and CSV export for the complete
  comparison history.
- Platform-native keyboard shortcuts and a link to the preserved ONNX Model
  Zoo distribution.

### Changed

- Image decoding and ONNX inference now run in the background, with stale-work
  cancellation preventing an older request from replacing newer results.
- Application, sidebar, toolbar, dialog, and color-map sizing is responsive and
  platform-independent.
- Updated the framework dependency to the released MDI 1.2.3 used by the final
  reference-book manuscript.

### Fixed

- Improved path handling across platforms, validation and error reporting,
  model-session cleanup, and recovery from incompatible model metadata.
- Clarified that displayed probabilities are model confidence values, not a
  guarantee of correctness or calibrated real-world accuracy.

### Documentation & Testing

- Expanded README documentation for models, profiles, normalization, results,
  exporting, preprocessing, and model acquisition.
- Added unit tests across application state, model profiles, result history,
  plotting, and view behavior.
- Added a generated 177-byte ONNX integration fixture that exercises real ONNX
  Runtime session creation, preprocessing, inference, and resource cleanup
  without requiring a large external model.

[Unreleased]: https://github.com/heddle/mdi-ml-classifier/compare/v1.0.1...develop
[1.0.1]: https://github.com/heddle/mdi-ml-classifier/releases/tag/v1.0.1
