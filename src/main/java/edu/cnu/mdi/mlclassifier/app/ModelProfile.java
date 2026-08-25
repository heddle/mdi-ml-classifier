package edu.cnu.mdi.mlclassifier.app;

import java.nio.file.Path;
import java.util.Objects;

import edu.cnu.mdi.mlclassifier.onnx.OnnxImageClassifier;

/** Persistent configuration required to use an ONNX classifier correctly. */
record ModelProfile(Path modelPath, Path labelsPath,
		OnnxImageClassifier.NormType normalization) {

	ModelProfile {
		modelPath = Objects.requireNonNull(modelPath, "modelPath")
				.toAbsolutePath().normalize();
		labelsPath = labelsPath == null ? null : labelsPath.toAbsolutePath().normalize();
		normalization = Objects.requireNonNull(normalization, "normalization");
	}
}
