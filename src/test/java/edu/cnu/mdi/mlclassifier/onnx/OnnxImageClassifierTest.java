package edu.cnu.mdi.mlclassifier.onnx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;

class OnnxImageClassifierTest {

	@Test
	void classifiesWithLocalMobileNetModelWhenAvailable() throws Exception {
		Path model = Path.of("models", "mobilenetv2-12.onnx");
		Path labels = Path.of("models", "imagenet_labels.txt");
		assumeTrue(Files.isRegularFile(model), "large ONNX model is not stored in Git");

		BufferedImage image = new BufferedImage(32, 24, BufferedImage.TYPE_INT_RGB);
		try (OnnxImageClassifier classifier = new OnnxImageClassifier(model, labels)) {
			var results = classifier.classify(image, 5);
			assertEquals(5, results.size());
			assertEquals(224, classifier.getInputWidth());
			assertEquals(224, classifier.getInputHeight());
		}
	}

	@Test
	void softmaxIsStableAndNormalized() {
		float[] probabilities = OnnxImageClassifier.softmax(new float[] { 1000, 1000, 999 });
		assertEquals(1.0, probabilities[0] + probabilities[1] + probabilities[2], 1.0e-6);
		assertEquals(probabilities[0], probabilities[1]);
		assertThrows(IllegalArgumentException.class, () -> OnnxImageClassifier.softmax(new float[0]));
		assertThrows(IllegalArgumentException.class,
				() -> OnnxImageClassifier.softmax(new float[] { Float.NaN }));
	}

	@Test
	void probabilityOutputIsNotSoftmaxedAgain() {
		float[] original = { 0.1f, 0.2f, 0.7f };
		float[] result = OnnxImageClassifier.probabilitiesFromOutput(original);
		assertArrayEquals(original, result);
		result[0] = 1f;
		assertEquals(0.1f, original[0], "returned probability array must be defensive");

		float[] logits = OnnxImageClassifier.probabilitiesFromOutput(new float[] { 1, 2, 3 });
		assertEquals(1.0, logits[0] + logits[1] + logits[2], 1.0e-6);
	}

	@Test
	void flattensSupportedOutputShapesAndRejectsOthers() {
		assertArrayEquals(new float[] { 1, 2 },
				OnnxImageClassifier.flattenToFloatArray(new float[][] { { 1, 2 } }));
		assertArrayEquals(new float[] { 3 },
				OnnxImageClassifier.flattenToFloatArray(new float[][][] { { { 3 } } }));
		assertThrows(IllegalArgumentException.class,
				() -> OnnxImageClassifier.flattenToFloatArray(new double[] { 1 }));
	}

	@Test
	void entropyHandlesCertainAndUniformDistributions() {
		assertEquals(0.0, OnnxImageClassifier.entropyBits(new float[] { 1, 0 }));
		assertEquals(2.0, OnnxImageClassifier.entropyBits(new float[] { .25f, .25f, .25f, .25f }), 1e-12);
		assertThrows(IllegalArgumentException.class,
				() -> OnnxImageClassifier.entropyBits(new float[] { -.1f, 1.1f }));
	}

	@Test
	void centerCropPreservesTargetAspectRatio() {
		assertEquals(new Rectangle(50, 0, 100, 100),
				OnnxImageClassifier.centerCrop(200, 100, 224, 224));
		assertEquals(new Rectangle(0, 50, 100, 100),
				OnnxImageClassifier.centerCrop(100, 200, 224, 224));
		assertEquals(new Rectangle(0, 0, 160, 90),
				OnnxImageClassifier.centerCrop(160, 90, 160, 90));
	}

	@Test
	void readsTrimmedNonEmptyLabels() throws Exception {
		assertEquals(List.of("tench", "goldfish"),
				OnnxImageClassifier.readLabels(Path.of("models", "imagenet_labels.txt")).subList(0, 2));
	}
}
