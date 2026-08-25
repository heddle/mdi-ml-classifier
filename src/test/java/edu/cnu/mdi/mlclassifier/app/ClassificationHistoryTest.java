package edu.cnu.mdi.mlclassifier.app;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.Test;

import edu.cnu.mdi.mlclassifier.model.ClassScore;
import edu.cnu.mdi.mlclassifier.model.InferenceSummary;
import edu.cnu.mdi.mlclassifier.onnx.OnnxImageClassifier;

class ClassificationHistoryTest {

	@Test
	void boundsHistoryAndExportsStructuredRuns() {
		ClassificationHistory history = new ClassificationHistory(1);
		history.add(run("first", .5));
		history.add(run("sorrel, \"horse\"", .997));

		assertEquals(1, history.runs().size());
		assertTrue(history.latestAsText().contains("sorrel, \"horse\""));
		String csv = history.asCsv();
		assertTrue(csv.startsWith("timestamp,image,model,model_bytes"));
		assertTrue(csv.contains("\"sorrel, \"\"horse\"\"\""));
		assertTrue(csv.contains(",25,0.04200000000,0.4200000000,"));
	}

	@Test
	void rejectsInvalidCapacity() {
		org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
				() -> new ClassificationHistory(0));
	}

	private static ClassificationRun run(String label, double probability) {
		return new ClassificationRun(Instant.parse("2026-08-25T12:00:00Z"),
				Path.of("images/horse,picture.jpg"),
				new ModelProfile(Path.of("models/resnet.onnx"), Path.of("models/labels.txt"),
						OnnxImageClassifier.NormType.RESNET),
				1024, List.of(new ClassScore(label, probability)),
				new InferenceSummary(25, -1, 4, 1, probability, .042, .42));
	}
}
