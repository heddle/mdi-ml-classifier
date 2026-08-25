package edu.cnu.mdi.mlclassifier.app;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Objects;

import edu.cnu.mdi.mlclassifier.model.ClassScore;
import edu.cnu.mdi.mlclassifier.model.InferenceSummary;

/** Immutable snapshot of one completed classification. */
record ClassificationRun(Instant timestamp, Path imagePath, ModelProfile profile,
		long modelBytes, List<ClassScore> results, InferenceSummary inference) {

	ClassificationRun {
		timestamp = Objects.requireNonNull(timestamp, "timestamp");
		profile = Objects.requireNonNull(profile, "profile");
		if (modelBytes < 0) {
			throw new IllegalArgumentException("modelBytes must not be negative");
		}
		results = List.copyOf(Objects.requireNonNull(results, "results"));
		if (results.isEmpty()) {
			throw new IllegalArgumentException("results must not be empty");
		}
	}

	ClassScore topResult() {
		return results.get(0);
	}
}
