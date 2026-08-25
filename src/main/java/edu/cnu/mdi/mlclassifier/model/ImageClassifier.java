package edu.cnu.mdi.mlclassifier.model;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/** Asynchronous image-classification contract used by the Swing view. */
public interface ImageClassifier {

    /** Classify an image and return the highest-scoring classes. */
    CompletableFuture<List<ClassScore>> classifyAsync(BufferedImage image, int topK);

    /** Return immutable descriptive metadata for the loaded model. */
    default List<String> getModelMetaData() {
        return List.of();
    }

    /** Return immutable diagnostics from the most recent inference. */
    default List<String> getInferenceOutput() {
        return List.of();
    }

	/** Return structured measurements from the most recent completed inference. */
	default Optional<InferenceSummary> getInferenceSummary() {
		return Optional.empty();
	}
}
