package edu.cnu.mdi.mlclassifier.model;

import java.util.Objects;

/**
 * A classifier result pairing a human-readable class label with its
 * probability.
 *
 * @param label non-null class label
 * @param score finite probability in the range {@code [0, 1]}
 */
public record ClassScore(String label, double score) {

	/** Validate the result at its public boundary. */
	public ClassScore {
		Objects.requireNonNull(label, "label");
		if (label.isBlank()) {
			throw new IllegalArgumentException("label must not be blank");
		}
		if (!Double.isFinite(score) || score < 0.0 || score > 1.0) {
			throw new IllegalArgumentException("score must be a finite probability in [0, 1]");
		}
	}
}
