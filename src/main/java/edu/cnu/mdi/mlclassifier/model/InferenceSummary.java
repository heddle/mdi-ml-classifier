package edu.cnu.mdi.mlclassifier.model;

/** Structured measurements from one completed inference operation. */
public record InferenceSummary(long durationMillis, double outputMinimum,
		double outputMaximum, double probabilitySum, double topConfidence,
		double entropyBits, double normalizedEntropyPercent) {

	/** Validate inference measurements at their public boundary. */
	public InferenceSummary {
		if (durationMillis < 0) {
			throw new IllegalArgumentException("durationMillis must not be negative");
		}
		if (!Double.isFinite(outputMinimum) || !Double.isFinite(outputMaximum)
				|| !Double.isFinite(probabilitySum) || !Double.isFinite(topConfidence)
				|| !Double.isFinite(entropyBits) || !Double.isFinite(normalizedEntropyPercent)) {
			throw new IllegalArgumentException("inference measurements must be finite");
		}
	}
}
