package edu.cnu.mdi.mlclassifier.app;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

import edu.cnu.mdi.mlclassifier.model.ClassScore;
import edu.cnu.mdi.mlclassifier.model.InferenceSummary;

/** Bounded in-memory history with text and CSV export formatting. */
final class ClassificationHistory {

	private final int maximumSize;
	private final List<ClassificationRun> runs = new ArrayList<>();

	ClassificationHistory(int maximumSize) {
		if (maximumSize < 1) {
			throw new IllegalArgumentException("maximumSize must be positive");
		}
		this.maximumSize = maximumSize;
	}

	void add(ClassificationRun run) {
		runs.add(0, Objects.requireNonNull(run, "run"));
		if (runs.size() > maximumSize) {
			runs.subList(maximumSize, runs.size()).clear();
		}
	}

	List<ClassificationRun> runs() {
		return List.copyOf(runs);
	}

	boolean isEmpty() {
		return runs.isEmpty();
	}

	void clear() {
		runs.clear();
	}

	String latestAsText() {
		if (runs.isEmpty()) {
			return "";
		}
		ClassificationRun run = runs.get(0);
		StringBuilder text = new StringBuilder();
		text.append("Time: ").append(run.timestamp()).append('\n');
		text.append("Image: ").append(run.imagePath() == null ? "in-memory image" : run.imagePath()).append('\n');
		text.append("Model: ").append(run.profile().modelPath()).append('\n');
		text.append("Labels: ").append(run.profile().labelsPath() == null ? "none" : run.profile().labelsPath()).append('\n');
		text.append("Normalization: ").append(run.profile().normalization()).append('\n');
		if (run.inference() != null) {
			text.append(String.format(Locale.ROOT, "Inference: %d ms; entropy %.3f bits (%.2f%% normalized)%n",
					run.inference().durationMillis(), run.inference().entropyBits(),
					run.inference().normalizedEntropyPercent()));
		}
		text.append("\nRank\tClass\tPredicted probability\n");
		for (int index = 0; index < run.results().size(); index++) {
			ClassScore score = run.results().get(index);
			text.append(String.format(Locale.ROOT, "%d\t%s\t%.8f%n",
					index + 1, score.label(), score.score()));
		}
		return text.toString();
	}

	String asCsv() {
		StringBuilder csv = new StringBuilder(
				"timestamp,image,model,model_bytes,labels,normalization,inference_ms,entropy_bits,normalized_entropy_percent,rank,class,predicted_probability\n");
		for (ClassificationRun run : runs) {
			InferenceSummary summary = run.inference();
			for (int index = 0; index < run.results().size(); index++) {
				ClassScore score = run.results().get(index);
				appendCsv(csv, run.timestamp().toString());
				appendCsv(csv, run.imagePath() == null ? "" : run.imagePath().toString());
				appendCsv(csv, run.profile().modelPath().toString());
				csv.append(run.modelBytes()).append(',');
				appendCsv(csv, run.profile().labelsPath() == null ? "" : run.profile().labelsPath().toString());
				appendCsv(csv, run.profile().normalization().name());
				csv.append(summary == null ? "" : summary.durationMillis()).append(',');
				csv.append(summary == null ? "" : format(summary.entropyBits())).append(',');
				csv.append(summary == null ? "" : format(summary.normalizedEntropyPercent())).append(',');
				csv.append(index + 1).append(',');
				appendCsv(csv, score.label());
				csv.append(format(score.score())).append('\n');
			}
		}
		return csv.toString();
	}

	private static String format(double value) {
		return String.format(Locale.ROOT, "%.10g", value);
	}

	private static void appendCsv(StringBuilder csv, String value) {
		csv.append('"').append(value.replace("\"", "\"\"")).append("\",");
	}
}
