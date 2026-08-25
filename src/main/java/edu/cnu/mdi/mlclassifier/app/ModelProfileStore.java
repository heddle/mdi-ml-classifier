package edu.cnu.mdi.mlclassifier.app;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;
import java.util.Optional;
import java.util.prefs.BackingStoreException;
import java.util.prefs.Preferences;

import edu.cnu.mdi.mlclassifier.onnx.OnnxImageClassifier;

/** Preference-backed model profiles keyed by normalized ONNX model path. */
final class ModelProfileStore {

	private static final String LAST_MODEL = "lastModel";
	private final Preferences preferences;

	ModelProfileStore(Preferences preferences) {
		this.preferences = Objects.requireNonNull(preferences, "preferences");
	}

	/** Save a profile and mark it as the last successfully loaded profile. */
	void save(ModelProfile profile) {
		Objects.requireNonNull(profile, "profile");
		Preferences node = profileNode(profile.modelPath());
		node.put("model", profile.modelPath().toString());
		if (profile.labelsPath() == null) {
			node.remove("labels");
		} else {
			node.put("labels", profile.labelsPath().toString());
		}
		node.put("normalization", profile.normalization().name());
		preferences.put(LAST_MODEL, profile.modelPath().toString());
		flush(node);
		flush(preferences);
	}

	Optional<ModelProfile> find(Path modelPath) {
		if (modelPath == null) {
			return Optional.empty();
		}
		Path normalized = modelPath.toAbsolutePath().normalize();
		Preferences node = profileNode(normalized);
		String storedModel = node.get("model", null);
		if (storedModel == null || !normalized.equals(Path.of(storedModel).toAbsolutePath().normalize())) {
			return Optional.empty();
		}
		return read(node);
	}

	Optional<ModelProfile> last() {
		String model = preferences.get(LAST_MODEL, null);
		return model == null || model.isBlank() ? Optional.empty() : find(Path.of(model));
	}

	void remove(Path modelPath) {
		if (modelPath == null) {
			return;
		}
		Path normalized = modelPath.toAbsolutePath().normalize();
		try {
			profileNode(normalized).removeNode();
			if (normalized.toString().equals(preferences.get(LAST_MODEL, null))) {
				preferences.remove(LAST_MODEL);
			}
			flush(preferences);
		} catch (BackingStoreException ignored) {
			// Losing a convenience profile is non-fatal.
		}
	}

	private Optional<ModelProfile> read(Preferences node) {
		try {
			Path model = Path.of(node.get("model", ""));
			String labelsValue = node.get("labels", null);
			Path labels = labelsValue == null || labelsValue.isBlank() ? null : Path.of(labelsValue);
			OnnxImageClassifier.NormType norm = OnnxImageClassifier.NormType.valueOf(
					node.get("normalization", OnnxImageClassifier.NormType.RESNET.name()));
			return Optional.of(new ModelProfile(model, labels, norm));
		} catch (IllegalArgumentException exception) {
			return Optional.empty();
		}
	}

	private Preferences profileNode(Path modelPath) {
		return preferences.node(profileKey(modelPath));
	}

	private static String profileKey(Path modelPath) {
		try {
			byte[] digest = MessageDigest.getInstance("SHA-256").digest(
					modelPath.toString().getBytes(StandardCharsets.UTF_8));
			return HexFormat.of().formatHex(digest);
		} catch (NoSuchAlgorithmException impossible) {
			throw new IllegalStateException("SHA-256 is unavailable", impossible);
		}
	}

	private static void flush(Preferences preferences) {
		try {
			preferences.flush();
		} catch (BackingStoreException ignored) {
			// Persistence failure is non-fatal for this convenience feature.
		}
	}
}
