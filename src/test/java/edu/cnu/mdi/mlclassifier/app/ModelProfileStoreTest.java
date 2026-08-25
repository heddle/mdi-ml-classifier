package edu.cnu.mdi.mlclassifier.app;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.UUID;
import java.util.prefs.BackingStoreException;
import java.util.prefs.Preferences;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import edu.cnu.mdi.mlclassifier.onnx.OnnxImageClassifier;

class ModelProfileStoreTest {

	private Preferences preferences;
	private ModelProfileStore store;

	@BeforeEach
	void createIsolatedStore() {
		preferences = Preferences.userRoot().node("mdi-classifier-tests/" + UUID.randomUUID());
		store = new ModelProfileStore(preferences);
	}

	@AfterEach
	void removeIsolatedStore() throws BackingStoreException {
		preferences.removeNode();
	}

	@Test
	void savesCompleteProfilesAndTracksLastSuccessfulModel() {
		ModelProfile resnet = new ModelProfile(Path.of("models/resnet.onnx"),
				Path.of("models/imagenet.txt"), OnnxImageClassifier.NormType.RESNET);
		ModelProfile mobile = new ModelProfile(Path.of("models/mobile.onnx"), null,
				OnnxImageClassifier.NormType.SCALE_NEG1_1);

		store.save(resnet);
		store.save(mobile);

		assertEquals(resnet, store.find(Path.of("models/resnet.onnx")).orElseThrow());
		assertEquals(mobile, store.last().orElseThrow());
	}

	@Test
	void removingLastProfileClearsLastSelection() {
		ModelProfile profile = new ModelProfile(Path.of("models/missing.onnx"), null,
				OnnxImageClassifier.NormType.SCALE_0_1);
		store.save(profile);

		store.remove(profile.modelPath());

		assertTrue(store.find(profile.modelPath()).isEmpty());
		assertTrue(store.last().isEmpty());
	}
}
