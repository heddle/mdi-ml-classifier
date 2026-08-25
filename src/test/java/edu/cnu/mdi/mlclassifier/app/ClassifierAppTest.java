package edu.cnu.mdi.mlclassifier.app;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.UUID;
import java.util.prefs.BackingStoreException;
import java.util.prefs.Preferences;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import edu.cnu.mdi.mlclassifier.onnx.OnnxImageClassifier;

class ClassifierAppTest {

    @AfterEach
    void clearConfiguration() {
        System.clearProperty(ClassifierApp.MODEL_PATH_PROPERTY);
        System.clearProperty(ClassifierApp.LABELS_PATH_PROPERTY);
    }

    @Test
    void commandLinePathsOverrideDefaults() {
        ClassifierApp.applyCommandLinePaths(new String[] {
                "--model=models/custom.onnx", "--labels=models/custom.txt" });

        assertEquals(Path.of("models/custom.onnx").toAbsolutePath().normalize(),
                ClassifierApp.configuredPath(ClassifierApp.MODEL_PATH_PROPERTY,
                        Path.of("ignored.onnx")));
        assertEquals(Path.of("models/custom.txt").toAbsolutePath().normalize(),
                ClassifierApp.configuredPath(ClassifierApp.LABELS_PATH_PROPERTY,
                        Path.of("ignored.txt")));
    }

	@Test
	void savedProfileIsTheDefaultButCommandLineStillWins() throws BackingStoreException {
		Preferences preferences = Preferences.userRoot().node(
				"mdi-classifier-tests/" + UUID.randomUUID());
		try {
			ModelProfileStore store = new ModelProfileStore(preferences);
			ModelProfile saved = new ModelProfile(Path.of("models/resnet.onnx"),
					Path.of("models/labels.txt"), OnnxImageClassifier.NormType.SCALE_0_1);
			store.save(saved);
			assertEquals(saved, ClassifierApp.startupProfile(store));

			System.setProperty(ClassifierApp.MODEL_PATH_PROPERTY, "models/explicit.onnx");
			assertEquals(Path.of("models/explicit.onnx").toAbsolutePath().normalize(),
					ClassifierApp.startupProfile(store).modelPath());
		} finally {
			preferences.removeNode();
		}
	}
}
