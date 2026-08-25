package edu.cnu.mdi.mlclassifier.app;

import java.awt.Desktop;
import java.awt.Toolkit;
import java.awt.datatransfer.StringSelection;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.EnumMap;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.prefs.Preferences;

import javax.swing.ButtonGroup;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JRadioButtonMenuItem;
import javax.swing.SwingUtilities;
import javax.swing.KeyStroke;

import ai.onnxruntime.OrtException;
import edu.cnu.mdi.app.BaseMDIApplication;
import edu.cnu.mdi.dialog.FileDialogs;
import edu.cnu.mdi.dialog.FileType;
import edu.cnu.mdi.io.RecentFiles;
import edu.cnu.mdi.io.RecentFilesMenu;
import edu.cnu.mdi.log.Log;
import edu.cnu.mdi.mlclassifier.model.ClassScore;
import edu.cnu.mdi.mlclassifier.onnx.OnnxImageClassifier;
import edu.cnu.mdi.mlclassifier.view.ImageClassifierView;
import edu.cnu.mdi.mlclassifier.view.PlotSupport;
import edu.cnu.mdi.splot.plot.PlotPanel;
import edu.cnu.mdi.splot.plot.PlotView;
import edu.cnu.mdi.util.PropertyUtils;
import edu.cnu.mdi.view.BaseView;
import edu.cnu.mdi.view.LogView;
import edu.cnu.mdi.view.ViewManager;

/** Desktop application for classifying images with an ONNX model. */
@SuppressWarnings("serial")
public class ClassifierApp extends BaseMDIApplication {
	public static final String MODEL_PATH_PROPERTY = "mdi.classifier.model";
	public static final String LABELS_PATH_PROPERTY = "mdi.classifier.labels";

	private PlotView plotView;
	private OnnxImageClassifier classifier;
	private ImageClassifierView imageView;
	private Path modelPath;
	private Path labelsPath;
	private OnnxImageClassifier.NormType normalization;
	private AtomicLong modelLoadSequence;
	private CompletableFuture<OnnxImageClassifier> modelLoadFuture;
	private RecentFiles recentModels;
	private RecentFilesMenu recentModelsHelper;
	private JMenu recentModelsMenu;
	private JMenuItem modelInfoItem;
	private EnumMap<OnnxImageClassifier.NormType, JRadioButtonMenuItem> normalizationItems;
	private ModelProfileStore modelProfiles;
	private ClassificationHistory classificationHistory;
	private JMenuItem showHistoryItem;
	private JMenuItem copyLatestItem;
	private JMenuItem saveHistoryItem;
	private JMenuItem clearHistoryItem;

	private static final FileType ONNX_FILE_TYPE = FileType.of("ONNX models", "onnx");
	private static final FileType LABEL_FILE_TYPE = FileType.of("Label files", "txt", "labels");
	private static final URI MODEL_ZOO_URI = URI.create("https://huggingface.co/onnxmodelzoo");
	/**
	 * Constructor.
	 *
	 * @param keyVals key-value pairs for configuring the base MDI application
	 */
	public ClassifierApp(Object... keyVals) {
		super(keyVals);
	}

	/**
	 * Create and register the initial set of views shown in the demo.
	 * <p>
	 * This method only builds views; it should not depend on the outer frame being
	 * shown or on final geometry.
	 */
	@Override
	protected void addInitialViews() {
		// BaseMDIApplication calls this override from its constructor, before this
		// subclass's instance-field initializers run. Initialize lifecycle state here.
		modelLoadSequence = new AtomicLong();
		normalizationItems = new EnumMap<>(OnnxImageClassifier.NormType.class);
		Preferences appPreferences = Preferences.userNodeForPackage(getClass());
		modelProfiles = new ModelProfileStore(appPreferences.node("modelProfiles"));
		classificationHistory = new ClassificationHistory(50);

		LogView logView = new LogView();
		ViewManager.getInstance().getViewMenu().addSeparator();
		logView.setVisible(false);

		plotView = new PlotView(PropertyUtils.TITLE, "Classification Results", PropertyUtils.FRACTION, 0.7,
				PropertyUtils.ASPECT, 1.2, PropertyUtils.VISIBLE, true);

		ModelProfile startupProfile = startupProfile(modelProfiles);
		modelPath = startupProfile.modelPath();
		labelsPath = startupProfile.labelsPath();
		normalization = startupProfile.normalization();

		imageView = new ImageClassifierView(unavailableClassifier());
		imageView.setResultConsumer(this::makeBarPlot);
		installModelMenu();
		installHistoryActions();
		if (Files.isRegularFile(modelPath)) {
			loadModel(modelPath, labelsPath, normalization, false);
		} else {
			String message = "ONNX model not found: " + modelPath.toAbsolutePath();
			Log.getInstance().warning(message);
			logView.setVisible(true);
			imageView.setStatusText("No model loaded. Choose Model > Open ONNX Model…");
			SwingUtilities.invokeLater(() -> offerModelRecovery(message));
		}


	}

	private static edu.cnu.mdi.mlclassifier.model.ImageClassifier unavailableClassifier() {
		return (image, topK) -> CompletableFuture.failedFuture(
				new IllegalStateException("No classifier model is loaded"));
	}

	private void installModelMenu() {
		recentModels = new RecentFiles(Preferences.userNodeForPackage(getClass())
				.node("models"), 10, "recentModel");
		recentModelsHelper = new RecentFilesMenu(recentModels, file -> {
			ModelProfile profile = modelProfiles.find(file.toPath())
					.orElse(new ModelProfile(file.toPath(), labelsPath, normalization));
			loadModel(profile, true);
		}, "models");

		JMenu modelMenu = new JMenu("Model");
		JMenuItem openModel = new JMenuItem("Open ONNX Model…");
		openModel.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_O,
				ImageClassifierView.menuShortcutMask()
						| InputEvent.SHIFT_DOWN_MASK));
		openModel.addActionListener(event -> chooseModel());
		modelMenu.add(openModel);

		recentModelsMenu = new JMenu("Recent Models");
		recentModelsHelper.rebuild(recentModelsMenu);
		modelMenu.add(recentModelsMenu);
		modelMenu.addSeparator();

		JMenuItem openLabels = new JMenuItem("Open Labels…");
		openLabels.addActionListener(event -> chooseLabels());
		modelMenu.add(openLabels);
		JMenuItem clearLabels = new JMenuItem("Use Class IDs (No Labels)");
		clearLabels.addActionListener(event -> {
			if (modelPath != null && Files.isRegularFile(modelPath)) {
				loadModel(modelPath, null, normalization, true);
			} else {
				labelsPath = null;
			}
		});
		modelMenu.add(clearLabels);

		JMenu normalizationMenu = new JMenu("Normalization");
		ButtonGroup normalizationGroup = new ButtonGroup();
		addNormalizationChoice(normalizationMenu, normalizationGroup,
				"ImageNet mean/std (ResNet)", OnnxImageClassifier.NormType.RESNET);
		addNormalizationChoice(normalizationMenu, normalizationGroup,
				"Scale to [0, 1]", OnnxImageClassifier.NormType.SCALE_0_1);
		addNormalizationChoice(normalizationMenu, normalizationGroup,
				"Scale to [-1, 1]", OnnxImageClassifier.NormType.SCALE_NEG1_1);
		modelMenu.add(normalizationMenu);
		modelMenu.addSeparator();

		modelInfoItem = new JMenuItem("Current Model Information…");
		modelInfoItem.setEnabled(false);
		modelInfoItem.addActionListener(event -> showModelInformation());
		modelMenu.add(modelInfoItem);
		JMenuItem findModels = new JMenuItem("Find Models Online…");
		findModels.addActionListener(event -> openModelZoo());
		modelMenu.add(findModels);
		BaseView.applyFocusFix(modelMenu, imageView);
		imageView.getJMenuBar().add(modelMenu);
	}

	private void addNormalizationChoice(JMenu menu, ButtonGroup group, String label,
			OnnxImageClassifier.NormType type) {
		JRadioButtonMenuItem item = new JRadioButtonMenuItem(label, normalization == type);
		item.addActionListener(event -> {
			if (normalization != type) {
				if (modelPath != null && Files.isRegularFile(modelPath)) {
					loadModel(modelPath, labelsPath, type, true);
				} else {
					normalization = type;
				}
			}
		});
		group.add(item);
		menu.add(item);
		normalizationItems.put(type, item);
	}

	private void chooseModel() {
		FileDialogs.openFile(imageView, "classifier-model", "Open ONNX Model",
				ONNX_FILE_TYPE).ifPresent(path -> loadModel(modelProfiles.find(path)
						.orElse(new ModelProfile(path, labelsPath, normalization)), true));
	}

	private void chooseLabels() {
		FileDialogs.openFile(imageView, "classifier-labels", "Open Class Labels",
				LABEL_FILE_TYPE).ifPresent(path -> {
					if (modelPath != null && Files.isRegularFile(modelPath)) {
						loadModel(modelPath, path, normalization, true);
					} else {
						labelsPath = path;
						imageView.setStatusText("Labels selected. Choose Model > Open ONNX Model…");
					}
				});
	}

	private void loadModel(Path requestedModel, Path requestedLabels,
			OnnxImageClassifier.NormType requestedNormalization, boolean showDialogOnFailure) {
		loadModel(new ModelProfile(requestedModel, requestedLabels, requestedNormalization),
				showDialogOnFailure);
	}

	private void loadModel(ModelProfile requestedProfile, boolean showDialogOnFailure) {
		Path normalizedModel = requestedProfile.modelPath();
		Path normalizedLabels = requestedProfile.labelsPath();
		OnnxImageClassifier.NormType requestedNormalization = requestedProfile.normalization();
		long request = modelLoadSequence.incrementAndGet();
		if (modelLoadFuture != null && !modelLoadFuture.isDone()) {
			modelLoadFuture.cancel(true);
		}
		imageView.setStatusText("Loading model " + normalizedModel.getFileName() + "…");
		modelLoadFuture = CompletableFuture.supplyAsync(() -> {
			try {
				if (!Files.isRegularFile(normalizedModel)) {
					throw new IOException("ONNX model not found: " + normalizedModel);
				}
				return normalizedLabels != null
						? new OnnxImageClassifier(normalizedModel, normalizedLabels, requestedNormalization)
						: new OnnxImageClassifier(normalizedModel, (List<String>) null, requestedNormalization);
			} catch (OrtException | IOException e) {
				throw new CompletionException(e);
			}
		});
		modelLoadFuture.whenComplete((loaded, error) -> SwingUtilities.invokeLater(() -> {
			if (request != modelLoadSequence.get()) {
				closeClassifier(loaded);
				return;
			}
			if (error != null) {
				Throwable root = error instanceof CompletionException && error.getCause() != null
						? error.getCause() : error;
				if (!(root instanceof CancellationException)) {
					handleModelLoadFailure(normalizedModel, root, showDialogOnFailure);
				}
				return;
			}
			OnnxImageClassifier previous = classifier;
			classifier = loaded;
			modelPath = normalizedModel;
			labelsPath = normalizedLabels;
			normalization = requestedNormalization;
			modelProfiles.save(new ModelProfile(modelPath, labelsPath, normalization));
			imageView.setActiveModelName(modelPath.getFileName().toString());
			imageView.setClassifier(loaded);
			recentModels.add(normalizedModel.toFile());
			recentModelsHelper.rebuild(recentModelsMenu);
			modelInfoItem.setEnabled(true);
			JRadioButtonMenuItem selectedNormalization = normalizationItems.get(normalization);
			if (selectedNormalization != null) {
				selectedNormalization.setSelected(true);
			}
			Log.getInstance().info("Loaded ONNX model: " + normalizedModel);
			if (previous != null) {
				CompletableFuture.runAsync(() -> closeClassifier(previous));
			}
		}));
	}

	private void handleModelLoadFailure(Path failedModel, Throwable failure, boolean showDialog) {
		if (!Files.isRegularFile(failedModel)) {
			recentModels.remove(failedModel.toFile());
			modelProfiles.remove(failedModel);
			recentModelsHelper.rebuild(recentModelsMenu);
		}
		String message = failure.getMessage() == null ? failure.toString() : failure.getMessage();
		Log.getInstance().warning("Unable to load ONNX model [" + failedModel + "]: " + message);
		imageView.setStatusText(classifier == null
				? "No model loaded. Choose Model > Open ONNX Model…"
				: "Could not replace model; the previous model remains active.");
		if (showDialog) {
			JOptionPane.showMessageDialog(imageView, message,
					"Open ONNX Model Failed", JOptionPane.ERROR_MESSAGE);
		} else if (classifier == null) {
			offerModelRecovery("The configured classifier could not be loaded:\n" + message);
		}
		JRadioButtonMenuItem activeNormalization = normalizationItems.get(normalization);
		if (activeNormalization != null) {
			activeNormalization.setSelected(true);
		}
	}

	private void offerModelRecovery(String message) {
		Object[] options = { "Choose Model…", "Cancel" };
		int choice = JOptionPane.showOptionDialog(imageView,
				message + "\n\nChoose an ONNX image-classification model now?",
				"Classifier Model Not Available", JOptionPane.DEFAULT_OPTION,
				JOptionPane.WARNING_MESSAGE, null, options, options[0]);
		if (choice == 0) {
			chooseModel();
		}
	}

	private void showModelInformation() {
		if (classifier == null) {
			return;
		}
		StringBuilder text = new StringBuilder();
		text.append("Model: ").append(modelPath).append('\n');
		text.append("Labels: ").append(labelsPath == null ? "none" : labelsPath).append('\n');
		text.append("Normalization: ").append(normalization).append("\n\n");
		classifier.getModelMetaData().forEach(line -> text.append(line).append('\n'));
		JOptionPane.showMessageDialog(imageView, text.toString(),
				"Current Model Information", JOptionPane.INFORMATION_MESSAGE);
	}

	private void openModelZoo() {
		try {
			if (!Desktop.isDesktopSupported()
					|| !Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
				throw new UnsupportedOperationException("Opening a browser is not supported on this desktop.");
			}
			Desktop.getDesktop().browse(MODEL_ZOO_URI);
		} catch (IOException | RuntimeException exception) {
			Log.getInstance().warning("Unable to open the model site: " + exception.getMessage());
			JOptionPane.showMessageDialog(imageView,
					"Open this address in a browser:\n" + MODEL_ZOO_URI,
					"ONNX Model Zoo", JOptionPane.INFORMATION_MESSAGE);
		}
	}

	private void installHistoryActions() {
		JMenu resultsMenu = findViewMenu("Results");
		if (resultsMenu == null) {
			return;
		}
		resultsMenu.addSeparator();
		showHistoryItem = new JMenuItem("Comparison History…");
		showHistoryItem.addActionListener(event -> showComparisonHistory());
		resultsMenu.add(showHistoryItem);
		copyLatestItem = new JMenuItem("Copy Latest Result");
		copyLatestItem.addActionListener(event -> copyLatestResult());
		resultsMenu.add(copyLatestItem);
		saveHistoryItem = new JMenuItem("Save History as CSV…");
		saveHistoryItem.addActionListener(event -> saveHistory());
		resultsMenu.add(saveHistoryItem);
		clearHistoryItem = new JMenuItem("Clear History");
		clearHistoryItem.addActionListener(event -> {
			int response = JOptionPane.showConfirmDialog(imageView,
					"Clear all classification comparison history?",
					"Clear Comparison History", JOptionPane.OK_CANCEL_OPTION,
					JOptionPane.QUESTION_MESSAGE);
			if (response == JOptionPane.OK_OPTION) {
				classificationHistory.clear();
				updateHistoryActions();
			}
		});
		resultsMenu.add(clearHistoryItem);
		updateHistoryActions();
	}

	private JMenu findViewMenu(String title) {
		for (int index = 0; index < imageView.getJMenuBar().getMenuCount(); index++) {
			JMenu menu = imageView.getJMenuBar().getMenu(index);
			if (menu != null && title.equals(menu.getText())) {
				return menu;
			}
		}
		return null;
	}

	private void recordClassification(List<ClassScore> results) {
		if (classifier == null || results == null || results.isEmpty()) {
			return;
		}
		long modelBytes = 0;
		try {
			modelBytes = Files.size(modelPath);
		} catch (IOException ignored) {
			// File size is useful comparison metadata, but not required.
		}
		classificationHistory.add(new ClassificationRun(Instant.now(),
				imageView.getCurrentImagePath(), new ModelProfile(modelPath, labelsPath, normalization),
				modelBytes, results, classifier.getInferenceSummary().orElse(null)));
		updateHistoryActions();
	}

	private void updateHistoryActions() {
		boolean available = !classificationHistory.isEmpty();
		showHistoryItem.setEnabled(available);
		copyLatestItem.setEnabled(available);
		saveHistoryItem.setEnabled(available);
		clearHistoryItem.setEnabled(available);
	}

	private void showComparisonHistory() {
		new ClassificationHistoryDialog(imageView, classificationHistory.runs()).setVisible(true);
	}

	private void copyLatestResult() {
		try {
			Toolkit.getDefaultToolkit().getSystemClipboard().setContents(
					new StringSelection(classificationHistory.latestAsText()), null);
			imageView.setStatusText("Latest classification copied to the clipboard.");
		} catch (RuntimeException exception) {
			Log.getInstance().warning("Unable to copy classification results: " + exception.getMessage());
			JOptionPane.showMessageDialog(imageView,
					"The clipboard is unavailable.\n" + exception.getMessage(),
					"Copy Results Failed", JOptionPane.ERROR_MESSAGE);
		}
	}

	private void saveHistory() {
		FileDialogs.saveFile(imageView, "classifier-results", "Save Classification History",
				"classification-history.csv", FileType.of("CSV files", "csv")).ifPresent(path -> {
					try {
						Files.writeString(path, classificationHistory.asCsv());
						imageView.setStatusText("Classification history saved to " + path.getFileName());
					} catch (IOException exception) {
						Log.getInstance().warning("Unable to save classification history: " + exception.getMessage());
						JOptionPane.showMessageDialog(imageView, exception.getMessage(),
								"Save Results Failed", JOptionPane.ERROR_MESSAGE);
					}
				});
	}

	private static void closeClassifier(OnnxImageClassifier toClose) {
		if (toClose == null) {
			return;
		}
		try {
			toClose.close();
		} catch (IOException e) {
			Log.getInstance().warning("Unable to close classifier cleanly: " + e.getMessage());
		}
	}

    // Plot the classification results using the associated PlotView.
	private void makeBarPlot(List<ClassScore> results) {
		recordClassification(results);
		PlotPanel plotPanel = PlotSupport.createBarPlot(results);
    	if (plotPanel != null) {
    		plotView.switchToPlotPanel(plotPanel);
    	}
    }

	/** Release the ONNX session and inference worker during application shutdown. */
	@Override
	protected void prepareForShutdown() {
		modelLoadSequence.incrementAndGet();
		if (modelLoadFuture != null) {
			modelLoadFuture.cancel(true);
		}
		closeClassifier(classifier);
		super.prepareForShutdown();
	}


	@Override
	protected String getApplicationId() {
		return "MDI-Classifier";
	}

	static Path configuredPath(String propertyName, Path fallback) {
		String configured = System.getProperty(propertyName);
		Path path = (configured == null || configured.isBlank()) ? fallback : Path.of(configured);
		return path.toAbsolutePath().normalize();
	}

	static ModelProfile startupProfile(ModelProfileStore profiles) {
		boolean explicitConfiguration = hasText(System.getProperty(MODEL_PATH_PROPERTY))
				|| hasText(System.getProperty(LABELS_PATH_PROPERTY));
		if (!explicitConfiguration) {
			var saved = profiles.last();
			if (saved.isPresent()) {
				return saved.get();
			}
		}
		Path model = configuredPath(MODEL_PATH_PROPERTY,
				Path.of("models", "mobilenetv2-12.onnx"));
		Path configuredLabels = configuredPath(LABELS_PATH_PROPERTY,
				Path.of("models", "imagenet_labels.txt"));
		Path labels = Files.isRegularFile(configuredLabels) ? configuredLabels : null;
		return new ModelProfile(model, labels, OnnxImageClassifier.NormType.RESNET);
	}

	private static boolean hasText(String value) {
		return value != null && !value.isBlank();
	}

	static void applyCommandLinePaths(String[] args) {
		if (args == null) {
			return;
		}
		for (String argument : args) {
			if (argument == null) continue;
			if (argument.startsWith("--model=")) {
				System.setProperty(MODEL_PATH_PROPERTY, argument.substring("--model=".length()));
			} else if (argument.startsWith("--labels=")) {
				System.setProperty(LABELS_PATH_PROPERTY, argument.substring("--labels=".length()));
			}
		}
	}

	/** Main entry point. */
	public static void main(String[] args) {
		applyCommandLinePaths(args);
		BaseMDIApplication.launch(() -> new ClassifierApp(
				PropertyUtils.TITLE, "MDI Machine Learning Classifier",
				PropertyUtils.FRACTION, 0.8));
	}
}
