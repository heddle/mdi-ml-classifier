package edu.cnu.mdi.mlclassifier.view;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.geom.Point2D.Double;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.prefs.Preferences;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.ButtonGroup;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JRadioButtonMenuItem;
import javax.swing.SwingConstants;

import edu.cnu.mdi.container.BaseContainer;
import edu.cnu.mdi.container.IContainer;
import edu.cnu.mdi.dialog.FileDialogs;
import edu.cnu.mdi.dialog.FileType;
import edu.cnu.mdi.feedback.FeedbackPane;
import edu.cnu.mdi.graphics.drawable.IDrawable;
import edu.cnu.mdi.graphics.toolbar.ToolBits;
import edu.cnu.mdi.io.RecentFiles;
import edu.cnu.mdi.io.RecentFilesMenu;
import edu.cnu.mdi.log.Log;
import edu.cnu.mdi.mlclassifier.model.ClassScore;
import edu.cnu.mdi.mlclassifier.model.ImageClassifier;
import edu.cnu.mdi.mlclassifier.onnx.OnnxImageClassifier;
import edu.cnu.mdi.swing.SwingSizingUtils;
import edu.cnu.mdi.transfer.FileDropHandler;
import edu.cnu.mdi.transfer.ImageFilters;
import edu.cnu.mdi.ui.fonts.Fonts;
import edu.cnu.mdi.util.PropertyUtils;
import edu.cnu.mdi.view.AbstractViewInfo;
import edu.cnu.mdi.view.BaseView;

/**
 * MDI view that displays a dropped image, runs ONNX classification, and exposes
 * image and inference details through the standard feedback pane.
 */
@SuppressWarnings("serial")
public class ImageClassifierView extends BaseView {

	// default side panel width (feedback)
	private static final int SIDE_PANEL_WIDTH = 250;
	private static final FileType IMAGE_FILE_TYPE = FileType.of(
			"Images", ImageIO.getReaderFileSuffixes());
	private static final int DEFAULT_TOP_K = 5;
	private static final int MAX_TOP_K = 100;
	private static final int[] TOP_K_CHOICES = { 1, 3, 5, 10, 20 };

	// status label
	private final JLabel statusLabel = new JLabel("Drop an image here (or use Image → Open Image)", SwingConstants.CENTER);

	private volatile ImageClassifier classifier;
	private final AtomicLong imageLoadSequence = new AtomicLong();
	private final AtomicLong classificationSequence = new AtomicLong();
	private CompletableFuture<BufferedImage> imageLoadFuture;
	private CompletableFuture<List<ClassScore>> classificationFuture;
	private RecentFiles recentImages;
	private RecentFilesMenu recentImagesHelper;
	private JMenu recentImagesMenu;
	private Preferences resultPreferences;
	private int topK;

	// current image
	private BufferedImage currentImage;

	// Optional: remember source for later (model metadata, etc.)
	private Path currentImagePath;

	// Current classification results
	private List<ClassScore> currentResults;

	// Rectangle where the image is drawn
	private Rectangle imageRect;

	// Consumer for classification results
	private Consumer<List<ClassScore>> classificationResultConsumer;

	/**
	 * Create an image-classifier view.
	 *
	 * @param classifier model runner owned by the application
	 * @param keyVals optional properties that override the view defaults
	 */
	public ImageClassifierView(ImageClassifier classifier, Object... keyVals) {
		super(viewProperties(keyVals));

		this.classifier = Objects.requireNonNull(classifier, "classifier");
		resultPreferences = Preferences.userNodeForPackage(getClass()).node("results");
		topK = validStoredTopK(resultPreferences.getInt("topK", DEFAULT_TOP_K));
		setFileFilter(ImageFilters.isActualImage);
		addStatusLabel();
		addFeedback();
		installImageMenu();
		installResultsMenu();

		// Set up drag and drop handling

		JComponent jc = (JComponent) getIContainer().getComponent();
		jc.setTransferHandler(new FileDropHandler(this));

		getIContainer().getFeedbackControl().addFeedbackProvider(this);

		IDrawable imageDrawer = new IDrawable() {

			@Override
			public void draw(Graphics2D g2, IContainer container) {
				drawImage(g2, container, currentImage);
			}
		};

		getIContainer().setBeforeDraw(imageDrawer);

	}

	/** Backward-compatible constructor for callers coupled to ONNX Runtime. */
	public ImageClassifierView(OnnxImageClassifier classifier, Object... keyVals) {
		this((ImageClassifier) classifier, keyVals);
	}

	@Override
	public AbstractViewInfo getViewInfo() {
		return new ImageClassifierViewInfo();
	}

	private static Object[] viewProperties(Object... overrides) {
		Object[] defaults = { PropertyUtils.TITLE, "Image Classifier", PropertyUtils.FRACTION, 0.7,
				PropertyUtils.ASPECT, 1.2, PropertyUtils.VISIBLE, true,
				PropertyUtils.TOOLBARBITS, ToolBits.INFO };
		if (overrides == null || overrides.length == 0) {
			return defaults;
		}
		Object[] properties = Arrays.copyOf(defaults, defaults.length + overrides.length);
		System.arraycopy(overrides, 0, properties, defaults.length, overrides.length);
		return properties;
	}

	// Add the status label below the image panel.
	private void addStatusLabel() {
		statusLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
		statusLabel.setFont(Fonts.defaultFont);
		statusLabel.setText("Model loaded. Drop an image above or use Image → Open Image…");
		statusLabel.setOpaque(true);
		statusLabel.setBackground(Color.lightGray);
		statusLabel.setForeground(Color.black);
		statusLabel.setBorder(BorderFactory.createLineBorder(Color.darkGray));
		add(statusLabel, BorderLayout.SOUTH);
	}

	// Add the feedback pane to the east side.
	private void addFeedback() {
		FeedbackPane fbp = initFeedback(Color.cyan, Color.black, 11);
		fbp.setPreferredSize(SwingSizingUtils.preferredSizeAtLeast(
				fbp, SIDE_PANEL_WIDTH, 1));
		add(fbp, BorderLayout.EAST);
		statusLabel.setBorder(BorderFactory.createLineBorder(Color.lightGray));
	}

	/**
	 * Set the consumer that will handle classification results.
	 *
	 * @param consumer a Consumer that processes a list of ClassScore objects
	 */
	public void setResultConsumer(Consumer<List<ClassScore>> consumer) {
		classificationResultConsumer = consumer;
	}

	/** @return the requested number of highest-scoring classifications */
	public int getTopK() {
		return topK;
	}

	/** @return the current image source, or {@code null} for an in-memory image */
	public Path getCurrentImagePath() {
		return currentImagePath;
	}

	/**
	 * Set the requested number of results. An image already on display is
	 * reclassified immediately.
	 *
	 * @param count result count from 1 through 100
	 */
	public void setTopK(int count) {
		if (count < 1 || count > MAX_TOP_K) {
			throw new IllegalArgumentException("topK must be between 1 and " + MAX_TOP_K);
		}
		if (!javax.swing.SwingUtilities.isEventDispatchThread()) {
			javax.swing.SwingUtilities.invokeLater(() -> setTopK(count));
			return;
		}
		if (topK == count) {
			return;
		}
		topK = count;
		resultPreferences.putInt("topK", count);
		if (currentImage != null) {
			setDecodedImage(currentImage, currentImagePath);
		}
	}

	/**
	 * Replace the classifier used by this view. If an image is already displayed,
	 * it is classified again with the replacement model.
	 *
	 * @param replacement the newly loaded classifier
	 */
	public void setClassifier(ImageClassifier replacement) {
		Objects.requireNonNull(replacement, "replacement");
		if (!javax.swing.SwingUtilities.isEventDispatchThread()) {
			javax.swing.SwingUtilities.invokeLater(() -> setClassifier(replacement));
			return;
		}
		classificationSequence.incrementAndGet();
		cancel(classificationFuture);
		classifier = replacement;
		currentResults = null;
		if (currentImage == null) {
			setStatusText("Model loaded. Drop an image above or use Image → Open Image…");
			getIContainer().refresh();
		} else {
			setDecodedImage(currentImage, currentImagePath);
		}
	}


	/**
	 * Set the image to display in this view and to be classified
	 *
	 * @param img        the image to display and classify
	 * @param sourcePath the optional source path of the image (may be null)
	 */
	public void setImage(BufferedImage img, Path sourcePath) {
		Objects.requireNonNull(img, "img");
		if (!javax.swing.SwingUtilities.isEventDispatchThread()) {
			javax.swing.SwingUtilities.invokeLater(() -> setImage(img, sourcePath));
			return;
		}
		imageLoadSequence.incrementAndGet();
		cancel(imageLoadFuture);
		setDecodedImage(img, sourcePath);
	}

	private void setDecodedImage(BufferedImage img, Path sourcePath) {
		long request = classificationSequence.incrementAndGet();
		cancel(classificationFuture);
		this.currentImagePath = sourcePath;
		currentImage = img;
		currentResults = null;
		setStatusText("Classifying image (top " + topK + ")...");

		classificationFuture = classifier.classifyAsync(img, topK);
		classificationFuture.whenComplete((results, err) -> {
			javax.swing.SwingUtilities.invokeLater(() -> {
					if (request != classificationSequence.get()) {
						return;
					}
					if (err != null) {
						Throwable root = (err instanceof CompletionException && err.getCause() != null) ? err.getCause()
								: err;
						if (root instanceof CancellationException) {
							return;
						}
						Log.getInstance().warning("ONNX inference failed: " + root.getMessage());
						setStatusText("Classification failed (see log).");
						return;
					}

					setStatusText("Classification complete.");
					currentResults = List.copyOf(results);
					if (classificationResultConsumer != null) {
						classificationResultConsumer.accept(currentResults);
					}
				});
			});
		getIContainer().refresh();
	}

	/**
	 * Update the status message displayed below the image.
	 *
	 * @param message the status message to display.
	 */
	public void setStatusText(String message) {
		Objects.requireNonNull(message, "message");
		if (!javax.swing.SwingUtilities.isEventDispatchThread()) {
			javax.swing.SwingUtilities.invokeLater(() -> setStatusText(message));
			return;
		}
		statusLabel.setText(message);
	}

	/**
	 * Handle files dropped on this view through drag and drop.
	 *
	 * @param files the dropped files.
	 */
	@Override
	public void filesDropped(List<File> files) {
		if (files == null || files.isEmpty()) {
			return;
		}
		openImageFile(files.get(0), false);
	}

	private void chooseImage() {
		FileDialogs.openFile(this, "classifier-image", "Open Image",
				IMAGE_FILE_TYPE).ifPresent(path -> openImageFile(path.toFile(), true));
	}

	/**
	 * Begin loading an image off the Swing EDT. A newer request supersedes an
	 * older load that has not completed.
	 */
	boolean openImageFile(File file, boolean showDialogOnFailure) {
		Objects.requireNonNull(file, "file");
		long request = imageLoadSequence.incrementAndGet();
		cancel(imageLoadFuture);
		setStatusText("Loading image...");
		imageLoadFuture = decodeImageAsync(file);
		imageLoadFuture.whenComplete((img, err) -> javax.swing.SwingUtilities.invokeLater(() -> {
			if (request != imageLoadSequence.get()) {
				return;
			}
			if (err == null) {
				setDecodedImage(img, file.toPath());
				recentImages.add(file);
				recentImagesHelper.rebuild(recentImagesMenu);
				Log.getInstance().info("Loaded image file: " + file.getAbsolutePath());
				return;
			}
			Throwable root = (err instanceof CompletionException && err.getCause() != null)
					? err.getCause() : err;
			if (root instanceof CancellationException) {
				return;
			}
			recentImages.remove(file);
			recentImagesHelper.rebuild(recentImagesMenu);
			Log.getInstance().warning("Error reading image file [" + file.getAbsolutePath() + "]: " + root.getMessage());
			setStatusText("Unable to read the image (see log).");
			if (showDialogOnFailure) {
				JOptionPane.showMessageDialog(this, root.getMessage(),
						"Open Image Failed", JOptionPane.ERROR_MESSAGE);
			}
		}));
		return true;
	}

	/** Decode an image without occupying the Swing event-dispatch thread. */
	CompletableFuture<BufferedImage> decodeImageAsync(File file) {
		return CompletableFuture.supplyAsync(() -> {
			try {
				BufferedImage image = ImageIO.read(file);
				if (image == null) {
					throw new IOException("The selected file is not a supported image.");
				}
				return image;
			} catch (IOException e) {
				throw new CompletionException(e);
			}
		});
	}

	private static void cancel(CompletableFuture<?> future) {
		if (future != null && !future.isDone()) {
			future.cancel(true);
		}
	}

	private void installImageMenu() {
		JMenuBar menuBar = getJMenuBar();
		if (menuBar == null) {
			menuBar = new JMenuBar();
			setJMenuBar(menuBar);
		}
		recentImages = new RecentFiles(Preferences.userNodeForPackage(getClass())
				.node("imageClassifier"), 10, "recentImage");
		recentImagesHelper = new RecentFilesMenu(recentImages,
				file -> openImageFile(file, true), "images");

		JMenu imageMenu = new JMenu("Image");
		JMenuItem openItem = new JMenuItem("Open Image…");
		openItem.addActionListener(event -> chooseImage());
		imageMenu.add(openItem);
		recentImagesMenu = new JMenu("Recent Images");
		recentImagesHelper.rebuild(recentImagesMenu);
		imageMenu.add(recentImagesMenu);
		BaseView.applyFocusFix(imageMenu, this);
		menuBar.add(imageMenu);
	}

	private void installResultsMenu() {
		JMenu resultsMenu = new JMenu("Results");
		JMenu topKMenu = new JMenu("Number of Classes");
		ButtonGroup group = new ButtonGroup();
		for (int count : TOP_K_CHOICES) {
			JRadioButtonMenuItem item = new JRadioButtonMenuItem(
					Integer.toString(count), topK == count);
			item.addActionListener(event -> setTopK(count));
			group.add(item);
			topKMenu.add(item);
		}
		resultsMenu.add(topKMenu);
		BaseView.applyFocusFix(resultsMenu, this);
		getJMenuBar().add(resultsMenu);
	}

	private static int validStoredTopK(int count) {
		return count >= 1 && count <= MAX_TOP_K ? count : DEFAULT_TOP_K;
	}

	// Draw the image centered and scaled to fit within the container.
	private void drawImage(Graphics2D g2, IContainer ctr, BufferedImage image) {
		if (image == null) {
			imageRect = null;
			return;
		}
		Objects.requireNonNull(ctr, "container");

        BaseContainer container = (BaseContainer) ctr;

        Rectangle bounds = container.getBounds();
	    int w = bounds.width;
	    int h = bounds.height;
	    int iw = image.getWidth();
	    int ih = image.getHeight();

		g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		double sx = (double) w / iw;
		double sy = (double) h / ih;
		double s = Math.min(sx, sy);

		int dw = (int) Math.round(iw * s);
		int dh = (int) Math.round(ih * s);

		int x = (w - dw) / 2;
		int y = (h - dh) / 2;

		g2.drawImage(image, x, y, dw, dh, null);
		imageRect = new Rectangle(x, y, dw, dh);

	}

	// Provide feedback strings showing screen and world coordinates
	// items will ad to the feedback when they are mouse-overed
	@Override
	public void getFeedbackStrings(IContainer container, Point pp, Double wp, List<String> feedbackStrings) {
		feedbackStrings.add(String.format("Screen Coordinates: (%d, %d)", pp.x, pp.y));

		if (imageRect != null && currentImage != null) {
			boolean inside = imageRect.contains(pp);
			String inImageStr = imageRect.contains(pp) ? "(inside image)" : "(outside image)";
			feedbackStrings.add(inImageStr);

			if (inside) {
				if (currentImagePath != null) {
					feedbackStrings.add("Source: " + currentImagePath.getFileName().toString());
				}
				int imgX = pp.x - imageRect.x;
				int imgY = pp.y - imageRect.y;
				String coordStrImg = String.format("Pixel: (%d, %d) %s", imgX, imgY, inImageStr);
				feedbackStrings.add(coordStrImg);

				Point imagePoint = imagePixelForDisplayPoint(imgX, imgY, imageRect.width, imageRect.height,
						currentImage.getWidth(), currentImage.getHeight());
				int clr = currentImage.getRGB(imagePoint.x, imagePoint.y);
				Color color = new Color(clr, true);
				feedbackStrings
						.add("Red: " + color.getRed() + " Green: " + color.getGreen() + " Blue: " + color.getBlue());


				if (currentResults != null && !currentResults.isEmpty()) {
					int maxClassesToShow = Math.min(5, currentResults.size());
					feedbackStrings.add(" "); // empty line
					feedbackStrings.add("$yellow$Top " + maxClassesToShow + " Classifications:");
					for (int i = 0; i < maxClassesToShow; i++) {
						ClassScore cs = currentResults.get(i);
						feedbackStrings.add(String.format("$yellow$  %s: %.4f%%", cs.label(), cs.score() * 100));
					}

					// top max of 3 cumulative probability

					double cumulativeProb = 0.0;
					int count = 0;
					feedbackStrings.add(" "); // empty line
					feedbackStrings.add("$orange$Cumulative Probability:");
					for (ClassScore cs : currentResults) {
						cumulativeProb += cs.score();
						count++;
						feedbackStrings.add(String.format("$orange$  Top-%d %.4f%%", count,
								cumulativeProb * 100));
						if (count >= 3) {
							break;
						}

					}

					feedbackStrings.add(" "); // empty line
					List<String> metaData = classifier.getModelMetaData();
					if (metaData != null && !metaData.isEmpty()) {
						feedbackStrings.add("$light green$Model Metadata:");
						for (String line : metaData) {
							feedbackStrings.add("$light green$  " + line);
						}
					}

					feedbackStrings.add(" "); // empty line
					List<String> inferenceOutput = classifier.getInferenceOutput();
					for (String line : inferenceOutput) {
						feedbackStrings.add("$white$" + line);
					}
				}
			}
		}

	}

	static Point imagePixelForDisplayPoint(int displayX, int displayY, int displayWidth, int displayHeight,
			int imageWidth, int imageHeight) {
		if (displayWidth <= 0 || displayHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) {
			throw new IllegalArgumentException("image and display dimensions must be positive");
		}
		int x = (int) Math.min(imageWidth - 1L,
				(long) Math.max(0, displayX) * imageWidth / displayWidth);
		int y = (int) Math.min(imageHeight - 1L,
				(long) Math.max(0, displayY) * imageHeight / displayHeight);
		return new Point(x, y);
	}

}
