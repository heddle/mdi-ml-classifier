package edu.cnu.mdi.mlclassifier.view;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.geom.Point2D.Double;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletionException;
import java.util.function.Consumer;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.SwingConstants;

import edu.cnu.mdi.container.BaseContainer;
import edu.cnu.mdi.container.IContainer;
import edu.cnu.mdi.feedback.FeedbackPane;
import edu.cnu.mdi.graphics.drawable.IDrawable;
import edu.cnu.mdi.log.Log;
import edu.cnu.mdi.mlclassifier.model.ClassScore;
import edu.cnu.mdi.mlclassifier.onnx.OnnxImageClassifier;
import edu.cnu.mdi.properties.PropertyUtils;
import edu.cnu.mdi.transfer.FileDropHandler;
import edu.cnu.mdi.transfer.ImageFilters;
import edu.cnu.mdi.ui.fonts.Fonts;
import edu.cnu.mdi.view.BaseView;

@SuppressWarnings("serial")
public class ImageClassifierView extends BaseView {

	// default side panel width (feedback)
	private static final int SIDE_PANEL_WIDTH = 230;

	// status label
	private final JLabel statusLabel = new JLabel("Drop an image here (or use File → Open)", SwingConstants.CENTER);

	private OnnxImageClassifier onnx; // model runner
	private boolean onnxReady = false;

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

	public ImageClassifierView(OnnxImageClassifier classifier, Object... keyVals) {
		super(PropertyUtils.TITLE, "Image Classifier", PropertyUtils.FRACTION, 0.7, PropertyUtils.ASPECT, 1.2,
				PropertyUtils.VISIBLE, true);

		onnx = classifier;
		onnxReady = true;
		setFileFilter(ImageFilters.isActualImage);
		addStatusLabel();
		addFeedback();

		// Set up drag and drop handling

		JComponent jc = (JComponent) getContainer().getComponent();
		jc.setTransferHandler(new FileDropHandler(this));

		getContainer().getFeedbackControl().addFeedbackProvider(this);

		IDrawable imageDrawer = new IDrawable() {

			@Override
			public void draw(Graphics2D g2, IContainer container) {
				drawImage(g2, container, currentImage);
			}
		};

		getContainer().setBeforeDraw(imageDrawer);

	}

	// Add the status label below the image panel.
	private void addStatusLabel() {
		statusLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
		statusLabel.setFont(Fonts.defaultFont);
		statusLabel.setText("Model loaded. Drop an image above to classify.");
		statusLabel.setOpaque(true);
		statusLabel.setBackground(Color.lightGray);
		statusLabel.setForeground(Color.black);
		statusLabel.setBorder(BorderFactory.createLineBorder(Color.darkGray));
		add(statusLabel, BorderLayout.SOUTH);
	}

	// Add the feedback pane to the east side.
	private void addFeedback() {
		FeedbackPane fbp = initFeedback();
		Dimension feedbackPref = fbp.getPreferredSize();
		fbp.setPreferredSize(new Dimension(SIDE_PANEL_WIDTH, feedbackPref.height));
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


	/**
	 * Set the image to display in this view and to be classified
	 *
	 * @param img        the image to display and classify
	 * @param sourcePath the optional source path of the image (may be null)
	 */
	public void setImage(BufferedImage img, Path sourcePath) {
		Objects.requireNonNull(img, "img");
		this.currentImagePath = sourcePath;
		currentImage = img;

		if (sourcePath != null) {
			setStatusText("Loaded image from " + sourcePath.getFileName().toString());
		} else {
			setStatusText("Image loaded");
		}

		setStatusText("Classifying image...");

		if (onnxReady && onnx != null) {
			onnx.classifyAsync(img, 5).whenComplete((results, err) -> {
				javax.swing.SwingUtilities.invokeLater(() -> {
					if (err != null) {
						Throwable root = (err instanceof CompletionException && err.getCause() != null) ? err.getCause()
								: err;
						Log.getInstance().warning("ONNX inference failed: " + root.getMessage());
						setStatusText("Classification failed (see log).");
						return;
					}

					setStatusText("Classification complete.");
					currentResults = results;
					if (classificationResultConsumer != null) {
						classificationResultConsumer.accept(results);
					}
				});
			});
		} else {
			runFakeInferenceAsync(img);
		}
		getContainer().refresh();
	}

	// Simulate an asynchronous inference process with fake results.
	private void runFakeInferenceAsync(BufferedImage img) {

		Thread t = new Thread(() -> {
			try {
				Thread.sleep(150);
			} catch (InterruptedException ignored) {
			}

			var results = java.util.List.of(new edu.cnu.mdi.mlclassifier.model.ClassScore("cat", 0.62),
					new edu.cnu.mdi.mlclassifier.model.ClassScore("dog", 0.21),
					new edu.cnu.mdi.mlclassifier.model.ClassScore("car", 0.09),
					new edu.cnu.mdi.mlclassifier.model.ClassScore("airplane", 0.05),
					new edu.cnu.mdi.mlclassifier.model.ClassScore("pizza", 0.03));

			javax.swing.SwingUtilities.invokeLater(() -> {
				if (classificationResultConsumer != null) {
					classificationResultConsumer.accept(results);
				}
			});
		}, "InferenceWorker");

		t.setDaemon(true);
		t.start();
	}

	/**
	 * Update the status message displayed below the image.
	 *
	 * @param message the status message to display.
	 */
	public void setStatusText(String message) {
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
			currentImagePath = null;
			return;
		}
		File file = files.get(0);
		try {
			BufferedImage img = ImageIO.read(file);
			if (img == null) {
				Log.getInstance().warning("The dropped file is not a valid image: " + file.getAbsolutePath());
				return;
			}
			currentImagePath = file.toPath();
			setImage(img, file.toPath());
			Log.getInstance().info("Loaded image file: " + file.getAbsolutePath());
		} catch (IOException e) {
			currentImagePath = null;
			Log.getInstance().warning("Error reading image file [" + file.getAbsolutePath() + "]: " + e.getMessage());
		}
	}

	// Draw the image centered and scaled to fit within the container.
	private void drawImage(Graphics2D g2, IContainer ctr, BufferedImage image) {
		if (image == null) {
			imageRect = null;
			return;
		}
	    Objects.requireNonNull(container, "container");

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

				int clr = currentImage.getRGB(
						(int) Math.round((double) imgX * currentImage.getWidth() / imageRect.width),
						(int) Math.round((double) imgY * currentImage.getHeight() / imageRect.height));
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
					ArrayList<String> metaData = onnx.getModelMetaData();
					if (metaData != null && !metaData.isEmpty()) {
						feedbackStrings.add("$light green$Model Metadata:");
						for (String line : metaData) {
							feedbackStrings.add("$light green$  " + line);
						}
					}

					feedbackStrings.add(" "); // empty line
					ArrayList<String> inferenceOutput = onnx.getInferenceOutput();
					for (String line : inferenceOutput) {
						feedbackStrings.add("$white$" + line);
					}
				}
			}
		}

	}

}
