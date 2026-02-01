package edu.cnu.mdi.mlclassifier.view;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletionException;
import java.util.function.Consumer;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

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

	private final ImagePanel imagePanel = new ImagePanel();
	private final JLabel statusLabel = new JLabel("Drop an image here (or use File → Open)", SwingConstants.CENTER);

	private OnnxImageClassifier onnx; // model runner
	private boolean onnxReady = false;

	// Optional: remember source for later (model metadata, etc.)
	private Path currentImagePath;

	private Consumer<List<ClassScore>> classificationResultConsumer;

	public ImageClassifierView(OnnxImageClassifier classifier, Object... keyVals) {
		super(PropertyUtils.TITLE, "Image Classifier", PropertyUtils.FRACTION, 0.7, PropertyUtils.ASPECT, 1.2,
				PropertyUtils.VISIBLE, true);

		onnx = classifier;
		onnxReady = true;
		setStatusText("Model loaded. Drop an image to classify.");
		// only allow valid image files to be dropped
		setFileFilter(ImageFilters.isActualImage);
		buildUI();

		// Set up drag and drop handling

		imagePanel.setTransferHandler(new FileDropHandler(this));

	}

	/**
	 * Set the consumer that will handle classification results.
	 *
	 * @param consumer a Consumer that processes a list of ClassScore objects
	 */
	public void setResultConsumer(Consumer<List<ClassScore>> consumer) {
		classificationResultConsumer = consumer;
	}

	// Build the user interface.
	private void buildUI() {
		// If BaseView is a JInternalFrame-like container:
		setLayout(new BorderLayout());

		statusLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
		statusLabel.setFont(Fonts.defaultFont);

		JPanel center = new JPanel(new BorderLayout());
		center.add(imagePanel, BorderLayout.CENTER);
		center.add(statusLabel, BorderLayout.SOUTH);

		add(center, BorderLayout.CENTER);

		// Give the view a sane minimum
		setMinimumSize(new Dimension(400, 300));
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
		imagePanel.setImage(img);

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
					if (classificationResultConsumer != null) {
						classificationResultConsumer.accept(results);
					}
				});
			});
		} else {
			runFakeInferenceAsync(img);
		}
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
			return;
		}
		File file = files.get(0);
		try {
			BufferedImage img = ImageIO.read(file);
			if (img == null) {
				Log.getInstance().warning("The dropped file is not a valid image: " + file.getAbsolutePath());
				return;
			}
			setImage(img, file.toPath());
			Log.getInstance().info("Loaded image file: " + file.getAbsolutePath());
		} catch (IOException e) {
			Log.getInstance().warning("Error reading image file [" + file.getAbsolutePath() + "]: " + e.getMessage());
		}
	}

	// ----------------------------------------------------------------
	// Inner panel that paints the image scaled-to-fit.
	// ----------------------------------------------------------------
	private static final class ImagePanel extends JPanel {
		private BufferedImage image;

		ImagePanel() {
			setOpaque(true);
		}

		void setImage(BufferedImage img) {
			this.image = img;
			repaint();
		}

		@Override
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);

			Graphics2D g2 = (Graphics2D) g.create();
			try {
				g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
				g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
				g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

				if (image == null) {
					// Optional: draw a subtle placeholder
					return;
				}

				int w = getWidth();
				int h = getHeight();

				int iw = image.getWidth();
				int ih = image.getHeight();

				double sx = (double) w / iw;
				double sy = (double) h / ih;
				double s = Math.min(sx, sy);

				int dw = (int) Math.round(iw * s);
				int dh = (int) Math.round(ih * s);

				int x = (w - dw) / 2;
				int y = (h - dh) / 2;

				g2.drawImage(image, x, y, dw, dh, null);
			} finally {
				g2.dispose();
			}
		}
	}
}
