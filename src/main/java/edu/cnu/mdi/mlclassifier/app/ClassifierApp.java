package edu.cnu.mdi.mlclassifier.app;

import java.awt.EventQueue;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import ai.onnxruntime.OrtException;
import edu.cnu.mdi.app.BaseMDIApplication;
import edu.cnu.mdi.log.Log;
import edu.cnu.mdi.mlclassifier.model.ClassScore;
import edu.cnu.mdi.mlclassifier.onnx.OnnxImageClassifier;
import edu.cnu.mdi.mlclassifier.view.ImageClassifierView;
import edu.cnu.mdi.mlclassifier.view.PlotSupport;
import edu.cnu.mdi.splot.plot.PlotPanel;
import edu.cnu.mdi.splot.plot.PlotView;
import edu.cnu.mdi.util.PropertyUtils;
import edu.cnu.mdi.view.LogView;
import edu.cnu.mdi.view.ViewManager;

/** Desktop application for classifying images with an ONNX model. */
@SuppressWarnings("serial")
public class ClassifierApp extends BaseMDIApplication {

	private PlotView plotView;
	private OnnxImageClassifier classifier;
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
		LogView logView = new LogView();
		ViewManager.getInstance().getViewMenu().addSeparator();
		logView.setVisible(false);

		plotView = new PlotView(PropertyUtils.TITLE, "Classification Results", PropertyUtils.FRACTION, 0.7,
				PropertyUtils.ASPECT, 1.2, PropertyUtils.VISIBLE, true);

		Path wd = Path.of(System.getProperty("user.dir"));

		String mobileNetModel = "models/mobilenetv2-12.onnx";
		Path modelPath  = wd.resolve(mobileNetModel);
		Path labelsPath = wd.resolve("models/imagenet_labels.txt");

		try {
		    classifier = new OnnxImageClassifier(modelPath, labelsPath);
		    ImageClassifierView imageView = new ImageClassifierView(classifier);
		    imageView.setResultConsumer(results -> makeBarPlot(results));
		} catch (OrtException | IOException e) {
			Log.getInstance().warning("Unable to load the classifier model: " + e.getMessage());
			logView.setVisible(true);
		}


	}

    // Plot the classification results using the associated PlotView.
    private void makeBarPlot(List<ClassScore> results) {
    	PlotPanel plotPanel = PlotSupport.createBarPlot(results);
    	if (plotPanel != null) {
    		plotView.switchToPlotPanel(plotPanel);
    	}
    }

	/** Release the ONNX session and inference worker during application shutdown. */
	@Override
	protected void prepareForShutdown() {
		try {
			if (classifier != null) {
				classifier.close();
			}
		} catch (IOException e) {
			Log.getInstance().warning("Unable to close the classifier cleanly: " + e.getMessage());
		} finally {
			super.prepareForShutdown();
		}
	}


	@Override
	protected String getApplicationId() {
		return "MDI-Classifier";
	}

	/** Main entry point. */
	public static void main(String[] args) {
		EventQueue.invokeLater(() -> {
			ClassifierApp app = new ClassifierApp(PropertyUtils.TITLE, "MDI Machine Learning Classifier",
					PropertyUtils.FRACTION, 0.8);
			app.setVisible(true);
		});
	}
}
