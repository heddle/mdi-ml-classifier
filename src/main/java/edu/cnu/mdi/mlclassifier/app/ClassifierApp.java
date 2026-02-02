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
import edu.cnu.mdi.properties.PropertyUtils;
import edu.cnu.mdi.splot.pdata.PlotDataException;
import edu.cnu.mdi.splot.plot.BarPlot;
import edu.cnu.mdi.splot.plot.PlotPanel;
import edu.cnu.mdi.splot.plot.PlotView;
import edu.cnu.mdi.view.LogView;

@SuppressWarnings("serial")
public class ClassifierApp extends BaseMDIApplication {

	private PlotView plotView;
	/**
	 * Constructor.
	 *
	 * @param keyVals key-value pairs for configuring the base MDI application
	 */
	public ClassifierApp(Object... keyVals) {
		super(keyVals);

		// Create internal views. (Do not depend on the outer frame being visible here.)
		addInitialViews();
	}

	/**
	 * Create and register the initial set of views shown in the demo.
	 * <p>
	 * This method only builds views; it should not depend on the outer frame being
	 * shown or on final geometry.
	 */
	private void addInitialViews() {
		LogView logView = new LogView();
		logView.setVisible(false);

		plotView = new PlotView(PropertyUtils.TITLE, "Classification Results", PropertyUtils.FRACTION, 0.7,
				PropertyUtils.ASPECT, 1.2, PropertyUtils.VISIBLE, true);

		String homeDir = System.getProperty("user.home");

		Path homePath = Path.of(homeDir);

		// Build the path relative to home
		Path modelPath = homePath.resolve("mdi-ml-classifier/models/resnet50-v2-7.onnx");
		Path labelsPath = homePath.resolve("mdi-ml-classifier/models/imagenet_labels.txt");
		
		
//		Path modelPath = Path.of("/Users/davidheddle/mdi-ml-classifier/models/resnet50-v2-7.onnx");
//		Path labelsPath = Path.of("/Users/davidheddle/mdi-ml-classifier/models/imagenet_labels.txt");

		try {
			OnnxImageClassifier classifier = new OnnxImageClassifier(modelPath, labelsPath);
			ImageClassifierView imageView = new ImageClassifierView(classifier);
			imageView.setResultConsumer(results -> makeBarPlot(results));
		} catch (OrtException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}


	}

    // Plot the classification results using the associated PlotView.
    private void makeBarPlot(List<ClassScore> results) {
    	PlotPanel plotPanel = PlotSupport.createBarPlot(results);
    	if (plotPanel != null) {
    		plotView.setPlotPanel(plotPanel);
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
