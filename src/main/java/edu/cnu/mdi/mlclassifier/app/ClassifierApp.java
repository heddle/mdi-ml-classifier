package edu.cnu.mdi.mlclassifier.app;

import java.awt.EventQueue;

import edu.cnu.mdi.app.BaseMDIApplication;
import edu.cnu.mdi.mlclassifier.view.ImageClassifierView;
import edu.cnu.mdi.properties.PropertyUtils;
import edu.cnu.mdi.splot.plot.PlotView;
import edu.cnu.mdi.view.LogView;

@SuppressWarnings("serial")
public class ClassifierApp extends BaseMDIApplication {

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

	    ImageClassifierView imageView = new ImageClassifierView();

	    PlotView plotView = new PlotView(
	        PropertyUtils.TITLE, "sPlot",
	        PropertyUtils.FRACTION, 0.7,
	        PropertyUtils.ASPECT, 1.2,
	        PropertyUtils.VISIBLE, true
	    );

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
