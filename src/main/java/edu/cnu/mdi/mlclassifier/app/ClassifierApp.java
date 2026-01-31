package edu.cnu.mdi.mlclassifier.app;

import java.awt.EventQueue;

import edu.cnu.mdi.app.BaseMDIApplication;
import edu.cnu.mdi.desktop.Desktop;
import edu.cnu.mdi.log.Log;
import edu.cnu.mdi.mlclassifier.view.ImageClassifierView;
import edu.cnu.mdi.properties.PropertyUtils;
import edu.cnu.mdi.splot.plot.PlotView;
import edu.cnu.mdi.util.Environment;
import edu.cnu.mdi.view.LogView;
import edu.cnu.mdi.view.ViewManager;
import edu.cnu.mdi.view.VirtualView;

@SuppressWarnings("serial")
public class ClassifierApp extends BaseMDIApplication {
	

	/** Virtual desktop view (optional). */
	private VirtualView virtualView;

	/** If true, install the VirtualView and place views into columns. */
	private final boolean enableVirtualDesktop = true;

	/** Number of "columns"/cells in the virtual desktop. */
	private final int virtualDesktopCols = 5;

	// Initial views
	private LogView logView;
	private ImageClassifierView imageClassifierView;
	private PlotView plotView;

	/**
	 * Constructor.
	 * @param keyVals key-value pairs for configuring the base MDI application
	 */
	public ClassifierApp(Object... keyVals) {
		super(keyVals);

		// Enable the framework-managed virtual desktop lifecycle (one-shot ready +
		// debounced relayout).
		prepareForVirtualDesktop();

		// Log environment information early.
		Log.getInstance().info(Environment.getInstance().toString());

		// Create internal views. (Do not depend on the outer frame being visible here.)
		addInitialViews();

		// Optionally create the virtual desktop overview.
		// Note: VirtualView now resolves its parent frame lazily in addNotify().
		if (enableVirtualDesktop) {
			virtualView = VirtualView.createVirtualView(virtualDesktopCols);
			virtualView.toFront();
		}
	}

	/**
	 * Create and register the initial set of views shown in the demo.
	 * <p>
	 * This method only builds views; it should not depend on the outer frame being
	 * shown or on final geometry.
	 */
	private void addInitialViews() {
		// Log view is useful but not always visible.
		logView = new LogView();
		logView.setVisible(false);
		ViewManager.getInstance().getViewMenu().addSeparator();

		imageClassifierView = new ImageClassifierView(); // your custom view
		plotView = new PlotView(PropertyUtils.TITLE, "sPlot", PropertyUtils.FRACTION, 0.7, PropertyUtils.ASPECT, 1.2,
				PropertyUtils.VISIBLE, true);

	}

	@Override
	protected String getApplicationId() {
		return "MDI-Classifier";
	}

	@Override	
	protected void onVirtualDesktopReady() {
	    standardVirtualDesktopReady(virtualView, this::restoreDefaultViewLocations, true);

	    Log.getInstance().info("Classifier is ready.");}

	/**
	 * Runs after the outer frame is resized or moved (debounced).
	 * <p>
	 * Keep this lightweight. Reconfiguring the virtual desktop updates its world
	 * sizing and refreshes the thumbnail items.
	 */
	@Override
	protected void onVirtualDesktopRelayout() {
		   standardVirtualDesktopRelayout(virtualView);
    }

	/**
	 * Places the demo views into a reasonable "default" arrangement on the virtual
	 * desktop.
	 * <p>
	 * If a user has a saved configuration, {@link Desktop#configureViews()} will
	 * typically override these positions.
	 */
	private void restoreDefaultViewLocations() {
		// Column 0: map centered; drawing upper-left
		virtualView.moveTo(imageClassifierView, 0, VirtualView.CENTER);

		// Column 1: plot view centered
		virtualView.moveTo(plotView, 1, VirtualView.TOPCENTER);

		// column 4: log view upper left (is not vis by default)
		virtualView.moveTo(logView, 3, VirtualView.CENTER);
	}

	/** Main entry point. */
	public static void main(String[] args) {
		EventQueue.invokeLater(() -> {
			ClassifierApp app= new ClassifierApp(
					PropertyUtils.TITLE, "MDI Machine Learning Classifier",
					PropertyUtils.FRACTION, 0.8);
			app.setVisible(true);
		});
	}
}
