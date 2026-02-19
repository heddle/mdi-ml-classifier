package edu.cnu.mdi.mlclassifier.view;

import java.util.List;

import edu.cnu.mdi.log.Log;
import edu.cnu.mdi.mlclassifier.model.ClassScore;
import edu.cnu.mdi.splot.pdata.PlotDataException;
import edu.cnu.mdi.splot.plot.BarPlot;
import edu.cnu.mdi.splot.plot.PlotPanel;

public class PlotSupport {

	/**
	 * Create a bar plot from classification results.
	 *
	 * @param results the list of ClassScore objects
	 * @return a PlotPanel containing the bar plot, or null if results are empty
	 */
	public static PlotPanel createBarPlot(List<ClassScore> results) {
		int n = results.size();
		if (n == 0) {
			return null;
		}

		double[] values = new double[n];
		String[] categories = new String[n];

		for (int i = 0; i < n; i++) {
			ClassScore cs = results.get(i);
			values[i] = cs.score();
			categories[i] = cs.label();
		}
		String title = "Classification Results";
		String xLabel = "Classes";
		String yLabel = "Scores";
		PlotPanel plotPanel = null;
		try {
			plotPanel = BarPlot.createBarPlot(title, values, categories, xLabel, yLabel);
		} catch (PlotDataException e) {
			Log.getInstance().warning("Error creating plot: " + e.getMessage());
		}
		return plotPanel;
	}
}
