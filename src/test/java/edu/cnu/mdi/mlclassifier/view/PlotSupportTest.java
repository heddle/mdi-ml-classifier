package edu.cnu.mdi.mlclassifier.view;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;

import org.junit.jupiter.api.Test;

class PlotSupportTest {

	@Test
	void handlesEmptyAndNullResultsExplicitly() {
		assertNull(PlotSupport.createBarPlot(List.of()));
		assertThrows(NullPointerException.class, () -> PlotSupport.createBarPlot(null));
	}
}
