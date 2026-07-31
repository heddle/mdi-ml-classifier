package edu.cnu.mdi.mlclassifier.view;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.awt.Point;

import org.junit.jupiter.api.Test;

class ImageClassifierViewTest {

	@Test
	void mapsEnlargedDisplayEdgesInsideSourceImage() {
		assertEquals(new Point(0, 0),
				ImageClassifierView.imagePixelForDisplayPoint(0, 0, 100, 100, 10, 10));
		assertEquals(new Point(9, 9),
				ImageClassifierView.imagePixelForDisplayPoint(99, 99, 100, 100, 10, 10));
	}

	@Test
	void rejectsInvalidDimensions() {
		assertThrows(IllegalArgumentException.class,
				() -> ImageClassifierView.imagePixelForDisplayPoint(0, 0, 0, 10, 10, 10));
	}
}
