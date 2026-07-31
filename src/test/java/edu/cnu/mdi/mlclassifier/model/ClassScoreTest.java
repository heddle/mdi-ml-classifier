package edu.cnu.mdi.mlclassifier.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ClassScoreTest {

	@Test
	void acceptsProbabilitiesAtBothEndpoints() {
		assertEquals(0.0, new ClassScore("zero", 0.0).score());
		assertEquals(1.0, new ClassScore("one", 1.0).score());
	}

	@Test
	void rejectsInvalidResults() {
		assertThrows(NullPointerException.class, () -> new ClassScore(null, 0.5));
		assertThrows(IllegalArgumentException.class, () -> new ClassScore("low", -0.1));
		assertThrows(IllegalArgumentException.class, () -> new ClassScore("high", 1.1));
		assertThrows(IllegalArgumentException.class, () -> new ClassScore("nan", Double.NaN));
	}
}
