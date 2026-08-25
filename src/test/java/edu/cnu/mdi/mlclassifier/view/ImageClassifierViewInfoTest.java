package edu.cnu.mdi.mlclassifier.view;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ImageClassifierViewInfoTest {

    @Test
    void documentsBothImageOpeningWorkflows() {
        String html = new ImageClassifierViewInfo().getAsHTML();
        assertTrue(html.contains("Open Image"));
        assertTrue(html.contains("Recent Images"));
        assertTrue(html.contains("ONNX"));
    }
}
