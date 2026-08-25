package edu.cnu.mdi.mlclassifier.view;

import java.util.List;

import edu.cnu.mdi.view.AbstractViewInfo;

/** Structured information displayed for the classifier view. */
public class ImageClassifierViewInfo extends AbstractViewInfo {

    @Override
    public String getTitle() {
        return "Image Classifier";
    }

    @Override
    public String getPurpose() {
        return "Demonstrates an end-to-end desktop machine-learning workflow: image loading, "
                + "model-specific preprocessing, asynchronous ONNX inference, ranked results, "
                + "diagnostic plotting, and interactive MDI feedback.";
    }

    @Override
    public List<String> getUsageBullets() {
        return List.of(
                "Drag an image onto the view or choose Image > Open Image….",
                "Reopen previously classified files from Image > Recent Images.",
                "Choose Model > Open ONNX Model… to replace the classifier; recently used models are remembered.",
                "Use Model > Open Labels… (or class IDs without labels) and Model > Normalization to match the selected model's training configuration.",
                "Choose Results > Number of Classes to change top-K output; the displayed image is reclassified automatically.",
                "Use Results > Comparison History to compare models, or copy and export completed classifications.",
                "Move the pointer over the displayed image to inspect source pixels, model metadata, and inference diagnostics.",
                "Review the Classification Results view for the top predicted classes.",
                "A newer image supersedes any older classification request still waiting to update the interface.");
    }

    @Override
    public String getTechnicalNotes() {
        return "The source is center-cropped to the model aspect ratio, resized, normalized, and arranged as NCHW or NHWC according to ONNX metadata. Inference runs on a dedicated worker so the Swing event-dispatch thread remains responsive. Outputs already forming a probability distribution are preserved; other finite outputs are converted with numerically stable softmax. Predicted probability expresses model confidence; it does not guarantee correctness or calibrated real-world accuracy.";
    }
}
