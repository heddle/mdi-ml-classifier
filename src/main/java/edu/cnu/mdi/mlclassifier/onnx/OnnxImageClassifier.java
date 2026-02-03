package edu.cnu.mdi.mlclassifier.onnx;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import edu.cnu.mdi.log.Log;
import edu.cnu.mdi.mlclassifier.model.ClassScore;

/**
 * Minimal ONNX image classifier wrapper using ONNX Runtime (CPU).
 * <p>
 * This class loads an ONNX model once, inspects the model's input tensor shape to infer:
 * <ul>
 *   <li>input tensor name</li>
 *   <li>image layout (NCHW vs NHWC)</li>
 *   <li>required image width/height</li>
 * </ul>
 *
 * <h2>Model expectations</h2>
 * <p>
 * This wrapper is designed for typical classification models with:
 * <ul>
 *   <li>one image input tensor of rank 4: [N,C,H,W] (NCHW) or [N,H,W,C] (NHWC)</li>
 *   <li>one output tensor containing logits or probabilities for classes:
 *       [N,numClasses] or [numClasses] (common: 1000 for ImageNet)</li>
 * </ul>
 *
 * <h2>Preprocessing</h2>
 * <p>
 * The input image is resized (bilinear) to the model input size and converted to float32.
 * By default, ImageNet normalization is applied:
 * <pre>
 * mean = [0.485, 0.456, 0.406]
 * std  = [0.229, 0.224, 0.225]
 * </pre>
 *
 * <h2>Threading</h2>
 * <p>
 * {@link #classifyAsync(BufferedImage, int)} runs inference off the EDT on a single dedicated
 * background thread. UI updates should be performed by the caller on the EDT.
 *
 * <h2>Resource management</h2>
 * <p>
 * The {@link OrtSession} is held for the lifetime of this classifier. Call {@link #close()}
 * when done (e.g., app shutdown) to release native resources.
 */
public final class OnnxImageClassifier implements Closeable {
	
	// Normalization types for different models
	public enum NormType { RESNET, SCALE_0_1, SCALE_NEG1_1 }


    /**
     * Simple descriptor of the model's image input.
     */
    public static final class ImageInputSpec {
        /** ONNX input tensor name. */
        public final String inputName;
        /** True if input layout is NCHW, false if NHWC. */
        public final boolean nchw;
        /** Model-required image width in pixels. */
        public final int width;
        /** Model-required image height in pixels. */
        public final int height;

        public ImageInputSpec(String inputName, boolean nchw, int width, int height) {
            this.inputName = inputName;
            this.nchw = nchw;
            this.width = width;
            this.height = height;
        }

        @Override
        public String toString() {
            return "ImageInputSpec{name=" + inputName + ", layout=" + (nchw ? "NCHW" : "NHWC")
                    + ", size=" + width + "x" + height + "}";
        }
    }

    // Keep a single environment for the process. OrtEnvironment is effectively a singleton.
    private static final OrtEnvironment ENV = OrtEnvironment.getEnvironment();

    private final OrtSession session;
    private final String inputName;
    private final String outputName; // may be null if multiple outputs (then we take first)
    private final boolean nchw;
    private final int inputW;
    private final int inputH;

    private final List<String> labels; // may be null
    private final ExecutorService exec;
    
    // Norm type for preprocessing default
    private NormType normType = NormType.RESNET; 


    // ImageNet defaults (common). Adjust if your model expects different preprocessing.
    private final float[] mean = {0.485f, 0.456f, 0.406f};
    private final float[] std  = {0.229f, 0.224f, 0.225f};
    
    //used for feedback
    private ArrayList<String> inferenceOutput = new ArrayList<String>();

    /**
     * Create a classifier with a model only (no labels). Class names will be "class_i".
     *
     * @param modelPath path to the ONNX model file
     * @throws OrtException if ONNX Runtime fails to create the session
     */
    public OnnxImageClassifier(Path modelPath) throws OrtException {
        this(modelPath, (List<String>) null);
    }

    /**
     * Create a classifier with a model and a labels file.
     *
     * @param modelPath path to the ONNX model file
     * @param labelsPath path to a text file containing one label per line (optional)
     * @throws OrtException if ONNX Runtime fails to create the session
     * @throws IOException if labelsPath is non-null and cannot be read
     */
    public OnnxImageClassifier(Path modelPath, Path labelsPath) throws OrtException, IOException {
        this(modelPath, (labelsPath != null && Files.exists(labelsPath)) ? readLabels(labelsPath) : null);
    }

    /**
     * Create a classifier with a model and in-memory labels.
     *
     * @param modelPath path to the ONNX model file
     * @param labels list of labels indexed by class id; if null or empty, labels are not used
     * @throws OrtException if ONNX Runtime fails to create the session
     */
    public OnnxImageClassifier(Path modelPath, List<String> labels) throws OrtException {
        Objects.requireNonNull(modelPath, "modelPath");
        
        //check the normalization type based on model name
        if (modelPath.toString().toLowerCase().contains("efficientnet")) {
            this.normType = NormType.SCALE_NEG1_1; // Common for ONNX-converted Lite models
        }

        // Create session
        this.session = ENV.createSession(modelPath.toString(), new OrtSession.SessionOptions());

        // Infer input spec from model
        ImageInputSpec spec = inferImageInputSpec(session);
        this.inputName = spec.inputName;
        this.nchw = spec.nchw;
        this.inputW = spec.width;
        this.inputH = spec.height;

        // Choose output
        String out = null;
        if (session.getOutputNames().size() == 1) {
            out = session.getOutputNames().iterator().next();
        }
        this.outputName = out;

        // Labels
        this.labels = (labels == null || labels.isEmpty()) ? null : List.copyOf(labels);

        // Executor for inference
        this.exec = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "OnnxInferenceWorker");
            t.setDaemon(true);
            return t;
        });

        // Helpful diagnostics
        session.getInputInfo().forEach((k, v) ->
                Log.getInstance().info("ONNX input: " + k + " -> " + v.getInfo()));
        session.getOutputInfo().forEach((k, v) ->
                Log.getInstance().info("ONNX output: " + k + " -> " + v.getInfo()));
        Log.getInstance().info("ONNX model loaded. input=" + inputName
                + " output=" + outputName
                + " layout=" + (nchw ? "NCHW" : "NHWC")
                + " size=" + inputW + "x" + inputH);
    }

    /**
     * Run classification asynchronously off the EDT.
     *
     * @param image input image
     * @param topK number of top classes to return (minimum 1)
     * @return future that completes with top-K {@link ClassScore} results
     */
    public CompletableFuture<List<ClassScore>> classifyAsync(BufferedImage image, int topK) {
        Objects.requireNonNull(image, "image");
        int k = Math.max(1, topK);
        return CompletableFuture.supplyAsync(() -> {
            try {
                return classify(image, k);
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        }, exec);
    }

    /**
     * Run classification synchronously on the caller thread.
     *
     * @param image input image
     * @param topK number of top classes to return (minimum 1)
     * @return top-K {@link ClassScore} results
     * @throws OrtException if ONNX Runtime inference fails
     */
    public List<ClassScore> classify(BufferedImage image, int topK) throws OrtException {
        Objects.requireNonNull(image, "image");
        inferenceOutput.clear();
        
        int k = Math.max(1, topK);

        float[] input = preprocess(image);

        // Build tensor shape with batch=1
        final long[] shape = nchw
                ? new long[]{1, 3, inputH, inputW}
                : new long[]{1, inputH, inputW, 3};

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(ENV, FloatBuffer.wrap(input), shape)) {

            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, inputTensor);

            long t0 = System.nanoTime();
            try (OrtSession.Result results = session.run(inputs)) {
                long dtMs = (System.nanoTime() - t0) / 1_000_000L;

                Object outObj;
                if (outputName != null) {
                    outObj = results.get(outputName).get().getValue();
                } else {
                    // Fallback: take first output.
                    outObj = results.iterator().next().getValue().getValue();
                }

                float[] logits = flattenToFloatArray(outObj);

                // Debug-friendly sanity checks (safe to keep, or gate behind a flag)
                float min = Float.POSITIVE_INFINITY;
                float max = Float.NEGATIVE_INFINITY;
                for (float v : logits) { 
                	min = Math.min(min, v); 
                	max = Math.max(max, v); 
                }
                
                inferenceOutput.add("ONNX inference: ");
                inferenceOutput.add("  time: " + dtMs + " ms");
                
                String logitsRange = String.format("  logits range: [%.4f, %.4f]", min, max);
                inferenceOutput.add(logitsRange);
                               
 
                float[] probs = softmax(logits);

                // Optional check: sum should be ~1.0
                float sum = 0f;
                float pMax = 0f;
				for (float p : probs) {
					sum += p;
					pMax = Math.max(pMax, p);
				}
				inferenceOutput.add("  probability sum: " + sum);
				inferenceOutput.add("  confidence (max): " + pMax);
				
				double ent = entropyBits(probs);
				inferenceOutput.add("  Uncertainty (entropy): " + String.format("%.4f bits", ent));
				
				for(String s : inferenceOutput) {
					Log.getInstance().info(s);
				}
	
                return topK(probs, k);
            }
        }
    }
    
    /**
     * Get inference output for feedback
     * @return inference output as ArrayList<String>
     */
    public ArrayList<String> getInferenceOutput(){
		return inferenceOutput;
	}

    /**
     * @return the inferred model input width (pixels)
     */
    public int getInputWidth() {
        return inputW;
    }

    /**
     * @return the inferred model input height (pixels)
     */
    public int getInputHeight() {
        return inputH;
    }

    /**
     * @return true if model expects NCHW layout, false for NHWC
     */
    public boolean isNchw() {
        return nchw;
    }
    
    /**
     * Compute Shannon entropy (base-2) of a probability distribution.
     *
     * @param probs array of probabilities (should sum to ~1)
     * @return entropy in bits
     */
    public static double entropyBits(float[] probs) {
        double h = 0.0;
        for (float p : probs) {
            if (p > 0f) {
                h -= p * (Math.log(p) / Math.log(2.0));
            }
        }
        return h;
    }


    /**
     * Release native resources. Safe to call once at shutdown.
     */
    @Override
    public void close() throws IOException {
        exec.shutdownNow();
        try {
            session.close();
        } catch (OrtException e) {
            throw new IOException(e);
        }
    }

    /**
     * Read labels from a text file (one label per line).
     *
     * @param labelsTxt labels text file
     * @return list of labels (trimmed, empty lines removed)
     * @throws IOException if reading fails
     */
    public static List<String> readLabels(Path labelsTxt) throws IOException {
        Objects.requireNonNull(labelsTxt, "labelsTxt");
        return Files.readAllLines(labelsTxt).stream()
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .toList();
    }

    /**
     * Infer image input tensor spec (name, layout, width, height) from the model.
     * <p>
     * This expects exactly one input tensor and rank-4 image shape:
     * NCHW: [N, C, H, W] or NHWC: [N, H, W, C]
     *
     * @param session ONNX session
     * @return inferred {@link ImageInputSpec}
     */
    public static ImageInputSpec inferImageInputSpec(OrtSession session) throws OrtException {
        Objects.requireNonNull(session, "session");

        Map<String, NodeInfo> inputs = session.getInputInfo();
        if (inputs.size() != 1) {
            throw new IllegalStateException("Expected exactly one model input, found " + inputs.size()
                    + ". Inputs=" + inputs.keySet());
        }

        Map.Entry<String, NodeInfo> e = inputs.entrySet().iterator().next();
        String name = e.getKey();
        NodeInfo ni = e.getValue();

        if (!(ni.getInfo() instanceof TensorInfo ti)) {
            throw new IllegalStateException("Model input is not a tensor: " + name + " -> " + ni.getInfo());
        }

        long[] shape = ti.getShape();
        if (shape.length != 4) {
            throw new IllegalStateException("Expected rank-4 image input, got shape="
                    + java.util.Arrays.toString(shape));
        }

        long d1 = shape[1];
        long d3 = shape[3];

        // Channel dimension is typically 1 or 3. If both could match, prefer NCHW.
        if (d1 == 1 || d1 == 3) {
            int h = safeDim(shape[2], "height");
            int w = safeDim(shape[3], "width");
            return new ImageInputSpec(name, true, w, h);
        }

        if (d3 == 1 || d3 == 3) {
            int h = safeDim(shape[1], "height");
            int w = safeDim(shape[2], "width");
            return new ImageInputSpec(name, false, w, h);
        }

        throw new IllegalStateException("Cannot infer image layout from input shape="
                + java.util.Arrays.toString(shape)
                + ". Expected channel dimension of 1 or 3.");
    }

    private static int safeDim(long dim, String label) {
        // -1 often used for batch only; H/W should be positive.
        if (dim <= 0 || dim > Integer.MAX_VALUE) {
            throw new IllegalStateException("Invalid " + label + " dimension: " + dim);
        }
        return (int) dim;
    }

    private float[] preprocess(BufferedImage src) {
        // 1. Resize image to model specs
        BufferedImage resized = new BufferedImage(inputW, inputH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(src, 0, 0, inputW, inputH, null);
        } finally {
            g.dispose();
        }

        int totalPixels = inputW * inputH;
        float[] out = new float[3 * totalPixels];

        if (nchw) {
            // [C, H, W] layout (Planar)
            int idxG = totalPixels;
            int idxB = 2 * totalPixels;

            for (int y = 0; y < inputH; y++) {
                for (int x = 0; x < inputW; x++) {
                    int rgb = resized.getRGB(x, y);
                    float rf = ((rgb >> 16) & 0xFF);
                    float gf = ((rgb >> 8) & 0xFF);
                    float bf = (rgb & 0xFF);

                    float[] normalized = applyNormalization(rf, gf, bf);

                    int p = y * inputW + x;
                    out[p] = normalized[0];        // R plane
                    out[idxG + p] = normalized[1]; // G plane
                    out[idxB + p] = normalized[2]; // B plane
                }
            }
        } else {
            // [H, W, C] layout (Interleaved)
            int i = 0;
            for (int y = 0; y < inputH; y++) {
                for (int x = 0; x < inputW; x++) {
                    int rgb = resized.getRGB(x, y);
                    float rf = ((rgb >> 16) & 0xFF);
                    float gf = ((rgb >> 8) & 0xFF);
                    float bf = (rgb & 0xFF);

                    float[] normalized = applyNormalization(rf, gf, bf);

                    out[i++] = normalized[0];
                    out[i++] = normalized[1];
                    out[i++] = normalized[2];
                }
            }
        }
        return out;
    }

    /**
     * Helper to centralize the math for different model requirements.
     */
    private float[] applyNormalization(float r, float g, float b) {
        switch (normType) {
            case RESNET:
                // Standard ImageNet Mean/Std subtraction
                return new float[] {
                    (r / 255.0f - mean[0]) / std[0],
                    (g / 255.0f - mean[1]) / std[1],
                    (b / 255.0f - mean[2]) / std[2]
                };
            case SCALE_NEG1_1:
                // Maps 0-255 to -1.0 to 1.0 (Common for TF exports)
                return new float[] {
                    (r - 127.5f) / 127.5f,
                    (g - 127.5f) / 127.5f,
                    (b - 127.5f) / 127.5f
                };
            case SCALE_0_1:
            default:
                // Simple 0.0 to 1.0 scaling (Common for ONNX Zoo Lite models)
                return new float[] { r / 255.0f, g / 255.0f, b / 255.0f };
        }
    }

    /**
     * Preprocess image into float array matching model layout.
     * <p>
     * Resizes to the inferred inputW/inputH, converts to RGB float in [0,1],
     * then applies ImageNet mean/std normalization.
     *
     * @param src source image
     * @return float array in CHW (NCHW) or HWC (NHWC) order (batch dimension not included)
     */
    private float[] Xpreprocess(BufferedImage src) {
        BufferedImage resized = new BufferedImage(inputW, inputH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(src, 0, 0, inputW, inputH, null);
        } finally {
            g.dispose();
        }

        if (nchw) {
            // [3, H, W] contiguous channels
            float[] out = new float[3 * inputW * inputH];
            int plane = inputW * inputH;

            int idxR = 0;
            int idxG = plane;
            int idxB = 2 * plane;

            for (int y = 0; y < inputH; y++) {
                for (int x = 0; x < inputW; x++) {
                    int rgb = resized.getRGB(x, y);
                    int r = (rgb >> 16) & 0xFF;
                    int gg = (rgb >> 8) & 0xFF;
                    int b = (rgb) & 0xFF;

                    float rf = (r / 255.0f - mean[0]) / std[0];
                    float gf = (gg / 255.0f - mean[1]) / std[1];
                    float bf = (b / 255.0f - mean[2]) / std[2];

                    int p = y * inputW + x;
                    out[idxR + p] = rf;
                    out[idxG + p] = gf;
                    out[idxB + p] = bf;
                }
            }
            return out;
        } else {
            // NHWC: [H, W, 3] interleaved per pixel
            float[] out = new float[inputW * inputH * 3];
            int i = 0;
            for (int y = 0; y < inputH; y++) {
                for (int x = 0; x < inputW; x++) {
                    int rgb = resized.getRGB(x, y);
                    int r = (rgb >> 16) & 0xFF;
                    int gg = (rgb >> 8) & 0xFF;
                    int b = (rgb) & 0xFF;

                    out[i++] = (r / 255.0f - mean[0]) / std[0];
                    out[i++] = (gg / 255.0f - mean[1]) / std[1];
                    out[i++] = (b / 255.0f - mean[2]) / std[2];
                }
            }
            return out;
        }
    }

    private List<ClassScore> topK(float[] probs, int k) {
        List<Map.Entry<Integer, Float>> idx = new ArrayList<>(probs.length);
        for (int i = 0; i < probs.length; i++) {
            idx.add(new AbstractMap.SimpleEntry<>(i, probs[i]));
        }
        idx.sort(Comparator.comparing(Map.Entry<Integer, Float>::getValue).reversed());

        int n = Math.min(k, idx.size());
        List<ClassScore> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int ci = idx.get(i).getKey();
            float p = idx.get(i).getValue();
            String name = (labels != null && ci < labels.size()) ? labels.get(ci) : ("class_" + ci);
            out.add(new ClassScore(name, p));
        }
        return out;
    }

    /**
     * Flatten common ONNX Runtime output shapes into a float[]:
     * <ul>
     *   <li>float[]</li>
     *   <li>float[][] where [1][N]</li>
     *   <li>float[][][] where [1][1][N] (rare)</li>
     * </ul>
     *
     * @param outObj raw output from ORT
     * @return flat float array
     */
    private static float[] flattenToFloatArray(Object outObj) {
        if (outObj instanceof float[] fa) {
            return fa;
        }
        if (outObj instanceof float[][] f2) {
            if (f2.length == 1) return f2[0];
        }
        if (outObj instanceof float[][][] f3) {
            if (f3.length == 1 && f3[0].length == 1) return f3[0][0];
        }
        throw new IllegalArgumentException("Unsupported ONNX output type: " + outObj.getClass());
    }

    /**
     * Numerically-stable softmax (float output).
     *
     * @param logits logits or unnormalized scores
     * @return probabilities that sum to ~1.0
     */
    private static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) max = Math.max(max, v);

        double sum = 0.0;
        double[] exps = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            double e = Math.exp(logits[i] - max);
            exps[i] = e;
            sum += e;
        }

        float[] p = new float[logits.length];
        if (sum == 0.0) return p;

        for (int i = 0; i < logits.length; i++) {
            p[i] = (float) (exps[i] / sum);
        }
        return p;
    }
}
