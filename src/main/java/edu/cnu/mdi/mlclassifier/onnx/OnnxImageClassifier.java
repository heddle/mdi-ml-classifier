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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import edu.cnu.mdi.log.Log;
import edu.cnu.mdi.mlclassifier.model.ClassScore;

/**
 * Minimal ONNX image classifier wrapper.
 *
 * Assumptions (typical vision classifier):
 * - input: float tensor [1, 3, H, W] in RGB order, normalized
 * - output: logits/probs [1, numClasses] (or [numClasses])
 *
 * You can adapt input/output name discovery once you know the exact model.
 */
public final class OnnxImageClassifier implements Closeable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final String outputName; // optional (can be null)
    private final int inputW;
    private final int inputH;

    // ImageNet defaults (common). Adjust per-model.
    private final float[] mean = {0.485f, 0.456f, 0.406f};
    private final float[] std  = {0.229f, 0.224f, 0.225f};

    private final ExecutorService exec;

    private final List<String> labels; // optional; if null, label = "class_i"

    public OnnxImageClassifier(Path modelPath) throws OrtException {
        this(modelPath, (List<String>) null, 224, 224);
    }

    public OnnxImageClassifier(Path modelPath, Path labelsPath) throws OrtException, IOException {
        this(modelPath,
             (labelsPath != null && Files.exists(labelsPath)) ? readLabels(labelsPath) : null,
             224, 224);
    }

    public OnnxImageClassifier(Path modelPath, List<String> labels) throws OrtException {
        this(modelPath, labels, 224, 224);
    }

    public OnnxImageClassifier(Path modelPath, List<String> labels, int inputW, int inputH) throws OrtException {
Log.getInstance().info("Loading ONNX model from: " + modelPath);

    	this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath.toString(), new OrtSession.SessionOptions());
        session.getInputInfo().forEach((k, v) ->
        Log.getInstance().info("ONNX input: " + k + " -> " + v.getInfo())
    );
    session.getOutputInfo().forEach((k, v) ->
        Log.getInstance().info("ONNX output: " + k + " -> " + v.getInfo())
    );

        // Most image models have exactly one input; keep it simple for v1.
        this.inputName = session.getInputNames().iterator().next();

        // If there is exactly one output, grab it; else you can pin by name later.
        String out = null;
        if (session.getOutputNames().size() == 1) {
            out = session.getOutputNames().iterator().next();
        }
        this.outputName = out;

        this.inputW = inputW;
        this.inputH = inputH;

        this.labels = (labels == null || labels.isEmpty()) ? null : List.copyOf(labels);

        this.exec = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "OnnxInferenceWorker");
            t.setDaemon(true);
            return t;
        });

        Log.getInstance().info("ONNX model loaded. input=" + inputName + " output=" + outputName
                + " size=" + inputW + "x" + inputH);
    }

    /** Run classification off-EDT. */
    public CompletableFuture<List<ClassScore>> classifyAsync(BufferedImage image, int topK) {
        Objects.requireNonNull(image, "image");
        int k = Math.max(1, topK);

        return CompletableFuture.supplyAsync(() -> {
            try {
                return classify(image, k);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }, exec);
    }

    /** Synchronous classify (runs on caller thread). */
    public List<ClassScore> classify(BufferedImage image, int topK) throws OrtException {
        float[] chw = preprocessToCHW(image, inputW, inputH);

        long[] shape = new long[] {1, 3, inputH, inputW};

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), shape)) {

            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, inputTensor);

            try (OrtSession.Result results = session.run(inputs)) {

                Object outObj;
                if (outputName != null) {
                    outObj = results.get(outputName).get().getValue();
                } else {
                    // fallback to first output
                    outObj = results.iterator().next().getValue().getValue();
                }

                float[] logits = flattenToFloatArray(outObj);
                float[] probs = softmax(logits);
                Log.getInstance().info("logits len=" + logits.length
                	    + " min=" + min(logits) + " max=" + max(logits));

                	Log.getInstance().info("probs sum=" + sum(probs)
                	    + " max=" + max(probs));


                return topK(probs, topK);
            }
        }
    }
    
    private float min(float[] arr) {
		float min = Float.MAX_VALUE;
		for (float v : arr) {
			if (v < min) {
				min = v;
			}
		}
		return min;
	}

	private float max(float[] arr) {
		float max = Float.MIN_VALUE;
		for (float v : arr) {
			if (v > max) {
				max = v;
			}
		}
		return max;
	}

	private float sum(float[] arr) {
		float sum = 0.0f;
		for (float v : arr) {
			sum += v;
		}
		return sum;
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
     * Convert BufferedImage -> RGB float CHW with normalization.
     * (Bilinear resize; no center-crop. Adjust if your model expects crop.)
     */
    private float[] preprocessToCHW(BufferedImage src, int w, int h) {
        BufferedImage resized = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(src, 0, 0, w, h, null);
        } finally {
            g.dispose();
        }

        float[] out = new float[3 * w * h];

        int idxR = 0;
        int idxG = w * h;
        int idxB = 2 * w * h;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = resized.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g2 = (rgb >> 8) & 0xFF;
                int b = (rgb) & 0xFF;

                // scale to 0..1
                float rf = (r / 255.0f - mean[0]) / std[0];
                float gf = (g2 / 255.0f - mean[1]) / std[1];
                float bf = (b / 255.0f - mean[2]) / std[2];

                int p = y * w + x;
                out[idxR + p] = rf;
                out[idxG + p] = gf;
                out[idxB + p] = bf;
            }
        }
        return out;
    }

    /**
     * Handles common ORT output shapes:
     * - float[]                         (already flat)
     * - float[][] with [1][N]
     */
    private static float[] flattenToFloatArray(Object outObj) {
        if (outObj instanceof float[] fa) {
            return fa;
        }
        if (outObj instanceof float[][] f2) {
            if (f2.length == 1) return f2[0];
        }
        if (outObj instanceof float[][][] f3) {
            // rare, but handle [1][1][N]
            if (f3.length == 1 && f3[0].length == 1) return f3[0][0];
        }
        throw new IllegalArgumentException("Unsupported ONNX output type: " + outObj.getClass());
    }

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
            p[i] = (float)(exps[i] / sum);
        }
        return p;
    }

    @Override
    public void close() throws IOException {
        exec.shutdownNow();
        try {
            session.close();
            env.close();
        } catch (OrtException e) {
            throw new IOException(e);
        }
    }

    // Helper for loading labels (optional)
    public static List<String> readLabels(Path labelsTxt) throws IOException {
        return Files.readAllLines(labelsTxt).stream()
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .toList();
    }
}
