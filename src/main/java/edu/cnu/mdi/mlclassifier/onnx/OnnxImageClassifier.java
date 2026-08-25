package edu.cnu.mdi.mlclassifier.onnx;

import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import edu.cnu.mdi.log.Log;
import edu.cnu.mdi.mlclassifier.model.ClassScore;
import edu.cnu.mdi.mlclassifier.model.ImageClassifier;
import edu.cnu.mdi.mlclassifier.model.InferenceSummary;

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
 * The input image is center-cropped to the model aspect ratio, resized with
 * bilinear interpolation, and converted to float32.
 * By default, ImageNet normalization is applied; constructors accepting
 * {@link NormType} allow models with different preprocessing requirements:
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
public final class OnnxImageClassifier implements Closeable, ImageClassifier {

	/** Supported input-pixel normalization schemes. */
	public enum NormType {
		/** ImageNet channel mean and standard-deviation normalization. */
		RESNET,
		/** Scale each channel from {@code [0,255]} to {@code [0,1]}. */
		SCALE_0_1,
		/** Scale each channel from {@code [0,255]} to {@code [-1,1]}. */
		SCALE_NEG1_1
	}


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

		/**
		 * Create an immutable image-input description.
		 *
		 * @param inputName tensor input name
		 * @param nchw {@code true} for NCHW; {@code false} for NHWC
		 * @param width required image width
		 * @param height required image height
		 */
        public ImageInputSpec(String inputName, boolean nchw, int width, int height) {
			this.inputName = Objects.requireNonNull(inputName, "inputName");
			if (width <= 0 || height <= 0) {
				throw new IllegalArgumentException("image dimensions must be positive");
			}
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
    private final String outputName;
    private final boolean nchw;
    private final int inputW;
    private final int inputH;

    private final List<String> labels; // may be null
    private final ExecutorService exec;
	private final AtomicBoolean closed = new AtomicBoolean();
	private final AtomicBoolean labelCountWarningLogged = new AtomicBoolean();

    // Norm type for preprocessing default
    private final NormType normType;


    // ImageNet defaults (common). Adjust if your model expects different preprocessing.
    private final float[] mean = {0.485f, 0.456f, 0.406f};
    private final float[] std  = {0.229f, 0.224f, 0.225f};

    //used for feedback
	private volatile List<String> inferenceOutput = List.of();
	private volatile InferenceSummary inferenceSummary;
	private final List<String> modelMetaData;

    /**
     * Create a classifier with a model only (no labels). Class names will be "class_i".
     *
     * @param modelPath path to the ONNX model file
     * @throws OrtException if ONNX Runtime fails to create the session
     */
    public OnnxImageClassifier(Path modelPath) throws OrtException {
		this(modelPath, (List<String>) null, NormType.RESNET);
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
		this(modelPath, labelsPath, NormType.RESNET);
    }

	/**
	 * Create a classifier with labels loaded from a file and explicit pixel
	 * normalization.
	 *
	 * @param modelPath model file
	 * @param labelsPath optional labels file
	 * @param normType model-specific pixel normalization
	 * @throws OrtException if ONNX Runtime cannot load the model
	 * @throws IOException if the labels file cannot be read
	 */
	public OnnxImageClassifier(Path modelPath, Path labelsPath, NormType normType)
			throws OrtException, IOException {
		this(modelPath, labelsPath == null ? null : readLabels(labelsPath), normType);
	}

    /**
     * Create a classifier with a model and in-memory labels.
     *
     * @param modelPath path to the ONNX model file
     * @param labels list of labels indexed by class id; if null or empty, labels are not used
     * @throws OrtException if ONNX Runtime fails to create the session
     */
    public OnnxImageClassifier(Path modelPath, List<String> labels) throws OrtException {
		this(modelPath, labels, NormType.RESNET);
	}

	/**
	 * Create a classifier with in-memory labels and explicit pixel normalization.
	 *
	 * @param modelPath model file
	 * @param labels labels indexed by class id, or {@code null}
	 * @param normType model-specific pixel normalization
	 * @throws OrtException if ONNX Runtime cannot load the model
	 */
	public OnnxImageClassifier(Path modelPath, List<String> labels, NormType normType) throws OrtException {
        Objects.requireNonNull(modelPath, "modelPath");
		this.normType = Objects.requireNonNull(normType, "normType");
		this.labels = (labels == null || labels.isEmpty()) ? null : List.copyOf(labels);

        // Create session
		OrtSession createdSession = null;
		ImageInputSpec spec;
		String selectedOutput;
		try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
			createdSession = ENV.createSession(modelPath.toString(), options);
			spec = inferImageInputSpec(createdSession);
			int outputCount = createdSession.getOutputNames().size();
			if (outputCount != 1) {
				throw new IllegalStateException(
						"Expected exactly one model output, found " + outputCount);
			}
			selectedOutput = createdSession.getOutputNames().iterator().next();
		} catch (OrtException | RuntimeException e) {
			if (createdSession != null) {
				try {
					createdSession.close();
				} catch (OrtException closeFailure) {
					e.addSuppressed(closeFailure);
				}
			}
			throw e;
		}
		this.session = createdSession;
        this.inputName = spec.inputName;
		this.outputName = selectedOutput;
        this.nchw = spec.nchw;
        this.inputW = spec.width;
        this.inputH = spec.height;

        // Store model meta data for feedback
		List<String> metadata = new ArrayList<>();
		metadata.add("ONNX model: " + modelPath.getFileName());
		metadata.add("input name: " + inputName);
		metadata.add("normalization: " + normType);
		if (normType == NormType.RESNET) {
			metadata.add(String.format("mean = [%.3f, %.3f, %.3f]", mean[0], mean[1], mean[2]));
			metadata.add(String.format("std  = [%.3f, %.3f, %.3f]", std[0], std[1], std[2]));
		}
		metadata.add("input layout: " + (nchw ? "NCHW" : "NHWC"));
		metadata.add("input size: " + inputW + " x " + inputH);
		modelMetaData = List.copyOf(metadata);

        // Executor for inference
        this.exec = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "OnnxInferenceWorker");
            t.setDaemon(true);
            return t;
        });

        // Helpful diagnostics
        session.getInputInfo().forEach((k, v) -> {
            Log.getInstance().info("ONNX input key: " + k);
            Log.getInstance().info("ONNX input value: " + v.getInfo());
         });


        session.getOutputInfo().forEach((k, v) ->
                Log.getInstance().info("ONNX output: " + k + " -> " + v.getInfo()));
        Log.getInstance().info("ONNX model loaded. input=" + inputName
                + " output=" + outputName
                + " layout=" + (nchw ? "NCHW" : "NHWC")
                + " size=" + inputW + "x" + inputH);

		for (String s : modelMetaData) {
			Log.getInstance().info(s);
		}
    }

    /**
     * Run classification asynchronously off the EDT.
     *
     * @param image input image
     * @param topK number of top classes to return (minimum 1)
     * @return future that completes with top-K {@link ClassScore} results
     */
    @Override
    public CompletableFuture<List<ClassScore>> classifyAsync(BufferedImage image, int topK) {
        Objects.requireNonNull(image, "image");
		requireOpen();
        int k = Math.max(1, topK);
        return CompletableFuture.supplyAsync(() -> {
            try {
                return classify(image, k);
            } catch (OrtException e) {
				throw new java.util.concurrent.CompletionException(e);
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
		requireOpen();

        int k = Math.max(1, topK);

        float[] input = preprocess(image);

        // Build tensor shape with batch=1
        final long[] shape = nchw
                ? new long[]{1, 3, inputH, inputW}
                : new long[]{1, inputH, inputW, 3};

		synchronized (session) {
			requireOpen();
			try (OnnxTensor inputTensor = OnnxTensor.createTensor(ENV, FloatBuffer.wrap(input), shape)) {

            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, inputTensor);

            long t0 = System.nanoTime();
            try (OrtSession.Result results = session.run(inputs)) {
                long dtMs = (System.nanoTime() - t0) / 1_000_000L;

				Object outObj = results.get(outputName)
						.orElseThrow(() -> new IllegalStateException("Model result omitted output " + outputName))
						.getValue();

				float[] output = flattenToFloatArray(outObj);

                // Debug-friendly sanity checks (safe to keep, or gate behind a flag)
                float min = Float.POSITIVE_INFINITY;
                float max = Float.NEGATIVE_INFINITY;
				for (float v : output) {
					if (!Float.isFinite(v)) {
						throw new IllegalArgumentException("Model output contains a non-finite value");
					}
                	min = Math.min(min, v);
                	max = Math.max(max, v);
                }

				List<String> diagnostics = new ArrayList<>();
				diagnostics.add("ONNX Inference: ");
				diagnostics.add("  time: " + dtMs + " ms");

				String outputRange = String.format("  output range: [%.4f, %.4f]", min, max);
				diagnostics.add(outputRange);

				float[] probs = probabilitiesFromOutput(output);
				if (labels != null && labels.size() != probs.length
						&& labelCountWarningLogged.compareAndSet(false, true)) {
					Log.getInstance().warning("Label count " + labels.size()
							+ " does not match model output count " + probs.length);
				}

                // Optional check: sum should be ~1.0
                float sum = 0f;
                float pMax = 0f;
				for (float p : probs) {
					sum += p;
					pMax = Math.max(pMax, p);
				}
				diagnostics.add(String.format("  probability sum: %7f", sum));
				diagnostics.add(String.format("  confidence (top-1 probability): %4f", pMax));

				double ent = entropyBits(probs);
				diagnostics.add("  uncertainty (entropy): " + String.format("%.3f bits", ent));
				double maxEnt = Math.log(probs.length) / Math.log(2.0); // log2(N)
				double nEnt = (maxEnt == 0.0) ? 0.0 : (ent / maxEnt) * 100.0;
				diagnostics.add("  uncertainty (normalized): " + String.format("%.2f%%", nEnt));
				inferenceSummary = new InferenceSummary(dtMs, min, max, sum, pMax, ent, nEnt);
				inferenceOutput = List.copyOf(diagnostics);
				for (String s : diagnostics) {
					Log.getInstance().info(s);
				}

                return topK(probs, k);
            }
			}
		}
    }

    /**
     * Get inference output for feedback
     * @return immutable snapshot of the most recent inference diagnostics
     */
    @Override
    public List<String> getInferenceOutput(){
		return inferenceOutput;
	}

	@Override
	public java.util.Optional<InferenceSummary> getInferenceSummary() {
		return java.util.Optional.ofNullable(inferenceSummary);
	}

    /**
	 * Get model meta data for feedback
	 *
	 * @return immutable model metadata
	 */
	@Override
	public List<String> getModelMetaData() {
		return modelMetaData;
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
		Objects.requireNonNull(probs, "probs");
        double h = 0.0;
        for (float p : probs) {
			if (!Float.isFinite(p) || p < 0f || p > 1f) {
				throw new IllegalArgumentException("probabilities must be finite and in [0, 1]");
			}
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
		if (!closed.compareAndSet(false, true)) {
			return;
		}
        exec.shutdownNow();
		synchronized (session) {
			try {
				session.close();
			} catch (OrtException e) {
				throw new IOException(e);
			}
		}
    }

	private void requireOpen() {
		if (closed.get()) {
			throw new IllegalStateException("classifier is closed");
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
	 * @throws OrtException if ONNX Runtime cannot inspect the model input
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
		if (ti.type != OnnxJavaType.FLOAT) {
			throw new IllegalStateException("Expected a float32 image input, got " + ti.type);
		}

        long[] shape = ti.getShape();
        if (shape.length != 4) {
            throw new IllegalStateException("Expected rank-4 image input, got shape="
                    + java.util.Arrays.toString(shape));
        }
		if (shape[0] != 1 && shape[0] != -1) {
			throw new IllegalStateException("Expected batch dimension 1 or dynamic, got " + shape[0]);
		}

        long d1 = shape[1];
        long d3 = shape[3];

		// This classifier produces RGB input, so only three-channel tensors are valid.
		if (d1 == 3) {
            int h = safeDim(shape[2], "height");
            int w = safeDim(shape[3], "width");
            return new ImageInputSpec(name, true, w, h);
        }

		if (d3 == 3) {
            int h = safeDim(shape[1], "height");
            int w = safeDim(shape[2], "width");
            return new ImageInputSpec(name, false, w, h);
        }

        throw new IllegalStateException("Cannot infer image layout from input shape="
                + java.util.Arrays.toString(shape)
				+ ". Expected a channel dimension of 3.");
    }

    private static int safeDim(long dim, String label) {
        // -1 often used for batch only; H/W should be positive.
        if (dim <= 0 || dim > Integer.MAX_VALUE) {
            throw new IllegalStateException("Invalid " + label + " dimension: " + dim);
        }
        return (int) dim;
    }

    private float[] preprocess(BufferedImage src) {
		// Center-crop to the model aspect ratio before resizing so the source is not
		// geometrically distorted.
        BufferedImage resized = new BufferedImage(inputW, inputH, BufferedImage.TYPE_INT_RGB);
		Rectangle crop = centerCrop(src.getWidth(), src.getHeight(), inputW, inputH);
        Graphics2D g = resized.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
			g.drawImage(src, 0, 0, inputW, inputH, crop.x, crop.y,
					crop.x + crop.width, crop.y + crop.height, null);
        } finally {
            g.dispose();
        }

		final int totalPixels = tensorElementCount(inputW, inputH) / 3;
		final int tensorElements = totalPixels * 3;
        float[] out = new float[tensorElements];

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

                    int p = y * inputW + x;
                    out[p] = normalizeChannel(rf, 0);        // R plane
                    out[idxG + p] = normalizeChannel(gf, 1); // G plane
                    out[idxB + p] = normalizeChannel(bf, 2); // B plane
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

                    out[i++] = normalizeChannel(rf, 0);
                    out[i++] = normalizeChannel(gf, 1);
                    out[i++] = normalizeChannel(bf, 2);
                }
            }
        }
		return out;
	}

	static int tensorElementCount(int width, int height) {
		if (width <= 0 || height <= 0) {
			throw new IllegalArgumentException("tensor dimensions must be positive");
		}
		try {
			return Math.multiplyExact(3, Math.multiplyExact(width, height));
		} catch (ArithmeticException exception) {
			throw new IllegalStateException(
					"Model image dimensions are too large: " + width + "x" + height,
					exception);
		}
	}

	static Rectangle centerCrop(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight) {
		if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) {
			throw new IllegalArgumentException("source and target dimensions must be positive");
		}
		double sourceAspect = (double) sourceWidth / sourceHeight;
		double targetAspect = (double) targetWidth / targetHeight;
		if (sourceAspect > targetAspect) {
			int width = Math.max(1, (int) Math.round(sourceHeight * targetAspect));
			return new Rectangle((sourceWidth - width) / 2, 0, width, sourceHeight);
		}
		int height = Math.max(1, (int) Math.round(sourceWidth / targetAspect));
		return new Rectangle(0, (sourceHeight - height) / 2, sourceWidth, height);
	}

    /** Normalize one channel without allocating a temporary array per pixel. */
    private float normalizeChannel(float value, int channel) {
        return switch (normType) {
            case RESNET -> (value / 255.0f - mean[channel]) / std[channel];
            case SCALE_NEG1_1 -> (value - 127.5f) / 127.5f;
            case SCALE_0_1 -> value / 255.0f;
        };
    }


    private List<ClassScore> topK(float[] probs, int k) {
		int n = Math.min(k, probs.length);
		Comparator<ScoredClass> worstFirst = Comparator
				.comparingDouble(ScoredClass::score)
				.thenComparing(Comparator.comparingInt(ScoredClass::index).reversed());
		PriorityQueue<ScoredClass> selected = new PriorityQueue<>(n, worstFirst);
		for (int index = 0; index < probs.length; index++) {
			ScoredClass candidate = new ScoredClass(index, probs[index]);
			if (selected.size() < n) {
				selected.add(candidate);
			} else if (worstFirst.compare(candidate, selected.peek()) > 0) {
				selected.poll();
				selected.add(candidate);
			}
		}

		List<ScoredClass> ranked = new ArrayList<>(selected);
		ranked.sort(Comparator.comparingDouble(ScoredClass::score).reversed()
				.thenComparingInt(ScoredClass::index));
        List<ClassScore> out = new ArrayList<>(n);
		for (ScoredClass scored : ranked) {
			int ci = scored.index();
			float p = scored.score();
            String name = (labels != null && ci < labels.size()) ? labels.get(ci) : ("class_" + ci);
            out.add(new ClassScore(name, p));
        }
		return List.copyOf(out);
    }

	private record ScoredClass(int index, float score) { }

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
	static float[] flattenToFloatArray(Object outObj) {
		Objects.requireNonNull(outObj, "outObj");
        if (outObj instanceof float[] fa) {
            return fa;
        }
        if (outObj instanceof float[][] f2) {
            if (f2.length == 1) {
				return f2[0];
			}
        }
        if (outObj instanceof float[][][] f3) {
            if (f3.length == 1 && f3[0].length == 1) {
				return f3[0][0];
			}
        }
        throw new IllegalArgumentException("Unsupported ONNX output type: " + outObj.getClass());
    }

    /**
     * Numerically-stable softmax (float output).
     *
     * @param logits logits or unnormalized scores
     * @return probabilities that sum to ~1.0
     */
	static float[] softmax(float[] logits) {
		Objects.requireNonNull(logits, "logits");
		if (logits.length == 0) {
			throw new IllegalArgumentException("model output is empty");
		}
        float max = Float.NEGATIVE_INFINITY;
		for (float v : logits) {
			if (!Float.isFinite(v)) {
				throw new IllegalArgumentException("logits must be finite");
			}
			max = Math.max(max, v);
		}

        double sum = 0.0;
        double[] exps = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            double e = Math.exp(logits[i] - max);
            exps[i] = e;
            sum += e;
        }

        float[] p = new float[logits.length];
        if (sum == 0.0) {
			return p;
		}

        for (int i = 0; i < logits.length; i++) {
            p[i] = (float) (exps[i] / sum);
        }
        return p;
    }

	/** Preserve probability output; otherwise convert logits with softmax. */
	static float[] probabilitiesFromOutput(float[] output) {
		Objects.requireNonNull(output, "output");
		if (output.length == 0) {
			throw new IllegalArgumentException("model output is empty");
		}
		double sum = 0.0;
		boolean probabilityRange = true;
		for (float value : output) {
			if (!Float.isFinite(value)) {
				throw new IllegalArgumentException("model output must be finite");
			}
			probabilityRange &= value >= 0.0f && value <= 1.0f;
			sum += value;
		}
		if (probabilityRange && Math.abs(sum - 1.0) <= 1.0e-4) {
			return output.clone();
		}
		return softmax(output);
	}
}
