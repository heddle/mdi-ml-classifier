package edu.cnu.mdi.mlclassifier.view;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.Point;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayDeque;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicReference;

import javax.swing.SwingUtilities;

import org.junit.jupiter.api.Test;

import edu.cnu.mdi.mlclassifier.model.ClassScore;
import edu.cnu.mdi.mlclassifier.model.ImageClassifier;

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

	@Test
	void lateResultFromOlderImageIsIgnored() throws Exception {
		QueuedClassifier classifier = new QueuedClassifier();
		AtomicReference<List<ClassScore>> accepted = new AtomicReference<>();
		AtomicReference<ImageClassifierView> viewRef = new AtomicReference<>();
		SwingUtilities.invokeAndWait(() -> {
			ImageClassifierView view = new ImageClassifierView(classifier);
			assertTrue(java.util.stream.IntStream.range(0, view.getJMenuBar().getMenuCount())
					.mapToObj(view.getJMenuBar()::getMenu)
					.anyMatch(menu -> menu != null && "Image".equals(menu.getText())));
			view.setResultConsumer(accepted::set);
			view.setImage(new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB), null);
			view.setImage(new BufferedImage(3, 3, BufferedImage.TYPE_INT_RGB), null);
			viewRef.set(view);
		});

		CompletableFuture<List<ClassScore>> oldFuture = classifier.futures.removeFirst();
		assertTrue(oldFuture.isCancelled());
		oldFuture.complete(List.of(new ClassScore("old", .9)));
		classifier.futures.removeFirst().complete(List.of(new ClassScore("new", .8)));
		SwingUtilities.invokeAndWait(() -> { });

		assertEquals("new", accepted.get().get(0).label());
		SwingUtilities.invokeAndWait(viewRef.get()::dispose);
	}

	@Test
	void newerFileLoadCancelsOlderDecode() throws Exception {
		QueuedClassifier classifier = new QueuedClassifier();
		AtomicReference<ControlledLoadingView> viewRef = new AtomicReference<>();
		SwingUtilities.invokeAndWait(() -> {
			ControlledLoadingView view = new ControlledLoadingView(classifier);
			view.openImageFile(new File("older.png"), false);
			view.openImageFile(new File("newer.png"), false);
			viewRef.set(view);
		});

		ControlledLoadingView view = viewRef.get();
		CompletableFuture<BufferedImage> older = view.loads.removeFirst();
		CompletableFuture<BufferedImage> newer = view.loads.removeFirst();
		assertTrue(older.isCancelled());
		newer.complete(new BufferedImage(7, 5, BufferedImage.TYPE_INT_RGB));
		SwingUtilities.invokeAndWait(() -> { });

		assertEquals(1, classifier.futures.size());
		assertEquals(7, classifier.images.removeFirst().getWidth());
		SwingUtilities.invokeAndWait(view::dispose);
	}

	@Test
	void replacingClassifierCancelsOldWorkAndReclassifiesCurrentImage() throws Exception {
		QueuedClassifier original = new QueuedClassifier();
		QueuedClassifier replacement = new QueuedClassifier();
		AtomicReference<ImageClassifierView> viewRef = new AtomicReference<>();
		SwingUtilities.invokeAndWait(() -> {
			ImageClassifierView view = new ImageClassifierView(original);
			view.setImage(new BufferedImage(4, 3, BufferedImage.TYPE_INT_RGB), null);
			view.setClassifier(replacement);
			viewRef.set(view);
		});

		assertTrue(original.futures.getFirst().isCancelled());
		assertEquals(1, replacement.futures.size());
		assertEquals(4, replacement.images.getFirst().getWidth());
		SwingUtilities.invokeAndWait(viewRef.get()::dispose);
	}

	@Test
	void changingTopKReclassifiesAndCancelsOlderRequest() throws Exception {
		QueuedClassifier classifier = new QueuedClassifier();
		AtomicReference<ImageClassifierView> viewRef = new AtomicReference<>();
		AtomicReference<Integer> originalTopK = new AtomicReference<>();
		SwingUtilities.invokeAndWait(() -> {
			ImageClassifierView view = new ImageClassifierView(classifier);
			originalTopK.set(view.getTopK());
			view.setTopK(3);
			view.setImage(new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB), null);
			view.setTopK(10);
			viewRef.set(view);
		});

		assertEquals(List.of(3, 10), classifier.topKs.stream().toList());
		assertTrue(classifier.futures.getFirst().isCancelled());
		assertEquals(10, viewRef.get().getTopK());
		assertThrows(IllegalArgumentException.class, () -> viewRef.get().setTopK(0));
		SwingUtilities.invokeAndWait(() -> {
			viewRef.get().setTopK(originalTopK.get());
			viewRef.get().dispose();
		});
	}

	private static final class QueuedClassifier implements ImageClassifier {
		private final ArrayDeque<CompletableFuture<List<ClassScore>>> futures = new ArrayDeque<>();
		private final ArrayDeque<BufferedImage> images = new ArrayDeque<>();
		private final ArrayDeque<Integer> topKs = new ArrayDeque<>();

		@Override
		public CompletableFuture<List<ClassScore>> classifyAsync(BufferedImage image, int topK) {
			CompletableFuture<List<ClassScore>> future = new CompletableFuture<>();
			images.add(image);
			topKs.add(topK);
			futures.add(future);
			return future;
		}
	}

	private static final class ControlledLoadingView extends ImageClassifierView {
		private static final long serialVersionUID = 1L;
		private final ArrayDeque<CompletableFuture<BufferedImage>> loads = new ArrayDeque<>();

		private ControlledLoadingView(ImageClassifier classifier) {
			super(classifier);
		}

		@Override
		CompletableFuture<BufferedImage> decodeImageAsync(File file) {
			CompletableFuture<BufferedImage> future = new CompletableFuture<>();
			loads.add(future);
			return future;
		}
	}
}
