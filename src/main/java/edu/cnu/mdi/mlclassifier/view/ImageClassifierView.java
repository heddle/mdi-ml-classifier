package edu.cnu.mdi.mlclassifier.view;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.datatransfer.DataFlavor;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import javax.imageio.ImageIO;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;
import javax.swing.TransferHandler;
import javax.swing.border.EmptyBorder;

import edu.cnu.mdi.properties.PropertyUtils;
import edu.cnu.mdi.view.BaseView;

@SuppressWarnings("serial")
public class ImageClassifierView extends BaseView {

    private final ImagePanel imagePanel = new ImagePanel();
    private final JLabel hintLabel = new JLabel("Drop an image here (or use File → Open)", SwingConstants.CENTER);

    // Optional: remember source for later (model metadata, etc.)
    private Path currentImagePath;

    public ImageClassifierView(Object... keyVals) {
        super(PropertyUtils.TITLE, "Image Classifier",
              PropertyUtils.FRACTION, 0.7,
              PropertyUtils.ASPECT, 1.2,
              PropertyUtils.VISIBLE, true);

        buildUI();
        installDnD();
    }

    private void buildUI() {
        // If BaseView is a JInternalFrame-like container:
        setLayout(new BorderLayout());

        hintLabel.setBorder(new EmptyBorder(10, 10, 10, 10));

        JPanel center = new JPanel(new BorderLayout());
        center.add(imagePanel, BorderLayout.CENTER);
        center.add(hintLabel, BorderLayout.SOUTH);

        add(center, BorderLayout.CENTER);

        // Give the view a sane minimum
        setMinimumSize(new Dimension(400, 300));
    }

    private void installDnD() {
        // Accept file drops onto the whole view (or just imagePanel)
        JComponent dropTarget = imagePanel;
        dropTarget.setTransferHandler(new TransferHandler() {
            @Override
            public boolean canImport(TransferSupport support) {
                return support.isDataFlavorSupported(DataFlavor.javaFileListFlavor);
            }

            @Override
            @SuppressWarnings("unchecked")
            public boolean importData(TransferSupport support) {
                if (!canImport(support)) {
                    return false;
                }
                try {
                    List<File> files = (List<File>) support.getTransferable()
                            .getTransferData(DataFlavor.javaFileListFlavor);
                    if (files == null || files.isEmpty()) {
                        return false;
                    }
                    File f = files.get(0);
                    BufferedImage img = ImageIO.read(f);
                    if (img == null) {
                        hintLabel.setText("Unsupported image format: " + f.getName());
                        return false;
                    }
                    setImage(img, f.toPath());
                    return true;
                } catch (Exception e) {
                    hintLabel.setText("Drop failed: " + e.getMessage());
                    return false;
                }
            }
        });
    }

    public void setImage(BufferedImage img, Path sourcePath) {
        Objects.requireNonNull(img, "img");
        this.currentImagePath = sourcePath;
        imagePanel.setImage(img);

        if (sourcePath != null) {
            hintLabel.setText(sourcePath.getFileName().toString());
        } else {
            hintLabel.setText("Image loaded");
        }

        // Future seam:
        // triggerInferenceAsync(img);
        // or notify listeners: imageLoaded(img, sourcePath);
    }

    // ----------------------------------------------------------------
    // Inner panel that paints the image scaled-to-fit.
    // ----------------------------------------------------------------
    private static final class ImagePanel extends JPanel {
        private BufferedImage image;

        ImagePanel() {
            setOpaque(true);
        }

        void setImage(BufferedImage img) {
            this.image = img;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            Graphics2D g2 = (Graphics2D) g.create();
            try {
                g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g2.setRenderingHint(RenderingHints.KEY_RENDERING,
                                    RenderingHints.VALUE_RENDER_QUALITY);
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                                    RenderingHints.VALUE_ANTIALIAS_ON);

                if (image == null) {
                    // Optional: draw a subtle placeholder
                    return;
                }

                int w = getWidth();
                int h = getHeight();

                int iw = image.getWidth();
                int ih = image.getHeight();

                double sx = (double) w / iw;
                double sy = (double) h / ih;
                double s = Math.min(sx, sy);

                int dw = (int) Math.round(iw * s);
                int dh = (int) Math.round(ih * s);

                int x = (w - dw) / 2;
                int y = (h - dh) / 2;

                g2.drawImage(image, x, y, dw, dh, null);
            } finally {
                g2.dispose();
            }
        }
    }
}
