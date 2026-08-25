package edu.cnu.mdi.mlclassifier.app;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.FontMetrics;
import java.awt.GraphicsConfiguration;
import java.awt.Rectangle;
import java.awt.Window;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.ListSelectionModel;
import javax.swing.SwingUtilities;
import javax.swing.border.EmptyBorder;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableColumn;

import edu.cnu.mdi.ui.fonts.Fonts;

/** Resizable comparison table for completed classification runs. */
@SuppressWarnings("serial")
final class ClassificationHistoryDialog extends JDialog {

	private static final int CELL_HORIZONTAL_PADDING = 24;
	private static final int DIALOG_MARGIN = 10;

	ClassificationHistoryDialog(Component parent, List<ClassificationRun> runs) {
		super(owner(parent), "Classification Comparison History",
				ModalityType.APPLICATION_MODAL);
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		setResizable(true);

		JTable table = createTable(runs);
		JScrollPane scrollPane = new JScrollPane(table);
		scrollPane.setPreferredSize(initialViewportSize(table, runs.size()));

		JButton close = new JButton("Close");
		close.setFont(Fonts.defaultFont);
		close.addActionListener(event -> dispose());
		JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT));
		buttons.add(close);

		JPanel content = new JPanel(new BorderLayout(0, DIALOG_MARGIN));
		content.setBorder(new EmptyBorder(DIALOG_MARGIN, DIALOG_MARGIN,
				DIALOG_MARGIN, DIALOG_MARGIN));
		content.add(scrollPane, BorderLayout.CENTER);
		content.add(buttons, BorderLayout.SOUTH);
		setContentPane(content);
		pack();
		setLocationRelativeTo(parent);
	}

	private static JTable createTable(List<ClassificationRun> runs) {
		String[] columns = { "Time", "Image", "Model", "Top class", "Confidence",
				"Inference (ms)", "Entropy (%)", "Model (MB)", "Normalization" };
		Object[][] rows = new Object[runs.size()][columns.length];
		for (int index = 0; index < runs.size(); index++) {
			ClassificationRun run = runs.get(index);
			rows[index][0] = run.timestamp().toString();
			rows[index][1] = run.imagePath() == null ? "in-memory" : run.imagePath().getFileName();
			rows[index][2] = run.profile().modelPath().getFileName();
			rows[index][3] = run.topResult().label();
			rows[index][4] = String.format("%.4f%%", run.topResult().score() * 100.0);
			rows[index][5] = run.inference() == null ? "" : run.inference().durationMillis();
			rows[index][6] = run.inference() == null ? ""
					: String.format("%.2f%%", run.inference().normalizedEntropyPercent());
			rows[index][7] = String.format("%.1f", run.modelBytes() / (1024.0 * 1024.0));
			rows[index][8] = run.profile().normalization();
		}
		DefaultTableModel model = new DefaultTableModel(rows, columns) {
			@Override
			public boolean isCellEditable(int row, int column) {
				return false;
			}
		};
		JTable table = new JTable(model);
		table.setFont(Fonts.defaultFont);
		table.getTableHeader().setFont(Fonts.defaultBoldFont);
		table.setRowHeight(Math.max(table.getRowHeight(),
				table.getFontMetrics(table.getFont()).getHeight() + 4));
		table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		table.setAutoCreateRowSorter(true);
		table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
		sizeColumnsToContent(table);
		return table;
	}

	private static void sizeColumnsToContent(JTable table) {
		FontMetrics bodyMetrics = table.getFontMetrics(table.getFont());
		FontMetrics headerMetrics = table.getTableHeader().getFontMetrics(
				table.getTableHeader().getFont());
		for (int columnIndex = 0; columnIndex < table.getColumnCount(); columnIndex++) {
			int width = headerMetrics.stringWidth(table.getColumnName(columnIndex));
			for (int row = 0; row < table.getRowCount(); row++) {
				Object value = table.getValueAt(row, columnIndex);
				width = Math.max(width, bodyMetrics.stringWidth(value == null ? "" : value.toString()));
			}
			TableColumn column = table.getColumnModel().getColumn(columnIndex);
			column.setPreferredWidth(width + CELL_HORIZONTAL_PADDING);
		}
	}

	private Dimension initialViewportSize(JTable table, int runCount) {
		int contentWidth = 0;
		for (int index = 0; index < table.getColumnCount(); index++) {
			contentWidth += table.getColumnModel().getColumn(index).getPreferredWidth();
		}
		int visibleRows = Math.max(8, Math.min(18, runCount));
		int contentHeight = table.getTableHeader().getPreferredSize().height
				+ visibleRows * table.getRowHeight();
		GraphicsConfiguration configuration = getGraphicsConfiguration();
		Rectangle screen = configuration == null
				? new Rectangle(contentWidth, contentHeight)
				: configuration.getBounds();
		int maximumWidth = Math.max(1, screen.width * 4 / 5);
		int maximumHeight = Math.max(1, screen.height * 2 / 3);
		return new Dimension(Math.min(contentWidth, maximumWidth),
				Math.min(contentHeight, maximumHeight));
	}

	private static Window owner(Component parent) {
		return parent == null ? null : SwingUtilities.getWindowAncestor(parent);
	}
}
