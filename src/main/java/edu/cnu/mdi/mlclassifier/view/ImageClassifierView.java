package edu.cnu.mdi.mlclassifier.view;

import edu.cnu.mdi.properties.PropertyUtils;
import edu.cnu.mdi.view.BaseView;

@SuppressWarnings("serial")
public class ImageClassifierView extends BaseView {

	public ImageClassifierView(Object... keyVals) {
		super(PropertyUtils.TITLE, "Image Classifier", PropertyUtils.FRACTION, 0.7, PropertyUtils.ASPECT, 1.2,
				PropertyUtils.VISIBLE, true);
		// Implementation of the image classifier view goes here.
	}
}
