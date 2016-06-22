/*
 * This code is a simple implementation to look for edges in images.
 * Only used for control flow divergence analysis.
 * (in usual cases it is recommend to apply a low pass filter before executing such code and using a more complex model...)
 */

#define MIN_PITCH_FACTOR 0.2f

__kernel void find_edge_pixels(__global float *image, __global int *steepnessBuffer, int width, int height) {
	unsigned int pixelId = get_global_id(0);
	unsigned int widthReduced = width - 2; // don't check border for edges
	int heightReduced = height - 2;

	// calculate position wthin image buffer
	unsigned int x = pixelId % widthReduced;
	unsigned int y = pixelId / widthReduced;

	unsigned int bufferPos = (y + 1) * width + 1 + x;

	if (bufferPos >= widthReduced * heightReduced)
		return;

	float pixelValue = image[bufferPos];
	int steepness = 0;

	float relationTop = pixelValue / image[bufferPos - width];
	float relationBottom = pixelValue / image[bufferPos + width];
	float relationLeft = pixelValue / image[bufferPos - 1];
	float relationRight = pixelValue / image[bufferPos + 1];

	__private int tmp = 0.0f;
	if (relationTop < (1.0f - MIN_PITCH_FACTOR) || relationTop > (1.0f + MIN_PITCH_FACTOR)) {
			steepness = max((fabs(relationTop - 1.0f) - MIN_PITCH_FACTOR), 0.0f) * 10;
	}

	if (relationBottom < (1.0f - MIN_PITCH_FACTOR) || relationBottom > (1.0f + MIN_PITCH_FACTOR)) {
			tmp = (int)(max((fabs(relationBottom - 1.0f) - MIN_PITCH_FACTOR) ,0.0f) * 10);
			if (tmp > steepness)
				steepness = tmp;
	}

	if (relationLeft < (1.0f - MIN_PITCH_FACTOR) || relationLeft > (1.0f + MIN_PITCH_FACTOR)){
			tmp = (int)(max((fabs(relationLeft - 1.0f) - MIN_PITCH_FACTOR) ,0.0f) * 10);
			if (tmp > steepness)
				steepness = tmp;
	}

	if (relationRight < (1.0f - MIN_PITCH_FACTOR) || relationRight > (1.0f + MIN_PITCH_FACTOR)) {
			tmp = (int)(max((fabs(relationRight - 1.0f) - MIN_PITCH_FACTOR) ,0.0f) * 10);
			if (tmp > steepness)
				steepness = tmp;
	}

	steepnessBuffer[pixelId] = steepness;
}

__kernel void find_edge_pixels_optimized(__global float *image, __global int *steepnessBuffer, int width, int height) {
	unsigned int pixelId = get_global_id(0);
	unsigned int widthReduced = width - 2; // don't check border for edges
	int heightReduced = height - 2;

	// calculate position wthin image buffer
	unsigned int x = pixelId % widthReduced;
	unsigned int y = pixelId / widthReduced;

	unsigned int bufferPos = (y + 1) * width + 1 + x;

	if (bufferPos >= widthReduced * heightReduced)
		return;

	float pixelValue = image[bufferPos];

	float relationTop = max(fabs(pixelValue / image[bufferPos - width] - 1.0f) - MIN_PITCH_FACTOR, 0.0f) * 10;
	float relationBottom = max(fabs(pixelValue / image[bufferPos + width] - 1.0f) - MIN_PITCH_FACTOR, 0.0f) * 10;
	float relationLeft = max(fabs(pixelValue / image[bufferPos - 1] - 1.0f) - MIN_PITCH_FACTOR, 0.0f) * 10;
	float relationRight = max(fabs(pixelValue / image[bufferPos + 1] - 1.0f) - MIN_PITCH_FACTOR, 0.0f) * 10;
	float maxRelation = max(max(max(relationTop, relationBottom), relationLeft), relationRight);

	steepnessBuffer[pixelId] = (int)maxRelation;
}