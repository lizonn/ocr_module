import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

class ImageProcessor:
    def __init__(self):
        pass

    def load_image(self, path):
        self.img = cv2.imread(path)
        return self.img

    def to_gray(self, img):
        return  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def contrast(self, img):
        img = img.copy()

        min_val = np.min(img)
        max_val = np.max(img)

        pixel_range = max_val - min_val

        img = (img - min_val) * (255.0 / pixel_range)
        img = np.round(img).astype(np.uint8)

        return img


    def as_white_background(self, im):
        colors = np.array(im).flatten()
        median = np.percentile(colors, 50)
        if median > 128:
            return im
        else:
            return 255 - im


    def get_histogram(self,im, horiz=False):
        if horiz:
            return np.array(im).mean(axis=0)
        else:
            return np.array(im).mean(axis=1)

    def get_hor_histogram(self,im):

        return np.array(im).sum(axis=(0, 1))


    def get_edges(self, im):
        if len(im.shape) > 2:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(im, 100, 200)

        return edges

    def get_lines_positions(self, im, threshold=.002):
        hist = self.get_histogram(self.get_edges(im))
        threshold = hist.min() + threshold * (hist.max() - hist.min())

        height = hist.shape[0]
        lines = []

        pos = 0
        while True:
            while pos < height and hist[pos] < threshold: pos += 1
            if pos == height:
                break

            line_start = pos
            while pos < height and hist[pos] >= threshold: pos += 1
            line_end = pos
            lines.append((line_start, line_end))
            if pos == height:
                break

        lines = [i for i in lines if abs(i[0] - i[1]) > 5]
        return lines

    def get_words_positions(self,line, threshold=0.1, space_threshold=0.5):
        if len(line.shape) > 2:
            line = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)

        hist = self.get_histogram(line, horiz=True)
        threshold = hist.max() - threshold * (hist.max() - hist.min())

        h_hist = self.get_histogram(line)
        letters_height = h_hist / h_hist.mean() / 2
        letters_height = 1 - np.round(letters_height)
        letters_height = letters_height.sum()
        space_threshold = int(round(letters_height * space_threshold))
        width = hist.shape[0]
        words = []

        pos = 0
        while True:
            while pos < width and hist[pos] >= threshold: pos += 1
            if pos == width:
                break

            word_start = pos
            while pos < width and hist[pos] < threshold: pos += 1
            word_end = pos
            new_word = (word_start, word_end)
            if len(words) != 0:
                prev_words = words[-1]
                if word_start - prev_words[1] < space_threshold:
                    words[-1] = (words[-1][0], word_end)
                else:
                    words.append(new_word)
            else:
                words.append(new_word)

            if pos == width:
                break

        words = [i for i in words if abs(i[0] - i[1]) > 5]
        return words

    def crop_word(self, img, word):
        start,end = word
        cropped_word = img[:, start:end]

        return cropped_word


    def get_letters_positions(self, word_image):
        gray = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated = cv2.dilate(threshold, kernel, iterations=1)

        labels = measure.label(dilated, connectivity=2, background=0)
        positions = []

        for label in np.unique(labels):
            if label == 0:
                continue

            mask = np.zeros(dilated.shape, dtype="uint8")
            mask[labels == label] = 255

            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 2 and h > 10:
                    positions.append((x, y, w, h))

        return positions


if __name__ == '__main__':

    processor = ImageProcessor()
    image = cv2.imread('images/noise.jpg')

    processed_image = processor.contrast(image)
    processed_image = processor.as_white_background(processed_image)
    lines = processor.get_lines_positions(processed_image)
    # words = processor.get_words_positions(processed_image)


    plt.imshow(processed_image,  cmap='gray')
    plt.title('Підготовлене зображення')
    plt.show()

    # виводить позиції рядків
    print("Line positions:", lines)