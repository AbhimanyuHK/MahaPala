import cv2
from matplotlib import pyplot as plt


class MahaPala:

    def __init__(self):
        self.plot_output_list = []
        self.method = cv2.TM_CCOEFF

    def input(self, input_file='data/samples/messi5.jpg'):
        if not input_file:
            raise Exception("Input file is required")

        self.img = cv2.imread(input_file, 0)
        self.img2 = self.img.copy()
        print(self.img)

    def sample(self, sample_file='data/samples/messi_face.jpg'):
        if not sample_file:
            raise Exception("Sample file is required")

        self.template = cv2.imread(sample_file, 0)
        self.w, self.h = self.template.shape[::-1]

    def output(self):
        pass

    def process(self, method=cv2.TM_CCOEFF):
        self.method = method
        img = self.img2.copy()

        # Apply template Matching
        res = cv2.matchTemplate(img, self.template, self.method)
        print(res)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if self.method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + self.w, top_left[1] + self.h)

        output = cv2.rectangle(img, top_left, bottom_right, 255, 2)
        # print(type(output), output)

        self.plot_output_list.append(
            (res, img, self.method)
        )

    def plot_output(self):
        for i, x in enumerate(self.plot_output_list):
            print(i)
            plt.figure(i)
            plt.subplot(1, 2, 1)
            plt.imshow(x[0], cmap='gray')
            plt.title('Matching Result')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(1, 2, 2)
            plt.imshow(x[1], cmap='gray')
            plt.title('Detected Point')
            plt.xticks([])
            plt.yticks([])

            plt.suptitle(x[2])

        plt.show()
