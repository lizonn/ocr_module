import cv2
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure




def preprocess_image_db(img):
    coords = np.column_stack(np.where(img > 0))

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    cropped_img = img[x_min:x_max+1, y_min:y_max+1]
    cropped_img = np.array(cropped_img, dtype=np.uint8)

    padded_img = cv2.copyMakeBorder(cropped_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    resized_img = cv2.resize(padded_img, (28, 28))

    return resized_img.flatten()


def preprocess_dataset(df):
    df_processed = df.copy()
    for i in range(len(df)):
        df_processed.iloc[i, 1:] = preprocess_image_db(df.iloc[i, 1:].values.reshape(28, 28)).flatten()

    return df_processed



if __name__ == '__main__':

    df = pd.read_csv("uaset.csv")

    processed_images = []

    for index, row in df.iterrows():
        try:
            image = row[1:].values.reshape([28, 28])
            processed_image = preprocess_image_db(image)
            processed_images.append(processed_image.flatten())
        except:
            print(index)

    processed_df = pd.DataFrame(processed_images, columns=df.columns[1:])
    processed_df.insert(0, 'label', df['label'])

    processed_df.to_csv("processed_uaset.csv", index=False)