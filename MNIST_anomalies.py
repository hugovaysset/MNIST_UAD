# Utility functions to add anomalies and plot them on images


import matplotlib.pyplot as plt
import numpy as np

flat_shape = 784
square_shape = 28


def set_anomaly(img):
    """
    Set a "draw line" anomaly on the given image
    """
    np.random.seed(0)

    modif_img = []

    if img.shape == (flat_shape,):  # flat format
        modif_img = np.array(img)
        germ = np.random.randint(0, flat_shape, size=1)

        for i in range(2):
            if germ + square_shape * i + i + 2 < flat_shape:
                modif_img[germ + square_shape * i + i] = .99
                modif_img[germ + square_shape * i + i + 1] = .5
                modif_img[germ + square_shape * i + i - 1] = .5

            if germ - square_shape * i - i - 2 > 0:
                modif_img[germ - square_shape * i - i] = .99
                modif_img[germ - square_shape * i - i + 1] = .5
                modif_img[germ - square_shape * i - i - 1] = .5

    elif img.shape[0] == square_shape:  # squared format
        modif_img = np.array(img)
        x, y = np.random.randint(0, square_shape, size=2)

        for i in range(2):
            if x + i < square_shape and y + i + 1 < square_shape:
                modif_img[x + i, y + i] = .99
                modif_img[x + i, y + i + 1] = .5
                modif_img[x + i, y + i - 1] = .5

            if x - i > 0 and y - i - 1 > 0:
                modif_img[x - i, y - i] = .99
                modif_img[x - i, y - i + 1] = .5
                modif_img[x - i, y - i - 1] = .5

    return modif_img


def predict_anomalies(model, ref, dims=(28, 28, 1)):
    """
    Make model predictions on reference and reference + anomalies
    tensor (bool): True if the model takes as inputs a rank-3 tensor (28, 28, 1)
    """
    predictions = model.predict(ref.reshape((ref.shape[0], *dims)))
    anomalies = np.array([set_anomaly(xi.reshape(dims)) for xi in ref])
    anomalies_pred = model.predict(anomalies.reshape((anomalies.shape[0], *dims)))

    if dims == (28, 28, 1):
        predictions = np.squeeze(predictions, axis=-1)
        anomalies = np.squeeze(anomalies, axis=-1)
        anomalies_pred = np.squeeze(anomalies_pred, axis=-1)

    return predictions, anomalies, anomalies_pred


def contour_anomalies(img, maskimg, legend="anomaly"):
    """
    Draw contour line on the edges of the pixels identified by the autoencoder
    as the anomaly
    """
    mapimg = (maskimg.reshape((square_shape, square_shape)) == True)
    ver_seg = np.where(mapimg[:, 1:] != mapimg[:, :-1])
    hor_seg = np.where(mapimg[1:, :] != mapimg[:-1, :])

    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0] + 1))
        l.append((p[1] + 1, p[0] + 1))
        l.append((np.nan, np.nan))

    for p in zip(*ver_seg):
        l.append((p[1] + 1, p[0]))
        l.append((p[1] + 1, p[0] + 1))
        l.append((np.nan, np.nan))

    segments = np.array(l)  # array of size Nx2
    segments[:, 0] = square_shape * segments[:, 0] / mapimg.shape[1]
    segments[:, 1] = square_shape * segments[:, 1] / mapimg.shape[0]

    # and now there isn't anything else to do than plot it
    img.plot(segments[:, 0], segments[:, 1], color='red', linewidth=2, label=legend)
    img.legend()


def plot_anomalies(ref, pred, anomalies, anomalies_pred, show_idx=0, threshold=0.5, ref_tensor=False):
    """
    Plot four images using matplotlib and contour the anomalies. Takes only 2D-arrays as inputs, if necessary
    remove the extra-axis using np.squeeze(ar, axis=-1)
    :param ref: ground truth image
    :param pred: prediction of the model on ref
    :param anomalies: image containing the anomaly
    :param anomalies_pred: prediction of the model on anomalies
    :param show_idx: index of the image to choose in the test set
    :param threshold: threshold for the contour of the predicted anomalies
    :param ref_tensor: if the model takes as input a rank-3 tensor, adds a dimension to the image
    :return:
    """
    fig, axis = plt.subplots(2, 2, figsize=(8, 8), sharex="all", sharey="all")

    if ref_tensor:
      ref = np.squeeze(ref, axis=-1)

    true_anomaly = np.abs(ref[show_idx] - anomalies[show_idx]) > 0
    predicted_anomaly = np.abs(anomalies_pred[show_idx] - anomalies[show_idx]) > threshold

    axis[0][0].imshow(ref[show_idx].reshape((square_shape, square_shape)))
    axis[0][0].set_title("Original")
    axis[0][1].imshow(pred[show_idx].reshape((square_shape, square_shape)))
    axis[0][1].set_title("Prediction on original")
    axis[1][0].imshow(anomalies[show_idx].reshape((square_shape, square_shape)))
    axis[1][0].set_title("Image wit anomaly")
    axis[1][1].imshow(anomalies_pred[show_idx].reshape((square_shape, square_shape)))
    axis[1][1].set_title("Prediction on anomaly")

    contour_anomalies(axis[1][0], true_anomaly, legend="GT anomaly")
    contour_anomalies(axis[1][1], predicted_anomaly, legend="Predicted anomalies")


def plot_predictions(model, inputs, n=5, dims=(28, 28, 1)):
    plt.figure(figsize=(10, 4.5))
    plt.viridis()

    predictions = model.predict(inputs[:n].reshape((n, *dims)))

    for i in range(n):
        # plot original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(inputs[i].reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original Images')

        # plot reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(predictions[i].reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed Images')
    plt.show()

