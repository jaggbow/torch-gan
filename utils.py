def image2vec(image):
    return image.view(image.size(0), 784)


def vec2image(vector):
    return vector.view(vector.size(0), 1, 28, 28)
